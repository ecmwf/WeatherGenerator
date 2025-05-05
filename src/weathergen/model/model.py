# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math
import warnings
from pathlib import Path

import astropy_healpix as hp
import astropy_healpix.healpy
import numpy as np
import torch
from astropy_healpix import healpy
from torch.utils.checkpoint import checkpoint

from weathergen.model.engines import (
    EmbeddingEngine,
    EnsPredictionHead,
    ForecastingEngine,
    GlobalAssimilationEngine,
    Local2GlobalAssimilationEngine,
    LocalAssimilationEngine,
    TargetPredictionEngine,
)
from weathergen.model.layers import MLP
from weathergen.model.utils import get_num_parameters
from weathergen.utils.logger import logger


class ModelParams(torch.nn.Module):
    def __init__(self):
        """initializes PyTorch Module and prepares for the creation of model parameters.
        Args:
            self : ModelParams object
        Returns:
            None
        Raises:
            noexcept
        """
        super(ModelParams, self).__init__()

    def create(self, cf):
        """creates positional encoding, HEALPix neighbourhood structure, batch-related parameters and saves them in self

        sinusoidal positional encoding: Sinusoidal positional encoding is used for the model to learn information about the relative
                                        or absolute position of the tokens in the sequence.

                                        For sinusoidal positional encoding we use sine and cosine in different frequencies. In the
                                        end we transform the embedding into a learnable parameter for the model and save it in self.

        HEALPix neighbourhood structure:    Saves all 8 neighbours for every HEALPix cell in local variable nbours and when it is an
                                            edge case with only 7 neighbours we add the cell itself as the 8th neighbour. After that
                                            the cell itself is added additionally to the neighbourhood structure of every cell. In
                                            the end nbours is transformed to a learnable parameter for the model and saved in self.

        batch related parameters creation:  We calculate the token_len for every query and store it in a tensor, which is then used as
                                            a learnable parameter for the model. Then we create a tensor for the attention for every
                                            cell and turn it into a learnable parameter as well.

        Args:
            self : ModelParams object
            cf : configuration
        Returns:
            ModelParams (self)
        Raises:
            AssertionError : If batch sizes for training and validation are not equal.
        """
        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**cf.healpix_level

        # positional encodings

        dim_embed = cf.ae_local_dim_embed
        len_token_seq = 1024
        position = torch.arange(0, len_token_seq).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim_embed, 2) * -(math.log(len_token_seq) / dim_embed))
        pe_embed = torch.zeros(len_token_seq, dim_embed, dtype=torch.float16)
        pe_embed[:, 0::2] = torch.sin(position * div[: pe_embed[:, 0::2].shape[1]])
        pe_embed[:, 1::2] = torch.cos(position * div[: pe_embed[:, 1::2].shape[1]])
        self.pe_embed = torch.nn.Parameter(pe_embed, requires_grad=False)

        dim_embed = cf.ae_global_dim_embed
        pe = torch.zeros(
            self.num_healpix_cells, cf.ae_local_num_queries, dim_embed, dtype=torch.float16
        )
        xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2) / dim_embed
        pe[..., 0::2] = 0.5 * torch.sin(torch.outer(8 * torch.arange(cf.ae_local_num_queries), xs))
        pe[..., 0::2] += (
            torch.sin(torch.outer(torch.arange(self.num_healpix_cells), xs))
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )
        pe[..., 1::2] = 0.5 * torch.cos(torch.outer(8 * torch.arange(cf.ae_local_num_queries), xs))
        pe[..., 1::2] += (
            torch.cos(torch.outer(torch.arange(self.num_healpix_cells), xs))
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )
        self.pe_global = torch.nn.Parameter(pe, requires_grad=False)

        # healpix neighborhood structure

        hlc = self.healpix_level
        num_healpix_cells = self.num_healpix_cells
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(np.arange(num_healpix_cells), 2**hlc, order="nested").transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        # nbors *and* self
        nbours = torch.empty((temp.shape[0], (temp.shape[1] + 1)), dtype=torch.int32)
        nbours[:, 0] = torch.arange(temp.shape[0])
        nbours[:, 1:] = torch.from_numpy(temp)
        self.hp_nbours = torch.nn.Parameter(nbours, requires_grad=False)

        # varlen index set for tokens
        assert cf.batch_size == cf.batch_size_validation
        bs = cf.batch_size
        nqs = 9
        s = [bs, self.num_healpix_cells, cf.ae_local_num_queries, cf.ae_global_dim_embed]
        pad = torch.zeros(1, dtype=torch.int32)
        if cf.target_cell_local_prediction:
            tokens_lens = torch.cat([pad, nqs * s[2] * torch.ones(bs * s[1], dtype=torch.int32)])
        else:
            tokens_lens = torch.cat([pad, nqs * s[1] * s[2] * torch.ones(bs, dtype=torch.int32)])
        self.tokens_lens = torch.nn.Parameter(tokens_lens, requires_grad=False)

        # precompute for varlen attention
        s = (self.num_healpix_cells, cf.ae_local_num_queries, cf.ae_global_dim_embed)
        # q_cells_lens = s[1] * torch.ones( s[0], dtype=torch.int32)
        q_cells_lens = torch.ones(s[0], dtype=torch.int32)
        q_cells_lens = torch.cat([torch.zeros(1, dtype=torch.int32), q_cells_lens])
        self.q_cells_lens = torch.nn.Parameter(q_cells_lens, requires_grad=False)

        return self


####################################################################################################
class Model(torch.nn.Module):
    #########################################
    def __init__(self, cf, sources_size, targets_num_channels, targets_coords_size):
        '''initializes PyTorch Module and prepares for the creation of model.
        Args:
            self : Model object
            cf : configuration with model parameters
            sources_size : list of number of channels for models
            targets_num_channels : list with size of each output sample for coordinates target embedding
            targets_coords_size : list with size of each input sample for coordinates target embedding
        Returns:
            None
        Raises:
            None
        '''
        super(Model, self).__init__()

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**self.healpix_level

        self.cf = cf
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size

    #########################################
    def create(self):
        '''create model buolding blocks and stores them in self(Model).

        embedding networks: Creates multiple embedding networks of different kinds based upon the configuration.

        local ae (assimilation engine): add neural network blocks to PyTorch ModuleList according to specified
                                        number of blocks for the local assimilation in the configuration

        ae (assimilation engine) adapter:   create adapters to transform local ae information to a format that
                                            can be used with the global ae

        learnable parameters:   creates a learnable Tensor with size ("number of healpix cells" * "number of
                                local queris" * "global embedding dimensions"). Overwrites last 10 values for
                                "global embedding dimensions" with metadata if ae_local_queries_per_cell is
                                True

        global ae (assimilation engine):    Uses transformer networks with self attention networks alternating
                                            between local and global attention based upon global attention
                                            density rate

        forecasting engine: Uses transformer networks with self attention networks alternating between local and
                            global attention based upon forecast attention density rate

        get parameters: gets parameters from self and config for further model components creation.

        embedding networks for coordinates: Creates embedding networks specifically for target coordinates and
                                            uses either a linear layer or a multi-layer perceptron (MLP), as
                                            defined in the config.

        prediction adapter: This component transforms the output from the global ae to something compatible with
                            the prediction engine. Uses an MLP if `cf.pred_adapter_kv` is True, otherwise it has
                            no impact.

        prediction engine:  Core prediction network with attention mechanism and MLPs per layer.

        prediction head:    Final layer to generate probabilistic distributions for the target variables instead
                            of just scalar values

        Args:
            self : Model object
        Returns:
            Model (self)
        Raises:
            AssertionError: If a wrong embedding network type was given
        '''
        cf = self.cf

        # separate embedding networks for differnt observation types
        self.embeds = EmbeddingEngine(cf, self.sources_size).create()

        ##############
        # local assimilation engine
        self.ae_local_blocks = LocalAssimilationEngine(cf).create()

        ##############
        # local -> global assimilation engine adapter
        self.ae_adapter = Local2GlobalAssimilationEngine(cf).create()

        ##############
        # learnable queries
        if cf.ae_local_queries_per_cell:
            s = (self.num_healpix_cells, cf.ae_local_num_queries, cf.ae_global_dim_embed)
            q_cells = torch.rand(s, requires_grad=True) / cf.ae_global_dim_embed
            # add meta data
            q_cells[:, :, -8:-6] = (
                (torch.arange(self.num_healpix_cells) / self.num_healpix_cells)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat((1, cf.ae_local_num_queries, 2))
            )
            theta, phi = healpy.pix2ang(
                nside=2**self.healpix_level, ipix=torch.arange(self.num_healpix_cells)
            )
            q_cells[:, :, -6:-3] = (
                torch.cos(theta).unsqueeze(1).unsqueeze(1).repeat((1, cf.ae_local_num_queries, 3))
            )
            q_cells[:, :, -3:] = (
                torch.sin(phi).unsqueeze(1).unsqueeze(1).repeat((1, cf.ae_local_num_queries, 3))
            )
            q_cells[:, :, -9] = torch.arange(cf.ae_local_num_queries)
            q_cells[:, :, -10] = torch.arange(cf.ae_local_num_queries)
        else:
            s = (1, cf.ae_local_num_queries, cf.ae_global_dim_embed)
            q_cells = torch.rand(s, requires_grad=True) / cf.ae_global_dim_embed
        self.q_cells = torch.nn.Parameter(q_cells, requires_grad=True)

        ##############
        # global assimilation engine
        self.ae_global_blocks = GlobalAssimilationEngine(cf, self.num_healpix_cells).create()

        ###############
        # forecasting engine
        self.fe_blocks = ForecastingEngine(cf, self.num_healpix_cells).create()

        ###############
        # embed coordinates yielding one query token for each target token
        dropout_rate = 0.1
        self.embed_target_coords = torch.nn.ModuleList()
        self.target_token_engines = torch.nn.ModuleList()
        self.pred_adapter_kv = torch.nn.ModuleList()
        self.pred_heads = torch.nn.ModuleList()

        for i_obs, si in enumerate(cf.streams):
            # extract and setup relevant parameters
            etc = si["embed_target_coords"]
            tro_type = si["target_readout"]["type"] if "type" in si["target_readout"] else "token"
            dim_embed = si["embed_target_coords"]["dim_embed"]
            dim_out = max(
                dim_embed,
                si["token_size"] * self.targets_num_channels[i_obs],
            )
            tr = si["target_readout"]
            num_layers = tr["num_layers"]
            tr_mlp_hidden_factor = tr["mlp_hidden_factor"] if "mlp_hidden_factor" in tr else 2
            tr_dim_head_proj = tr["dim_head_proj"] if "dim_head_proj" in tr else None
            softcap = tr["softcap"] if "softcap" in tr else 0.0

            if tro_type == "obs_value":
                # fixed dimension for obs_value type
                dims_embed = [si["embed_target_coords"]["dim_embed"] for _ in range(num_layers + 1)]
            else:
                if cf.pred_dyadic_dims:
                    coord_dim = self.geoinfo_sizes[i_obs] * si["token_size"]
                    dims_embed = torch.tensor(
                        [dim_out // 2**i for i in range(num_layers - 1, -1, -1)] + [dim_out]
                    )
                    dims_embed[dims_embed < coord_dim] = dims_embed[
                        torch.where(dims_embed >= coord_dim)[0][0]
                    ]
                    dims_embed = dims_embed.tolist()
                else:
                    dims_embed = torch.linspace(
                        dim_embed, dim_out, num_layers + 1, dtype=torch.int32
                    ).tolist()

            logger.info("{} :: coord embed: :: {}".format(si["name"], dims_embed))

            dim_coord_in = self.targets_coords_size[i_obs]

            # embedding network for coordinates
            if etc["net"] == "linear":
                self.embed_target_coords.append(torch.nn.Linear(dim_coord_in, dims_embed[0]))
            elif etc["net"] == "mlp":
                self.embed_target_coords.append(
                    MLP(
                        dim_coord_in,
                        dims_embed[0],
                        hidden_factor=8,
                        with_residual=False,
                        dropout_rate=dropout_rate,
                    )
                )
            else:
                assert False

            # obs-specific adapter for tokens
            if cf.pred_adapter_kv:
                self.pred_adapter_kv.append(
                    MLP(
                        cf.ae_global_dim_embed,
                        cf.ae_global_dim_embed,
                        hidden_factor=2,
                        with_residual=True,
                        dropout_rate=dropout_rate,
                        norm_type=cf.norm_type,
                    )
                )
            else:
                self.pred_adapter_kv.append(torch.nn.Identity())

            # target prediction engines
            tte = TargetPredictionEngine(
                cf,
                dims_embed,
                dim_coord_in,
                tr_dim_head_proj,
                tr_mlp_hidden_factor,
                softcap,
                tro_type,
            ).create()

            self.target_token_engines.append(tte)

            # ensemble prediction heads to provide probabilistic prediction
            self.pred_heads.append(
                EnsPredictionHead(
                    dims_embed[-1],
                    self.targets_num_channels[i_obs],
                    si["pred_head"]["num_layers"],
                    si["pred_head"]["ens_size"],
                    norm_type=cf.norm_type,
                )
            )

        return self

    #########################################
    def freeze_weights_forecast(self):
        """Freeze model weights
        Args:
            self : Model object
        Returns:
            Model (self)
        Raises:
            None
        """

        # freeze everything
        for p in self.parameters():
            p.requires_grad = False
        self.q_cells.requires_grad = False

        # unfreeze forecast part
        for p in self.fe_blocks.parameters():
            p.requires_grad = True

        return self

    #########################################
    def print_num_parameters(self):
        """print number of parameters for entire model and model building blocks
        Args:
            self : Model object
        Returns:
            None
        Raises:
            None
        """

        cf = self.cf
        num_params_embed = [get_num_parameters(embed) for embed in self.embeds]
        num_params_total = get_num_parameters(self)
        num_params_ae_local = get_num_parameters(self.ae_local_blocks)
        num_params_ae_global = get_num_parameters(self.ae_global_blocks)

        num_params_q_cells = np.prod(self.q_cells.shape) if self.q_cells.requires_grad else 0
        num_params_ae_adapater = get_num_parameters(self.ae_adapter)

        num_params_fe = get_num_parameters(self.fe_blocks)

        num_params_pred_adapter = [get_num_parameters(kv) for kv in self.pred_adapter_kv]
        num_params_embed_tcs = [get_num_parameters(etc) for etc in self.embed_target_coords]
        num_params_tte = [get_num_parameters(tte) for tte in self.target_token_engines]
        num_params_preds = [get_num_parameters(head) for head in self.pred_heads]

        print("-----------------")
        print(f"Total number of trainable parameters: {num_params_total:,}")
        print("Number of parameters:")
        print("  Embedding networks:")
        [
            print("    {} : {:,}".format(si["name"], np))
            for si, np in zip(cf.streams, num_params_embed, strict=False)
        ]
        print(f" Local assimilation engine: {num_params_ae_local:,}")
        print(f" Local-global adapter: {num_params_ae_adapater:,}")
        print(f" Learnable queries: {num_params_q_cells:,}")
        print(f" Global assimilation engine: {num_params_ae_global:,}")
        print(f" Forecast engine: {num_params_fe:,}")
        print(" kv-adapter, coordinate embedding, prediction networks and prediction heads:")
        zps = zip(
            cf.streams,
            num_params_pred_adapter,
            num_params_embed_tcs,
            num_params_tte,
            num_params_preds,
            strict=False,
        )
        [
            print("    {} : {:,} / {:,} / {:,} / {:,}".format(si["name"], np0, np1, np2, np3))
            for si, np0, np1, np2, np3 in zps
        ]
        print("-----------------")

    #########################################
    def load(self, run_id, epoch=-1):
        """checks for missing and unused keys when loading model parameters.

        Gets model state by selecting the run and its unique epoch.
        Args:
            self : Model object
            run_id : specifies the desired training run
            epoch : specifies the desired model state by epoch, default -1 selects latest epoch
        Returns:
            None
        Raises:
            None
        """

        path_run = Path(self.cf.model_path) / run_id
        epoch_id = f"epoch{epoch:05d}" if epoch != -1 and epoch is not None else "latest"
        filename = f"{run_id}_{epoch_id}.chkpt"

        params = torch.load(
            path_run / filename, map_location=torch.device("cpu"), weights_only=True
        )
        params_renamed = {}
        for k in params.keys():
            params_renamed[k.replace("module.", "")] = params[k]
        mkeys, ukeys = self.load_state_dict(params_renamed, strict=False)
        # mkeys, ukeys = self.load_state_dict( params, strict=False)

        if len(mkeys) > 0:
            logger.warning(f"Missing keys when loading model: {mkeys}")

        if len(ukeys) > 0:
            logger.warning(f"Unused keys when loading model: {mkeys}")

    #########################################
    def forward_jac(self, *args):
        """performs a simplified forward pass of the model for Jacobian computation.
        Args:
            self : Model object
            *args : tuple with the following arguments
                    model_params : model parameters
                    batch : streams_data :  used to initialize first tokens for pre-processing and deliver
                                            positional information for prediction
                            source_cell_lens :  used to identify range of tokens to use from generated tokens
                                                in cell embedding
                            target_coords_idxs : indices of target coordinates.
            forecast_steps : Number of forecast steps to iterate (irrelevant for the output)
        Returns:
            tuple of first prediction result without any forecasting
        Raises:
            AssertionError: if batch_size != 1
            AssertionError: if tro_type != "obs_value"
            AssertionError: type(tte_kv) != torch.nn.Identity
        """

        sources = args[:-1]
        sources_lens = args[-1]
        # no-op when satisfied but needed for Jacobian
        sources_lens = sources_lens.to(torch.int64).cpu()

        preds_all = self.forward(sources, sources_lens)

        return tuple(preds_all[0])

    #########################################
    def forward(self, model_params, batch, forecast_offset, forecast_steps):
        """Performs the forward pass of the model to generate forecasts

        First it processes thorugh the embedding network, the local assimilation network and the global
        assimilation network. The output tokens are then used in the prediction and forecast steps that are
        repeated as often as the given number of 'forecast_steps'. We always use the tokens for predictions and
        then forecast new tokens based upon the current ones. All prediction results are collected in a list and
        returned at the end.
        Args:
            self : Model object
            model_params : model parameters
            batch : streams_data :  used to initialize first tokens for pre-processing and deliver positional
                                    information for prediction
                    source_cell_lens :  used to identify range of tokens to use from generated tokens in cell
                                        embedding
                    target_coords_idxs : indices of target coordinates.
            forecast_offset : 
            forecast_steps : Number of forecast steps to iterate
        Returns:
            a list containing all prediction results from the forecasting steps
        Raises:
            AssertionError: if batch_size != 1
            AssertionError: if tro_type != "obs_value"
            AssertionError: type(tte_kv) != torch.nn.Identity
        """

        (streams_data, source_cell_lens, target_coords_idxs) = batch

        # embed
        tokens = self.embed_cells(model_params, streams_data)

        # local assimilation engine and adapter
        tokens = self.assimilate_local(model_params, tokens, source_cell_lens)

        tokens = self.assimilate_global(model_params, tokens)

        # roll-out in latent space
        preds_all = []
        for fstep in range(forecast_offset, forecast_offset + forecast_steps):
            # prediction
            preds_all += [
                self.predict(
                    model_params,
                    fstep,
                    tokens,
                    streams_data,
                    target_coords_idxs,
                )
            ]

            tokens = tokens + self.forecast(model_params, tokens)

        # prediction for final step
        preds_all += [
            self.predict(
                model_params,
                forecast_offset + forecast_steps,
                tokens,
                streams_data,
                target_coords_idxs,
            )
        ]

        return preds_all

    #########################################
    def embed_cells(self, model_params, streams_data):
        """create tokens for local assimilation
        Args:
            self : Model object
            model_params : model parameters
            streams_data : used to initialize first tokens for pre-processing
        Returns:
            tokens for local assimilation
        Raises:
            None
        """

        # code.interact( local=locals())
        source_tokens_lens = torch.stack(
            [
                torch.stack(
                    [
                        s.source_tokens_lens if len(s.source_tokens_lens) > 0 else torch.tensor([])
                        for s in stl_b
                    ]
                )
                for stl_b in streams_data
            ]
        )
        offsets_base = source_tokens_lens.sum(1).sum(0).cumsum(0)
        tokens_all = torch.empty(
            (int(offsets_base[-1]), self.cf.ae_local_dim_embed), dtype=torch.float16, device="cuda"
        )

        for _, sb in enumerate(streams_data):
            for _, (s, embed) in enumerate(zip(sb, self.embeds, strict=False)):
                if not s.source_empty():
                    idxs = s.source_idxs_embed
                    idxs_pe = s.source_idxs_embed_pe

                    # create full scatter index (there's no broadcasting which is likely highly inefficient)
                    idxs = idxs.unsqueeze(1).repeat((1, self.cf.ae_local_dim_embed))
                    x_embed = embed(s.source_tokens_cells, s.source_centroids).flatten(0, 1)
                    # there's undocumented limitation in flash_attn that will make embed fail if
                    # #tokens is too large; code below is a work around
                    # x_embed = torch.cat( [embed( s_c, c_c).flatten(0,1)
                    #                 for s_c,c_c in zip( torch.split( s.source_tokens_cells, 49152),
                    #                                     torch.split( s.source_centroids, 49152))])

                    # scatter write to reorder from per stream to per cell ordering
                    tokens_all.scatter_(0, idxs, x_embed + model_params.pe_embed[idxs_pe])

        return tokens_all

    #########################################
    def assimilate_local(self, model_params, tokens, cell_lens):
        """create tokens for global assimilation
        Args:
            self : Model object
            model_params : model parameters
            tokens : input tokens to be processed by local assimilation
            cell_lens : used to identify range of tokens to use from generated tokens in cell embedding
        Returns:
            tokens for global assimilation
        Raises:
            None
        """

        batch_size = self.cf.batch_size if self.training else self.cf.batch_size_validation

        s = self.q_cells.shape
        # print( f'{np.prod(np.array(tokens.shape))} :: {np.prod(np.array(s))}'
        #        + ':: {np.prod(np.array(tokens.shape))/np.prod(np.array(s))}')
        # TODO: test if positional encoding is needed here
        if self.cf.ae_local_queries_per_cell:
            tokens_global = (self.q_cells + model_params.pe_global).repeat(batch_size, 1, 1)
        else:
            tokens_global = (
                self.q_cells.repeat(self.num_healpix_cells, 1, 1) + model_params.pe_global
            )
        q_cells_lens = torch.cat(
            [model_params.q_cells_lens[0].unsqueeze(0)]
            + [model_params.q_cells_lens[1:] for _ in range(batch_size)]
        )

        # # local assimilation model
        # for block in self.ae_local_blocks :
        #   tokens = checkpoint( block, tokens, cell_lens, use_reentrant=False)

        # for block in self.ae_adapter :
        #   tokens_global = checkpoint( block, tokens_global, tokens, q_cells_lens, cell_lens, use_reentrant=False)

        # work around to bug in flash attention for hl>=5

        cell_lens = cell_lens[1:]
        clen = self.num_healpix_cells // (2 if self.cf.healpix_level <= 5 else 8)
        tokens_global_all = []
        zero_pad = torch.zeros(1, device="cuda", dtype=torch.int32)
        for i in range((cell_lens.shape[0]) // clen):
            # make sure we properly catch all elements in last chunk
            i_end = (i + 1) * clen if i < (cell_lens.shape[0] // clen) - 1 else cell_lens.shape[0]
            l0, l1 = (
                (0 if i == 0 else cell_lens[: i * clen].cumsum(0)[-1]),
                cell_lens[:i_end].cumsum(0)[-1],
            )

            tokens_c = tokens[l0:l1]
            tokens_global_c = tokens_global[i * clen : i_end]
            cell_lens_c = torch.cat([zero_pad, cell_lens[i * clen : i_end]])
            q_cells_lens_c = q_cells_lens[: cell_lens_c.shape[0]]

            if l0 == l1 or tokens_c.shape[0] == 0:
                tokens_global_all += [tokens_global_c]
                continue

            for block in self.ae_local_blocks:
                tokens_c = checkpoint(block, tokens_c, cell_lens_c, use_reentrant=False)

            for block in self.ae_adapter:
                tokens_global_c = checkpoint(
                    block,
                    tokens_global_c,
                    tokens_c,
                    q_cells_lens_c,
                    cell_lens_c,
                    use_reentrant=False,
                )

            tokens_global_all += [tokens_global_c]

        tokens_global = torch.cat(tokens_global_all)

        # recover batch dimension and build global token list
        tokens_global = (
            tokens_global.reshape([batch_size, self.num_healpix_cells, s[-2], s[-1]])
            + model_params.pe_global
        ).flatten(1, 2)

        return tokens_global

    #########################################
    def assimilate_global(self, model_params, tokens):
        """create tokens for forecasting
        Args:
            self : Model object
            model_params : model parameters (not used)
            tokens : input tokens to be pre-processed by global assimilation
        Returns:
            tokens for first forecasting step and prediction
        Raises:
            None
        """

        # global assimilation engine and adapter
        for block in self.ae_global_blocks:
            tokens = checkpoint(block, tokens, use_reentrant=False)

        return tokens

    #########################################
    def forecast(self, model_params, tokens):
        """iterating through the model by block

        Here we iterate through the model and one step goes though one fe_block updating the tokens and then
        using them as input in the next iteration with the current index.
        Args:
            self : Model object
            model_params : model parameters (never used)
            tokens : input tokens to be processed by the model.
        Returns:
            processed tokens
        Raises:
            ValueError: For unexpected arguments in checkpoint method
        """

        for it, block in enumerate(self.fe_blocks):
            aux_info = torch.tensor([it], dtype=torch.float32, device="cuda")
            tokens = checkpoint(block, tokens, aux_info, use_reentrant=False)

        return tokens

    #########################################
    def predict(self, model_params, fstep, tokens, streams_data, target_coords_idxs):
        """generates predictions for specific target coordinates based on the current forecast iteration

        takes the current forecast and target coordinate information to generate prediction. It uses the
        prediction head on the tokens and coordinate information to transform its prediction into physical
        space.

        Args:
            self : Model object
            model_params : model parameters
            fstep : Number of forecast steps to iterate
            tokens : input tokens to be processed by the model
            streams_data : used to initialize target coordinates tokens and index information
            target_coords_idxs : indices of target coordinates.
        Returns:
            prediction output tokens in physical representation
        Raises:
            AssertionError: if batch_size != 1
            AssertionError: if tro_type != "obs_value"
            AssertionError: type(tte_kv) != torch.nn.Identity
        """

        # fp32, i32 = torch.float32, torch.int32
        batch_size = self.cf.batch_size if self.training else self.cf.batch_size_validation

        s = [batch_size, self.num_healpix_cells, self.cf.ae_local_num_queries, tokens.shape[-1]]
        tokens_stream = (tokens.reshape(s) + model_params.pe_global).flatten(0, 1)
        tokens_stream = tokens_stream[model_params.hp_nbours.flatten()].flatten(0, 1)

        # pair with tokens from assimilation engine to obtain target tokens
        preds_tokens = []
        for ii, (tte, tte_kv) in enumerate(
            zip(self.target_token_engines, self.pred_adapter_kv, strict=False)
        ):
            si = self.cf.streams[ii]
            tc_embed = self.embed_target_coords[ii]

            assert batch_size == 1

            # embed token coords, concatenating along batch dimension (which is taking care of through
            # the varlen attention)
            tc_tokens = torch.cat(
                [
                    checkpoint(
                        tc_embed,
                        streams_data[i_b][ii].target_coords[fstep],
                        use_reentrant=False,
                    )
                    if len(streams_data[i_b][ii].target_coords[fstep].shape) > 1
                    else streams_data[i_b][ii].target_coords[fstep]
                    for i_b in range(len(streams_data))
                ]
            )

            if torch.isnan(tc_tokens).any():
                nn = si["name"]
                logger.warning(
                    f"Skipping prediction for {nn} because of {torch.isnan(tc_tokens).sum()} NaN in tc_tokens."
                )
                preds_tokens += [torch.tensor([], device=tc_tokens.device)]
                continue
            if tc_tokens.shape[0] == 0:
                preds_tokens += [torch.tensor([], device=tc_tokens.device)]
                continue

            # TODO: how to support tte_kv efficiently, generate 1-ring neighborhoods here or on a per
            #       stream basis
            assert type(tte_kv) == torch.nn.Identity

            # lens for varlen attention
            tcs_lens = target_coords_idxs[ii][fstep]
            # coord information for learnable layer norm
            tcs_aux = torch.cat(
                [streams_data[i_b][ii].target_coords[fstep] for i_b in range(len(streams_data))]
            )

            # apply prediction engine
            for ib, block in enumerate(tte):
                if self.cf.pred_self_attention and ib % 3 == 1:
                    tc_tokens = checkpoint(block, tc_tokens, tcs_lens, tcs_aux, use_reentrant=False)
                else:
                    tc_tokens = checkpoint(
                        block,
                        tc_tokens,
                        tokens_stream,
                        tcs_lens,
                        model_params.tokens_lens,
                        tcs_aux,
                        use_reentrant=False,
                    )

            # final prediction head to map back to physical space
            preds_tokens += [checkpoint(self.pred_heads[ii], tc_tokens, use_reentrant=False)]

        return preds_tokens
