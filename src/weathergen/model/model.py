# ruff: noqa: T201
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

from weathergen.common.config import Config
from weathergen.model.engines import (
    EmbeddingEngine,
    EnsPredictionHead,
    ForecastingEngine,
    GlobalAssimilationEngine,
    Local2GlobalAssimilationEngine,
    LocalAssimilationEngine,
    TargetPredictionEngine,
    TargetPredictionEngineClassic,
)
from weathergen.model.layers import MLP, NamedLinear
from weathergen.model.parametrised_prob_dist import LatentInterpolator
from weathergen.model.utils import get_num_parameters
from weathergen.utils.logger import logger
from weathergen.utils.utils import get_dtype


class ModelParams(torch.nn.Module):
    """Creation of query and embedding parameters of the model."""

    def __init__(self) -> None:
        super(ModelParams, self).__init__()

    def create(self, cf: Config) -> "ModelParams":
        """Creates positional embedding for each grid point for each stream used after stream
        embedding, positional embedding for all stream assimilated cell-level local embedding,
        initializing queries for local-to-global adapters, HEALPix neighbourhood based parameter
        initializing for target prediction.

        Sinusoidal positional encoding: Harmonic positional encoding based upon sine and cosine for
            both per stream after stream embedding and per cell level for local assimilation.

        HEALPix neighbourhood structure: Determine the neighbors for each cell and initialize each
            with its own cell number as well as the cell numbers of its neighbors. If a cell has
            fewer than eight neighbors, use its own cell number to fill the remaining slots.

        Query len based parameter creation: Calculate parameters for the calculated token length at
            each cell after local assimilation.

        Args:
            cf : Configuration
        """
        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**cf.healpix_level
        self.dtype = get_dtype(cf.attention_dtype)

        # positional encodings

        dim_embed = cf.ae_local_dim_embed
        len_token_seq = 1024
        position = torch.arange(0, len_token_seq).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim_embed, 2) * -(math.log(len_token_seq) / dim_embed))
        pe_embed = torch.zeros(len_token_seq, dim_embed, dtype=self.dtype)
        pe_embed[:, 0::2] = torch.sin(position * div[: pe_embed[:, 0::2].shape[1]])
        pe_embed[:, 1::2] = torch.cos(position * div[: pe_embed[:, 1::2].shape[1]])
        self.pe_embed = torch.nn.Parameter(pe_embed, requires_grad=False)

        dim_embed = cf.ae_global_dim_embed
        pe = torch.zeros(
            self.num_healpix_cells, cf.ae_local_num_queries, dim_embed, dtype=self.dtype
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
        assert cf.batch_size_per_gpu == cf.batch_size_validation_per_gpu
        bs = cf.batch_size_per_gpu
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
    """WeatherGenerator model architecture

    WeatherGenerator consists of the following components:

    embeds: embedding networks: Stream specific embedding networks.

    ae_local_blocks: Local assimilation engine: transformer based network to combine different input
        streams per healpix cell.

    ae_adapter: Assimilation engine adapter: Adapter to transform local assimilation engine
        information to the global assimilation engine.

    ae_global_blocks: Global assimilation engine: Transformer network alternating between local and
        global attention based upon global attention density rate.

    fe_blocks: Forecasting engine: Transformer network using the output of global attention to
        advance the latent representation in time.

    embed_target_coords: Embedding networks for coordinates: Initializes embedding networks tailored
        for metadata embedded target coordinates. The architecture is either a linear layer or a
        multi-layer perceptron, determined by the configuration of the embedding target coordinate
        networks.

    pred_adapter_kv: Prediction adapter: Adapter to transform the global assimilation/forecasting
        engine output to the prediction engine. Uses an MLP if `cf.pred_adapter_kv` is True,
        otherwise it uses an identity function.

    target_token_engines: Prediction engine: Transformer based prediction network that generates
        output corresponding to target coordinates.

    pred_heads: Prediction head: Final layers using target token engines output for mapping target
        coordinates to its physical space.
    """

    #########################################
    def __init__(self, cf: Config, sources_size, targets_num_channels, targets_coords_size):
        """
        Args:
            cf : Configuration with model parameters
            sources_size : List of number of channels for models
            targets_num_channels : List with size of each output sample for coordinates target
                embedding
            targets_coords_size : List with size of each input sample for coordinates target
                embedding
        """
        super(Model, self).__init__()

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**self.healpix_level

        self.cf = cf
        self.dtype = get_dtype(self.cf.attention_dtype)
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size

    #########################################
    def create(self) -> "Model":
        """Create each individual module of the model"""
        cf = self.cf

        # separate embedding networks for differnt observation types
        self.embeds = EmbeddingEngine(cf, self.sources_size).create()

        ##############
        # local assimilation engine
        self.ae_local_blocks = LocalAssimilationEngine(cf).create()

        if cf.latent_noise_kl_weight > 0.0:
            self.interpolate_latents = LatentInterpolator(
                gamma=cf.latent_noise_gamma,
                dim=cf.ae_local_dim_embed,
                use_additive_noise=cf.latent_noise_use_additive_noise,
                deterministic=cf.latent_noise_deterministic_latents,
            )

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
        if isinstance(cf.forecast_steps, int):
            assert not (cf.forecast_steps > 0 and cf.fe_num_blocks == 0), (
                "Empty forecast engine (fe_num_blocks = 0), but forecast_steps > 0"
            )
        else:
            assert not (min(cf.forecast_steps) > 0 and cf.fe_num_blocks == 0), (
                "Empty forecast engine (fe_num_blocks = 0), but forecast_steps[i] > 0 for some i"
            )

        self.fe_blocks = ForecastingEngine(cf, self.num_healpix_cells).create()

        ###############
        # embed coordinates yielding one query token for each target token
        dropout_rate = cf.embed_dropout_rate
        self.embed_target_coords = torch.nn.ModuleList()
        self.target_token_engines = torch.nn.ModuleList()
        self.pred_adapter_kv = torch.nn.ModuleList()
        self.pred_heads = torch.nn.ModuleList()

        for i_obs, si in enumerate(cf.streams):
            stream_name = si.get("name", i_obs)

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
                self.embed_target_coords.append(
                    NamedLinear(
                        f"embed_target_coords_{stream_name}",
                        in_features=dim_coord_in,
                        out_features=dims_embed[0],
                        bias=False,
                    )
                )
            elif etc["net"] == "mlp":
                self.embed_target_coords.append(
                    MLP(
                        dim_coord_in,
                        dims_embed[0],
                        hidden_factor=8,
                        with_residual=False,
                        dropout_rate=dropout_rate,
                        norm_eps=self.cf.mlp_norm_eps,
                        stream_name=f"embed_target_coords_{stream_name}",
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
                        norm_eps=self.cf.mlp_norm_eps,
                        stream_name=f"pred_adapter_kv_{stream_name}",
                    )
                )
            else:
                self.pred_adapter_kv.append(torch.nn.Identity())

            # target prediction engines
            tte_version = (
                TargetPredictionEngine
                if cf.decoder_type != "PerceiverIOCoordConditioning"
                else TargetPredictionEngineClassic
            )
            tte = tte_version(
                cf,
                dims_embed,
                dim_coord_in,
                tr_dim_head_proj,
                tr_mlp_hidden_factor,
                softcap,
                tro_type,
                stream_name=stream_name,
            )

            self.target_token_engines.append(tte)

            # ensemble prediction heads to provide probabilistic prediction
            final_activation = si["pred_head"].get("final_activation", "Identity")
            logger.debug(
                f"{final_activation} activation as prediction head output of {si['name']} stream"
            )
            self.pred_heads.append(
                EnsPredictionHead(
                    dims_embed[-1],
                    self.targets_num_channels[i_obs],
                    si["pred_head"]["num_layers"],
                    si["pred_head"]["ens_size"],
                    norm_type=cf.norm_type,
                    final_activation=final_activation,
                    stream_name=stream_name,
                )
            )

        return self

    #########################################
    def freeze_weights_forecast(self) -> "Model":
        """Freezes core model weights and makes forecasting engine weights trainable"""

        # freeze everything
        for p in self.parameters():
            p.requires_grad = False
        self.q_cells.requires_grad = False

        # unfreeze forecast part
        for p in self.fe_blocks.parameters():
            p.requires_grad = True

        return self

    #########################################
    def print_num_parameters(self) -> None:
        """Print number of parameters for entire model and each module used to build the model"""

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
    def load(self, run_id: str, epoch: str = -1) -> None:
        """Loads model state from checkpoint and checks for missing and unused keys.
        Args:
            run_id : model_id of the trained model
            epoch : The epoch to load. Default (-1) is the latest epoch
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
        sources = args[:-1]
        sources_lens = args[-1]
        # no-op when satisfied but needed for Jacobian
        sources_lens = sources_lens.to(torch.int64).cpu()

        preds_all = self.forward(sources, sources_lens)

        return tuple(preds_all[0])

    #########################################
    def forward(self, model_params: ModelParams, batch, forecast_offset: int, forecast_steps: int):
        """Performs the forward pass of the model to generate forecasts

        Tokens are processed through the model components, which were defined in the create method.
        Args:
            model_params : Query and embedding parameters
            batch :
                streams_data : Contains tokenized source data and target data for each dataset and
                    each stream
                source_cell_lens : Used to identify range of tokens to use from generated tokens in
                    cell embedding
                target_coords_idxs : Indices of target coordinates for each dataset.
            forecast_offset : Starting index for iteration
            forecast_steps : Number of forecast steps to calculate from forecast_offset
        Returns:
            A list containing all prediction results
        """

        (streams_data, source_cell_lens, target_coords_idxs) = batch

        # embed
        tokens = self.embed_cells(model_params, streams_data)

        # local assimilation engine and adapter
        tokens, posteriors = self.assimilate_local(model_params, tokens, source_cell_lens)

        tokens = self.assimilate_global(model_params, tokens)

        # roll-out in latent space
        preds_all = []
        tokens_all = [tokens]
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

            tokens = self.forecast(model_params, tokens)
            tokens_all += [tokens]

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

        # now encode the targets into the latent space for all fsteps
        if self.cf.get("encode_targets_latent", False):
            with torch.no_grad():
                tokens_targets = []
                tokens_targets_srclk = self.embed_cells_targets_srclk(model_params, streams_data)
                for fstep in range(len(tokens_targets_srclk)):
                    tokens_target, _ = self.assimilate_local(
                        model_params, tokens_targets_srclk[fstep], source_cell_lens
                    )
                    tokens_target = self.assimilate_global(model_params, tokens_target)
                    tokens_target_det = tokens_target.detach()  # explicitly detach as well
                    tokens_targets.append(tokens_target_det)

        return_dict = {"preds_all": preds_all, "posteriors": posteriors}
        if self.cf.get("encode_targets_latent", False):
            return_dict["tokens_all"] = tokens_all
            return_dict["tokens_targets"] = tokens_targets

        return return_dict

    #########################################
    def embed_cells(self, model_params: ModelParams, streams_data) -> torch.Tensor:
        """Embeds input data for each stream separately and rearranges it to cell-wise order
        Args:
            model_params : Query and embedding parameters
            streams_data : Used to initialize first tokens for pre-processing
        Returns:
            Tokens for local assimilation
        """

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
            (int(offsets_base[-1]), self.cf.ae_local_dim_embed), dtype=self.dtype, device="cuda"
        )

        for _, sb in enumerate(streams_data):
            for _, (s, embed) in enumerate(zip(sb, self.embeds, strict=False)):
                if not s.source_empty():
                    idxs = s.source_idxs_embed
                    idxs_pe = s.source_idxs_embed_pe

                    # create full scatter index
                    # (there's no broadcasting which is likely highly inefficient)
                    idxs = idxs.unsqueeze(1).repeat((1, self.cf.ae_local_dim_embed))
                    x_embed = embed(s.source_tokens_cells, s.source_centroids).flatten(0, 1)
                    # there's undocumented limitation in flash_attn that will make embed fail if
                    # #tokens is too large; code below is a work around
                    # x_embed = torch.cat(
                    #     [
                    #         embed(s_c, c_c).flatten(0, 1)
                    #         for s_c, c_c in zip(
                    #             torch.split(s.source_tokens_cells, 49152),
                    #             torch.split(s.source_centroids, 49152),
                    #         )
                    #     ]
                    # )

                    # scatter write to reorder from per stream to per cell ordering
                    tokens_all.scatter_(0, idxs, x_embed + model_params.pe_embed[idxs_pe])

        return tokens_all

    def embed_cells_targets_srclk(self, model_params: ModelParams, streams_data) -> torch.Tensor:
        """Embeds target data similar to source tokens for each fstep and stream separately and rearranges it to cell-wise order
        Args:
            model_params : Query and embedding parameters
            streams_data : Used to initialize first tokens for pre-processing
        Returns:
            Tokens for local assimilation
        """
        with torch.no_grad():
            target_srclk_tokens_lens = torch.stack(
                [
                    torch.stack(
                        [
                            torch.stack(
                                [
                                    s.target_srclk_tokens_lens[fstep]
                                    if len(s.target_srclk_tokens_lens[fstep]) > 0
                                    else torch.tensor([])
                                    for fstep in range(len(s.target_srclk_tokens_lens))
                                ]
                            )
                            for s in stl_b
                        ]
                    )
                    for stl_b in streams_data
                ]
            )
            offsets_base = target_srclk_tokens_lens.sum(1).sum(0).cumsum(1)
            num_fsteps = target_srclk_tokens_lens.shape[
                2
            ]  # TODO: KCT, if there are diff no of tokens per fstep, this may fail
            tokens_all = []
            for fstep in range(num_fsteps):
                tokens_all.append(
                    torch.empty(
                        (int(offsets_base[fstep][-1]), self.cf.ae_local_dim_embed),
                        dtype=self.dtype,
                        device="cuda",
                    )
                )

            tokens_all_scattered = []
            for _, sb in enumerate(streams_data):
                for _, (s, embed) in enumerate(zip(sb, self.embeds, strict=False)):
                    for fstep in range(num_fsteps):
                        # TODO: KCT: should we actually remove the below check and just return an empty tensor?
                        if not (s.target_srclk_tokens_lens[fstep].sum() == 0):
                            idxs = s.target_srclk_idxs_embed[fstep]
                            idxs_pe = s.target_srclk_idxs_embed_pe[fstep]

                            # create full scatter index
                            # (there's no broadcasting which is likely highly inefficient)
                            idxs = idxs.unsqueeze(1).repeat((1, self.cf.ae_local_dim_embed))

                            x_embed = embed(
                                s.target_srclk_tokens_cells[fstep], s.target_srclk_centroids[fstep]
                            ).flatten(0, 1)

                            # scatter write to reorder from per stream to per cell ordering
                            tokens_all_fstep = tokens_all[fstep]
                            tokens_all_fstep.scatter_(
                                0, idxs, x_embed + model_params.pe_embed[idxs_pe]
                            )
                            tokens_all_scattered.append(tokens_all_fstep)

        return tokens_all_scattered

    #########################################
    def assimilate_local(
        self, model_params: ModelParams, tokens: torch.Tensor, cell_lens: torch.Tensor
    ) -> torch.Tensor:
        """Processes embedded tokens locally and prepares them for the global assimilation
        Args:
            model_params : Query and embedding parameters
            tokens : Input tokens to be processed by local assimilation
            cell_lens : Used to identify range of tokens to use from generated tokens in cell
                embedding
        Returns:
            Tokens for global assimilation
        """

        batch_size = (
            self.cf.batch_size_per_gpu if self.training else self.cf.batch_size_validation_per_gpu
        )

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
        # for block in self.ae_local_blocks:
        #     tokens = checkpoint(block, tokens, cell_lens, use_reentrant=False)

        # if self.cf.latent_noise_kl_weight > 0.0:
        #     tokens, posteriors = self.interpolate_latents.interpolate_with_noise(
        #         tokens, sampling=self.training
        #     )
        # else:
        #     tokens, posteriors = tokens, 0.0

        # for block in self.ae_adapter:
        #     tokens_global = checkpoint(
        #         block,
        #         tokens_global,
        #         tokens,
        #         q_cells_lens,
        #         cell_lens,
        #         use_reentrant=False,
        #     )

        # work around to bug in flash attention for hl>=5

        cell_lens = cell_lens[1:]
        clen = self.num_healpix_cells // (2 if self.cf.healpix_level <= 5 else 8)
        tokens_global_all = []
        posteriors = []
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

            if self.cf.latent_noise_kl_weight > 0.0:
                tokens_c, posteriors_c = self.interpolate_latents.interpolate_with_noise(
                    tokens_c, sampling=self.training
                )
                posteriors += [posteriors_c]
            else:
                tokens_c, posteriors = tokens_c, 0.0

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

        return tokens_global, posteriors

    #########################################
    def assimilate_global(self, model_params: ModelParams, tokens: torch.Tensor) -> torch.Tensor:
        """Performs transformer based global assimilation in latent space
        Args:
            model_params : Query and embedding parameters (never used)
            tokens : Input tokens to be pre-processed by global assimilation
        Returns:
            Latent representation of the model
        """

        # global assimilation engine and adapter
        for block in self.ae_global_blocks:
            tokens = checkpoint(block, tokens, use_reentrant=False)

        return tokens

    #########################################
    def forecast(self, model_params: ModelParams, tokens: torch.Tensor) -> torch.Tensor:
        """Advances latent space representation in time

        Args:
            model_params : Query and embedding parameters (never used)
            tokens : Input tokens to be processed by the model.
        Returns:
            Processed tokens
        Raises:
            ValueError: For unexpected arguments in checkpoint method
        """

        for it, block in enumerate(self.fe_blocks):
            aux_info = torch.tensor([it], dtype=torch.float32, device="cuda")
            tokens = checkpoint(block, tokens, aux_info, use_reentrant=False)

        return tokens

    #########################################
    def predict(
        self,
        model_params: ModelParams,
        fstep: int,
        tokens: torch.Tensor,
        streams_data,
        target_coords_idxs,
    ) -> list[torch.Tensor]:
        """Predict outputs at the specific target coordinates based on the input weather state and
        pre-training task and projects the latent space representation back to physical space.

        Args:
            model_params : Query and embedding parameters
            fstep : Number of forecast steps
            tokens : Tokens from global assimilation engine
            streams_data : Used to initialize target coordinates tokens and index information
            target_coords_idxs : Indices of target coordinates
        Returns:
            Prediction output tokens in physical representation for each target_coords.
        """

        batch_size = (
            self.cf.batch_size_per_gpu if self.training else self.cf.batch_size_validation_per_gpu
        )

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

            ## embed token coords, concatenating along batch dimension
            # (which is taking care of through the varlen attention)
            with torch.amp.autocast("cuda", dtype=torch.float32, enabled=False):
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

            # skip when coordinate embeddings yields nan (i.e. the coord embedding network diverged)
            if torch.isnan(tc_tokens).any():
                nn = si["name"]
                logger.warning(
                    (
                        f"Skipping prediction for {nn} because",
                        f" of {torch.isnan(tc_tokens).sum()} NaN in tc_tokens.",
                    )
                )
                preds_tokens += [torch.tensor([], device=tc_tokens.device)]
                continue

            # skip empty lengths
            if tc_tokens.shape[0] == 0:
                preds_tokens += [torch.tensor([], device=tc_tokens.device)]
                continue

            # TODO: how to support tte_kv efficiently,
            #  generate 1-ring neighborhoods here or on a per stream basis
            assert isinstance(tte_kv, torch.nn.Identity)

            # lens for varlen attention
            tcs_lens = target_coords_idxs[ii][fstep]
            # coord information for learnable layer norm
            tcs_aux = torch.cat(
                [streams_data[i_b][ii].target_coords[fstep] for i_b in range(len(streams_data))]
            )

            tc_tokens = tte(
                latent=tokens_stream,
                output=tc_tokens,
                latent_lens=model_params.tokens_lens,
                output_lens=tcs_lens,
                coordinates=tcs_aux,
            )

            # final prediction head to map back to physical space
            preds_tokens += [checkpoint(self.pred_heads[ii], tc_tokens, use_reentrant=False)]

        return preds_tokens
