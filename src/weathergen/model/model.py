# ruff: noqa: T201
# (C) Copyright 2025 WeatherGenerator contributors.

#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses
import logging
import math
import warnings

import astropy_healpix as hp
import astropy_healpix.healpy
import numpy as np
import torch
import torch.nn as nn
from astropy_healpix import healpy
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.datasets.utils import healpix_verts_rots, r3tos2
from weathergen.model.engines import (
    EmbeddingEngine,
    EnsPredictionHead,
    ForecastingEngine,
    GlobalAssimilationEngine,
    Local2GlobalAssimilationEngine,
    LocalAssimilationEngine,
    QueryAggregationEngine,
    TargetPredictionEngine,
    TargetPredictionEngineClassic,
)
from weathergen.model.layers import MLP, NamedLinear
from weathergen.model.parametrised_prob_dist import LatentInterpolator
from weathergen.model.utils import get_num_parameters
from weathergen.utils.distributed import is_root
from weathergen.utils.utils import get_dtype

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelOutput:
    """
    A dataclass to encapsulate the model output and give a clear API.
    """

    physical: dict[str, torch.Tensor]
    latent: dict[str, torch.Tensor]


class ModelParams(torch.nn.Module):
    """Creation of query and embedding parameters of the model."""

    def __init__(self, cf) -> None:
        super(ModelParams, self).__init__()

        self.cf = cf

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**cf.healpix_level
        self.dtype = get_dtype(cf.attention_dtype)

        bs = cf.batch_size_per_gpu
        nqs = 9
        s = [bs, self.num_healpix_cells, cf.ae_local_num_queries, cf.ae_global_dim_embed]

        ### POSITIONAL EMBEDDINGS ###
        len_token_seq = 1024
        self.pe_embed = torch.nn.Parameter(
            torch.zeros(len_token_seq, cf.ae_local_dim_embed, dtype=self.dtype), requires_grad=False
        )

        pe = torch.zeros(
            self.num_healpix_cells,
            cf.ae_local_num_queries,
            cf.ae_global_dim_embed,
            dtype=self.dtype,
        )
        self.pe_global = torch.nn.Parameter(pe, requires_grad=False)

        ### ROPE COORDS ###
        self.rope_2D = cf.get("rope_2D", False)
        if self.rope_2D:
            self.rope_coords = torch.nn.Parameter(
                torch.zeros(
                    bs,
                    self.num_healpix_cells * cf.ae_local_num_queries,
                    2,
                    dtype=self.dtype,
                ),
                requires_grad=False,
            )
        else:
            self.rope_coords = None

        ### HEALPIX NEIGHBOURS ###
        hlc = self.healpix_level
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(
                np.arange(self.num_healpix_cells), 2**hlc, order="nested"
            ).transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        self.hp_nbours = torch.nn.Parameter(
            torch.empty((temp.shape[0], (temp.shape[1] + 1)), dtype=torch.int32),
            requires_grad=False,
        )
        # self.hp_nbours = torch.empty((temp.shape[0], (temp.shape[1] + 1)), dtype=torch.int32)

        if cf.target_cell_local_prediction:
            tokens_lens_value = nqs * s[2]
        else:
            tokens_lens_value = nqs * s[1] * s[2]
        self.tokens_lens = torch.nn.Parameter(
            tokens_lens_value * torch.ones(bs * s[1] + 1, dtype=torch.int32), requires_grad=False
        )
        self.tokens_lens.data[0] = 0

        self.q_cells_lens = torch.nn.Parameter(
            torch.ones(self.num_healpix_cells + 1, dtype=torch.int32), requires_grad=False
        )
        self.q_cells_lens.data[0] = 0

    def create(self, cf: Config) -> "ModelParams":
        self.reset_parameters(cf)
        return self

    def reset_parameters(self, cf: Config) -> "ModelParams":
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

        # positional encodings

        bs = cf.batch_size_per_gpu
        dim_embed = cf.ae_local_dim_embed
        len_token_seq = 1024
        self.pe_embed.data.fill_(0.0)
        position = torch.arange(0, len_token_seq, device=self.pe_embed.device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim_embed, 2, device=self.pe_embed.device)
            * -(math.log(len_token_seq) / dim_embed),
        )
        self.pe_embed.data[:, 0::2] = torch.sin(position * div[: self.pe_embed[:, 0::2].shape[1]])
        self.pe_embed.data[:, 1::2] = torch.cos(position * div[: self.pe_embed[:, 1::2].shape[1]])

        dim_embed = cf.ae_global_dim_embed

        if self.rope_2D:
            # Precompute per-cell center coordinates (lat, lon in radians) for 2D RoPE.
            # Shape: (num_healpix_cells, ae_local_num_queries, 2)
            verts, _ = healpix_verts_rots(self.healpix_level, 0.5, 0.5)
            coords = r3tos2(verts.to(self.rope_coords.device)).to(self.rope_coords.dtype)
            coords = coords.unsqueeze(1).repeat(1, cf.ae_local_num_queries, 1)
            coords_flat = coords.flatten(0, 1).unsqueeze(0).repeat(bs, 1, 1)
            self.rope_coords.data.copy_(coords_flat)

            # Clear pe_global when using 2D RoPE
            self.pe_global.data.fill_(0.0)
        else:
            # Original pe_global initialization
            self.pe_global.data.fill_(0.0)
            xs = (
                2.0
                * np.pi
                * torch.arange(0, dim_embed, 2, device=self.pe_global.device)
                / dim_embed
            )
            self.pe_global.data[..., 0::2] = 0.5 * torch.sin(
                torch.outer(
                    8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs
                )
            )
            self.pe_global.data[..., 0::2] += (
                torch.sin(
                    torch.outer(
                        torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs
                    )
                )
                .unsqueeze(1)
                .repeat((1, cf.ae_local_num_queries, 1))
            )
            self.pe_global.data[..., 1::2] = 0.5 * torch.cos(
                torch.outer(
                    8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs
                )
            )
            self.pe_global.data[..., 1::2] += (
                torch.cos(
                    torch.outer(
                        torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs
                    )
                )
                .unsqueeze(1)
                .repeat((1, cf.ae_local_num_queries, 1))
            )

        # healpix neighborhood structure

        hlc = self.healpix_level
        num_healpix_cells = self.num_healpix_cells
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(np.arange(num_healpix_cells), 2**hlc, order="nested").transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        # nbors *and* self
        self.hp_nbours.data[:, 0] = torch.arange(temp.shape[0], device=self.hp_nbours.device)
        self.hp_nbours.data[:, 1:] = torch.from_numpy(temp).to(self.hp_nbours.device)

        # varlen index set for tokens
        assert cf.batch_size_per_gpu == cf.batch_size_validation_per_gpu
        nqs = 9
        s = [bs, self.num_healpix_cells, cf.ae_local_num_queries, cf.ae_global_dim_embed]
        if cf.target_cell_local_prediction:
            tokens_lens_value = nqs * s[2]
        else:
            tokens_lens_value = nqs * s[1] * s[2]
        self.tokens_lens.data.fill_(tokens_lens_value)
        self.tokens_lens.data[0] = 0

        # precompute for varlen attention
        self.q_cells_lens.data.fill_(1)
        self.q_cells_lens.data[0] = 0

        # ensure all params have grad set to False

        return


####################################################################################################
class Model(torch.nn.Module):
    """WeatherGenerator model architecture

    WeatherGenerator consists of the following components:

    embeds: embedding networks: Stream specific embedding networks.

    ae_local_blocks: Local assimilation engine: transformer based network to combine different input
        streams per healpix cell.

    ae_adapter: Assimilation engine adapter: Adapter to transform local assimilation engine
        information to the global assimilation engine.

    ae_aggregation_blocks: Query aggregation engine: after the learnable queries are created per
        non-masked healpix cell, this engine combines information from all non-masked cells by
        using dense attention layers.

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

        self.ae_aggregation_engine: QueryAggregationEngine | None = None
        self.ae_global_engine: GlobalAssimilationEngine | None = None
        self.ae_local_engine: LocalAssimilationEngine | None = None
        self.ae_local_global_engine: Local2GlobalAssimilationEngine | None = None
        self.embed_engine: EmbeddingEngine | None = None
        self.embed_target_coords = None
        self.forecast_engine: ForecastingEngine | None = None
        self.interpolate_latents: LatentInterpolator | None = None
        self.pred_adapter_kv = None
        self.pred_heads = None
        self.q_cells: torch.Tensor | None = None
        self.stream_names: list[str] = None
        self.target_token_engines = None

    #########################################
    def create(self) -> "Model":
        """Create each individual module of the model"""
        cf = self.cf

        # determine stream names once so downstream components use consistent keys
        self.stream_names = [str(stream_cfg["name"]) for stream_cfg in cf.streams]
        # separate embedding networks for differnt observation types
        self.embed_engine = EmbeddingEngine(cf, self.sources_size, self.stream_names)

        ##############
        # local assimilation engine
        self.ae_local_engine = LocalAssimilationEngine(cf)

        if cf.latent_noise_kl_weight > 0.0:
            self.interpolate_latents = LatentInterpolator(
                gamma=cf.latent_noise_gamma,
                dim=cf.ae_local_dim_embed,
                use_additive_noise=cf.latent_noise_use_additive_noise,
                deterministic=cf.latent_noise_deterministic_latents,
            )

        ##############
        # local -> global assimilation engine adapter
        self.ae_local_global_engine = Local2GlobalAssimilationEngine(cf)

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
        # query aggregation engine
        self.ae_aggregation_engine = QueryAggregationEngine(cf, self.num_healpix_cells)

        ##############
        # global assimilation engine
        self.ae_global_engine = GlobalAssimilationEngine(cf, self.num_healpix_cells)

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

        self.forecast_engine = ForecastingEngine(cf, self.num_healpix_cells)

        ###############
        # embed coordinates yielding one query token for each target token
        dropout_rate = cf.embed_dropout_rate
        self.embed_target_coords = torch.nn.ModuleDict()
        self.target_token_engines = torch.nn.ModuleDict()
        self.pred_adapter_kv = torch.nn.ModuleDict()
        self.pred_heads = torch.nn.ModuleDict()

        for i_obs, si in enumerate(cf.streams):
            stream_name = self.stream_names[i_obs]

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

            if is_root():
                logger.info("{} :: coord embed: :: {}".format(si["name"], dims_embed))

            dim_coord_in = self.targets_coords_size[i_obs]

            # embedding network for coordinates
            if etc["net"] == "linear":
                self.embed_target_coords[stream_name] = NamedLinear(
                    f"embed_target_coords_{stream_name}",
                    in_features=dim_coord_in,
                    out_features=dims_embed[0],
                    bias=False,
                )
            elif etc["net"] == "mlp":
                self.embed_target_coords[stream_name] = MLP(
                    dim_coord_in,
                    dims_embed[0],
                    hidden_factor=8,
                    with_residual=False,
                    dropout_rate=dropout_rate,
                    norm_eps=self.cf.mlp_norm_eps,
                    stream_name=f"embed_target_coords_{stream_name}",
                )
            else:
                assert False

            # obs-specific adapter for tokens
            if cf.pred_adapter_kv:
                self.pred_adapter_kv[stream_name] = MLP(
                    cf.ae_global_dim_embed,
                    cf.ae_global_dim_embed,
                    hidden_factor=2,
                    with_residual=True,
                    dropout_rate=dropout_rate,
                    norm_type=cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                    stream_name=f"pred_adapter_kv_{stream_name}",
                )
            else:
                self.pred_adapter_kv[stream_name] = torch.nn.Identity()

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

            self.target_token_engines[stream_name] = tte

            # ensemble prediction heads to provide probabilistic prediction
            final_activation = si["pred_head"].get("final_activation", "Identity")
            if is_root():
                logger.debug(
                    f"{final_activation} activation of prediction head of {si['name']} stream"
                )
            self.pred_heads[stream_name] = EnsPredictionHead(
                dims_embed[-1],
                self.targets_num_channels[i_obs],
                si["pred_head"]["num_layers"],
                si["pred_head"]["ens_size"],
                norm_type=cf.norm_type,
                final_activation=final_activation,
                stream_name=stream_name,
            )

        return self

    def reset_parameters(self):
        def _reset_params(module):
            if isinstance(module, nn.Linear | nn.LayerNorm):
                module.reset_parameters()
            else:
                pass

        self.apply(_reset_params)

    #########################################
    def print_num_parameters(self) -> None:
        """Print number of parameters for entire model and each module used to build the model"""

        cf = self.cf
        num_params_embed = [
            get_num_parameters(self.embed_engine.embeds[name]) for name in self.stream_names
        ]
        num_params_total = get_num_parameters(self)
        num_params_ae_local = get_num_parameters(self.ae_local_engine.ae_local_blocks)
        num_params_ae_global = get_num_parameters(self.ae_global_engine.ae_global_blocks)

        num_params_q_cells = np.prod(self.q_cells.shape) if self.q_cells.requires_grad else 0
        num_params_ae_adapater = get_num_parameters(self.ae_local_global_engine.ae_adapter)
        num_params_ae_aggregation = get_num_parameters(
            self.ae_aggregation_engine.ae_aggregation_blocks
        )

        num_params_fe = get_num_parameters(self.forecast_engine.fe_blocks)

        num_params_pred_adapter = [
            get_num_parameters(self.pred_adapter_kv[name]) for name in self.stream_names
        ]
        num_params_embed_tcs = [
            get_num_parameters(self.embed_target_coords[name]) for name in self.stream_names
        ]
        num_params_tte = [
            get_num_parameters(self.target_token_engines[name]) for name in self.stream_names
        ]
        num_params_preds = [get_num_parameters(self.pred_heads[name]) for name in self.stream_names]

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
        print(f" Query Aggregation engine: {num_params_ae_aggregation:,}")
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
    def rename_old_state_dict(self, params: dict) -> dict:
        """Checks if model from checkpoint is from the old model version and if so renames
        the parameters accordingly to the new model version.

        Args:
            params : Dictionary with (old) model parameters from checkpoint
        Returns:
            new_params : Dictionary with (renamed) model parameters
        """
        params_cleanup = {
            "embeds": "embed_engine.embeds",  # EmbeddingEngine
            "ae_local_blocks": "ae_local_engine.ae_local_blocks",  # LocalAssimilationEngine
            "ae_adapter": "ae_local_global_engine.ae_adapter",  # Local2GlobalAssimilationEngine
            "ae_global_blocks": "ae_global_engine.ae_global_blocks",  # GlobalAssimilationEngine
            "fe_blocks": "forecast_engine.fe_blocks",  # ForecastingEngine
        }

        new_params = {}

        for k, v in params.items():
            new_k = k
            prefix = ""

            # Strip "module." (prefix for DataParallel or DistributedDataParallel)
            if new_k.startswith("module."):
                prefix = "module."
                new_k = new_k[len(prefix) :]

            first_w, rest = new_k.split(".", 1) if "." in new_k else (new_k, "")
            # Only check first word (root level modules) to avoid false matches.
            if first_w in params_cleanup:
                new_k = params_cleanup[first_w] + "." + rest

            new_k = prefix + new_k
            new_params[new_k] = v

        return new_params

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

            if self.training:
                # Impute noise to the latent state
                noise_std = self.cf.get("impute_latent_noise_std", 0.0)
                if noise_std > 0.0:
                    tokens = tokens + torch.randn_like(tokens) * torch.norm(tokens) * noise_std

            # Apply 2D RoPE coords only on the first forecast step
            tokens = self.forecast(model_params, tokens, fstep, apply_rope = fstep == forecast_offset)

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

        latents = {}
        latents["posteriors"] = posteriors

        return ModelOutput(physical=preds_all, latent=latents)

    #########################################
    def embed_cells(self, model_params: ModelParams, streams_data) -> torch.Tensor:
        """Embeds input data for each stream separately and rearranges it to cell-wise order
        Args:
            model_params : Query and embedding parameters
            streams_data : Used to initialize first tokens for pre-processing
        Returns:
            Tokens for local assimilation
        """

        device = next(self.parameters()).device
        tokens_all = self.embed_engine(streams_data, model_params.pe_embed, self.dtype, device)

        return tokens_all

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

        # local assimilation model
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
        tokens_global_unmasked_all = []
        posteriors = []
        zero_pad = torch.zeros(1, device=tokens.device, dtype=torch.int32)
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

            # local assimilation model
            tokens_c = self.ae_local_engine(tokens_c, cell_lens_c, use_reentrant=False)

            if self.cf.latent_noise_kl_weight > 0.0:
                tokens_c, posteriors_c = self.interpolate_latents.interpolate_with_noise(
                    tokens_c, sampling=self.training
                )
                posteriors += [posteriors_c]
            else:
                tokens_c, posteriors = tokens_c, 0.0

            # create mask for global tokens, without first element (used for padding)
            mask_c = cell_lens_c[1:].to(torch.bool)
            tokens_global_unmasked_c = tokens_global_c[mask_c]
            q_cells_lens_unmasked_c = torch.cat([zero_pad, q_cells_lens_c[1:][mask_c]])
            cell_lens_unmasked_c = torch.cat([zero_pad, cell_lens_c[1:][mask_c]])

            if l0 == l1 or tokens_c.shape[0] == 0:
                tokens_global_unmasked_all += [tokens_global_unmasked_c]
                continue

            # local to global adapter engine
            tokens_global_unmasked_c = self.ae_local_global_engine(
                tokens_c,
                tokens_global_unmasked_c,
                q_cells_lens_unmasked_c,
                cell_lens_unmasked_c,
                use_reentrant=False,
            )

            tokens_global_unmasked_all += [tokens_global_unmasked_c]

        tokens_global_unmasked = torch.cat(tokens_global_unmasked_all)

        # query aggregation engine on the query tokens in unmasked cells
        # (applying this here assumes batch_size=1)
        # permute to use ae_local_num_queries as the batchsize and no_of_tokens
        # as seq len for flash attention
        tokens_global_unmasked = torch.permute(tokens_global_unmasked, [1, 0, 2])
        tokens_global_unmasked = self.ae_aggregation_engine(
            tokens_global_unmasked, coords=None, use_reentrant=False
        )
        tokens_global_unmasked = torch.permute(tokens_global_unmasked, [1, 0, 2])

        # create mask from cell lens
        mask = cell_lens.to(torch.bool)

        # fill empty tensor using mask for positions of unmasked tokens
        tokens_global[mask] = tokens_global_unmasked.to(tokens_global.dtype)

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
        tokens = self.ae_global_engine(tokens, coords=model_params.rope_coords, use_reentrant=False)

        return tokens

    #########################################
    def forecast(
        self, model_params: ModelParams, tokens: torch.Tensor, fstep: int, apply_rope: bool = False
    ) -> torch.Tensor:
        """Advances latent space representation in time

        Args:
            model_params : Query and embedding parameters (never used)
            tokens : Input tokens to be processed by the model.
            fstep: Current forecast step index (can be used as aux info).
            apply_rope: Whether to apply 2D RoPE coords (only on first forecast step).
        Returns:
            Processed tokens
        Raises:
            ValueError: For unexpected arguments in checkpoint method
        """

        if apply_rope and model_params.rope_2D:
            tokens = self.forecast_engine(tokens, fstep, coords=model_params.rope_coords)
        else:
            tokens = self.forecast_engine(tokens, fstep, coords=None)

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
        tokens_stream = tokens.reshape(s).flatten(0, 1)
        tokens_stream = tokens_stream[model_params.hp_nbours.flatten()].flatten(0, 1)

        # pair with tokens from assimilation engine to obtain target tokens
        preds_tokens = []
        for idx, stream_name in enumerate(self.stream_names):
            si = self.cf.streams[idx]
            tte = self.target_token_engines[stream_name]
            tte_kv = self.pred_adapter_kv[stream_name]
            tc_embed = self.embed_target_coords[stream_name]

            assert batch_size == 1

            ## embed token coords, concatenating along batch dimension
            # (which is taking care of through the varlen attention)
            # arguably we should to the mixed precision policy when creating the model in FSDP
            tc_tokens = torch.cat(
                [
                    checkpoint(
                        tc_embed,
                        streams_data[i_b][idx].target_coords[fstep],
                        use_reentrant=False,
                    )
                    if len(streams_data[i_b][idx].target_coords[fstep].shape) > 1
                    else streams_data[i_b][idx].target_coords[fstep]
                    for i_b in range(len(streams_data))
                ]
            )

            # skip when coordinate embeddings yields nan (i.e. the coord embedding network diverged)
            if torch.isnan(tc_tokens).any():
                nn = si["name"]
                if is_root():
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
            tcs_lens = target_coords_idxs[idx][fstep]
            # coord information for learnable layer norm
            tcs_aux = torch.cat(
                [streams_data[i_b][idx].target_coords[fstep] for i_b in range(len(streams_data))]
            )

            tc_tokens = tte(
                latent=tokens_stream,
                output=tc_tokens,
                latent_lens=model_params.tokens_lens,
                output_lens=tcs_lens,
                coordinates=tcs_aux,
            )

            # final prediction head to map back to physical space
            preds_tokens += [
                checkpoint(self.pred_heads[stream_name], tc_tokens, use_reentrant=False)
            ]

        return preds_tokens
