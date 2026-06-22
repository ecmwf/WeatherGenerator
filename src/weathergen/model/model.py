# ruff: noqa: T201
# (C) Copyright 2025 WeatherGenerator contributors.

#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import math
import warnings

import astropy_healpix as hp
import astropy_healpix.healpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.utils import healpix_verts_rots, r3tos2
from weathergen.model.encoder import EncoderModule
from weathergen.model.engines import (
    BilinearDecoder,
    DeepSSLFusion,
    EnsPredictionHead,
    ForecastingEngine,
    IdentityEngine,
    LatentPredictionHeadIdentity,
    LatentPredictionHeadMLP,
    LatentPredictionHeadTransformer,
    LatentState,
    TargetPredictionEngine,
    TargetPredictionEngineClassic,
)
from weathergen.model.layers import MLP, NamedLinear
from weathergen.model.utils import get_num_parameters
from weathergen.utils.distributed import is_root
from weathergen.utils.utils import get_dtype, is_stream_forcing

logger = logging.getLogger(__name__)

type StreamName = str


class ModelOutput:
    """
    Representation of model output
    """

    physical: list[dict[StreamName, torch.Tensor]]
    latent: list[dict[str, torch.Tensor | LatentState]]
    latent_deep: list[dict[str, list[torch.Tensor]]] | None

    def __init__(self, len_output: int) -> None:
        self.physical = [{} for _ in range(len_output)]
        self.latent = [{} for _ in range(len_output)]
        self.latent_deep = None

    def add_physical_prediction(
        self, fstep: int, stream_name: StreamName, pred: torch.Tensor
    ) -> None:
        self.physical[fstep][stream_name] = pred

    def add_latent_prediction(self, fstep: int, latent_name: str, pred: torch.Tensor) -> None:
        self.latent[fstep][latent_name] = pred

    def add_deep_latent_prediction(
        self, fstep: int, name: str, level_preds: list[torch.Tensor]
    ) -> None:
        if self.latent_deep is None:
            self.latent_deep = [{} for _ in range(len(self.physical))]
        self.latent_deep[fstep][name] = level_preds

    def get_physical_prediction(
        self, fstep: int, stream_name: StreamName | None = None, sample_idx: int | None = None
    ):
        pred = self.physical[fstep]
        if stream_name is not None:
            pred = pred.get(stream_name, None)
            if sample_idx is not None:
                assert sample_idx < len(pred), "Invalid sample index."
                pred = pred[sample_idx]
        return pred

    def get_latent_prediction(self, fstep: int):
        return self.latent[fstep]


class ModelParams(torch.nn.Module):
    """Creation of query and embedding parameters of the model."""

    def __init__(self, cf) -> None:
        super(ModelParams, self).__init__()

        self.cf = cf

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**cf.healpix_level
        self.dtype = get_dtype(cf.attention_dtype)

        # Positional embeddings
        self.max_tokens_local_per_cell = cf.get("ae_local_max_tokens_per_cell", 64)
        self.pe_embed = torch.nn.Parameter(
            torch.zeros(self.max_tokens_local_per_cell, cf.ae_local_dim_embed, dtype=self.dtype),
            requires_grad=False,
        )

        pe = torch.zeros(
            self.num_healpix_cells,
            cf.ae_local_num_queries,
            cf.ae_global_dim_embed,
            dtype=self.dtype,
        )
        self.pe_global = torch.nn.Parameter(pe, requires_grad=False)

        # RoPE coordinates
        self.rope_2D = cf.get("rope_2D", False)
        if self.rope_2D:
            self.num_extra_tokens = cf.num_register_tokens + cf.num_class_tokens
            total_tokens = (
                self.num_healpix_cells + self.num_extra_tokens
            ) * cf.ae_local_num_queries
            self.register_buffer(
                "rope_coords",
                torch.zeros(
                    1,
                    total_tokens,
                    2,
                    dtype=self.dtype,
                ),
            )
            self.register_buffer(
                "rope_cell_coords",
                torch.zeros(
                    self.num_healpix_cells,
                    2,
                    dtype=self.dtype,
                ),
            )
        else:
            self.rope_coords = None
            self.rope_cell_coords = None

        # HEALPix neighbours
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

        dim_embed = cf.ae_local_dim_embed
        token_idx_bias = 16
        freq_bias = 8
        self.pe_embed.data.fill_(0.0)
        position = torch.arange(
            token_idx_bias,
            token_idx_bias + self.max_tokens_local_per_cell,
            device=self.pe_embed.device,
        ).unsqueeze(1)
        div = torch.exp(
            torch.arange(freq_bias, freq_bias + dim_embed, 2, device=self.pe_embed.device)
            * -(math.log(self.max_tokens_local_per_cell) / dim_embed),
        )
        self.pe_embed.data[:, 0::2] = torch.sin(position * div[: self.pe_embed[:, 0::2].shape[1]])
        self.pe_embed.data[:, 1::2] = torch.cos(position * div[: self.pe_embed[:, 1::2].shape[1]])

        dim_embed = cf.ae_global_dim_embed

        if self.rope_2D:
            # Precompute per-cell center coordinates (lat, lon in radians) for 2D RoPE.
            # Shape: (num_healpix_cells, ae_local_num_queries, 2)
            verts, _ = healpix_verts_rots(self.healpix_level, 0.5, 0.5)
            coords = r3tos2(verts.to(self.rope_coords.device)).to(self.rope_coords.dtype)
            # Per-cell coords for QueryAggregationEngine (no query expansion)
            self.rope_cell_coords.data.copy_(coords)
            coords = coords.unsqueeze(1).repeat(1, cf.ae_local_num_queries, 1)
            coords_flat = coords.flatten(0, 1).unsqueeze(0)
            offset = self.num_extra_tokens * cf.ae_local_num_queries
            self.rope_coords.data.fill_(0.0)
            self.rope_coords.data[:, offset : offset + coords_flat.shape[1], :].copy_(coords_flat)

        # pe_global: always initialized. RoPE handles relative position in Q/K, but pe_global
        # provides per-cell token identity which is critical for masked cells that have no
        # content from local assimilation. Without it, masked cells are identical and the
        # teacher representation (evaluated without dropout) collapses to low rank.
        self.pe_global.data.fill_(0.0)
        xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=self.pe_global.device) / dim_embed
        self.pe_global.data[..., 0::2] = 0.5 * torch.sin(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
        )
        self.pe_global.data[..., 0::2] += (
            torch.sin(
                torch.outer(torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs)
            )
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )
        self.pe_global.data[..., 1::2] = 0.5 * torch.cos(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
        )
        self.pe_global.data[..., 1::2] += (
            torch.cos(
                torch.outer(torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs)
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

        # precompute for varlen attention
        self.q_cells_lens.data.fill_(1)
        self.q_cells_lens.data[0] = 0

        # ensure all params have grad set to False

        return


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

        self.embed_target_coords = None
        self.encoder: EncoderModule | None = None
        self.forecast_engine: ForecastingEngine | IdentityEngine | None = None
        self.pred_heads = None
        self.q_cells: torch.Tensor | None = None
        self.stream_names: list[str] = None
        self.target_token_engines = None

        assert cf.get("forecast", {}).get("att_dense_rate", 1.0) == 1.0, (
            "Local attention not adapted for register tokens"
        )
        self.num_register_tokens = cf.num_register_tokens
        self.latent_heads = None
        self.latent_pre_norm = None
        self.register_buffer("latent_perturbation_log_sigma", None)
        self.deep_ssl_fusion: DeepSSLFusion | None = None
        self.deep_ssl_level_projections: nn.ModuleDict | None = None
        # auxiliary tokens
        self.class_token_idxs = list(
            range(cf.num_register_tokens, cf.num_register_tokens + cf.num_class_tokens)
        )
        self.register_token_idxs = list(range(cf.num_register_tokens))
        self.aux_token_idxs = list(range(cf.num_register_tokens + cf.num_class_tokens))
        self.num_aux_tokens = cf.num_register_tokens + cf.num_class_tokens


    def _create_latent_pred_head(
        self, global_cfg, name, loss_cfg, use_class_token, use_patch_token
    ):
        if loss_cfg["head"].lower() == "mlp":
            return LatentPredictionHeadMLP(
                name,
                global_cfg.ae_global_dim_embed,
                loss_cfg,
                use_class_token=use_class_token,
                use_patch_token=use_patch_token,
                default_mlp_type=global_cfg.get("mlp_type", "mlp"),
            )
        elif loss_cfg["head"].lower() == "transformer":
            return LatentPredictionHeadTransformer(
                global_cfg,
                name,
                global_cfg.ae_global_dim_embed,
                loss_cfg,
                use_class_token=use_class_token,
                use_patch_token=use_patch_token,
            )
        elif loss_cfg["head"].lower() == "identity":
            return LatentPredictionHeadIdentity()
        else:
            assert False, f"Unknown latent prediction head type {loss_cfg['head']}"

    def create(self) -> "Model":
        """Create each individual module of the model"""
        cf = self.cf

        self.encoder = EncoderModule(
            cf, self.sources_size, self.targets_num_channels, self.targets_coords_size
        )

        mode_cfg = cf.training_config
        if cf.fe_num_blocks > 0:
            self.forecast_engine = ForecastingEngine(cf, mode_cfg, self.num_healpix_cells)
        else:
            self.forecast_engine = IdentityEngine()

        # embed coordinates yielding one query token for each target token
        dropout_rate = cf.embed_dropout_rate
        self.embed_target_coords = torch.nn.ModuleDict()
        self.target_token_engines = torch.nn.ModuleDict()
        self.pred_heads = torch.nn.ModuleDict()

        # determine stream names once so downstream components use consistent keys
        self.stream_names = [str(stream_cfg["name"]) for stream_cfg in cf.streams]

        for i_stream, _ in enumerate(cf.streams):
            stream_name = self.stream_names[i_stream]

        loss_terms = [
            v.type for _, v in cf.training_config.losses.items() if v.get("enabled", True)
        ]
        if cf.validation_config.get("losses"):
            loss_terms += [
                v.type for _, v in cf.validation_config.losses.items() if v.get("enabled", True)
            ]

        if "LossPhysical" in loss_terms:
            for i_stream, si in enumerate(cf.streams):
                stream_name = self.stream_names[i_stream]

                # skip decoder if channels are empty
                if is_stream_forcing(si):
                    continue

                # skip for the moment to ensure target embedding and tte exist (ordering of
                # cf.streams is random)
                if si.get("pred_spatial_shared") is None:
                    # extract and setup relevant parameters
                    etc = si["embed_target_coords"]
                    tr = si["target_readout"]
                    num_layers = tr["num_layers"]
                    tr_mlp_hidden_factor = (
                        tr["mlp_hidden_factor"] if "mlp_hidden_factor" in tr else 2
                    )
                    tr_mlp_type = tr.get("mlp_type", cf.get("mlp_type", "mlp"))
                    tr_dim_head_proj = tr["dim_head_proj"] if "dim_head_proj" in tr else None
                    softcap = tr["softcap"] if "softcap" in tr else 0.0

                    dims_embed = [
                        si["embed_target_coords"]["dim_embed"] for _ in range(num_layers + 1)
                    ]

                    if is_root():
                        logger.info("{} :: coord embed: :: {}".format(si["name"], dims_embed))

                    dim_coord_in = self.targets_coords_size[i_stream]

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
                            mlp_type=self.cf.get("mlp_type", "mlp"),
                            norm_eps=self.cf.mlp_norm_eps,
                            name=f"embed_target_coords_{stream_name}",
                        )
                    else:
                        assert False

                    if cf.decoder_type == "Linear":
                        tte = BilinearDecoder(
                            stream_name,
                            dims_embed[0],
                            cf.ae_global_dim_embed,
                            self.targets_num_channels[i_stream],
                        )
                    else:
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
                            tr_mlp_type,
                            softcap,
                            stream_config=si,
                        )

                    self.target_token_engines[stream_name] = tte

                    # ensemble prediction heads to provide probabilistic prediction
                    final_activation = si["pred_head"].get("final_activation", "Identity")
                    if is_root():
                        logger.debug(
                            f"{final_activation} activation of pred head of {si['name']} stream"
                        )
                    self.pred_heads[stream_name] = EnsPredictionHead(
                        dims_embed[-1],
                        self.targets_num_channels[i_stream],
                        si["pred_head"]["num_layers"],
                        si["pred_head"]["ens_size"],
                        norm_type=cf.norm_type,
                        final_activation=final_activation,
                        stream_name=stream_name,
                    )

            # iterate again to setup shared spatial pred heads if specified in config
            for i_stream, si in enumerate(cf.streams):
                stream_name = self.stream_names[i_stream]

                # skip decoder if channels are empty
                if is_stream_forcing(si):
                    continue

                pred_spatial_shared = si.get("pred_spatial_shared")
                if pred_spatial_shared is not None:
                    if pred_spatial_shared not in self.stream_names:
                        msg = f"Stream {stream_name} has pred_spatial_shared={pred_spatial_shared}"
                        msg += " but no stream with that name found."
                        raise ValueError(msg)
                    if pred_spatial_shared == stream_name:
                        msg = f"Stream {stream_name} has pred_spatial_shared={pred_spatial_shared}"
                        msg += "but cannot share with itself."
                        raise ValueError(msg)
                    logger.debug(
                        f"{stream_name} shares spatial prediction head with {pred_spatial_shared}."
                    )

                    self.embed_target_coords[stream_name] = self.embed_target_coords[
                        pred_spatial_shared
                    ]
                    self.target_token_engines[stream_name] = self.target_token_engines[
                        pred_spatial_shared
                    ]

                    idx_shared_s = [
                        i for i, so in enumerate(cf.streams) if so["name"] == pred_spatial_shared
                    ]
                    assert (len(idx_shared_s)) == 1
                    si_other = cf.streams[idx_shared_s[0]]
                    dims_embed = [
                        si_other["embed_target_coords"]["dim_embed"] for _ in range(num_layers + 1)
                    ]

                    # ensemble prediction heads to provide probabilistic prediction
                    final_activation = si["pred_head"].get("final_activation", "Identity")
                    if is_root():
                        logger.debug(
                            f"{final_activation} activation of pred head of {si['name']} stream"
                        )
                    self.pred_heads[stream_name] = EnsPredictionHead(
                        dims_embed[-1],
                        self.targets_num_channels[i_stream],
                        si["pred_head"]["num_layers"],
                        si["pred_head"]["ens_size"],
                        norm_type=cf.norm_type,
                        final_activation=final_activation,
                        stream_name=stream_name,
                    )

        # Latent heads for losses
        self.latent_heads = nn.ModuleDict()
        # Encoder output is normalized inside the encoder via `ae_global_trailing_layer_norm`, so
        # the physical decoders and the SSL heads/teacher target consume the same normalized
        # representation. Identity here avoids a redundant second norm.
        # TODO remove from code entirely
        self.latent_pre_norm = nn.Identity()

        ssl_losses_cfgs = [
            v
            for _, v in cf.training_config.losses.items()
            if v.type == "LossLatentSSLStudentTeacher" and v.get("enabled", True)
        ]

        # TODO: support multiple LossLatentSSLStudentTeacher terms
        assert len(ssl_losses_cfgs) <= 1, "To be implemented."
        for ssl_target_losses in ssl_losses_cfgs:
            self.latent_pre_norm = nn.Identity()
            for loss, loss_conf in ssl_target_losses.loss_fcts.items():
                if loss == "iBOT":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=True,
                        use_patch_token=True,
                    )
                elif loss == "JEPA":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=False,
                        use_patch_token=True,
                    )
                elif loss == "DINO":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=True,
                        use_patch_token=False,
                    )

        # Latent-perturbation noise scale (learnable or fixed)
        # torch.zeros() is used (not torch.tensor) so this respects the meta-device context used by FSDP. The actual value is filled in reset_parameters().
        num_mem = cf.get("latent_perturbation_num_members", 0)
        if num_mem > 1:
            if cf.get("latent_perturbation_sigma_learnable", True):
                self.latent_perturbation_log_sigma = nn.Parameter(torch.zeros(1))
            else:
                self.register_buffer("latent_perturbation_log_sigma", torch.zeros(1))

        # Deep SSL fusion and per-level projections (student only)
        deep_ssl_cfg = cf.training_config.get("deep_ssl", None)
        if deep_ssl_cfg and deep_ssl_cfg.get("enabled", False) and deep_ssl_cfg.get("tap_after"):
            num_taps = len(deep_ssl_cfg.tap_after)
            num_levels = num_taps + 1  # taps + final output
            hidden_factor = deep_ssl_cfg.get("fusion_hidden_factor", 2)
            self.deep_ssl_fusion = DeepSSLFusion(num_levels, cf.ae_global_dim_embed, hidden_factor)

            # Per-level output projections: one per latent head per level
            # Skip identity heads (teacher model) — they have no learnable projection
            self.deep_ssl_level_projections = nn.ModuleDict()
            for head_name, head in self.latent_heads.items():
                out_dim = self._get_latent_head_out_dim(head)
                if out_dim < 0:
                    continue
                self.deep_ssl_level_projections[head_name] = nn.ModuleList(
                    [nn.Linear(out_dim, out_dim, bias=False) for _ in range(num_levels)]
                )

        return self

    @staticmethod
    def _get_latent_head_out_dim(head: nn.Module) -> int:
        """Infer the output dimension of a latent prediction head."""
        if isinstance(head, LatentPredictionHeadMLP):
            return head.blocks.layers[-1].out_features
        elif isinstance(head, LatentPredictionHeadTransformer):
            return head.blocks[-1].out_features
        elif isinstance(head, LatentPredictionHeadIdentity):
            return -1  # identity head: projection is also identity
        else:
            raise ValueError(f"Cannot determine output dim for head type {type(head)}")

    def reset_parameters(self):
        def _reset_params(module):
            if isinstance(module, nn.Linear | nn.LayerNorm):
                module.reset_parameters()
            else:
                pass

        self.apply(_reset_params)

        if self.latent_perturbation_log_sigma is not None:
            sigma_init = self.cf.get("latent_perturbation_sigma_init", 0.01)
            with torch.no_grad():
                self.latent_perturbation_log_sigma.fill_(math.log(sigma_init))

    def print_num_parameters(self) -> None:
        """Print number of parameters for entire model and each module used to build the model"""

        cf = self.cf
        num_params_embed = [
            get_num_parameters(self.encoder.embed_engine.embeds[name]) for name in self.stream_names
        ]
        num_params_total = get_num_parameters(self)
        num_params_ae_local = get_num_parameters(self.encoder.ae_local_engine.ae_local_blocks)
        num_params_ae_global = get_num_parameters(self.encoder.ae_global_engine.ae_global_blocks)

        num_params_q_cells = (
            np.prod(self.encoder.q_cells.shape) if self.encoder.q_cells.requires_grad else 0
        )
        num_params_q_aux = (
            np.prod(self.encoder.q_aux.shape) if self.encoder.q_aux.requires_grad else 0
        )
        num_params_ae_adapter = get_num_parameters(self.encoder.ae_local_global_engine)

        num_params_ae_aggregation = get_num_parameters(
            self.encoder.ae_aggregation_engine.ae_aggregation_blocks
        )

        num_params_latent_heads = get_num_parameters(self.latent_heads)
        num_params_latent_heads += get_num_parameters(self.latent_pre_norm)

        num_params_fe = get_num_parameters(self.forecast_engine.fe_blocks)

        mdict = self.embed_target_coords
        num_params_embed_tcs = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.stream_names
        ]
        mdict = self.target_token_engines
        num_params_tte = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.stream_names
        ]
        mdict = self.pred_heads
        num_params_preds = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.stream_names
        ]

        print("-----------------")
        print(f"Total number of trainable parameters: {num_params_total:,}")
        print("Number of parameters:")
        print("  Embedding networks:")
        [
            print("    {} : {:,}".format(si["name"], np))
            for si, np in zip(cf.streams, num_params_embed, strict=False)
        ]
        print(f" Local assimilation engine: {num_params_ae_local:,}")
        print(f" Local-global adapter: {num_params_ae_adapter:,}")
        print(f" Learnable spatial queries: {num_params_q_cells:,}")
        print(f" Learnable auxiliary queries: {num_params_q_aux:,}")
        print(f" Query Aggregation engine: {num_params_ae_aggregation:,}")
        print(f" Global assimilation engine: {num_params_ae_global:,}")
        print(f" Latent prediction heads and pre-norm: {num_params_latent_heads:,}")
        if self.deep_ssl_fusion is not None:
            num_params_deep_ssl = get_num_parameters(self.deep_ssl_fusion)
            if self.deep_ssl_level_projections is not None:
                num_params_deep_ssl += get_num_parameters(self.deep_ssl_level_projections)
            print(f" Deep SSL fusion + level projections: {num_params_deep_ssl:,}")
        print(f" Forecast engine: {num_params_fe:,}")
        print(" coordinate embedding, prediction networks and prediction heads:")
        zps = zip(
            cf.streams,
            num_params_embed_tcs,
            num_params_tte,
            num_params_preds,
            strict=False,
        )
        [
            print("   {} : {:,} / {:,} / {:,}".format(si["name"], np0, np1, np2))
            for si, np0, np1, np2 in zps
        ]
        print("-----------------")

    def tokens_to_latent_state(self, tokens_post_norm, tokens) -> LatentState:
        """
        Extract separate parts from global latent space representation and store in LatentState
        """
        toks_pn = tokens_post_norm
        return LatentState(
            register_tokens=toks_pn[:, self.register_token_idxs] if toks_pn is not None else None,
            class_token=toks_pn[:, self.class_token_idxs] if tokens_post_norm is not None else None,
            patch_tokens=toks_pn[:, self.num_aux_tokens :] if toks_pn is not None else None,
            z_pre_norm=tokens,
        )

    def forward(self, model_params: ModelParams, batch: ModelBatch) -> ModelOutput:
        """Forward pass of the model

        Tokens are processed through the model components, which were defined in the create method.
        Args:
            model_params : Query and embedding parameters
            batch
        Returns:
            A list containing all prediction results
        """

        output = ModelOutput(batch.get_output_len())

        tokens, posteriors, intermediates = self.encoder(model_params, batch)
        output.add_latent_prediction(0, "posteriors", posteriors)

        # recover batch dimension and separate input_steps
        shape = (len(batch), batch.get_num_steps(), *tokens.shape[1:])
        # collapse along input step dimension
        tokens = tokens.reshape(shape).sum(axis=1)

        # reshape intermediates the same way as tokens
        for i, inter in enumerate(intermediates):
            intermediates[i] = inter.reshape(shape).sum(axis=1)

        # Allow for pushforward trick
        p_fwd = self.cf.training_config.get("forecast", {}).get("pushforward", False)
        # roll-out in latent space, iterate and generate output over requested output steps
        for step in batch.get_output_idxs():
            without_grad = p_fwd and self.training and step != max(batch.get_output_idxs())
            if without_grad:
                # Pushforward mode: advance tokens without grad; no decoding with torch.no_grad():
                tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
                continue

            if self.forecast_engine:
                tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
            if "masking" in self.cf.training_config.training_mode:
                # decoder predictions
                output = self.predict_decoders(model_params, step, tokens, batch, output)
            if "student_teacher" in self.cf.training_config.training_mode:
                # latent predictions (raw and with SSL heads)
                output = self.predict_latent(model_params, step, tokens, batch, output, intermediates)

        return output

    def predict_latent(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: ModelBatch,
        output: ModelOutput,
        intermediates: list[torch.Tensor] | None = None,
    ) -> ModelOutput:
        """
        Compute latent predictions
        """

        # safe latent prediction
        tokens_post_norm = self.latent_pre_norm(tokens) if step == 0 else None
        noise_pre_predictor_std = self.cf.get("noise_pre_predictor_std", 0)
        if noise_pre_predictor_std > 0 and self.training:
            tokens_post_norm = (
                tokens_post_norm
                + torch.randn_like(tokens_post_norm)
                * torch.norm(tokens_post_norm)
                * noise_pre_predictor_std
            )

        latent_state = self.tokens_to_latent_state(tokens_post_norm, tokens)
        output.add_latent_prediction(step, "latent_state", latent_state)

        # latent predictions for SSL training
        for name, head in self.latent_heads.items():
            output.add_latent_prediction(step, name, head(latent_state))

        # deep SSL: multi-level predictions (only at fstep 0, matching existing SSL)
        if intermediates and step == 0:
            all_levels = intermediates + [tokens]

            if self.deep_ssl_fusion is not None:
                # Student path: fuse all levels, predict once, project per level
                fused = self.deep_ssl_fusion(all_levels)
                fused_post_norm = self.latent_pre_norm(fused)
                fused_latent = self.tokens_to_latent_state(fused_post_norm, fused)
                for name, head in self.latent_heads.items():
                    predictor_out = head(fused_latent)
                    projections = self.deep_ssl_level_projections[name]
                    level_preds = [proj(predictor_out) for proj in projections]
                    output.add_deep_latent_prediction(step, name, level_preds)
            else:
                # Teacher path: independent per-level prediction (no fusion)
                level_preds_per_head: dict[str, list[torch.Tensor]] = {
                    name: [] for name in self.latent_heads
                }
                for level_tokens in all_levels:
                    level_post_norm = self.latent_pre_norm(level_tokens)
                    level_state = self.tokens_to_latent_state(level_post_norm, level_tokens)
                    for name, head in self.latent_heads.items():
                        level_preds_per_head[name].append(head(level_state))
                for name, preds in level_preds_per_head.items():
                    output.add_deep_latent_prediction(step, name, preds)

        return output

    def predict_decoders(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: ModelBatch,
        output: ModelOutput,
    ) -> ModelOutput:
        """
        Compute decoder-based predictions

        Predict outputs at the specific target coordinates based on the input weather state and
        pre-training task and projects the latent space representation back to physical space.

        Args:
            model_params : Query and embedding parameters
            fstep : Number of forecast steps
            tokens : Tokens from global assimilation engine
            streams_data : Used to initialize target coordinates tokens and index information
                List of StreamData len(streams_data) == batch_size_per_gpu
            target_coords_idxs : Indices of target coordinates
        Returns:
            Prediction output tokens in physical representation for each target_coords.
        """
        # Empty dicts evaluate to False in python
        if not self.pred_heads:
            return output

        # ── Strip aux tokens ─────────────────────────────────────────────
        # tokens: [B, num_aux + H*Q, D] → [B, H*Q, D]
        tokens = tokens[:, self.num_aux_tokens:]

        B = len(batch)
        H = self.num_healpix_cells
        Q = self.cf.ae_local_num_queries   # 1 per default config
        D = tokens.shape[-1]
        assert tokens.shape == (B, H * Q, D), f"unexpected token shape {tokens.shape}"
        # tokens_nbors_lens uses fill_value=9 (one entry per neighbour cell), which is only correct when Q=1.  
        # If Q>1 each neighbour contributes Q tokens and the value must change to 9*Q.
        assert Q == 1, "predict_decoders assumes ae_local_num_queries==1; update tokens_nbors_lens fill_value if Q>1"

        # ── Latent Gaussian perturbation ────────────────────────────────
        # Sample M independent members by treating each as an extra batch element 
        # The entire decoder forward pass runs once, fully vectorised.
        # Layout convention (used throughout): M is the OUTER axis, B is the INNER axis.
        #   tokens_tiled[m*B + b]  belongs to member m, batch item b
        M = self.cf.get("latent_perturbation_num_members", 0)
        if M > 1 and self.latent_perturbation_log_sigma is not None:
            tokens_std = tokens.std().item()
            tokens_abs = tokens.abs().mean().item()

            sigma_log = torch.exp(self.latent_perturbation_log_sigma).item()

            #logger.info(
            #    f"tokens_std={tokens_std:.4f} "
            #    f"tokens_abs={tokens_abs:.4f} "
            #    f"sigma={sigma_log:.4f} "
            #    f"sigma/tokens_std={sigma_log/tokens_std:.4f}"
           #) )
            override = self.cf.get("latent_perturbation_sigma_override", None)
            #logger.info(f"sigma_override={override}")
            if override is not None:
                sigma = torch.tensor(
                    override,
                    device=tokens.device,
                    dtype=tokens.dtype,
                )
            else:
                sigma = torch.exp(self.latent_perturbation_log_sigma).to(tokens.dtype)
            #logger.info(f"effective_sigma={sigma.item()}")
            eps = torch.randn(M, B, H * Q, D, device=tokens.device, dtype=tokens.dtype)
            tokens_tiled = (tokens.unsqueeze(0) + sigma * eps).reshape(M * B, H * Q, D)
            latent_spread = (
                tokens_tiled.reshape(M, B, H * Q, D)
                .std(dim=0)
                .mean()
            )

            #logger.info(
            #    f"latent_spread={latent_spread.item():.6f}"
            #)
        else:
            M = 1
            tokens_tiled = tokens
        # tokens_tiled: [M*B, H, D]  (Q=1 so H*Q == H)
        assert tokens_tiled.shape == (M * B, H * Q, D)

        # ── Flatten to cell level ────────────────────────────────────────
        # tokens_flat[i*H + h]  =  D-dim token for (member i//B, batch i%B, cell h)
        tokens_flat = tokens_tiled.reshape(M * B, H, Q, D).flatten(0, 1)  # [M*B*H, Q, D]
        assert tokens_flat.shape == (M * B * H, Q, D)

        # ── Neighbour token gather ───────────────────────────────────────
        # hp_nbours[h, k] is the k-th neighbour cell index of cell h; values in [0, H).
        #
        # For the i-th (m, b) pair (i = m*B + b) the neighbour of cell h lives at flat row
        #   i*H + nbr   in tokens_flat   (NOT just at `nbr`)
        # We therefore add a per-(m,b) offset of i*H to every cell index.
        MB = M * B
        cell_offsets = torch.arange(MB, device=tokens.device).view(MB, 1, 1) * H
        idxs = (model_params.hp_nbours.unsqueeze(0) + cell_offsets).flatten(0, 1)  # [M*B*H, 9]
        assert idxs.shape == (MB * H, 9)
        assert idxs.min() >= 0 and idxs.max() < MB * H, \
            f"neighbour indices out of range [0, {MB * H}): min={idxs.min()}, max={idxs.max()}"

        # Gather neighbours: [M*B*H, 9, Q, D] → flatten (9, Q) → [M*B*H*9*Q, D]
        tokens_nbors = tokens_flat[idxs.flatten()].flatten(0, 1)  # [M*B*H * 9*Q, D]
        assert tokens_nbors.shape == (MB * H * 9 * Q, D)

        # KV sequence lengths for varlen attention.
        # attention.py does cumsum(x_kv_lens) before flash_attn, so these are INDIVIDUAL counts
        # (not already cumulative).  M*B*H groups × 9 neighbour cells (each with Q=1 token).
        tokens_nbors_lens = tokens_flat.new_zeros(MB * H + 1, dtype=torch.int32)
        tokens_nbors_lens[1:] = 9 * Q

        # ── Per-stream decoding ──────────────────────────────────────────
        for stream_name in self.stream_names:
            # Collect target coordinates for this stream + step across the batch.
            t_coords = [
                batch.samples[i_b].streams_data[stream_name].target_coords[step]
                for i_b in range(B)
            ]
            t_coords_lens = [len(t) for t in t_coords]  # per-batch-item point counts
            t_coords = torch.cat(t_coords)               # [P, coord_dim]
            P = t_coords.shape[0]

            if P == 0:
                continue

            tc_tokens = checkpoint(
                self.embed_target_coords[stream_name], t_coords, use_reentrant=False
            )  # [P, embed_dim]
            assert tc_tokens.shape[0] == P

            if torch.isnan(tc_tokens).any():
                logger.warning(
                    f"Skipping prediction for {stream_name} because"
                    f" of {torch.isnan(tc_tokens).sum()} NaN in tc_tokens."
                )
                pred = torch.tensor([], device=tc_tokens.device)

            else:
                # Build query-group lens for varlen attention.
                # tcs_lens = [0, c_0, ..., c_{N-1}] where N = B*H, individual counts.
                tcls = torch.cat([
                    sample.streams_data[stream_name].target_coords_lens[step]
                    for sample in batch.samples
                ])
                tcs_lens = torch.cat([torch.zeros(1, dtype=torch.int32, device=tcls.device), tcls])
                N = tcs_lens.shape[0] - 1   # must equal B*H for the KV pairing to be valid
                assert N == B * H, \
                    f"expected N={B * H} query groups (B*H), got N={N}"

                # Tile M times: M-major ordering matches tokens_tiled / tokens_nbors layout.
                #   tc_tokens_in: [M*P, embed_dim]  — [P pts for m=0 | P pts for m=1 | ...]
                #   tcs_lens_in:  [0, c_0,..,c_{N-1}, c_0,..,c_{N-1}, ...] (M repetitions)
                tc_tokens_in = tc_tokens.repeat(M, 1)
                tcs_lens_in = torch.cat([
                    torch.zeros(1, dtype=torch.int32, device=tcs_lens.device),
                    tcs_lens[1:].repeat(M),
                ])
                assert tc_tokens_in.shape == (M * P, tc_tokens.shape[-1])
                assert tcs_lens_in.shape[0] == M * N + 1

                if self.cf.decoder_type == "Linear":
                    pred = self.target_token_engines[stream_name](
                        tc_tokens_in,
                        tokens_tiled.reshape(-1, D),   # [M*B*H*Q, D]  all latent tokens flat
                        tcs_lens_in,
                    )
                    # [M*P, C] → [M, P, C]
                    assert pred.shape[0] == M * P
                    pred = pred.reshape(M, P, pred.shape[-1])
                else:
                    # Decode all M members in a SINGLE varlen call. Both query groups
                    # (tcs_lens_in) and KV groups (tokens_nbors_lens) are M-major with N=B*H
                    # groups per member, so member m's queries pair with member m's KVs. The
                    # decoder params are therefore used exactly ONCE per forward — no per-member
                    # reuse — which keeps the per-block activation checkpointing inside the
                    # engine DDP-safe (reuse + checkpointing is what trips "marked ready twice").
                    # The single call launches M*B*H attention groups; this must stay under the
                    # CUDA grid-dim cap (65535), which holds for M<=4 at healpix-5 with B=1.
                    assert M * N <= 65535, (
                        f"M*B*H={M * N} exceeds the CUDA grid-dim limit (65535); "
                        f"lower latent_perturbation_num_members"
                    )
                    tc_tokens_out = self.target_token_engines[stream_name](
                        latent=tokens_nbors,
                        output=tc_tokens_in,
                        latent_lens=tokens_nbors_lens,
                        output_lens=tcs_lens_in,
                        coordinates=t_coords.repeat(M, 1),
                    )  # [M*P, embed_dim]
                    pred = self.pred_heads[stream_name](tc_tokens_out)
                    # pred: [E, M*P, C]
                    # The M blocks in dim-1 are contiguous (repeat(M,1) ordering), so we can
                    # split on M and permute: [E, M*P, C] → [M, E, P, C] → [M*E, P, C]
                    E, pts_M, C = pred.shape
                    assert pts_M == M * P, f"expected {M * P} pts in pred, got {pts_M}"
                    pred_members = (
                        pred.reshape(E, M, P, C)
                        .permute(1, 0, 2, 3)
                    )
                        #.reshape(M * E, P, C)
                    #)
                    if M > 1:
                        output_spread = pred_members.std(dim=0).mean()
                        #logger.info(
                        #    f"{stream_name}: "
                        #    f"latent_spread={latent_spread.item():.6f} "
                        #    f"output_spread={output_spread.item():.6f} "
                        #    f"gain={output_spread.item()/latent_spread.item() + 1e-8:.6f}"
                        #)
                    pred = pred_members.reshape(M * E,P,C)

                assert pred.shape[1] == P, f"expected {P} points in dim=1, got {pred.shape[1]}"

            # ── Unpack into per-batch-item list ─────────────────────────
            # pred: [M or M*E, P, C]  →  list of B tensors, each [M or M*E, pts_b, C]
            pred = torch.split(pred, t_coords_lens, dim=1)
            output.add_physical_prediction(step, stream_name, pred)

        return output
