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
import typing

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.domain import Domain
from weathergen.datasets.domain_pyramid import build_domain_pyramid
from weathergen.datasets.utils import healpix_verts_rots, r3tos2
from weathergen.model.encoder import EncoderModule
from weathergen.model.engines import (
    BilinearDecoder,
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
from weathergen.model.latent_cascade import LatentCascade
from weathergen.model.layers import MLP, NamedLinear
from weathergen.model.norms import AdaLayerNorm
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

    def __init__(self, len_output: int) -> None:
        self.physical = [{} for _ in range(len_output)]
        self.latent = [{} for _ in range(len_output)]

    def add_physical_prediction(
        self, fstep: int, stream_name: StreamName, pred: torch.Tensor
    ) -> None:
        self.physical[fstep][stream_name] = pred

    def add_latent_prediction(self, fstep: int, latent_name: str, pred: torch.Tensor) -> None:
        self.latent[fstep][latent_name] = pred

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


def latent_level_name(level: int) -> str:
    """Name of a coarser latent level in the model's latent predictions."""
    return f"latent_state_hl{level}"


class ModelParams(torch.nn.Module):
    """Creation of query and embedding parameters of the model."""

    def __init__(self, cf, level: int | None = None, domain: Domain | None = None) -> None:
        """
        Args:
            cf : Configuration
            level, domain : latent level and its domain (multi-resolution latent, see
                ModelParamsPyramid). Default: cf.healpix_level and the `domain:` block.
        """
        super(ModelParams, self).__init__()

        self.cf = cf

        self.healpix_level = cf.healpix_level if level is None else level
        # healpix cells of the (possibly regional) domain
        self.domain = Domain.from_config(cf) if domain is None else domain
        assert self.domain.healpix_level == self.healpix_level
        self.num_healpix_cells = len(self.domain)
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

        # HEALPix neighbours (compact indexing; missing and out-of-domain neighbours -> self)
        temp = self.domain.neighbours_compact()
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

    def params_for(self, level: int) -> "ModelParams":
        """Parameters of a latent level (single latent level: this object)."""
        assert level == self.healpix_level, f"no model parameters for latent level {level}"
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
            if not self.domain.is_global:
                verts = verts[torch.from_numpy(self.domain.active_cells)]
            coords = r3tos2(verts.to(self.rope_coords.device)).to(self.rope_coords.dtype)
            if not self.domain.is_global:
                # r3tos2 returns the healpix azimuth in (-pi, pi]. Because of the +180 deg
                # shift in theta_phi_to_standard_coords, that seam lies on the Greenwich
                # meridian; move it to 180 deg geographic so that a regional domain
                # crossing 0 deg (e.g. Scandinavia) has continuous RoPE coordinates.
                coords[:, 1] = torch.remainder(coords[:, 1], 2 * torch.pi)
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

        # healpix neighborhood structure (compact indexing, see __init__)
        temp = self.domain.neighbours_compact()
        # nbors *and* self
        self.hp_nbours.data[:, 0] = torch.arange(temp.shape[0], device=self.hp_nbours.device)
        self.hp_nbours.data[:, 1:] = torch.from_numpy(temp).to(self.hp_nbours.device)

        # precompute for varlen attention
        self.q_cells_lens.data.fill_(1)
        self.q_cells_lens.data[0] = 0

        # ensure all params have grad set to False

        return


class ModelParamsPyramid(torch.nn.Module):
    """
    Model parameters of a multi-resolution latent: one `ModelParams` per latent level,
    each sized to that level's domain (pe_global, rope coords, hp_nbours, q_cells_lens).

    `params_for(level)` returns the tables of one level. Any other attribute is read from
    the finest level (e.g. the level-independent `pe_embed`).
    """

    def __init__(self, cf) -> None:
        super().__init__()
        self.cf = cf
        self.pyramid = build_domain_pyramid(cf)
        self.levels = self.pyramid.levels
        self._primary_level = self.pyramid.finest
        self._params = torch.nn.ModuleDict(
            {
                str(lvl): ModelParams(cf, level=lvl, domain=self.pyramid.domain(lvl))
                for lvl in self.levels
            }
        )

    def params_for(self, level: int) -> ModelParams:
        """The ModelParams tables of a latent level."""
        return self._params[str(level)]

    def create(self, cf: Config) -> "ModelParamsPyramid":
        self.reset_parameters(cf)
        return self

    def reset_parameters(self, cf: Config) -> None:
        for p in self._params.values():
            p.reset_parameters(cf)

    def __getattr__(self, name):
        # nn.Module attributes (_params, cf, pyramid, ...) resolve normally; anything else
        # is delegated to the finest level
        try:
            return super().__getattr__(name)
        except AttributeError:
            params = super().__getattr__("_params")
            return getattr(params[str(self.__dict__["_primary_level"])], name)


def build_model_params(cf) -> ModelParams | ModelParamsPyramid:
    """ModelParams for a single latent level, ModelParamsPyramid for `latent_levels`."""
    if build_domain_pyramid(cf).is_single:
        return ModelParams(cf)
    return ModelParamsPyramid(cf)


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

        # latent levels (one unless `latent_levels` is configured); the single-level
        # attributes refer to the finest level
        self.domain_pyramid = build_domain_pyramid(cf)
        self.healpix_level = self.domain_pyramid.finest
        self.num_healpix_cells = len(self.domain_pyramid.domain(self.healpix_level))

        self.cf = cf
        self.dtype = get_dtype(self.cf.attention_dtype)
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size

        self.embed_target_coords = None
        self.encoder: EncoderModule | None = None
        self.forecast_engine: ForecastingEngine | IdentityEngine | None = None
        # multi-resolution latent: forecast engines of the coarser levels + level coupling
        self._forecast_per_level: torch.nn.ModuleDict | None = None
        self.forecast_cascade: LatentCascade | None = None
        self.pred_heads = None
        self.q_cells: torch.Tensor | None = None
        self.streams: dict[str, typing.Any] = cf.streams
        self.target_token_engines = None

        assert cf.get("forecast", {}).get("att_dense_rate", 1.0) == 1.0, (
            "Local attention not adapted for register tokens"
        )
        self.num_register_tokens = cf.num_register_tokens
        self.latent_heads = None
        self.latent_pre_norm = None
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

        # multi-resolution latent: one forecast engine per coarser level (own weights, the
        # finest level uses forecast_engine) and a latent cascade after every forecast step
        if not self.domain_pyramid.is_single:
            if cf.fe_num_blocks > 0:
                self._forecast_per_level = torch.nn.ModuleDict(
                    {
                        str(lvl): ForecastingEngine(
                            cf, mode_cfg, len(self.domain_pyramid.domain(lvl))
                        )
                        for lvl in self.domain_pyramid.levels
                        if lvl != self.healpix_level
                    }
                )
            self.forecast_cascade = LatentCascade.from_config(
                self.domain_pyramid, cf, "forecast_cascade"
            )

        # embed coordinates yielding one query token for each target token
        dropout_rate = cf.embed_dropout_rate
        self.embed_target_coords = torch.nn.ModuleDict()
        self.target_token_engines = torch.nn.ModuleDict()
        self.pred_heads = torch.nn.ModuleDict()

        # determine stream names once so downstream components use consistent keys
        loss_terms = [
            v.type for _, v in cf.training_config.losses.items() if v.get("enabled", True)
        ]
        if cf.validation_config.get("losses"):
            loss_terms += [
                v.type for _, v in cf.validation_config.losses.items() if v.get("enabled", True)
            ]

        if "LossPhysical" in loss_terms:
            for i_stream, (stream_name, si) in enumerate(self.streams.items()):
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
            for i_stream, (stream_name, si) in enumerate(self.streams.items()):
                # skip decoder if channels are empty
                if is_stream_forcing(si):
                    continue

                pred_spatial_shared = si.get("pred_spatial_shared")
                if pred_spatial_shared is not None:
                    if pred_spatial_shared not in self.streams.keys():
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

                    assert pred_spatial_shared in self.streams.keys()
                    si_other = self.streams[pred_spatial_shared]
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
        self.latent_pre_norm = nn.LayerNorm(cf.ae_global_dim_embed)

        ssl_losses_cfgs = [
            v
            for _, v in cf.training_config.losses.items()
            if v.type == "LossLatentSSLStudentTeacher" and v.get("enabled", True)
        ]

        # TODO: support multiple LossLatentSSLStudentTeacher terms
        assert len(ssl_losses_cfgs) <= 1, "To be implemented."
        for ssl_target_losses in ssl_losses_cfgs:
            self.latent_pre_norm = nn.LayerNorm(cf.ae_global_dim_embed)
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

        return self

    def reset_parameters(self):
        def _reset_params(module):
            if isinstance(module, AdaLayerNorm | nn.Linear | nn.LayerNorm):
                module.reset_parameters()
            else:
                pass

        self.apply(_reset_params)

        # learnable cell queries are plain Parameters (one bank per latent level), not
        # covered by the line above: re-draw them (uninitialised memory after to_empty)
        self.encoder.reset_q_cells()

        # multi-resolution latent: the line above resets every Linear, including the
        # zero-initialised last projections of the cascade blocks; re-apply those
        for cascade in self.latent_cascades():
            cascade.init_residual_branches()

    def latent_cascades(self) -> list[LatentCascade]:
        """The latent cascades of a multi-resolution latent (empty for one level)."""
        cascades = [getattr(self.encoder, "latent_cascade", None), self.forecast_cascade]
        return [c for c in cascades if c is not None]

    def print_num_parameters(self) -> None:
        """Print number of parameters for entire model and each module used to build the model"""

        num_params_embed = [
            get_num_parameters(self.encoder.embed_engine.embeds[name])
            for name in self.streams.keys()
        ]
        num_params_total = get_num_parameters(self)
        num_params_ae_local = get_num_parameters(self.encoder.ae_local_engine.ae_local_blocks)
        num_params_ae_global = get_num_parameters(self.encoder.ae_global_engine.ae_global_blocks)

        num_params_q_cells = (
            np.prod(self.encoder.q_cells.shape) if self.encoder.q_cells.requires_grad else 0
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
            for name in self.streams.keys()
        ]
        mdict = self.target_token_engines
        num_params_tte = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.streams.keys()
        ]
        mdict = self.pred_heads
        num_params_preds = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.streams.keys()
        ]

        print("-----------------")
        print(f"Total number of trainable parameters: {num_params_total:,}")
        print("Number of parameters:")
        print("  Embedding networks:")
        [
            print("    {} : {:,}".format(si["name"], np))
            for si, np in zip(self.streams.values(), num_params_embed, strict=False)
        ]
        print(f" Local assimilation engine: {num_params_ae_local:,}")
        print(f" Local-global adapter: {num_params_ae_adapter:,}")
        print(f" Learnable queries: {num_params_q_cells:,}")
        print(f" Query Aggregation engine: {num_params_ae_aggregation:,}")
        print(f" Global assimilation engine: {num_params_ae_global:,}")
        print(f" Latent prediction heads and pre-norm: {num_params_latent_heads:,}")
        print(f" Forecast engine: {num_params_fe:,}")
        print(" coordinate embedding, prediction networks and prediction heads:")
        zps = zip(
            self.streams.keys(),
            num_params_embed_tcs,
            num_params_tte,
            num_params_preds,
            strict=False,
        )
        for stream_name, np0, np1, np2 in zps:
            print(f"   {stream_name} : {np0:,} / {np1:,} / {np2:,}")

        # multi-resolution latent: the lines above are the finest level; coarser levels and
        # the cascades are listed here
        if not self.domain_pyramid.is_single:
            enc = self.encoder
            print(f" Multi-resolution latent, finest level hl{self.healpix_level} listed above.")
            for lvl in self.domain_pyramid.levels:
                if lvl == self.healpix_level:
                    continue
                key = str(lvl)
                print(f"  Level hl{lvl}:")
                if not enc.share_local_assimilation:
                    n_local = get_num_parameters(enc._ae_local_per_level[key])
                    n_adapter = get_num_parameters(enc._ae_local_global_per_level[key])
                    print(f"   Local assimilation engine: {n_local:,}")
                    print(f"   Local-global adapter: {n_adapter:,}")
                q = enc._q_cells_param_dict[key]
                print(f"   Learnable queries: {(q.numel() if q.requires_grad else 0):,}")
                n_agg = get_num_parameters(enc._ae_aggregation_per_level[key])
                print(f"   Query Aggregation engine: {n_agg:,}")
                n_global = get_num_parameters(enc._ae_global_per_level[key])
                print(f"   Global assimilation engine: {n_global:,}")
                if self._forecast_per_level is not None:
                    n_fe = get_num_parameters(self._forecast_per_level[key])
                    print(f"   Forecast engine: {n_fe:,}")
            for name, cascade in (
                ("Latent cascade", enc.latent_cascade),
                ("Forecast cascade", self.forecast_cascade),
            ):
                print(f"  {name}: {get_num_parameters(cascade):,}")
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

        tokens, posteriors = self.encoder(model_params, batch)

        # multi-resolution latent: the encoder returns one latent per level. The finest
        # level plays the role of the single latent (latent outputs, SSL heads); all levels
        # are forecast and read by the decoders.
        tokens_lvl = None
        if isinstance(tokens, dict):
            tokens_lvl, posteriors = tokens, posteriors[self.healpix_level]

        output.add_latent_prediction(0, "posteriors", posteriors)

        # recover batch dimension and separate input_steps
        if tokens_lvl is None:
            shape = (len(batch), batch.get_num_source_steps(), *tokens.shape[1:])
            # collapse along input step dimension
            tokens = tokens.reshape(shape).sum(axis=1)
        else:
            tokens_lvl = {
                lvl: t.reshape((len(batch), batch.get_num_source_steps(), *t.shape[1:])).sum(axis=1)
                for lvl, t in tokens_lvl.items()
            }
            tokens = tokens_lvl[self.healpix_level]

        # Allow for pushforward trick
        p_fwd = self.cf.training_config.get("forecast", {}).get("pushforward", False)
        # roll-out in latent space, iterate and generate output over requested output steps
        for step in batch.get_output_idxs():
            without_grad = p_fwd and self.training and step != max(batch.get_output_idxs())
            if without_grad:
                # Pushforward mode: advance tokens without grad; no decoding with torch.no_grad():
                if tokens_lvl is None:
                    tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
                else:
                    tokens_lvl = self.forecast_levels(model_params, tokens_lvl, step)
                    tokens = tokens_lvl[self.healpix_level]
                continue

            if tokens_lvl is None:
                tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
            else:
                tokens_lvl = self.forecast_levels(model_params, tokens_lvl, step)
                tokens = tokens_lvl[self.healpix_level]
            # decoder predictions
            output = self.predict_decoders(
                model_params, step, tokens, batch, output, tokens_lvl=tokens_lvl
            )
            # latent predictions (raw and with SSL heads)
            output = self.predict_latent(
                model_params, step, tokens, batch, output, tokens_lvl=tokens_lvl
            )

        return output

    def forecast_levels(self, model_params, tokens_lvl: dict, step: int) -> dict:
        """
        Multi-resolution latent: advance every level's latent with its own forecast engine,
        then couple the levels with the forecast cascade (inert with num_cycles: 0).
        """
        out = {}
        for lvl, t in tokens_lvl.items():
            if lvl == self.healpix_level or self._forecast_per_level is None:
                engine = self.forecast_engine
            else:
                engine = self._forecast_per_level[str(lvl)]
            out[lvl] = engine(t, step, model_params.params_for(lvl).rope_coords)
        return self.forecast_cascade(out)

    def predict_latent(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: ModelBatch,
        output: ModelOutput,
        tokens_lvl: dict | None = None,
    ) -> ModelOutput:
        """
        Compute latent predictions

        tokens_lvl : multi-resolution latent only: {level: tokens}. The finest level is the
            latent_state above; every coarser level is also stored, as the raw latent
            "latent_state_hl<level>" (no SSL heads), so that it can be written to the output.
        """

        # safe latent prediction
        tokens_post_norm = self.latent_pre_norm(tokens) if step == 0 else None
        latent_state = self.tokens_to_latent_state(tokens_post_norm, tokens)
        output.add_latent_prediction(step, "latent_state", latent_state)

        # latent predictions for SSL training
        for name, head in self.latent_heads.items():
            output.add_latent_prediction(step, name, head(latent_state))

        if tokens_lvl is not None:
            for lvl, tokens_l in tokens_lvl.items():
                if lvl != self.healpix_level:
                    output.add_latent_prediction(
                        step, latent_level_name(lvl), self.tokens_to_latent_state(None, tokens_l)
                    )

        return output

    def predict_decoders(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: ModelBatch,
        output: ModelOutput,
        tokens_lvl: dict | None = None,
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
            tokens_lvl : multi-resolution latent only: {level: tokens}; each stream then
                reads the latent levels listed in its decode levels (see
                `predict_decoders_kv_multi_level`)
        Returns:
            Prediction output tokens in physical representation for each target_coords.
        """
        # Empty dicts evaluate to False in python
        if not self.pred_heads:
            return output

        # remove register  and class tokens
        tokens = tokens[:, self.num_aux_tokens :]

        # get 1-ring neighborhood for prediction
        batch_size = len(batch)
        s = [batch_size, self.num_healpix_cells, self.cf.ae_local_num_queries, tokens.shape[-1]]
        idxs = model_params.hp_nbours.unsqueeze(0).repeat((batch_size, 1, 1)).flatten(0, 1)
        tokens_nbors = tokens.reshape(s).flatten(0, 1)[idxs.flatten()].flatten(0, 1)
        # TODO: precompute in model_params?
        tokens_nbors_lens = torch.full(
            (s[0] * s[1] + 1,), fill_value=9, dtype=torch.int32, device=tokens_nbors.device
        )
        tokens_nbors_lens[0] = 0
        tokens_flat = tokens.reshape(-1, s[-1])

        # pair with tokens from assimilation engine to obtain target tokens
        for stream_name in self.streams.keys():
            # multi-resolution latent: key/value pool of this stream's decoder
            if tokens_lvl is not None:
                tokens_nbors, tokens_nbors_lens, tokens_flat = self.predict_decoders_kv_multi_level(
                    model_params, stream_name, tokens_lvl, batch_size
                )

            # extract target coords for current stream and fstep and convert to one tensor
            t_coords = [
                batch.samples[i_b].streams_data[stream_name].target_coords[step]
                for i_b in range(batch_size)
            ]
            t_coords_lens = [len(t) for t in t_coords]
            t_coords = torch.cat(t_coords)

            if len(t_coords) == 0:
                continue

            # embed token coords
            tc_embed = self.embed_target_coords[stream_name]
            tc_tokens = checkpoint(tc_embed, t_coords, use_reentrant=False)

            # skip when coordinate embeddings yields nan (i.e. the coord embedding network diverged)
            if torch.isnan(tc_tokens).any():
                logger.warning(
                    (
                        f"Skipping prediction for {stream_name} because",
                        f" of {torch.isnan(tc_tokens).sum()} NaN in tc_tokens.",
                    )
                )
                pred = torch.tensor([], device=tc_tokens.device)

            # skip empty lengths
            elif tc_tokens.shape[0] == 0:
                pred = torch.tensor([], device=tc_tokens.device)

            else:
                # lens for varlen attention
                tcls = torch.cat(
                    [
                        sample.streams_data[stream_name].target_coords_lens[step]
                        for sample in batch.samples
                    ]
                )
                tcs_lens = torch.cat([torch.zeros(1, dtype=torch.int32, device=tcls.device), tcls])

                if self.cf.decoder_type == "Linear":
                    pred = self.target_token_engines[stream_name](
                        tc_tokens,
                        tokens_flat,  # collapse the batch and token dimensions
                        tcs_lens,
                    ).unsqueeze(0)  # add ensemble dim: shape is then [1, preds_per_coord, channels]
                else:
                    tc_tokens = self.target_token_engines[stream_name](
                        latent=tokens_nbors,
                        output=tc_tokens,
                        latent_lens=tokens_nbors_lens,
                        output_lens=tcs_lens,
                        coordinates=t_coords,
                    )

                    # final prediction head to map back to physical space
                    pred = self.pred_heads[stream_name](tc_tokens)

            # recover batch dimension (ragged, so as list)
            pred = torch.split(pred, t_coords_lens, dim=1)
            output.add_physical_prediction(step, stream_name, pred)

        return output

    def predict_decoders_kv_multi_level(
        self, model_params, stream_name: str, tokens_lvl: dict, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Key/value pool of a stream's decoder for the multi-resolution latent.

        The target coordinates of a stream are tokenized at its encode level `ls`, so the
        decoder has one varlen segment per cell of that level (and batch sample). Each
        segment holds, for every decode level `l` of the stream:
          * l == ls : the cell's own 1-ring (cell + 8 neighbours);
          * l <  ls : the 1-ring of the cell's ancestor at level l (exact HEALPix nested
                      parent), i.e. the large-scale context. If the ancestor lies outside
                      level l's domain, this part is left out of the segment (masked).
        Default for a fine stream over levels [5, 8]: 9 + 9 = 18 keys per cell.

        Returns (tokens_nbors, tokens_nbors_lens, tokens_flat) with the layout used by
        the target prediction engines (and the Linear decoder).
        """
        ls = self.encoder.stream_encode_level[stream_name]
        nq = self.cf.ae_local_num_queries
        num_cells_s = len(self.domain_pyramid.domain(ls))

        parts, valids = [], []
        for lvl in self.encoder.stream_decode_levels[stream_name]:
            toks_l = tokens_lvl[lvl][:, self.num_aux_tokens :]
            dim = toks_l.shape[-1]
            num_cells_l = len(self.domain_pyramid.domain(lvl))
            hp_nbours = model_params.params_for(lvl).hp_nbours.to(torch.long)
            dev = toks_l.device
            # 1-ring of every cell of level lvl, with an offset per batch sample
            b_off = (torch.arange(batch_size, device=dev) * num_cells_l).view(-1, 1, 1)
            idxs = (hp_nbours.unsqueeze(0) + b_off).flatten()
            ring = toks_l.reshape(batch_size * num_cells_l, nq, dim)[idxs]
            ring = ring.reshape(batch_size * num_cells_l, hp_nbours.shape[1] * nq, dim)

            if lvl == ls:
                valid = torch.ones(batch_size * num_cells_s, dtype=torch.bool, device=dev)
            else:
                anc = torch.as_tensor(
                    self.domain_pyramid.ancestor_of(ls, lvl), dtype=torch.long, device=dev
                )
                rows = anc.clamp(min=0).repeat(batch_size) + (
                    torch.arange(batch_size, device=dev).repeat_interleave(num_cells_s)
                    * num_cells_l
                )
                ring = ring[rows]
                valid = (anc >= 0).repeat(batch_size)
            parts.append(ring)
            valids.append(valid.unsqueeze(1).expand(-1, ring.shape[1]))

        kv = torch.cat(parts, dim=1)  # (batch_size * num_cells_s, keys per cell, dim)
        mask = torch.cat(valids, dim=1)
        tokens_nbors = kv[mask]
        tokens_nbors_lens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=kv.device),
                mask.sum(dim=1).to(torch.int32),
            ]
        )
        toks_s = tokens_lvl[ls][:, self.num_aux_tokens :]
        return tokens_nbors, tokens_nbors_lens, toks_s.reshape(-1, toks_s.shape[-1])
