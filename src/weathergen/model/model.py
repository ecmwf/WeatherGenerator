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
import warnings

import astropy_healpix as hp
import astropy_healpix.healpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import (
    Config,
    get_decoder_healpix_level,
    get_encoder_configs,
    get_healpix_level,
)
from weathergen.datasets.batch import BatchSamples, MultiBatch
from weathergen.datasets.utils import healpix_verts_rots, hp_level_to_num_cells, r3tos2
from weathergen.model.encoder import MultiEncoder
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
from weathergen.model.layers import MLP, NamedLinear
from weathergen.model.norms import AdaLayerNorm
from weathergen.model.utils import exchange_latents, get_num_parameters
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


def _rope_coordinates(cf: Config, healpix_level: int, device):
    if not cf.get("rope_2D", False):
        return None, None

    verts, _ = healpix_verts_rots(healpix_level, 0.5, 0.5)
    cell_coords = r3tos2(verts.to(device)).to(get_dtype(cf.attention_dtype))
    num_queries = cf.ae_local_num_queries
    offset = (cf.num_register_tokens + cf.num_class_tokens) * num_queries
    coords = torch.zeros(
        1,
        offset + cell_coords.shape[0] * num_queries,
        2,
        device=device,
        dtype=cell_coords.dtype,
    )
    coords[:, offset:] = cell_coords.repeat_interleave(num_queries, dim=0)
    return coords, cell_coords


class _EncoderParams(torch.nn.Module):
    """Non-trainable positional state for one encoder."""

    rope_coords: torch.Tensor | None
    rope_cell_coords: torch.Tensor | None

    def __init__(self, cf: Config):
        super().__init__()
        self.cf = cf
        self.num_healpix_cells = hp_level_to_num_cells(cf.healpix_level)
        dtype = get_dtype(cf.attention_dtype)
        self.register_buffer(
            "pe_embed",
            torch.zeros(
                cf.get("ae_local_max_tokens_per_cell", 64), cf.ae_local_dim_embed, dtype=dtype
            ),
        )
        self.register_buffer(
            "pe_global",
            torch.zeros(
                self.num_healpix_cells,
                cf.ae_local_num_queries,
                cf.ae_global_dim_embed,
                dtype=dtype,
            ),
        )
        self.register_buffer("rope_coords", None)
        self.register_buffer("rope_cell_coords", None)
        self.register_buffer(
            "q_cells_lens", torch.ones(self.num_healpix_cells + 1, dtype=torch.int32)
        )

    def reset_parameters(self):
        cf = self.cf
        pe_embed = self.pe_embed
        max_tokens, dim_embed = pe_embed.shape
        token_idx_bias = 16
        freq_bias = 8
        position = torch.arange(
            token_idx_bias, token_idx_bias + max_tokens, device=pe_embed.device
        ).unsqueeze(1)
        div = torch.exp(
            torch.arange(freq_bias, freq_bias + dim_embed, 2, device=pe_embed.device)
            * -(math.log(max_tokens) / dim_embed)
        )
        pe_embed[:, 0::2] = torch.sin(position * div[: pe_embed[:, 0::2].shape[1]])
        pe_embed[:, 1::2] = torch.cos(position * div[: pe_embed[:, 1::2].shape[1]])

        self.rope_coords, self.rope_cell_coords = _rope_coordinates(
            cf, cf.healpix_level, pe_embed.device
        )

        # Preserve per-cell identity even when RoPE is disabled or a cell is masked.
        pe_global = self.pe_global
        dim_embed = cf.ae_global_dim_embed
        xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=pe_global.device) / dim_embed
        pe_global[..., 0::2] = 0.5 * torch.sin(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=pe_global.device), xs)
        )
        pe_global[..., 0::2] += (
            torch.sin(
                torch.outer(torch.arange(self.num_healpix_cells, device=pe_global.device), xs)
            )
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )
        pe_global[..., 1::2] = 0.5 * torch.cos(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=pe_global.device), xs)
        )
        pe_global[..., 1::2] += (
            torch.cos(
                torch.outer(torch.arange(self.num_healpix_cells, device=pe_global.device), xs)
            )
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )
        self.q_cells_lens.fill_(1)
        self.q_cells_lens[0] = 0


class _GridParams(torch.nn.Module):
    """One neighbor table per distinct native grid."""

    def __init__(self, level: int):
        super().__init__()
        self.healpix_level = level
        self.num_healpix_cells = hp_level_to_num_cells(level)
        self.register_buffer("hp_nbours", torch.empty(self.num_healpix_cells, 9, dtype=torch.int32))

    def reset_parameters(self):
        with warnings.catch_warnings(action="ignore"):
            neighbours = hp.neighbours(
                np.arange(self.num_healpix_cells), 2**self.healpix_level, order="nested"
            ).transpose()
        for i, row in enumerate(neighbours):
            row[row == -1] = i
        self.hp_nbours[:, 0] = torch.arange(self.num_healpix_cells, device=self.hp_nbours.device)
        self.hp_nbours[:, 1:] = torch.from_numpy(neighbours).to(self.hp_nbours.device)


class ModelParams(torch.nn.Module):
    """Independent encoder positional state and downstream grid parameters."""

    def __init__(self, cf: Config):
        super().__init__()
        self.cf = cf
        encoder_configs = get_encoder_configs(cf)
        self.encoders = torch.nn.ModuleDict(
            {name: _EncoderParams(encoder_cf) for name, encoder_cf in encoder_configs.items()}
        )
        self.healpix_level = get_healpix_level(cf)
        self.num_healpix_cells = hp_level_to_num_cells(self.healpix_level)
        self.grids = torch.nn.ModuleDict(
            {
                str(level): _GridParams(level)
                for level in sorted({encoder.healpix_level for encoder in encoder_configs.values()})
            }
        )

    def create(self, cf: Config) -> "ModelParams":
        self.reset_parameters(cf)
        return self

    def reset_parameters(self, cf: Config) -> None:
        for params in self.encoders.values():
            params.reset_parameters()
        for grid in self.grids.values():
            grid.reset_parameters()


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
        self.encoder_configs = get_encoder_configs(cf)
        self.encoder_levels = {
            name: encoder.healpix_level for name, encoder in self.encoder_configs.items()
        }
        for name, encoder in self.encoder_configs.items():
            for key in (
                "ae_global_dim_embed",
                "ae_local_num_queries",
                "num_register_tokens",
                "num_class_tokens",
            ):
                if encoder[key] != cf[key]:
                    raise ValueError(f"Encoder {name!r} must use the model's {key}.")
        self.decoder_levels = {name: get_decoder_healpix_level(cf, name) for name in cf.streams}
        self.decoder_latent = cf.get("decoder_latent", "common")
        if self.decoder_latent not in ("common", "encoder"):
            raise ValueError("decoder_latent must be 'common' or 'encoder'.")

        self.healpix_level = get_healpix_level(cf)
        self.num_healpix_cells = hp_level_to_num_cells(self.healpix_level)

        self.cf = cf
        self.dtype = get_dtype(self.cf.attention_dtype)
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size

        self.embed_target_coords = None
        self.encoder: MultiEncoder | None = None
        self.forecast_engines = nn.ModuleDict()
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
        queries = cf.ae_local_num_queries
        self.class_token_idxs = list(
            range(
                cf.num_register_tokens * queries,
                (cf.num_register_tokens + cf.num_class_tokens) * queries,
            )
        )
        self.register_token_idxs = list(range(cf.num_register_tokens * queries))
        self.num_aux_tokens = (cf.num_register_tokens + cf.num_class_tokens) * queries
        self.aux_token_idxs = list(range(self.num_aux_tokens))

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

        self.encoder = MultiEncoder(cf, self.sources_size)

        self.forecast_engines = nn.ModuleDict(
            {
                str(level): (
                    ForecastingEngine(cf, cf.training_config, hp_level_to_num_cells(level))
                    if cf.fe_num_blocks > 0
                    else IdentityEngine()
                )
                for level in sorted(set(self.encoder_levels.values()))
            }
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

    def print_num_parameters(self) -> None:
        """Print number of parameters for entire model and each module used to build the model"""

        encoders = self.encoder.encoders.values()
        num_params_embed = [
            sum(
                get_num_parameters(encoder.embed_engine.embeds[name])
                for encoder in encoders
                if name in encoder.embed_engine.embeds
            )
            for name in self.streams.keys()
        ]
        num_params_total = get_num_parameters(self)
        num_params_ae_local = sum(
            get_num_parameters(encoder.ae_local_engine.ae_local_blocks) for encoder in encoders
        )
        num_params_ae_global = sum(
            get_num_parameters(encoder.ae_global_engine.ae_global_blocks) for encoder in encoders
        )

        num_params_q_cells = sum(
            encoder.q_cells.numel() for encoder in encoders if encoder.q_cells.requires_grad
        )
        num_params_ae_adapter = sum(
            get_num_parameters(encoder.ae_local_global_engine) for encoder in encoders
        )

        num_params_ae_aggregation = sum(
            get_num_parameters(encoder.ae_aggregation_engine.ae_aggregation_blocks)
            for encoder in encoders
        )

        num_params_latent_heads = get_num_parameters(self.latent_heads)
        num_params_latent_heads += get_num_parameters(self.latent_pre_norm)

        num_params_fe = sum(
            get_num_parameters(engine.fe_blocks) for engine in self.forecast_engines.values()
        )

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

    def forward(self, model_params: ModelParams, batch: MultiBatch) -> ModelOutput:
        """Forward pass of the model

        Tokens are processed through the model components, which were defined in the create method.
        Args:
            model_params : Query and embedding parameters
            batch : Shared data and encoder-named model batches
        Returns:
            A list containing all prediction results
        """

        samples = batch.batch.get_source_samples()
        output = ModelOutput(samples.get_output_len())

        latents, posteriors = self.encoder(model_params, batch)
        output.add_latent_prediction(0, "posteriors", posteriors)
        for name, tokens in latents.items():
            encoder_samples = batch.batches[name].get_source_samples()
            shape = (
                encoder_samples.get_num_source_steps(),
                len(encoder_samples),
                *tokens.shape[1:],
            )
            latents[name] = tokens.reshape(shape).sum(dim=0)
        latents, common_tokens = exchange_latents(
            latents,
            self.encoder_levels,
            self.healpix_level,
            self.num_aux_tokens,
            self.cf.ae_local_num_queries,
        )

        # Allow for pushforward trick
        p_fwd = self.cf.training_config.get("forecast", {}).get("pushforward", False)
        # roll-out in latent space, iterate and generate output over requested output steps
        for step in samples.get_output_idxs():
            latents = {
                name: self.forecast_engines[str(self.encoder_levels[name])](
                    tokens, step, model_params.encoders[name].rope_coords
                )
                for name, tokens in latents.items()
            }
            latents, common_tokens = exchange_latents(
                latents,
                self.encoder_levels,
                self.healpix_level,
                self.num_aux_tokens,
                self.cf.ae_local_num_queries,
            )
            without_grad = p_fwd and self.training and step != max(samples.get_output_idxs())
            if without_grad:
                continue
            output = self.predict_decoders(
                model_params, step, common_tokens, latents, samples, output
            )
            output = self.predict_latent(model_params, step, common_tokens, samples, output)

        return output

    def predict_latent(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: BatchSamples,
        output: ModelOutput,
    ) -> ModelOutput:
        """
        Compute latent predictions
        """

        # safe latent prediction
        tokens_post_norm = self.latent_pre_norm(tokens) if step == 0 else None
        latent_state = self.tokens_to_latent_state(tokens_post_norm, tokens)
        output.add_latent_prediction(step, "latent_state", latent_state)

        # latent predictions for SSL training
        for name, head in self.latent_heads.items():
            output.add_latent_prediction(step, name, head(latent_state))

        return output

    def predict_decoders(
        self,
        model_params: ModelParams,
        step: int,
        common_tokens: torch.Tensor,
        latents: dict[str, torch.Tensor],
        batch: BatchSamples,
        output: ModelOutput,
    ) -> ModelOutput:
        """Decode each stream on its configured common or encoder-owned native grid."""
        # Empty dicts evaluate to False in python
        if not self.pred_heads:
            return output

        batch_size = len(batch)
        prepared = {}

        # pair with tokens from assimilation engine to obtain target tokens
        for stream_name in self.streams.keys():
            if is_stream_forcing(self.streams[stream_name]):
                continue
            # extract target coords for current stream and fstep and convert to one tensor
            t_coords = [
                batch.samples[i_b].streams_data[stream_name].target_coords[step]
                for i_b in range(batch_size)
            ]
            t_coords_lens = [len(t) for t in t_coords]
            t_coords = torch.cat(t_coords)

            if len(t_coords) == 0:
                continue

            name = (
                None
                if self.decoder_latent == "common"
                else self.streams[stream_name].get("encoder", "default")
            )
            if name not in prepared:
                selected = common_tokens if name is None else latents[name]
                level = self.decoder_levels[stream_name]
                cells = hp_level_to_num_cells(level)
                queries = self.cf.ae_local_num_queries
                spatial = selected[:, self.num_aux_tokens :].reshape(
                    batch_size, cells, queries, selected.shape[-1]
                )
                neighbors = neighbor_lens = None
                if self.cf.decoder_type != "Linear":
                    neighbors = spatial[:, model_params.grids[str(level)].hp_nbours.long()]
                    neighbors = neighbors.reshape(-1, selected.shape[-1])
                    neighbor_lens = torch.full(
                        (batch_size * cells + 1,),
                        9 * queries,
                        dtype=torch.int32,
                        device=selected.device,
                    )
                    neighbor_lens[0] = 0
                prepared[name] = (spatial.flatten(0, 2), neighbors, neighbor_lens)
            tokens, tokens_nbors, tokens_nbors_lens = prepared[name]

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
                        tokens,
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
