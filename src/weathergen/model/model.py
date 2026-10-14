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
import typing

import numpy as np
import torch
import torch.nn as nn

from weathergen.common.config import (
    Config,
    get_decoder_configs,
    get_encoder_configs,
    get_healpix_level,
    get_healpix_levels,
)
from weathergen.datasets.batch import BatchSamples
from weathergen.datasets.domain import Domain
from weathergen.datasets.utils import healpix_verts_rots, r3tos2
from weathergen.model.decoder import DecoderModule, PhysicalPrediction
from weathergen.model.encoder import EncoderModule, MultiEncoder
from weathergen.model.engines import (
    LatentPredictionHeadIdentity,
    LatentPredictionHeadMLP,
    LatentPredictionHeadTransformer,
    LatentState,
    MultiLevelForecastingEngine,
)
from weathergen.model.norms import AdaLayerNorm
from weathergen.model.utils import get_num_parameters
from weathergen.utils.utils import get_dtype

type StreamName = str


class ModelOutput:
    """
    Representation of model output
    """

    physical: list[dict[str, dict[StreamName, tuple[PhysicalPrediction, ...]]]]
    latent: list[dict[str, torch.Tensor | LatentState]]

    def __init__(self, len_output: int) -> None:
        self.physical = [{} for _ in range(len_output)]
        self.latent = [{} for _ in range(len_output)]

    def add_physical_prediction(
        self,
        fstep: int,
        decoder_name: str,
        stream_name: StreamName,
        pred: tuple[PhysicalPrediction, ...],
    ) -> None:
        self.physical[fstep].setdefault(decoder_name, {})[stream_name] = pred

    def add_latent_prediction(self, fstep: int, latent_name: str, pred: torch.Tensor) -> None:
        self.latent[fstep][latent_name] = pred

    def get_physical_prediction(
        self,
        fstep: int,
        decoder_name: str,
        stream_name: StreamName | None = None,
        sample_idx: int | None = None,
    ):
        pred = self.physical[fstep].get(decoder_name, {})
        if stream_name is not None:
            pred = pred.get(stream_name, None)
            if sample_idx is not None:
                assert sample_idx < len(pred), "Invalid sample index."
                pred = pred[sample_idx]
        return pred

    def get_latent_prediction(self, fstep: int):
        return self.latent[fstep]


def _rope_coordinates(cf: Config, device):
    if not cf.get("rope_2D", False):
        return None, None

    verts = torch.cat(
        [
            healpix_verts_rots(level, active_cells=Domain.from_config(cf, level).active_cells)[0]
            for level in get_healpix_levels(cf)
        ]
    )
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
        self.domain = Domain.from_config(cf, get_healpix_level(cf))
        self.num_healpix_cells = len(self.domain)
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

        self.rope_coords, self.rope_cell_coords = _rope_coordinates(cf, pe_embed.device)

        # Preserve per-cell identity even when RoPE is disabled or a cell is masked.
        pe_global = self.pe_global
        dim_embed = cf.ae_global_dim_embed
        xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=pe_global.device) / dim_embed
        cell_ids = torch.tensor(self.domain.active_cells, device=pe_global.device)
        pe_global[..., 0::2] = 0.5 * torch.sin(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=pe_global.device), xs)
        )
        pe_global[..., 0::2] += (
            torch.sin(torch.outer(cell_ids, xs))
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )
        pe_global[..., 1::2] = 0.5 * torch.cos(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=pe_global.device), xs)
        )
        pe_global[..., 1::2] += (
            torch.cos(torch.outer(cell_ids, xs))
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )
        self.q_cells_lens.fill_(1)
        self.q_cells_lens[0] = 0


class ModelParams(torch.nn.Module):
    """Independent encoder positional state and downstream grid parameters."""

    rope_coords: torch.Tensor | None

    def __init__(self, cf: Config):
        super().__init__()
        self.cf = cf
        encoder_configs = get_encoder_configs(cf)
        self.encoders = torch.nn.ModuleDict(
            {name: _EncoderParams(encoder_cf) for name, encoder_cf in encoder_configs.items()}
        )
        self.healpix_levels = get_healpix_levels(cf)
        self.domains = {level: Domain.from_config(cf, level) for level in self.healpix_levels}
        self.num_healpix_cells = sum(map(len, self.domains.values()))
        self.level_slices = {}
        offset = 0
        for level in self.healpix_levels:
            count = len(self.domains[level])
            self.level_slices[level] = slice(offset, offset + count)
            offset += count
        self.register_buffer("rope_coords", None)
        neighbours = {
            level: np.column_stack([np.arange(len(domain)), domain.neighbours_compact()])
            for level, domain in self.domains.items()
        }
        self.register_buffer(
            "hp_nbours",
            torch.tensor(
                np.concatenate(
                    [
                        neighbours[level] + self.level_slices[level].start
                        for level in self.healpix_levels
                    ]
                ),
                dtype=torch.int64,
            ),
        )
        # Only overlapping parent/child cells participate in cross-level attention.
        for coarse, fine in zip(self.healpix_levels, self.healpix_levels[1:], strict=False):
            parents = self.domains[coarse].to_compact[
                self.domains[fine].active_cells // (4 ** (fine - coarse))
            ]
            fine_cells = np.flatnonzero(parents >= 0)
            coarse_cells, counts = np.unique(parents[fine_cells], return_counts=True)
            for name, values, dtype in (
                ("fine", fine_cells, torch.int64),
                ("coarse", coarse_cells, torch.int64),
                ("counts", counts, torch.int32),
            ):
                self.register_buffer(
                    f"cascade_{name}_{coarse}_{fine}", torch.tensor(values, dtype=dtype)
                )
        for name, decoder_cf in get_decoder_configs(cf).items():
            level = decoder_cf.healpix_level
            indices = neighbours[level] + self.level_slices[level].start
            self.register_buffer(
                f"decode_{name}_indices", torch.tensor(indices.reshape(-1), dtype=torch.int64)
            )
            self.register_buffer(
                f"decode_{name}_lengths",
                torch.full(
                    (len(indices),),
                    indices.shape[1] * decoder_cf.ae_local_num_queries,
                    dtype=torch.int32,
                ),
            )

    def create(self, cf: Config) -> "ModelParams":
        self.reset_parameters(cf)
        return self

    def reset_parameters(self, cf: Config) -> None:
        for params in self.encoders.values():
            params.reset_parameters()
        self.rope_coords, _ = _rope_coordinates(cf, self.hp_nbours.device)


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

    decoders: Named physical readouts, each owning coordinate embeddings, target token engines,
        and prediction heads for its selected streams.
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

        self.healpix_levels = get_healpix_levels(cf)
        self.domains = {level: Domain.from_config(cf, level) for level in self.healpix_levels}
        self.num_healpix_cells = sum(map(len, self.domains.values()))

        self.cf = cf
        self.dtype = get_dtype(self.cf.attention_dtype)
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size

        self.decoders = nn.ModuleDict()
        self.encoder: MultiEncoder | None = None
        self.forecast_engine: MultiLevelForecastingEngine | None = None
        self.q_cells: torch.Tensor | None = None
        self.streams: dict[str, typing.Any] = cf.streams

        assert cf.get("forecast", {}).get("att_dense_rate", 1.0) == 1.0, (
            "Local attention not adapted for register tokens"
        )
        self.num_register_tokens = cf.num_register_tokens
        self.latent_heads = None
        self.latent_pre_norm = None
        # auxiliary tokens
        num_queries = cf.ae_local_num_queries
        self.class_token_idxs = list(
            range(
                cf.num_register_tokens * num_queries,
                (cf.num_register_tokens + cf.num_class_tokens) * num_queries,
            )
        )
        self.register_token_idxs = list(range(cf.num_register_tokens * num_queries))
        self.num_aux_tokens = (cf.num_register_tokens + cf.num_class_tokens) * num_queries
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

        self.encoder = MultiEncoder(
            cf, self.sources_size, self.targets_num_channels, self.targets_coords_size
        )

        self.forecast_engine = MultiLevelForecastingEngine(cf, cf.training_config, self.domains)

        loss_terms = [
            v.type for _, v in cf.training_config.losses.items() if v.get("enabled", True)
        ]
        if cf.validation_config.get("losses"):
            loss_terms += [
                v.type for _, v in cf.validation_config.losses.items() if v.get("enabled", True)
            ]

        if "LossPhysical" in loss_terms:
            target_channels = dict(zip(self.streams, self.targets_num_channels, strict=False))
            target_coords = dict(zip(self.streams, self.targets_coords_size, strict=False))
            self.decoders = nn.ModuleDict(
                {
                    name: DecoderModule(decoder_cf, target_channels, target_coords)
                    for name, decoder_cf in get_decoder_configs(cf).items()
                }
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

        encoders = [
            module for module in self.encoder.modules() if isinstance(module, EncoderModule)
        ]
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

        num_params_fe = get_num_parameters(self.forecast_engine)

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
        print(" Named physical decoders:")
        for name, decoder in self.decoders.items():
            print(f"   {name}: {get_num_parameters(decoder):,}")
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

    def forward(self, model_params: ModelParams, batch: BatchSamples) -> ModelOutput:
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
        output.add_latent_prediction(0, "posteriors", posteriors)

        # recover batch dimension and separate input_steps
        shape = (batch.get_num_source_steps(), len(batch), *tokens.shape[1:])
        # EmbeddingEngine packs input steps before batch samples.
        tokens = tokens.reshape(shape).sum(axis=0)

        # Allow for pushforward trick
        p_fwd = self.cf.training_config.get("forecast", {}).get("pushforward", False)
        # roll-out in latent space, iterate and generate output over requested output steps
        for step in batch.get_output_idxs():
            without_grad = p_fwd and self.training and step != max(batch.get_output_idxs())
            tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
            if self.forecast_engine.enabled:
                tokens = self.encoder.couple(tokens, model_params)
            if without_grad:
                continue

            # decoder predictions
            output = self.predict_decoders(model_params, step, tokens, batch, output)
            # latent predictions (raw and with SSL heads)
            output = self.predict_latent(model_params, step, tokens, batch, output)

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
        tokens: torch.Tensor,
        batch: BatchSamples,
        output: ModelOutput,
    ) -> ModelOutput:
        """Compute independent decoder- and stream-keyed physical predictions."""
        tokens = tokens[:, self.num_aux_tokens :]
        cells = tokens.reshape(
            len(batch), self.num_healpix_cells, self.cf.ae_local_num_queries, tokens.shape[-1]
        )
        for name, decoder in self.decoders.items():
            predictions = decoder(model_params, name, step, cells, batch)
            for stream_name, pred in predictions.items():
                output.add_physical_prediction(step, name, stream_name, pred)
        return output
