# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import dataclasses

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.datasets.batch import SampleMetaData
from weathergen.model.attention import (
    MultiCrossAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlenSlicedQ,
    MultiSelfAttentionHead,
    MultiSelfAttentionHeadLocal,
    MultiSelfAttentionHeadVarlen,
)
from weathergen.model.blocks import CrossAttentionBlock, OriginalPredictionBlock, SelfAttentionBlock
from weathergen.model.embeddings import (
    StreamEmbedLinear,
    StreamEmbedTransformer,
)
from weathergen.model.layers import MLP
from weathergen.model.utils import ActivationFactory
from weathergen.utils.utils import get_dtype


class EmbeddingEngine(torch.nn.Module):
    name: "EmbeddingEngine"

    def __init__(self, cf: Config, sources_size, stream_names: list[str]) -> None:
        """
        Initialize the EmbeddingEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param sources_size: List of source sizes for each stream.
        :param stream_names: Ordered list of stream identifiers aligned with cf.streams.
        """
        super(EmbeddingEngine, self).__init__()
        self.cf = cf
        self.sources_size = sources_size  # KCT:iss130, what is this?
        self.embeds = torch.nn.ModuleDict()
        self.stream_names = list(stream_names)

        assert len(self.stream_names) == len(self.cf.streams), (
            "stream_names must align with cf.streams"
        )

        for i, (si, stream_name) in enumerate(zip(self.cf.streams, self.stream_names, strict=True)):
            if si.get("diagnostic", False) or self.sources_size[i] == 0:
                self.embeds[stream_name] = torch.nn.Identity()
                continue

            if si["embed"]["net"] == "transformer":
                self.embeds[stream_name] = StreamEmbedTransformer(
                    mode=self.cf.embed_orientation,
                    num_tokens=si["embed"]["num_tokens"],
                    token_size=si["token_size"],
                    num_channels=self.sources_size[i],
                    dim_embed=si["embed"]["dim_embed"],
                    dim_out=self.cf.ae_local_dim_embed,
                    num_blocks=si["embed"]["num_blocks"],
                    num_heads=si["embed"]["num_heads"],
                    dropout_rate=self.cf.embed_dropout_rate,
                    norm_type=self.cf.norm_type,
                    unembed_mode=self.cf.embed_unembed_mode,
                    stream_name=stream_name,
                )
            elif si["embed"]["net"] == "linear":
                self.embeds[stream_name] = StreamEmbedLinear(
                    self.sources_size[i] * si["token_size"],
                    self.cf.ae_local_dim_embed,
                    stream_name=stream_name,
                )
            else:
                raise ValueError("Unsupported embedding network type")

    # TODO: remove device from arg list
    def forward(self, input_step, sample, pe_embed, dtype, device):
        offsets_base = torch.cumsum(sample.source_cell_lens[input_step][1:], 0)

        tokens_all = torch.empty(
            (int(offsets_base[-1]), self.cf.ae_local_dim_embed), dtype=dtype, device=device
        )

        # iterate over all streams
        for stream_name, s_data in sample.streams_data.items():
            # embedding network
            embed = self.embeds[stream_name]

            # skip empty stream
            if s_data.source_empty():
                continue

            idxs = s_data.source_idxs_embed[input_step].to(device)
            idxs_pe = s_data.source_idxs_embed_pe[input_step].to(device)

            # create full scatter index
            # (there's no broadcasting which is likely highly inefficient)
            idxs = idxs.unsqueeze(1).repeat((1, self.cf.ae_local_dim_embed))
            x_embed = embed(s_data.source_tokens_cells[input_step]).flatten(0, 1)
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
            tokens_all.scatter_(0, idxs, x_embed + pe_embed[idxs_pe])

        return tokens_all


class LocalAssimilationEngine(torch.nn.Module):
    name: "LocalAssimilationEngine"

    def __init__(self, cf: Config) -> None:
        """
        Initialize the LocalAssimilationEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        """
        super(LocalAssimilationEngine, self).__init__()
        self.cf = cf
        self.ae_local_blocks = torch.nn.ModuleList()

        for _ in range(self.cf.ae_local_num_blocks):
            self.ae_local_blocks.append(
                MultiSelfAttentionHeadVarlen(
                    self.cf.ae_local_dim_embed,
                    num_heads=self.cf.ae_local_num_heads,
                    dropout_rate=self.cf.ae_local_dropout_rate,
                    with_qk_lnorm=self.cf.ae_local_with_qk_lnorm,
                    with_flash=self.cf.with_flash_attention,
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.norm_eps,
                    attention_dtype=get_dtype(self.cf.attention_dtype),
                )
            )
            self.ae_local_blocks.append(
                MLP(
                    self.cf.ae_local_dim_embed,
                    self.cf.ae_local_dim_embed,
                    with_residual=True,
                    dropout_rate=self.cf.ae_local_dropout_rate,
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )

    def forward(self, tokens_c, cell_lens_c, use_reentrant):
        for block in self.ae_local_blocks:
            tokens_c = checkpoint(block, tokens_c, cell_lens_c, use_reentrant=use_reentrant)
        return tokens_c


class Local2GlobalAssimilationEngine(torch.nn.Module):
    name: "Local2GlobalAssimilationEngine"

    def __init__(self, cf: Config) -> None:
        """
        Initialize the Local2GlobalAssimilationEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        """
        super(Local2GlobalAssimilationEngine, self).__init__()
        self.cf = cf
        self.ae_adapter = torch.nn.ModuleList()

        self.ae_adapter.append(
            MultiCrossAttentionHeadVarlenSlicedQ(
                self.cf.ae_global_dim_embed,
                self.cf.ae_local_dim_embed,
                num_slices_q=self.cf.ae_local_num_queries,
                dim_head_proj=self.cf.ae_adapter_embed,
                num_heads=self.cf.ae_adapter_num_heads,
                with_residual=self.cf.ae_adapter_with_residual,
                with_qk_lnorm=self.cf.ae_adapter_with_qk_lnorm,
                dropout_rate=self.cf.ae_adapter_dropout_rate,
                with_flash=self.cf.with_flash_attention,
                norm_type=self.cf.norm_type,
                norm_eps=self.cf.norm_eps,
                attention_dtype=get_dtype(self.cf.attention_dtype),
            )
        )

        ae_adapter_num_blocks = cf.get("ae_adapter_num_blocks", 2)
        for _ in range(ae_adapter_num_blocks - 1):
            self.ae_adapter.append(
                MLP(
                    self.cf.ae_global_dim_embed,
                    self.cf.ae_global_dim_embed,
                    with_residual=True,
                    dropout_rate=self.cf.ae_adapter_dropout_rate,
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )
            self.ae_adapter.append(
                MultiCrossAttentionHeadVarlenSlicedQ(
                    self.cf.ae_global_dim_embed,
                    self.cf.ae_local_dim_embed,
                    num_slices_q=self.cf.ae_local_num_queries,
                    dim_head_proj=self.cf.ae_adapter_embed,
                    num_heads=self.cf.ae_adapter_num_heads,
                    with_residual=self.cf.ae_adapter_with_residual,
                    with_qk_lnorm=self.cf.ae_adapter_with_qk_lnorm,
                    dropout_rate=self.cf.ae_adapter_dropout_rate,
                    with_flash=self.cf.with_flash_attention,
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.norm_eps,
                    attention_dtype=get_dtype(self.cf.attention_dtype),
                )
            )

    def forward(self, tokens_c, tokens_global_c, q_cells_lens_c, cell_lens_c, use_reentrant):
        for block in self.ae_adapter:
            tokens_global_c = checkpoint(
                block,
                tokens_global_c,
                tokens_c,
                q_cells_lens_c,
                cell_lens_c,
                use_reentrant=use_reentrant,
            )
        return tokens_global_c


class QueryAggregationEngine(torch.nn.Module):
    name: "QueryAggregationEngine"

    def __init__(self, cf: Config, num_healpix_cells: int) -> None:
        """
        Initialize the QueryAggregationEngine with the configuration.

        This engine is used for aggregating information from all query tokens coming
        from healpix cells, that are not masked.

        :param cf: Configuration object containing parameters for the engine.
        :param num_healpix_cells: Number of healpix cells used for local queries.
        """
        super(QueryAggregationEngine, self).__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells

        self.ae_aggregation_blocks = torch.nn.ModuleList()

        global_rate = int(1 / self.cf.ae_aggregation_att_dense_rate)
        for i in range(self.cf.ae_aggregation_num_blocks):
            ## Alternate between local and global attention
            #  as controlled by cf.ae_dense_local_att_dense_rate
            # Last block is always global attention
            if i % global_rate == 0 or i + 1 == self.cf.ae_aggregation_num_blocks:
                self.ae_aggregation_blocks.append(
                    MultiSelfAttentionHead(
                        self.cf.ae_global_dim_embed,
                        num_heads=self.cf.ae_aggregation_num_heads,
                        dropout_rate=self.cf.ae_aggregation_dropout_rate,
                        with_qk_lnorm=self.cf.ae_aggregation_with_qk_lnorm,
                        with_flash=self.cf.with_flash_attention,
                        norm_type=self.cf.norm_type,
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                    )
                )
            else:
                self.ae_aggregation_blocks.append(
                    MultiSelfAttentionHeadLocal(
                        self.cf.ae_global_dim_embed,
                        num_heads=self.cf.ae_aggregation_num_heads,
                        qkv_len=self.num_healpix_cells * self.cf.ae_local_num_queries,
                        block_factor=self.cf.ae_aggregation_block_factor,
                        dropout_rate=self.cf.ae_aggregation_dropout_rate,
                        with_qk_lnorm=self.cf.ae_aggregation_with_qk_lnorm,
                        with_flash=self.cf.with_flash_attention,
                        norm_type=self.cf.norm_type,
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                    )
                )
            # MLP block
            self.ae_aggregation_blocks.append(
                MLP(
                    self.cf.ae_global_dim_embed,
                    self.cf.ae_global_dim_embed,
                    with_residual=True,
                    dropout_rate=self.cf.ae_aggregation_dropout_rate,
                    hidden_factor=self.cf.ae_aggregation_mlp_hidden_factor,
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )

    def forward(self, tokens, use_reentrant):
        for block in self.ae_aggregation_blocks:
            tokens = checkpoint(block, tokens, use_reentrant=use_reentrant)
        return tokens


class GlobalAssimilationEngine(torch.nn.Module):
    name: "GlobalAssimilationEngine"

    def __init__(self, cf: Config, num_healpix_cells: int) -> None:
        """
        Initialize the GlobalAssimilationEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param num_healpix_cells: Number of healpix cells used for local queries.
        """
        super(GlobalAssimilationEngine, self).__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells

        self.ae_global_blocks = torch.nn.ModuleList()

        global_rate = int(1 / self.cf.ae_global_att_dense_rate)
        for i in range(self.cf.ae_global_num_blocks):
            ## Alternate between local and global attention
            #  as controlled by cf.ae_global_att_dense_rate
            # Last block is always global attention
            if i % global_rate == 0 or i + 1 == self.cf.ae_global_num_blocks:
                self.ae_global_blocks.append(
                    MultiSelfAttentionHead(
                        self.cf.ae_global_dim_embed,
                        num_heads=self.cf.ae_global_num_heads,
                        dropout_rate=self.cf.ae_global_dropout_rate,
                        with_qk_lnorm=self.cf.ae_global_with_qk_lnorm,
                        with_flash=self.cf.with_flash_attention,
                        norm_type=self.cf.norm_type,
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                    )
                )
            else:
                self.ae_global_blocks.append(
                    MultiSelfAttentionHeadLocal(
                        self.cf.ae_global_dim_embed,
                        num_heads=self.cf.ae_global_num_heads,
                        qkv_len=self.num_healpix_cells * self.cf.ae_local_num_queries,
                        block_factor=self.cf.ae_global_block_factor,
                        dropout_rate=self.cf.ae_global_dropout_rate,
                        with_qk_lnorm=self.cf.ae_global_with_qk_lnorm,
                        with_flash=self.cf.with_flash_attention,
                        norm_type=self.cf.norm_type,
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                    )
                )
            # MLP block
            self.ae_global_blocks.append(
                MLP(
                    self.cf.ae_global_dim_embed,
                    self.cf.ae_global_dim_embed,
                    with_residual=True,
                    dropout_rate=self.cf.ae_global_dropout_rate,
                    hidden_factor=self.cf.ae_global_mlp_hidden_factor,
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )

        self.ae_global_blocks.append(
            torch.nn.LayerNorm(self.cf.ae_global_dim_embed, elementwise_affine=False)
        )

    def forward(self, tokens, use_reentrant):
        for block in self.ae_global_blocks:
            tokens = checkpoint(block, tokens, use_reentrant=use_reentrant)
        return tokens


class ForecastingEngine(torch.nn.Module):
    name: "ForecastingEngine"

    def __init__(self, cf: Config, num_healpix_cells: int) -> None:
        """
        Initialize the ForecastingEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param num_healpix_cells: Number of healpix cells used for local queries.
        """
        super(ForecastingEngine, self).__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells
        self.fe_blocks = torch.nn.ModuleList()

        global_rate = int(1 / self.cf.forecast_att_dense_rate)
        if self.cf.forecast_policy is not None:
            for i in range(self.cf.fe_num_blocks):
                # Alternate between global and local attention
                if (i % global_rate == 0) or i + 1 == self.cf.ae_global_num_blocks:
                    self.fe_blocks.append(
                        MultiSelfAttentionHead(
                            self.cf.ae_global_dim_embed,
                            num_heads=self.cf.fe_num_heads,
                            dropout_rate=self.cf.fe_dropout_rate,
                            with_qk_lnorm=self.cf.fe_with_qk_lnorm,
                            with_flash=self.cf.with_flash_attention,
                            norm_type=self.cf.norm_type,
                            dim_aux=(1 if cf.forecast_with_step_conditioning else None),
                            norm_eps=self.cf.norm_eps,
                            attention_dtype=get_dtype(self.cf.attention_dtype),
                            with_noise_conditioning=self.cf.fe_diffusion_model,
                        )
                    )
                else:
                    self.fe_blocks.append(
                        MultiSelfAttentionHeadLocal(
                            self.cf.ae_global_dim_embed,
                            num_heads=self.cf.fe_num_heads,
                            qkv_len=self.num_healpix_cells * self.cf.ae_local_num_queries,
                            block_factor=self.cf.ae_global_block_factor,
                            dropout_rate=self.cf.fe_dropout_rate,
                            with_qk_lnorm=self.cf.fe_with_qk_lnorm,
                            with_flash=self.cf.with_flash_attention,
                            norm_type=self.cf.norm_type,
                            dim_aux=(1 if cf.forecast_with_step_conditioning else None),
                            norm_eps=self.cf.norm_eps,
                            attention_dtype=get_dtype(self.cf.attention_dtype),
                            with_noise_conditioning=self.cf.fe_diffusion_model,
                        )
                    )
                # Add MLP block
                self.fe_blocks.append(
                    MLP(
                        self.cf.ae_global_dim_embed,
                        self.cf.ae_global_dim_embed,
                        with_residual=True,
                        dropout_rate=self.cf.fe_dropout_rate,
                        norm_type=self.cf.norm_type,
                        dim_aux=(1 if cf.forecast_with_step_conditioning else None),
                        norm_eps=self.cf.mlp_norm_eps,
                        with_noise_conditioning=self.cf.fe_diffusion_model,
                    )
                )

        def init_weights_final(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, mean=0, std=0.001)

        for block in self.fe_blocks:
            block.apply(init_weights_final)

    def forward(
        self,
        tokens: torch.Tensor,
        fstep: int,
        meta_info: SampleMetaData = None,
        noise_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        # predict residual to last time step if requested
        forecast_residual = self.cf.get("forecast_residual", False)
        if forecast_residual:
            tokens_in = tokens

        # aux_info is forecast step, if not disabled with cf.forecast_with_step_conditioning
        # aux_info = torch.tensor([fstep], dtype=torch.float32, device="cuda")
        aux_info = None
        if self.cf.fe_diffusion_model:
            assert noise_emb is not None, (
                "Noise embedding must be provided for diffusion forecast engine"
            )
            for block in self.fe_blocks:
                tokens = checkpoint(block, tokens, noise_emb, aux_info, use_reentrant=False)
        else:
            for block in self.fe_blocks:
                tokens = checkpoint(block, tokens, aux_info, use_reentrant=False)

        return tokens if not forecast_residual else (tokens_in + tokens)


class EnsPredictionHead(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        dim_out,
        ens_num_layers,
        ens_size,
        stream_name: str,
        norm_type="LayerNorm",
        hidden_factor=2,
        final_activation: None | str = None,
    ):
        """Constructor"""

        super(EnsPredictionHead, self).__init__()

        self.name = f"EnsPredictionHead_{stream_name}"

        dim_internal = dim_embed * hidden_factor
        # norm = torch.nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm
        enl = ens_num_layers

        self.pred_heads = torch.nn.ModuleList()
        for i in range(ens_size):
            self.pred_heads.append(torch.nn.ModuleList())

            # self.pred_heads[-1].append( norm( dim_embed))
            self.pred_heads[-1].append(
                torch.nn.Linear(dim_embed, dim_out if enl == 1 else dim_internal)
            )

            for i in range(ens_num_layers - 1):
                self.pred_heads[-1].append(torch.nn.GELU())
                self.pred_heads[-1].append(
                    torch.nn.Linear(dim_internal, dim_out if enl - 2 == i else dim_internal)
                )

            # Add optional final non-linear activation
            if final_activation is not None and enl >= 1:
                fal = ActivationFactory.get(final_activation)
                self.pred_heads[-1].append(fal)

    #########################################
    def forward(self, toks):
        preds = []
        for pred_head in self.pred_heads:
            cpred = toks
            for block in pred_head:
                cpred = block(cpred)
            preds.append(cpred)
        preds = torch.stack(preds, 0)

        return preds


class TargetPredictionEngineClassic(nn.Module):
    def __init__(
        self,
        cf,
        dims_embed,
        dim_coord_in,
        tr_dim_head_proj,
        tr_mlp_hidden_factor,
        softcap,
        tro_type,
        stream_name: str,
    ):
        """
        Initialize the TargetPredictionEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param dims_embed: List of embedding dimensions for each layer.
        :param dim_coord_in: Input dimension for coordinates.
        :param tr_dim_head_proj: Dimension for head projection.
        :param tr_mlp_hidden_factor: Hidden factor for the MLP layers.
        :param softcap: Softcap value for the attention layers.
        :param tro_type: Type of target readout (e.g., "obs_value").
        """
        super(TargetPredictionEngineClassic, self).__init__()
        self.name = f"TargetPredictionEngine_{stream_name}"

        self.cf = cf
        self.dims_embed = dims_embed
        self.dim_coord_in = dim_coord_in
        self.tr_dim_head_proj = tr_dim_head_proj
        self.tr_mlp_hidden_factor = tr_mlp_hidden_factor
        self.softcap = softcap
        self.tro_type = tro_type
        self.tte = torch.nn.ModuleList()

        for i in range(len(self.dims_embed) - 1):
            # Multi-Cross Attention Head
            self.tte.append(
                MultiCrossAttentionHeadVarlen(
                    dim_embed_q=self.dims_embed[i],
                    dim_embed_kv=self.cf.ae_global_dim_embed,
                    num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
                    dim_head_proj=self.tr_dim_head_proj,
                    with_residual=True,
                    with_qk_lnorm=True,
                    dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                    with_flash=self.cf.with_flash_attention,
                    norm_type=self.cf.norm_type,
                    softcap=self.softcap,
                    dim_aux=self.dim_coord_in,
                    norm_eps=self.cf.norm_eps,
                    attention_dtype=get_dtype(self.cf.attention_dtype),
                )
            )

            # Optional Self-Attention Head
            if self.cf.pred_self_attention:
                self.tte.append(
                    MultiSelfAttentionHeadVarlen(
                        dim_embed=self.dims_embed[i],
                        num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
                        dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                        with_qk_lnorm=True,
                        with_flash=self.cf.with_flash_attention,
                        norm_type=self.cf.norm_type,
                        dim_aux=self.dim_coord_in,
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                    )
                )

            # MLP Block
            self.tte.append(
                MLP(
                    self.dims_embed[i],
                    self.dims_embed[i + 1],
                    with_residual=(self.cf.pred_dyadic_dims or self.tro_type == "obs_value"),
                    hidden_factor=self.tr_mlp_hidden_factor,
                    dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                    norm_type=self.cf.norm_type,
                    dim_aux=(self.dim_coord_in if self.cf.pred_mlp_adaln else None),
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )

    def forward(self, latent, output, latent_lens, output_lens, coordinates):
        tc_tokens = output
        tcs_lens = output_lens
        tokens_stream = latent
        tokens_lens = latent_lens
        tcs_aux = coordinates

        for ib, block in enumerate(self.tte):
            if self.cf.pred_self_attention and ib % 3 == 1:
                tc_tokens = checkpoint(block, tc_tokens, tcs_lens, tcs_aux, use_reentrant=False)
            else:
                tc_tokens = checkpoint(
                    block,
                    tc_tokens,
                    tokens_stream,
                    tcs_lens,
                    tokens_lens,
                    tcs_aux,
                    use_reentrant=False,
                )
        return tc_tokens


class TargetPredictionEngine(nn.Module):
    def __init__(
        self,
        cf,
        dims_embed,
        dim_coord_in,
        tr_dim_head_proj,
        tr_mlp_hidden_factor,
        softcap,
        tro_type,
        stream_name: str,
    ):
        """
        Initialize the TargetPredictionEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param dims_embed: List of embedding dimensions for each layer.
        :param dim_coord_in: Input dimension for coordinates.
        :param tr_dim_head_proj: Dimension for head projection.
        :param tr_mlp_hidden_factor: Hidden factor for the MLP layers.
        :param softcap: Softcap value for the attention layers.
        :param tro_type: Type of target readout (e.g., "obs_value").

        the decoder_type decides the how the conditioning is done

        PerceiverIO: is a simple CrossAttention layer with no MLP or Adaptive LayerNorm
        AdaLayerNormConditioning: only conditions via the Adaptive LayerNorm
        CrossAttentionConditioning: conditions via the CrossAttention layer but also uses an MLP
        CrossAttentionAdaNormConditioning: conditions via the CrossAttention layer and
            Adaptive LayerNorm
        PerceiverIOCoordConditioning: The conditioning is the coordinates and is a modified Adaptive
            LayerNorm that does not scale after the layer is applied
        """
        super(TargetPredictionEngine, self).__init__()
        self.name = f"TargetPredictionEngine_{stream_name}"

        self.cf = cf
        self.dims_embed = dims_embed
        self.dim_coord_in = dim_coord_in
        self.tr_dim_head_proj = tr_dim_head_proj
        self.tr_mlp_hidden_factor = tr_mlp_hidden_factor
        self.softcap = softcap
        self.tro_type = tro_type

        # For backwards compatibility
        from omegaconf import OmegaConf

        self.cf = OmegaConf.merge(
            OmegaConf.create({"decoder_type": "PerceiverIOCoordConditioning"}), self.cf
        )

        attention_kwargs = {
            "with_qk_lnorm": True,
            "dropout_rate": 0.1,  # Assuming dropout_rate is 0.1
            "with_flash": self.cf.with_flash_attention,
            "norm_type": self.cf.norm_type,
            "softcap": self.softcap,
            "dim_aux": self.dim_coord_in,
            "norm_eps": self.cf.norm_eps,
            "attention_dtype": get_dtype(self.cf.attention_dtype),
        }
        self.tte = nn.ModuleList()
        self.output_in_norm = nn.LayerNorm(self.dims_embed[0])
        self.latent_in_norm = nn.LayerNorm(self.cf.ae_global_dim_embed)
        self.final_norm = nn.Identity()  # nn.RMSNorm(self.dims_embed[-1])
        self.dropout = nn.Dropout(0.2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 9, self.cf.ae_global_dim_embed))
        dim_aux = self.cf.ae_global_dim_embed

        for ith, dim in enumerate(self.dims_embed[:-1]):
            if self.cf.decoder_type == "PerceiverIO":
                # a single cross attention layer as per https://arxiv.org/pdf/2107.14795
                self.tte.append(
                    CrossAttentionBlock(
                        dim_q=dim,
                        dim_kv=dim_aux,
                        dim_aux=dim_aux,
                        num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
                        with_self_attn=False,
                        with_adanorm=False,
                        with_mlp=False,
                        attention_kwargs=attention_kwargs,
                    )
                )
            elif self.cf.decoder_type == "AdaLayerNormConditioning":
                self.tte.append(
                    SelfAttentionBlock(
                        dim=dim,
                        dim_aux=dim_aux,
                        num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
                        attention_kwargs=attention_kwargs,
                        with_adanorm=True,
                        dropout_rate=0.1,
                    )
                )
            elif self.cf.decoder_type == "CrossAttentionConditioning":
                self.tte.append(
                    CrossAttentionBlock(
                        dim_q=dim,
                        dim_kv=self.cf.ae_global_dim_embed,
                        dim_aux=dim_aux,
                        num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
                        with_self_attn=True,
                        with_adanorm=False,
                        with_mlp=True,
                        dropout_rate=0.1,
                        attention_kwargs=attention_kwargs,
                    )
                )
            elif self.cf.decoder_type == "CrossAttentionAdaNormConditioning":
                self.tte.append(
                    CrossAttentionBlock(
                        dim_q=dim,
                        dim_kv=dim_aux,
                        dim_aux=dim_aux,
                        num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
                        with_self_attn=True,
                        with_adanorm=True,
                        with_mlp=True,
                        dropout_rate=0.1,
                        attention_kwargs=attention_kwargs,
                    )
                )
            elif self.cf.decoder_type == "PerceiverIOCoordConditioning":
                self.tte.append(
                    OriginalPredictionBlock(
                        config=self.cf,
                        dim_in=dim,
                        dim_out=self.dims_embed[ith + 1],
                        dim_kv=dim_aux,
                        dim_aux=self.dim_coord_in,
                        num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
                        attention_kwargs=attention_kwargs,
                        tr_dim_head_proj=tr_dim_head_proj,
                        tr_mlp_hidden_factor=tr_mlp_hidden_factor,
                        tro_type=tro_type,
                        mlp_norm_eps=self.cf.mlp_norm_eps,
                    )
                )
            else:
                raise NotImplementedError(
                    f"{self.cf.decoder_type} is not implemented for prediction heads"
                )

    def forward(self, latent, output, latent_lens, output_lens, coordinates):
        latent = (
            self.dropout(self.latent_in_norm(latent + self.pos_embed))
            if self.cf.decoder_type != "PerceiverIOCoordConditioning"
            else latent
        )
        for layer in self.tte:
            if isinstance(layer, OriginalPredictionBlock):
                output = checkpoint(
                    layer,
                    latent=latent.flatten(0, 1),
                    output=output,
                    coords=coordinates,
                    latent_lens=latent_lens,
                    output_lens=output_lens,
                    use_reentrant=False,
                )
            elif isinstance(layer, CrossAttentionBlock):
                output = checkpoint(
                    layer,
                    x=output,
                    x_kv=latent.flatten(0, 1),
                    x_lens=output_lens,
                    aux=latent[:, 0],
                    x_kv_lens=latent_lens,
                    use_reentrant=False,
                )
            else:
                output = checkpoint(
                    layer,
                    x=output,
                    x_lens=output_lens,
                    aux=latent[:, 0],
                    use_reentrant=False,
                )
        output = (
            self.final_norm(output)
            if self.cf.decoder_type != "PerceiverIOCoordConditioning"
            else output
        )
        return output


@dataclasses.dataclass
class LatentState:
    """
    A dataclass to encapsulate the latent state
    """

    register_tokens: torch.Tensor
    latent_tokens: torch.Tensor
