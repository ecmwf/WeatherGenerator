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
from typing import Callable

import torch
import torch.nn as nn
from omegaconf import OmegaConf
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
from weathergen.model.positional_encoding import get_rope_mode
from weathergen.model.utils import ActivationFactory
from weathergen.utils.utils import get_dtype

class EmbeddingEngine(torch.nn.Module):
    name: "EmbeddingEngine"

    def __init__(self, cf: Config, sources_size) -> None:
        """
        Initialize the EmbeddingEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param sources_size: List of source sizes for each stream.
        """
        super(EmbeddingEngine, self).__init__()
        self.cf = cf
        self.dtype = get_dtype(self.cf.mixed_precision_dtype)
        self.sources_size = sources_size  # KCT:iss130, what is this?
        self.embeds = torch.nn.ModuleDict()
        self.stream_names = [str(stream_cfg["name"]) for stream_cfg in cf.streams]

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
                    mlp_type=self.cf.get("mlp_type", "mlp"),
                    use_xsa=self.cf.get("use_xsa", False),
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

    def forward(self, batch, pe_embed):
        num_steps_input = batch.get_num_steps()

        num_tokens = torch.sum(batch.tokens_lens, 2).flatten().sum().item()
        tokens_all = torch.empty(
            (num_tokens, self.cf.ae_local_dim_embed), dtype=self.dtype, device=batch.get_device()
        )

        # iterate over all streams
        x_embeds = []
        for stream_name in self.stream_names:
            # collect all source tokens from all input_steps and all samples in the batch
            sdata = []
            for istep in range(num_steps_input):
                for sample in batch.get_samples():
                    sdata += [sample.streams_data[stream_name].source_tokens_cells[istep]]

            if all(s is None for s in sdata):
                continue

            sdata = torch.cat(sdata).to(tokens_all.dtype)
            # skip empty stream
            if sdata.numel() == 0:
                continue
            
            # embedding from physical space to per patch latent representation
            x_embeds += [self.embeds[stream_name](sdata).flatten(0, 1)]

        # switch from stream to cell-based ordering and apply per cell positional encoding

        # if the assert is hit, max_number_tokens_local_per_cell in config needs to be increased
        max_tokens = self.cf.get("ae_local_max_tokens_per_cell", 64)
        assert batch.tokens_lens.flatten(0, 2).sum(0).max() <= max_tokens, (
            "max number of tokens per cell for positional encoding exceeded."
        )
        " Increase ae_local_max_tokens_per_cell in config."

        if batch.tokens_lens.shape[2] == 1:
            # trivial with one stream
            tokens_all = torch.cat(x_embeds)

        else:
            scatter_idxs = self.get_scatter_idxs_vectorized(batch)
            scatter_idxs = scatter_idxs.unsqueeze(1).repeat((1, self.cf.ae_local_dim_embed))

            # actual scatter operation and apply per cell positional encoding
            tokens_all.scatter_(0, scatter_idxs, torch.cat(x_embeds))

        pe_idxs = self.get_pe_idxs_vectorized(batch)
        tokens_all = tokens_all + pe_embed[pe_idxs]

        return tokens_all

    def get_pe_idxs_vectorized(self, batch):
        """
        Compute per cell indices into positional encoding
        """

        tok_counts = batch.tokens_lens.permute([2, 0, 1, 3]).sum(0).flatten()
        rows = torch.arange(tok_counts.max(), device=tok_counts.device).unsqueeze(0)
        rows = rows.expand(tok_counts.shape[0], -1)
        pe_idxs = rows[rows < tok_counts.unsqueeze(1)]

        return pe_idxs

    def get_scatter_idxs(self, batch):
        """
        Compute reordering index so that tokens from different streams but same cell are
        continguous

        Simple version (reference implementation)
        """

        dev = batch.get_device()
        # batch.tokens_lens : (num_steps_input, num_samples, num_streams, num_cells)
        # flatten leasds to streams x tokens per cell (across all cells for input steps and samples)
        tok_counts = batch.tokens_lens.permute([2, 0, 1, 3]).flatten(1, -1)

        scatter_idxs = []
        for i in range(len(tok_counts)):
            for j in range(tok_counts.shape[1]):
                if tok_counts[i, j] == 0:
                    continue
                # offset from preceding cells
                offset = tok_counts[:, :j].flatten().sum()
                # offset from preceding streams in cells
                offset += tok_counts[:i, j].sum()
                # scatter idxs is offset and idxs for all tokens in cell for current stream
                scatter_idxs += [offset[i, j] + torch.arange(tok_counts[i, j], device=dev)]

        scatter_idxs = torch.cat(scatter_idxs).to(torch.int64)

        return scatter_idxs

    def get_scatter_idxs_vectorized(self, batch):
        """
        Compute reordering index so that tokens from different streams but same cell are
        continguous

        Vectorized version
        """

        dev = batch.get_device()
        # batch.tokens_lens : (num_steps_input, num_samples, num_streams, num_cells)
        # flatten leasds to streams x tokens per cell (across all cells for input steps and samples)
        tok_counts = batch.tokens_lens.permute([2, 0, 1, 3]).flatten(1, -1)

        # partial sums for per cell offsets
        pad = torch.zeros((1, tok_counts.shape[1]), dtype=torch.int64, device=dev)
        offset = torch.cat([pad, tok_counts.cumsum(0)])[:-1]
        offset[:, 1:] += tok_counts.sum(0).cumsum(0)[:-1]

        ranges = torch.arange(tok_counts.max(), device=dev).repeat((tok_counts.numel(), 1))
        idxs = (offset.flatten() + ranges.transpose(1, 0)).transpose(1, 0)
        # select idxs[i][:ranges[i]] for each i; vectorized version
        col_indices = torch.arange(idxs.shape[1], device=dev).unsqueeze(0)
        valid_mask = col_indices < tok_counts.flatten().unsqueeze(1)
        scatter_idxs = idxs[valid_mask].to(torch.int64)

        return scatter_idxs


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
                    use_xsa=self.cf.get("use_xsa", False),
                    norm_type=self.cf.norm_type,
                    qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
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
                    mlp_type=self.cf.get("mlp_type", "mlp"),
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )

    def forward(self, tokens_c, cell_lens_c, use_reentrant):
        for block in self.ae_local_blocks:
            tokens_c = block(tokens_c, cell_lens_c)
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
                qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
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
                    mlp_type=self.cf.get("mlp_type", "mlp"),
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
                    qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                    norm_eps=self.cf.norm_eps,
                    attention_dtype=get_dtype(self.cf.attention_dtype),
                )
            )

    def forward(self, tokens_c, tokens_global_c, q_cells_lens_c, cell_lens_c):
        for block in self.ae_adapter:
            tokens_global_c = block(
                tokens_global_c,
                tokens_c,
                q_cells_lens_c,
                cell_lens_c,
            )
        return tokens_global_c


class Local2GlobalSumEngine(torch.nn.Module):
    """Alternative to Local2GlobalAssimilationEngine.

    Instead of cross-attention (Q=learnable query, KV=local tokens), this engine
    sums local tokens per cell and projects to global dim. Masked cells are filled
    externally by the encoder using the learnable query + pe_global (unchanged).

    Forward signature matches Local2GlobalAssimilationEngine; tokens_global_c and
    q_cells_lens_c are unused (masked-cell filling happens in the encoder).
    """

    name: "Local2GlobalSumEngine"

    def __init__(self, cf: Config) -> None:
        super(Local2GlobalSumEngine, self).__init__()
        self.cf = cf
        self.proj = torch.nn.Linear(cf.ae_local_dim_embed, cf.ae_global_dim_embed, bias=False)
        ae_adapter_num_blocks = cf.get("ae_adapter_num_blocks", 2)
        self.mlp_blocks = torch.nn.ModuleList()
        for _ in range(ae_adapter_num_blocks - 1):
            self.mlp_blocks.append(
                MLP(
                    cf.ae_global_dim_embed,
                    cf.ae_global_dim_embed,
                    with_residual=True,
                    dropout_rate=cf.ae_adapter_dropout_rate,
                    norm_type=cf.norm_type,
                    norm_eps=cf.mlp_norm_eps,
                )
            )

    def forward(self, tokens_c, tokens_global_c, q_cells_lens_c, cell_lens_c):
        # tokens_c:        (total_local_tokens, local_dim)
        # tokens_global_c: (num_unmasked_cells, num_queries, global_dim) — unused
        # cell_lens_c:     (num_unmasked_cells + 1,) with 0 at index 0
        num_cells = cell_lens_c.shape[0] - 1
        cell_counts = cell_lens_c[1:]

        # scatter-sum local tokens into per-cell summaries
        cell_idx = torch.repeat_interleave(
            torch.arange(num_cells, device=tokens_c.device, dtype=torch.long), cell_counts
        )
        cell_sums = torch.zeros(
            num_cells, tokens_c.shape[-1], device=tokens_c.device, dtype=tokens_c.dtype
        )
        cell_sums.scatter_add_(0, cell_idx.unsqueeze(1).expand_as(tokens_c), tokens_c)

        # project to global dim and match (num_cells, num_queries, global_dim)
        num_queries = tokens_global_c.shape[1]
        out = self.proj(cell_sums).unsqueeze(1).expand(-1, num_queries, -1)

        for blk in self.mlp_blocks:
            out = blk(out)

        return out


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
        rope_mode = get_rope_mode(self.cf)

        self.ae_aggregation_blocks = torch.nn.ModuleList()

        global_rate = int(1 / self.cf.ae_aggregation_att_dense_rate)
        for i in range(self.cf.ae_aggregation_num_blocks):
            ## Alternate between local and global attention
            #  as controlled by cf.ae_dense_local_att_dense_rate
            # Last block is always global attention
            if i % global_rate == 0 or i + 1 == self.cf.ae_aggregation_num_blocks:
                self.ae_aggregation_blocks.append(
                    MultiSelfAttentionHeadVarlen(
                        self.cf.ae_global_dim_embed,
                        num_heads=self.cf.ae_aggregation_num_heads,
                        dropout_rate=self.cf.ae_aggregation_dropout_rate,
                        with_qk_lnorm=self.cf.ae_aggregation_with_qk_lnorm,
                        with_flash=self.cf.with_flash_attention,
                        use_xsa=self.cf.get("use_xsa", False),
                        norm_type=self.cf.norm_type,
                        qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                        rope_mode=rope_mode,
                    )
                )
            else:
                assert False, "Incompatible with batchsize > 1 here"
                self.ae_aggregation_blocks.append(
                    MultiSelfAttentionHeadLocal(
                        self.cf.ae_global_dim_embed,
                        num_heads=self.cf.ae_aggregation_num_heads,
                        qkv_len=self.num_healpix_cells * self.cf.ae_local_num_queries,
                        block_factor=self.cf.ae_aggregation_block_factor,
                        dropout_rate=self.cf.ae_aggregation_dropout_rate,
                        with_qk_lnorm=self.cf.ae_aggregation_with_qk_lnorm,
                        with_flash=self.cf.with_flash_attention,
                        use_xsa=self.cf.get("use_xsa", False),
                        norm_type=self.cf.norm_type,
                        qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
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
                    mlp_type=self.cf.get("mlp_type", "mlp"),
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )

    def forward(self, tokens, batch_lens, use_reentrant, coords=None):
        for block in self.ae_aggregation_blocks:
            aux_info = None
            if isinstance(block, MultiSelfAttentionHeadVarlen):
                tokens = block(tokens, x_lens=batch_lens, coords=coords)
            else:
                tokens = block(tokens, coords, aux_info)
        return tokens


class GlobalAssimilationEngine(torch.nn.Module):
    name: "GlobalAssimilationEngine"

    def __init__(
        self, cf: Config, num_healpix_cells: int, tap_global_layers: set[int] | None = None
    ) -> None:
        """
        Initialize the GlobalAssimilationEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param num_healpix_cells: Number of healpix cells used for local queries.
        :param tap_global_layers: Logical layer indices at which to collect intermediate
            representations for deep self-supervision. None means disabled.
        """
        super(GlobalAssimilationEngine, self).__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells
        self.tap_global_layers = tap_global_layers
        rope_mode = get_rope_mode(self.cf)

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
                        use_xsa=self.cf.get("use_xsa", False),
                        norm_type=self.cf.norm_type,
                        qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                        rope_mode=rope_mode,
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
                        use_xsa=self.cf.get("use_xsa", False),
                        norm_type=self.cf.norm_type,
                        qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                        rope_mode=rope_mode,
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
                    mlp_type=self.cf.get("mlp_type", "mlp"),
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )
        if self.cf.get("ae_global_trailing_layer_norm", False):
            self.ae_global_blocks.append(
                torch.nn.LayerNorm(self.cf.ae_global_dim_embed, elementwise_affine=False)
            )

    def forward(self, tokens, coords=None):
        aux_info = None
        intermediates: list[torch.Tensor] = []
        logical_layer = 0
        for block in self.ae_global_blocks:
            tokens = checkpoint(block, tokens, coords, aux_info, use_reentrant=False)
            if isinstance(block, MLP):
                if self.tap_global_layers and logical_layer in self.tap_global_layers:
                    intermediates.append(tokens)
                logical_layer += 1
        return tokens, intermediates


class IdentityEngine(torch.nn.Module):
    """Identity engine that passes tokens through unchanged."""

    def __init__(self):
        super().__init__()
        self.fe_blocks = torch.nn.ModuleList()

    def forward(self, tokens, *args, **kwargs):
        return tokens


class ForecastingEngine(torch.nn.Module):
    name: "ForecastingEngine"

    def __init__(self, cf: Config, mode_cfg, num_healpix_cells: int, dim_aux: int = None) -> None:
        """
        Initialize the ForecastingEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param num_healpix_cells: Number of healpix cells used for local queries.
        """
        super(ForecastingEngine, self).__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells
        rope_mode = get_rope_mode(self.cf)
        self.fe_blocks = torch.nn.ModuleList()

        global_rate = int(1 / self.cf.forecast_att_dense_rate)
        if mode_cfg.get("forecast", {}).get("policy") is not None:
            for i in range(self.cf.fe_num_blocks):
                # Alternate between global and local attention
                if (i % global_rate == 0) or i + 1 == self.cf.fe_num_blocks:
                    self.fe_blocks.append(
                        MultiSelfAttentionHead(
                            self.cf.ae_global_dim_embed,
                            num_heads=self.cf.fe_num_heads,
                            dropout_rate=self.cf.fe_dropout_rate,
                            with_qk_lnorm=self.cf.fe_with_qk_lnorm,
                            with_flash=self.cf.with_flash_attention,
                            use_xsa=self.cf.get("use_xsa", False),
                            norm_type=self.cf.norm_type,
                            qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                            dim_aux=dim_aux,
                            norm_eps=self.cf.norm_eps,
                            attention_dtype=get_dtype(self.cf.attention_dtype),
                            rope_mode=rope_mode,
                            is_dit=self.cf.fe_diffusion_model,
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
                            use_xsa=self.cf.get("use_xsa", False),
                            norm_type=self.cf.norm_type,
                            qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                            dim_aux=dim_aux,
                            norm_eps=self.cf.norm_eps,
                            attention_dtype=get_dtype(self.cf.attention_dtype),
                            rope_mode=rope_mode,
                            is_dit=self.cf.fe_diffusion_model,
                        )
                    )
                # Add MLP block
                self.fe_blocks.append(
                    MLP(
                        self.cf.ae_global_dim_embed,
                        self.cf.ae_global_dim_embed,
                        num_layers=2,
                        with_residual=True,
                        dropout_rate=self.cf.fe_dropout_rate,
                        mlp_type=self.cf.get("mlp_type", "mlp"),
                        norm_type=self.cf.norm_type,
                        dim_aux=dim_aux,
                        norm_eps=self.cf.mlp_norm_eps,
                        is_dit=self.cf.fe_diffusion_model,
                    )
                )
                # Optionally, add LayerNorm after i-th layer
                if i in self.cf.get("fe_layer_norm_after_blocks", []):
                    self.fe_blocks.append(
                        torch.nn.LayerNorm(self.cf.ae_global_dim_embed, elementwise_affine=False)
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
        ada_ln_aux: torch.Tensor = None,
        coords: torch.Tensor = None,
    ) -> torch.Tensor:
        # aux_info is forecast step, if not disabled with cf.forecast_with_step_conditioning
        # aux_info = torch.tensor([fstep], dtype=torch.float32, device="cuda")
        if self.training:
            # Impute noise to the latent state
            noise_std = self.cf.get("fe_impute_latent_noise_std", 0.0)
            if noise_std > 0.0:
                tokens = tokens + torch.randn_like(tokens) * torch.norm(tokens) * noise_std

        # predict residual to last time step if requested
        forecast_residual = self.cf.get("forecast_residual", False)
        if forecast_residual:
            tokens_in = tokens

        if self.cf.fe_diffusion_model:
            assert noise_emb is not None, (
                "noise_emb must be provided for diffusion model conditioning"
            )
            for block in self.fe_blocks:
                if isinstance(block, torch.nn.LayerNorm):
                    tokens = checkpoint(block, tokens, use_reentrant=False)
                else:
                    assert ada_ln_aux is None, (
                        "ada_ln_aux should not be provided when diffusion model conditioning is disabled"
                    )
                    tokens = checkpoint(block, tokens, coords, noise_emb, use_reentrant=False)
        else:
            for block in self.fe_blocks:
                if isinstance(block, torch.nn.LayerNorm):
                    tokens = checkpoint(block, tokens, use_reentrant=False)
                else:
                    tokens = checkpoint(block, tokens, coords, ada_ln_aux, use_reentrant=False)

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
        tr_mlp_type,
        softcap,
        stream_config: dict,
    ):
        """
        Initialize the TargetPredictionEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param dims_embed: List of embedding dimensions for each layer.
        :param dim_coord_in: Input dimension for coordinates.
        :param tr_dim_head_proj: Dimension for head projection.
        :param tr_mlp_hidden_factor: Hidden factor for the MLP layers.
        :param softcap: Softcap value for the attention layers.
        """
        super(TargetPredictionEngineClassic, self).__init__()
        self.name = f"TargetPredictionEngine_{stream_config['name']}"

        self.cf = cf
        self.dims_embed = dims_embed
        self.dim_coord_in = dim_coord_in
        self.tr_dim_head_proj = tr_dim_head_proj
        self.tr_mlp_hidden_factor = tr_mlp_hidden_factor
        self.tr_mlp_type = tr_mlp_type
        self.softcap = softcap
        self.tte = torch.nn.ModuleList()

        for i in range(len(self.dims_embed) - 1):
            # Multi-Cross Attention Head
            self.tte.append(
                MultiCrossAttentionHeadVarlen(
                    dim_embed_q=self.dims_embed[i],
                    dim_embed_kv=self.cf.ae_global_dim_embed,
                    num_heads=stream_config["target_readout"]["num_heads"],
                    dim_head_proj=self.tr_dim_head_proj,
                    with_residual=True,
                    with_qk_lnorm=True,
                    dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                    with_flash=self.cf.with_flash_attention,
                    norm_type=self.cf.norm_type,
                    qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
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
                        num_heads=stream_config["target_readout"]["num_heads"],
                        dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                        with_qk_lnorm=True,
                        with_flash=self.cf.with_flash_attention,
                        use_xsa=self.cf.get("use_xsa", False),
                        norm_type=self.cf.norm_type,
                        qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
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
                    with_residual=True,
                    hidden_factor=self.tr_mlp_hidden_factor,
                    dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                    mlp_type=self.tr_mlp_type,
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
        tr_mlp_type,
        softcap,
        stream_config: dict,
    ):
        """
        Initialize the TargetPredictionEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param dims_embed: List of embedding dimensions for each layer.
        :param dim_coord_in: Input dimension for coordinates.
        :param tr_dim_head_proj: Dimension for head projection.
        :param tr_mlp_hidden_factor: Hidden factor for the MLP layers.
        :param softcap: Softcap value for the attention layers.

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
        self.name = f"TargetPredictionEngine_{stream_config['name']}"

        self.cf = cf
        self.dims_embed = dims_embed
        self.dim_coord_in = dim_coord_in
        self.tr_dim_head_proj = tr_dim_head_proj
        self.tr_mlp_hidden_factor = tr_mlp_hidden_factor
        self.tr_mlp_type = tr_mlp_type
        self.softcap = softcap

        # For backwards compatibility

        self.cf = OmegaConf.merge(
            OmegaConf.create({"decoder_type": "PerceiverIOCoordConditioning"}), self.cf
        )

        attention_kwargs = {
            "with_qk_lnorm": True,
            "dropout_rate": 0.1,  # Assuming dropout_rate is 0.1
            "with_flash": self.cf.with_flash_attention,
            "norm_type": self.cf.norm_type,
            "qk_norm_type": self.cf.qk_norm_type,
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
                        mlp_type=self.tr_mlp_type,
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
                        mlp_type=self.tr_mlp_type,
                        use_xsa=self.cf.get("use_xsa", False),
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
                        mlp_type=self.tr_mlp_type,
                        use_xsa=self.cf.get("use_xsa", False),
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
                        mlp_type=self.tr_mlp_type,
                        use_xsa=self.cf.get("use_xsa", False),
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
                        tr_mlp_type=tr_mlp_type,
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


class DeepSSLFusion(nn.Module):
    """Concatenate multi-level representations along channel dim, fuse with MLP.

    Used by the student in deep self-supervision (V-JEPA 2.1 style): all intermediate
    encoder levels are concatenated and projected back to the embedding dimension.
    """

    def __init__(self, num_levels: int, dim_embed: int, hidden_factor: int = 2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_levels * dim_embed, hidden_factor * dim_embed, bias=False),
            nn.GELU(),
            nn.Linear(hidden_factor * dim_embed, dim_embed, bias=False),
        )

    def forward(self, levels: list[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(levels, dim=-1))


@dataclasses.dataclass
class LatentState:
    """
    A dataclass to encapsulate the latent state aka the intput to latent heads.
    """

    class_token: torch.Tensor
    register_tokens: torch.Tensor
    patch_tokens: torch.Tensor
    z_pre_norm: torch.Tensor


class LatentPredictionHeadTransformer(nn.Module):
    def __init__(
        self,
        cf: Config,
        name: str,
        in_dim: int,
        loss_conf,
        use_class_token: bool,
        use_patch_token: bool,
    ):
        super().__init__()

        self.name = name

        out_dim, num_blocks, num_heads, with_qk_lnorm, intermediate_dim, dropout_rate = (
            loss_conf["out_dim"],
            loss_conf["num_blocks"],
            loss_conf["num_heads"],
            loss_conf["with_qk_lnorm"],
            loss_conf["intermediate_dim"],
            loss_conf["dropout_rate"],
        )

        self.global_cf = cf
        self.use_class_token = use_class_token
        self.use_patch_token = use_patch_token

        self.blocks = nn.ModuleList()

        # first map to intermediate_dim to introduce a bottleneck
        self.blocks.append(nn.Linear(in_dim, intermediate_dim, bias=False))

        for _ in range(num_blocks):
            self.blocks.append(
                MultiSelfAttentionHead(
                    intermediate_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    with_qk_lnorm=with_qk_lnorm,
                    with_flash=self.global_cf.with_flash_attention,
                    use_xsa=self.global_cf.get("use_xsa", False),
                    norm_type=self.global_cf.norm_type,
                    qk_norm_type=self.global_cf.qk_norm_type,
                    # dim_aux=dim_aux,
                    norm_eps=self.global_cf.norm_eps,
                    attention_dtype=get_dtype(self.global_cf.attention_dtype),
                )
            )
            # Add MLP block
            self.blocks.append(
                MLP(
                    intermediate_dim,
                    intermediate_dim,
                    hidden_factor=4,
                    with_residual=True,
                    dropout_rate=dropout_rate,
                    mlp_type=loss_conf.get("mlp_type", self.global_cf.get("mlp_type", "mlp")),
                    norm_type=self.global_cf.norm_type,
                    # dim_aux=dim_aux,
                    norm_eps=self.global_cf.mlp_norm_eps,
                )
            )

        # finally map from intermediate_dim to the out_dim
        self.blocks.append(nn.Linear(intermediate_dim, out_dim, bias=False))

    def forward(self, x: LatentState):
        # we concatenate the patch and class tokens to process them together
        # We concatenate in the token dimension [Batch, Tokens, Dim]
        patch_class_tokens = []
        if self.use_class_token:
            patch_class_tokens.append(x.class_token)
        if self.use_patch_token:
            patch_class_tokens.append(x.patch_tokens)
        patch_class_tokens = torch.cat(patch_class_tokens, dim=1)

        for _b_idx, block in enumerate(self.blocks):
            if isinstance(block, torch.nn.modules.normalization.LayerNorm):
                patch_class_tokens = block(patch_class_tokens)
            else:
                patch_class_tokens = checkpoint(block, patch_class_tokens, use_reentrant=False)
        return patch_class_tokens


class LatentPredictionHeadIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        return

    def forward(self, x: LatentState):
        return x.patch_tokens


class LatentPredictionHeadMLP(nn.Module):
    def __init__(
        self,
        name,
        in_dim: int,
        loss_conf,
        use_class_token: bool,
        use_patch_token: bool,
        default_mlp_type: str = "mlp",
    ):
        super().__init__()

        self.name = name

        out_dim, num_layers, hidden_factor = (
            loss_conf["out_dim"],
            loss_conf["num_layers"],
            loss_conf["hidden_factor"],
        )

        self.use_class_token = use_class_token
        self.use_patch_token = use_patch_token

        # Create an MLP block
        self.blocks = MLP(
            in_dim,
            out_dim,
            num_layers,
            hidden_factor,
            mlp_type=loss_conf.get("mlp_type", default_mlp_type),
        )

    def forward(self, x: LatentState):
        outputs = []
        if self.use_class_token:
            outputs.append(self.blocks(x.class_token))
        if self.use_patch_token:
            outputs.append(self.blocks(x.patch_tokens))

        return torch.cat(outputs, dim=1)


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DiffusionPrediction:
    """Deferred diffusion prediction returned by LatentPredictionHeadDiffusion during training.

    The predictor head cannot denoise during the student forward pass because teacher targets
    are not yet available. This object carries the student conditioning and a reference to the
    denoising method so the loss module can complete the prediction.
    """

    conditioning: torch.Tensor  # (B, N, D_inner) projected student tokens
    denoise_and_loss: Callable  # (teacher_target, mask) -> (loss, info_dict)


class LatentPredictionHeadDiffusion(nn.Module):
    """Diffusion-based JEPA predictor head.

    Instead of directly predicting teacher targets, this head learns a conditional denoising
    model: given student latents as conditioning, it denoises noised teacher targets.

    Training: returns a ``DiffusionPrediction`` (deferred — loss module completes denoising).
    Inference: runs iterative EDM sampling from pure noise, returns a tensor.
    """

    def __init__(
        self,
        cf: Config,
        name: str,
        in_dim: int,
        loss_conf,
        use_class_token: bool,
        use_patch_token: bool,
    ):
        super().__init__()
        self.name = name
        self.global_cf = cf
        self.use_class_token = use_class_token
        self.use_patch_token = use_patch_token

        # Dimensions
        self.out_dim = loss_conf["out_dim"]
        intermediate_dim = loss_conf["intermediate_dim"]
        num_blocks = loss_conf["num_blocks"]
        num_heads = loss_conf["num_heads"]
        dropout_rate = loss_conf["dropout_rate"]
        with_qk_lnorm = loss_conf["with_qk_lnorm"]

        # EDM parameters (from loss_conf, with fallbacks to global config)
        self.sigma_data = loss_conf.get("sigma_data", cf.get("sigma_data", 1.0))
        self.sigma_min = loss_conf.get("sigma_min", cf.get("sigma_min", 0.002))
        self.sigma_max = loss_conf.get("sigma_max", cf.get("sigma_max", 80.0))
        self.p_mean = loss_conf.get("p_mean", cf.get("p_mean", -1.2))
        self.p_std = loss_conf.get("p_std", cf.get("p_std", 1.2))
        self.rho = loss_conf.get("rho", cf.get("rho", 7))
        self.num_inference_steps = loss_conf.get("num_inference_steps", 20)
        freq_embed_dim = loss_conf.get("frequency_embedding_dim", 256)

        # Noise embedder (reuses the same architecture as DiffusionForecastEngine)
        noise_embed_dim = intermediate_dim
        self.noise_embedder = nn.Sequential(
            nn.Linear(freq_embed_dim, noise_embed_dim),
            nn.SiLU(),
            nn.Linear(noise_embed_dim, noise_embed_dim),
        )
        self.freq_embed_dim = freq_embed_dim

        # Conditioning projection: student tokens -> intermediate_dim
        self.cond_proj = nn.Linear(in_dim, intermediate_dim, bias=False)

        # Target projection: target/noised tokens -> intermediate_dim
        self.target_proj = nn.Linear(self.out_dim, intermediate_dim, bias=False)

        # Denoising transformer blocks (with DiT-style noise conditioning)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                MultiSelfAttentionHead(
                    intermediate_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    with_qk_lnorm=with_qk_lnorm,
                    with_flash=cf.with_flash_attention,
                    use_xsa=cf.get("use_xsa", False),
                    norm_type=cf.norm_type,
                    qk_norm_type=cf.get("qk_norm_type", cf.norm_type),
                    dim_aux=noise_embed_dim,
                    norm_eps=cf.norm_eps,
                    attention_dtype=get_dtype(cf.attention_dtype),
                    is_dit=True,
                )
            )
            self.blocks.append(
                MLP(
                    intermediate_dim,
                    intermediate_dim,
                    hidden_factor=4,
                    with_residual=True,
                    dropout_rate=dropout_rate,
                    mlp_type=loss_conf.get("mlp_type", cf.get("mlp_type", "mlp")),
                    norm_type=cf.norm_type,
                    dim_aux=noise_embed_dim,
                    norm_eps=cf.mlp_norm_eps,
                    is_dit=True,
                )
            )

        # Output projection: intermediate_dim -> out_dim
        self.out_proj = nn.Linear(intermediate_dim, self.out_dim, bias=False)

        # Initialize output projection to near-zero for stable start
        nn.init.normal_(self.out_proj.weight, mean=0, std=0.001)

    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal positional embedding for noise level."""
        half = self.freq_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_embed_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def _embed_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """Embed noise level sigma into a conditioning vector."""
        c_noise = sigma.log() / 4
        t_freq = self._timestep_embedding(c_noise)
        return self.noise_embedder(t_freq)

    def _edm_precondition(
        self,
        x_noised: torch.Tensor,
        cond: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """EDM-preconditioned denoising: D(x; sigma) = c_skip*x + c_out*F(c_in*x, cond; sigma)."""
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()

        noise_emb = self._embed_noise(sigma)  # (B, D_noise)

        # Project and concatenate: [cond_tokens ; noised_target_tokens]
        target_scaled = self.target_proj(c_in * x_noised)  # (B, N, D_inner)
        combined = torch.cat([cond, target_scaled], dim=1)  # (B, 2N, D_inner)

        for block in self.blocks:
            combined = checkpoint(block, combined, None, noise_emb, use_reentrant=False)

        # Slice out the target portion (second half)
        n_target = x_noised.shape[1]
        f_out = combined[:, -n_target:]
        f_out = self.out_proj(f_out)  # (B, N, out_dim)

        return c_skip * x_noised + c_out * f_out

    def _training_denoise_and_loss(
        self,
        cond: torch.Tensor,
        teacher_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Called by the loss module. Noises teacher target, denoises, computes masked loss.

        Args:
            cond: Student conditioning (B, N, D_inner) from forward().
            teacher_target: Clean teacher patch tokens (B, N, D_target).
            mask: Boolean mask (B, N) — True where loss should be computed.

        Returns:
            loss: Scalar loss value.
            info: Dict with diagnostic values.
        """
        b = teacher_target.shape[0]
        device = teacher_target.device

        # Sample sigma from log-normal: ln(sigma) ~ N(p_mean, p_std^2)
        eta = torch.randn(b, device=device)
        sigma = (eta * self.p_std + self.p_mean).exp()  # (B,)

        # Add noise
        noise = torch.randn_like(teacher_target)
        x_noised = teacher_target + sigma.view(b, 1, 1) * noise

        # Denoise
        denoised = self._edm_precondition(x_noised, cond, sigma)

        # EDM loss weight: lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        loss_weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2  # (B,)

        # Per-token MSE, then mask
        per_token_mse = (denoised - teacher_target).pow(2).mean(dim=-1)  # (B, N)
        mask_f = mask.to(per_token_mse.dtype)
        token_counts = mask_f.sum(dim=-1).clamp(min=1.0)  # (B,)

        # Per-sample masked loss, weighted by noise level
        per_sample_loss = (per_token_mse * mask_f).sum(dim=-1) / token_counts  # (B,)
        weighted_loss = (loss_weight * per_sample_loss)  # (B,)

        # Mean over samples with at least one masked token
        valid = token_counts > 0
        loss = weighted_loss[valid].mean() if valid.any() else weighted_loss.sum() * 0.0

        info = {
            "sigma_mean": sigma.mean().item(),
            "sigma_std": sigma.std().item(),
            "loss_weight_mean": loss_weight.mean().item(),
        }
        return loss, info

    def forward(self, x: LatentState) -> DiffusionPrediction | torch.Tensor:
        """Forward pass.

        Training: returns DiffusionPrediction (deferred denoising).
        Inference: returns denoised tensor via iterative sampling.
        """
        # Extract and project student tokens
        tokens = []
        if self.use_class_token:
            tokens.append(x.class_token)
        if self.use_patch_token:
            tokens.append(x.patch_tokens)
        student_tokens = torch.cat(tokens, dim=1)
        cond = self.cond_proj(student_tokens)  # (B, N, D_inner)

        if self.training:
            return DiffusionPrediction(
                conditioning=cond,
                denoise_and_loss=lambda target, mask: self._training_denoise_and_loss(
                    cond, target, mask
                ),
            )

        return self._inference_forward(cond)

    @torch.no_grad()
    def _inference_forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Iterative EDM sampling from pure noise, conditioned on student latents."""
        b, n_cond, _ = cond.shape
        device = cond.device
        num_steps = self.num_inference_steps

        # Training-aligned sigma bounds (same logic as DiffusionForecastEngine)
        sigma_max_train = math.exp(self.p_mean + 3.0 * self.p_std)
        sigma_max_eff = min(self.sigma_max, sigma_max_train)
        sigma_min_eff = max(self.sigma_min, self.sigma_data * 0.01)

        # Time step discretization (EDM Eq. 5)
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (
            sigma_max_eff ** (1 / self.rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min_eff ** (1 / self.rho) - sigma_max_eff ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        # Start from scaled noise
        x = torch.randn(b, n_cond, self.out_dim, device=device) * t_steps[0].float()

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:], strict=False)):
            sigma = torch.full((b,), t_cur.item(), device=device)
            sigma_next = t_next.item()

            # Euler step
            denoised = self._edm_precondition(x, cond, sigma)
            d_cur = (x - denoised) / t_cur.float()
            x_next = x + (t_next - t_cur).float() * d_cur

            # 2nd order correction (Heun's method)
            if i < num_steps - 1:
                sigma_n = torch.full((b,), sigma_next, device=device)
                denoised_next = self._edm_precondition(x_next, cond, sigma_n)
                d_prime = (x_next - denoised_next) / t_next.float()
                x_next = x + (t_next - t_cur).float() * (0.5 * d_cur + 0.5 * d_prime)

            x = x_next

        return x


class EfficientBilinear(torch.nn.Module):
    def __init__(self, in_dim_lhs, in_dim_rhs, out, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out, in_dim_lhs, in_dim_rhs))
        self.bias = nn.Parameter(torch.zeros(out)) if bias else 0.0
        self.total_in = in_dim_lhs * in_dim_rhs

    def forward(self, x_lhs, x_rhs):
        return torch.einsum("bi,oij,bj->bo", x_lhs, self.weight, x_rhs) + self.bias

    def reset_parameters(self):
        if isinstance(self.weight, nn.Parameter):
            bound = math.sqrt(2.0 / self.total_in)
            nn.init.uniform_(self.weight, -bound, bound)
        if isinstance(self.bias, nn.Parameter):
            nn.init.zeros_(self.bias)


class BilinearDecoder(nn.Module):
    def __init__(self, stream_name, coord_dim, latent_dim, out_dim):
        super().__init__()

        self.name = f"BilinearDecoder_{stream_name}"
        self.latent_dim = latent_dim
        self.bilin = EfficientBilinear(coord_dim, latent_dim, out_dim)

    def forward(self, coords_md, latent_nd, tcs_lens_n1):
        """
        Using Noam Shazeer notation
        N = Number of latent tokens*batch_size (N1 means N+1)
        M = Number of coordinates to decode
        D = Hidden dimension
        """
        latent_md = torch.repeat_interleave(latent_nd, tcs_lens_n1[1:], 0)
        return self.bilin(coords_md, latent_md)
