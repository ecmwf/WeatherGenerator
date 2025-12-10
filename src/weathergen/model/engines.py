# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
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
    """Embedding engine for the model."""

    name: "EmbeddingEngine"

    def __init__(self, cf: Config, sources_size) -> None:
        """Initialize the EmbeddingEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param sources_size: Tensor of number of channels for each stream
        """
        super(EmbeddingEngine, self).__init__()
        self.cf = cf
        self.sources_size = sources_size
        self.embeds = torch.nn.ModuleList()

        for i, si in enumerate(self.cf.streams):
            stream_name = si.get("name", i)

            if si.get("diagnostic", False) or self.sources_size[i] == 0:
                self.embeds.append(torch.nn.Identity())
                continue

            if si["embed"]["net"] == "transformer":
                self.embeds.append(
                    StreamEmbedTransformer(
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
                        embed_size_centroids=self.cf.embed_size_centroids,
                        unembed_mode=self.cf.embed_unembed_mode,
                        stream_name=stream_name,
                    )
                )
            elif si["embed"]["net"] == "linear":
                self.embeds.append(
                    StreamEmbedLinear(
                        self.sources_size[i] * si["token_size"],
                        self.cf.ae_local_dim_embed,
                        stream_name=stream_name,
                    )
                )
            else:
                raise ValueError("Unsupported embedding network type")

    def forward(self, streams_data, pe_embed, dtype, device):
        """Forward pass of the embedding engine.

        :param streams_data: Tensor of streams data.
        :param pe_embed: Positional encoding embeddings.
        :param dtype: Data type for the embeddings.
        :param device: Device to run the embeddings on.

        :return tokens_all: Embedded tokens.
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
            (int(offsets_base[-1]), self.cf.ae_local_dim_embed), dtype=dtype, device=device
        )

        for _, sb in enumerate(streams_data):
            for _, (s, embed) in enumerate(zip(sb, self.embeds, strict=False)):
                if not s.source_empty():
                    idxs = s.source_idxs_embed.to(device)
                    idxs_pe = s.source_idxs_embed_pe.to(device)

                    # create full scatter index
                    idxs = idxs.unsqueeze(
                        1
                    ).expand(
                        -1, self.cf.ae_local_dim_embed
                    )  # expand works as the tensor is only read (!do not apply in-place write 
                       # operations on it!)
                    try:
                        x_embed = embed(s.source_tokens_cells, s.source_centroids).flatten(0, 1)
                    except RuntimeError:
                        # there's undocumented limitation in flash_attn that will make embed fail if
                        # #tokens is too large; code below is a work around
                        x_embed = torch.cat(
                            [
                                embed(s_c, c_c).flatten(0, 1)
                                for s_c, c_c in zip(
                                    torch.split(s.source_tokens_cells, 49152),
                                    torch.split(s.source_centroids, 49152), strict=False,
                                )
                            ]
                        )

                    # scatter write to reorder from per stream to per cell ordering
                    tokens_all.scatter_(0, idxs, x_embed + pe_embed[idxs_pe])
        return tokens_all


class LocalAssimilationEngine(torch.nn.Module):
    """Local assimilation engine for the model.
    
    The LocalAssimilationEngine is responsible for fusing information from different input
    streams (e.g., satellite, station data) within each HEALPix cell. It operates locally,
    meaning attention is computed only among tokens belonging to the same cell. This step 
    aggregates high-resolution, heterogeneous input data into a unified cell-level 
    representation before global interaction takes place. It uses a sequence of self-attention 
    blocks and MLPs.
    """

    name: "LocalAssimilationEngine"

    def __init__(self, cf: Config) -> None:
        """Initialize the LocalAssimilationEngine with the configuration.

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
        """Forward pass of the local assimilation engine.

        :param tokens_c: Tokens to be assimilated.
        :param cell_lens_c: Cell lengths for the tokens.
        :param use_reentrant: Whether to use reentrant mode.

        :return tokens_c: Assimilated tokens.
        """
        for block in self.ae_local_blocks:
            tokens_c = checkpoint(block, tokens_c, cell_lens_c, use_reentrant=use_reentrant)
        return tokens_c


class Local2GlobalAssimilationEngine(torch.nn.Module):
    """Local2GlobalAssimilationEngine for the model."""

    name: "Local2GlobalAssimilationEngine"

    def __init__(self, cf: Config) -> None:
        """Initialize the Local2GlobalAssimilationEngine with the configuration.

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
        """Forward pass of the local to global assimilation engine.

        :param tokens_c: Tokens to be assimilated.
        :param tokens_global_c: Global tokens to be assimilated.
        :param q_cells_lens_c: Query cell lengths for the tokens.
        :param cell_lens_c: Cell lengths for the tokens.
        :param use_reentrant: Whether to use reentrant mode.

        :return tokens_global_c: Assimilated tokens.
        """
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


class GlobalAssimilationEngine(torch.nn.Module):
    """Global assimilation engine for the model.
    
    The GlobalAssimilationEngine processes the unified cell-level representations generated by
    the LocalAssimilationEngine. Its primary role is to model long-range dependencies and
    physical interactions across the entire globe. It alternates between local attention 
    (focusing on neighboring cells) and global attention (fully connected or sparse global 
    patterns) to efficiently propagate information. This engine transforms the local latents 
    into a globally consistent state representation.
    """

    name: "GlobalAssimilationEngine"

    def __init__(self, cf: Config, num_healpix_cells: int) -> None:
        """Initialize the GlobalAssimilationEngine with the configuration.

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

    def forward(self, tokens, use_reentrant):
        """Forward pass of the global assimilation engine.

        :param tokens: Tokens to be assimilated.
        :param use_reentrant: Whether to use reentrant mode.

        :return tokens: Assimilated tokens.
        """
        for block in self.ae_global_blocks:
            tokens = checkpoint(block, tokens, use_reentrant=use_reentrant)
        return tokens


class ForecastingEngine(torch.nn.Module):
    """Forecasting engine for the model."""

    name: "ForecastingEngine"

    def __init__(self, cf: Config, num_healpix_cells: int) -> None:
        """Initialize the ForecastingEngine with the configuration.

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
                            dim_aux=1,
                            norm_eps=self.cf.norm_eps,
                            attention_dtype=get_dtype(self.cf.attention_dtype),
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
                            dim_aux=1,
                            norm_eps=self.cf.norm_eps,
                            attention_dtype=get_dtype(self.cf.attention_dtype),
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
                        dim_aux=1,
                        norm_eps=self.cf.mlp_norm_eps,
                    )
                )

        def init_weights_final(m):
            """Initialize the weights of the forecasting engine."""
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, mean=0, std=0.001)

        for block in self.fe_blocks:
            block.apply(init_weights_final)

    def forward(self, tokens, fstep):
        """Forward pass of the forecasting engine.

        :param tokens: Tokens to be forecasted.
        :param fstep: Forecast step.

        :return tokens: Forecasted tokens.
        """
        aux_info = torch.tensor([fstep], dtype=torch.float32, device="cuda")
        for block in self.fe_blocks:
            tokens = checkpoint(block, tokens, aux_info, use_reentrant=False)

        return tokens


class EnsPredictionHead(torch.nn.Module):
    """Ensemble prediction head for the model."""

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
        """Initialize the EnsPredictionHead with the configuration.

        :param dim_embed: Dimension of the embedding.
        :param dim_out: Dimension of the output.
        :param ens_num_layers: Number of layers in the ensemble.
        :param ens_size: Size of the ensemble.
        :param stream_name: Name of the stream.
        :param norm_type: Type of normalization.
        :param hidden_factor: Hidden factor to create an internal dimension.
        :param final_activation: Optional final activation function.
        """

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
        """Forward pass of the EnsPredictionHead.

        :param toks: Tokens to be predicted.

        :return preds: Ensemble predictions.
        """
        preds = []
        for pred_head in self.pred_heads:
            cpred = toks
            for block in pred_head:
                cpred = block(cpred)
            preds.append(cpred)
        preds = torch.stack(preds, 0)

        return preds


class TargetPredictionEngineClassic(nn.Module):
    """Target prediction engine for the model.
    
    The TargetPredictionEngineClassic is a specialized decoding module that projects the global
    latent states back to specific target coordinates (e.g., station locations). It typically 
    employs a PerceiverIO-style architecture where target coordinate embeddings query the 
    latent state via cross-attention. This engine is "Classic" in the sense that it strictly 
    follows the original design with coordinate conditioning and optional self-attention, 
    without the flexible decoder types found in the newer `TargetPredictionEngine`.
    """

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
        """Initialize the TargetPredictionEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param dims_embed: Tensor of embedding dimensions for each layer.
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
        """Forward pass of the TargetPredictionEngineClassic.

        :param latent: Latent tokens.
        :param output: Output tokens.
        :param latent_lens: Lengths of the latent tokens.
        :param output_lens: Lengths of the output tokens.
        :param coordinates: Coordinates of the tokens.

        :returns tc_tokens: Output tokens.
        """
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
    """TargetPredictionEngine for the model.
    
    The TargetPredictionEngine handles the decoding of the latent representation into the target
    observational space. Unlike the Classic version which solely relies on a fixed
    PerceiverIO-like structure with coordinate conditioning, this engine is configurable via
    `decoder_type`. It supports various conditioning mechanisms, allowing for experimentation
    with how the latent state and auxiliary information (like coordinates) are fused to generate
    predictions. It includes normalization, optional positional embeddings and a flexible 
    sequence of decoding blocks.
    """

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
        """Initialize the TargetPredictionEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param dims_embed: Tensor of embedding dimensions for each layer.
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
        """Forward pass of the TargetPredictionEngine.

        :param latent: Latent tokens.
        :param output: Output tokens.
        :param latent_lens: Lengths of the latent tokens.
        :param output_lens: Lengths of the output tokens.
        :param coordinates: Coordinates of the tokens.

        :return output: Output tokens.
        """
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
