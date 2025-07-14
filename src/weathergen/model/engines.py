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

from weathergen.model.attention import (
    MultiCrossAttentionHead_Varlen,
    MultiCrossAttentionHead_Varlen_SlicedQ,
    MultiSelfAttentionHead,
    MultiSelfAttentionHead_Local,
    MultiSelfAttentionHead_Varlen,
)
from weathergen.model.activations import SwiGLU
from weathergen.model.blocks import (
    SelfAttentionBlock,
    CrossAttentionBlock,
)
from weathergen.model.embeddings import (
    StreamEmbedLinear,
    StreamEmbedTransformer,
)
from weathergen.model.layers import MLP
from weathergen.utils.config import Config, get_dtype


class EmbeddingEngine:
    def __init__(self, cf: Config, sources_size) -> None:
        """
        Initialize the EmbeddingEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param sources_size: List of source sizes for each stream.
        """
        self.cf = cf
        self.sources_size = sources_size  # KCT:iss130, what is this?
        self.embeds = torch.nn.ModuleList()

    def create(self) -> torch.nn.ModuleList:
        """
        Creates and returns the module list (embeds).

        :return: torch.nn.ModuleList containing the embedding layers.
        """
        for i, si in enumerate(self.cf.streams):
            if "diagnostic" in si and si["diagnostic"]:
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
                        norm_type=self.cf.norm_type,
                        embed_size_centroids=self.cf.embed_size_centroids,
                        unembed_mode=self.cf.embed_unembed_mode,
                    )
                )
            elif si["embed"]["net"] == "linear":
                self.embeds.append(
                    StreamEmbedLinear(
                        self.sources_size[i] * si["token_size"], self.cf.ae_local_dim_embed
                    )
                )
            else:
                raise ValueError("Unsupported embedding network type")
        return self.embeds


class LocalAssimilationEngine:
    def __init__(self, cf: Config) -> None:
        """
        Initialize the LocalAssimilationEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        """
        self.cf = cf
        self.ae_local_blocks = torch.nn.ModuleList()

    def create(self) -> torch.nn.ModuleList:
        """
        Creates and returns the module list (ae_local_blocks).

        :return: torch.nn.ModuleList containing the local assimilation blocks.
        """
        for _ in range(self.cf.ae_local_num_blocks):
            self.ae_local_blocks.append(
                MultiSelfAttentionHead_Varlen(
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
        return self.ae_local_blocks


class Local2GlobalAssimilationEngine:
    def __init__(self, cf: Config) -> None:
        """
        Initialize the Local2GlobalAssimilationEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        """
        self.cf = cf
        self.ae_adapter = torch.nn.ModuleList()

    def create(self) -> torch.nn.ModuleList:
        """
        Creates and returns the module list (ae_adapter).

        :return: torch.nn.ModuleList containing the local-to-global assimilation adapter blocks.
        """
        self.ae_adapter.append(
            MultiCrossAttentionHead_Varlen_SlicedQ(
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
            MultiCrossAttentionHead_Varlen_SlicedQ(
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
        return self.ae_adapter


class GlobalAssimilationEngine:
    def __init__(self, cf: Config, num_healpix_cells: int) -> None:
        """
        Initialize the GlobalAssimilationEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param num_healpix_cells: Number of healpix cells used for local queries.
        """
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells

        self.ae_global_blocks = torch.nn.ModuleList()

    def create(self) -> torch.nn.ModuleList:
        """
        Creates and returns the module list (ae_global_blocks).

        :return: torch.nn.ModuleList containing the global assimilation blocks.
        """
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
                    MultiSelfAttentionHead_Local(
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
        return self.ae_global_blocks


class ForecastingEngine:
    def __init__(self, cf: Config, num_healpix_cells: int) -> None:
        """
        Initialize the ForecastingEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param num_healpix_cells: Number of healpix cells used for local queries.
        """
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells
        self.fe_blocks = torch.nn.ModuleList()

    def create(self) -> torch.nn.ModuleList:
        """
        Creates and returns the module list (fe_blocks).

        :return: torch.nn.ModuleList containing the forecasting blocks.
        """
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
                        MultiSelfAttentionHead_Local(
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
        return self.fe_blocks


class EnsPredictionHead(torch.nn.Module):
    #########################################
    def __init__(
        self, dim_embed, dim_out, ens_num_layers, ens_size, norm_type="LayerNorm", hidden_factor=2
    ):
        """Constructor"""

        super(EnsPredictionHead, self).__init__()

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

    #########################################
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, toks):
        preds = []
        for pred_head in self.pred_heads:
            cpred = toks
            for block in pred_head:
                cpred = block(cpred)
            preds.append(cpred)
        preds = torch.stack(preds, 0)

        return preds


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
        super(TargetPredictionEngine, self).__init__()

        self.cf = cf
        self.dims_embed = dims_embed
        self.dim_coord_in = dim_coord_in
        self.tr_dim_head_proj = tr_dim_head_proj
        self.tr_mlp_hidden_factor = tr_mlp_hidden_factor
        self.softcap = softcap
        self.tro_type = tro_type

        attention_kwargs = {
            "dropout_rate": 0.1,  # Assuming dropout_rate is 0.1
            "with_flash": self.cf.with_flash_attention,
            "norm_type": self.cf.norm_type,
            "softcap": self.softcap,
            "dim_aux": self.dim_coord_in,
            "norm_eps": self.cf.norm_eps,
            "attention_dtype": get_dtype(self.cf.attention_dtype),
        }
        self.dim_adapter = nn.Sequential(
            SwiGLU(),
            nn.Linear(self.cf.ae_global_dim_embed // 2, self.dims_embed[0]),
            nn.LayerNorm(self.dims_embed[0]),
        )
        self.cross_attn = MultiCrossAttentionHead_Varlen(
            dim_embed_q=self.dims_embed[0],
            dim_embed_kv=self.dims_embed[0],
            num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
            with_residual=True,
            with_qk_lnorm=False,
            **attention_kwargs,
        )
        attention_kwargs["with_qk_lnorm"] = True

        prev_dim = self.dims_embed[0]
        # prev_dim = self.cf.ae_global_dim_embed
        self.tte = nn.ModuleList()
        for ith, next_dim in enumerate(self.dims_embed[1:]):
            if self.cf.decoder_type == "PerceiverIO":
                # a single cross attention layer as per https://arxiv.org/pdf/2107.14795
                self.tte.append(
                    CrossAttentionBlock(
                        dim=prev_dim,
                        dim_aux=next_dim,
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
                        dim=prev_dim,
                        dim_aux=prev_dim,
                        num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
                        attention_kwargs=attention_kwargs,
                        with_adanorm=True,
                    )
                )
            elif self.cf.decoder_type == "CrossAttentionConditioning":
                self.tte.append(
                    CrossAttentionBlock(
                        dim=prev_dim,
                        dim_aux=next_dim,
                        num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
                        with_self_attn=True,
                        with_adanorm=False,
                        with_mlp=True,
                        attention_kwargs=attention_kwargs,
                    )
                )
            elif self.cf.decoder_type == "CrossAttentionAdaNormConditioning":
                self.tte.append(
                    CrossAttentionBlock(
                        dim=prev_dim,
                        dim_aux=next_dim,
                        num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
                        with_self_attn=True,
                        with_adanorm=True,
                        with_mlp=True,
                        attention_kwargs=attention_kwargs,
                    )
                )
            else:
                raise NotImplementedError(
                    f"{self.cf.decoder_type} is not implemented for prediction heads"
                )
            prev_dim = self.dims_embed[ith]

    def forward(self, latent, output_cond, latent_lens, output_cond_lens):
        latent = checkpoint(self.dim_adapter, latent, use_reentrant=False)
        latent = checkpoint(
            self.cross_attn,
            x_q=output_cond,
            x_kv=latent,
            x_kv_lens=latent_lens,
            x_q_lens=output_cond_lens,
            use_reentrant=False,
        )
        for layer in self.tte:
            latent = checkpoint(
                layer,
                x=latent,
                x_lens=output_cond_lens,
                aux=output_cond,
                aux_lens=output_cond_lens,
                use_reentrant=False,
            )
        return latent
