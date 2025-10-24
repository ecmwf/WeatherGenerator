# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch.nn as nn

from weathergen.model.attention import (
    MultiCrossAttentionHeadVarlen,
    MultiSelfAttentionHeadVarlen,
)
from weathergen.model.layers import MLP, MoEMLP
from weathergen.model.norms import AdaLayerNormLayer
from weathergen.utils.utils import get_dtype

import logging
logger = logging.getLogger(__name__)

class SelfAttentionBlock(nn.Module):
    """
    A self attention block, i.e., adaptive layer norm with multi head self attenttion and adaptive
    layer norm with a FFN.
    """

    def __init__(self, dim, dim_aux, with_adanorm, num_heads, dropout_rate, **kwargs):
        super().__init__()

        self.with_adanorm = with_adanorm

        self.mhsa = MultiSelfAttentionHeadVarlen(
            dim_embed=dim,
            num_heads=num_heads,
            with_residual=False,
            **kwargs["attention_kwargs"],
        )
        if self.with_adanorm:
            self.mhsa_block = AdaLayerNormLayer(dim, dim_aux, self.mhsa, dropout_rate)
        else:
            self.ln_sa = nn.LayerNorm(dim, eps=kwargs["attention_kwargs"]["norm_eps"])
            self.mhsa_block = lambda x, _, **kwargs: self.mhsa(self.ln_sa(x), **kwargs) + x

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        use_moe_ffn = (kwargs.get("ffn_mlp_type", "dense") == "moe")
        ffn_hidden_factor = kwargs.get("ffn_hidden_factor", 4)
        moe_kwargs = kwargs.get("moe_kwargs", {})  # e.g. num_experts, top_k, router_noisy_std
        
        if use_moe_ffn:
            self.mlp = MoEMLP(
                dim_in=dim,
                dim_out=dim,
                hidden_factor=ffn_hidden_factor,
                dropout_rate=0.1,
                nonlin=nn.GELU,              # internal block constructs nonlin()
                with_residual=False,
                norm_type=kwargs["attention_kwargs"]["norm_type"],
                dim_aux=(dim_aux if self.with_adanorm else None),
                norm_eps=kwargs["attention_kwargs"]["norm_eps"],
                **moe_kwargs,                # <- e.g. num_experts=8, top_k=2, router_noisy_std=0.01
            )
        else:
            self.mlp = MLP(
                dim_in=dim,
                dim_out=dim,
                hidden_factor=4,
                dropout_rate=0.1,
                nonlin=approx_gelu,
                with_residual=False,
            )
        if self.with_adanorm:
            self.mlp_fn = lambda x, **kwargs: self.mlp(x)
            self.mlp_block = AdaLayerNormLayer(dim, dim_aux, self.mlp_fn, dropout_rate)
        else:
            self.ln_mlp = nn.LayerNorm(norm_eps=kwargs["attention_kwargs"]["norm_eps"])
            self.mlp_block = lambda x, _, **kwargs: self.mlp(self.ln_mlp(x), None, **kwargs) + x

        self.initialise_weights()
        if self.with_adanorm:
            # Has to happen after the basic weight init to ensure it is zero!
            self.mhsa_block.initialise_weights()
            self.mlp_block.initialise_weights()

    def initialise_weights(self):
        # Initialise transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x, x_lens, aux=None):
        # we have aux_lens as arg to be consistent with the CrossAttentionBlock
        assert self.with_adanorm ^ (aux is None), "Conditioning is not being used"
        x = self.mhsa_block(x, aux, x_lens=x_lens)
        x = self.mlp_block(x, aux)
        return x


class CrossAttentionBlock(nn.Module):
    """
    A cross attention block, i.e., adaptive layer norm with cross attenttion and adaptive layer norm
    with a FFN.
    """

    def __init__(
        self,
        dim_q,
        dim_kv,
        dim_aux,
        with_self_attn,
        with_adanorm,
        with_mlp,
        num_heads,
        dropout_rate,
        **kwargs,
    ):
        super().__init__()

        self.with_adanorm = with_adanorm
        self.with_self_attn = with_self_attn
        self.with_mlp = with_mlp

        if with_self_attn:
            self.mhsa = MultiSelfAttentionHeadVarlen(
                dim_embed=dim_q,
                num_heads=num_heads,
                with_residual=False,
                **kwargs["attention_kwargs"],
            )
            if self.with_adanorm:
                self.mhsa_block = AdaLayerNormLayer(dim_q, dim_aux, self.mhsa, dropout_rate)
            else:
                self.ln_sa = nn.LayerNorm(dim_q, eps=kwargs["attention_kwargs"]["norm_eps"])
                self.mhsa_block = lambda x, _, **kwargs: self.mhsa(self.ln_sa(x), **kwargs) + x

        self.cross_attn = MultiCrossAttentionHeadVarlen(
            dim_embed_q=dim_q,
            dim_embed_kv=dim_kv,
            num_heads=num_heads,
            with_residual=False,
            **kwargs["attention_kwargs"],
        )
        if self.with_adanorm:
            self.cross_attn_block = AdaLayerNormLayer(dim_q, dim_aux, self.cross_attn, dropout_rate)
        else:
            self.ln_ca = nn.LayerNorm(dim_q, eps=kwargs["attention_kwargs"]["norm_eps"])
            self.cross_attn_block = (
                lambda x, _, **kwargs: self.cross_attn(self.ln_ca(x), **kwargs) + x
            )

        if self.with_mlp:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            
            use_moe_ffn = (kwargs.get("ffn_mlp_type", "dense") == "moe")
            ffn_hidden_factor = kwargs.get("ffn_hidden_factor", 4)
            moe_kwargs = kwargs.get("moe_kwargs", {})
            
            if use_moe_ffn:
                self.mlp = MoEMLP(
                    dim_in=dim_q,
                    dim_out=dim_q,
                    hidden_factor=ffn_hidden_factor,
                    dropout_rate=0.1,
                    nonlin=nn.GELU,          # internal block constructs nonlin()
                    with_residual=False,
                    norm_type=kwargs["attention_kwargs"]["norm_type"],
                    dim_aux=(dim_aux if self.with_adanorm else None),
                    norm_eps=kwargs["attention_kwargs"]["norm_eps"],
                    **moe_kwargs,            # <- e.g. num_experts=8, top_k=2, router_noisy_std=0.01
                )
            else:
                self.mlp = MLP(
                    dim_in=dim_q,
                    dim_out=dim_q,
                    hidden_factor=4,
                    nonlin=approx_gelu,
                    with_residual=False,
                )
            if self.with_adanorm:
                self.mlp_fn = lambda x, **kwargs: self.mlp(x)
                self.mlp_block = AdaLayerNormLayer(dim_q, dim_aux, self.mlp_fn, dropout_rate)
            else:
                self.ln_mlp = nn.LayerNorm(eps=kwargs["attention_kwargs"]["norm_eps"])
                self.mlp_block = lambda x, _, **kwargs: self.mlp(self.ln_mlp(x)) + x
        else:
            self.mlp_block = lambda x, _, **kwargs: x

        self.initialise_weights()
        if self.with_adanorm:
            # Has to happen after the basic weight init to ensure it is zero!
            self.mhsa_block.initialise_weights()
            self.cross_attn_block.initialise_weights()
            self.mlp_block.initialise_weights()

    def initialise_weights(self):
        # Initialise transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x, x_kv, aux, x_kv_lens=None, x_lens=None):
        x = self.cross_attn_block(x, aux, x_kv=x_kv, x_lens=x_lens, x_kv_lens=x_kv_lens)
        if self.with_self_attn:
            x = self.mhsa_block(x, aux, x_lens=x_lens)
        x = self.mlp_block(x, aux, x_lens=x_lens)
        return x


class OriginalPredictionBlock(nn.Module):
    def __init__(
        self,
        config,
        dim_in,
        dim_out,
        dim_kv,
        dim_aux,
        num_heads,
        attention_kwargs,
        tr_dim_head_proj,
        tr_mlp_hidden_factor,
        tro_type,
        mlp_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__()

        self.cf = config
        self.tro_type = tro_type
        self.tr_dim_head_proj = tr_dim_head_proj
        self.tr_mlp_hidden_factor = tr_mlp_hidden_factor

        self.block = nn.ModuleList()

        # Multi-Cross Attention Head
        self.block.append(
            MultiCrossAttentionHeadVarlen(
                dim_in,
                self.cf.ae_global_dim_embed,
                self.cf.streams[0]["target_readout"]["num_heads"],
                dim_head_proj=self.tr_dim_head_proj,
                with_residual=True,
                with_qk_lnorm=True,
                dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                with_flash=self.cf.with_flash_attention,
                norm_type=attention_kwargs["norm_type"],
                softcap=0.0,
                dim_aux=dim_aux,
                norm_eps=attention_kwargs["norm_eps"],
                attention_dtype=get_dtype(self.cf.attention_dtype),
            )
        )

        # Optional Self-Attention Head
        if self.cf.pred_self_attention:
            self.block.append(
                MultiSelfAttentionHeadVarlen(
                    dim_in,
                    num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
                    dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                    with_qk_lnorm=True,
                    with_flash=self.cf.with_flash_attention,
                    norm_type=self.cf.norm_type,
                    dim_aux=dim_aux,
                    norm_eps=self.cf.norm_eps,
                    attention_dtype=get_dtype(self.cf.attention_dtype),
                )
            )

        # MLP Block
        # Add MoE option
        use_moe = getattr(self.cf, "decoder_mlp_type", "dense") == "moe"
        logger.info(
            "[MoE] Decoder head: type=%s%s",
            "moe" if use_moe else "dense",
            ("" if not use_moe else
            f" (experts={getattr(self.cf,'moe_num_experts',None)}, top_k={getattr(self.cf,'moe_top_k',None)})"),
        )

        if use_moe:
            self.block.append(
                MoEMLP(
                    dim_in,
                    dim_out,
                    hidden_factor=self.tr_mlp_hidden_factor,
                    dropout_rate=0.1,
                    with_residual=True,                        # mirror dense
                    norm_type=self.cf.norm_type,
                    dim_aux=(dim_aux if self.cf.pred_mlp_adaln else None),
                    norm_eps=self.cf.mlp_norm_eps,
                    num_experts=getattr(self.cf, "moe_num_experts", 8),
                    top_k=getattr(self.cf, "moe_top_k", 2),
                    router_noisy_std=getattr(self.cf, "moe_router_noisy_std", 0.0),
                    use_checkpoint=getattr(self.cf, "moe_use_checkpoint", False),
                )
            )
        else:
            self.block.append(
                MLP(
                    dim_in,
                    dim_out,
                    with_residual=True,
                    hidden_factor=self.tr_mlp_hidden_factor,
                    dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                    norm_type=self.cf.norm_type,
                    dim_aux=(dim_aux if self.cf.pred_mlp_adaln else None),
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )

    def forward(self, latent, output, coords, latent_lens, output_lens):
        for layer in self.block:
            if isinstance(layer, MultiCrossAttentionHeadVarlen):
                output = layer(output, latent, output_lens, latent_lens, coords)
            else:
                output = layer(output, output_lens, coords)
        return output
