# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch.nn as nn

from weathergen.model.norms import AdaLayerNormLayer
from weathergen.model.attention import (
    MultiSelfAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlen,
)
from weathergen.model.layers import MLP


class SelfAttentionBlock(nn.Module):
    """
    A self attention block, i.e., adaptive layer norm with multi head self attenttion and adaptive layer norm with a FFN.
    """

    def __init__(self, dim, dim_aux, with_adanorm=True, num_heads=8, dropout_rate=0.1, **kwargs):
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
    A cross attention block, i.e., adaptive layer norm with cross attenttion and adaptive layer norm with a FFN.
    """

    def __init__(
        self,
        dim_q,
        dim_kv,
        dim_aux,
        with_self_attn=True,
        with_adanorm=True,
        with_mlp=True,
        num_heads=8,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__()

        self.with_adanorm = with_adanorm
        self.with_self_attn = with_self_attn
        self.with_mlp = with_self_attn

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
                self.mhsa_block = (
                    lambda x, _, **kwargs: self.mhsa(self.ln_sa(x), **kwargs) + x
                )

        self.cross_attn = MultiCrossAttentionHeadVarlen(
            dim_embed_q=dim_q,
            dim_embed_kv=dim_kv,
            num_heads=num_heads,
            with_residual=False,
            **kwargs["attention_kwargs"],
        )
        if self.with_adanorm:
            self.cross_attn_block = AdaLayerNormLayer(
                dim_q, dim_aux, self.cross_attn, dropout_rate
            )
        else:
            self.ln_ca = nn.LayerNorm(dim_q,eps=kwargs["attention_kwargs"]["norm_eps"])
            self.cross_attn_block = (
                lambda x, _, **kwargs: self.cross_attn(self.ln_ca(x), **kwargs) + x
            )

        if self.with_mlp:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
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
                self.ln_mlp = nn.LayerNorm(dim_q, eps=kwargs["attention_kwargs"]["norm_eps"])
                self.mlp_block = (
                    lambda x, _, **kwargs: self.mlp(self.ln_mlp(x)) + x
                )
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
