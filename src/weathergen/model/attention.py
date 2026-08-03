# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import partial

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from weathergen.model.layers import LinearNormConditioning
from weathergen.model.norms import AdaLayerNorm, AdaLNZero, RMSNorm
from weathergen.model.positional_encoding import rotary_pos_emb_2d

"""
Attention blocks used by WeatherGenerator.

Some blocks optionally apply 2D RoPE. When enabled, the caller must provide per-token 2D
coordinates aligned with the token order (lat, lon in radians).
"""


def _apply_xsa(attn_out: torch.Tensor, self_values: torch.Tensor) -> torch.Tensor:
    attn_out_float = attn_out.float()
    self_values_float = self_values.float()
    denom = (
        self_values_float.pow(2)
        .sum(dim=-1, keepdim=True)
        .clamp_min(torch.finfo(self_values_float.dtype).eps)
    )
    proj = (attn_out_float * self_values_float).sum(dim=-1, keepdim=True) / denom
    return (attn_out_float - (proj * self_values_float)).to(attn_out.dtype)


class MultiSelfAttentionHeadVarlen(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
        with_2d_rope=False,
        use_xsa=False,
    ):
        super(MultiSelfAttentionHeadVarlen, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.with_flash = with_flash
        self.softcap = softcap
        self.with_residual = with_residual
        self.use_xsa = use_xsa
        self.with_2d_rope = with_2d_rope
        self.use_xsa = use_xsa

        assert dim_embed % num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        if dim_aux is not None:
            self.lnorm = AdaLayerNorm(dim_embed, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm = norm(dim_embed, eps=norm_eps)
        self.proj_heads_q = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_v = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_out = torch.nn.Linear(dim_embed, dim_embed, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype

        assert with_flash, "Only flash attention supported at the moment"

    def forward(self, x, x_lens, ada_ln_aux=None, coords=None):
        if self.with_residual:
            x_in = x
        x = self.lnorm(x) if ada_ln_aux is None else self.lnorm(x, ada_ln_aux)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x.shape[0], self.num_heads, x.shape[-1] // self.num_heads]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype)
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x).reshape(s)

        if self.with_2d_rope:
            if coords is None:
                raise ValueError("coords must be provided when with_2d_rope=True")
            qs, ks = rotary_pos_emb_2d(qs, ks, coords, unsqueeze_dim=1)

        # set dropout rate according to training/eval mode as required by flash_attn
        dropout_rate = self.dropout_rate if self.training else 0.0

        cum_x_lens = torch.cumsum(x_lens, 0, dtype=torch.int32)
        # ordering of tensors (seq, heads, embed) (which differs from torch's flash attention implt)
        outs = flash_attn_varlen_func(
            qs,
            ks,
            vs,
            cum_x_lens,
            cum_x_lens,
            x_lens.max(),
            x_lens.max(),
            softcap=self.softcap,
            dropout_p=dropout_rate,
        )

        if self.use_xsa:
            outs = _apply_xsa(outs, vs)

        out = self.proj_out(outs.flatten(-2, -1))

        if self.with_residual:
            out = out + x_in

        return out


class MultiSelfAttentionHeadVarlenFlex(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
        use_xsa=False,
    ):
        super(MultiSelfAttentionHeadVarlenFlex, self).__init__()

        self.num_heads = num_heads
        self.with_flash = with_flash
        self.softcap = softcap
        self.with_residual = with_residual
        self.use_xsa = use_xsa

        assert dim_embed % num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        self.lnorm = norm(dim_embed, eps=norm_eps)
        self.proj_heads_q = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_v = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_out = torch.nn.Linear(dim_embed, dim_embed, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)
        self.dtype = attention_dtype

        assert with_flash, "Only flash attention supported at the moment"

        def att(qs, ks, vs, x_mask):
            def sparsity_mask(score, b, h, q_idx, kv_idx):
                return (q_idx // 16) == (kv_idx % 16)

            return flex_attention(qs, ks, vs, score_mod=sparsity_mask)

        self.compiled_flex_attention = torch.compile(att, dynamic=False)

    def forward(self, x, x_lens=None):
        if self.with_residual:
            x_in = x
        x = self.lnorm(x)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x.shape[0], 1, self.num_heads, -1]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype).permute([1, 2, 0, 3])
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype).permute([1, 2, 0, 3])
        vs = self.proj_heads_v(x).reshape(s).permute([1, 2, 0, 3])

        outs = self.compiled_flex_attention(qs, ks, vs).transpose(1, 2).squeeze()

        if self.use_xsa:
            outs = _apply_xsa(outs, vs.transpose(1, 2).squeeze())

        out = self.dropout(self.proj_out(outs.flatten(-2, -1)))
        if self.with_residual:
            out = out + x_in

        return out


class MultiSelfAttentionHeadLocal(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        num_heads,
        qkv_len,
        block_factor,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
        use_xsa=False,
        with_2d_rope=False,
        is_dit=False,
        dit_is_cond=False,
    ):
        super(MultiSelfAttentionHeadLocal, self).__init__()

        self.num_heads = num_heads
        self.with_flash = with_flash
        self.softcap = softcap
        self.with_residual = with_residual
        self.use_xsa = use_xsa
        self.with_2d_rope = with_2d_rope
        self.dtype = attention_dtype
        self.use_xsa = use_xsa

        assert dim_embed % num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        self.is_dit = is_dit
        self.dit_is_cond = dit_is_cond
        if is_dit:
            if dit_is_cond:
                assert dim_aux is not None, "For DIT, need to provide dim_aux for ada layer norm"
            assert dim_aux is None, "conditioning not yet implemented for DIT attention"
            assert with_residual, "DIT attention should always have residual connection"
            self.lnorm = (
                AdaLNZero(dim_embed, dim_aux, norm_eps=norm_eps)
                if dim_aux is not None
                else norm(dim_embed, eps=norm_eps)
            )
            self.noise_conditioning = LinearNormConditioning(
                latent_space_dim=dim_embed, dtype=attention_dtype
            )
        elif dim_aux is not None:
            self.lnorm = AdaLayerNorm(dim_embed, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm = norm(dim_embed, eps=norm_eps)

        self.proj_heads_q = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_v = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_out = torch.nn.Linear(dim_embed, dim_embed, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        assert with_flash, "Only flash attention supported."

        # define block mask
        def mask_block_local(batch, head, idx_q, idx_kv):
            return (idx_q // block_factor) == (idx_kv // block_factor)

        self.block_mask = create_block_mask(
            mask_block_local, B=None, H=None, Q_LEN=qkv_len, KV_LEN=qkv_len
        )
        # compile for efficiency
        self.flex_attention = torch.compile(flex_attention, dynamic=False)

    def forward(self, x, coords=None, emb=None, ada_ln_aux=None):
        if self.with_residual:
            x_in = x

        # Handle ada_ln_aux conditioning
        if self.is_dit:
            if self.dit_is_cond:
                x, cond_gate = self.lnorm(x, ada_ln_aux)
            else:
                x = self.lnorm(x)
                cond_gate = 1
            x, noise_gate = self.noise_conditioning(x, emb)
            gate = cond_gate * noise_gate
        else:
            x = self.lnorm(x, ada_ln_aux) if ada_ln_aux is not None else self.lnorm(x)

        # project onto heads
        s = [x.shape[0], x.shape[1], self.num_heads, -1]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype).permute([0, 2, 1, 3])
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype).permute([0, 2, 1, 3])
        vs = self.proj_heads_v(x).reshape(s).permute([0, 2, 1, 3])

        if self.with_2d_rope:
            if coords is None:
                raise ValueError("coords must be provided when with_2d_rope=True")
            qs, ks = rotary_pos_emb_2d(qs, ks, coords, unsqueeze_dim=1)

        outs = self.flex_attention(qs, ks, vs, block_mask=self.block_mask).transpose(1, 2)

        if self.use_xsa:
            outs = _apply_xsa(outs, vs.transpose(1, 2))

        out = self.proj_out(self.dropout(outs.flatten(-2, -1)))

        if self.with_residual:
            out = x_in + out * gate if self.is_dit else x_in + out

        return out


class MultiCrossAttentionHeadVarlen(torch.nn.Module):
    def __init__(
        self,
        dim_embed_q,
        dim_embed_kv,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        super(MultiCrossAttentionHeadVarlen, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.with_residual = with_residual
        self.with_flash = with_flash
        self.softcap = softcap

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

        if dim_aux is not None:
            self.lnorm_in_q = AdaLayerNorm(dim_embed_q, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm_in_q = norm(dim_embed_q, eps=norm_eps)
        self.lnorm_in_kv = norm(dim_embed_kv, eps=norm_eps)

        self.proj_heads_q = torch.nn.Linear(dim_embed_q, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(
            dim_embed_kv, num_heads * self.dim_head_proj, bias=False
        )
        self.proj_heads_v = torch.nn.Linear(
            dim_embed_kv, num_heads * self.dim_head_proj, bias=False
        )

        self.proj_out = torch.nn.Linear(self.dim_head_proj * num_heads, dim_embed_q, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        assert with_flash, "Only flash attention supported at the moment"

    def forward(self, x_q, x_kv, x_q_lens=None, x_kv_lens=None, ada_ln_aux=None):
        if self.with_residual:
            x_q_in = x_q
        x_q = self.lnorm_in_q(x_q) if ada_ln_aux is None else self.lnorm_in_q(x_q, ada_ln_aux)
        x_kv = self.lnorm_in_kv(x_kv)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x_q.shape[0], self.num_heads, self.dim_head_proj]
        qs = self.lnorm_q(self.proj_heads_q(x_q).reshape(s)).to(self.dtype)
        s = [x_kv.shape[0], self.num_heads, self.dim_head_proj]
        ks = self.lnorm_k(self.proj_heads_k(x_kv).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x_kv).reshape(s)

        # set dropout rate according to training/eval mode as required by flash_attn
        dropout_rate = self.dropout_rate if self.training else 0.0

        if x_kv_lens is not None:
            cum_x_q_lens = torch.cumsum(x_q_lens, 0, dtype=torch.int32)
            cum_x_kv_lens = torch.cumsum(x_kv_lens, 0, dtype=torch.int32)
            outs = flash_attn_varlen_func(
                qs,
                ks,
                vs,
                cum_x_q_lens,
                cum_x_kv_lens,
                x_q_lens.max(),
                x_kv_lens.max(),
                softcap=self.softcap,
                dropout_p=dropout_rate,
            )
        else:
            assert False

        outs = self.proj_out(outs.flatten(-2, -1))
        if self.with_residual:
            outs = x_q_in + outs

        return outs


class MultiCrossAttentionHeadVarlenSlicedQ(torch.nn.Module):
    def __init__(
        self,
        dim_embed_q,
        dim_embed_kv,
        num_slices_q,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        super(MultiCrossAttentionHeadVarlenSlicedQ, self).__init__()

        self.num_slices_q = num_slices_q
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.with_residual = with_residual
        self.with_flash = with_flash
        self.softcap = softcap

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

        if dim_aux is not None:
            self.lnorm_in_q = AdaLayerNorm(dim_embed_q, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm_in_q = norm(dim_embed_q, eps=norm_eps)
        self.lnorm_in_kv = norm(dim_embed_kv, eps=norm_eps)

        assert num_heads % num_slices_q == 0
        num_heads_r = num_heads
        self.proj_heads_q = torch.nn.ModuleList()
        for _ in range(num_slices_q):
            self.proj_heads_q.append(
                torch.nn.Linear(dim_embed_q, num_heads_r * self.dim_head_proj, bias=False)
            )
        self.proj_heads_k = torch.nn.Linear(
            dim_embed_kv, num_heads_r * self.dim_head_proj, bias=False
        )
        self.proj_heads_v = torch.nn.Linear(
            dim_embed_kv, num_heads_r * self.dim_head_proj, bias=False
        )

        self.proj_out = torch.nn.Linear(self.dim_head_proj * num_heads, dim_embed_q, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        assert with_flash, "Only flash attention supported at the moment"

    def forward(self, x_q, x_kv, x_q_lens=None, x_kv_lens=None, ada_ln_aux=None):
        if self.with_residual:
            x_q_in = x_q
        x_q = self.lnorm_in_q(x_q) if ada_ln_aux is None else self.lnorm_in_q(x_q, ada_ln_aux)
        x_kv = self.lnorm_in_kv(x_kv)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x_q.shape[0], self.num_heads, self.dim_head_proj]
        qs = [
            self.lnorm_q(head_proj(x_q_i).reshape(s)).to(self.dtype)
            for head_proj, x_q_i in zip(self.proj_heads_q, x_q.transpose(1, 0), strict=False)
        ]
        s = [x_kv.shape[0], self.num_heads, self.dim_head_proj]
        ks = self.lnorm_k(self.proj_heads_k(x_kv).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x_kv).reshape(s)

        # set dropout rate according to training/eval mode as required by flash_attn
        dropout_rate = self.dropout_rate if self.training else 0.0

        cum_x_q_lens = torch.cumsum(x_q_lens, 0, dtype=torch.int32)
        cum_x_kv_lens = torch.cumsum(x_kv_lens, 0, dtype=torch.int32)
        outs = []
        for _i, qs_i in enumerate(qs):
            outs += [
                flash_attn_varlen_func(
                    qs_i,
                    ks,
                    vs,
                    cum_x_q_lens,
                    cum_x_kv_lens,
                    x_q_lens.max(),
                    x_kv_lens.max(),
                    softcap=self.softcap,
                    dropout_p=dropout_rate,
                )
            ]

        outs = self.proj_out(torch.stack(outs).transpose(1, 0).flatten(-2, -1))
        if self.with_residual:
            outs = x_q_in + outs.reshape(x_q_in.shape)

        return outs


class MultiSelfAttentionHead(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        softcap=0.0,
        norm_type="LayerNorm",
        qk_norm_type=None,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
        use_xsa=False,
        with_2d_rope=False,
        is_dit=False,  # should only be True for diffusion model
        dit_is_cond=False,  # whether the attention is used for conditioning in the diffusion model (as opposed to denoising). Should only be True for cross attention layers in the diffusion model, and will control whether ada_ln_aux is applied to the input or output of the attention layer
    ):
        super(MultiSelfAttentionHead, self).__init__()

        self.num_heads = num_heads
        self.with_flash = with_flash
        self.softcap = softcap
        self.dropout_rate = dropout_rate
        self.with_residual = with_residual
        self.use_xsa = use_xsa
        self.with_2d_rope = with_2d_rope
        self.use_xsa = use_xsa

        assert dim_embed % num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        self.is_dit = is_dit
        self.dit_is_cond = dit_is_cond

        if is_dit:
            if dit_is_cond:
                assert dim_aux is not None, "For DIT, need to provide dim_aux for ada layer norm"
            assert with_residual, "DIT attention should always have residual connection"
            self.lnorm = (
                AdaLNZero(dim_embed, dim_aux, norm_eps=norm_eps)
                if dim_aux is not None
                else norm(dim_embed, eps=norm_eps)
            )
            self.noise_conditioning = LinearNormConditioning(
                latent_space_dim=dim_embed, dtype=attention_dtype
            )  # TODO: Do I need to pass dtype?
        elif dim_aux is not None:
            self.lnorm = AdaLayerNorm(dim_embed, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm = norm(dim_embed, eps=norm_eps)

        self.proj_heads_q = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_v = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_out = torch.nn.Linear(dim_embed, dim_embed, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        if with_flash:
            self.att = torch.nn.functional.scaled_dot_product_attention
        else:
            self.att = self.attention
            self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, coords=None, emb=None, ada_ln_aux=None):
        if self.with_residual:
            x_in = x

        # Handle ada_ln_aux conditioning
        if self.is_dit:
            if self.dit_is_cond:
                x, cond_gate = self.lnorm(x, ada_ln_aux)
            else:
                x = self.lnorm(x)
                cond_gate = 1
            x, noise_gate = self.noise_conditioning(x, emb)
            gate = cond_gate * noise_gate
        else:
            x = self.lnorm(x, ada_ln_aux) if ada_ln_aux is not None else self.lnorm(x)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [*([x.shape[0], 1] if len(x.shape) == 2 else x.shape[:-1]), self.num_heads, -1]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype)
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x).reshape(s).to(self.dtype)

        if self.with_2d_rope:
            if coords is None:
                raise ValueError("coords must be provided when with_2d_rope=True")
            qs, ks = rotary_pos_emb_2d(qs, ks, coords, unsqueeze_dim=2)

        # set dropout rate according to training/eval mode as required by flash_attn
        dropout_rate = self.dropout_rate if self.training else 0.0

        # ordering of tensors (seq, heads, embed) (which differs from torch's flash attention implt)
        outs = flash_attn_func(qs, ks, vs, softcap=self.softcap, dropout_p=dropout_rate)

        if self.use_xsa:
            outs = _apply_xsa(outs, vs)

        out = self.proj_out(outs.flatten(-2, -1))

        if self.with_residual:
            out = x_in + out * gate if self.is_dit else out + x_in

        return out


class MultiCrossAttentionHead(torch.nn.Module):
    def __init__(
        self,
        dim_embed_q,
        dim_embed_kv,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
        is_dit=False,
    ):
        super(MultiCrossAttentionHead, self).__init__()

        self.num_heads = num_heads
        self.with_residual = with_residual
        self.with_flash = with_flash
        self.is_dit = is_dit

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        assert dim_embed_q % num_heads == 0
        self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

        if is_dit:
            assert with_residual
            self.lnorm_in_q = norm(dim_embed_q, eps=norm_eps)
            self.noise_conditioning = LinearNormConditioning(
                latent_space_dim=dim_embed_q, dtype=attention_dtype
            )
        else:
            self.lnorm_in_q = norm(dim_embed_q, eps=norm_eps)
        self.lnorm_in_kv = norm(dim_embed_kv, eps=norm_eps)

        self.proj_heads_q = torch.nn.Linear(dim_embed_q, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(
            dim_embed_kv, num_heads * self.dim_head_proj, bias=False
        )
        self.proj_heads_v = torch.nn.Linear(
            dim_embed_kv, num_heads * self.dim_head_proj, bias=False
        )

        self.proj_out = torch.nn.Linear(self.dim_head_proj * num_heads, dim_embed_q, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        self.att = torch.nn.functional.scaled_dot_product_attention
        self.softmax = torch.nn.Softmax(dim=-1)

    #########################################
    def forward(self, x_q, x_kv, emb=None):
        if self.with_residual:
            x_q_in = x_q

        if self.is_dit:
            x_q = self.lnorm_in_q(x_q)
            x_q, gate = self.noise_conditioning(x_q, emb)
        else:
            x_q = self.lnorm_in_q(x_q)
        x_kv = self.lnorm_in_kv(x_kv)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x_q.shape[0], -1, self.num_heads, self.dim_head_proj]
        qs = self.lnorm_q(self.proj_heads_q(x_q).reshape(s)).to(self.dtype).transpose(-3, -2)
        s = [x_kv.shape[0], -1, self.num_heads, self.dim_head_proj]
        ks = self.lnorm_k(self.proj_heads_k(x_kv).reshape(s)).to(self.dtype).transpose(-3, -2)
        vs = self.proj_heads_v(x_kv).reshape(s).transpose(-3, -2)

        # correct ordering of tensors with seq dimension second but last is critical
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            outs = self.att(qs, ks, vs).transpose(2, 1)

        outs = self.dropout(self.proj_out(outs.flatten(-2, -1)))
        if self.with_residual:
            outs = x_q_in + outs * gate if self.is_dit else x_q_in + outs

        return outs
