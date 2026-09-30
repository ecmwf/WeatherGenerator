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
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from abc import ABC, abstractmethod
from dataclasses import dataclass

from weathergen.model.norms import AdaLayerNorm, RMSNorm
from weathergen.model.positional_encoding import rotary_pos_emb_2d

"""
Attention blocks used by WeatherGenerator.

Some blocks optionally apply 2D RoPE. When enabled, the caller must provide per-token 2D
coordinates aligned with the token order (lat, lon in radians).
"""


@dataclass
class SeqLens:
    """Packed-sequence metadata for varlen attention (computed once per forward)."""
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    max_q: int
    max_kv: int

    @classmethod
    def from_lens(cls, q_lens, kv_lens=None) -> "SeqLens":
        kv_lens = kv_lens if kv_lens is not None else q_lens
        
        # flash_attn_varlen requires cu_seqlens to have a leading 0.
        # F.pad adds a 0 at the beginning of the cumsum tensor.
        cu_q = F.pad(torch.cumsum(q_lens, 0, dtype=torch.int32), (1, 0))
        cu_kv = F.pad(torch.cumsum(kv_lens, 0, dtype=torch.int32), (1, 0))
        
        return cls(
            cu_q=cu_q,
            cu_kv=cu_kv,
            max_q=q_lens.max().item(),
            max_kv=kv_lens.max().item(),
        )

    
class AttentionKernel(torch.nn.Module, ABC):
    """Abstract base class for attention kernels."""
    
    @abstractmethod
    def __call__(
        self,
        qs,
        ks,
        vs,
        x_q_lens=None,
        x_kv_lens=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        softcap=0.0,
        dropout_p=0.0
    ):
        raise NotImplementedError("Attention kernels must implement the __call__ method.")


class FlashKernel(AttentionKernel):
    def __init__(self):
        pass

    def __call__(
        self,
        q,
        k,
        v,
        seqlens: SeqLens | None = None,
        softcap=0.0,
        dropout_p=0.0,
    ):
        """Wrapper for FlashAttention (batched or varlen)."""
        if seqlens is not None:
            return flash_attn_varlen_func(
                q,
                k,
                v,
                seqlens.cu_q,
                seqlens.cu_kv,
                seqlens.max_q,
                seqlens.max_kv,
                softcap=softcap,
                dropout_p=dropout_p,
            )

        return flash_attn_func(q, k, v, softcap=softcap, dropout_p=dropout_p)


class SDPAKernel(AttentionKernel):
    def __init__(self):
        pass

    def __call__(
        self,
        qs,
        ks,
        vs,
        seqlens: SeqLens | None = None,
        softcap=0.0,
        dropout_p=0.0,
    ):
        """Wrapper for scaled_dot_product_attention."""
        if seqlens is not None:
            raise NotImplementedError("SDPA does not support packed sequences.")

        if softcap != 0.0:
            raise NotImplementedError("SDPA does not support softcap.")

        qs = qs.transpose(1, 2)
        ks = ks.transpose(1, 2)
        vs = vs.transpose(1, 2)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            outs = torch.nn.functional.scaled_dot_product_attention(
                qs,
                ks,
                vs,
                dropout_p=dropout_p
            )

        return outs.transpose(1, 2)


class BaseAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        super(BaseAttention, self).__init__()

        # values assigned by _make_qk_lnorms() in each subclass __init__
        self.lnorm_q = None
        self.lnorm_k = None

        # values assigned by _make_proj_heads() in subclasses that use standard projections
        self.proj_heads_q = None
        self.proj_heads_k = None
        self.proj_heads_v = None
        self.proj_out = None

        self.num_heads = num_heads
        self.with_residual = with_residual
        self.norm_eps = norm_eps
        self.with_qk_lnorm = with_qk_lnorm
        self.dtype = attention_dtype
        self.dropout_rate = dropout_rate
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        if norm_type == "LayerNorm":
            self.norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=self.norm_eps)
        else:
            self.norm = partial(RMSNorm, eps=self.norm_eps)

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            self.qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=self.norm_eps)
        else:
            self.qk_norm = partial(RMSNorm, eps=self.norm_eps)

    def _make_qk_lnorms(self):
        if self.with_qk_lnorm:
            lnorm = self.qk_norm
            self.lnorm_q = lnorm(self.dim_head_proj)
            self.lnorm_k = lnorm(self.dim_head_proj)
        else:
            self.lnorm_q = torch.nn.Identity()
            self.lnorm_k = torch.nn.Identity()

    def _make_proj_heads(self, dim_embed, dim_embed_kv=None):
        dim_embed_kv = dim_embed_kv if dim_embed_kv else dim_embed

        self.proj_heads_q = torch.nn.Linear(
            dim_embed, self.num_heads * self.dim_head_proj, bias=False
        )
        self.proj_heads_k = torch.nn.Linear(
            dim_embed_kv, self.num_heads * self.dim_head_proj, bias=False
        )
        self.proj_heads_v = torch.nn.Linear(
            dim_embed_kv, self.num_heads * self.dim_head_proj, bias=False
        )
        self.proj_out = torch.nn.Linear(self.num_heads * self.dim_head_proj, dim_embed, bias=False)


# MultiSelfAttentionHeadVarlen, MultiCrossAttentionHeadVarlen,
# MultiSelfAttentionHead, MultiCrossAttentionHead
class Attention(BaseAttention):
    def __init__(
        self,
        dim_embed,
        num_heads,
        dim_embed_kv=None,
        dim_head_proj=None,
        softcap=0.0,
        dim_aux=None,
        with_2d_rope=False,
        kernel=None,
        **kwargs,
    ):
        super(Attention, self).__init__(num_heads=num_heads, dim_head_proj=dim_head_proj, **kwargs)

        self.softcap = softcap
        self.with_2d_rope = with_2d_rope

        assert dim_embed % self.num_heads == 0
        self.dim_head_proj = dim_embed // self.num_heads if dim_head_proj is None else dim_head_proj

        self._make_qk_lnorms()

        if dim_aux is not None:
            self.lnorm = AdaLayerNorm(dim_embed, dim_aux, norm_eps=self.norm_eps)
        else:
            self.lnorm = self.norm(dim_embed)

        self.lnorm_in_kv = self.norm(dim_embed_kv) if dim_embed_kv is not None else None

        self._make_proj_heads(dim_embed, dim_embed_kv)

        self.att = kernel if kernel is not None else FlashKernel()

    def forward(self, x, x_kv=None, seqlens: SeqLens | None = None, ada_ln_aux=None, coords=None):
        if self.with_residual:
            x_in = x
        x = self.lnorm(x) if ada_ln_aux is None else self.lnorm(x, ada_ln_aux)
        x_kv = self.lnorm_in_kv(x_kv) if x_kv is not None else x

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        if seqlens is not None:
            s_q = [x.shape[0], self.num_heads, self.dim_head_proj]
            s_kv = [x_kv.shape[0], self.num_heads, self.dim_head_proj]
        else:
            s_q = [*([x.shape[0], 1] if len(x.shape) == 2 else x.shape[:-1]), self.num_heads, -1]
            s_kv = [
                *([x_kv.shape[0], 1] if len(x_kv.shape) == 2 else x_kv.shape[:-1]),
                self.num_heads,
                -1,
            ]

        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s_q)).to(self.dtype)
        ks = self.lnorm_k(self.proj_heads_k(x_kv).reshape(s_kv)).to(self.dtype)
        vs = self.proj_heads_v(x_kv).reshape(s_kv).to(self.dtype)

        if self.with_2d_rope:
            if coords is None:
                raise ValueError("coords must be provided when with_2d_rope=True")
            unsqueeze_dim = 1 if x_lens is not None else 2
            qs, ks = rotary_pos_emb_2d(qs, ks, coords, unsqueeze_dim=unsqueeze_dim)

        # set dropout rate according to training/eval mode as required by flash_attn
        dropout_rate = self.dropout_rate if self.training else 0.0

        outs = self.att(qs, ks, vs, seqlens=seqlens, softcap=self.softcap, dropout_p=dropout_rate)

        out = self.proj_out(outs.flatten(-2, -1))

        if self.with_residual:
            out += x_in

        return out


class MultiSelfAttentionHeadVarlenFlex(BaseAttention):
    def __init__(
        self,
        dim_embed,
        num_heads,
        dim_head_proj=None,
        softcap=0.0,
        **kwargs,
    ):
        super(MultiSelfAttentionHeadVarlenFlex, self).__init__(
            num_heads=num_heads, dim_head_proj=dim_head_proj, **kwargs
        )

        self.softcap = softcap

        assert dim_embed % self.num_heads == 0
        self.dim_head_proj = dim_embed // self.num_heads if dim_head_proj is None else dim_head_proj

        self._make_qk_lnorms()

        self.lnorm = self.norm(dim_embed)

        self._make_proj_heads(dim_embed)

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

        out = self.dropout(self.proj_out(outs.flatten(-2, -1)))
        if self.with_residual:
            out += x_in

        return out


class MultiSelfAttentionHeadLocal(BaseAttention):
    def __init__(
        self,
        dim_embed,
        num_heads,
        qkv_len,
        block_factor,
        dim_head_proj=None,
        softcap=0.0,
        dim_aux=None,
        with_2d_rope=False,
        **kwargs,
    ):
        super(MultiSelfAttentionHeadLocal, self).__init__(
            num_heads=num_heads, dim_head_proj=dim_head_proj, **kwargs
        )

        self.softcap = softcap
        self.with_2d_rope = with_2d_rope

        assert dim_embed % self.num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        self._make_qk_lnorms()

        if dim_aux is not None:
            self.lnorm = AdaLayerNorm(dim_embed, dim_aux, norm_eps=self.norm_eps)
        else:
            self.lnorm = self.norm(dim_embed)

        self._make_proj_heads(dim_embed)

        # define block mask
        def mask_block_local(batch, head, idx_q, idx_kv):
            return (idx_q // block_factor) == (idx_kv // block_factor)

        self.block_mask = create_block_mask(
            mask_block_local, B=None, H=None, Q_LEN=qkv_len, KV_LEN=qkv_len
        )
        # compile for efficiency
        self.flex_attention = torch.compile(flex_attention, dynamic=False)

    def forward(self, x, coords=None, ada_ln_aux=None):
        if self.with_residual:
            x_in = x
        x = self.lnorm(x) if ada_ln_aux is None else self.lnorm(x, ada_ln_aux)

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

        out = self.proj_out(self.dropout(outs.flatten(-2, -1)))
        if self.with_residual:
            out += x_in

        return out


class MultiCrossAttentionHeadVarlenSlicedQ(BaseAttention):
    def __init__(
        self,
        dim_embed_q,
        dim_embed_kv,
        num_heads,
        num_slices_q,
        dim_head_proj=None,
        softcap=0.0,
        dim_aux=None,
        **kwargs,
    ):
        super(MultiCrossAttentionHeadVarlenSlicedQ, self).__init__(
            num_heads=num_heads, dim_head_proj=dim_head_proj, **kwargs
        )

        self.num_slices_q = num_slices_q
        self.softcap = softcap

        self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

        self._make_qk_lnorms()

        if dim_aux is not None:
            self.lnorm_in_q = AdaLayerNorm(dim_embed_q, dim_aux, norm_eps=self.norm_eps)
        else:
            self.lnorm_in_q = self.norm(dim_embed_q)
        self.lnorm_in_kv = self.norm(dim_embed_kv)

        assert self.num_heads % num_slices_q == 0
        num_heads_r = self.num_heads
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
