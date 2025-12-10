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

from weathergen.model.norms import AdaLayerNorm, RMSNorm


class MultiSelfAttentionHeadVarlen(torch.nn.Module):
    """Multi-head self-attention with variable length sequences.
    
    This module implements multi-head self-attention for variable length sequences packed into a
    single tensor. It leverages FlashAttention's variable length API (`flash_attn_varlen_func`)
    to efficiently handle batches of sequences with differing lengths without padding, using
    cumulative length indices to define sequence boundaries.
    """
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
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        """Initialize the MultiSelfAttentionHeadVarlen module.
        
        :param dim_embed: Embedding dimension.
        :param num_heads: Number of attention heads.
        :param dim_head_proj: Dimension of the projection head.
        :param dropout_rate: Dropout rate.
        :param with_residual: Whether to use residual connections.
        :param with_qk_lnorm: Whether to use layer normalization for query and key.
        :param with_flash: Whether to use flash attention.
        :param norm_type: Type of normalization.
        :param softcap: Softcap for attention.
        :param dim_aux: Dimension of auxiliary data.
        :param norm_eps: Epsilon for normalization.
        :param attention_dtype: Data type for attention.
        """
        super(MultiSelfAttentionHeadVarlen, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.with_flash = with_flash
        self.softcap = softcap
        self.with_residual = with_residual

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

        lnorm = norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype

        assert with_flash, "Only flash attention supported at the moment"

    def forward(self, x, x_lens, ada_ln_aux=None):
        """Forward pass of the MultiSelfAttentionHeadVarlen module.
        
        :param x: Input tensor.
        :param x_lens: Lengths of the input sequences.
        :param ada_ln_aux: Auxiliary data for adaptive layer normalization.

        :return out: Output tensor.
        """
        if self.with_residual:
            x_in = x
        x = self.lnorm(x) if ada_ln_aux is None else self.lnorm(x, ada_ln_aux)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x.shape[0], self.num_heads, -1]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype)
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x).reshape(s)

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

        out = self.proj_out(outs.flatten(-2, -1))

        if self.with_residual:
            out = out + x_in

        return out


class MultiSelfAttentionHeadVarlenFlex(torch.nn.Module):
    """Multi-head self-attention with variable length sequences and flex attention.
    
    This module implements multi-head self-attention using PyTorch's FlexAttention. It allows
    for defining custom sparse attention patterns via a score modification function. This is
    particularly useful for optimizing attention mechanisms where full NxN interactions are not
    required or desired, enabling flexible and efficient attention computations.
    """
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
        softcap=0.0,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        """Initialize the MultiSelfAttentionHeadVarlenFlex module.
        
        :param dim_embed: Embedding dimension.
        :param num_heads: Number of attention heads.
        :param dim_head_proj: Dimension of the projection head.
        :param dropout_rate: Dropout rate.
        :param with_residual: Whether to use residual connections.
        :param with_qk_lnorm: Whether to use layer normalization for query and key.
        :param with_flash: Whether to use flash attention.
        :param norm_type: Type of normalization.
        :param softcap: Softcap for attention.
        :param norm_eps: Epsilon for normalization.
        :param attention_dtype: Data type for attention.
        """
        super(MultiSelfAttentionHeadVarlenFlex, self).__init__()

        self.num_heads = num_heads
        self.with_flash = with_flash
        self.softcap = softcap
        self.with_residual = with_residual

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

        lnorm = norm if with_qk_lnorm else torch.nn.Identity
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
        """Forward pass of the MultiSelfAttentionHeadVarlenFlex module.
        
        :param x: Input tensor.
        :param x_lens: Lengths of the input sequences.

        :return out: Output tensor.
        """
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
            out = out + x_in

        return out


class MultiSelfAttentionHeadLocal(torch.nn.Module):
    """Multi-head self-attention with local (block-wise) attention.

    This module implements local (block-wise) multi-head self-attention. It restricts attention
    to local blocks defined by `block_factor`, meaning tokens only attend to other tokens within
    the same block. This effectively reduces the computational complexity from quadratic to
    linear with respect to the sequence length (for a fixed block size), making it suitable for
    processing long sequences where local interactions dominate."""
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
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        """Initialize the MultiSelfAttentionHeadLocal module.
        
        :param dim_embed: Embedding dimension.
        :param num_heads: Number of attention heads.
        :param qkv_len: Length of the query, key, and value.
        :param block_factor: Block factor.
        :param dim_head_proj: Dimension of the projection head.
        :param dropout_rate: Dropout rate.
        :param with_residual: Whether to use residual connections.
        :param with_qk_lnorm: Whether to use layer normalization for query and key.
        :param with_flash: Whether to use flash attention.
        :param norm_type: Type of normalization.
        :param softcap: Softcap for attention.
        :param dim_aux: Dimension of the auxiliary data.
        :param norm_eps: Epsilon for normalization.
        :param attention_dtype: Data type for attention.
        """
        super(MultiSelfAttentionHeadLocal, self).__init__()

        self.num_heads = num_heads
        self.with_flash = with_flash
        self.softcap = softcap
        self.with_residual = with_residual

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

        lnorm = norm if with_qk_lnorm else torch.nn.Identity
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

    def forward(self, x, ada_ln_aux=None):
        """Forward pass of the MultiSelfAttentionHeadLocal module.
        
        :param x: Input tensor.
        :param ada_ln_aux: Auxiliary data for adaptive layer normalization.

        :return out: Output tensor.
        """
        if self.with_residual:
            x_in = x
        x = self.lnorm(x) if ada_ln_aux is None else self.lnorm(x, ada_ln_aux)

        # project onto heads
        s = [x.shape[0], x.shape[1], self.num_heads, -1]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype).permute([0, 2, 1, 3])
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype).permute([0, 2, 1, 3])
        vs = self.proj_heads_v(x).reshape(s).permute([0, 2, 1, 3])

        outs = self.flex_attention(qs, ks, vs, block_mask=self.block_mask).transpose(1, 2)

        out = self.proj_out(self.dropout(outs.flatten(-2, -1)))
        if self.with_residual:
            out = x_in + out

        return out


class MultiCrossAttentionHeadVarlen(torch.nn.Module):
    """Multi-head cross-attention with variable length sequences.
    
    This module implements multi-head cross-attention for variable length sequences. Similar to
    the self-attention variant, it uses FlashAttention (`flash_attn_varlen_func`) to handle
    packed sequences of queries and keys/values with different lengths. It ensures correct masking
    and efficient computation for cases where both source and target sequences vary in length.
    """
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
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        """Initialize the MultiCrossAttentionHeadVarlen module.
        
        :param dim_embed_q: Embedding dimension of the query.
        :param dim_embed_kv: Embedding dimension of the key and value.
        :param num_heads: Number of attention heads.
        :param dim_head_proj: Dimension of the projection head.
        :param dropout_rate: Dropout rate.
        :param with_residual: Whether to use residual connections.
        :param with_qk_lnorm: Whether to use layer normalization for query and key.
        :param with_flash: Whether to use flash attention.
        :param norm_type: Type of normalization.
        :param softcap: Softcap for attention.
        :param dim_aux: Dimension of the auxiliary data.
        :param norm_eps: Epsilon for normalization.
        :param attention_dtype: Data type for attention.
        """
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

        lnorm = norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        assert with_flash, "Only flash attention supported at the moment"

    def forward(self, x_q, x_kv, x_q_lens=None, x_kv_lens=None, ada_ln_aux=None):
        """Forward pass of the MultiCrossAttentionHeadVarlen module.
        
        :param x_q: Query tensor.
        :param x_kv: Key and value tensor.
        :param x_q_lens: Lengths of the query sequences.
        :param x_kv_lens: Lengths of the key and value sequences.
        :param ada_ln_aux: Auxiliary data for adaptive layer normalization.

        :return outs: Output tensors.
        """
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
    """Multi-head cross-attention with variable length sequences and sliced queries.
    
    This module implements a memory-efficient variant of multi-head cross-attention where the
    query projection is sliced into chunks. This allows processing extremely large query sets
    (e.g., global queries against local latents) by computing attention for subsets of queries
    sequentially. This approach reduces peak memory usage significantly, enabling the model to
    scale to higher resolutions or larger query counts."""
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
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        """Initialize the MultiCrossAttentionHeadVarlenSlicedQ module.
        
        :param dim_embed_q: Embedding dimension of the query.
        :param dim_embed_kv: Embedding dimension of the key and value.
        :param num_slices_q: Number of slices for the query.
        :param num_heads: Number of attention heads.
        :param dim_head_proj: Dimension of the projection head.
        :param dropout_rate: Dropout rate.
        :param with_residual: Whether to use residual connections.
        :param with_qk_lnorm: Whether to use layer normalization for query and key.
        :param with_flash: Whether to use flash attention.
        :param norm_type: Type of normalization.
        :param softcap: Softcap for attention.
        :param dim_aux: Dimension of the auxiliary data.
        :param norm_eps: Epsilon for normalization.
        :param attention_dtype: Data type for attention.
        """
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

        lnorm = norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        assert with_flash, "Only flash attention supported at the moment"

    def forward(self, x_q, x_kv, x_q_lens=None, x_kv_lens=None, ada_ln_aux=None):
        """Forward pass of the MultiCrossAttentionHeadVarlenSlicedQ module.
        
        :param x_q: Query tensor.
        :param x_kv: Key and value tensor.
        :param x_q_lens: Lengths of the query sequences.
        :param x_kv_lens: Lengths of the key and value sequences.
        :param ada_ln_aux: Auxiliary data for adaptive layer normalization.
        
        :return outs: Output tensors.
        """
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
    """Multi-head self-attention."""
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
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        """Initialize the MultiSelfAttentionHead module.
        
        :param dim_embed: Embedding dimension.
        :param num_heads: Number of attention heads.
        :param dim_head_proj: Dimension of the projection head.
        :param dropout_rate: Dropout rate.
        :param with_residual: Whether to use residual connections.
        :param with_qk_lnorm: Whether to use layer normalization for query and key.
        :param with_flash: Whether to use flash attention.
        :param softcap: Softcap for attention.
        :param norm_type: Type of normalization.
        :param dim_aux: Dimension of the auxiliary data.
        :param norm_eps: Epsilon for normalization.
        :param attention_dtype: Data type for attention.
        """
        super(MultiSelfAttentionHead, self).__init__()

        self.num_heads = num_heads
        self.with_flash = with_flash
        self.softcap = softcap
        self.dropout_rate = dropout_rate
        self.with_residual = with_residual

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

        lnorm = norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        if with_flash:
            self.att = torch.nn.functional.scaled_dot_product_attention
        else:
            self.att = self.attention
            self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, ada_ln_aux=None):
        """Forward pass of the MultiSelfAttentionHead module.
        
        :param x: Input tensor.
        :param ada_ln_aux: Auxiliary data for adaptive layer normalization.
        
        :return out: Output tensor.
        """
        if self.with_residual:
            x_in = x
        x = self.lnorm(x) if ada_ln_aux is None else self.lnorm(x, ada_ln_aux)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [*([x.shape[0], 1] if len(x.shape) == 2 else x.shape[:-1]), self.num_heads, -1]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype)
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x).reshape(s).to(self.dtype)

        # set dropout rate according to training/eval mode as required by flash_attn
        dropout_rate = self.dropout_rate if self.training else 0.0

        # ordering of tensors (seq, heads, embed) (which differs from torch's flash attention implt)
        outs = flash_attn_func(qs, ks, vs, softcap=self.softcap, dropout_p=dropout_rate)

        out = self.proj_out(outs.flatten(-2, -1))
        if self.with_residual:
            out = out + x_in

        return out


class MultiCrossAttentionHead(torch.nn.Module):
    """Multi-head cross-attention."""
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
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        """Initialize the MultiCrossAttentionHead module.
        
        :param dim_embed_q: Embedding dimension of the query.
        :param dim_embed_kv: Embedding dimension of the key and value.
        :param num_heads: Number of attention heads.
        :param dim_head_proj: Dimension of the projection head.
        :param dropout_rate: Dropout rate.
        :param with_residual: Whether to use residual connections.
        :param with_qk_lnorm: Whether to use layer normalization for query and key.
        :param with_flash: Whether to use flash attention.
        :param norm_type: Type of normalization.
        :param norm_eps: Epsilon for normalization.
        :param attention_dtype: Data type for attention.
        """
        super(MultiCrossAttentionHead, self).__init__()

        self.num_heads = num_heads
        self.with_residual = with_residual
        self.with_flash = with_flash

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        assert dim_embed_q % num_heads == 0
        self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

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

        lnorm = norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        self.att = torch.nn.functional.scaled_dot_product_attention
        self.softmax = torch.nn.Softmax(dim=-1)

    #########################################
    def forward(self, x_q, x_kv):
        """Forward pass of the MultiCrossAttentionHead module.
        
        :param x_q: Query tensor.
        :param x_kv: Key and value tensor.
        
        :return outs: Output tensors.
        """
        if self.with_residual:
            x_q_in = x_q
        x_q, x_kv = self.lnorm_in_q(x_q), self.lnorm_in_kv(x_kv)

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
            outs = x_q_in + outs

        return outs
