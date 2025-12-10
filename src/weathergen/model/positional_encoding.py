# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import numpy as np
import torch


####################################################################################################
def positional_encoding_harmonic(x):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device
    dtype = x.dtype

    len_token_seq = x.shape[-2]
    pe = torch.zeros(len_token_seq, dim_embed, device=dev, dtype=dtype)
    position = torch.arange(0, len_token_seq, device=dev, dtype=dtype).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim_embed, 2, device=dev, dtype=dtype) * -(math.log(10000) / dim_embed)
    )

    pe[:, 0::2] = torch.sin(position * div[: pe[:, 0::2].shape[1]])
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    x = x + pe

    return x


####################################################################################################
def positional_encoding_harmonic_idx(x, s_idx):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device

    len_token_seq = x.shape[0]
    pe = torch.zeros(x.shape[-2:], device=dev)
    pos = (s_idx + 1) * torch.ones(len_token_seq, device=dev)
    xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=dev) / dim_embed

    pe[:, 0::2] = torch.sin(torch.outer(pos, xs))
    pe[:, 1::2] = torch.cos(torch.outer(pos, xs))
    x = x + pe

    return x


####################################################################################################
def positional_encoding_harmonic_global(x):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device

    pe = torch.zeros(x.shape[-3], x.shape[-2], dim_embed, device=dev)
    xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=dev) / dim_embed
    pe[..., 0::2] = 0.5 * torch.sin(torch.outer(8 * torch.arange(x.shape[-2], device=dev), xs))
    pe[..., 0::2] += (
        torch.sin(torch.outer(torch.arange(x.shape[-3], device=dev), xs))
        .unsqueeze(1)
        .repeat((1, x.shape[-2], 1))
    )
    pe[..., 1::2] = 0.5 * torch.cos(torch.outer(8 * torch.arange(x.shape[-2], device=dev), xs))
    pe[..., 1::2] += (
        torch.cos(torch.outer(torch.arange(x.shape[-3], device=dev), xs))
        .unsqueeze(1)
        .repeat((1, x.shape[-2], 1))
    )
    x = x + pe

    return x


####################################################################################################
def positional_encoding_harmonic_coord(x, lats, lons):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device

    pe = torch.zeros(x.shape[0], dim_embed, device=dev)
    xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=dev) / dim_embed
    pe[..., 0::2] = 0.5 * torch.sin(torch.outer(lats, xs))
    pe[..., 1::2] = 0.5 * torch.cos(torch.outer(lons, xs))[..., : pe[..., 1::2].shape[-1]]
    x = x + pe

    return x


####################################################################################################
# Rotary positional embeddings (2D) adapted from Qwen3 & LLama for reuse in WeatherGenerator.
# rotate_half () and apply_rotary_pos_emb () from:
# https://github.com/qiuzh20/gated_attention/blob/main/modeling_qwen3.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine embedding tensor.
        sin: Sine embedding tensor.
        position_ids: Deprecated and unused; present for API compatibility.
        unsqueeze_dim: Dimension along which to unsqueeze cos/sin for broadcasting.
    """

    cos = cos.unsqueeze(unsqueeze_dim).to(dtype=q.dtype)
    sin = sin.unsqueeze(unsqueeze_dim).to(dtype=q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


####################################################################################################
def rotary_embedding_2d(coords, dim, base=10000.0):
    """Create 2D RoPE embeddings from latitude/longitude coordinates.

    Args:
        coords: Tensor of shape (..., 2) with coordinates in radians (lat, lon).
        dim: Head dimension to encode; must be divisible by 4.
        base: RoPE base frequency.

    Returns:
        Tuple of (cos, sin) tensors with shape (..., dim).
    """

    if coords.shape[-1] != 2:
        raise ValueError(f"coords last dimension must be 2 (lat, lon); got {coords.shape[-1]}")
    if dim % 4 != 0:
        raise ValueError(f"2D rotary embeddings require dim to be divisible by 4; got {dim}")

    # Split the rotary frequencies evenly between latitude and longitude to stay local to each cell.
    half_dim = dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half_dim, 2, device=coords.device, dtype=coords.dtype) / half_dim)
    )

    lat, lon = coords.unbind(dim=-1)
    freq_lat = lat.unsqueeze(-1) * inv_freq
    freq_lon = lon.unsqueeze(-1) * inv_freq

    freqs = torch.cat((freq_lat, freq_lon), dim=-1)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = torch.cos(emb)
    sin = torch.sin(emb)

    return cos, sin


####################################################################################################
def apply_rotary_pos_emb_2d(q, k, coords, base=10000.0, unsqueeze_dim=1):
    """Convenience wrapper that builds 2D RoPE embeddings and applies them to q/k."""

    cos, sin = rotary_embedding_2d(coords, q.shape[-1], base=base)
    return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
