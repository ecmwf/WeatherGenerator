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
# 2D Rotary Position Embedding
# https://github.com/naver-ai/rope-vit/blob/main/models/vit_rope.py
####################################################################################################
def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
                    
    return freqs_cis


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    # for Query Aggregation engine fallback
    else:
        # fallback for cases where rotary frequencies already include head dim
        shape = [1] * ndim
        shape[-freqs_cis.ndim :] = freqs_cis.shape
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
