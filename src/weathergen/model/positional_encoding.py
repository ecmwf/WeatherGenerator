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
import torch.nn


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


####################################################################################################
def positional_encoding_harmonic(x):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device

    len_token_seq = x.shape[-2]
    pe = torch.zeros(len_token_seq, dim_embed, device=dev)
    position = torch.arange(0, len_token_seq).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim_embed, 2) * -(math.log(10000) / dim_embed))

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
