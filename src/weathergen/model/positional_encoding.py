# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import math
from functools import lru_cache

import healpy as hp
import numpy as np
import numpy.typing as npt
import torch

# Suppress verbose healpy transform messages during spherical RoPE coefficient precomputation.
logging.getLogger("healpy").setLevel(logging.WARNING)


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
# The functions rotate_half() and apply_rotary_pos_emb() below are derived from LLaMA and Qwen3
# models, originally developed by Meta Platforms, Inc., The Qwen team, Alibaba Group and the
# HuggingFace Inc. team, licensed under the Apache License, Version 2.0.
# Source: https://github.com/qiuzh20/gated_attention/blob/main/modeling_qwen3.py


def rotate_half(x):
    """Rotates half the hidden dims of the input."""

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine embedding tensor.
        sin: Sine embedding tensor.
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

    assert coords.shape[-1] == 2, (
        f"coords last dimension must be 2 (lat, lon); got {coords.shape[-1]}"
    )
    assert dim % 4 == 0, f"2D rotary embeddings require dim to be divisible by 4; got {dim}"

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
def rotary_pos_emb_2d(q, k, coords, base=10000.0, unsqueeze_dim=1):
    """Convenience wrapper that builds 2D RoPE embeddings and applies them to q/k."""

    cos, sin = rotary_embedding_2d(coords, q.shape[-1], base=base)
    return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

# Spherical RoPE
def _max_supported_spherical_band(dim_embed: int, num_heads: int) -> int:
    head_dim = dim_embed // num_heads
    max_complex = (head_dim - (head_dim % 2)) // 2
    return max(0, (max_complex - 1) // 2)


def get_rope_mode(cf, logger=None) -> str:
    """Resolve RoPE mode, including temporary backwards compatibility for rope_2D."""

    rope_mode = cf.get("rope_mode", "none") or "none"
    rope_2d = cf.get("rope_2D", None)
    if rope_2d is not None:
        if logger is not None:
            logger.warning(
                "Config key 'rope_2D' is deprecated and will be removed. Use 'rope_mode' "
                "with one of: none, 2d, spherical."
            )
        if rope_mode == "none":
            rope_mode = "2d" if rope_2d else "none"
    return rope_mode


def get_rope_spherical_band(cf) -> int:
    """Resolve spherical band index, supporting explicit config or automatic selection."""

    rope_spherical_band = cf.get("rope_spherical_band", None)
    if rope_spherical_band is not None:
        return int(rope_spherical_band)

    candidates = [
        _max_supported_spherical_band(cf.ae_global_dim_embed, cf.ae_aggregation_num_heads),
        _max_supported_spherical_band(cf.ae_global_dim_embed, cf.ae_global_num_heads),
    ]
    if cf.get("fe_num_blocks", 0) > 0:
        candidates.append(_max_supported_spherical_band(cf.ae_global_dim_embed, cf.fe_num_heads))
    return min(candidates)


def apply_rope(qs, ks, coords, rope_mode, unsqueeze_dim):
    rope_mode = rope_mode or "none"
    if rope_mode == "none":
        return qs, ks
    if coords is None:
        raise ValueError(f"coords must be provided when rope_mode={rope_mode}")
    if rope_mode == "2d":
        return rotary_pos_emb_2d(qs, ks, coords, unsqueeze_dim=unsqueeze_dim)
    if rope_mode == "spherical":
        return rotary_pos_emb_spherical(qs, ks, coords, unsqueeze_dim=unsqueeze_dim)
    raise ValueError(f"Unsupported rope_mode={rope_mode}")


def rotary_pos_emb_spherical(
    q: torch.Tensor,
    k: torch.Tensor,
    coeffs: tuple[torch.Tensor, torch.Tensor],
    unsqueeze_dim: int = 1,
):
    """Apply spherical-harmonic RoPE-style modulation to q/k using precomputed coefficients.

    Both q and k are multiplied by Y_lm(omega) at their respective positions. Under the real-pair
    representation of complex modes, the attention dot product is equivalent to
    Re[sum_m Y_lm(omega_r) Y_lm*(omega_s) q_m k_m*].
    """

    coeff_real, coeff_imag = coeffs
    return (
        _apply_complex_modulation(q, coeff_real, coeff_imag, unsqueeze_dim),
        _apply_complex_modulation(k, coeff_real, coeff_imag, unsqueeze_dim),
    )


def _apply_complex_modulation(
    x: torch.Tensor,
    coeff_real: torch.Tensor,
    coeff_imag: torch.Tensor,
    unsqueeze_dim: int,
) -> torch.Tensor:
    coeff_real = coeff_real.unsqueeze(unsqueeze_dim).to(dtype=x.dtype)
    coeff_imag = coeff_imag.unsqueeze(unsqueeze_dim).to(dtype=x.dtype)
    num_complex = coeff_real.shape[-1]
    max_complex = (x.shape[-1] - (x.shape[-1] % 2)) // 2
    if num_complex > max_complex:
        raise ValueError(
            f"Spherical RoPE requires {num_complex} complex modes but the head only supports "
            f"{max_complex}. Reduce rope_spherical_band or increase the head dimension."
        )
    num_rotary_dims = 2 * num_complex
    if num_rotary_dims == 0:
        return x

    x_rot = x[..., :num_rotary_dims].reshape(*x.shape[:-1], num_complex, 2)
    x_real = x_rot[..., 0]
    x_imag = x_rot[..., 1]
    out_real = (x_real * coeff_real) - (x_imag * coeff_imag)
    out_imag = (x_real * coeff_imag) + (x_imag * coeff_real)
    out = torch.stack((out_real, out_imag), dim=-1).flatten(-2, -1)
    if num_rotary_dims < x.shape[-1]:
        out = torch.cat((out, x[..., num_rotary_dims:]), dim=-1)
    return out


def build_spherical_rope_coeff_tensors(
    nside: int,
    band: int,
    num_local_queries: int,
    num_extra_tokens: int,
    device=None,
    dtype=torch.float32,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """Build spherical-RoPE coefficient tensors for cell-level, extra tokens, and packed tokens."""

    real_maps, imag_maps = _healpy_band_maps(nside, band)
    cell_real = torch.as_tensor(real_maps, device=device, dtype=dtype)
    cell_imag = torch.as_tensor(imag_maps, device=device, dtype=dtype)

    extra_real = torch.ones(
        num_extra_tokens, cell_real.shape[-1], device=cell_real.device, dtype=cell_real.dtype
    )
    extra_imag = torch.zeros_like(extra_real)
    packed_extra_real = (
        extra_real.unsqueeze(1).repeat(1, num_local_queries, 1).flatten(0, 1).unsqueeze(0)
    )
    packed_extra_imag = (
        extra_imag.unsqueeze(1).repeat(1, num_local_queries, 1).flatten(0, 1).unsqueeze(0)
    )

    packed_real = cell_real.unsqueeze(1).repeat(1, num_local_queries, 1).flatten(0, 1).unsqueeze(0)
    packed_imag = cell_imag.unsqueeze(1).repeat(1, num_local_queries, 1).flatten(0, 1).unsqueeze(0)

    return (
        (cell_real, cell_imag),
        (extra_real, extra_imag),
        (packed_extra_real, packed_extra_imag),
        (packed_real, packed_imag),
    )



@lru_cache(maxsize=32)
def _healpy_band_maps(
    nside: int, band: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Precompute one spherical-harmonic band on the HEALPix grid using healpy.

    The returned columns store the complex coefficients Y_lm(omega) for fixed l=band and
    m=-l,...,+l. These are the position factors used in spherical RoPE:

        q_m^omega = Y_lm(omega) q_m,    k_m^omega = Y_lm(omega) k_m.

    The following attention dot product then implicitly forms
    Y_lm(omega_r) Y_lm*(omega_s), matching the spherical harmonics addition-theorem
    structure.
    """

    num_pixels = hp.nside2npix(nside)
    real_maps = np.zeros((num_pixels, 2 * band + 1), dtype=np.float64)
    imag_maps = np.zeros((num_pixels, 2 * band + 1), dtype=np.float64)
    alm_size = hp.sphtfunc.Alm.getsize(band, band)

    for m in range(0, band + 1):
        # healpy stores alm only for m >= 0 and alm2map reconstructs a real field. Setting
        # a_lm=1 gives 2 Re[Y_lm] for m>0, while a_lm=i gives -2 Im[Y_lm]. We combine these
        # two real maps below to recover the complex coefficient Y_lm itself.
        alm_real = np.zeros(alm_size, dtype=np.complex128)
        alm_real[hp.sphtfunc.Alm.getidx(band, band, m)] = 1.0
        real_map = hp.alm2map(alm_real, nside=nside, lmax=band, mmax=band, pol=False)
        real_map = hp.reorder(real_map, r2n=True)

        if m == 0:
            # Y_l0 is real, and healpy returns it directly because there is no -m counterpart
            # to merge into the real map.
            real_maps[:, band] = real_map
            continue

        alm_imag = np.zeros(alm_size, dtype=np.complex128)
        alm_imag[hp.sphtfunc.Alm.getidx(band, band, m)] = 1.0j
        imag_map = hp.alm2map(alm_imag, nside=nside, lmax=band, mmax=band, pol=False)
        imag_map = hp.reorder(imag_map, r2n=True)

        pos_idx = band + m
        neg_idx = band - m
        sign = -1.0 if m % 2 else 1.0

        # Columns are ordered as m=-l,...,+l, hence band+m for +m and band-m for -m.
        # The negative-order mode follows the standard convention
        # Y_l,-m = (-1)^m Y_lm*.
        real_maps[:, pos_idx] = real_map / 2.0
        imag_maps[:, pos_idx] = -imag_map / 2.0
        real_maps[:, neg_idx] = sign * real_map / 2.0
        imag_maps[:, neg_idx] = sign * imag_map / 2.0

    return real_maps, imag_maps
