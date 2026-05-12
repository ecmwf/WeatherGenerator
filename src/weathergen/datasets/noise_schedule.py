"""Cosine noise schedule and noise application for self-flow augmentation.

Implements variance-preserving (VP) noise: x_t = alpha_t * x_0 + sigma_t * eps,
with alpha^2 + sigma^2 = 1 (cosine schedule). t=0 is clean, t=1 is pure noise.
"""

import math

import numpy as np
import torch


class CosineNoiseSchedule:
    """Variance-preserving cosine schedule."""

    def alpha_sigma(self, t: float) -> tuple[float, float]:
        alpha = math.cos(t * math.pi / 2)
        sigma = math.sin(t * math.pi / 2)
        return alpha, sigma


def sample_noise_levels(
    rng: np.random.Generator,
    schedule: CosineNoiseSchedule,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[float, float, float, float]:
    """Sample s < t and return (alpha_s, sigma_s, alpha_t, sigma_t)."""
    u1, u2 = rng.uniform(t_min, t_max, size=2)
    s_raw, t_raw = float(min(u1, u2)), float(max(u1, u2))
    alpha_s, sigma_s = schedule.alpha_sigma(s_raw)
    alpha_t, sigma_t = schedule.alpha_sigma(t_raw)
    return alpha_s, sigma_s, alpha_t, sigma_t


def apply_self_flow_noise(
    data: torch.Tensor,
    noise_mask: torch.Tensor,
    alpha_s: float,
    sigma_s: float,
    alpha_t: float,
    sigma_t: float,
    rng_seed: int,
) -> torch.Tensor:
    """VP noise with per-cell levels: noise_mask=True -> level t, else -> level s.

    Args:
        data: shape [num_points, num_vars]
        noise_mask: bool tensor, shape [num_points] (True = high noise at level t)
        alpha_s, sigma_s: noise level for non-masked points
        alpha_t, sigma_t: noise level for masked points
        rng_seed: seed for reproducible noise realization

    Returns:
        New tensor with noised data, same shape and dtype as input.
    """
    gen = torch.Generator().manual_seed(rng_seed)
    eps = torch.randn(data.shape, dtype=data.dtype, generator=gen)
    # Per-point alpha/sigma based on noise_mask
    alpha = torch.where(noise_mask[:, None], alpha_t, alpha_s)
    sigma = torch.where(noise_mask[:, None], sigma_t, sigma_s)
    return alpha * data + sigma * eps


def apply_uniform_noise(
    data: torch.Tensor,
    alpha: float,
    sigma: float,
    rng_seed: int,
) -> torch.Tensor:
    """Uniform VP noise at level s. Same seed -> same eps as apply_self_flow_noise.

    Args:
        data: shape [num_points, num_vars]
        alpha, sigma: noise level parameters
        rng_seed: seed for reproducible noise realization (same seed = same eps)

    Returns:
        New tensor with noised data, same shape and dtype as input.
    """
    gen = torch.Generator().manual_seed(rng_seed)
    eps = torch.randn(data.shape, dtype=data.dtype, generator=gen)
    return alpha * data + sigma * eps
