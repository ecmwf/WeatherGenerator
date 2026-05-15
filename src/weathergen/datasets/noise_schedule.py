"""Noise schedules and noise application for self-flow augmentation.

Implements variance-preserving (VP) noise: x_t = alpha_t * x_0 + sigma_t * eps,
with alpha^2 + sigma^2 = 1.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray


class NoiseSchedule(Protocol):
    """Interface for scalar VP noise schedules."""

    def alpha_sigma(self, t: float) -> tuple[float, float]:
        """Return the VP coefficients at noise coordinate ``t``."""


DEFAULT_VARIABLE_COVARIANCE_EIGENVALUES: dict[str, tuple[float, ...]] = {
    "t": (
        9.003102596830489,
        2.293351658543038,
        0.6748801083907654,
        0.2604952628281507,
        0.09891307429514709,
        0.08263724819193823,
        0.05768254592530663,
        0.04190002514522106,
        0.02868342247319774,
        0.018737974953089695,
        0.011013001979555728,
        0.007630323366702682,
        0.006520842322586829,
        0.004351893127165177,
    ),
    "q": (
        8.394306318307711,
        1.2059738965026323,
        0.7092965676610652,
        0.5829882550553874,
        0.4300830817278935,
        0.3187069082052382,
        0.21523107544510497,
        0.1717038250154352,
        0.12416142934086521,
        0.11052846444467718,
        0.10200146060827858,
        0.08437577734477344,
        0.032619192060751985,
    ),
    "z": (
        12.318100395742182,
        0.9935742873765985,
        0.3933829732401214,
        0.03218962999589178,
        0.009788407172634683,
        0.0029283911285314227,
        0.001461108379491582,
        0.0005823541536375106,
        0.00032003128133065357,
        0.00018680089044569518,
        0.00009760131405860466,
        0.00003563728854203592,
        0.00001688517499230172,
    ),
    "u": (
        8.913235337386716,
        2.507673143850246,
        0.7827225333260602,
        0.4574805969615945,
        0.199798045399375,
        0.14437941697101112,
        0.09277260698892087,
        0.06475068523587617,
        0.04397533320094085,
        0.02956532764857135,
        0.025222696933906528,
        0.019031891725549283,
        0.011105653968501782,
        0.009656046027393781,
    ),
    "v": (
        7.599048297205972,
        2.725523295962741,
        0.7630413968805377,
        0.5762015933695852,
        0.33344609575102374,
        0.2790374303946277,
        0.18349997471849738,
        0.1309986034101711,
        0.09421573441116741,
        0.06649958989218436,
        0.04295032232242902,
        0.03741428614740747,
        0.02641032531289457,
        0.012535769503125379,
    ),
}


def variable_group_from_channel(channel_name: str) -> str | None:
    """Map ERA5 channel names to the self-flow covariance groups."""

    if channel_name == "2t" or channel_name.startswith("t_"):
        return "t"
    if channel_name.startswith("q_"):
        return "q"
    if channel_name.startswith("z_"):
        return "z"
    if channel_name == "10u" or channel_name.startswith("u_"):
        return "u"
    if channel_name == "10v" or channel_name.startswith("v_"):
        return "v"
    return None


def covariance_information_from_sigma(eigenvalues: Sequence[float], sigma: float) -> float:
    """Gaussian mutual information per covariance mode for VP noise."""

    if not 0.0 < sigma <= 1.0:
        raise ValueError(f"sigma must be in (0, 1], got {sigma}.")

    eigenvalues_np = _positive_eigenvalues(eigenvalues)
    snr = (1.0 - sigma**2) / sigma**2
    return float(0.5 * np.log1p(snr * eigenvalues_np).mean())


def _positive_eigenvalues(eigenvalues: Sequence[float]) -> NDArray:
    eigenvalues_np = np.asarray(eigenvalues, dtype=np.float64)
    if eigenvalues_np.ndim != 1 or eigenvalues_np.size == 0:
        raise ValueError("Covariance eigenvalues must be a non-empty 1D sequence.")
    if not np.all(np.isfinite(eigenvalues_np)):
        raise ValueError("Covariance eigenvalues must be finite.")
    if np.any(eigenvalues_np <= 0.0):
        raise ValueError("Covariance eigenvalues must be positive.")
    return eigenvalues_np


class CosineNoiseSchedule:
    """Variance-preserving cosine schedule.

    This schedule is linear in angle: ``alpha = cos(pi * t / 2)`` and
    ``sigma = sin(pi * t / 2)``.
    """

    def alpha_sigma(self, t: float) -> tuple[float, float]:
        alpha = math.cos(t * math.pi / 2)
        sigma = math.sin(t * math.pi / 2)
        return alpha, sigma


class LinearInformationNoiseSchedule:
    """VP schedule with information gain linear in ``t``.

    For normalized Gaussian data, ``I(t) = -log(sigma(t))`` is a useful
    scalar information proxy. This schedule makes that quantity linear over
    ``[t_min, t_max]`` while matching the cosine schedule's endpoint noise
    levels by default.
    """

    def __init__(
        self,
        t_min: float = 0.1,
        t_max: float = 0.9,
        reference_schedule: NoiseSchedule | None = None,
    ) -> None:
        if not 0.0 < t_min < t_max <= 1.0:
            raise ValueError(
                "LinearInformationNoiseSchedule requires 0.0 < t_min < t_max <= 1.0 "
                f"(got t_min={t_min}, t_max={t_max})."
            )

        reference_schedule = reference_schedule or CosineNoiseSchedule()
        _, sigma_min = reference_schedule.alpha_sigma(t_min)
        _, sigma_max = reference_schedule.alpha_sigma(t_max)

        if not 0.0 < sigma_min < sigma_max <= 1.0:
            raise ValueError(
                "Reference schedule must provide increasing sigma values in (0, 1] "
                f"over [t_min, t_max] (got sigma_min={sigma_min}, sigma_max={sigma_max})."
            )

        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)

    def alpha_sigma(self, t: float) -> tuple[float, float]:
        if not self.t_min <= t <= self.t_max:
            raise ValueError(
                f"Noise coordinate t={t} is outside the linear-information range "
                f"[{self.t_min}, {self.t_max}]."
            )

        tau = (t - self.t_min) / (self.t_max - self.t_min)
        log_sigma = (1.0 - tau) * self.log_sigma_min + tau * self.log_sigma_max
        sigma = math.exp(log_sigma)
        alpha = math.sqrt(1.0 - sigma**2)
        return alpha, sigma

    def information(self, t: float) -> float:
        """Return ``-log(sigma(t))`` for diagnostics."""

        _, sigma = self.alpha_sigma(t)
        return -math.log(sigma)


class CovarianceLinearInformationNoiseSchedule:
    """VP schedule with covariance-aware Gaussian information linear in ``t``.

    For data with covariance eigenvalues ``lambda_k``, the Gaussian mutual
    information under VP noise is ``0.5 * mean(log(1 + SNR * lambda_k))``.
    This schedule inverts that curve so equal-width bins in ``t`` have equal
    covariance-aware information gain.
    """

    def __init__(
        self,
        eigenvalues: Sequence[float],
        t_min: float = 0.1,
        t_max: float = 0.9,
        reference_schedule: NoiseSchedule | None = None,
    ) -> None:
        if not 0.0 < t_min < t_max <= 1.0:
            raise ValueError(
                "CovarianceLinearInformationNoiseSchedule requires "
                f"0.0 < t_min < t_max <= 1.0 (got t_min={t_min}, t_max={t_max})."
            )

        reference_schedule = reference_schedule or CosineNoiseSchedule()
        _, sigma_min = reference_schedule.alpha_sigma(t_min)
        _, sigma_max = reference_schedule.alpha_sigma(t_max)

        if not 0.0 < sigma_min < sigma_max <= 1.0:
            raise ValueError(
                "Reference schedule must provide increasing sigma values in (0, 1] "
                f"over [t_min, t_max] (got sigma_min={sigma_min}, sigma_max={sigma_max})."
            )

        self.eigenvalues = _positive_eigenvalues(eigenvalues)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.information_min_noise = self.information_from_sigma(self.sigma_min)
        self.information_max_noise = self.information_from_sigma(self.sigma_max)

    def information_from_sigma(self, sigma: float) -> float:
        return covariance_information_from_sigma(self.eigenvalues, sigma)

    def alpha_sigma(self, t: float) -> tuple[float, float]:
        if not self.t_min <= t <= self.t_max:
            raise ValueError(
                f"Noise coordinate t={t} is outside the covariance-linear-information "
                f"range [{self.t_min}, {self.t_max}]."
            )

        tau = (t - self.t_min) / (self.t_max - self.t_min)
        target_information = (
            1.0 - tau
        ) * self.information_min_noise + tau * self.information_max_noise
        sigma = self._sigma_for_information(target_information)
        alpha = math.sqrt(1.0 - sigma**2)
        return alpha, sigma

    def information(self, t: float) -> float:
        _, sigma = self.alpha_sigma(t)
        return self.information_from_sigma(sigma)

    def _sigma_for_information(self, target_information: float) -> float:
        lo = self.sigma_min
        hi = self.sigma_max
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.information_from_sigma(mid) > target_information:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


class VariableCovarianceLinearInformationNoiseSchedule:
    """Channel-aware covariance-linear schedule for ERA5 variable groups."""

    def __init__(
        self,
        variable_eigenvalues: Mapping[str, Sequence[float]] | None = None,
        t_min: float = 0.1,
        t_max: float = 0.9,
        reference_schedule: NoiseSchedule | None = None,
    ) -> None:
        reference_schedule = reference_schedule or CosineNoiseSchedule()
        variable_eigenvalues = variable_eigenvalues or DEFAULT_VARIABLE_COVARIANCE_EIGENVALUES
        self.group_schedules = {
            group: CovarianceLinearInformationNoiseSchedule(
                eigenvalues,
                t_min=t_min,
                t_max=t_max,
                reference_schedule=reference_schedule,
            )
            for group, eigenvalues in variable_eigenvalues.items()
        }
        self.fallback_schedule = LinearInformationNoiseSchedule(
            t_min=t_min,
            t_max=t_max,
            reference_schedule=reference_schedule,
        )

    def alpha_sigma(self, t: float) -> tuple[float, float]:
        """Return fallback scalar coefficients for non-grouped variables."""

        return self.fallback_schedule.alpha_sigma(t)

    def alpha_sigma_for_group(self, group: str, t: float) -> tuple[float, float]:
        if group not in self.group_schedules:
            raise ValueError(f"Unsupported covariance group '{group}'.")
        return self.group_schedules[group].alpha_sigma(t)

    def alpha_sigma_for_channels(
        self,
        t: float,
        channel_names: Sequence[str],
    ) -> tuple[NDArray, NDArray]:
        alpha_values = []
        sigma_values = []
        fallback_alpha, fallback_sigma = self.fallback_schedule.alpha_sigma(t)

        for channel_name in channel_names:
            group = variable_group_from_channel(channel_name)
            if group is None or group not in self.group_schedules:
                alpha, sigma = fallback_alpha, fallback_sigma
            else:
                alpha, sigma = self.group_schedules[group].alpha_sigma(t)
            alpha_values.append(alpha)
            sigma_values.append(sigma)

        return (
            np.asarray(alpha_values, dtype=np.float32),
            np.asarray(sigma_values, dtype=np.float32),
        )

    def information(self, t: float, group: str | None = None) -> float:
        if group is None:
            return self.fallback_schedule.information(t)
        if group not in self.group_schedules:
            raise ValueError(f"Unsupported covariance group '{group}'.")
        return self.group_schedules[group].information(t)


def sample_noise_coordinates(
    rng: np.random.Generator,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[float, float]:
    """Sample raw self-flow coordinates ``s < t`` from the configured range."""

    u1, u2 = rng.uniform(t_min, t_max, size=2)
    return float(min(u1, u2)), float(max(u1, u2))


def sample_noise_levels(
    rng: np.random.Generator,
    schedule: NoiseSchedule,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[float, float, float, float]:
    """Sample s < t and return (alpha_s, sigma_s, alpha_t, sigma_t)."""
    s_raw, t_raw = sample_noise_coordinates(rng, t_min=t_min, t_max=t_max)
    alpha_s, sigma_s = schedule.alpha_sigma(s_raw)
    alpha_t, sigma_t = schedule.alpha_sigma(t_raw)
    return alpha_s, sigma_s, alpha_t, sigma_t


def _coefficient_tensor(
    value: float | Sequence[float] | NDArray | torch.Tensor,
    data: torch.Tensor,
    name: str,
) -> torch.Tensor:
    coefficient = torch.as_tensor(value, dtype=data.dtype, device=data.device)
    if coefficient.ndim == 0:
        return coefficient
    if coefficient.ndim != 1:
        raise ValueError(f"{name} must be scalar or a 1D channel vector.")
    if coefficient.shape[0] != data.shape[1]:
        raise ValueError(f"{name} has {coefficient.shape[0]} channels, expected {data.shape[1]}.")
    return coefficient[None, :]


def apply_self_flow_noise(
    data: torch.Tensor,
    noise_mask: torch.Tensor,
    alpha_s: float | Sequence[float] | NDArray | torch.Tensor,
    sigma_s: float | Sequence[float] | NDArray | torch.Tensor,
    alpha_t: float | Sequence[float] | NDArray | torch.Tensor,
    sigma_t: float | Sequence[float] | NDArray | torch.Tensor,
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
    noise_mask = noise_mask.to(device=data.device, dtype=torch.bool)
    alpha_s_tensor = _coefficient_tensor(alpha_s, data, "alpha_s")
    sigma_s_tensor = _coefficient_tensor(sigma_s, data, "sigma_s")
    alpha_t_tensor = _coefficient_tensor(alpha_t, data, "alpha_t")
    sigma_t_tensor = _coefficient_tensor(sigma_t, data, "sigma_t")
    alpha = torch.where(noise_mask[:, None], alpha_t_tensor, alpha_s_tensor)
    sigma = torch.where(noise_mask[:, None], sigma_t_tensor, sigma_s_tensor)
    return alpha * data + sigma * eps


def apply_uniform_noise(
    data: torch.Tensor,
    alpha: float | Sequence[float] | NDArray | torch.Tensor,
    sigma: float | Sequence[float] | NDArray | torch.Tensor,
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
    alpha_tensor = _coefficient_tensor(alpha, data, "alpha")
    sigma_tensor = _coefficient_tensor(sigma, data, "sigma")
    return alpha_tensor * data + sigma_tensor * eps
