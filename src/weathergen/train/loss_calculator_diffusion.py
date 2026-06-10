# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import torch
from omegaconf import DictConfig
from torch import Tensor

from weathergen.train.loss_calculator import LossCalculator
from weathergen.train.utils import TRAIN, Stage

_logger = logging.getLogger(__name__)


def edm_noise_weight(
    noise_level_rn: Tensor,
    sigma_data: float,
    p_mean: float,
    p_std: float,
    noise_distribution: str = "log_normal",
) -> Tensor:
    """EDM loss weight λ(σ) = (σ² + σ_data²) / (σ·σ_data)².

    σ is reconstructed from the per-sample noise level:
      - ``log_uniform``: ``noise_level_rn`` is log σ directly → σ = exp(noise_level_rn)
      - otherwise (``log_normal``): ``noise_level_rn`` is η ~ N(0,1) → σ = exp(η·p_std + p_mean)
    """
    if noise_distribution == "log_uniform":
        sigma = noise_level_rn.exp()
    else:
        sigma = (noise_level_rn * p_std + p_mean).exp()
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2


class DiffusionLossCalculator(LossCalculator):
    """LossCalculator that applies the EDM noise weight λ(σ) to every loss term in diffusion training.

    The loss modules (``LossLatentDiffusion``, ``LossPhysical``) return raw, unweighted losses; this
    calculator multiplies each active term by the same per-batch λ(σ) at combine time. Because the
    weight is shared, it factors out of the sum, so the intended ``w_phys : w_latent`` ratio between
    the per-term ``weight``s is preserved at every noise level. All diffusion σ-weighting lives here;
    the base ``LossCalculator`` and the loss modules carry no σ logic.
    """

    def __init__(self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str):
        super().__init__(cf, mode_cfg, stage, device)
        # Parameters needed to reconstruct σ from the per-sample noise level.
        self.sigma_data = cf.sigma_data
        self.p_mean = cf.p_mean
        self.p_std = cf.p_std
        self.noise_distribution = cf.get("noise_distribution", "log_normal")

    @staticmethod
    def _noise_level(target):
        """Return ``noise_level_rn`` carried by a target, or None if absent."""
        aux = getattr(target, "aux_outputs", None) or {}
        return aux.get("noise_level_rn", None)

    def _find_noise_level(self, targets_and_aux):
        """Return the diffusion driver's noise level, or None if no diffusion term is present."""
        for target in targets_and_aux.values():
            noise_level = self._noise_level(target)
            if noise_level is not None:
                return noise_level
        return None

    def _term_scale(self, term_name, calculator, targets_and_aux):
        # Validation uses unweighted loss, so leave every term unscaled outside training.
        if self.stage != TRAIN:
            return 1.0

        # λ(σ) weights all loss terms equally in diffusion training (latent + physical). Since the
        # loss modules now return raw losses, this is the single point where λ(σ) is applied.
        # No target carries noise_level_rn → non-diffusion training → no scaling (base behavior).
        noise_level_rn = self._find_noise_level(targets_and_aux)
        if noise_level_rn is None:
            return 1.0

        eta = torch.tensor([noise_level_rn], device=self.device, dtype=torch.float32)
        return edm_noise_weight(
            eta, self.sigma_data, self.p_mean, self.p_std, self.noise_distribution
        )


def make_loss_calculator(
    cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str
) -> LossCalculator:
    """Select the loss calculator for a mode config.

    Returns ``DiffusionLossCalculator`` when a diffusion (latent) loss term is configured, so the
    EDM noise weight is propagated to the physical loss; otherwise the plain ``LossCalculator``.
    """
    losses = mode_cfg.get("losses", {}) or {}
    is_diffusion = any(v.get("type") == "LossLatentDiffusion" for v in losses.values())
    cls = DiffusionLossCalculator if is_diffusion else LossCalculator
    return cls(cf, mode_cfg, stage, device)
