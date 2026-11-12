# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from omegaconf import DictConfig

from weathergen.model.flow_matching import GaussianPath
from weathergen.train.loss_modules.loss_module_latent_diffusion import LossLatentDiffusionBase
from weathergen.utils.train_logger import Stage


class LossFlowMatching(LossLatentDiffusionBase):
    """Conditional flow-matching / denoising-score-matching (/ EDM) loss in latent space.

    Regresses the network's prediction (velocity / noise / score / **denoiser**, per
    ``diffusion.flow_matching.prediction_type``) against the analytic conditional target
    (course Eq. 37 / Alg. 4 / Eq. 40; denoiser -> ``z``, as in LossLatentDiffusion).

    FlowMatchingForecastEngine returns the clean-latent estimate ``D_t(x_t)`` and
    DiffusionLatentTargetEncoder provides the clean latent ``z`` together with the per-sample
    noise level (``t`` for condot, ``sigma`` for ve). Since every parameterization is affine in
    ``D`` at fixed ``x_t``, ``pred - target = a_t (D - z)`` (see
    ``GaussianPath.denoiser_to_target_scale``), so scaling both by ``a_t`` reproduces the
    parameterization-space regression exactly for difference-based loss functions (mse, ...)
    without the loss needing the engine's ``eps``.

    During **training**, the ve (EDM) path is additionally weighted by the EDM
    ``lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma*sigma_data)^2``; condot is unweighted.
    **Validation** is unweighted and computed in x0 space: unlike the training loss, it IS
    comparable across velocity/noise/score/denoiser runs, because each parameterization
    regresses a target with a different scale.
    """

    diffusion_type = "flow_matching"
    name = "LossFlowMatching"

    def __init__(
        self,
        cf: DictConfig,
        mode_cfg: DictConfig,
        stage: Stage,
        device: str,
        **loss_fcts: dict,
    ):
        super().__init__(cf, mode_cfg, stage, device, **loss_fcts)
        fm_cf = self.diff_cf.flow_matching
        self.path = GaussianPath.from_config(cf)
        self.prediction_type = fm_cf.get("prediction_type", "velocity")
        self.sigma_data = fm_cf.get("sigma_data", 1.0)

    def _get_noise_weight(self, sigma: torch.Tensor) -> torch.Tensor | None:
        if self.path.kind == "ve":
            return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        return None

    def _to_loss_space(self, tokens: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if self.stage == "val":
            return tokens
        t = self.path.path_variable(sigma, device=tokens.device)
        return self.path.denoiser_to_target_scale(t, self.prediction_type) * tokens
