# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Adapted from https://github.com/mlexpos/adana/blob/main/src/optim/adana.py

import torch
from torch.optim import Optimizer


class ADana(Optimizer):
    """
    ADana (Adaptive Damped Nesterov Acceleration) optimizer.

    Log-time scheduling of momentum and weight decay without tau probability estimator.
    Two modes:
    - clipsnr=None: No SNR clipping (base ADana)
    - clipsnr=float: SNR clipping enabled (Dana-MK4)

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (initial value for param_groups).
        lr_peak: Peak scheduled learning rate, used as reference for schedule_factor
            in weight decay. If None, defaults to lr.
        delta: Delta parameter for EMA coefficient (default: 8.0).
        kappa: Kappa parameter for effective time scaling (default: 0.85).
        epsilon: Small constant for numerical stability (default: 1e-8).
        weight_decay: Weight decay parameter (default: 0.0).
        clipsnr: SNR clipping parameter. None disables clipping (default: None).
        wd_decaying: Whether to decay weight decay over time (default: True).
        wd_ts: Timescale for weight decay decay (default: 1.0).
        gamma_3_factor: Scaling factor for the g3 (long-momentum) term (default: 1.0).
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        lr_peak: float | None = None,
        delta: float = 8.0,
        kappa: float = 0.85,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        clipsnr: float | None = None,
        wd_decaying: bool = True,
        wd_ts: float = 1.0,
        gamma_3_factor: float = 1.0,
    ):
        defaults = dict(
            lr=lr,
            delta=delta,
            epsilon=epsilon,
            weight_decay=weight_decay,
            weighted_step_count=0,
        )
        # lr_peak is the reference for schedule_factor (weight decay modulation).
        # schedule_factor = group['lr'] / lr_peak, so it stays in [0, 1] during training.
        self.lr = lr_peak if lr_peak is not None else lr
        self.delta = delta
        self.kappa = kappa
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clipsnr = clipsnr
        self.wd_decaying = wd_decaying
        self.wd_ts = wd_ts
        self.gamma_3_factor = gamma_3_factor

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            g2 = group["lr"]
            g3 = group["lr"]
            schedule_factor = group["lr"] / self.lr  # γ(t) without peak LR
            time_factor = schedule_factor
            group["weighted_step_count"] += time_factor
            delta = group["delta"]
            wd = group["weight_decay"]
            epsilon = group["epsilon"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m, v = state["m"], state["v"]
                state["step"] += 1

                step = state["step"]
                alpha = delta / (delta + step)

                # Update first moment (EMA of gradient)
                m.lerp_(grad, alpha)

                # Update second moment (EMA of gradient squared)
                v.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)

                # Compute sqrt(v) + epsilon once and reuse
                sqrt_v_eps = torch.sqrt(v).add_(epsilon)

                # Compute normalization term
                norm_term = 1.0 / sqrt_v_eps
                m_norm_term = torch.abs(m) * norm_term

                # Compute momentum factor and alpha factor using step-based time
                effective_time = 1.0 + step
                mfac = m_norm_term

                if self.clipsnr is not None:
                    alpha_factor = torch.clamp(
                        (effective_time ** (1 - self.kappa)) * mfac,
                        max=self.clipsnr,
                    )
                else:
                    alpha_factor = (effective_time ** (1 - self.kappa)) * mfac

                # Compute g3 term (momentum-based update), scaled by gamma_3_factor
                g3_term = (
                    (-g3) * self.gamma_3_factor * (torch.sign(m) * alpha_factor + m * norm_term)
                )

                # Compute g2 term (gradient-based update)
                g2_term = (-g2) * grad * norm_term

                # Combine updates
                p.add_(g2_term + g3_term)

                # Apply independent weight decay (paper convention):
                # WD is multiplied by schedule γ(t) but NOT by peak LR γ*
                if self.wd_decaying:
                    wd_factor = -wd / (1 + step / self.wd_ts) * schedule_factor
                else:
                    wd_factor = -wd * schedule_factor
                p.mul_(1 + wd_factor)

        return loss
