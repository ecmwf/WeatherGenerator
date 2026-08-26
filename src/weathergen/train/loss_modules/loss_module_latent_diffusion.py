# ruff: noqa: T201

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

import weathergen.train.loss_modules.loss_functions as loss_fns
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

_logger = logging.getLogger(__name__)


class LossLatentDiffusion(LossModuleBase):
    """
    Calculates loss in latent space.
    """

    def __init__(
        self,
        cf: DictConfig,
        mode_cfg: DictConfig,
        stage: Stage,
        device: str,
        **loss_fcts: dict,
    ):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.name = "LossLatentDiff"

        self.sigma_data = self.cf.sigma_data
        self.rho = self.cf.rho
        self.p_mean = self.cf.p_mean
        self.p_std = self.cf.p_std
        self.noise_distribution = self.cf.get("noise_distribution", "log_normal")

        # Dynamically load loss functions based on configuration and stage
        self.loss_fcts = [
            [
                getattr(loss_fns, name),
                params.get("weight", 1.0),
                name,
            ]
            for name, params in loss_fcts.items()
        ]

        self.random_target = None

    def _get_noise_weight(self, noise_level_rn):
        if self.noise_distribution == "log_uniform":
            sigma = noise_level_rn.exp()
        else:
            sigma = (noise_level_rn * self.p_std + self.p_mean).exp()

        # Select the per-noise-level loss weighting. The default "edm" weight is only
        # balanced when the denoiser uses EDM preconditioning (c_skip/c_out). When the
        # model predicts x0 directly (c_skip=0, c_out=1 in diffusion.denoise), that weight
        # behaves like ~1/sigma^2 and lets trivial near-clean (low-sigma) samples dominate
        # the gradient. The alternatives below are appropriate for direct x0-prediction.
        weighting = self.cf.get("diffusion_loss_weighting", "edm")
        if weighting == "edm":
            # lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
            return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        elif weighting == "min_snr_gamma":
            # Min-SNR-gamma weighting (Hang et al., 2023) for direct x0-prediction.
            # SNR = sigma_data^2 / sigma^2; capping at gamma prevents low-noise samples
            # from dominating while still down-weighting the untrained high-noise tail.
            gamma = self.cf.get("diffusion_min_snr_gamma", 5.0)
            snr = (self.sigma_data / sigma) ** 2
            return torch.clamp(snr, max=gamma) / self.sigma_data**2
        elif weighting == "uniform_x0":
            # Uniform weight in x0-space, normalised by the data variance.
            return torch.ones_like(sigma) / self.sigma_data**2
        else:
            raise ValueError(
                f"Unknown diffusion_loss_weighting '{weighting}'. "
                "Expected one of: 'edm', 'min_snr_gamma', 'uniform_x0'."
            )

    def _get_fstep_weights(self, forecast_steps):
        timestep_weight_config = self.cf.get("timestep_weight")
        if timestep_weight_config is None:
            return [1.0 for _ in range(forecast_steps)]
        weights_timestep_fct = getattr(loss_fns, timestep_weight_config[0])
        return weights_timestep_fct(forecast_steps, timestep_weight_config[1])

    def _loss_per_loss_function(
        self,
        loss_fct,
        target: torch.Tensor,
        pred: torch.Tensor,
        noise_weight: torch.Tensor = 1.0,
    ):
        """
        Compute loss for given loss function
        """

        loss, loss_chs = loss_fct(target=target, pred=pred)
        loss = noise_weight * loss

        return loss

    def compute_loss(self, preds: dict, targets: dict, **kwargs) -> LossValues:
        losses_all: dict[str, Tensor] = {
            f"{self.name}.{loss_fct_name}": torch.zeros(
                1,
                device=self.device,
            )
            for _, _, loss_fct_name in self.loss_fcts
        }

        pred_tokens_all = [
            pl["latent_state"].z_pre_norm for pl in preds.latent if pl and "latent_state" in pl
        ]
        target_tokens_all = [latent["diffusion_latent"] for latent in targets.latent if latent]

        # Remove the register and class tokens (prepended by the encoder) from the
        # predictions and targets so the diffusion loss is computed on healpix-cell
        # latents only.
        num_extra_tokens = self.cf.num_register_tokens + self.cf.num_class_tokens
        if num_extra_tokens > 0:
            pred_tokens_all = [tokens[:, num_extra_tokens:] for tokens in pred_tokens_all]
            target_tokens_all = [tokens[:, num_extra_tokens:] for tokens in target_tokens_all]

        # In ensemble mode predict_latent is not called, so latent predictions are absent.
        # In diffusion-rollout mode the model instead stores a per-step `latent_state` to
        # carry the rolled-forward state forward (for continuation / physical decoding), not
        # a per-fstep denoising prediction — so pred and target fstep counts do not
        # correspond and the latent diffusion loss is not meaningful. In all of these
        # inference-only cases, return a zero loss rather than crashing.
        is_ensemble = self.cf.get("fe_diffusion_num_ensemble_members", 1) > 1
        is_rollout = self.cf.get("diffusion_rollout", False)
        if not pred_tokens_all or is_ensemble or is_rollout:
            nan = torch.tensor(torch.nan).to(self.device)
            return LossValues(
                loss=torch.zeros(1, device=self.device),
                losses_all={f"{self.name}.{n}": nan for _, _, n in self.loss_fcts},
                stddev_all={"latent": nan},
            )

        eta = torch.tensor(
            [targets.aux_outputs["noise_level_rn"]], device=self.device, dtype=torch.float32
        )
        fsteps = len(target_tokens_all)

        # During validation, use unweighted loss (no noise-level scaling)
        noise_weight = 1.0 if self.stage == "val" else self._get_noise_weight(eta)
        fstep_loss_weights = self._get_fstep_weights(fsteps)

        loss_fsteps = torch.tensor(0.0, device=self.device, requires_grad=True)
        ctr_fsteps = 0
        for target_tokens, pred_tokens, fstep_loss_weight in zip(
            target_tokens_all, pred_tokens_all, fstep_loss_weights, strict=True
        ):
            # the first entry in tokens_all is the source itself, so skip it
            loss_fstep = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_loss_fcts = 0
            # if forecast_offset==0, then the timepoints correspond.
            # Otherwise targets don't encode the source timestep, so we don't need to skip
            for loss_fct, loss_fct_weight, loss_fct_name in self.loss_fcts:
                loss_lfct = self._loss_per_loss_function(
                    loss_fct,
                    target=target_tokens,
                    pred=pred_tokens,
                    noise_weight=noise_weight,
                )

                losses_all[f"{self.name}.{loss_fct_name}"] += loss_lfct  # TODO: break into fsteps

                # Add the weighted and normalized loss from this loss function to the total
                # batch loss
                loss_fstep = loss_fstep + (loss_fct_weight * loss_lfct)
                ctr_loss_fcts += 1 if loss_lfct > 0.0 else 0

            # Always add loss_fstep to keep the computational graph connected to model
            # parameters, even when ctr_loss_fcts==0 (e.g. NaN outputs from bf16 overflow).
            # GradScaler will detect NaN/inf gradients and skip the optimizer step safely.
            # Using `else 0` (a Python int) disconnects the graph and causes
            # GradScaler.step() to assert "No inf checks were recorded".
            loss_fsteps = loss_fsteps + (
                loss_fstep * fstep_loss_weight / ctr_loss_fcts
                if ctr_loss_fcts > 0
                else loss_fstep * 0.0
            )
            ctr_fsteps += 1 if ctr_loss_fcts > 0 else 0

        loss = loss_fsteps / (ctr_fsteps if ctr_fsteps > 0 else 1.0)

        for _, loss_values in losses_all.items():
            loss_values /= ctr_fsteps if ctr_fsteps > 0 else 1.0
            loss_values[loss_values == 0.0] = torch.nan

        return LossValues(
            loss=loss,
            losses_all=losses_all,
            stddev_all={"latent": torch.tensor(torch.nan).to(self.device)},
        )
