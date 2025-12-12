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

import weathergen.train.loss_modules.loss_functions as losses
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
        loss_fcts: list,
        stage: Stage,
        device: str,
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

        # Dynamically load loss functions based on configuration and stage
        self.loss_fcts = [[getattr(losses, name), w, name] for name, w in loss_fcts]

    def _get_noise_weight(self, eta):
        sigma = (eta * self.p_std + self.p_mean).exp()
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def _get_fstep_weights(self, forecast_steps):
        timestep_weight_config = self.cf.get("timestep_weight")
        if timestep_weight_config is None:
            return [1.0 for _ in range(forecast_steps)]
        weights_timestep_fct = getattr(losses, timestep_weight_config[0])
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

        loss_val = noise_weight * loss_fct(target=target, mu=pred)

        return loss_val

    def compute_loss(
        self,
        preds: dict,
        targets: dict,
    ) -> LossValues:
        losses_all: dict[str, Tensor] = {
            f"{self.name}.{loss_fct_name}": torch.zeros(
                1,
                device=self.device,
            )
            for _, _, loss_fct_name in self.loss_fcts
        }

        pred_tokens_all = [pl["latent_state"].latent_tokens for pl in preds.latent if pl]
        target_tokens_all = targets.latent
        eta = torch.tensor([targets.aux_outputs["noise_level_rn"]], device=self.device)
        fsteps = len(target_tokens_all)

        noise_weight = self._get_noise_weight(eta)
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

            loss_fsteps = loss_fsteps + (
                loss_fstep * fstep_loss_weight / ctr_loss_fcts if ctr_loss_fcts > 0 else 0
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
