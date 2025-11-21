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


class LossLatent(LossModuleBase):
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
        self.name = "LossLatent"

        # Dynamically load loss functions based on configuration and stage
        self.loss_fcts = [
            [getattr(losses, name if name != "mse" else "mse_channel_location_weighted"), w]
            for name, w in loss_fcts
        ]

    def _loss_per_loss_function(
        self,
        loss_fct,
        target: torch.Tensor,
        pred: torch.Tensor,
    ):
        """
        Compute loss for given loss function
        """

        loss_val = loss_fct(target=target, ens=None, mu=pred)

        return loss_val

    def compute_loss(
        self,
        preds: list[list[Tensor]],
        targets: list[list[any]],
    ) -> LossValues:
        return super().compute_loss(preds, targets)

        ### FROM KEREM's PR
        # losses_all: Tensor = torch.zeros(
        #     len(self.loss_fcts),
        #     device=self.device,
        # )

        # loss_fsteps_lat = torch.tensor(0.0, device=self.device, requires_grad=True)
        # ctr_fsteps_lat = 0
        # # TODO: KCT, do we need the below per fstep?
        # for fstep in range(
        #     1, len(preds)
        # ):  # the first entry in tokens_all is the source itself, so skip it
        #     loss_fstep = torch.tensor(0.0, device=self.device, requires_grad=True)
        #     ctr_loss_fcts = 0
        #     # if forecast_offset==0, then the timepoints correspond.
        #     # Otherwise targets don't encode the source timestep, so we don't need to skip
        #     fstep_targs = fstep if self.cf.forecast_offset == 0 else fstep - 1
        #     for i_lfct, (loss_fct, loss_fct_weight) in enumerate(self.loss_fcts_lat):
        #         loss_lfct = self._loss_per_loss_function(
        #             loss_fct,
        #             stream_info=None,
        #             target=targets[fstep_targs],
        #             pred=preds[fstep],
        #         )

        #         losses_all[i_lfct] += loss_lfct  # TODO: break into fsteps

        #         # Add the weighted and normalized loss from this loss function to the total
        #         # batch loss
        #         loss_fstep = loss_fstep + (loss_fct_weight * loss_lfct)
        #         ctr_loss_fcts += 1 if loss_lfct > 0.0 else 0

        #     loss_fsteps_lat = loss_fsteps_lat + (
        #         loss_fstep / ctr_loss_fcts if ctr_loss_fcts > 0 else 0
        #     )
        #     ctr_fsteps_lat += 1 if ctr_loss_fcts > 0 else 0

        # loss = loss_fsteps_lat / (ctr_fsteps_lat if ctr_fsteps_lat > 0 else 1.0)

        # losses_all /= ctr_fsteps_lat if ctr_fsteps_lat > 0 else 1.0
        # losses_all[losses_all == 0.0] = torch.nan

        # return LossValues(loss=loss, losses_all=losses_all)
