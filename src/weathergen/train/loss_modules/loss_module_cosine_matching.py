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
import torch.nn.functional as F
from omegaconf import DictConfig

from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

_logger = logging.getLogger(__name__)


class LossLatent(LossModuleBase):
    """
    Band hinge on per-token cosine similarity between consecutive FE latent steps.

    Penalises tokens whose cosine similarity to the previous step falls outside
    [cosine_low, cosine_high]. Both bounds are enforced as soft hinge losses so
    the FE is free inside the sweet-spot and pays a quadratic penalty outside it.

    cos_sim_to_prev is computed in model.forward() per patch token and stored in
    output.latent[step]["cos_sim_to_prev"] — no target calculator needed.
    """

    def __init__(
        self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str, **loss_fcts
    ):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.name = "LossLatent"

        params = next(iter(loss_fcts.values()), {}) if loss_fcts else {}
        self.cosine_low = params.get("cosine_low", 0.68)
        self.cosine_high = params.get("cosine_high", 0.78)

    def compute_loss(self, preds, targets, metadata, **kwargs) -> LossValues:
        acc_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        count = 0

        for step_pred in preds.latent:
            cos_sim = step_pred.get("cos_sim_to_prev", None)
            if cos_sim is None:
                continue
            step_loss = (
                F.relu(cos_sim - self.cosine_high) ** 2 + F.relu(self.cosine_low - cos_sim) ** 2
            ).mean()
            acc_loss = acc_loss + step_loss
            count += 1

        loss = acc_loss / count if count > 0 else acc_loss
        return LossValues(
            loss=loss, losses_all={"cosine_band": loss.detach().item()}, stddev_all={}
        )
