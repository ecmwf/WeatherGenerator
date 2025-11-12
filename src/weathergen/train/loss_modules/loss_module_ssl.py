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

from omegaconf import DictConfig

import torch.nn.functional as F
import torch
from torch import Tensor

import weathergen.train.loss_modules.loss as loss_fns
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

_logger = logging.getLogger(__name__)

class LossLatentSSLStudentTeacher(LossModuleBase):
    """
    Manages and computes the overall loss for a WeatherGenerator model pretraining using
    DINO/iBOT/JEPA/BYOL style losses.

    This class handles the initialization and application of various loss functions,
    It provides both the main loss for backpropagation and detailed loss metrics for logging.
    """

    valid_loss_names = set(["DINO", "iBOT", "JEPA"])

    def __init__(
        self,
        cf: DictConfig,
        losses: list,
        stage: Stage,
        device: str,
    ):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.name = "LossLatentSSLStudentTeacher"
        self.local_cf = cf["training_mode_config"]["losses"][self.name]

        # Dynamically load loss functions based on configuration and stage
        self.losses = {
            name: (self.local_cf[name]["weight"], get_loss_function_ssl(name))
            for name in losses
            if name in self.valid_loss_names
        }

    def compute_loss(
        self,
        preds: dict,
        targets: dict,
    ) -> LossValues:
        # gradient loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # initialize dictionaries for detailed loss tracking and standard deviation statistics
        # create tensor for each stream
        losses_all: dict[str, Tensor] = {loss: 0.0 for loss in self.losses}

        for name, (weight, loss_fn) in self.losses.items():
            loss_value = loss_fn(preds.latent[name], targets[name]).mean()
            loss += weight * loss_value
            losses_all[name] = loss_value.item()

        return loss


def get_loss_function_ssl(name):
    if name == "iBOT":
        return loss_fns.masked_student_teacher_patch_softmax
    elif name == "DINO":
        return loss_fns.student_teacher_global_softmax
    elif name == "JEPA":
        return F.l1_loss
    else:
        raise NotImplementedError(
            f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
        )

