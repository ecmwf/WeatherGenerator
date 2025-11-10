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

import numpy as np
from omegaconf import DictConfig

import torch
from torch import Tensor
import torch.nn.functional as F

import weathergen.train.loss as losses
from weathergen.train.loss import stat_loss_fcts
from weathergen.train.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import TRAIN, VAL, Stage

_logger = logging.getLogger(__name__)


class LossLatentSSLStudentTeacher(LossModuleBase):
    """
    Manages and computes the overall loss for a WeatherGenerator model pretraining using
    DINO/iBOT/JEPA/BYOL style losses.

    This class handles the initialization and application of various loss functions,
    It provides both the main loss for backpropagation and detailed loss metrics for logging.
    """

    valid_loss_names = set("DINO", "iBOT", "JEPA")

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

        # Dynamically load loss functions based on configuration and stage
        self.losses = {
            name: get_loss_function_ssl(name) for name in losses if name in self.valid_loss_names
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
        losses_all: dict[str, Tensor] = { loss : 0.0
            for loss in self.losses
        }

        for name, loss_fn in losses:
            loss_value = loss_fn(preds.latent[name], targets[name]).mean()
            loss += loss_value
            losses_all[name] = loss_value.item()

        return loss





def get_loss_function_ssl(name):
    if name == "iBOT":
        return losses.masked_student_teacher_patch_softmax
    elif name == "DINO":
        return losses.student_teacher_global_softmax
    elif name == "JEPA":
        return F.l1_loss
    else:
        raise NotImplementedError(
            f"{name} is not an implemented loss for the LossLatentSSLStudentTeacher"
        )
