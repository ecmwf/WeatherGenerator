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

from weathergen.train.loss_calculator_base import LossValues
from weathergen.train.loss_calculator_classes import LossCalculatorLatent, LossCalculatorPhysical
from weathergen.utils.train_logger import TRAIN, Stage

_logger = logging.getLogger(__name__)


class LossCalculator:
    """
    Manages and computes the overall loss for a WeatherGenerator model during
    training and validation stages.
    """

    def __init__(
        self,
        cf: DictConfig,
        stage: Stage,
        device: str,
    ):
        """
        Initializes the LossCalculator.

        This sets up the configuration, the operational stage (training or validation),
        the device for tensor operations, and initializes the list of loss functions
        based on the provided configuration.

        Args:
            cf: The OmegaConf DictConfig object containing model and training configurations.
                It should specify 'loss_fcts' for training and 'loss_fcts_val' for validation.
            stage: The current operational stage, either TRAIN or VAL.
                   This dictates which set of loss functions (training or validation) will be used.
            device: The computation device, such as 'cpu' or 'cuda:0', where tensors will reside.
        """
        self.cf = cf
        self.stage = stage
        self.device = device

        loss_fcts = cf.loss_fcts if stage == TRAIN else cf.loss_fcts_val

        loss_fcts_physical = [[name, w] for name, w in loss_fcts if name.split(":")[0] != "latent"]
        loss_fcts_latent = [
            [name.split(":")[1], w] for name, w in loss_fcts if name.split(":")[0] == "latent"
        ]

        calculator_configs = []

        if loss_fcts_physical:
            calculator_configs.append((LossCalculatorPhysical, loss_fcts_physical, "physical"))
        if loss_fcts_latent:
            calculator_configs.append((LossCalculatorLatent, loss_fcts_latent, "latent"))

        self.loss_calculators = [
            (Cls(cf=cf, loss_fcts=losses, stage=stage, device=self.device), type)
            for (Cls, losses, type) in calculator_configs
        ]

    def compute_loss(
        self,
        preds: dict,
        targets: dict,
    ):
        loss_values = {}
        loss = 0
        for calculator, type in self.loss_calculators:
            loss_values[type] = calculator.compute_loss(preds=preds[type], targets=targets[type])
            loss += loss_values[type].loss

        losses_all = {}
        stddev_all = {}
        for _, v in loss_values.items():
            losses_all.update(v.losses_all)
            stddev_all.update(v.stddev_all)
        return LossValues(loss=loss, losses_all=losses_all, stddev_all=stddev_all)
