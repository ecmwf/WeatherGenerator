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
from collections import defaultdict

import torch
from omegaconf import DictConfig

import weathergen.train.loss_modules as LossModules
from weathergen.model.model import ModelOutput
from weathergen.train.target_and_aux_module_base import TargetAuxOutput
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
        self.loss_hist = []
        self.losses_unweighted_hist = []
        self.stddev_unweighted_hist = []

        training_config = cf.get("training_config")
        loss_configs = [(t.num_samples, t.loss) for t in training_config.model_input]

        calculator_configs = []
        for num_samples, lc in loss_configs:
            for _ in range(num_samples):
                calculator_configs += (
                    lc.training if stage == TRAIN else lc.get("validation", lc.training)
                )

        calculator_configs = [
            (getattr(LossModules, Cls), config)
            for t in calculator_configs
            for (Cls, config) in t.items()
        ]

        self.loss_calculators = [
            (config.weight, Cls(cf=cf, loss_fcts=config.loss_fcts, stage=stage, device=self.device))
            for (Cls, config) in calculator_configs
        ]

    def compute_loss(
        self,
        preds: ModelOutput,
        targets: TargetAuxOutput,
    ):
        losses_all = defaultdict(dict)
        stddev_all = defaultdict(dict)
        loss = torch.tensor(0.0, requires_grad=True)

        for weight, calculator in self.loss_calculators:
            loss_values = calculator.compute_loss(preds=preds, targets=targets)
            loss = loss + weight * loss_values.loss
            losses_all[calculator.name] = loss_values.losses_all
            losses_all[calculator.name]["loss_avg"] = loss_values.loss
            stddev_all[calculator.name] = loss_values.stddev_all

        # Keep histories for logging
        self.loss_hist += [loss.detach()]
        self.losses_unweighted_hist += [losses_all]
        self.stddev_unweighted_hist += [stddev_all]

        return loss
