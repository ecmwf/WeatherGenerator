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
from copy import deepcopy

import torch
from omegaconf import DictConfig

import weathergen.train.loss_modules as LossModules
from weathergen.model.model import ModelOutput
from weathergen.train.target_and_aux_module_base import TargetAuxOutput
from weathergen.utils.train_logger import Stage

_logger = logging.getLogger(__name__)


class LossCalculator:
    """
    Manages and computes the overall loss for a WeatherGenerator model during
    training and validation stages.
    """

    def __init__(
        self,
        cf: DictConfig,
        mode_cfg: DictConfig,
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

        loss_term_configs = deepcopy(mode_cfg.losses)

        self.loss_calculators = dict(
            [
                (
                    loss_term_name,
                    [
                        (
                            params.get("weight", 1.0),
                            getattr(LossModules, params.type)(
                                cf, mode_cfg, stage, self.device, **params.loss_fcts
                            ),
                        )
                    ],
                )
                for loss_term_name, params in loss_term_configs.items()
            ]
        )

    def compute_loss(
        self,
        preds: ModelOutput,
        targets_and_aux: TargetAuxOutput,
        metadata: dict,
    ):
        losses_all = defaultdict(dict)
        stddev_all = defaultdict(dict)
        loss = torch.tensor(0.0, requires_grad=True)
        for loss_term_name, calc_term in self.loss_calculators.items():
            target = targets_and_aux[loss_term_name]
            for weight, calculator in calc_term:
                if weight > 0.0:
                    loss_values = calculator.compute_loss(
                        preds=preds, targets=target, metadata=metadata
                    )
                    loss = loss + weight * loss_values.loss
                    losses_all[calculator.name] = loss_values.losses_all
                    losses_all[calculator.name]["loss_avg"] = loss_values.loss
                    stddev_all[calculator.name] = loss_values.stddev_all

        # Keep histories for logging
        self.loss_hist += [loss.detach()]
        self.losses_unweighted_hist += [losses_all]
        self.stddev_unweighted_hist += [stddev_all]

        return loss
