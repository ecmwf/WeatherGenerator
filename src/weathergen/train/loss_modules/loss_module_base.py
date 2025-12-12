# ruff: noqa: T201

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses
from abc import abstractmethod
from collections import defaultdict

from torch import Tensor

from weathergen.common.config import Config
from weathergen.utils.train_logger import Stage

type StreamName = str


@dataclasses.dataclass
class LossValues:
    """
    A dataclass to encapsulate the loss components returned by each loss module.

    This provides a structured way to return the primary loss used for optimization,
    along with detailed per-stream/per-channel/per-loss-function losses for logging,
    and standard deviations for ensemble scenarios.
    """

    # The primary scalar loss value for optimization.
    loss: Tensor
    # Dictionaries containing loss values for each stream, channel, and loss function, as
    # well as standard deviations when operating with ensembles (e.g., when training with CRPS).
    losses_all: defaultdict
    stddev_all: defaultdict


class LossModuleBase:
    def __init__(self):
        """
        Base class for loss modules.

        Args:
            cf: The OmegaConf DictConfig object containing model and training configurations.
                It should specify 'loss_fcts' for training and 'loss_fcts_val' for validation.
            stage: The current operational stage, either TRAIN or VAL.
                   This dictates which set of loss functions (training or validation) will be used.
            device: The computation device, such as 'cpu' or 'cuda:0', where tensors will reside.
        """
        self.cf: Config | None = None
        self.stage: Stage
        self.loss_fcts = []

    @abstractmethod
    def compute_loss(
        self,
        preds: dict,
        targets: dict,
    ) -> LossValues:
        """
        Computes loss given predictions and targets and returns values of LossValues dataclass.
        """

        raise NotImplementedError()
