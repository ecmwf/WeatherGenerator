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

import torch
from torch import Tensor

from weathergen.common.config import Config
from weathergen.utils.train_logger import Stage


@dataclasses.dataclass
class LossValues:
    """
    A dataclass to encapsulate the various loss components computed by the LossCalculator.

    This provides a structured way to return the primary loss used for optimization,
    along with detailed per-stream/per-channel/per-loss-function losses for logging,
    and standard deviations for ensemble scenarios.
    """

    # The primary scalar loss value for optimization.
    loss: Tensor
    # Dictionaries containing detailed loss values for each stream, channel, and loss function, as
    # well as standard deviations when operating with ensembles (e.g., when training with CRPS).
    losses_all: dict[str, Tensor]
    stddev_all: dict[str, Tensor]


class LossCalculatorBase:
    def __init__(self):
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
        self.cf: Config | None = None
        self.stage: Stage
        self.loss_fcts = []

    @staticmethod
    def _loss_per_loss_function(
        loss_fct,
        target: torch.Tensor,
        pred: torch.Tensor,
        substep_masks: list[torch.Tensor],
        weights_channels: torch.Tensor,
        weights_locations: torch.Tensor,
    ):
        """
        Compute loss for given loss function
        """

        loss_lfct = torch.tensor(0.0, device=target.device, requires_grad=True)
        losses_chs = torch.zeros(target.shape[-1], device=target.device, dtype=torch.float32)

        ctr_substeps = 0
        for mask_t in substep_masks:
            assert mask_t.sum() == len(weights_locations) if weights_locations is not None else True

            loss, loss_chs = loss_fct(
                target[mask_t], pred[:, mask_t], weights_channels, weights_locations
            )

            # accumulate loss
            loss_lfct = loss_lfct + loss
            losses_chs = losses_chs + loss_chs.detach() if len(loss_chs) > 0 else losses_chs
            ctr_substeps += 1 if loss > 0.0 else 0

        # normalize over forecast steps in window
        losses_chs /= ctr_substeps if ctr_substeps > 0 else 1.0

        # TODO: substep weight
        loss_lfct = loss_lfct / (ctr_substeps if ctr_substeps > 0 else 1.0)

        return loss_lfct, losses_chs

    # def _get_weights(self, stream_info):

    # def _update_weights(self, stream_info):
