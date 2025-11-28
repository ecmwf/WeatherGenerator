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
import torch
from omegaconf import DictConfig
from torch import Tensor

import weathergen.train.loss_modules.loss_functions as losses
from weathergen.train.loss_modules.loss_functions import stat_loss_fcts
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import TRAIN, VAL, Stage

_logger = logging.getLogger(__name__)


class LossPhysical(LossModuleBase):
    """
    Manages and computes the overall loss for a WeatherGenerator model during
    training and validation stages.

    This class handles the initialization and application of various loss functions,
    applies channel-specific weights, constructs masks for missing data, and
    aggregates losses across different data streams, channels, and forecast steps.
    It provides both the main loss for backpropagation and detailed loss metrics for logging.
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
        self.name = "LossPhysical"

        # Dynamically load loss functions based on configuration and stage
        self.loss_fcts = [
            [getattr(losses, name if name != "mse" else "mse_channel_location_weighted"), w, name]
            for name, w in loss_fcts
        ]

    def _get_weights(self, stream_info):
        """
        Get weights for current stream
        """

        device = self.device

        # Determine stream and channel loss weights based on the current stage
        if self.stage == TRAIN:
            # set loss_weights to 1. when not specified
            stream_info_loss_weight = stream_info.get("loss_weight", 1.0)
            weights_channels = (
                torch.tensor(stream_info["target_channel_weights"]).to(
                    device=device, non_blocking=True
                )
                if "target_channel_weights" in stream_info
                else None
            )
        elif self.stage == VAL:
            # in validation mode, always unweighted loss
            stream_info_loss_weight = 1.0
            weights_channels = None

        return stream_info_loss_weight, weights_channels

    def _get_fstep_weights(self, forecast_steps):
        timestep_weight_config = self.cf.get("timestep_weight")
        if timestep_weight_config is None:
            return [1.0 for _ in range(forecast_steps)]
        weights_timestep_fct = getattr(losses, timestep_weight_config[0])
        return weights_timestep_fct(forecast_steps, timestep_weight_config[1])

    def _get_location_weights(self, stream_info, stream_data, forecast_offset, fstep):
        location_weight_type = stream_info.get("location_weight", None)
        if location_weight_type is None:
            return None
        weights_locations_fct = getattr(losses, location_weight_type)
        weights_locations = weights_locations_fct(stream_data, forecast_offset, fstep)
        weights_locations = weights_locations.to(device=self.device, non_blocking=True)

        return weights_locations

    def _get_substep_masks(self, stream_info, fstep, stream_data):
        """
        Find substeps and create corresponding masks (reused across loss functions)
        """

        tok_spacetime = stream_info.get("tokenize_spacetime", None)
        target_times = stream_data.target_times_raw[self.cf.forecast_offset + fstep]
        target_times_unique = np.unique(target_times) if tok_spacetime else [target_times]
        substep_masks = []
        for t in target_times_unique:
            # find substep
            mask_t = torch.tensor(t == target_times).to(self.device, non_blocking=True)
            substep_masks.append(mask_t)

        return substep_masks

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

    def compute_loss(
        self,
        preds: dict,
        targets: dict,
    ) -> LossValues:
        """
        Computes the total loss for a given batch of predictions and corresponding
        stream data.

        The computed loss is:

        Mean_{stream}( Mean_{fsteps}( Mean_{loss_fcts}( loss_fct( target, pred, weigths) )))

        This method orchestrates the calculation of the overall loss by iterating through
        different data streams, forecast steps, channels, and configured loss functions.
        It applies weighting, handles NaN values through masking, and accumulates
        detailed loss metrics for logging.

        Args:
            preds: A nested list of prediction tensors. The outer list represents forecast steps,
                   the inner list represents streams. Each tensor contains predictions for that
                   step and stream.
            streams_data: A nested list representing the input batch data. The outer list is for
                          batch items, the inner list for streams. Each element provides an object
                          (e.g., dataclass instance) containing target data and metadata.

        Returns:
            A ModelLoss dataclass instance containing:
            - loss: The loss for back-propagation.
            - losses_all: A dictionary mapping stream names to a tensor of per-channel and
                          per-loss-function losses, normalized by non-empty targets/forecast steps.
            - stddev_all: A dictionary mapping stream names to a tensor of mean standard deviations
                          of predictions for channels with statistical loss functions, normalized.
        """

        preds = preds.physical
        streams_data = targets.physical

        # gradient loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        # counter for non-empty targets
        ctr_streams = 0

        # initialize dictionaries for detailed loss tracking and standard deviation statistics
        # create tensor for each stream
        losses_all: dict[str, Tensor] = {
            f"{self.name}.{st.name}.{loss_fct_name}": torch.zeros(
                (len(st[str(self.stage) + "_target_channels"])),
                device=self.device,
            )
            for st in self.cf.streams
            for _, _, loss_fct_name in self.loss_fcts
        }
        stddev_all: dict[str, Tensor] = {
            f"{self.name}.{st.name}.{loss_fct_name}": torch.zeros(
                len(stat_loss_fcts), device=self.device
            )
            for st in self.cf.streams
            for _, _, loss_fct_name in self.loss_fcts
        }

        # TODO: iterate over batch dimension
        i_batch = 0
        for i_stream_info, stream_info in enumerate(self.cf.streams):
            # extract target tokens for current stream from the specified forecast offset onwards
            targets = streams_data[i_batch][i_stream_info].target_tokens[self.cf.forecast_offset :]

            stream_data = streams_data[i_batch][i_stream_info]

            fstep_loss_weights = self._get_fstep_weights(len(targets))

            loss_fsteps = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_fsteps = 0

            stream_is_spoof = streams_data[i_batch][i_stream_info].is_spoof()
            if stream_is_spoof:
                spoof_weight = torch.tensor(0.0, device=self.device, requires_grad=False)
            else:
                spoof_weight = torch.tensor(1.0, device=self.device, requires_grad=False)

            for fstep, (target, fstep_weight) in enumerate(
                zip(targets, fstep_loss_weights, strict=False)
            ):
                # skip if either target or prediction has no data points
                pred = preds[fstep][i_stream_info]
                if not (target.shape[0] > 0 and pred.shape[0] > 0):
                    continue

                # reshape prediction tensor to match target's dimensions: extract data/coords and
                # remove token dimension if it exists.
                # expected final shape of pred is [ensemble_size, num_samples, num_channels].
                pred = pred.reshape([pred.shape[0], *target.shape])
                assert pred.shape[1] > 0

                # get weigths for current streams
                stream_loss_weight, weights_channels = self._get_weights(stream_info)

                # get weights for locations
                weights_locations = self._get_location_weights(
                    stream_info, stream_data, self.cf.forecast_offset, fstep
                )

                # get masks for sub-time steps
                substep_masks = self._get_substep_masks(stream_info, fstep, stream_data)

                # accumulate loss from different loss functions
                loss_fstep = torch.tensor(0.0, device=self.device, requires_grad=True)
                ctr_loss_fcts = 0
                for loss_fct, loss_fct_weight, loss_fct_name in self.loss_fcts:
                    # loss for current loss function
                    loss_lfct, loss_lfct_chs = self._loss_per_loss_function(
                        loss_fct,
                        target,
                        pred,
                        substep_masks,
                        weights_channels,
                        weights_locations,
                    )
                    losses_all[f"{self.name}.{stream_info.name}.{loss_fct_name}"] += (
                        spoof_weight * loss_lfct_chs
                    )

                    # Add the weighted and normalized loss from this loss function to the total
                    # batch loss
                    loss_fstep = loss_fstep + (
                        loss_fct_weight * loss_lfct * stream_loss_weight * fstep_weight
                    )
                    ctr_loss_fcts += 1 if loss_lfct > 0.0 else 0

                loss_fsteps = loss_fsteps + (loss_fstep / ctr_loss_fcts if ctr_loss_fcts > 0 else 0)
                ctr_fsteps += 1 if ctr_loss_fcts > 0 else 0

            loss = loss + ((spoof_weight * loss_fsteps) / (ctr_fsteps if ctr_fsteps > 0 else 1.0))
            ctr_streams += 1 if ctr_fsteps > 0 and not stream_is_spoof else 0

            # normalize by forecast step
            if ctr_fsteps > 0:
                for _, _, loss_fct_name in self.loss_fcts:
                    losses_all[f"{self.name}.{stream_info.name}.{loss_fct_name}"] /= ctr_fsteps
                    stddev_all[f"{self.name}.{stream_info.name}.{loss_fct_name}"] /= ctr_fsteps

            # replace channels without information by nan to exclude from further computations
            for _, _, loss_fct_name in self.loss_fcts:
                key = f"{self.name}.{stream_info.name}.{loss_fct_name}"
                losses_all[key][losses_all[key] == 0.0] = torch.nan
                stddev_all[key][stddev_all[key] == 0.0] = torch.nan

        # normalize by all targets and forecast steps that were non-empty
        # (with each having an expected loss of 1 for an uninitalized neural net)
        loss = loss / ctr_streams

        # Return all computed loss components encapsulated in a ModelLoss dataclass
        return LossValues(loss=loss, losses_all=losses_all, stddev_all=stddev_all)
