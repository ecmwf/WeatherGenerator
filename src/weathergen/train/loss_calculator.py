# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses
import logging

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

import weathergen.train.loss as losses
from weathergen.train.loss import stat_loss_fcts
from weathergen.utils.train_logger import TRAIN, VAL, Stage

_logger = logging.getLogger(__name__)


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


class LossCalculator:
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

        # Dynamically load loss functions based on configuration and stage
        loss_fcts = cf.loss_fcts if stage == TRAIN else cf.loss_fcts_val
        self.loss_fcts = [[getattr(losses, name), w] for name, w in loss_fcts]

    @staticmethod
    def _construct_masks(
        target_times_raw: np.array, mask_nan: Tensor, tok_spacetime: bool
    ) -> list[Tensor]:
        """
        Constructs a list of boolean masks for target data.

        If 'tok_spacetime' is enabled, masks are generated for unique intermediate time steps
        within a single forecast step and combined with a NaN mask. Otherwise, a single mask
        for non-NaN values is returned. This is useful for datasets where targets might have
        sub-timestep granularity.

        Args:
            target_times_raw: A NumPy array containing raw time values for targets
                              within a single forecast step.
            mask_nan: A PyTorch Tensor indicating non-NaN values for the specific channel.
            tok_spacetime: A boolean flag indicating whether spacetime tokenization is active,
                           which influences mask construction.

        Returns:
            A list of PyTorch boolean Tensors, where each tensor is a combined mask for
            a unique time point or simply the non-NaN mask.
        """
        masks = []
        if tok_spacetime:
            t_unique = np.unique(target_times_raw)
            for t in t_unique:
                mask_t = Tensor(t == target_times_raw).to(mask_nan)
                masks.append(torch.logical_and(mask_t, mask_nan))
        else:
            masks.append(mask_nan)
        return masks

    @staticmethod
    def _compute_loss_with_mask(
        target: Tensor, pred: Tensor, mask: np.array, i_ch: int, loss_fct: losses, ens: bool
    ) -> Tensor:
        """
        Computes the loss for a specific channel using a given mask.

        This helper function applies a chosen loss function to the masked target and prediction
        data for a single channel, handling ensemble predictions by calculating mean and standard
        deviation over the ensemble dimension.

        Args:
            target: The ground truth target tensor.
            pred: The prediction tensor, potentially with an ensemble dimension.
            mask: A boolean mask tensor, indicating which elements to consider for loss computation.
            i_ch: The index of the channel for which to compute the loss.
            loss_fct: The specific loss function to apply. It is expected to accept
                      (masked_target, masked_pred, pred_mean, pred_std).
            ens: A boolean flag indicating whether 'pred' contains an ensemble dimension.

        Returns:
            The computed loss value for the masked data, or a tensor with value 0 if no
            valid data points are present under the mask.
        """
        if mask.sum().item() > 0:
            # Only compute loss if there are non-NaN values
            return loss_fct(
                target[mask, i_ch],
                pred[:, mask, i_ch],
                pred[:, mask, i_ch].mean(0),
                (pred[:, mask, i_ch].std(0) if ens else torch.zeros(1, device=pred.device)),
            )
        else:
            # If no valid data under the mask, return 0 to avoid errors and not contribute to loss
            return torch.tensor(0.0, device=pred.device)

    def _compute_loss_per_loss_function(
        self,
        loss_fct,
        i_lfct,
        i_batch,
        i_strm,
        strm,
        fstep,
        streams_data,
        target,
        pred,
        mask_nan,
        channel_loss_weight,
        losses_all,
    ):
        tok_spacetime = strm["tokenize_spacetime"] if "tokenize_spacetime" in strm else False
        ens = pred.shape[0] > 1

        # compute per channel loss
        loss_lfct = torch.tensor(0.0, device=self.device, requires_grad=True)
        ctr_chs = 0

        # loop over all channels within the current stream and forecast step
        for i_ch in range(target.shape[-1]):
            # construct masks based on spacetime tokenization setting
            masks = self._construct_masks(
                target_times_raw=streams_data[i_batch][i_strm].target_times_raw[
                    self.cf.forecast_offset + fstep
                ],
                mask_nan=mask_nan[:, i_ch],
                tok_spacetime=tok_spacetime,
            )
            ctr_substeps = 0
            for mask in masks:
                loss_ch = self._compute_loss_with_mask(
                    target=target,
                    pred=pred,
                    mask=mask,
                    i_ch=i_ch,
                    loss_fct=loss_fct,
                    ens=ens,
                )
                # accumulate weighted loss for this loss function and channel
                loss_lfct = loss_lfct + (channel_loss_weight[i_ch] * loss_ch)
                ctr_chs += 1 if loss_ch > 0.0 else 0
                ctr_substeps += 1 if loss_ch > 0.0 else 0
                # for detailed logging
                losses_all[strm.name][i_ch, i_lfct] += loss_ch.item()

            # normalize over forecast steps in window
            losses_all[strm.name][i_ch, i_lfct] /= ctr_substeps if ctr_substeps > 0 else 0.0

        # normalize the accumulated loss for the current loss function
        loss_lfct = loss_lfct / ctr_chs if (ctr_chs > 0) else loss_lfct

        return loss_lfct, losses_all

    def compute_loss(
        self,
        preds: list[list[Tensor]],
        streams_data: list[
            list[any]
        ],  # Assuming Stream is a dataclass/object for each stream in a batch
    ) -> LossValues:
        """
        Computes the total loss for a given batch of predictions and corresponding
        stream data.

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

        # gradient loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        # counter for non-empty targets
        ctr_streams = 0

        # initialize dictionaries for detailed loss tracking and standard deviation statistics
        # create tensor for each stream
        losses_all: dict[str, Tensor] = {
            st.name: torch.zeros(
                (len(st[str(self.stage) + "_target_channels"]), len(self.loss_fcts)),
                device=self.device,
            )
            for st in self.cf.streams
        }
        stddev_all: dict[str, Tensor] = {
            st.name: torch.zeros(len(stat_loss_fcts), device=self.device) for st in self.cf.streams
        }

        # TODO: iterate over batch dimension
        i_batch = 0
        for i_strm, strm in enumerate(self.cf.streams):
            # extract target tokens for current stream from the specified forecast offset onwards
            targets = streams_data[i_batch][i_strm].target_tokens[self.cf.forecast_offset :]

            loss_fstep = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_fsteps = 0

            for fstep, target in enumerate(targets):
                # skip if either target or prediction has no data points
                pred = preds[fstep][i_strm]
                if not (target.shape[0] > 0 and pred.shape[0] > 0):
                    continue

                num_channels = len(strm[str(self.stage) + "_target_channels"])

                # Determine stream and channel loss weights based on the current stage
                if self.stage == TRAIN:
                    # set loss_weights to 1. when not specified
                    strm_loss_weight = strm["loss_weight"] if "loss_weight" in strm else 1.0
                    channel_loss_weight = (
                        strm["channel_weight"]
                        if "channel_weight" in strm
                        else np.ones(num_channels)
                    )
                elif self.stage == VAL:
                    # in validation mode, always unweighted loss
                    strm_loss_weight = 1.0
                    channel_loss_weight = np.ones(num_channels)

                # reshape prediction tensor to match target's dimensions: extract data/coords and
                # remove token dimension if it exists.
                # expected final shape of pred is [ensemble_size, num_samples, num_channels].
                pred = pred.reshape([pred.shape[0], *target.shape])
                assert pred.shape[1] > 0

                mask_nan = ~torch.isnan(target)
                # if all valid predictions are masked out by NaNs, skip this forecast step
                if pred[:, mask_nan].shape[1] == 0:
                    continue

                # accumulate loss from different loss functions and across channels
                for i_lfct, (loss_fct, loss_fct_weight) in enumerate(self.loss_fcts):
                    loss_lfct, losses_all = self._compute_loss_per_loss_function(
                        loss_fct,
                        i_lfct,
                        i_batch,
                        i_strm,
                        strm,
                        fstep,
                        streams_data,
                        target,
                        pred,
                        mask_nan,
                        channel_loss_weight,
                        losses_all,
                    )

                    # Update statistical deviation metrics if the current loss function is
                    # recognized as statistical
                    if loss_fct.__name__ in stat_loss_fcts:
                        indx = stat_loss_fcts.index(loss_fct.__name__)
                        stddev_all[strm.name][indx] += pred[:, mask_nan].std(0).mean().item()

                    # Add the weighted and normalized loss from this loss function to the total
                    # batch loss
                    loss_fstep = loss_fstep + (loss_fct_weight * loss_lfct * strm_loss_weight)
                    ctr_fsteps += 1 if loss_lfct > 0.0 else 0

            loss = loss + (loss_fstep / (ctr_fsteps if ctr_fsteps > 0 else 1.0))
            ctr_streams += 1 if ctr_fsteps > 0 else 0

            # normalize by forecast step
            losses_all[strm.name] /= ctr_fsteps if ctr_fsteps > 0 else 1.0
            stddev_all[strm.name] /= ctr_fsteps if ctr_fsteps > 0 else 1.0

            # replace channels without information by nan to exclude from further computations
            losses_all[strm.name][losses_all[strm.name] == 0.0] = torch.nan
            stddev_all[strm.name][stddev_all[strm.name] == 0.0] = torch.nan

        if loss == 0.0:
            # streams_data[i] are samples in batch
            # streams_data[i][0] is stream 0 (sample_idx is identical for all streams per sample)
            _logger.warning(
                f"Loss is 0.0 for sample(s): {[sd[0].sample_idx.item() for sd in streams_data]}."
                + "This will likely lead to errors in the optimization step."
            )

        # normalize by all targets and forecast steps that were non-empty
        # (with each having an expected loss of 1 for an uninitalized neural net)
        loss = loss / ctr_streams

        # Return all computed loss components encapsulated in a ModelLoss dataclass
        return LossValues(loss=loss, losses_all=losses_all, stddev_all=stddev_all)
