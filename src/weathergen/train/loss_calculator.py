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
    
    losses_all_lat: Tensor


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
        self.loss_fcts = [
            [getattr(losses, name if name != "mse" else "mse_channel_location_weighted"), w]
            for name, w in loss_fcts
        ]
        
        loss_fcts_lat = cf.get("latent_loss_fcts") if stage == TRAIN else cf.get("latent_loss_fcts_val")
        if loss_fcts_lat:
            self.loss_fcts_lat = [
                [getattr(losses, name), w]
                for name, w in loss_fcts_lat
            ]
        else:
            self.loss_fcts_lat = []


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
                torch.tensor(stream_info["channel_weight"]).to(device=device, non_blocking=True)
                if "channel_weight" in stream_info
                else None
            )
        elif self.stage == VAL:
            # in validation mode, always unweighted loss
            stream_info_loss_weight = 1.0
            weights_channels = None

        return stream_info_loss_weight, weights_channels

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
        stream_info,
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
            assert mask_t.sum() == len(weights_locations) if weights_locations else True

            loss, loss_chs = loss_fct(
                target[mask_t], pred[:, mask_t], weights_channels, weights_locations
            )

            # accumulate loss
            loss_lfct = loss_lfct + loss
            losses_chs += loss_chs.detach()
            ctr_substeps += 1 if loss > 0.0 else 0

        # normalize over forecast steps in window
        losses_chs /= ctr_substeps if ctr_substeps > 0 else 1.0

        # TODO: substep weight
        loss_lfct = loss_lfct / (ctr_substeps if ctr_substeps > 0 else 1.0)

        return loss_lfct, losses_chs
    
    def _loss_per_loss_function_lat(
        loss_fct,
        stream_info,
        target: torch.Tensor,
        pred: torch.Tensor,
    ):
        """
        Compute loss for given loss function
        """

        loss_val = loss_fct(target=target, ens=None, mu=pred)

        return loss_val

    def compute_loss(
        self,
        preds: list[list[Tensor]],
        streams_data: list[list[any]],
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
        
        losses_all_lat: Tensor= torch.zeros(
                len(self.loss_fcts_lat),
                device=self.device,
            )
        

        if self.cf.get("encode_targets_latent", False):
            # unpack the predictions/tokens from the latent space if the latent space tokens are encoded
            preds, tokens_all, tokens_targets = preds
                
        # TODO: iterate over batch dimension
        i_batch = 0
        for i_stream_info, stream_info in enumerate(self.cf.streams):
            
            # 1. First go through the losses in the physical space
            
            # extract target tokens for current stream from the specified forecast offset onwards
            targets = streams_data[i_batch][i_stream_info].target_tokens[self.cf.forecast_offset :]

            stream_data = streams_data[i_batch][i_stream_info]

            # TODO: set from stream info
            weights_locations = None

            loss_fsteps = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_fsteps = 0
            for fstep, target in enumerate(targets):
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

                # get masks for sub-time steps
                substep_masks = self._get_substep_masks(stream_info, fstep, stream_data)

                # accumulate loss from different loss functions
                loss_fstep = torch.tensor(0.0, device=self.device, requires_grad=True)
                ctr_loss_fcts = 0
                for i_lfct, (loss_fct, loss_fct_weight) in enumerate(self.loss_fcts):
                    # loss for current loss function
                    loss_lfct, loss_lfct_chs = LossCalculator._loss_per_loss_function(
                        loss_fct,
                        stream_info,
                        target,
                        pred,
                        substep_masks,
                        weights_channels,
                        weights_locations,
                    )
                    losses_all[stream_info.name][:, i_lfct] += loss_lfct_chs

                    # Add the weighted and normalized loss from this loss function to the total
                    # batch loss
                    loss_fstep = loss_fstep + (loss_fct_weight * loss_lfct * stream_loss_weight)
                    ctr_loss_fcts += 1 if loss_lfct > 0.0 else 0

                loss_fsteps = loss_fsteps + (loss_fstep / ctr_loss_fcts if ctr_loss_fcts > 0 else 0)
                ctr_fsteps += 1 if ctr_loss_fcts > 0 else 0
                
        
                      
            loss = loss + (loss_fsteps / (ctr_fsteps if ctr_fsteps > 0 else 1.0))
            ctr_streams += 1 if ctr_fsteps > 0 else 0
            

            # normalize by forecast step
            losses_all[stream_info.name] /= ctr_fsteps if ctr_fsteps > 0 else 1.0
            stddev_all[stream_info.name] /= ctr_fsteps if ctr_fsteps > 0 else 1.0

            # replace channels without information by nan to exclude from further computations
            losses_all[stream_info.name][losses_all[stream_info.name] == 0.0] = torch.nan
            stddev_all[stream_info.name][stddev_all[stream_info.name] == 0.0] = torch.nan
            
        
        # 2. Now go through the losses in the latent space, if applicable
        if self.loss_fcts_lat:
            loss_fsteps_lat = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_fsteps_lat = 0
            # TODO: KCT, do we need the below per fstep?
            for fstep in range(1, len(tokens_all)): # the first entry in tokens_all is the source itself, so skip it
                loss_fstep = torch.tensor(0.0, device=self.device, requires_grad=True)
                ctr_loss_fcts = 0
                # if forecast_offset==0, then the timepoints correspond. Otherwise targets don't encode the source timestep, so we don't need to skip
                fstep_targs = fstep if self.cf.forecast_offset == 0 else fstep -1
                for i_lfct, (loss_fct, loss_fct_weight) in enumerate(self.loss_fcts_lat):
                    loss_lfct = LossCalculator._loss_per_loss_function_lat(
                        loss_fct,
                        stream_info=None,
                        target=tokens_all[fstep],
                        pred=tokens_targets[fstep_targs]
                    )
                    
                    losses_all_lat[i_lfct] += loss_lfct # TODO: break into fsteps
                    
                    # Add the weighted and normalized loss from this loss function to the total
                    # batch loss
                    loss_fstep = loss_fstep + (loss_fct_weight * loss_lfct)
                    ctr_loss_fcts += 1 if loss_lfct > 0.0 else 0
                    
                loss_fsteps_lat = loss_fsteps_lat + (loss_fstep / ctr_loss_fcts if ctr_loss_fcts > 0 else 0)
                ctr_fsteps_lat += 1 if ctr_loss_fcts > 0 else 0
                
            loss = loss + (loss_fsteps_lat / (ctr_fsteps_lat if ctr_fsteps_lat > 0 else 1.0))
            ctr_streams =  ctr_streams  if ctr_streams > 0 else 1 # TODO: KCT, check the logic here
            
            losses_all_lat /= ctr_fsteps_lat if ctr_fsteps_lat > 0 else 1.0
            losses_all_lat[losses_all_lat == 0.0] = torch.nan

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
        return LossValues(loss=loss, losses_all=losses_all, stddev_all=stddev_all, losses_all_lat=losses_all_lat)
