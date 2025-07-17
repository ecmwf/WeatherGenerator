# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


# loss_module = LossModule(
#     loss_fcts,
#     stage=TRAIN/VAL,
#     self.devices[0],
#     cf_streams=self.cf.streams
# )

# loss = loss_module.compute_loss(
#     preds=preds,
#     targets=targets,
#     stream_data=stream_data
# )

import logging

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

import weathergen.train.loss as losses
from weathergen.train.loss import stat_loss_fcts
from weathergen.utils.train_logger import TRAIN, VAL, Stage

_logger = logging.getLogger(__name__)


class LossModule:
    def __init__(
        self,
        cf: DictConfig,
        stage: Stage = TRAIN,
        device: str = "cpu",
    ):
        self.cf = cf
        self.stage = stage
        self.device = device

        loss_fcts = cf.loss_fcts if stage == TRAIN else cf.loss_fcts_val
        self.loss_fcts = [[getattr(losses, name), w] for name, w in loss_fcts]

        # self.stream_loss_weight = {} —> normalize
        # self.channel_loss_weight = {} —> normalize
    
    @staticmethod
    def _construct_masks(
        target_times_raw: np.array,
        mask_nan: Tensor
    ):
        """
        Construct a list of masks for intermediate time steps within one forecast step. Only applies for specific datasets.
        """
        masks = []
        t_unique = Tensor(np.unique(target_times_raw)).to(mask_nan)
        for t in t_unique:
            mask_t = t==target_times_raw
            masks.append(torch.logical_and(mask_t, mask_nan))
        return masks

    def compute_loss(
        self,
        preds: Tensor,
        targets: Tensor,
        targets_coords: Tensor,  # TODO: verify type
        streams_data: None,  # TODO: determine type
    ):
        # TODO: Rethink counters (ctr_ftarget and ctr_chs)
        ctr_ftarget = 0

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        # Create list storing losses for each stream
        losses_all: dict[str, Tensor] = {
            st.name: torch.zeros(
                (len(st[str(self.stage) + "_target_channels"]), len(self.loss_fcts)),
                device=self.device,
            )
            for st in self.cf.streams  # No nan here as it's divided so any remaining 0 become nan
        }  # Create tensor for each stream
        stddev_all: dict[str, Tensor] = {
            st.name: torch.zeros(len(stat_loss_fcts), device=self.device) for st in self.cf.streams
        }  # Create tensor for each stream
        # assert len(targets) == len(preds) and len(preds) == len(self.cf.streams)

        assert len(targets) == self.cf.forecast_steps + 1, "Length of targets does not match number of forecast_steps."
        for fstep in range(len(targets)):
            for i_strm, (target, strm) in enumerate(
                zip(targets[fstep], self.cf.streams, strict=False)
            ):
                pred = preds[fstep][i_strm]
                if not (target.shape[0] > 0 and pred.shape[0] > 0):
                    continue

                num_channels = len(strm[str(self.stage) + "_target_channels"])

                # set loss_weights to 1. when not specified
                obs_loss_weight = strm["loss_weight"] if "loss_weight" in strm else 1.0
                channel_loss_weight = (
                    strm["channel_weight"] if "channel_weight" in strm else np.ones(num_channels)
                )
                # in validation mode, always unweighted loss is computed
                obs_loss_weight = 1.0 if self.stage == VAL else obs_loss_weight
                channel_loss_weight = (
                    np.ones(num_channels) if self.stage == VAL else channel_loss_weight
                )

                # extract data/coords and remove token dimension if it exists
                pred = pred.reshape([pred.shape[0], *target.shape])
                assert pred.shape[1] > 0

                mask_nan = ~torch.isnan(target)
                if pred[:, mask_nan].shape[1] == 0:
                    continue
                ens = pred.shape[0] > 1

                tok_spacetime = strm["tokenize_spacetime"] if "tokenize_spacetime" in strm else False
                i_batch = 0 # TODO: Iterate over batch dimension here in future
                # accumulate loss from different loss functions and channels
                for j, (loss_fct, w) in enumerate(self.loss_fcts):
                    # compute per channel loss
                    val = torch.tensor(0.0, device=self.device, requires_grad=True)
                    # TODO: Rethink counters (ctr_ftarget and ctr_chs)
                    ctr_chs = 0.0



                    # TODO: Dataclass as new issue


                    # loop over all channels
                    for i in range(target.shape[-1]):

                        if tok_spacetime:
                            # if stream is internal time step, build masks separately per step
                            masks = self._construct_masks(
                                target_times_raw=streams_data[i_batch][i_strm].target_times_raw[fstep],
                                mask_nan=mask_nan[:, i]
                            )
                        else:
                            masks = [mask_nan[:, i]]
                        
                        # Compute the actual loss, apply loss weighting, and add it to the losses_all list
                        for mask in masks:
                            # only compute loss if there are non-NaN values
                            if mask.sum().item() > 0:
                                temp = loss_fct(
                                    target[mask, i],
                                    pred[:, mask, i],
                                    pred[:, mask, i].mean(0),
                                    (
                                        pred[:, mask, i].std(0)
                                        if ens
                                        else torch.zeros(1, device=pred.device)
                                    ),
                                )
                                # TODO: Implement separate loss weighting functionality over space, channels, time, etc.
                                val = val + channel_loss_weight[i] * temp
                                losses_all[strm.name][i, j] += temp.item()
                                ctr_chs += 1

                    val = val / ctr_chs if (ctr_chs > 0) else val

                    if loss_fct.__name__ in stat_loss_fcts:
                        indx = stat_loss_fcts.index(loss_fct.__name__)
                        stddev_all[strm.name][indx] += pred[:, mask_nan].std(0).mean().item()
                    # ignore NaNs so that training can continue even if one pred-net diverges
                    loss = loss + (
                        (w * val * obs_loss_weight)
                        if not torch.isnan(val)
                        else torch.tensor(0.0, requires_grad=True)
                    )
                ctr_ftarget += 1

        if loss == 0.0:
            # streams_data[i] are samples in batch
            # streams_data[i][0] is stream 0 (sample_idx is identical for all streams per sample)
            _logger.warning(
                f"Loss is 0.0 for sample(s): {[sd[0].sample_idx.item() for sd in streams_data]}."
                + "This will likely lead to errors in the optimization step."
            )

        # normalize by all targets and forecast steps that were non-empty
        # (with each having an expected loss of 1 for an uninitalized neural net)
        loss = loss / ctr_ftarget

        losses_all = {k: v / ctr_ftarget for k, v in losses_all.items()}
        stddev_all = {k: v / ctr_ftarget for k, v in stddev_all.items()}

        return loss, losses_all, stddev_all
