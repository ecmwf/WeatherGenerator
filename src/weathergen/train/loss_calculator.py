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
class ModelLoss:
    # Resolve types and add descriptions
    loss: list
    losses_all: list
    stddev_all: list


class LossCalculator:
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

    @staticmethod
    def _construct_masks(target_times_raw: np.array, mask_nan: Tensor, tok_spacetime: bool):
        """
        Construct a list of masks for intermediate time steps within one forecast step. Only applies for specific datasets.
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
    def _compute_loss_with_mask(target, pred, mask, i_ch, loss_fct, ens):
        # only compute loss if there are non-NaN values
        if mask.sum().item() > 0:
            return loss_fct(
                target[mask, i_ch],
                pred[:, mask, i_ch],
                pred[:, mask, i_ch].mean(0),
                (pred[:, mask, i_ch].std(0) if ens else torch.zeros(1, device=pred.device)),
            )
        else:
            return 0

    def compute_loss(
        self,
        preds: Tensor,
        streams_data: None,  # TODO: determine type
    ) -> ModelLoss:
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

        i_batch = 0  # TODO: Iterate over batch dimension here in future
        for i_strm, strm in enumerate(self.cf.streams):
            targets = streams_data[i_batch][i_strm].target_tokens[self.cf.forecast_offset :]
            assert len(targets) == self.cf.forecast_offset + self.cf.forecast_steps, (
                "Length of targets does not match number of forecast_steps."
            )
            for fstep, target in enumerate(targets):
                pred = preds[fstep][i_strm]
                # Only compute loss if target and prediction exists
                if not (target.shape[0] > 0 and pred.shape[0] > 0):
                    continue

                num_channels = len(strm[str(self.stage) + "_target_channels"])

                if self.stage == TRAIN:
                    # set loss_weights to 1. when not specified
                    obs_loss_weight = strm["loss_weight"] if "loss_weight" in strm else 1.0
                    channel_loss_weight = (
                        strm["channel_weight"]
                        if "channel_weight" in strm
                        else np.ones(num_channels)
                    )
                elif self.stage == VAL:
                    # in validation mode, always unweighted loss is computed
                    obs_loss_weight = 1.0
                    channel_loss_weight = np.ones(num_channels)

                # extract data/coords and remove token dimension if it exists. Shape of pred is
                # [ensemble, target_points, target_channels]
                pred = pred.reshape([pred.shape[0], *target.shape])
                assert pred.shape[1] > 0

                mask_nan = ~torch.isnan(target)
                if pred[:, mask_nan].shape[1] == 0:
                    continue
                ens = pred.shape[0] > 1

                tok_spacetime = (
                    strm["tokenize_spacetime"] if "tokenize_spacetime" in strm else False
                )
                # accumulate loss from different loss functions and channels
                for i_lfct, (loss_fct, w) in enumerate(self.loss_fcts):
                    # compute per channel loss
                    val = torch.tensor(0.0, device=self.device, requires_grad=True)
                    ctr_chs = 0.0

                    # Loop over all channels, construct masks and compute loss accordingly
                    for i_ch in range(target.shape[-1]):
                        masks = self._construct_masks(
                            target_times_raw=streams_data[i_batch][i_strm].target_times_raw[
                                self.cf.forecast_offset + fstep
                            ],
                            mask_nan=mask_nan[:, i_ch],
                            tok_spacetime=tok_spacetime,
                        )
                        for mask in masks:
                            temp = self._compute_loss_with_mask(
                                target=target,
                                pred=pred,
                                mask=mask,
                                i_ch=i_ch,
                                loss_fct=loss_fct,
                                ens=ens,
                            )
                            val = val + channel_loss_weight[i_ch] * temp  # Channel weighting
                            # TODO: Add latitude weighting
                            losses_all[strm.name][i_ch, i_lfct] += temp.item()
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

        return ModelLoss(loss=loss, losses_all=losses_all, stddev_all=stddev_all)
