# pylint: disable=bad-builtin
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

import numpy as np
import torch
from omegaconf import DictConfig

import weathergen.train.loss_modules.loss_functions as loss_fns
from weathergen.common.io import physical_output_name
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.train.utils import TRAIN, VAL, Stage

_logger = logging.getLogger(__name__)


def get_num_samples(config) -> np.typing.NDArray:
    """
    Get number of samples in source/target config
    """
    return np.array([s_cfg.get("num_samples", 1) for _, s_cfg in config.items()])


class DynamicLossEMA:
    """
    Tracks and applies dynamic channel weights using an Exponential Moving Average (EMA)
    of inverse MSE, as described in Samudra 2.
    """

    def __init__(self, cfg: dict | None, streams_cfg: dict, device: str):
        self.enabled = cfg is not None
        if self.enabled:
            self.window = cfg.get("window", 100)
            self.L = cfg.get("L", 20.0)
            self.channel_weights_ema = {}
            for stream_name, stream_info in streams_cfg.items():
                num_channels = len(stream_info.train_target_channels)
                self.channel_weights_ema[stream_name] = torch.ones(num_channels, device=device)

    def get_weights(
        self, stream_name: str, weights_channels_static: torch.Tensor | None
    ) -> torch.Tensor | None:
        if not self.enabled:
            return None

        ema = self.channel_weights_ema[stream_name]
        if ema.numel() > 0:
            l_min = ema.min().clamp(min=1e-6)
            # Clamp max weight to L * min weight as per Samudra 2 paper
            clamped_ema = ema.clamp(max=self.L * l_min)
            # Normalize so mean is 1.0 to preserve overall learning rate scale
            weights_channels = clamped_ema / clamped_ema.mean()
        else:
            weights_channels = ema.clone()

        if weights_channels_static is not None and weights_channels_static.numel() > 0:
            weights_channels = weights_channels * weights_channels_static

        return weights_channels

    def update(self, stream_name: str, loss_lfct_chs: torch.Tensor):
        if not self.enabled:
            return

        with torch.no_grad():
            mse_per_chan = loss_lfct_chs.detach().clamp(min=1e-6)
            inv_mse = 1.0 / mse_per_chan
            self.channel_weights_ema[stream_name] = (
                1.0 - 1.0 / self.window
            ) * self.channel_weights_ema[stream_name] + (1.0 / self.window) * inv_mse


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
        mode_cfg: DictConfig,
        stage: Stage,
        device: str,
        **loss_fcts,
    ):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.mode_cfg = mode_cfg
        self.stage = stage
        self.device = device
        self.name = "LossPhysical"

        # Dynamic Loss state (extract it before parsing the actual loss functions)
        self.dynamic_loss_cfg = loss_fcts.get("dynamic_loss")
        self.forecast_offset = self.mode_cfg.forecast.offset

        # dynamically load loss functions based on configuration and stage
        self.loss_fcts = [
            [
                getattr(loss_fns, name),
                params.get("weight", 1.0),
                name,
            ]
            for name, params in loss_fcts.items()
            if name != "dynamic_loss"
        ]

        self.dynamic_loss_ema = DynamicLossEMA(
            self.dynamic_loss_cfg if self.stage == TRAIN else None,
            {
                physical_output_name(decoder, stream): self.cf.streams[stream]
                for decoder, settings in self.cf.decoders.items()
                for stream in settings.streams
            },
            self.device,
        )

    def _get_weights(self, stream_name, stream_info):
        """
        Get weights for current stream
        """

        device = self.device

        # Determine stream and channel loss weights based on the current stage
        if self.stage == TRAIN:
            # set loss_weights to 1. when not specified
            stream_info_loss_weight = stream_info.get("loss_weight", 1.0)
            weights_channels_static = (
                torch.tensor(stream_info["target_channel_weights"]).to(
                    device=device, non_blocking=True
                )
                if stream_info.get("target_channel_weights")
                else None
            )
        elif self.stage == VAL:
            # in validation mode, always unweighted loss
            stream_info_loss_weight = 1.0
            weights_channels_static = None

        if self.dynamic_loss_ema.enabled:
            weights_channels = self.dynamic_loss_ema.get_weights(
                stream_name, weights_channels_static
            )
        else:
            weights_channels = (
                weights_channels_static
                if weights_channels_static is None or weights_channels_static.numel() > 0
                else None
            )

        return stream_info_loss_weight, weights_channels

    def _get_output_step_weights(self, len_forecast_steps):
        timestep_weight_config = self.mode_cfg.get("forecast", {}).get("timestep_weight", {})
        if len(timestep_weight_config) == 0:
            return [1.0 for _ in range(len_forecast_steps)]
        weights_timestep_fct = getattr(loss_fns, list(timestep_weight_config.keys())[0])
        decay_factor = list(timestep_weight_config.values())[0]["decay_factor"]
        return weights_timestep_fct(len_forecast_steps, decay_factor)

    def _get_location_weights(self, stream_info, target_coords, substep_masks):
        location_weight_type = stream_info.get("location_weight", None)
        if location_weight_type is None:
            return [None for _ in substep_masks]

        target_coords = target_coords.to(self.device, non_blocking=True)
        weights_locations_fct = getattr(loss_fns, location_weight_type)
        weights_locations = [weights_locations_fct(target_coords[mask]) for mask in substep_masks]

        return weights_locations

    def _get_substep_masks(self, stream_info, output_step, target_times):
        """
        Find substeps and create corresponding masks (reused across loss functions)
        """

        tok_spacetime = stream_info.get("tokenize_spacetime", None)
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
        weights_locations: list[torch.Tensor],
    ):
        """
        Compute loss for given loss function
        """

        loss_lfct = torch.tensor(0.0, device=target.device, requires_grad=True)
        losses_chs = torch.zeros(target.shape[-1], device=target.device, dtype=torch.float32)

        ctr_substeps = 0
        for i_t, mask_t in enumerate(substep_masks):
            assert (
                mask_t.sum() == len(weights_locations[i_t])
                if weights_locations[i_t] is not None
                else True
            )

            loss, loss_chs = loss_fct(
                target[mask_t], pred[:, mask_t], weights_channels, weights_locations[i_t]
            )

            # accumulate loss
            loss_lfct = loss_lfct + loss
            losses_chs = losses_chs + loss_chs.detach() if len(loss_chs) > 0 else losses_chs
            ctr_substeps += 1

        # normalize over forecast steps in window
        losses_chs /= ctr_substeps if ctr_substeps > 0 else 1.0

        # TODO: substep weight
        loss_lfct = loss_lfct / (ctr_substeps if ctr_substeps > 0 else 1.0)

        return loss_lfct, losses_chs

    def compute_loss(self, preds: dict, targets: dict, metadata) -> LossValues:
        """Average independent decoder-stream losses against selected native target rows.

        Detailed metrics retain the same ``decoder::stream`` identifier as saved output.
        Each decoder owns its dynamic channel weights, including when streams overlap.
        """
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        output_info = metadata[1]
        step_weights = self._get_output_step_weights(len(targets.output_idxs))
        if len(targets.physical) > len(targets.output_idxs):
            step_weights.insert(0, None)
        losses_all = {}
        output_count = 0

        for decoder_name, settings in self.cf.decoders.items():
            for stream_name in settings.streams:
                stream_info = self.cf.streams[stream_name]
                output_name = physical_output_name(decoder_name, stream_name)
                target_channels = (
                    stream_info.val_target_channels
                    if self.stage == VAL
                    else stream_info.train_target_channels
                )
                stream_weight, channel_weights = self._get_weights(output_name, stream_info)
                metrics = defaultdict(lambda: defaultdict(dict))
                losses_all[output_name] = metrics
                if self.dynamic_loss_ema.enabled and channel_weights is not None:
                    for channel, weight in zip(target_channels, channel_weights, strict=True):
                        metrics["mse_ema_weight"][channel][str(self.forecast_offset)] = weight

                stream_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                step_count = 0
                for step, (pred_step, target_step) in enumerate(
                    zip(preds.physical, targets.physical, strict=True)
                ):
                    predictions = pred_step.get(decoder_name, {}).get(stream_name, ())
                    if not predictions or step_weights[step] is None:
                        continue
                    target_data = target_step[stream_name]
                    step_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    sample_count = 0
                    step_metrics = defaultdict(list)
                    for prediction, pred_params in zip(predictions, output_info, strict=True):
                        target_idx_native = pred_params.global_params.get("correspondence", -1)
                        matches = [
                            i
                            for i, info in enumerate(target_data["target_metda_data"])
                            if info[stream_name].global_params["idx"] == target_idx_native
                        ]
                        if not matches:
                            continue
                        assert len(matches) == 1
                        target_idx = matches[0]
                        if target_data["is_spoof"][target_idx] or not prediction.indices.numel():
                            continue
                        indices = prediction.indices
                        native_target = target_data["target"][target_idx]
                        target = native_target[indices.to(native_target.device)]
                        coords = target_data["target_coords"][target_idx]
                        coords = coords[indices.to(coords.device)]
                        times = target_data["target_times"][target_idx][
                            indices.detach().cpu().numpy()
                        ]
                        pred = prediction.values
                        assert pred.shape[1:] == target.shape
                        masks = self._get_substep_masks(stream_info, step, times)
                        location_weights = self._get_location_weights(stream_info, coords, masks)
                        sample_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        active = False
                        for loss_fct, weight, name in self.loss_fcts:
                            if name not in pred_params.global_params["loss"]:
                                continue
                            value, channel_values = self._loss_per_loss_function(
                                loss_fct, target, pred, masks, channel_weights, location_weights
                            )
                            sample_loss = sample_loss + weight * value
                            step_metrics[name].append(channel_values)
                            active = True
                            if (
                                self.dynamic_loss_ema.enabled
                                and step == self.forecast_offset
                                and name == "mse"
                            ):
                                self.dynamic_loss_ema.update(output_name, channel_values)
                        if active:
                            step_loss = step_loss + sample_loss
                            sample_count += 1
                    if sample_count:
                        stream_loss = stream_loss + step_weights[step] * step_loss / sample_count
                        step_count += 1
                    for name, values in step_metrics.items():
                        for channel, value in zip(
                            target_channels, torch.stack(values).mean(0), strict=True
                        ):
                            metrics[name][channel][str(step)] = value
                if step_count:
                    loss = loss + stream_weight * stream_loss / step_count
                    output_count += 1
                for channel_metrics in metrics.values():
                    values = [
                        value for steps in channel_metrics.values() for value in steps.values()
                    ]
                    channel_metrics["avg"] = torch.stack(values).mean()

        if output_count:
            loss = loss / output_count
        else:
            _logger.warning("No non-empty decoder targets contributed to physical loss.")
        return LossValues(loss=loss, losses_all=losses_all, stddev_all=None)
