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
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.train.utils import TRAIN, VAL, Stage

_logger = logging.getLogger(__name__)

VALID_TIME_AGGREGATIONS = frozenset({"diff", "mean", "min", "max"})


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
        self._warned_time_agg_mismatches = set()

        # Dynamic Loss state (extract it before parsing the actual loss functions)
        self.dynamic_loss_cfg = loss_fcts.get("dynamic_loss")
        self.forecast_offset = self.mode_cfg.forecast.offset

        self.time_aggregation_types = {
            name: self._parse_time_aggregation_types(name, params)
            for name, params in loss_fcts.items()
            if name != "dynamic_loss"
        }
        self.with_time_aggregation = any(self.time_aggregation_types.values())

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
            self.cf.streams,
            self.device,
        )

    @staticmethod
    def _parse_time_aggregation_types(loss_name: str, params: dict) -> list[str]:
        time_aggregation_types = params.get("time_aggregation_types", [])
        invalid = sorted(set(time_aggregation_types) - VALID_TIME_AGGREGATIONS)
        if invalid:
            raise ValueError(
                f"Unsupported time aggregation types for loss '{loss_name}': {invalid}. "
                f"Supported values are {sorted(VALID_TIME_AGGREGATIONS)}."
            )
        return list(time_aggregation_types)

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
    def _nan_reduce(values: torch.Tensor, op_name: str, dim: int) -> torch.Tensor:
        if op_name == "mean":
            return torch.nanmean(values, dim=dim)

        if op_name == "min":
            mask_nan = torch.isnan(values)
            reduced = torch.amin(torch.where(mask_nan, torch.inf, values), dim=dim)
            return torch.where(mask_nan.all(dim=dim), torch.full_like(reduced, torch.nan), reduced)

        if op_name == "max":
            mask_nan = torch.isnan(values)
            reduced = torch.amax(torch.where(mask_nan, -torch.inf, values), dim=dim)
            return torch.where(mask_nan.all(dim=dim), torch.full_like(reduced, torch.nan), reduced)

        raise ValueError(f"Unsupported time aggregation operation: {op_name}")

    def _compute_time_aggregate_loss(
        self,
        loss_fct,
        agg_op: str,
        pred_time: torch.Tensor,
        target_time: torch.Tensor,
        weights_channels: torch.Tensor | None,
        weights_locations: list[torch.Tensor | None],
        step_weights: list[float],
    ):
        if agg_op == "diff":
            if target_time.shape[0] < 2:
                return None

            loss_agg = torch.tensor(0.0, device=target_time.device, requires_grad=True)
            losses_chs = torch.zeros(target_time.shape[-1], device=target_time.device)
            ctr_steps = 0
            pred_diff = pred_time[:, 1:] - pred_time[:, :-1]
            target_diff = target_time[1:] - target_time[:-1]

            for step_idx in range(target_diff.shape[0]):
                loss_step, loss_step_chs = loss_fct(
                    target_diff[step_idx],
                    pred_diff[:, step_idx],
                    weights_channels,
                    weights_locations[step_idx],
                )
                step_weight = step_weights[step_idx] if step_idx < len(step_weights) else 1.0
                loss_agg = loss_agg + step_weight * loss_step
                losses_chs = losses_chs + step_weight * loss_step_chs.detach()
                ctr_steps += 1 if loss_step > 0.0 else 0

            if ctr_steps == 0:
                return None

            return loss_agg / ctr_steps, losses_chs / ctr_steps

        pred_agg = self._nan_reduce(pred_time, agg_op, dim=1)
        target_agg = self._nan_reduce(target_time, agg_op, dim=0)
        weights_locations_agg = next((weights for weights in weights_locations if weights is not None), None)
        return loss_fct(target_agg, pred_agg, weights_channels, weights_locations_agg)

    def _can_time_aggregate(self, stream_name: str, loss_name: str, entries: list[dict]) -> bool:
        if len(entries) < 2:
            return False

        target_shapes = {tuple(entry["target"].shape) for entry in entries}
        pred_shapes = {tuple(entry["pred"].shape) for entry in entries}
        if len(target_shapes) == 1 and len(pred_shapes) == 1:
            return True

        warn_key = (stream_name, loss_name)
        if warn_key not in self._warned_time_agg_mismatches:
            _logger.warning(
                "Skipping time aggregation for stream '%s' and loss '%s' because forecast "
                "steps do not share a consistent tensor shape.",
                stream_name,
                loss_name,
            )
            self._warned_time_agg_mismatches.add(warn_key)

        return False

    @staticmethod
    def _get_shared_time_agg_location_weights(weights_locations):
        if not weights_locations:
            return None

        first_weights_location = weights_locations[0]
        if not any(weight is not None for weight in weights_locations):
            return first_weights_location

        if first_weights_location is None:
            return None

        if any(
            weight is None or not torch.equal(weight, first_weights_location)
            for weight in weights_locations
        ):
            return None

        return first_weights_location

    def _record_time_agg_candidate(
        self,
        time_agg_records,
        stream_name: str,
        correspondence_idx: int,
        loss_fct_name: str,
        timestep_idx: int,
        target: torch.Tensor,
        pred: torch.Tensor,
        weights_channels: torch.Tensor | None,
        weights_locations,
        output_step_weight,
        loss_fct,
        loss_fct_weight: float,
    ):
        time_aggregation_types = self.time_aggregation_types.get(loss_fct_name, [])
        if not time_aggregation_types:
            return

        record_key = (stream_name, correspondence_idx, loss_fct_name)
        time_agg_records[record_key].append(
            {
                "timestep_idx": timestep_idx,
                "target": target,
                "pred": pred,
                "weights_channels": weights_channels,
                "weights_locations": self._get_shared_time_agg_location_weights(
                    weights_locations
                ),
                "output_step_weight": (
                    float(output_step_weight) if output_step_weight is not None else 1.0
                ),
                "loss_fct": loss_fct,
                "loss_fct_weight": loss_fct_weight,
                "time_aggregation_types": time_aggregation_types,
            }
        )

    def _apply_time_aggregate_losses(
        self,
        stream_name: str,
        target_channels,
        losses_all,
        time_agg_records,
    ):
        if not self.with_time_aggregation:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        aggregate_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        aggregate_terms = set()
        for (agg_stream_name, _, loss_name), entries in time_agg_records.items():
            if agg_stream_name != stream_name:
                continue

            entries = sorted(entries, key=lambda entry: entry["timestep_idx"])
            if not self._can_time_aggregate(stream_name, loss_name, entries):
                continue

            pred_time = torch.stack([entry["pred"] for entry in entries], dim=1)
            target_time = torch.stack([entry["target"] for entry in entries], dim=0)
            weights_locations = [entry["weights_locations"] for entry in entries]
            step_weights = [entry["output_step_weight"] for entry in entries[:-1]]

            for agg_op in entries[0]["time_aggregation_types"]:
                agg_result = self._compute_time_aggregate_loss(
                    entries[0]["loss_fct"],
                    agg_op,
                    pred_time,
                    target_time,
                    entries[0]["weights_channels"],
                    weights_locations,
                    step_weights,
                )
                if agg_result is None:
                    continue

                agg_loss, agg_loss_chs = agg_result
                aggregate_loss = aggregate_loss + entries[0]["loss_fct_weight"] * agg_loss
                aggregate_terms.add(agg_op)

                agg_loss_name = f"{loss_name}_time_{agg_op}"
                losses_all[stream_name]["aggregate"][agg_loss_name] = defaultdict(dict)
                for ch_n, v in zip(target_channels, agg_loss_chs, strict=True):
                    losses_all[stream_name]["aggregate"][agg_loss_name][ch_n] = v

        return aggregate_loss, len(aggregate_terms)

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
            ctr_substeps += 1 if loss > 0.0 else 0

        # normalize over forecast steps in window
        losses_chs /= ctr_substeps if ctr_substeps > 0 else 1.0

        # TODO: substep weight
        loss_lfct = loss_lfct / (ctr_substeps if ctr_substeps > 0 else 1.0)

        return loss_lfct, losses_chs

    def compute_loss(self, preds: dict, targets: dict, metadata) -> LossValues:
        """
        Computes the total loss for a given batch of predictions and corresponding
        stream data.

        The computed loss is:

        Mean_{stream}( Mean_{output_steps}( Mean_{loss_fcts}( loss_fct( target, pred, weigths) )))

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
        losses_all = defaultdict(dict)

        source2target_idxs, output_info, target2source_idxs, target_info = metadata

        # TODO: iterate over batch dimension
        for stream_name, stream_info in self.cf.streams.items():
            # TODO: avoid this
            target_channels = (
                stream_info.val_target_channels
                if self.stage == "val"
                else stream_info.train_target_channels
            )

            losses_all[stream_name] = defaultdict(dict)

            stream_loss_weight, weights_channels = self._get_weights(stream_name, stream_info)
            if self.dynamic_loss_ema.enabled and weights_channels is not None:
                losses_all[stream_name][str(self.forecast_offset)]["mse_ema_weight"] = {}
                for ch_n, w in zip(target_channels, weights_channels, strict=True):
                    losses_all[stream_name][str(self.forecast_offset)]["mse_ema_weight"][ch_n] = (
                        w.item()
                    )

            # TODO: make nicer
            output_step_loss_weights = self._get_output_step_weights(len(targets.output_idxs))
            if len(targets.physical) - len(targets.output_idxs) > 0:
                output_step_loss_weights.insert(0, None)

            # loss_stream: loss for given stream
            loss_stream = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_timesteps = 0
            time_agg_records = defaultdict(list) if self.with_time_aggregation else None
            for timestep_idx, (preds_cur, target_cur) in enumerate(
                zip(preds.physical, targets.physical, strict=True)
            ):
                preds_batch = preds_cur.get(stream_name, [])
                if not preds_batch:
                    # skip to next timestep if preds of current timestep are empty
                    continue

                targets_batch = target_cur[stream_name]["target"]
                targets_coords_batch = target_cur[stream_name]["target_coords"]
                targets_times_batch = target_cur[stream_name]["target_times"]
                targets_params = target_cur[stream_name]["target_metda_data"]
                targets_is_spoof = target_cur[stream_name]["is_spoof"]

                output_step_weight = output_step_loss_weights[timestep_idx]

                # loss_timestep: loss for given timestep
                loss_timestep = torch.tensor(0.0, device=self.device, requires_grad=True)
                ctr_batch = 0
                for pred, pred_params in zip(preds_batch, output_info, strict=True):
                    # source has a unique target but index is not invariant with multiple
                    # target_aux calculators
                    target_idx_native = pred_params.global_params.get("correspondence", -1)
                    target_idx = [
                        i
                        for i, t in enumerate(targets_params)
                        if t[stream_name].global_params["idx"] == target_idx_native
                    ]
                    # source/model_input has no target for physical loss
                    if len(target_idx) == 0:
                        continue
                    # source -> target correspondence has to be unique
                    assert len(target_idx) == 1
                    target_idx = target_idx[0]

                    # current target data
                    target = targets_batch[target_idx]
                    target_times = targets_times_batch[target_idx]

                    # get masks for sub-time steps
                    substep_masks = self._get_substep_masks(stream_info, timestep_idx, target_times)

                    # get weights for locations
                    weights_locations = self._get_location_weights(
                        stream_info, targets_coords_batch[target_idx], substep_masks
                    )

                    # loss_st_corr: loss for give source-target correspondence
                    loss_st_corr = torch.tensor(0.0, device=self.device, requires_grad=True)
                    ctr_loss_fcts = 0
                    for loss_fct, loss_fct_weight, loss_fct_name in self.loss_fcts:
                        # skip is loss is not computed for this sample
                        if loss_fct_name not in pred_params.global_params["loss"]:
                            continue

                        # spoofed inputs are masked in the output calculations
                        is_spoof = targets_is_spoof[target_idx]
                        sw = 0.0 if is_spoof else 1.0
                        spoof_weight = torch.tensor(sw, device=self.device, requires_grad=False)

                        # skip if either target or prediction has no data points
                        if not (target.shape[0] > 0 and pred.shape[0] > 0):
                            continue

                        # reshape prediction tensor to match target's dimensions: extract
                        # data/coords and remove token dimension if it exists.
                        # expected shape of pred is [ensemble_size, num_samples, num_channels].
                        pred = pred.reshape([pred.shape[0], *target.shape])
                        assert pred.shape[1] > 0

                        losses_all[stream_name][str(timestep_idx)][loss_fct_name] = defaultdict(
                            dict
                        )
                        # loss_lfct: loss for given loss function aggregated over all channels
                        # loss_lfct_chs: loss for given loss function per channel
                        loss_lfct, loss_lfct_chs = self._loss_per_loss_function(
                            loss_fct,
                            target,
                            pred,
                            substep_masks,
                            weights_channels,
                            weights_locations,
                        )

                        for ch_n, v in zip(target_channels, loss_lfct_chs, strict=True):
                            losses_all[stream_name][str(timestep_idx)][loss_fct_name][ch_n] = (
                                spoof_weight * v if v != 0.0 and not is_spoof else torch.nan
                            )

                        # Update EMA for dynamic loss if enabled
                        if (
                            self.dynamic_loss_ema.enabled
                            and timestep_idx == self.forecast_offset
                            and loss_fct_name == "mse"
                            and not is_spoof
                        ):
                            self.dynamic_loss_ema.update(stream_name, loss_lfct_chs)

                        # Add the weighted and normalized loss from this loss function to the total
                        # batch loss
                        loss_cur_w = spoof_weight * loss_fct_weight * loss_lfct * output_step_weight
                        loss_st_corr = loss_st_corr + loss_cur_w
                        ctr_loss_fcts += 1 if (loss_cur_w > 0.0 and not is_spoof) else 0

                        if self.with_time_aggregation and not is_spoof:
                            self._record_time_agg_candidate(
                                time_agg_records,
                                stream_name,
                                pred_params.global_params.get("correspondence", -1),
                                loss_fct_name,
                                timestep_idx,
                                target,
                                pred,
                                weights_channels,
                                weights_locations,
                                output_step_weight,
                                loss_fct,
                                loss_fct_weight,
                            )

                    loss_timestep = loss_timestep + loss_st_corr
                    ctr_batch += 1 if ctr_loss_fcts > 0.0 else 0

                loss_stream = loss_stream + loss_timestep
                ctr_timesteps += 1 if ctr_batch > 0 else 0

            aggregate_terms = 0
            if self.with_time_aggregation:
                aggregate_loss, aggregate_terms = self._apply_time_aggregate_losses(
                    stream_name,
                    target_channels,
                    losses_all,
                    time_agg_records,
                )
                loss_stream = loss_stream + aggregate_loss

            denom = ctr_timesteps + aggregate_terms
            denom = denom if denom > 0 else 1.0
            loss = loss + (stream_loss_weight * loss_stream) / denom

            ctr_streams += 1 if ctr_timesteps > 0 else 0

        # normalize by all targets and forecast steps that were non-empty
        # (with each having an expected loss of 1 for an uninitalized neural net)
        if loss == 0.0:
            _logger.warning(
                "Loss is 0.0, likely incorrect configuration. Check stream"
                " support time and training configuration."
            )
        loss = loss / ctr_streams if ctr_streams > 0 else loss

        def _nested_dict():
            return defaultdict(dict)

        # Reorder losses_all to [stream_name][loss_fct_name][ch_n][output_step]
        reordered_losses = defaultdict(dict)
        for stream_name, output_step_dict in losses_all.items():
            reordered_losses[stream_name] = defaultdict(_nested_dict)
            for output_step, lfct_dict in output_step_dict.items():
                for loss_fct_name, ch_dict in lfct_dict.items():
                    for ch_n, v in ch_dict.items():
                        reordered_losses[stream_name][loss_fct_name][ch_n][output_step] = v

        # Calculate per stream, per lfct average across channels and output_steps
        for stream_name, lfct_dict in reordered_losses.items():
            for loss_fct_name, ch_dict in lfct_dict.items():
                reordered_losses[stream_name][loss_fct_name]["avg"] = 0
                count = 0
                for ch_n, output_step_dict in ch_dict.items():
                    if ch_n != "avg":
                        for _, v in output_step_dict.items():
                            v = 0.0 if type(v) is float and np.isnan(v) else v
                            reordered_losses[stream_name][loss_fct_name]["avg"] += v
                            count += 1
                reordered_losses[stream_name][loss_fct_name]["avg"] /= count

        # Return all computed loss components encapsulated in a ModelLoss dataclass
        return LossValues(loss=loss, losses_all=reordered_losses, stddev_all=None)
