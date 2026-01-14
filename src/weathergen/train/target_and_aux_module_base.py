# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class TargetAuxOutput:
    """
    A dataclass to encapsulate the TargetAndAuxCalculator output and give a clear API.
    """

    num_forecast_steps: int

    physical: dict[str, torch.Tensor]
    latent: dict[str, torch.Tensor]
    aux_outputs: dict[str, torch.Tensor]


class TargetAndAuxModuleBase:
    def __init__(self, cf, model, **kwargs):
        pass

    def reset(self):
        pass

    def update_state_pre_backward(self, istep, batch, model, **kwargs) -> None:
        pass

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        pass

    def compute(self, istep, batch, *args, **kwargs) -> TargetAuxOutput:
        pass

    def to_device(self, device) -> TargetAndAuxModuleBase:
        return self


class PhysicalTargetAndAux(TargetAndAuxModuleBase):
    def __init__(self, cf, model, **kwargs):
        return

    def reset(self):
        return

    def update_state_pre_backward(self, istep, batch, model, **kwargs):
        return

    def update_state_post_opt_step(self, istep, batch, model, **kwargs):
        return

    def compute(self, istep, batch, *args, **kwargs) -> TargetAuxOutput:
        # TODO: properly retrieve/define these
        stream_names = [k for k, _ in batch.samples[0].streams_data.items()]
        forecast_steps = batch.get_forecast_steps()

        # collect all targets, concatenating across batch dimension since this is also how it
        # happens for predictions in the model
        targets = {}
        for stream_name in stream_names:
            # collect targets for all forecast steps
            targets[stream_name] = []
            for fstep in range(forecast_steps):
                targets_cur, target_times_cur, target_coords_cur, meta_data = [], [], [], []
                is_spoof = []
                for sample in batch.samples:
                    targets_cur += [sample.streams_data[stream_name].target_tokens[fstep]]
                    target_times_cur += [sample.streams_data[stream_name].target_times_raw[fstep]]
                    target_coords_cur += [sample.streams_data[stream_name].target_coords_raw[fstep]]
                    meta_data += [sample.meta_info]
                    is_spoof += [sample.streams_data[stream_name].is_spoof()]

                targets[stream_name].append(
                    {
                        "target": targets_cur,
                        "target_times": target_times_cur,
                        "target_coords": target_coords_cur,
                        "target_metda_data": meta_data,
                        "is_spoof": is_spoof,
                    }
                )

        aux_outputs = {}
        return TargetAuxOutput(forecast_steps, targets, None, aux_outputs)

    def to_device(self, device) -> PhysicalTargetAndAux:
        return self
