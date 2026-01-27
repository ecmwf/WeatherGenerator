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

from weathergen.model.engines import LatentState

type StreamName = str


@dataclasses.dataclass
class TargetAuxOutput:
    """
    A dataclass to encapsulate the TargetAndAuxCalculator output and give a clear API.
    """

    output_idxs: list[int]

    physical: list[dict[StreamName, torch.Tensor]]
    latent: list[dict[str, torch.Tensor | LatentState]]
    aux_outputs: dict[str, torch.Tensor]

    def __init__(self, len_target: int, output_idxs: list) -> None:
        self.output_idxs = output_idxs
        self.physical = [{} for _ in range(len_target)]
        self.latent = [{} for _ in range(len_target)]
        self.aux_outputs = {}

    def add_physical_target(
        self, timestep_idx: int, stream_name: StreamName, pred: torch.Tensor
    ) -> None:
        self.physical[timestep_idx][stream_name] = pred

    def add_latent_target(self, timestep_idx: int, latent_name: str, pred: torch.Tensor) -> None:
        self.latent[timestep_idx][latent_name] = pred

    def get_physical_target(
        self,
        timestep_idx: int,
        stream_name: StreamName | None = None,
        sample_idx: int | None = None,
    ):
        pred = self.physical[timestep_idx]
        if stream_name is not None:
            pred = pred.get(stream_name, None)
            if sample_idx is not None:
                assert sample_idx < len(pred), "Invalid sample index."
                pred = pred[sample_idx]
        return pred

    def get_latent_target(self, timestep_idx: int):
        return self.latent[timestep_idx]


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

    def compute(self, bidx, batch, model_params, model) -> TargetAuxOutput:
        # TODO: properly retrieve/define these
        stream_names = [k for k, _ in batch.samples[0].streams_data.items()]
        output_idxs = batch.get_output_idxs()
        assert len(output_idxs) > 0

        targets = TargetAuxOutput(batch.get_output_len(), output_idxs)

        # collect all targets, concatenating across batch dimension since this is also how it
        # happens for predictions in the model
        for stream_name in stream_names:
            # collect targets for all forecast steps
            for step in output_idxs:
                targets_cur, target_times_cur, target_coords_cur, meta_data = [], [], [], []
                is_spoof = []
                for sample in batch.samples:
                    targets_cur += [sample.streams_data[stream_name].target_tokens[step]]
                    target_times_cur += [sample.streams_data[stream_name].target_times_raw[step]]
                    target_coords_cur += [sample.streams_data[stream_name].target_coords_raw[step]]
                    meta_data += [sample.meta_info]
                    is_spoof += [sample.streams_data[stream_name].is_spoof()]

                targets_step = {
                    "target": targets_cur,
                    "target_times": target_times_cur,
                    "target_coords": target_coords_cur,
                    "target_metda_data": meta_data,
                    "is_spoof": is_spoof,
                }

                targets.add_physical_target(step, stream_name, targets_step)

        return targets

    def to_device(self, device) -> PhysicalTargetAndAux:
        return self
