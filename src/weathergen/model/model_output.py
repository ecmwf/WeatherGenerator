# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from weathergen.datasets.batch import BatchSamples
from weathergen.model.engines import LatentState

type StreamName = str


class ModelOutput:
    """
    Representation of model output
    """

    physical: list[dict[StreamName, torch.Tensor]]
    latent: list[dict[str, torch.Tensor | LatentState]]
    latent_deep: list[dict[str, list[torch.Tensor]]] | None

    def __init__(
        self,
        forecast_steps: list[int],
        forecast_offset: int,
        source_samples: BatchSamples,
    ) -> None:
        self.forecast_offset = forecast_offset
        # the first chunk keeps its leading forecast_offset steps as empty slots, so that
        # concatenating the chunks of a rollout stays indexed by global forecast step
        base = 0 if forecast_steps[0] == forecast_offset else forecast_steps[0]
        self.forecast_steps = list(range(base, forecast_steps[-1] + 1))

        self.physical: list[dict[StreamName, torch.Tensor]] = [{} for _ in self.forecast_steps]
        self.latent: list[dict[str, torch.Tensor | LatentState]] = [{} for _ in self.forecast_steps]
        self.batch_samples = source_samples
        self.latent_deep = None

    def chunk_idx(self, fstep: int) -> int:
        """Index of forecast step fstep into chunk-local data, e.g. predictions."""
        return fstep - self.forecast_steps[0]

    def fstep_idx(self, fstep: int) -> int:
        """Index of forecast step fstep into batch-global data, e.g. target coordinates."""
        return fstep

    def add_physical_prediction(
        self, fstep: int, stream_name: StreamName, pred: torch.Tensor
    ) -> None:
        self.physical[fstep][stream_name] = pred

    def add_latent_prediction(self, fstep: int, latent_name: str, pred: torch.Tensor) -> None:
        self.latent[fstep][latent_name] = pred

    def add_deep_latent_prediction(
        self, fstep: int, name: str, level_preds: list[torch.Tensor]
    ) -> None:
        if self.latent_deep is None:
            self.latent_deep = [{} for _ in range(len(self.physical))]
        self.latent_deep[fstep][name] = level_preds

    def get_physical_prediction(
        self, fstep: int, stream_name: StreamName | None = None, sample_idx: int | None = None
    ):
        pred = self.physical[fstep]
        if stream_name is not None:
            pred = pred.get(stream_name, None)
            if sample_idx is not None:
                assert sample_idx < len(pred), "Invalid sample index."
                pred = pred[sample_idx]
        return pred

    def get_latent_prediction(self, fstep: int):
        return self.latent[fstep]
