"""
Data structures for student-teacher multi-view training.

Provides clean separation between:
  - Model data (StreamData objects containing tensors)
  - View metadata (spatial masks, strategies, relationships)
"""

from dataclasses import dataclass

import numpy as np
import torch

from weathergen.common.config import Config
from weathergen.datasets.stream_data import StreamData

# TODO: Add a store for a random number for diffusion
# TODO: GetTimestep to get the timestep
# TODO: GetMetaData: then this gets the right rn for the timestep!

@dataclass
class SampleMetaData:
    # masking strategy
    # masking_strategy: str

    # parameters for masking strategy
    masking_params: Config | dict

    mask: torch.Tensor | None = None

    noise_level_rn: float | None = None

class Sample:
    # keys: stream name, values: SampleMetaData
    meta_info: dict

    # data for all streams
    # keys: stream_name, values: StreamData
    streams_data: dict[str, StreamData | None]
    forecast_dt: int | None

    # TODO:
    # these two need to live in ModelBatch as they are flattened!
    # this should be a dict also lives in ModelBatch
    source_cell_lens: list[torch.Tensor] | None
    target_coords_idx: list[torch.Tensor] | None

    def __init__(self, streams: dict) -> None:
        # TODO: can we pass this right away?
        self.meta_info = {}

        self.streams_data = {}
        for stream_info in streams:
            self.streams_data[stream_info["name"]] = None

        self.source_cell_lens: list[torch.Tensor] | None = None
        self.target_coords_idx: list[torch.Tensor] | None = None

        self.forecast_dt: int | None = None

    def add_stream_data(self, stream_name: str, stream_data: StreamData) -> None:
        """
        Add data for stream @stream_name to sample
        """
        assert self.streams_data.get(stream_name, -1) != -1, "stream name does not exist"
        self.streams_data[stream_name] = stream_data

    def add_meta_info(self, stream_name: str, meta_info: SampleMetaData) -> None:
        """
        Add metadata for stream @stream_name to sample
        """
        self.meta_info[stream_name] = meta_info

    def set_preprocessed(self, source_cell_lens, target_coords_idx):
        """
        Set preprocessed data for sample
        """
        self.source_cell_lens = source_cell_lens
        self.target_coords_idx = target_coords_idx

    def set_forecast_dt(self, forecast_dt: int) -> None:
        """
        Set forecast_dt for sample
        """
        self.forecast_dt = forecast_dt

    # TODO: complete interface, e.g get_stream

    def get_stream_data(self, stream_name: str) -> StreamData:
        """
        Get data for stream @stream_name from sample
        """
        assert self.streams_data.get(stream_name, -1) != -1, "stream name does not exist"
        return self.streams_data[stream_name]

class ModelBatch:
    """
    Container for all data and metadata for one training batch.
    """

    # source samples (for model)
    source_samples: list[Sample]

    # target samples (for TargetAuxCalculator)
    target_samples: list[Sample]

    # index of corresponding target (for source samples) or source (for target samples)
    # these are in 1-to-1 corresponding for classical training modes (MTM, forecasting) but
    # can be more complex for strategies like student-teacher training
    source2target_matching_idxs: np.typing.NDArray[np.int32]
    target2source_matching_idxs: np.typing.NDArray[np.int32]

    def __init__(self, streams, num_source_samples: int, num_target_samples: int) -> None:
        """ """

        self.source_samples = [Sample(streams) for _ in range(num_source_samples)]
        self.target_samples = [Sample(streams) for _ in range(num_target_samples)]

        self.source2target_matching_idxs = np.full(num_source_samples, -1, dtype=np.int32)
        # self.target_source_matching_idxs = np.full(num_target_samples, -1, dtype=np.int32)
        self.target2source_matching_idxs = [[] for _ in range(num_target_samples)]

    def add_source_stream(
        self,
        source_sample_idx: int,
        target_sample_idx: int,
        stream_name: str,
        stream_data: StreamData,
        source_meta_info: SampleMetaData,
    ) -> None:
        """
        Add data for one stream to sample @source_sample_idx
        """
        self.source_samples[source_sample_idx].add_stream_data(stream_name, stream_data)

        # add the meta_info
        self.source_samples[source_sample_idx].add_meta_info(stream_name, source_meta_info)
        

        assert target_sample_idx < len(self.target_samples), "invalid value for target_sample_idx"
        self.source2target_matching_idxs[source_sample_idx] = target_sample_idx

    def add_target_stream(
        self,
        target_sample_idx: int,
        source_sample_idx: int | list[int],
        stream_name: str,
        stream_data: StreamData,
        target_meta_info: SampleMetaData,
    ) -> None:
        """
        Add data for one stream to sample @target_sample_idx
        """
        self.target_samples[target_sample_idx].add_stream_data(stream_name, stream_data)

        # add the meta_info -- for target we have different
        self.target_samples[target_sample_idx].add_meta_info(stream_name, target_meta_info)

        if isinstance(source_sample_idx, int):
            assert source_sample_idx < len(self.source_samples), "invalid value for source_sample_idx"
        else:
            assert all(idx < len(self.source_samples) for idx in source_sample_idx), "invalid value for source_sample_idx"
        self.target2source_matching_idxs[target_sample_idx] = source_sample_idx

    def len_sources(self) -> int:
        """
        Number of source samples
        """
        return len(self.source_samples)

    def len_targets(self) -> int:
        """
        Number of target samples
        """
        return len(self.target_samples)

    def get_source_sample(self, idx: int) -> Sample:
        """
        Get a source sample
        """
        return self.source_samples[idx]

    def get_target_sample(self, idx: int) -> Sample:
        """
        Get a target sample
        """
        return self.target_samples[idx]

    def get_source_idx_for_target(self, target_idx: int) -> int:
        """
        Get index of source sample for a given target sample index
        """
        return int(self.target2source_matching_idxs[target_idx])

    def get_target_idx_for_source(self, source_idx: int) -> int:
        """
        Get index of target sample for a given source sample index
        """
        return int(self.source2target_matching_idxs[source_idx])
