"""
Data structures for student-teacher multi-view training.

Provides clean separation between:
  - Model data (StreamData objects containing tensors)
  - View metadata (spatial masks, strategies, relationships)
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch

from weathergen.common.config import Config
from weathergen.datasets.stream_data import StreamData


@dataclass
class SampleMetaData:
    # sample parameters (masking)
    params: Config | dict

    mask: torch.Tensor | None = None

    global_params: dict | None = None


class Sample:
    # keys: stream name, values: SampleMetaData
    meta_info: dict[str | SampleMetaData]

    # data for all streams
    # keys: stream_name, values: StreamData
    streams_data: dict[str, StreamData | None]

    def __init__(self, streams: dict) -> None:
        self.meta_info = {}

        self.streams_data = {}
        for stream_info in streams:
            self.streams_data[stream_info["name"]] = None

    def to_device(self, device) -> None:
        for key in self.meta_info.keys():
            self.meta_info[key].mask = (
                self.meta_info[key].mask.to(device, non_blocking=True)
                if self.meta_info[key].mask is not None
                else None
            )

        for key, val in self.streams_data.items():
            if val is not None:
                self.streams_data[key] = val.to_device(device)

    def is_empty(self) -> bool:
        """
        Check if sample is empty
        """
        return np.all(
            np.array([s.empty() if s is not None else True for _, s in self.streams_data.items()])
        )

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

    def get_stream_data(self, stream_name: str) -> StreamData:
        """
        Get data for stream @stream_name from sample
        """
        assert self.streams_data.get(stream_name, -1) != -1, "stream name does not exist"
        return self.streams_data[stream_name]

    def get_forecast_steps(self) -> int:
        for _, sdata in self.streams_data.items():
            forecast_dt = sdata.get_forecast_steps()
        return forecast_dt


class BatchSamples:
    """
    Container for source or target samples
    """

    samples: list[Sample]
    tokens_lens: torch.Tensor | None
    device: str | None

    def __init__(self, streams: dict, num_samples: int) -> None:
        self.samples = [Sample(streams) for _ in range(num_samples)]
        self.tokens_lens = None
        self.device = None

    def __len__(self) -> int:
        return len(self.samples)

    def to_device(self, device):
        for sample in self.samples:
            sample.to_device(device)

        self.tokens_lens = (
            self.tokens_lens.to(device, non_blocking=True) if self.tokens_lens is not None else None
        )

        self.device = device

        return self

    def get_samples(self) -> list[Sample]:
        return self.samples

    def get_subset(self, subset: list | None = None):
        if subset is None:
            return self
        else:
            assert len(list(set(subset))) == len(subset), "subset contains duplicates"
            # create copy and then select subset for samples and tokens_lens
            bs = copy.deepcopy(self)
            bs.samples = [bs.samples[i] for i in subset]
            torch_idxs = torch.tensor(subset, dtype=torch.long, device=bs.tokens_lens.device)
            bs.tokens_lens = torch.index_select(bs.tokens_lens, 1, torch_idxs)
            return bs

    def get_num_steps(self) -> int:
        """
        Get number of input/source steps from smallest of all available streams
        """
        # TODO: define explicitly
        lens = [
            len(stream.source_tokens_cells) for _, stream in self.samples[0].streams_data.items()
        ]

        return min(lens)

    def get_forecast_steps(self) -> int:
        """
        Get forecast steps
        """
        # use sample 0 since the number of forecast steps is constant across batch
        # TODO: fix use of sample 0
        return self.samples[0].get_forecast_steps()

    def get_device(self) -> str | torch.device:
        """
        Get device of tensors in the batch
        """
        return self.device


class ModelBatch:
    """
    Container for all data and metadata for one training batch.
    """

    # source samples (for model)
    source_samples: BatchSamples

    # target samples (for TargetAuxCalculator)
    target_samples: BatchSamples

    # index of corresponding target (for source samples) or source (for target samples)
    # these are in 1-to-1 corresponding for classical training modes (e.g. MTM, forecasting) but
    # can be more complex for strategies like student-teacher training
    source2target_matching_idxs: np.typing.NDArray[np.int32]
    target2source_matching_idxs: np.typing.NDArray[np.int32]

    # device of the tensors in the batch
    device: str | torch.device

    def __init__(self, streams: dict, num_source_samples: int, num_target_samples: int) -> None:
        """ """

        self.source_samples = BatchSamples(streams, num_source_samples)
        self.target_samples = BatchSamples(streams, num_target_samples)

        self.source2target_matching_idxs = np.full(num_source_samples, -1, dtype=np.int32)
        self.target2source_matching_idxs = [[] for _ in range(num_target_samples)]

    def to_device(self, device):  # -> ModelBatch
        """
        Move batch to device
        """

        self.source_samples.to_device(device)
        self.target_samples.to_device(device)

        self.device = device

        return self

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
        self.source_samples.samples[source_sample_idx].add_stream_data(stream_name, stream_data)

        # add the meta_info
        self.source_samples.samples[source_sample_idx].add_meta_info(stream_name, source_meta_info)

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
        self.target_samples.samples[target_sample_idx].add_stream_data(stream_name, stream_data)

        # add the meta_info -- for target we have different
        self.target_samples.samples[target_sample_idx].add_meta_info(stream_name, target_meta_info)

        if isinstance(source_sample_idx, int):
            assert source_sample_idx < len(self.source_samples), (
                "invalid value for source_sample_idx"
            )
        else:
            assert all(idx < len(self.source_samples) for idx in source_sample_idx), (
                "invalid value for source_sample_idx"
            )
        self.target2source_matching_idxs[target_sample_idx] = source_sample_idx

    def is_empty(self):
        """
        Check if batch is empty
        """
        source_empty = np.all(
            np.array([s.is_empty() if s is not None else True for s in self.source_samples.samples])
        )
        target_empty = np.all(
            np.array([s.is_empty() if s is not None else True for s in self.target_samples.samples])
        )
        return source_empty or target_empty

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
        return self.source_samples.samples[idx]

    def get_source_samples(self, subset: list | None = None) -> BatchSamples:
        """
        Get source samples
        """
        return self.source_samples.get_subset(subset)

    def get_target_sample(self, idx: int) -> Sample:
        """
        Get a target sample
        """
        return self.target_samples.samples[idx]

    def get_target_samples(self, subset: list | None = None) -> BatchSamples:
        """
        Get target samples
        """
        return self.target_samples.get_subset(subset)

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

    def get_forecast_steps(self) -> int:
        """
        Get forecast steps
        """
        # use sample 0 since the number of forecast steps is constant across batch
        # TODO: fix use of sample 0
        return self.source_samples.get_forecast_steps()

    def get_device(self) -> str | torch.device:
        """
        Get device of tensors in the batch
        """
        return self.device

    def get_num_source_steps(self) -> int:
        """
        Get number of input/source steps from smallest of all available streams
        """
        # TODO: define explicitly
        lens = [
            len(stream.source_tokens_cells)
            for _, stream in self.target_samples.samples[0].streams_data.items()
        ]

        return min(lens)

    def get_num_target_steps(self) -> int:
        """
        Get number of input/source steps from smallest of all available streams
        """
        # TODO: define explicitly
        # TODO: ensure that num_input_steps is constant across batch with different strategies
        lens = [
            len(stream.target_tokens)
            for _, stream in self.target_samples.samples[0].streams_data.items()
        ]

        return min(lens)
