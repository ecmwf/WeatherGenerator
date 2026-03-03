"""
Data structures for student-teacher multi-view training.

Provides clean separation between:
  - Model data (StreamData objects containing tensors)
  - View metadata (spatial masks, strategies, relationships)
"""

from __future__ import annotations  # allow forward references in typehints

import copy
from dataclasses import dataclass

import numpy as np
import torch

from weathergen.common.config import Config
from weathergen.datasets.data_reader_base import DTRange
from weathergen.datasets.stream_data import StreamData


@dataclass
class SampleMetaData:
    # sample parameters (masking)
    params: Config | dict

    mask: torch.Tensor | None = None

    global_params: dict | None = None


class Sample:
    def __init__(self, streams: dict) -> None:
        self.meta_info: dict[str, SampleMetaData] = {}
        self.streams_data: dict[str, StreamData | None] = {}

        for stream_info in streams:
            self.streams_data[stream_info["name"]] = None

    @property
    def sample_idx(self) -> int:
        if not self.is_empty():
            # choose 'first' streams stream_data
            stream_data = next(iter(self.streams_data.values()))
            # TODO check: is reliable sinsce?
            # since set with tidx and not idx in MultiStreamDataSampler
            idx = stream_data.sample_idx

            assert all(
                stream_data.sample_idx == idx for stream_data in self.streams_data.values()
            ), "sample_idx should be identical for all streams within one sample."

            return idx
        else:
            msg = "Cannot infer sample_idx without any streams added."
            raise RuntimeError(msg)

    def pin_memory(self):
        """Pin all tensors in this Sample to CPU pinned memory"""

        # Pin StreamData objects in streams_data dict
        if hasattr(self, "streams_data") and isinstance(self.streams_data, dict):
            for _stream_name, stream_data in self.streams_data.items():
                if stream_data is not None and hasattr(stream_data, "pin_memory"):
                    stream_data.pin_memory()

        # Pin tensors in meta_info
        if hasattr(self, "meta_info") and isinstance(self.meta_info, dict):
            for _key, meta_data in self.meta_info.items():
                if isinstance(meta_data, SampleMetaData):
                    # Pin mask tensor
                    if meta_data.mask is not None and isinstance(meta_data.mask, torch.Tensor):
                        meta_data.mask = meta_data.mask.pin_memory()

        return self

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
        return all(s.empty() if s is not None else True for s in self.streams_data.values())

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


class BatchSamples:
    """
    Container for source or target samples
    """

    samples: list[Sample]
    tokens_lens: torch.Tensor | None
    output_steps: int
    output_idxs: list[int]
    device: str | None

    def __init__(self, streams: dict, num_samples: int, output_steps, output_idxs) -> None:
        self.samples = [Sample(streams) for _ in range(num_samples)]
        self.tokens_lens = None
        self.output_steps = output_steps
        self.output_idxs = output_idxs
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

    @property
    def sample_idxs(self) -> list[int]:
        return [sample.sample_idx for sample in self.samples]

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

    # unused
    def get_num_steps(self) -> int:
        """
        Get number of input/source steps from smallest of all available streams
        """
        # TODO: define explicitly
        lens = [
            len(stream.source_tokens_cells) for _, stream in self.samples[0].streams_data.items()
        ]

        return min(lens)

    # unused
    def get_output_idxs(self) -> int:
        """
        Get forecast indices
        """
        return self.output_idxs

    # unused
    def get_output_len(self) -> int:
        """
        Get length of output
        """
        return self.output_steps

    # unused
    def get_device(self) -> str | torch.device:
        """
        Get device of tensors in the batch
        """
        return self.device

    # unused
    def pin_memory(self):
        """Pin all tensors in this batch to CPU pinned memory"""

        # pin all samples
        for sample in self.samples:
            sample.pin_memory()

        # pin source_tokens_lens
        if isinstance(self.tokens_lens, torch.Tensor):
            self.tokens_lens = self.tokens_lens.pin_memory()

        return self


class ModelBatch:
    """
    Container for all data and metadata for one training batch.

    The data a instance contains is associated with one particular inital sampling window.
    From this initial data multiple forecast windows are derived by offsetting the initial window.
    Multiple samples for one forecast window can be generated
    by sampling differently masked versions of the data.
    Note that `output_offset` is the difference in forecast steps
    between corresponding source and target samples.
    Source and target samples are in 1-to-1 correspondance for classical training modes
    (e.g. MTM, forecasting), but can be more complex for strategies like student-teacher training.
    This relationship is expressed in `source2target_matching_idxs`
    and `target2source_matching_idxs`.


    Attributes:
        init_time: Initial sampling window used to costruct samples.
        source_samples: sources for model
        target_samples: targets for TargetAuxCalculator
        source2target_matching_idxs: index of corresponding target indices.
        target2source_matching_idxs: index of corresponding source indices.
        output_idxs: Forecast step indices (including offset) for this batch. The number of
          forecast steps is constant per batch, this amortized through data parallel training.
        device: device of the tensors in this batch.
    """

    def __init__(
        self,
        init_time: DTRange,
        streams: dict,
        num_source_samples: int,
        num_target_samples: int,
        output_offset: int,
        output_steps: int,
    ) -> None:
        """
        Initialize new ModelBatch.

        Args:
            init_time: Initial sampling window used to costruct samples.
            streams: global StreamConfig.
            num_source_samples: Number of differently masked source samples for one input window.
            num_target_samples: Number of differently masked target samples for one input window.
            output_offset: forecast offset for this batch.
            output_steps: number of forecast steps for this batch.
        """
        self.init_time: DTRange = init_time

        # define forecast indices
        self.output_offset: BatchSamples = output_offset
        self.output_steps: BatchSamples = output_steps
        self.output_idxs: list[int] = list(range(output_offset, output_steps))

        self.source_samples = BatchSamples(
            streams, num_source_samples, output_steps, self.output_idxs
        )
        self.target_samples = BatchSamples(
            streams, num_target_samples, output_steps, self.output_idxs
        )

        self.source2target_matching_idxs: np.typing.NDArray[np.int32] = np.full(
            num_source_samples, -1, dtype=np.int32
        )
        self.target2source_matching_idxs: np.typing.NDArray[np.int32] = [
            [] for _ in range(num_target_samples)
        ]
        self.device: torch.device | None = None

    def pin_memory(self) -> ModelBatch:
        """Pin all tensors in this batch to CPU pinned memory"""

        # pin source samples
        self.source_samples.pin_memory()

        # pin target samples
        self.target_samples.pin_memory()

        return self

    def to_device(self, device: str | torch.device) -> ModelBatch:
        """
        Move batch to device
        """
        self.device = torch.device(device)

        self.source_samples.to_device(self.device)
        self.target_samples.to_device(self.device)

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
        source_empty = all(
            s.is_empty() if s is not None else True for s in self.source_samples.samples
        )
        target_empty = all(
            s.is_empty() if s is not None else True for s in self.target_samples.samples
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

    def get_output_idxs(self) -> int:
        """
        Get valid output steps
        """
        return self.output_idxs

    def get_output_len(self) -> int:
        """
        Get length of output
        """
        return self.output_steps

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
