"""
Data structures for student-teacher multi-view training.

Provides clean separation between:
  - Model data (StreamData objects containing tensors)
  - View metadata (spatial masks, strategies, relationships)
"""

from dataclasses import dataclass

import numpy as np

from weathergen.datasets.stream_data import StreamData

# TODO: Add a store for a random number for diffusion
# TODO: GetTimestep to get the timestep
# TODO: GetData: get the streamdata
# TODO: GetMetaData: then this gets the right rn for the timestep!


@dataclass
class ViewMetadata:
    """
    Metadata describing how a view was generated.

    This captures the spatial selection (which cells/tokens were kept),
    the strategy used (random, healpix, etc.), and hierarchical parameters.

    Attributes:
        view_id: Unique identifier (e.g., "teacher_global", "student_local_0")
        keep_mask: Boolean array [num_healpix_cells] at data level indicating kept cells
        strategy: Name of selection strategy ("random", "healpix_level_2", etc.)
        healpix_level: HEALPix level for hierarchical selection (None if not applicable)
        rate: Fraction of data kept (e.g., 0.5 = 50% kept); None if fixed count
        parent_view_id: ID of the parent view this is a subset of (None for teacher)
    """

    # Core identifiers and selection description
    view_id: str
    keep_mask: np.typing.NDArray  # [num_cells] bool at data level
    strategy: str  # e.g. "random", "healpix", "channel"

    # Hierarchical/quantitative description of selection
    healpix_level: int | None = None
    rate: float | None = None
    parent_view_id: str | None = None  # For students: which teacher they belong to

    # Optional extras for future/other training paradigms
    loss_type: str | None = None  # e.g. DINO, JEPA
    strategy_config: dict | None = None  # e.g. {rate: 0.5, hl_mask: 3, overlap: "disjoint"}


class SampleMetaData:
    # masking strategy
    masking_strategy: str

    # parameters for masking strategy
    masking_params: dict


class Sample:
    # keys: stream name, values: SampleMetaData
    meta_info: dict

    # data for all streams
    # keys: stream_name, values: StreamData
    streams_data: dict

    def __init__(self, streams: dict) -> None:
        # TODO: can we pass this right away?
        self.meta_info = {}

        self.streams_data = {}
        for stream_info in streams:
            self.streams_data[stream_info["name"]] = None

    def add_stream_data(self, stream_name: str, stream_data: StreamData) -> None:
        """
        Add data for stream @stream_name to sample
        """
        assert self.streams_data.get(stream_name, -1) != -1, "stream name does not exist"
        self.streams_data[stream_name] = stream_data

    # TODO: complete interface, e.g get_stream

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
    source_matching_idx: np.typing.NDArray[np.int32]
    target_matching_idx: np.typing.NDArray[np.int32]

    def __init__(self, streams, num_source_samples: int, num_target_samples: int) -> None:
        """ """

        self.source_samples = [Sample(streams) for _ in range(num_source_samples)]
        self.target_samples = [Sample(streams) for _ in range(num_target_samples)]

        self.source_target_matching_idxs = np.full(num_source_samples, -1, dtype=np.int32)
        # self.target_source_matching_idxs = np.full(num_target_samples, -1, dtype=np.int32)
        self.target_source_matching_idxs = [[] for _ in range(num_target_samples)]

    def add_source_stream(
        self,
        source_sample_idx: int,
        target_sample_idx: int,
        stream_name: str,
        stream_data: StreamData,
    ) -> None:
        """
        Add data for one stream to sample @source_sample_idx
        """
        self.source_samples[source_sample_idx].add_stream_data(stream_name, stream_data)

        assert target_sample_idx < len(self.target_samples), "invalid value for target_sample_idx"
        self.source_target_matching_idxs[source_sample_idx] = target_sample_idx

    def add_target_stream(
        self,
        target_sample_idx: int,
        source_sample_idx: int | list[int],
        stream_name: str,
        stream_data: StreamData,
    ) -> None:
        """
        Add data for one stream to sample @target_sample_idx
        """
        self.target_samples[target_sample_idx].add_stream_data(stream_name, stream_data)

        if isinstance(source_sample_idx, int):
            assert source_sample_idx < len(self.source_samples), "invalid value for source_sample_idx"
        else:
            assert all(idx < len(self.source_samples) for idx in source_sample_idx), "invalid value for source_sample_idx"
        self.target_source_matching_idxs[target_sample_idx] = source_sample_idx

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
        return int(self.source_target_matching_idxs[target_idx])

    def get_target_idx_for_source(self, source_idx: int) -> int:
        """
        Get index of target sample for a given source sample index
        """
        return int(self.source_target_matching_idxs[source_idx])
