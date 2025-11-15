"""
Data structures for student-teacher multi-view training.

Provides clean separation between:
  - Model data (StreamData objects containing tensors)
  - View metadata (spatial masks, strategies, relationships)
"""

from dataclasses import dataclass, field

import numpy as np
import torch

# TODO: SampleMetadata !! Just a dictionary to store a random number for diffusion
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

    view_id: str
    keep_mask: np.ndarray  # [num_cells] bool at data level
    strategy: str  # e.g., "random", "healpix_level_2"
    healpix_level: int | None
    rate: float | None
    parent_view_id: str | None = None  # For students: which teacher they belong to


# TODO: This doesn't handle the masking case, and we probably want it to,
# where the model_inputs are the correct data for the masked source (and target?). Or target becomes the target?
# Also should this model batch contain the source_cell_lens and target_coords_idx?
# Every sample is n different [streams]...each view is a different dictionary corresponding to one model input
# to get epsilon in there...
# batches is for parallelism, but needs to all be in a tensor... [b, n, dim_embedding]? [b x n, dim_embedding]


# NOTE: this only stores the student source_cell_lens and target_coords_idx,
# because the teacher ones are already provided separately in (model_batches, source_cell_lens, target_coords_idx, forecast_dt)
# ^^^^^^ teacher ones ^^^^^^
# However, we should probably store them all here for consistency. This needs changes to the model, so not done now.
# The forecast_dt is provided separately?


@dataclass
class ModelBatch:
    """
    Container for all data and metadata for one training batch.

    Attributes:
        model_inputs: List of student views, each containing StreamData for all streams
        targets: List containing teacher view with StreamData for all streams
        view_metadata: List of ViewMetadata describing each view (teacher + students)
        batch_info: Optional dict with batch-level info (sample indices, forecast steps, etc.)
        student_source_cell_lens: List of source cell lengths for each student view
        student_target_coords_idx: List of target coordinate indices for each student view
    """

    model_inputs: list[list[any]]  # [n_students][n_streams]
    targets: list[list[any]]  # [1][n_streams] (teacher)
    view_metadata: list[ViewMetadata]
    batch_info: dict | None = field(default_factory=dict)

    # Offsets for student views (populated when needed for future student-teacher training)
    # TODO: rename to model_input...source_cell/target_coords...
    student_source_cell_lens: list | None = None  # [n_students] each is a tensor
    student_target_coords_idx: list | None = None  # [n_students] each is a list of lists

    # TODO: this also needs target_source_cell_lens and target_target_coords_idx for teacher views
    # TODO fix this ridiculous naming

    # TODO: add the timestep as an optional int for the model_inputs when we have multiple timesteps for the diffusion model...
    # TODO add the forecast_dt as an optional int ?

    def to_device(self, device):
        """Move all StreamData objects to the specified device."""
        for student_view in self.model_inputs:
            for stream_data in student_view:
                stream_data.to_device(device)

        for teacher_batch in self.targets:
            for stream_data in teacher_batch:
                stream_data.to_device(device)

        # Move student offsets if they exist
        if self.student_source_cell_lens is not None:
            self.student_source_cell_lens = [
                lens.to(device) if isinstance(lens, torch.Tensor) else lens
                for lens in self.student_source_cell_lens
            ]

        if self.student_target_coords_idx is not None:
            # This is list[list[list[tensor]]], need to move all tensors
            self.student_target_coords_idx = [
                [
                    [t.to(device) if isinstance(t, torch.Tensor) else t for t in stream]
                    for stream in student_idx
                ]
                for student_idx in self.student_target_coords_idx
            ]

        return self
