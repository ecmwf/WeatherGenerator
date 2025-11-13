"""
Data structures for student-teacher multi-view training.

Provides clean separation between:
  - Model data (StreamData objects containing tensors)
  - View metadata (spatial masks, strategies, relationships)
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from weathergen.datasets.stream_data import StreamData


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
    keep_mask: np.ndarray          # [num_cells] bool at data level
    strategy: str                  # e.g., "random", "healpix_level_2"
    healpix_level: Optional[int]
    rate: Optional[float]
    parent_view_id: Optional[str] = None  # For students: which teacher they belong to


# TODO: This doesn't handle the masking case, and we probably want it to,
# where the model_inputs are the correct data for the masked source (and target?). Or target becomes the target?
# Also should this model batch contain the source_cell_lens and target_coords_idx?

@dataclass
class ModelBatch:
    """
    Container for all data and metadata for one training batch.
    
    Wraps StreamData objects (which hold the actual tensors) and associates
    them with metadata describing how views were generated and their relationships.
    
    Training modes:
      - Student-teacher (JEPA-style):
          model_inputs: [list[list[StreamData]]] - [n_students][n_streams]
          targets: [list[list[StreamData]]] - [1][n_streams] (teacher)
          view_metadata: [list of ViewMetadata: teacher first, then students]
    
    Attributes:
        model_inputs: List of student views, each containing StreamData for all streams
        targets: List containing teacher view with StreamData for all streams
        view_metadata: List of ViewMetadata describing each view (teacher + students)
        batch_info: Optional dict with batch-level info (sample indices, forecast steps, etc.)
    """
    model_inputs: list[list[any]]   # [n_students][n_streams], the student views (ideally later for masking too)
    targets: list[list[any]]        # [1][n_streams], teacher view
    view_metadata: list[ViewMetadata]
    batch_info: Optional[dict] = field(default_factory=dict)
    
    def to_device(self, device):
        """
        Move all StreamData objects to the specified device.
        
        Args:
            device: device where we want to move
        
        Returns:
            self
        """
        # Move all student views to device
        for student_view in self.model_inputs:
            for stream_data in student_view:
                stream_data.to_device(device)
        
        # Move teacher view to device
        for teacher_batch in self.targets:
            for stream_data in teacher_batch:
                stream_data.to_device(device)
        
        return self