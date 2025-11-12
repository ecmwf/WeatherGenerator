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


@dataclass
class ModelBatch:
    """
    Container for all data and metadata for one training batch.
    
    Wraps StreamData objects (which hold the actual tensors) and associates
    them with metadata describing how views were generated and their relationships.
    
    Training modes:
      - Masking (MAE-style):
          model_inputs: [single StreamData with unmasked tokens]
          targets: [single StreamData with masked tokens]
          view_metadata: [single ViewMetadata describing mask]
      
      - Student-teacher (JEPA-style):
          model_inputs: [list of StreamData, one per local/student view]
          targets: [single StreamData with teacher/global view]
          view_metadata: [list of ViewMetadata: teacher first, then students]
    
    Attributes:
        model_inputs: List of StreamData fed to encoder (students in ST, single in masking)
        targets: List of StreamData for loss computation (teacher in ST, masked in masking)
        view_metadata: List of ViewMetadata describing each view (teacher + students)
        batch_info: Optional dict with batch-level info (sample indices, forecast steps, etc.)
    """
    model_inputs: list[StreamData]   # >= 1; students in ST mode, single view in masking
    targets: list[StreamData]        # teacher in ST mode; masked tokens in masking
    view_metadata: list[ViewMetadata]
    batch_info: Optional[dict] = field(default_factory=dict)