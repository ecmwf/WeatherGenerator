from dataclasses import dataclass

from numpy.typing import NDArray
from torch import Tensor

@dataclass
class TrainerPredictions:
    # Denormalized predictions
    preds_all: list[list[list[Tensor]]]
    # Denormalized targets
    targets_all: list[list[list[Tensor]]]
    # Raw target coordinates
    targets_coords_raw: list[list[Tensor]]
    # Raw target timestamps
    targets_times_raw: list[list[NDArray]]
    # Target lengths
    targets_lens: list[list[list[int]]]
    tokens_all: list[Tensor]
    tokens_coords_raw: list[NDArray]
    tokens_times_raw: list[NDArray]
