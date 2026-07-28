# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch


def select_packed_cell_shard(
    tokens: torch.Tensor,
    cell_lens: torch.Tensor,
    num_cells: int,
    cell_start: int,
    cell_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a HEALPix-cell range from a cell-packed token tensor.

    ``cell_lens`` is flattened in ``(input_step, sample, cell)`` order. Token
    counts vary by cell, so slicing the first tensor dimension directly would
    split cells. The returned tokens remain packed in the same order, restricted
    to ``[cell_start, cell_end)`` for every input-step/sample row.
    """

    if cell_lens.numel() % num_cells:
        raise ValueError("cell_lens does not contain a whole number of HEALPix grids")
    if not 0 <= cell_start < cell_end <= num_cells:
        raise ValueError(
            f"invalid HEALPix cell range [{cell_start}, {cell_end}) for {num_cells} cells"
        )

    cell_lens_2d = cell_lens.reshape(-1, num_cells)
    selected_cells = torch.zeros_like(cell_lens_2d, dtype=torch.bool)
    selected_cells[:, cell_start:cell_end] = True
    selected_tokens = torch.repeat_interleave(
        selected_cells.flatten(), cell_lens.to(dtype=torch.long)
    )
    if selected_tokens.numel() != tokens.shape[0]:
        raise ValueError(
            f"packed token length mismatch: cell_lens describes {selected_tokens.numel()} "
            f"tokens, tensor has {tokens.shape[0]}"
        )

    return tokens[selected_tokens], cell_lens_2d[:, cell_start:cell_end].flatten()
