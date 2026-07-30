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


def select_healpix_neighborhood_shard(
    tokens: torch.Tensor,
    hp_neighbours: torch.Tensor,
    cell_start: int,
    cell_end: int,
) -> torch.Tensor:
    """Pack nine-token HEALPix neighbourhoods for one contiguous cell shard.

    Args:
        tokens: Dense tensor in ``(sample, cell, query, channel)`` order.
        hp_neighbours: Global ``(cell, 9)`` lookup containing self and its
            eight neighbours.
        cell_start: First cell owned by the spatial rank.
        cell_end: Exclusive end of the rank's cell range.
    """

    if tokens.ndim != 4:
        raise ValueError("tokens must have shape (sample, cell, query, channel)")
    num_cells = tokens.shape[1]
    if hp_neighbours.shape != (num_cells, 9):
        raise ValueError(
            f"expected HEALPix neighbour lookup with shape ({num_cells}, 9), "
            f"got {tuple(hp_neighbours.shape)}"
        )
    if not 0 <= cell_start < cell_end <= num_cells:
        raise ValueError(
            f"invalid HEALPix cell range [{cell_start}, {cell_end}) for {num_cells} cells"
        )

    local_neighbours = hp_neighbours[cell_start:cell_end].to(dtype=torch.long)
    return tokens[:, local_neighbours].flatten(0, 3)


def reassemble_packed_cell_shards(
    shards: list[torch.Tensor],
    shard_cell_lens: list[torch.Tensor],
    cells_per_shard: int,
) -> torch.Tensor:
    """Reassemble rank-packed predictions into sample-major HEALPix order."""

    if len(shards) != len(shard_cell_lens) or not shards:
        raise ValueError("shards and shard_cell_lens must be non-empty and have equal length")
    if any(lens.numel() % cells_per_shard for lens in shard_cell_lens):
        raise ValueError("shard_cell_lens does not contain whole local HEALPix grids")

    batch_size = shard_cell_lens[0].numel() // cells_per_shard
    if any(lens.numel() != batch_size * cells_per_shard for lens in shard_cell_lens):
        raise ValueError("all spatial shards must describe the same batch size")

    rank_offsets = [0] * len(shards)
    sample_predictions = []
    for sample_idx in range(batch_size):
        sample_shards = []
        for rank, (rank_pred, rank_lens) in enumerate(zip(shards, shard_cell_lens, strict=True)):
            lens = rank_lens.reshape(batch_size, cells_per_shard)
            sample_len = int(lens[sample_idx].sum().item())
            offset = rank_offsets[rank]
            sample_shards.append(rank_pred[:, offset : offset + sample_len])
            rank_offsets[rank] += sample_len
        sample_predictions.append(torch.cat(sample_shards, dim=1))
    return torch.cat(sample_predictions, dim=1)
