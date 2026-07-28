# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from weathergen.model.spatial_parallel import select_packed_cell_shard
from weathergen.utils import distributed


def test_select_packed_cell_shard_preserves_cell_boundaries_across_rows():
    cell_lens = torch.tensor(
        [
            [1, 0, 2, 1, 3, 0, 1, 2],
            [0, 2, 1, 0, 1, 2, 0, 1],
        ],
        dtype=torch.int32,
    )
    tokens = torch.arange(cell_lens.sum(), dtype=torch.float32).unsqueeze(1)

    shard, shard_lens = select_packed_cell_shard(
        tokens, cell_lens.flatten(), num_cells=8, cell_start=2, cell_end=4
    )

    assert shard_lens.tolist() == [2, 1, 1, 0]
    assert shard.squeeze(1).tolist() == [1, 2, 3, 12]


def test_eight_shards_cover_every_packed_token_once_and_keep_gradients():
    num_cells = 16
    cell_lens = torch.tensor(
        [
            [0, 1, 2, 0, 1, 3, 0, 2, 1, 0, 2, 1, 0, 1, 2, 1],
            [1, 0, 1, 2, 0, 1, 2, 0, 3, 1, 0, 1, 2, 0, 1, 1],
        ],
        dtype=torch.int32,
    )
    tokens = torch.arange(cell_lens.sum(), dtype=torch.float32, requires_grad=True)
    shard_width = num_cells // 8

    selected = []
    for rank in range(8):
        shard, _ = select_packed_cell_shard(
            tokens,
            cell_lens.flatten(),
            num_cells,
            rank * shard_width,
            (rank + 1) * shard_width,
        )
        selected.append(shard)
        shard.sum().backward(retain_graph=rank < 7)

    assert sum(shard.numel() for shard in selected) == tokens.numel()
    assert torch.equal(tokens.grad, torch.ones_like(tokens))


@pytest.mark.parametrize(
    ("num_cells", "cell_start", "cell_end"),
    [(8, -1, 1), (8, 3, 3), (8, 0, 9)],
)
def test_select_packed_cell_shard_rejects_invalid_ranges(num_cells, cell_start, cell_end):
    with pytest.raises(ValueError, match="invalid HEALPix cell range"):
        select_packed_cell_shard(
            torch.arange(num_cells),
            torch.ones(num_cells, dtype=torch.int32),
            num_cells,
            cell_start,
            cell_end,
        )


def test_spatial_parallel_size_requires_whole_rank_groups(monkeypatch):
    monkeypatch.setattr(distributed, "get_world_size", lambda: 16)
    assert distributed.get_encoder_spatial_parallel_size({"encoder_spatial_parallel_size": 4}) == 4
    assert distributed.get_encoder_spatial_parallel_size({"encoder_spatial_parallel_size": 8}) == 8

    with pytest.raises(ValueError, match="must be divisible"):
        distributed.get_encoder_spatial_parallel_size({"encoder_spatial_parallel_size": 6})
