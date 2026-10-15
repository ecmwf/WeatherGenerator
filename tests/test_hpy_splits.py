# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Equivalence of vectorized hpy_cell_splits / hpy_splits vs the original loop.

The original implementations are frozen here. TokenizerMasking always converts
coords to torch before tokenize_space, so the cases use torch coordinates.
"""

import numpy as np
import pytest
import torch
from astropy_healpix.healpy import ang2pix

from weathergen.datasets.tokenizer_utils import (
    hpy_cell_splits,
    hpy_splits,
    numpy_argsort_args,
    theta_phi_to_standard_coords,
)


def _hpy_cell_splits_original(coords: torch.Tensor, hl: int):
    """Pre-rewrite hpy_cell_splits: per-cell index lists, unsorted by latitude."""
    thetas, phis = theta_phi_to_standard_coords(coords)
    hpy_idxs = ang2pix(2**hl, thetas, phis, nest=True)

    hpy_idxs_ord = np.argsort(hpy_idxs, **numpy_argsort_args)
    splits = np.flatnonzero(np.diff(hpy_idxs[hpy_idxs_ord]))

    hpy_idxs_ord_temp = np.split(hpy_idxs_ord, splits + 1)
    hpy_idxs_ord_split = [np.array([], dtype=np.int64) for _ in range(12 * 4**hl)]
    for b, x in zip(np.unique(np.unique(hpy_idxs[hpy_idxs_ord])), hpy_idxs_ord_temp, strict=True):
        hpy_idxs_ord_split[b] = x

    return (hpy_idxs_ord_split, thetas, phis)


def _hpy_splits_original(
    coords: torch.Tensor, hl: int, token_size: int, pad_tokens: bool, offset_step: int = 0
):
    """Pre-rewrite hpy_splits: per-cell torch.argsort / cat / split."""
    (hpy_idxs_ord_split, thetas, phis) = _hpy_cell_splits_original(coords, hl)

    thetas_sorted = [torch.argsort(thetas[idxs], stable=True) for idxs in hpy_idxs_ord_split]
    if pad_tokens:
        rem = [
            token_size - (len(idxs) % token_size if len(idxs) % token_size != 0 else token_size)
            for idxs in hpy_idxs_ord_split
        ]
    else:
        rem = np.zeros(len(hpy_idxs_ord_split), dtype=np.int32)

    offset = (1 if pad_tokens else 0) + offset_step
    int32 = torch.int32
    idxs_ord = [
        list(
            torch.split(
                torch.cat(
                    (torch.from_numpy(np.take(idxs, ts) + offset), torch.zeros(r, dtype=int32))
                ),
                token_size,
            )
        )
        if len(idxs) > 0
        else []
        for idxs, ts, r in zip(hpy_idxs_ord_split, thetas_sorted, rem, strict=True)
    ]

    idxs_ord_lens = [[len(a) for a in aa] for aa in idxs_ord]
    return idxs_ord, idxs_ord_lens


def _split_by_counts(idxs_ord, counts):
    """Recover the original per-cell index lists from the vectorized return value."""
    cells = []
    start = 0
    for count in counts:
        cells.append(idxs_ord[start : start + int(count)])
        start += int(count)
    return cells


def _coords(n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-89.0, 89.0, n).astype(np.float32)
    lon = rng.uniform(-180.0, 180.0, n).astype(np.float32)
    return torch.tensor(np.stack([lat, lon], axis=1))


def _assert_hpy_splits_equal(old_idxs, old_lens, new_idxs, new_lens):
    assert old_lens == new_lens
    assert len(old_idxs) == len(new_idxs)
    for cell_i, (old_cell, new_cell) in enumerate(zip(old_idxs, new_idxs, strict=True)):
        assert len(old_cell) == len(new_cell), f"cell {cell_i}: token count"
        for tok_i, (old_tok, new_tok) in enumerate(zip(old_cell, new_cell, strict=True)):
            assert old_tok.shape == new_tok.shape, f"cell {cell_i} token {tok_i}: shape"
            assert torch.equal(old_tok, new_tok), f"cell {cell_i} token {tok_i}: values"


# ---------------------------------------------------------------------------
# Cases: production always passes torch coords. Empty streams never reach these
# functions (TokenizerMasking skips them).
# ---------------------------------------------------------------------------

HPY_CASES = [
    pytest.param(_coords(1, 0), 3, 4, True, 0, id="one_point"),
    pytest.param(_coords(8000, 1), 3, 4, True, 0, id="dense_padded"),
    pytest.param(_coords(8000, 1), 3, 4, False, 0, id="dense_unpadded"),
    pytest.param(_coords(8000, 1), 3, 4, True, 17, id="dense_offset"),
    pytest.param(
        torch.tensor(
            np.stack(
                [np.full(16, 45.0, np.float32), np.linspace(10.0, 10.01, 16, dtype=np.float32)],
                axis=1,
            )
        ),
        3,
        4,
        True,
        0,
        id="exact_multiple_of_token_size",
    ),
    pytest.param(
        torch.tensor(
            np.stack(
                [
                    np.array([45.0] * 8 + [10.0] * 3, np.float32),
                    np.array([20.0] * 8 + [40.0] * 3, np.float32),
                ],
                axis=1,
            )
        ),
        3,
        3,
        True,
        0,
        id="tied_coordinates",
    ),
    pytest.param(_coords(7, 2), 3, 4, True, 0, id="remainder_token"),
    pytest.param(_coords(2000, 3), 5, 4, True, 0, id="healpix_level_5"),
]


@pytest.mark.parametrize("coords, hl, token_size, pad_tokens, offset_step", HPY_CASES)
def test_hpy_splits_matches_original(coords, hl, token_size, pad_tokens, offset_step):
    old_idxs, old_lens = _hpy_splits_original(coords, hl, token_size, pad_tokens, offset_step)
    new_idxs, new_lens = hpy_splits(coords, hl, token_size, pad_tokens, offset_step)
    _assert_hpy_splits_equal(old_idxs, old_lens, new_idxs, new_lens)


@pytest.mark.parametrize("coords, hl, token_size, pad_tokens, offset_step", HPY_CASES)
def test_hpy_cell_splits_matches_original(coords, hl, token_size, pad_tokens, offset_step):
    """Same points per cell; within a cell the new order is the original latitude sort."""
    del token_size, pad_tokens, offset_step
    old_cells, thetas, _phis = _hpy_cell_splits_original(coords, hl)
    new_idxs, new_counts = hpy_cell_splits(coords, hl)

    assert len(old_cells) == len(new_counts)
    assert int(new_counts.sum()) == coords.shape[0]
    new_cells = _split_by_counts(new_idxs, new_counts)

    for cell_i, (old_cell, new_cell) in enumerate(zip(old_cells, new_cells, strict=True)):
        assert old_cell.shape[0] == new_cell.shape[0], f"cell {cell_i}: occupancy"
        np.testing.assert_array_equal(
            np.sort(old_cell), np.sort(new_cell), err_msg=f"cell {cell_i}: member indices"
        )
        if old_cell.shape[0] == 0:
            continue
        theta_order = torch.argsort(thetas[old_cell], stable=True).numpy()
        np.testing.assert_array_equal(
            np.take(old_cell, theta_order),
            new_cell,
            err_msg=f"cell {cell_i}: latitude order",
        )
