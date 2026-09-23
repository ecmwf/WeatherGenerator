"""Standalone tests for vectorized hpy_splits.

The hl=1 case asserts hand-checked tokens and does not call a frozen original.
TokenizerMasking always converts coords to torch before tokenize_space.
"""

import torch
from astropy_healpix.healpy import ang2pix

from weathergen.datasets.tokenizer_utils import hpy_splits, theta_phi_to_standard_coords


def test_hpy_splits_known_layout_hl1():
    """Hand-checked nest layout at hl=1 (48 cells). Does not use the frozen original.

    Four lat/lon points (degrees). Nest ids are ``ang2pix(nside=2, ..., nest=True)``
    after ``theta_phi_to_standard_coords`` (same as production):

      row 0:  80N,   0E  -> cell 11, theta smallest of the three northern points
      row 1:  70N,   0E  -> cell 11, theta largest of those three
      row 2:  75N,   0E  -> cell 11, theta between row 0 and row 1
      row 3:  80S, 180E  -> cell 32

    Theta order in cell 11 is therefore rows 0, 2, 1. token_size=2.

    pad_tokens=True stores index+1; leftover slots are 0 (padding row):
      cell 11: [1, 3] then [2, 0]
      cell 32: [4, 0]

    pad_tokens=False stores raw rows; last token may be short:
      cell 11: [0, 2] then [1]
      cell 32: [3]
    """
    hl = 1
    token_size = 2
    num_cells = 12 * 4**hl
    cell_north = 11
    cell_south = 32
    coords = torch.tensor(
        [
            [80.0, 0.0],
            [70.0, 0.0],
            [75.0, 0.0],
            [-80.0, 180.0],
        ],
        dtype=torch.float32,
    )

    thetas, phis = theta_phi_to_standard_coords(coords)
    cells = ang2pix(2**hl, thetas, phis, nest=True)
    assert cells.tolist() == [cell_north, cell_north, cell_north, cell_south]
    assert float(thetas[0]) < float(thetas[2]) < float(thetas[1])

    idxs, lens = hpy_splits(coords, hl, token_size, pad_tokens=True, offset_step=0)
    assert len(idxs) == num_cells
    for cell_i, (toks, tok_lens) in enumerate(zip(idxs, lens, strict=True)):
        if cell_i == cell_north:
            assert tok_lens == [2, 2]
            assert torch.equal(toks[0], torch.tensor([1, 3], dtype=toks[0].dtype))
            assert torch.equal(toks[1], torch.tensor([2, 0], dtype=toks[1].dtype))
        elif cell_i == cell_south:
            assert tok_lens == [2]
            assert torch.equal(toks[0], torch.tensor([4, 0], dtype=toks[0].dtype))
        else:
            assert toks == []
            assert tok_lens == []

    idxs, lens = hpy_splits(coords, hl, token_size, pad_tokens=False, offset_step=0)
    for cell_i, (toks, tok_lens) in enumerate(zip(idxs, lens, strict=True)):
        if cell_i == cell_north:
            assert tok_lens == [2, 1]
            assert torch.equal(toks[0], torch.tensor([0, 2], dtype=toks[0].dtype))
            assert torch.equal(toks[1], torch.tensor([1], dtype=toks[1].dtype))
        elif cell_i == cell_south:
            assert tok_lens == [1]
            assert torch.equal(toks[0], torch.tensor([3], dtype=toks[0].dtype))
        else:
            assert toks == []
            assert tok_lens == []
