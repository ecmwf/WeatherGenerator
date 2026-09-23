"""Tests for vectorized get_target_coords_local.

Equivalence cases compare against the frozen split/cat path. The standalone
two-cell case asserts hand-checked columns and does not call the original.
"""

import numpy as np
import pytest
import torch
from astropy_healpix.healpy import ang2pix

from weathergen.datasets.tokenizer import Tokenizer
from weathergen.datasets.tokenizer_utils import (
    _rotate_points_per_cell,
    get_target_coords_local,
    theta_phi_to_standard_coords,
)
from weathergen.datasets.utils import s2tor3, vecs_to_rots


def _locs_to_cell_coords_ctrs(healpix_centers_rots: torch.Tensor, locs: list[torch.Tensor]):
    """Frozen pre-rewrite helper. Removed from dataset utils; only used here."""
    all_points = torch.cat(locs, dim=0)
    lengths = torch.tensor([len(s) for s in locs], device=all_points.device)
    batch_indices = torch.repeat_interleave(
        torch.arange(len(locs), device=all_points.device), lengths
    )
    rotations_selected = healpix_centers_rots[batch_indices]
    return torch.bmm(rotations_selected, all_points.unsqueeze(-1)).squeeze(-1)


def _locs_to_ctr_coords(ctrs_r3, locs: list[torch.Tensor]) -> list:
    """Frozen pre-rewrite helper. Removed from dataset utils; only used here."""
    ctrs_rots = vecs_to_rots(ctrs_r3).to(torch.float32)
    all_points = torch.cat(locs, dim=0)
    lengths = torch.tensor([len(s) for s in locs], device=all_points.device)
    batch_indices = torch.repeat_interleave(
        torch.arange(len(locs), device=all_points.device), lengths
    )
    rotated_points = torch.bmm(ctrs_rots[batch_indices], all_points.unsqueeze(-1)).squeeze(-1)
    return list(torch.split(rotated_points, lengths.tolist()))


def _get_target_coords_local_original(
    stream_id,
    hlc,
    masked_points_per_cell,
    coords,
    target_geoinfos,
    target_times,
    verts_rots,
    verts_local,
    nctrs,
):
    """Pre-rewrite get_target_coords_local: split per cell, then locs_to_* helpers."""
    del hlc
    target_coords = s2tor3(*theta_phi_to_standard_coords(coords))
    tcs = torch.split(target_coords, masked_points_per_cell.tolist())

    if target_coords.shape[0] == 0:
        return torch.tensor([])

    verts00_rots, verts10_rots, verts11_rots, verts01_rots, vertsmm_rots = verts_rots

    a = torch.zeros(
        [
            *target_coords.shape[:-1],
            1 + target_geoinfos.shape[1] + target_times.shape[1] + 5 * (3 * 5) + 3 * 8,
        ]
    )
    a[0] = stream_id
    geoinfo_offset = 1
    a[..., geoinfo_offset : geoinfo_offset + target_times.shape[1]] = target_times
    geoinfo_offset += target_times.shape[1]
    a[..., geoinfo_offset : geoinfo_offset + target_geoinfos.shape[1]] = target_geoinfos
    geoinfo_offset += target_geoinfos.shape[1]

    ref = torch.tensor([1.0, 0.0, 0.0])

    tcs_lens = torch.tensor([tt.shape[0] for tt in tcs], dtype=torch.int32)
    tcs_lens_mask = tcs_lens > 0
    tcs_lens = tcs_lens[tcs_lens_mask]

    vls = torch.cat(
        [
            vl.repeat([tt, 1, 1])
            for tt, vl in zip(tcs_lens, verts_local[tcs_lens_mask], strict=False)
        ],
        0,
    )
    vls = vls.transpose(0, 1)

    zi = 0
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - _locs_to_cell_coords_ctrs(
        verts00_rots, tcs
    )

    zi = 3
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[0]

    zi = 15
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - _locs_to_cell_coords_ctrs(
        verts10_rots, tcs
    )

    zi = 18
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[1]

    zi = 30
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - _locs_to_cell_coords_ctrs(
        verts11_rots, tcs
    )

    zi = 33
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[2]

    zi = 45
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - _locs_to_cell_coords_ctrs(
        verts01_rots, tcs
    )

    zi = 48
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[3]

    zi = 60
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - _locs_to_cell_coords_ctrs(
        vertsmm_rots, tcs
    )

    zi = 63
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[4]

    tcs_ctrs = torch.cat([ref - torch.cat(_locs_to_ctr_coords(c, tcs)) for c in nctrs], -1)
    zi = 75
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + (3 * 8))] = tcs_ctrs

    zi = 99
    a[..., (geoinfo_offset + zi) :] = target_coords[..., (geoinfo_offset + 2) :]

    a[..., 98] = np.sin(coords[:, 0])
    a[..., 97] = np.cos(coords[:, 0])
    a[..., 96] = np.sin(coords[:, 1])
    a[..., 95] = np.cos(coords[:, 1])

    return a


def _pack_coords_by_cell(coords: torch.Tensor, hl: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Order points by nested healpix cell, matching TokenizerMasking packing."""
    thetas, phis = theta_phi_to_standard_coords(coords)
    hpy = ang2pix(2**hl, np.asarray(thetas), np.asarray(phis), nest=True)
    order = np.argsort(hpy, kind="stable")
    packed = coords[torch.as_tensor(order, dtype=torch.long)]
    counts = np.bincount(hpy, minlength=12 * 4**hl)
    return packed, torch.from_numpy(counts.astype(np.int32))


def _random_latlon(n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-89.0, 89.0, n).astype(np.float32)
    lon = rng.uniform(-180.0, 180.0, n).astype(np.float32)
    return torch.tensor(np.stack([lat, lon], axis=1))


def _geo_times(n: int, n_geo: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    geo = torch.tensor(rng.normal(size=(n, n_geo)).astype(np.float32))
    times = torch.tensor(rng.normal(size=(n, 5)).astype(np.float32))
    return geo, times


@pytest.fixture(scope="module")
def target_geometry():
    hl = 2
    tok = Tokenizer(hl)
    return {
        "hl": hl,
        "verts_rots": tok.hpy_verts_rots_target,
        "verts_local": tok.hpy_verts_local_target,
        "nctrs": tok.hpy_nctrs_target,
    }


def test_rotate_points_per_cell_toy_example():
    """4 cells / 5 points: empty, two in cell 1, empty, three in cell 3."""
    r1 = torch.eye(3, dtype=torch.float32)
    r3 = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    cell_rots = torch.stack([torch.eye(3), r1, torch.eye(3), r3])
    points = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    counts = torch.tensor([0, 2, 0, 3], dtype=torch.int32)

    got = _rotate_points_per_cell(cell_rots, points, counts)
    expected = torch.stack(
        [
            r1 @ points[0],
            r1 @ points[1],
            r3 @ points[2],
            r3 @ points[3],
            r3 @ points[4],
        ]
    )
    assert torch.equal(got, expected)


def test_get_target_coords_local_known_two_cells():
    """Two packed points, identity vertex frames, north-pole neighbor centers.

    Does not call the frozen original. Geometry is synthetic (2 cells), not Tokenizer.

      row 0 / cell 0: lat=0, lon=-180  ->  p = (1, 0, 0)
      row 1 / cell 1: lat=90, lon=0    ->  p = (0, 0, 1)

    Vertex rotations are I, so each vertex offset is ``ref - p`` with ref=(1,0,0).
    Neighbor centers are (0,0,1); ``vecs_to_rots`` then maps (x,y,z) -> (z,y,-x),
    so the 8 neighbor offsets are ``ref - (z, y, -x)``.
    ``verts_local`` is 0.1 in cell 0 and 0.2 in cell 1 (copied into each vertex block).
    """
    coords = torch.tensor([[0.0, -180.0], [90.0, 0.0]], dtype=torch.float32)
    counts = torch.tensor([1, 1], dtype=torch.int32)
    n_cells = 2
    times = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
        dtype=torch.float32,
    )
    geo = torch.zeros((2, 0), dtype=torch.float32)
    ident = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(n_cells, 1, 1)
    verts_rots = [ident.clone() for _ in range(5)]
    verts_local = torch.zeros((n_cells, 5, 12), dtype=torch.float32)
    verts_local[0] = 0.1
    verts_local[1] = 0.2
    nctrs = torch.tensor([0.0, 0.0, 1.0]).expand(8, n_cells, 3).contiguous()

    got = get_target_coords_local(
        stream_id=7.0,
        hlc=0,
        masked_points_per_cell=counts,
        coords=coords,
        target_geoinfos=geo,
        target_times=times,
        verts_rots=verts_rots,
        verts_local=verts_local,
        nctrs=nctrs,
    )

    p = s2tor3(*theta_phi_to_standard_coords(coords))
    torch.testing.assert_close(p[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(p[1], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5, rtol=0)

    assert got.shape == (2, 105)
    assert got[0, 0].item() == 7.0
    assert got[1, 0].item() == 0.0
    torch.testing.assert_close(got[:, 1:6], times)

    ref = torch.tensor([1.0, 0.0, 0.0])
    vertex_off = ref - p
    neighbor_off = ref - torch.stack(
        [torch.tensor([p[i, 2], p[i, 1], -p[i, 0]]) for i in range(2)]
    )
    for row, cell in ((0, 0), (1, 1)):
        local = torch.full((12,), 0.1 if cell == 0 else 0.2)
        for base in (6, 21, 36, 51, 66):
            torch.testing.assert_close(got[row, base : base + 3], vertex_off[row], atol=1e-5, rtol=0)
            torch.testing.assert_close(got[row, base + 3 : base + 15], local, atol=0, rtol=0)
        # First four neighbor triples (81:93). Columns 95-98 are then overwritten
        # with sin/cos of lat/lon (production layout).
        for k in range(4):
            col = 81 + 3 * k
            torch.testing.assert_close(got[row, col : col + 3], neighbor_off[row], atol=1e-5, rtol=0)

    # Production overwrites these four columns after the neighbor block.
    torch.testing.assert_close(got[:, 98], torch.sin(coords[:, 0]))
    torch.testing.assert_close(got[:, 97], torch.cos(coords[:, 0]))
    torch.testing.assert_close(got[:, 96], torch.sin(coords[:, 1]))
    torch.testing.assert_close(got[:, 95], torch.cos(coords[:, 1]))


def test_get_target_coords_local_empty():
    empty = torch.zeros((0, 2))
    counts = torch.zeros(12 * 4**2, dtype=torch.int32)
    dummy_rots = [torch.eye(3).unsqueeze(0).repeat(counts.shape[0], 1, 1) for _ in range(5)]
    verts_local = torch.zeros((counts.shape[0], 5, 12))
    nctrs = torch.zeros((8, counts.shape[0], 3))
    geo, times = torch.zeros((0, 0)), torch.zeros((0, 5))

    got = get_target_coords_local(1.0, 2, counts, empty, geo, times, dummy_rots, verts_local, nctrs)
    old = _get_target_coords_local_original(
        1.0, 2, counts, empty, geo, times, dummy_rots, verts_local, nctrs
    )
    assert got.numel() == 0
    assert old.numel() == 0


CASES = [
    pytest.param(1, 0, 0, id="one_point"),
    pytest.param(200, 0, 0, id="dense_no_geoinfos"),
    pytest.param(200, 3, 1, id="dense_with_geoinfos"),
    pytest.param(7, 2, 2, id="sparse"),
]


@pytest.mark.parametrize("n_points, n_geo, seed", CASES)
def test_get_target_coords_local_matches_original(target_geometry, n_points, n_geo, seed):
    hl = target_geometry["hl"]
    coords, counts = _pack_coords_by_cell(_random_latlon(n_points, seed), hl)
    geo, times = _geo_times(coords.shape[0], n_geo, seed + 10)

    kwargs = dict(
        stream_id=3.0,
        hlc=hl,
        masked_points_per_cell=counts,
        coords=coords,
        target_geoinfos=geo,
        target_times=times,
        verts_rots=target_geometry["verts_rots"],
        verts_local=target_geometry["verts_local"],
        nctrs=target_geometry["nctrs"],
    )
    old = _get_target_coords_local_original(**kwargs)
    new = get_target_coords_local(**kwargs)
    assert old.shape == new.shape
    torch.testing.assert_close(new, old, atol=0.0, rtol=0.0, equal_nan=True)
