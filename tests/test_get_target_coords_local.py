"""Equivalence of vectorized get_target_coords_local vs the original split/cat path.

The original implementation is frozen here. Production always packs target points
in healpix-cell order with ``masked_points_per_cell`` counts, matching TokenizerMasking.
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
from weathergen.datasets.utils import locs_to_cell_coords_ctrs, locs_to_ctr_coords, s2tor3


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
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - locs_to_cell_coords_ctrs(
        verts00_rots, tcs
    )

    zi = 3
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[0]

    zi = 15
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - locs_to_cell_coords_ctrs(
        verts10_rots, tcs
    )

    zi = 18
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[1]

    zi = 30
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - locs_to_cell_coords_ctrs(
        verts11_rots, tcs
    )

    zi = 33
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[2]

    zi = 45
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - locs_to_cell_coords_ctrs(
        verts01_rots, tcs
    )

    zi = 48
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[3]

    zi = 60
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - locs_to_cell_coords_ctrs(
        vertsmm_rots, tcs
    )

    zi = 63
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[4]

    tcs_ctrs = torch.cat([ref - torch.cat(locs_to_ctr_coords(c, tcs)) for c in nctrs], -1)
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

    tcs = torch.split(points, counts.tolist())
    assert torch.equal(got, locs_to_cell_coords_ctrs(cell_rots, tcs))


def test_rotate_points_per_cell_matches_locs_to_cell_coords_ctrs(target_geometry):
    hl = target_geometry["hl"]
    cell_rots = target_geometry["verts_rots"][0]
    coords, counts = _pack_coords_by_cell(_random_latlon(200, seed=0), hl)
    points = s2tor3(*theta_phi_to_standard_coords(coords)).to(torch.float32)
    tcs = torch.split(points, counts.tolist())

    got = _rotate_points_per_cell(cell_rots, points, counts)
    old = locs_to_cell_coords_ctrs(cell_rots, tcs)
    assert torch.equal(got, old)


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
