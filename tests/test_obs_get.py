# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Equivalence of one obs row-slice vs three orthogonal column reads."""

from pathlib import Path

import numpy as np
import pytest

from weathergen.datasets.data_reader_obs import _columns_from_block


def _oindex_triple(block, coords_idx, geoinfo_idx, channels_idx):
    coords = block[:, coords_idx]
    geoinfos = (
        block[:, geoinfo_idx]
        if len(geoinfo_idx) > 0
        else np.zeros((block.shape[0], 0), np.float32)
    )
    data = block[:, channels_idx]
    return coords, geoinfos, data


@pytest.mark.parametrize("n_rows,n_cols", [(0, 8), (1, 8), (128, 24), (400, 34)])
def test_columns_from_block_matches_oindex(n_rows, n_cols):
    rng = np.random.default_rng(0)
    block = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    block[::7] = np.nan
    coords_idx = [1, 2]
    geoinfo_idx = [3, 4, 5]
    channels_idx = np.array([6, 7, 4], dtype=np.int64)
    original = _oindex_triple(block, coords_idx, geoinfo_idx, channels_idx)
    vectorized = _columns_from_block(block, coords_idx, geoinfo_idx, channels_idx)
    for a, b in zip(original, vectorized, strict=True):
        assert a.dtype == b.dtype
        assert np.array_equal(a, b, equal_nan=True)


def test_columns_from_block_empty_geoinfo():
    block = np.arange(20, dtype=np.float32).reshape(5, 4)
    original = _oindex_triple(block, [0, 1], [], [2, 3])
    vectorized = _columns_from_block(block, [0, 1], [], [2, 3])
    for a, b in zip(original, vectorized, strict=True):
        np.testing.assert_array_equal(a, b)
    assert vectorized[1].shape == (5, 0)
    assert vectorized[1].dtype == np.float32


@pytest.mark.skipif(
    not Path(
        "/e/data1/slmet/ml_training/observations-ea-ofb-0001-1979-2025-combined-surface-v5.zarr"
    ).is_dir(),
    reason="obs zarr not available",
)
def test_real_zarr_row_slice_matches_triple_oindex():
    import zarr

    path = "/e/data1/slmet/ml_training/observations-ea-ofb-0001-1979-2025-combined-surface-v5.zarr"
    g = zarr.open(path, mode="r")
    d = g["data"]
    h0 = int(
        (np.datetime64("2023-06-15T00") - np.datetime64("1970-01-01T00")) / np.timedelta64(1, "h")
    )
    hr = g["idx_197001010000_1"]
    start, end = int(hr[h0]), int(hr[h0 + 6])
    coords_idx, geoinfo_idx, channels_idx = [1, 2], [4, 5], [6, 7, 8, 9, 10]
    orig_coords = d.oindex[start:end, coords_idx]
    orig_geo = d.oindex[start:end, geoinfo_idx]
    orig_data = d.oindex[start:end, channels_idx]
    coords, geo, data = _columns_from_block(
        np.asarray(d[start:end]), coords_idx, geoinfo_idx, channels_idx
    )
    np.testing.assert_array_equal(coords, orig_coords)
    np.testing.assert_array_equal(geo, orig_geo)
    assert np.array_equal(data, orig_data, equal_nan=True)
