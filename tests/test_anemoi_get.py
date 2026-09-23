# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Equivalence of the vectorized anemoi _get post-process vs the original transpose+slice."""

import numpy as np
import pytest

from weathergen.datasets.data_reader_anemoi import _take_var_axis


def _take_var_axis_original(arr, var_idx):
    data = arr.transpose([0, 2, 1]).reshape((arr.shape[0] * arr.shape[2], -1))
    return data[:, list(var_idx)]


@pytest.mark.parametrize("n_time,n_var,n_grid", [(1, 16, 64), (6, 113, 128), (6, 113, 1024)])
def test_take_var_axis_matches_original(n_time, n_var, n_grid):
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n_time, n_var, n_grid)).astype(np.float32)
    idx = [0, 3, 7, n_var - 1, 2]
    original = _take_var_axis_original(arr, idx)
    vectorized = _take_var_axis(arr, idx)
    np.testing.assert_array_equal(vectorized, original)
    assert vectorized.dtype == arr.dtype


def test_take_var_axis_empty_selection():
    arr = np.zeros((2, 8, 16), dtype=np.float32)
    original = _take_var_axis_original(arr, [])
    vectorized = _take_var_axis(arr, [])
    np.testing.assert_array_equal(vectorized, original)
    assert vectorized.shape == (32, 0)


def test_take_var_axis_numpy_idx():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((3, 10, 20)).astype(np.float32)
    idx = np.array([9, 0, 4], dtype=np.int64)
    np.testing.assert_array_equal(_take_var_axis(arr, idx), _take_var_axis_original(arr, idx))


def test_take_var_axis_known_two_times():
    """Hand-checked (time, var, grid) gather. Does not use the frozen original.

    Cube is 2 times × 3 vars × 2 grid points; ``arr[t, v, g] = 100*t + 10*v + g``.
    ``var_idx = [2, 0]`` keeps vars 2 then 0. Rows are time-major, then grid:

      t0 g0 -> [20, 0]
      t0 g1 -> [21, 1]
      t1 g0 -> [120, 100]
      t1 g1 -> [121, 101]
    """
    arr = np.array(
        [
            [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]],
            [[100.0, 101.0], [110.0, 111.0], [120.0, 121.0]],
        ],
        dtype=np.float32,
    )
    out = _take_var_axis(arr, [2, 0])
    expected = np.array(
        [
            [20.0, 0.0],
            [21.0, 1.0],
            [120.0, 100.0],
            [121.0, 101.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.float32
    assert out.flags.c_contiguous
