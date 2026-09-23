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
