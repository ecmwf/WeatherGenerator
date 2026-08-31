# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Equivalence of vectorized DataReaderBase._normalize vs the original channel loop."""

import numpy as np
import pytest

from weathergen.datasets.data_reader_base import DataReaderBase


def _normalize_original(data, idx, mean, stdev, name):
    """Pre-rewrite per-channel loop."""
    if data.shape[-1] != len(idx):
        raise ValueError(
            f"incorrect number of {name} channels: expected {len(idx)}, got {data.shape[-1]}"
        )
    for i, ch in enumerate(idx):
        data[..., i] = (data[..., i] - mean[ch]) / stdev[ch]
    return data


def _run_both(data, idx, mean, stdev):
    original = _normalize_original(data.copy(), idx, mean, stdev, "target")
    vectorized = DataReaderBase._normalize(data.copy(), idx, mean, stdev, "target")
    return original, vectorized


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalize_matches_original_array_stats(dtype):
    rng = np.random.default_rng(0)
    n_full, n_sel = 20, 7
    idx = [1, 3, 4, 8, 9, 15, 19]
    mean = rng.normal(size=n_full)
    stdev = rng.uniform(0.1, 2.0, size=n_full)
    data = rng.normal(size=(128, n_sel)).astype(dtype)
    original, vectorized = _run_both(data, idx, mean, stdev)
    np.testing.assert_array_equal(vectorized, original)
    assert vectorized.dtype == data.dtype


def test_normalize_matches_original_dict_stats():
    rng = np.random.default_rng(1)
    idx = [2, 5, 11]
    mean = {ch: float(rng.normal()) for ch in idx}
    stdev = {ch: float(rng.uniform(0.2, 3.0)) for ch in idx}
    data = rng.normal(size=(32, 16, 3)).astype(np.float32)
    original, vectorized = _run_both(data, idx, mean, stdev)
    np.testing.assert_array_equal(vectorized, original)


def test_normalize_matches_original_numpy_idx():
    rng = np.random.default_rng(2)
    idx = np.array([0, 2, 4], dtype=np.int64)
    mean = rng.normal(size=6)
    stdev = rng.uniform(0.5, 1.5, size=6)
    data = rng.normal(size=(64, 3)).astype(np.float32)
    original, vectorized = _run_both(data, idx, mean, stdev)
    np.testing.assert_array_equal(vectorized, original)


def test_normalize_matches_original_float32_stats():
    rng = np.random.default_rng(4)
    idx = [0, 1, 2]
    mean = rng.normal(size=3).astype(np.float32)
    stdev = rng.uniform(0.2, 2.0, size=3).astype(np.float32)
    data = rng.normal(size=(64, 3)).astype(np.float32)
    original, vectorized = _run_both(data, idx, mean, stdev)
    np.testing.assert_array_equal(vectorized, original)


def test_normalize_is_inplace():
    rng = np.random.default_rng(3)
    idx = [0, 1]
    mean = np.array([0.0, 1.0])
    stdev = np.array([1.0, 2.0])
    data = rng.normal(size=(8, 2)).astype(np.float32)
    out = DataReaderBase._normalize(data, idx, mean, stdev, "target")
    assert out is data


def test_normalize_empty_rows():
    idx = [1, 4]
    mean = np.arange(6, dtype=np.float64)
    stdev = np.ones(6)
    data = np.zeros((0, 2), dtype=np.float32)
    original, vectorized = _run_both(data, idx, mean, stdev)
    np.testing.assert_array_equal(vectorized, original)
    assert vectorized.shape == (0, 2)


def test_normalize_rejects_wrong_channel_count():
    data = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="incorrect number of target channels"):
        DataReaderBase._normalize(data, [0, 1], np.zeros(5), np.ones(5), "target")


def test_normalize_matches_original_across_row_chunks():
    rng = np.random.default_rng(5)
    n_sel = 12
    idx = list(range(n_sel))
    mean = rng.normal(size=n_sel)
    stdev = rng.uniform(0.2, 2.0, size=n_sel)
    data = rng.normal(size=(9000, n_sel)).astype(np.float32)
    original, vectorized = _run_both(data, idx, mean, stdev)
    np.testing.assert_array_equal(vectorized, original)
    out = DataReaderBase._normalize(data, idx, mean, stdev, "target")
    assert out is data


def test_normalize_matches_original_3d_array_stats():
    rng = np.random.default_rng(6)
    idx = [0, 2, 3, 5, 6, 7, 8, 9]
    mean = rng.normal(size=10)
    stdev = rng.uniform(0.3, 1.5, size=10)
    data = rng.normal(size=(4, 2048, len(idx))).astype(np.float32)
    original, vectorized = _run_both(data, idx, mean, stdev)
    np.testing.assert_array_equal(vectorized, original)


def test_normalize_matches_original_strided_rows():
    rng = np.random.default_rng(8)
    idx = list(range(12))
    mean = rng.normal(size=12)
    stdev = rng.uniform(0.2, 2.0, size=12)
    data = rng.normal(size=(2000, 12)).astype(np.float32)[::2]
    original, vectorized = _run_both(data, idx, mean, stdev)
    np.testing.assert_array_equal(vectorized, original)
    out = DataReaderBase._normalize(data, idx, mean, stdev, "target")
    assert out is data
