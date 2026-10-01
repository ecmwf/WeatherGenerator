# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from weathergen.evaluate.io.data.target_sources import (
    AnemoiGridMatcher,
    AnemoiTargetSource,
    TargetRequest,
)

N_DATES, N_ENS, N_GRID = 24, 2, 50
ALL_VARS = ["2t", "10u", "10v", "msl", "z_500"]
DATES = np.arange("2023-01-01T00", N_DATES, dtype="datetime64[h]").astype("datetime64[s]")
RNG = np.random.default_rng(0)
LATS = RNG.uniform(-90, 90, N_GRID)
LONS = RNG.uniform(0, 360, N_GRID)
DATA = RNG.standard_normal((N_DATES, len(ALL_VARS), N_ENS, N_GRID)).astype(np.float32)


class FakeAnemoiDataset:
    """Minimal stand-in for an anemoi dataset; records every slice read."""

    def __init__(self, select=None, reads=None):
        # ``select`` returns variables in dataset order, not the requested order.
        self.variables = [v for v in ALL_VARS if select is None or v in select]
        self._var_idx = [ALL_VARS.index(v) for v in self.variables]
        self.dates = DATES
        self.latitudes = LATS
        self.longitudes = LONS
        self.shape = (N_DATES, len(self.variables), N_ENS, N_GRID)
        self.reads = reads if reads is not None else []

    def __getitem__(self, key):
        self.reads.append((key.start, key.stop))
        return DATA[key][:, self._var_idx]


def make_source(**kwargs):
    reads = []
    source = AnemoiTargetSource(["fake.zarr"], **kwargs)
    source._open = lambda select=None: FakeAnemoiDataset(select, reads)
    return source, reads


def expected(date_idx, grid_idx, channels, member=0):
    out = np.full((len(grid_idx), len(channels)), np.nan, dtype=np.float32)
    for c, ch in enumerate(channels):
        if ch in ALL_VARS:
            out[:, c] = DATA[date_idx, ALL_VARS.index(ch), member, grid_idx]
    return out


def gridded_coords(n_times):
    return np.tile(np.stack([LATS, LONS], axis=1), (n_times, 1))


def test_gridded_substeps_identity_grid():
    source, _ = make_source()
    coords = gridded_coords(3)
    channels = ["msl", "2t"]  # different order than the dataset
    requests = [
        TargetRequest(0, k, N_GRID, DATES[5 + k], coords, np.arange(k * N_GRID, (k + 1) * N_GRID))
        for k in range(3)
    ]
    targets = source.fill_targets(requests, channels)
    for k, target in enumerate(targets):
        np.testing.assert_array_equal(target, expected(5 + k, np.arange(N_GRID), channels))


def test_missing_channel_is_nan():
    source, _ = make_source()
    req = TargetRequest(0, 0, N_GRID, DATES[0], gridded_coords(1))
    (target,) = source.fill_targets([req], ["2t", "not_in_dataset"])
    np.testing.assert_array_equal(target[:, 0], DATA[0, 0, 0])
    assert np.isnan(target[:, 1]).all()


def test_each_date_read_once_and_cached_across_calls():
    source, reads = make_source(n_threads=1)
    coords = gridded_coords(1)
    # Sample s, fstep f -> date s + f: 4 samples x 3 fsteps share 6 dates.
    requests = [
        TargetRequest(s, f, N_GRID, DATES[s + f], coords) for s in range(4) for f in range(3)
    ]
    targets = source.fill_targets(requests, ["2t"])
    read_dates = [d for lo, hi in reads for d in range(lo, hi)]
    assert sorted(read_dates) == list(range(6))
    assert reads == [(0, 6)]  # one contiguous slice
    for req, target in zip(requests, targets, strict=True):
        s, f = req.result_idx, req.target_idx
        np.testing.assert_array_equal(target, expected(s + f, np.arange(N_GRID), ["2t"]))

    reads.clear()
    source.fill_targets(requests[:3], ["2t"])
    assert reads == []


def test_runs_split_by_memory_budget_and_threads():
    # A date read always decompresses every variable of the date.
    bytes_per_date = len(ALL_VARS) * N_ENS * N_GRID * 4
    # Budget of 8 dates over 4 threads -> runs of at most 2 consecutive dates.
    source, reads = make_source(n_threads=4, max_read_bytes=8 * bytes_per_date)
    assert source._read_plan(np.arange(3), bytes_per_date)[1] == 2
    coords = gridded_coords(1)
    date_idx = [0, 1, 2, 3, 4, 5, 10, 11, 20]
    requests = [TargetRequest(0, i, N_GRID, DATES[d], coords) for i, d in enumerate(date_idx)]
    targets = source.fill_targets(requests, ["10v"])
    assert sorted(reads) == [(0, 2), (2, 4), (4, 6), (10, 12), (20, 21)]
    for d, target in zip(date_idx, targets, strict=True):
        np.testing.assert_array_equal(target, expected(d, np.arange(N_GRID), ["10v"]))


def test_scatter_rows_with_own_times_use_nearest_grid_point():
    source, _ = make_source()
    pick = np.array([7, 3, 3, 42, 0])
    # Small offsets: the nearest grid point is still the picked one.
    coords = np.stack([LATS[pick] + 1e-3, LONS[pick] - 360.0 + 1e-3], axis=1)
    row_dates = np.array([2, 9, 2, 9, 4])
    req = TargetRequest(0, 0, len(pick), DATES[row_dates], coords)
    (target,) = source.fill_targets([req], ["z_500", "10u"])
    for r in range(len(pick)):
        np.testing.assert_array_equal(
            target[r], expected(row_dates[r], pick[r : r + 1], ["z_500", "10u"])[0]
        )


def test_ensemble_member_and_empty_request():
    source, _ = make_source(ensemble_member=1)
    req = TargetRequest(0, 0, N_GRID, DATES[3], gridded_coords(1))
    empty = TargetRequest(0, 1, 0, np.datetime64("NaT"), gridded_coords(1))
    target, empty_target = source.fill_targets([req, empty], ["2t"])
    np.testing.assert_array_equal(target, expected(3, np.arange(N_GRID), ["2t"], member=1))
    assert empty_target.shape == (0, 1)


def test_missing_date_raises():
    source, _ = make_source()
    req = TargetRequest(0, 0, N_GRID, np.datetime64("2030-01-01T00", "s"), gridded_coords(1))
    with pytest.raises(ValueError, match="not found in anemoi dataset"):
        source.fill_targets([req], ["2t"])


def test_grid_matcher_identity_needs_no_tree():
    matcher = AnemoiGridMatcher(LATS, LONS)
    idx = matcher.indices(gridded_coords(2))
    np.testing.assert_array_equal(idx, np.tile(np.arange(N_GRID), 2))
    assert matcher._tree is None


def test_from_inference_config_selects_anemoi_streams():
    cfg = {
        "data_path_anemoi": "/data",
        "streams": {
            "ERA5": {"type": "anemoi", "filenames": ["a.zarr", "b.zarr"]},
            "OBS": {"type": "obs", "filenames": ["o.zarr"]},
        },
    }
    source = AnemoiTargetSource.from_inference_config(cfg, "ERA5", {"ensemble_member": 2})
    assert source.filenames == ["/data/a.zarr", "/data/b.zarr"]
    assert source.ensemble_member == 2
    assert AnemoiTargetSource.from_inference_config(cfg, "OBS") is None
    assert AnemoiTargetSource.from_inference_config(cfg, "missing") is None
