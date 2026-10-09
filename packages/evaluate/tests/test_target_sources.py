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

from weathergen.evaluate.io.data.target_sources import AnemoiTargetSource, TargetRequest

N_DATES, N_ENS, N_GRID = 24, 2, 50
ALL_VARS = ["2t", "10u", "10v", "msl", "z_500"]
DATES = np.arange("2023-01-01T00", N_DATES, dtype="datetime64[h]").astype("datetime64[s]")
RNG = np.random.default_rng(0)
LATS = RNG.uniform(-90, 90, N_GRID)
LONS = RNG.uniform(0, 360, N_GRID)
DATA = RNG.standard_normal((N_DATES, len(ALL_VARS), N_ENS, N_GRID)).astype(np.float32)


class FakeAnemoiDataset:
    """Minimal stand-in for an anemoi dataset; records every slice read."""

    variables, dates, latitudes, longitudes, shape = ALL_VARS, DATES, LATS, LONS, DATA.shape

    def __init__(self, reads):
        self.reads = reads

    def __getitem__(self, key):
        self.reads.append((key.start, key.stop))
        return DATA[key]


def make_source(**kwargs):
    reads = []
    source = AnemoiTargetSource(["fake.zarr"], **kwargs)
    source._open = lambda: FakeAnemoiDataset(reads)
    return source, reads


def expected(date_idx, grid_idx, channels, member=0):
    out = np.full((len(grid_idx), len(channels)), np.nan, dtype=np.float32)
    for c, ch in enumerate(channels):
        if ch in ALL_VARS:
            out[:, c] = DATA[date_idx, ALL_VARS.index(ch), member, grid_idx]
    return out


def test_substeps_and_missing_channel():
    source, _ = make_source()
    channels = ["msl", "2t", "not_in_dataset"]
    requests = [TargetRequest(0, k, N_GRID, DATES[5 + k]) for k in range(3)]
    for k, target in enumerate(source.fill_targets(requests, channels)):
        np.testing.assert_array_equal(target, expected(5 + k, np.arange(N_GRID), channels))


def test_each_date_read_once_and_cached_across_calls():
    source, reads = make_source(n_threads=1)
    # Sample s, fstep f -> date s + f: 4 samples x 3 fsteps share 6 dates.
    requests = [TargetRequest(s, f, N_GRID, DATES[s + f]) for s in range(4) for f in range(3)]
    targets = source.fill_targets(requests, ["2t"])
    assert reads == [(0, 6)]  # one contiguous slice
    for req, target in zip(requests, targets, strict=True):
        expect = expected(req.result_idx + req.target_idx, np.arange(N_GRID), ["2t"])
        np.testing.assert_array_equal(target, expect)

    reads.clear()
    source.fill_targets(requests[:3], ["2t"])
    assert reads == []


def test_runs_split_by_memory_budget():
    # A date read decompresses every variable; 8 dates over 4 threads -> runs of 2.
    bytes_per_date = len(ALL_VARS) * N_ENS * N_GRID * 4
    source, reads = make_source(n_threads=4, max_read_bytes=8 * bytes_per_date)
    date_idx = [0, 1, 2, 3, 4, 5, 10, 11, 20]
    requests = [TargetRequest(0, i, N_GRID, DATES[d]) for i, d in enumerate(date_idx)]
    targets = source.fill_targets(requests, ["10v"])
    assert sorted(reads) == [(0, 2), (2, 4), (4, 6), (10, 12), (20, 21)]
    for d, target in zip(date_idx, targets, strict=True):
        np.testing.assert_array_equal(target, expected(d, np.arange(N_GRID), ["10v"]))


def test_rows_with_own_times_and_repeated_grid():
    source, _ = make_source()
    row_dates = np.repeat([9, 2], N_GRID)  # the grid twice, each block at its own time
    (target,) = source.fill_targets([TargetRequest(0, 0, 2 * N_GRID, DATES[row_dates])], ["z_500"])
    np.testing.assert_array_equal(target[:N_GRID], expected(9, np.arange(N_GRID), ["z_500"]))
    np.testing.assert_array_equal(target[N_GRID:], expected(2, np.arange(N_GRID), ["z_500"]))
    (target,) = source.fill_targets([TargetRequest(0, 0, 2 * N_GRID, DATES[4])], ["z_500"])
    np.testing.assert_array_equal(target, expected(4, np.tile(np.arange(N_GRID), 2), ["z_500"]))


def test_ensemble_member_and_empty_request():
    source, _ = make_source(ensemble_member=1)
    req = TargetRequest(0, 0, N_GRID, DATES[3])
    empty = TargetRequest(0, 1, 0, np.array([], dtype="datetime64[s]"))
    target, empty_target = source.fill_targets([req, empty], ["2t"])
    np.testing.assert_array_equal(target, expected(3, np.arange(N_GRID), ["2t"], member=1))
    assert empty_target.shape == (0, 1)


def test_missing_date_or_grid_mismatch_raises():
    source, _ = make_source()
    with pytest.raises(ValueError, match="not found in anemoi dataset"):
        source.fill_targets(
            [TargetRequest(0, 0, N_GRID, np.datetime64("2030-01-01T00", "s"))], ["2t"]
        )
    with pytest.raises(ValueError, match="do not match"):
        source.fill_targets([TargetRequest(0, 0, N_GRID - 1, DATES[0])], ["2t"])
    # Grid interleaved by time (point-major): valid times change within a grid block.
    interleaved = DATES[np.tile([2, 9], N_GRID)]
    with pytest.raises(ValueError, match="change within"):
        source.fill_targets([TargetRequest(0, 0, 2 * N_GRID, interleaved)], ["2t"])


def test_from_inference_config_selects_anemoi_streams():
    cfg = {
        "data_path_anemoi": "/data",
        "streams": {
            "ERA5": {"type": "anemoi", "filenames": ["a.zarr", "b.zarr"]},
            "OPERAN": {"type": "anemoi_operan", "filenames": ["op.zarr"]},
            "OBS": {"type": "obs", "filenames": ["o.zarr"]},
        },
    }
    source = AnemoiTargetSource.from_inference_config(cfg, "ERA5", {"ensemble_member": 2})
    assert source.filenames == ["/data/a.zarr", "/data/b.zarr"]
    assert source.ensemble_member == 2
    assert AnemoiTargetSource.from_inference_config(cfg, "OPERAN").filenames == ["/data/op.zarr"]
    assert AnemoiTargetSource.from_inference_config(cfg, "OBS") is None
    assert AnemoiTargetSource.from_inference_config(cfg, "missing") is None
