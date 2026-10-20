# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Reader data whose datapoints have no values, e.g. the targets of inference without targets."""

import numpy as np
import pytest
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import DTRange, ReaderData
from weathergen.datasets.multi_stream_data_sampler import collect_datasources
from weathergen.datasets.tokenizer_masking import readerdata_to_torch
from weathergen.datasets.tokenizer_utils import tokenize_spacetime

N_POINTS, N_CHANNELS = 5, 3


def _without_values(cls=ReaderData, n_points: int = N_POINTS):
    coords = np.stack([np.linspace(-80, 80, n_points), np.linspace(0, 300, n_points)], axis=1)
    return cls(
        coords=coords.astype(np.float32),
        geoinfos=np.zeros((n_points, 1), dtype=np.float32),
        data=np.zeros((0, N_CHANNELS), dtype=np.float32),
        datetimes=np.repeat(np.datetime64("2020-01-01T00:00"), n_points),
    )


def test_datapoints_without_values_are_not_empty():
    assert not _without_values(IOReaderData).is_empty()
    assert _without_values(IOReaderData, n_points=0).is_empty()


def test_create_and_combine_accept_datapoints_without_values():
    rdata = IOReaderData.create(_without_values())
    combined = IOReaderData.combine([rdata, _without_values(IOReaderData, n_points=2)])

    assert combined.coords.shape == (N_POINTS + 2, 2)
    assert combined.data.shape == (0, N_CHANNELS)


def test_create_rejects_a_mismatching_number_of_values():
    rdata = _without_values()
    rdata.data = np.zeros((N_POINTS - 1, N_CHANNELS), dtype=np.float32)
    with pytest.raises(AssertionError):
        IOReaderData.create(rdata)


def test_remove_nan_and_shuffle_keep_points_without_values():
    rdata = _without_values()
    rdata.coords[0, 0] = np.nan

    rdata = rdata.remove_nan_coords_and_geoinfos()
    assert rdata.coords.shape == (N_POINTS - 1, 2) and rdata.data.shape == (0, N_CHANNELS)

    rdata = rdata.shuffle(np.random.default_rng(0), shuffle=True, num_subset=2)
    assert rdata.coords.shape == (2, 2) and rdata.datetimes.shape == (2,)
    assert rdata.data.shape == (0, N_CHANNELS)


N_TIMES, N_VARS, N_GRID = 2, 4, 6
T0 = np.datetime64("2020-01-01T00:00", "ms")
STEP = np.timedelta64(6, "h")


class _Cube:
    """Stand-in for an anemoi dataset: (time, var, ensemble, grid) and its dates."""

    def __init__(self, readable: bool = True):
        self.dates = T0 + STEP * np.arange(4)
        rng = np.random.default_rng(0)
        self._values = rng.random((4, N_VARS, 1, N_GRID), dtype=np.float32)
        self.readable = readable

    def __getitem__(self, key):
        assert self.readable, "points without values must not read the data"
        return self._values[key]


def _anemoi_reader(ds: _Cube, geoinfo_idx: list[int]) -> DataReaderAnemoi:
    reader = object.__new__(DataReaderAnemoi)
    reader.stream_info = {"name": "ERA5"}
    reader.ds, reader.len = ds, len(ds.dates)
    reader.source_idx, reader.target_idx, reader.geoinfo_idx = [0, 1], [1, 2], geoinfo_idx
    reader._latlon = np.stack([np.linspace(-80, 80, N_GRID), np.linspace(0, 300, N_GRID)], axis=1)
    window = DTRange(T0 + STEP, T0 + STEP * (N_TIMES + 1))
    reader._get_dataset_idxs = lambda idx: (np.arange(1, N_TIMES + 1), window)
    return reader


@pytest.mark.parametrize("geoinfo_idx", [[], [3]])
def test_anemoi_target_points_match_the_target_data(geoinfo_idx):
    reader = _anemoi_reader(_Cube(), geoinfo_idx)
    target = reader.get_target(0)
    points = reader.get_target_points(0)

    np.testing.assert_array_equal(points.coords, target.coords)
    np.testing.assert_array_equal(points.geoinfos, target.geoinfos)
    np.testing.assert_array_equal(points.datetimes, target.datetimes)
    assert points.len() == N_TIMES * N_GRID
    assert points.data.shape == (0, len(reader.target_idx))


def test_anemoi_target_points_without_geoinfos_do_not_read():
    reader = _anemoi_reader(_Cube(readable=False), geoinfo_idx=[])

    assert reader.get_target_points(0).len() == N_TIMES * N_GRID


class _Stream:
    """Stand-in for a reader in collect_datasources: records which getter was used."""

    def __init__(self):
        self.stream_info = {"name": "ERA5"}
        self.calls = []

    def get_target(self, idx):
        self.calls.append("get_target")
        rdata = _without_values()
        rdata.data = np.ones((N_POINTS, N_CHANNELS), dtype=np.float32)
        return rdata

    def get_target_points(self, idx):
        self.calls.append("get_target_points")
        return _without_values()

    def normalize_target_channels(self, data):
        return data

    def normalize_geoinfos(self, geoinfos):
        return geoinfos


@pytest.mark.parametrize(
    ("with_values", "getter", "n_values"),
    [(True, "get_target", N_POINTS), (False, "get_target_points", 0)],
)
def test_collect_targets_reads_values_only_with_targets(with_values, getter, n_values):
    stream = _Stream()
    rng = np.random.default_rng(0)
    rdata = collect_datasources([stream], 0, "target", rng, with_values=with_values)

    assert stream.calls == [getter]
    assert not rdata.is_empty() and rdata.data.shape == (n_values, N_CHANNELS)


def test_spacetime_tokenization_of_points_without_values():
    rdata = readerdata_to_torch(IOReaderData.create(_without_values()))
    rdata.datetimes[1::2] = rdata.datetimes[0] + STEP
    with_values = readerdata_to_torch(IOReaderData.create(_without_values()))
    with_values.datetimes = rdata.datetimes
    with_values.data = torch.ones((N_POINTS, N_CHANNELS))

    idxs, lens = tokenize_spacetime(rdata, 4, 1)
    idxs_ref, lens_ref = tokenize_spacetime(with_values, 4, 1)

    assert lens == lens_ref and sum(map(sum, lens)) >= N_POINTS
    for cell, cell_ref in zip(idxs, idxs_ref, strict=True):
        assert all(torch.equal(t, t_ref) for t, t_ref in zip(cell, cell_ref, strict=True))
