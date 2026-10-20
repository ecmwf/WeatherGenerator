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

from weathergen.common.io import IOReaderData
from weathergen.datasets.data_reader_base import ReaderData

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
