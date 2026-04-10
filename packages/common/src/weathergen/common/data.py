# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses
import logging

import numpy as np
from numpy import datetime64
from numpy.typing import NDArray

type DType = np.float32
type NPDT64 = datetime64

_logger = logging.getLogger(__name__)

_DT_ZERO = np.datetime64("1850-01-01T00:00")


@dataclasses.dataclass
class DTRange:
    """
    Defines a time window for indexing into datasets.

    It is defined as numpy datetime64 objects.
    """

    start: NPDT64
    end: NPDT64

    def __post_init__(self):
        assert self.start < self.end, "start time must be before end time"
        assert self.start > _DT_ZERO, "start time must be after 1850-01-01T00:00"


@dataclasses.dataclass
class ReaderData:
    """
    Wrapper for return values from DataReader.get_source and DataReader.get_target.
    """

    coords: NDArray[DType]
    geoinfos: NDArray[DType]
    data: NDArray[DType]
    datetimes: NDArray[NPDT64]
    is_spoof: bool = False

    def __len__(self):
        return len(self.data)

    @classmethod
    def create(cls, other: "ReaderData") -> "ReaderData":
        """
        Create an instance from another ReaderData instance.

        Parameters
        ------
            other: ReaderData
                Input data.

        Returns
        ------
            ReaderData:
                Has the same underlying data as `other`.
        """
        assert other is not None, "Input cannot be None."

        assert isinstance(other, ReaderData), (
            f"Expected input of type ReaderData. Got {type(other)}"
        )

        coords = np.asarray(other.coords)
        geoinfos = np.asarray(other.geoinfos)
        data = np.asarray(other.data)
        datetimes = np.asarray(other.datetimes)

        n_datapoints = len(data)

        assert coords.shape == (n_datapoints, 2), "number of datapoints do not match data"
        assert geoinfos.shape[0] == n_datapoints, "number of datapoints do not match data"
        assert datetimes.shape[0] == n_datapoints, "number of datapoints do not match data"

        return cls(**dataclasses.asdict(other))

    @classmethod
    def combine(cls, others: list["ReaderData"]) -> "ReaderData":
        """
        Create an instance from ReaderData instance by combining multiple ones.

        Parameters
        ------
            others: list[ReaderData]
                A list of input datas to combine.

        Returns
        ------
            ReaderData
                Instance with concatenated input data.
        """
        assert others is not None, "Input cannot be None."
        assert isinstance(others, list), f"Input must be a List. Got {type(others)}"
        assert len(others) > 0, len(others)

        first = others[0]
        coords = np.zeros((0, first.coords.shape[1]), dtype=first.coords.dtype)
        geoinfos = np.zeros((0, first.geoinfos.shape[1]), dtype=first.geoinfos.dtype)
        data = np.zeros((0, first.data.shape[1]), dtype=first.data.dtype)
        datetimes = np.array([], dtype=first.datetimes.dtype)
        is_spoof = True

        for item in others:
            n_datapoints = len(item.data)
            assert item.coords.shape == (n_datapoints, 2), "number of datapoints do not match"
            assert item.geoinfos.shape[0] == n_datapoints, "number of datapoints do not match"
            assert item.datetimes.shape[0] == n_datapoints, "number of datapoints do not match"

            coords = np.concatenate([coords, item.coords])
            geoinfos = np.concatenate([geoinfos, item.geoinfos])
            data = np.concatenate([data, item.data])
            datetimes = np.concatenate([datetimes, item.datetimes])
            is_spoof = is_spoof and item.is_spoof

        return cls(coords, geoinfos, data, datetimes, is_spoof)

    @staticmethod
    def empty(num_data_fields: int, num_geo_fields: int) -> "ReaderData":
        """
        Create an empty ReaderData object

        Parameters
        ------
            num_data_fields: int
                Number of data fields.
            num_geo_fields:
                Number of geo fields.

        Returns
        -------
        ReaderData
            Empty ReaderData object
        """
        return ReaderData(
            coords=np.zeros((0, 2), dtype=np.float32),
            geoinfos=np.zeros((0, num_geo_fields), dtype=np.float32),
            data=np.zeros((0, num_data_fields), dtype=np.float32),
            datetimes=np.zeros((0,), dtype=np.datetime64),
            is_spoof=False,
        )

    def is_empty(self):
        """
        Test if data object is empty
        """
        return len(self) == 0

    def remove_nan_coords_and_geoinfos(self) -> "ReaderData":
        """
        Remove all data points where coords or geoinfos contain NaN

        Returns
        -------
        self
        """
        idx_valid = ~np.isnan(self.coords)
        # filter should be if any (of the two) coords is NaN
        idx_valid = np.logical_and(idx_valid[:, 0], idx_valid[:, 1])

        # also filter rows where any geoinfo field is NaN
        idx_valid_geoinfos = ~np.isnan(self.geoinfos).any(axis=1)
        idx_valid = np.logical_and(idx_valid, idx_valid_geoinfos)

        # apply
        return ReaderData(
            self.coords[idx_valid],
            self.geoinfos[idx_valid],
            self.data[idx_valid],
            self.datetimes[idx_valid],
        )

    def shuffle(self, rng, shuffle: bool, num_subset: int) -> "ReaderData":
        """
        Drop a random subset of points as specified by num_subset
        num_subset = -1 indicates no points to be dropped

        Returns
        -------
        self
        """

        # nothing to be done
        if num_subset < 0 and shuffle is False:
            return self

        num_datapoints = self.coords.shape[0]
        if (num_datapoints == 0) or (num_datapoints < num_subset and shuffle is False):
            return self

        # only shuffling
        if num_subset == -1 and shuffle is True:
            num_subset = num_datapoints

        # ensure num_subset <= num_datapoints
        num_subset = min(num_subset, num_datapoints)

        idxs_subset = rng.choice(num_datapoints, num_subset, replace=False)
        if shuffle is False:
            idxs_subset = np.sort(idxs_subset)

        self.coords = self.coords[idxs_subset]
        self.geoinfos = self.geoinfos[idxs_subset]
        self.data = self.data[idxs_subset]
        self.datetimes = self.datetimes[idxs_subset]

        return self


def check_reader_data(rdata: ReaderData, dtr: DTRange) -> None:
    """
    Check that ReaderData is valid

    Parameters
    ----------
    rdata :
        ReaderData to check
    dtr :
        datetime range of window for which the rdata is valid

    Returns
    -------
    None
    """

    # Validate dimensions
    assert rdata.coords.ndim == 2, f"coords must be 2D {rdata.coords.shape}"
    assert rdata.coords.shape[1] == 2, (
        f"coords must have 2 columns (lat, lon), got {rdata.coords.shape}"
    )
    assert rdata.geoinfos.ndim == 2, f"geoinfos must be 2D, got {rdata.geoinfos.shape}"
    assert rdata.data.ndim == 2, f"data must be 2D {rdata.data.shape}"
    assert rdata.datetimes.ndim == 1, f"datetimes must be 1D {rdata.datetimes.shape}"

    # Validate consistency of lengths
    n_points = rdata.coords.shape[0]
    assert n_points == rdata.data.shape[0], "coords and data must have same length"
    assert n_points == rdata.geoinfos.shape[0], "geoinfos and data must have same length"

    # Check that all fields have the same length
    assert (
        rdata.coords.shape[0]
        == rdata.geoinfos.shape[0]
        == rdata.data.shape[0]
        == rdata.datetimes.shape[0]
    ), (
        f"coords, geoinfos, data and datetimes must have the same length "
        f"{rdata.coords.shape[0]}, {rdata.geoinfos.shape[0]}, {rdata.data.shape[0]}, "
        f"{rdata.datetimes.shape[0]}"
    )

    # Check that all datetimes fall within the specified range
    assert np.logical_and(rdata.datetimes >= dtr.start, rdata.datetimes < dtr.end).all(), (
        f"datetimes for data points violate window {dtr}."
    )
