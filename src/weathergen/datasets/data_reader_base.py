# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy import datetime64, timedelta64
from numpy.typing import NDArray

from weathergen.utils.better_abc import ABCMeta, abstract_attribute

_logger = logging.getLogger(__name__)

# The numpy date time 64 time (nanosecond precision)
NPDT64: TypeAlias = datetime64
# The numpy delta time 64 time (nanosecond precision)
NPTDel64: TypeAlias = timedelta64

DType: TypeAlias = np.float32  # The type for the data in the datasets.

"""
The type for indexing into datasets. It is a multiple of hours.
"""
TIndex: TypeAlias = np.int32


_DT_ZERO = np.datetime64("1850-01-01T00:00")


@dataclass
class TimeIndexRange:
    """
    Defines a time window for indexing into datasets.

    It is defined as number of hours since the start of the dataset.
    """

    start: TIndex
    end: TIndex


@dataclass
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


def str_to_datetime64(s: str | int | NPDT64) -> NPDT64:
    """
    Convert a string to a numpy datetime64 object.
    """
    if isinstance(s, datetime64):
        return s
    format_str = "%Y%m%d%H%M%S"
    return np.datetime64(datetime.datetime.strptime(str(s), format_str))


class TimeWindowHandler:
    """
    Handler for time windows and translation of indices to times
    """

    def __init__(
        self,
        t_start: str | int | NPDT64,
        t_end: str | int | NPDT64,
        t_window_len_hours: int,
        t_window_step_hours: int,
    ):
        """
        Parameters
        ----------
                start : int
            start time
        end : int
            end time
        t_window_len : int
            length of data window
        t_window_step :
            delta hours between start times of windows

        """
        self.t_start: NPDT64 = str_to_datetime64(t_start)
        self.t_end: NPDT64 = str_to_datetime64(t_end)
        self.t_window_len: NPTDel64 = np.timedelta64(t_window_len_hours, "h")
        self.t_window_step: NPTDel64 = np.timedelta64(t_window_step_hours, "h")

        assert self.t_start < self.t_end, "end datetime has to be in the past of start datetime"
        assert self.t_start > _DT_ZERO, "start datetime has to be >= 1850-01-01T00:00."
        # TODO: check t_start and t_end are aligned with t_window_step and in hours

    def get_index_range(self) -> TimeIndexRange:
        """
        Temporal window corresponding to index

        Parameters
        ----------
        idx :
            index of temporal window

        Returns
        -------
            start and end of temporal window
        """

        idx_start: TIndex = np.int32(0)
        idx_end = np.int32((self.t_end - self.t_start) // self.t_window_step)
        assert idx_start <= idx_end, f"time window idxs invalid: {idx_start} <= {idx_end}"

        return TimeIndexRange(idx_start, idx_end)

    # TODO: unused
    # def get_absolute_index(self, idx: int) -> tuple[np.int64, np.int64]:
    #     """
    #     Absolute index (in sec) with respect to reference base time

    #     Parameters
    #     ----------
    #     idx :
    #         index of temporal window

    #     Returns
    #     -------
    #         start and end of absolute indices
    #     """

    #     idx_start = (self.t_start + self.t_window_step * idx - self.zero_time).seconds
    #     idx_end = (idx_start + self.t_window_len - self.zero_time).seconds
    #     assert idx_start <= idx_end, "time window idxs invalid"

    #     return idx_start, idx_end

    def window(self, idx: TIndex) -> DTRange:
        """
        Temporal window corresponding to index

        Parameters
        ----------
        idx :
            index of temporal window

        Returns
        -------
            start and end of temporal window
        """

        t_start_win = self.t_start + self.t_window_step * idx
        t_end_win = t_start_win + self.t_window_len

        return DTRange(t_start_win, t_end_win)


@dataclass
class ReaderData:
    """
    Wrapper for return values from DataReader.get_source and DataReader.get_target
    """

    coords: NDArray[DType]
    geoinfos: NDArray[DType]
    data: NDArray[DType]
    datetimes: NDArray[NPDT64]

    @staticmethod
    def empty(num_data_fields: int, num_geo_fields: int) -> "ReaderData":
        """
        Create an empty ReaderData object

        Returns
        -------
        ReaderData
            Empty ReaderData object
        """
        # TODO: it should also get the right shapes for data and geoinfos
        return ReaderData(
            coords=np.zeros((0, 2), dtype=np.float32),
            geoinfos=np.zeros((0, num_geo_fields), dtype=np.float32),
            data=np.zeros((0, num_data_fields), dtype=np.float32),
            datetimes=np.zeros((0,), dtype=np.datetime64),
        )

    def is_empty(self):
        return self.len() == 0

    def len(self):
        """
        Length of data

        Returns
        -------
        length of data
        """
        return len(self.data)


def check_reader_data(rdata: ReaderData) -> None:
    """
    Check that ReaderData is valid

    Parameters
    ----------
    rdata : ReaderData
        ReaderData to check

    Returns
    -------
    None
    """

    assert rdata.coords.ndim == 2, f"coords must be 2D {rdata.coords.shape}"
    assert rdata.coords.shape[1] == 2, (
        f"coords must have 2 columns (lat, lon), got {rdata.coords.shape}"
    )
    assert rdata.geoinfos.ndim == 2, f"geoinfos must be 2D, got {rdata.geoinfos.shape}"
    assert rdata.data.ndim == 2, f"data must be 2D {rdata.data.shape}"
    assert rdata.data.shape[1] > 0, f"data must have at least one channel {rdata.data.shape}"
    assert rdata.datetimes.ndim == 1, f"datetimes must be 1D {rdata.datetimes.shape}"

    assert rdata.coords.shape[0] == rdata.data.shape[0], "coords and data must have same length"
    assert rdata.geoinfos.shape[0] == rdata.data.shape[0], "geoinfos and data must have same length"

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


class DataReaderBase(metaclass=ABCMeta):
    """
    Base class for data readers.
    """

    # The fields that need to be set by the child classes
    source_channels: list[str] = abstract_attribute()
    target_channels: list[str] = abstract_attribute()
    geoinfo_channels: list[str] = abstract_attribute()

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
    ) -> None:
        """
        Parameters
        ----------
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        self.time_window_handler = tw_handler

        # variables that need to be set / properly initialized by child classes that provide
        # concrete implementation
        # TODO: move these fields to abstract fields

        self.source_idx = []
        self.target_idx = []
        self.geoinfo_idx = []

        self.mean = np.zeros(0)
        self.stdev = np.ones(0)
        self.mean_geoinfo = np.zeros(0)
        self.stdev_geoinfo = np.ones(0)

    @abstractmethod
    def length(self) -> int:
        """The length of this dataset. Must be constant."""
        pass

    def __len__(self) -> int:
        """
        Length of dataset

        Parameters
        ----------
        None

        Returns
        -------
        length of dataset
        """

        return self.length()

    def get_source(self, idx: TIndex) -> ReaderData:
        """
        Get source data for idx

        Parameters
        ----------
        idx : int
            Index of temporal window

        Returns
        -------
        source data (coords, geoinfos, data, datetimes)
        """

        rdata = self._get(idx, self.source_idx)

        # TODO: check that geocoords adhere to conventions in debug mode

        return rdata

    def get_target(self, idx: TIndex) -> ReaderData:
        """
        Get target data for idx

        Parameters
        ----------
        idx : int
            Index of temporal window

        Returns
        -------
        target data (coords, geoinfos, data, datetimes)
        """

        rdata = self._get(idx, self.target_idx)

        # TODO: check that geocoords adhere to conventions in debug mode

        return rdata

    @abstractmethod
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for window

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels

        Returns
        -------
        data (coords, geoinfos, data, datetimes)
        """

        raise NotImplementedError()

    def get_source_num_channels(self) -> int:
        """
        Get number of source channels

        Parameters
        ----------
        None

        Returns
        -------
        number of source channels
        """
        return len(self.source_idx)

    def get_target_num_channels(self) -> int:
        """
        Get number of target channels

        Parameters
        ----------
        None

        Returns
        -------
        number of target channels
        """
        return len(self.target_idx)

    def get_coords_size(self) -> int:
        """
        Get size of coords

        Parameters
        ----------
        None

        Returns
        -------
        size of coords
        """
        return 2

    def get_geoinfo_size(self) -> int:
        """
        Get size of geoinfos

        Parameters
        ----------
        None

        Returns
        -------
        size of geoinfos
        """
        return len(self.geoinfo_idx)

    def normalize_coords(self, coords: NDArray[DType]) -> NDArray[DType]:
        """
        Normalize coordinates

        Parameters
        ----------
        coords :
            coordinates to be normalized

        Returns
        -------
        Normalized coordinates
        """
        coords[..., 0] = np.sin(np.deg2rad(coords[..., 0]))
        coords[..., 1] = np.sin(0.5 * np.deg2rad(coords[..., 1]))

        return coords

    def normalize_geoinfos(self, geoinfos: NDArray[DType]) -> NDArray[DType]:
        """
        Normalize geoinfos

        Parameters
        ----------
        geoinfos :
            geoinfos to be normalized

        Returns
        -------
        Normalized geoinfo
        """

        assert geoinfos.shape[-1] == len(self.geoinfo_idx), "incorrect number of geoinfo channels"
        for i, _ in enumerate(self.geoinfo_idx):
            geoinfos[..., i] = (geoinfos[..., i] - self.mean_geoinfo[i]) / self.stdev_geoinfo[i]

        return geoinfos

    def normalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        """
        Normalize source channels

        Parameters
        ----------
        data :
            data to be normalized

        Returns
        -------
        Normalized data
        """
        assert source.shape[-1] == len(self.source_idx), "incorrect number of source channels"
        for i, ch in enumerate(self.source_idx):
            source[..., i] = (source[..., i] - self.mean[ch]) / self.stdev[ch]

        return source

    def normalize_target_channels(self, target: NDArray[DType]) -> NDArray[DType]:
        """
        Normalize target channels

        Parameters
        ----------
        data :
            data to be normalized

        Returns
        -------
        Normalized data
        """
        assert target.shape[-1] == len(self.target_idx), "incorrect number of target channels"
        for i, ch in enumerate(self.target_idx):
            target[..., i] = (target[..., i] - self.mean[ch]) / self.stdev[ch]

        return target

    def denormalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        """
        Denormalize source channels

        Parameters
        ----------
        data :
            data to be denormalized

        Returns
        -------
        Denormalized data
        """
        assert source.shape[-1] == len(self.source_idx), "incorrect number of source channels"
        for i, ch in enumerate(self.source_idx):
            source[..., i] = (source[..., i] * self.stdev[ch]) + self.mean[ch]

        return source

    def denormalize_target_channels(self, data: NDArray[DType]) -> NDArray[DType]:
        """
        Denormalize target channels

        Parameters
        ----------
        data :
            data to be denormalized (target or pred)

        Returns
        -------
        Denormalized data
        """
        assert data.shape[-1] == len(self.target_idx), "incorrect number of target channels"
        for i, ch in enumerate(self.target_idx):
            data[..., i] = (data[..., i] * self.stdev[ch]) + self.mean[ch]

        return data


class DataReaderTimestep(DataReaderBase):
    """
    An abstract class for data readers that provide data at fixed time intervals.

    On top of all the fields to be defined in DataReaderBase, they must define the following fields:

    """

    # The start time of the dataset.
    data_start_time: NPDT64
    # The end time of the dataset (possibly none).
    data_end_time: NPDT64 | None = None
    # The period of the dataset, i.e. the time interval between two consecutive samples.
    # It is also called 'frequency' in Anemoi.
    period: NPTDel64
    # The subsampling rate, i.e. the number of samples to skip between two consecutive samples.
    window_subsampling_rate: int | None = None

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        data_start_time: NPDT64 | None,
        data_end_time: NPDT64 | None,
        period: NPTDel64,
        window_subsampling_rate: int | None,
    ) -> None:
        super().__init__(tw_handler)
        self.data_start_time = data_start_time or tw_handler.t_start
        self.data_end_time = data_end_time
        self.period = period
        self.window_subsampling_rate = window_subsampling_rate

        assert window_subsampling_rate is None or window_subsampling_rate > 0, (
            window_subsampling_rate
        )

    def _get_dataset_idxs(self, idx: TIndex) -> NDArray[np.int32]:
        """
        Get dataset indexes for a given time window index.

        Parameters
        ----------
        idx : TIndex
            Index of the time window.

        Returns
        -------
        NDArray[np.int32]
            Array of dataset indexes corresponding to the time window.
        """
        return get_dataset_indexes_periodic(
            self.data_start_time,
            self.data_end_time,
            self.period,
            idx,
            self.time_window_handler,
            self.window_subsampling_rate,
        )


# to avoid rounding issues
# The basic time precision is 1 second.
t_epsilon = np.timedelta64(1, "s")


def get_dataset_indexes_periodic(
    data_start_time: NPDT64,
    data_end_time: NPDT64 | None,
    period: NPTDel64,
    idx: TIndex,
    tw_handler: TimeWindowHandler,
    subsampling_rate: int | None,
) -> NDArray[np.int32]:
    """
    Get dataset indexes for a given time window index, when the dataset is periodic.

    Keeping this function separate for testing purposes.

    Parameters
    ----------
    data_start_time : NPDT64
        Start time of the dataset.
    data_end_time : NPDT64
        End time of the dataset (possibly none).
    period : NPTDel64
    idx : TIndex
        Index of the time window.
    tw_handler : TimeWindowHandler
        Handler for time windows.
    subsampling_rate : int | None
        Subsampling rate. If not or set to 1, then no subsampling is applied.

    Returns
    -------
    NDArray[np.int32]
        Array of dataset indexes corresponding to the time window.

    dataset_start_time and period must be aligned with the time window handler.
    """
    # Function is separated from the class to allow testing without instantiating the class.
    dtr = tw_handler.window(idx)
    # If there is no overlap with the dataset, return empty array
    # TODO: boundary conditions
    if dtr.end < data_start_time or (data_end_time is not None and dtr.start > data_end_time):
        return np.array([], dtype=np.int32)

    # For simplicity, assuming the window starts inside the dataset.
    assert dtr.start >= data_start_time, (dtr, data_start_time, data_end_time)
    # relative time in dataset
    delta_t_start = dtr.start - data_start_time
    assert isinstance(delta_t_start, timedelta64), "delta_t_start must not be None"
    start_didx = delta_t_start // period
    if (delta_t_start % period) > np.timedelta64(0, "s"):
        start_didx += 1

    end_didx = start_didx + int((dtr.end - dtr.start - t_epsilon) / period)
    if subsampling_rate is None:
        subsampling_rate = 1
    assert subsampling_rate > 0, ("Subsampling rate must be positive", subsampling_rate)

    # TODO: add subsampling (but not implemented yet in the code anyway)
    return np.arange(start_didx, end_didx + 1, step=subsampling_rate, dtype=np.int32)
