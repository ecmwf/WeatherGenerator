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

import numpy as np
import pandas as pd
from numpy import datetime64, timedelta64
from numpy.typing import NDArray

from weathergen.utils.better_abc import ABCMeta, abstract_attribute

_logger = logging.getLogger(__name__)

# The numpy date time 64 time (nanosecond precision)
type NPDT64 = datetime64
# The numpy delta time 64 time (nanosecond precision)
type NPTDel64 = timedelta64

type DType = np.float32  # The type for the data in the datasets.

"""
The type for indexing into datasets. It is a multiple of hours.
"""
type TIndex = np.int64


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
    s_str = str(s)

    supported_formats = [
        "%Y%m%d%H%M%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]

    for fmt in supported_formats:
        try:
            dt_obj = datetime.datetime.strptime(s_str, fmt)
            return np.datetime64(dt_obj)
        except ValueError:
            pass

    raise ValueError(f"Unable to parse the date string '{s}'. Original string might be invalid.")


def str_to_timedelta(s: str | datetime.timedelta) -> pd.Timedelta:
    """
    Convert a string or datetime.timedelta object to a pd.Timedelta object.
    The string format is expected to be "HH:MM:SS".
    Hours are not limited to two digits. Minutes and seconds must be in the range 0-59.
    """

    if not isinstance(s, str) and not isinstance(s, datetime.timedelta):
        raise TypeError("Input must be a string or a datetime.timedelta object")
    if isinstance(s, datetime.timedelta):
        # If input is a timedelta object, convert it directly to pd.Timedelta
        return pd.Timedelta(s)
    if isinstance(s, str):
        # ensure that the string is in "HH:MM:SS" format
        parts = s.split(":")
        if not len(parts) == 3:
            raise ValueError("String must be in 'HH:MM:SS' format")
        if not all(part.isdigit() for part in parts):
            raise ValueError("String must be in 'HH:MM:SS' format")
        # ensure that minutes and seconds do not exceed 59
        if int(parts[1]) > 59 or int(parts[2]) > 59:
            raise ValueError("Minutes and seconds must be in the range 0-59")
    return pd.to_timedelta(s)


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
        start :
            start time
        end :
            end time
        t_window_len :
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

        idx_start: TIndex = np.int64(0)
        idx_end = np.int64((self.t_end - self.t_start) // self.t_window_step)
        assert idx_start <= idx_end, f"time window idxs invalid: {idx_start} <= {idx_end}"

        return TimeIndexRange(idx_start, idx_end)

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
    is_spoof: bool = False

    @staticmethod
    def empty(num_data_fields: int, num_geo_fields: int) -> "ReaderData":
        """
        Create an empty ReaderData object

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
        return self.len() == 0

    def len(self):
        """
        Length of data

        Returns
        -------
        length of data
        """
        return len(self.data)

    def remove_nan_coords(self) -> "ReaderData":
        """
        Remove all data points where coords are NaN

        Returns
        -------
        self
        """
        idx_valid = ~np.isnan(self.coords)
        # filter should be if any (of the two) coords is NaN
        idx_valid = np.logical_and(idx_valid[:, 0], idx_valid[:, 1])

        # apply
        return ReaderData(
            self.coords[idx_valid],
            self.geoinfos[idx_valid],
            self.data[idx_valid],
            self.datetimes[idx_valid],
        )


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

    assert rdata.coords.ndim == 2, f"coords must be 2D {rdata.coords.shape}"
    assert rdata.coords.shape[1] == 2, (
        f"coords must have 2 columns (lat, lon), got {rdata.coords.shape}"
    )
    assert rdata.geoinfos.ndim == 2, f"geoinfos must be 2D, got {rdata.geoinfos.shape}"
    assert rdata.data.ndim == 2, f"data must be 2D {rdata.data.shape}"
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

    assert np.logical_and(rdata.datetimes >= dtr.start, rdata.datetimes < dtr.end).all(), (
        f"datetimes for data points violate window {dtr}."
    )


class DataReaderBase(metaclass=ABCMeta):
    """
    Base class for data readers.

    Coordinates must be provided in standard geographical format:
    latitude in degrees from -90 (South) to +90 (North),
    and longitude in degrees from -180 (West) to +180 (East).
    """

    # The fields that need to be set by the child classes
    source_channels: list[str] = abstract_attribute()
    target_channels: list[str] = abstract_attribute()
    geoinfo_channels: list[str] = abstract_attribute()
    source_idx: list[int] = abstract_attribute()
    target_idx: list[int] = abstract_attribute()
    geoinfo_idx: list[int] = abstract_attribute()
    target_channel_weights: list[float] = abstract_attribute()

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        stream_info: dict,
    ) -> None:
        """
        Parameters
        ----------
        tw_handler :
            time window handler
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        self.time_window_handler = tw_handler
        self.stream_info = stream_info
        self.target_channel_weights = None

    def init_empty(self) -> None:
        """
        Initialize
        """
        # pylint: disable=attribute-defined-outside-init
        self.source_channels = []
        self.target_channels = []
        self.geoinfo_channels = []
        self.source_idx = []
        self.target_idx = []
        self.geoinfo_idx = []
        self.target_channel_weights = []

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

    def parse_target_channel_weights(
        self,
    ) -> list[float] | None:
        target_channel_weights = [
            self.stream_info["channel_weights"].get(ch, 1.0)
            if self.stream_info.get("channel_weights", None)
            else 1.0
            for ch in self.target_channels
        ]

        if self.stream_info.get("channel_weights", None) is not None:
            # Check whether all given channel_weights could be matched to a channel.
            ch_unmatched = [
                ch for ch in self.stream_info["channel_weights"] if ch not in self.target_channels
            ]
            if len(ch_unmatched) > 0:
                _logger.info(
                    f"Unmatched channel_weights in {self.stream_info.name}: {ch_unmatched}"
                )

        return target_channel_weights

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

    @staticmethod
    def _normalize(
        data: NDArray[DType],
        idx: list[int],
        mean: dict[int, float],
        stdev: dict[int, float],
        name: str,
    ) -> NDArray[DType]:
        """
        Helper function to normalize data

        Parameters
        ----------
        data :
            data to be normalized
        idx :
            indices of channels to be normalized
        mean :
            mean values for channels
        stdev :
            standard deviation values for channels
        name :
            name of the data (for error messages)

        Returns
        -------
        Normalized data
        """
        if data.shape[-1] != len(idx):
            raise ValueError(
                f"incorrect number of {name} channels: expected {len(idx)}, got {data.shape[-1]}"
            )
        for i, ch in enumerate(idx):
            data[..., i] = (data[..., i] - mean[ch]) / stdev[ch]

        return data

    @staticmethod
    def _denormalize(
        data: NDArray[DType],
        idx: list[int],
        mean: dict[int, float],
        stdev: dict[int, float],
        name: str,
    ) -> NDArray[DType]:
        """
        Helper function to denormalize data

        Parameters
        ----------
        data :
            data to be denormalized
        idx :
            indices of channels to be denormalized
        mean :
            mean values for channels
        stdev :
            standard deviation values for channels
        name :
            name of the data (for error messages)

        Returns
        -------
        Denormalized data
        """
        if data.shape[-1] != len(idx):
            raise ValueError(
                f"incorrect number of {name} channels: expected {len(idx)}, got {data.shape[-1]}"
            )
        for i, ch in enumerate(idx):
            data[..., i] = (data[..., i] * stdev[ch]) + mean[ch]

        return data

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
        source :
            data to be normalized

        Returns
        -------
        Normalized source data
        """
        return self._normalize(source, self.source_idx, self.mean, self.stdev, "source")

    def normalize_target_channels(self, target: NDArray[DType]) -> NDArray[DType]:
        """
        Normalize target channels

        Parameters
        ----------
        target :
            data to be normalized

        Returns
        -------
        Normalized target data
        """
        return self._normalize(target, self.target_idx, self.mean, self.stdev, "target")

    def denormalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        """
        Denormalize source channels

        Parameters
        ----------
        source :
            data to be denormalized

        Returns
        -------
        Denormalized source data
        """
        return self._denormalize(source, self.source_idx, self.mean, self.stdev, "source")

    def denormalize_target_channels(self, target: NDArray[DType]) -> NDArray[DType]:
        """
        Denormalize target channels

        Parameters
        ----------
        target :
            data to be denormalized (target or pred)

        Returns
        -------
        Denormalized target data
        """
        return self._denormalize(target, self.target_idx, self.mean, self.stdev, "target")


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

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        stream_info: dict,
        data_start_time: NPDT64 | None = None,
        data_end_time: NPDT64 | None = None,
        period: NPTDel64 | None = None,
    ) -> None:
        """
        Parameters
        ----------
        tw_handler :
            time window handler
        stream_info :
            information about stream
        data_start_time :
            start time of dataset
        end_start_time :
            end time of dataset
        period :
            period / frequency of dataset

        Returns
        -------
        None
        """

        super().__init__(tw_handler, stream_info)
        self.data_start_time = data_start_time or tw_handler.t_start
        self.data_end_time = data_end_time
        self.period = period

    def _get_dataset_idxs(self, idx: TIndex) -> tuple[NDArray[np.int64], DTRange]:
        """
        Get dataset indexes for a given time window index.

        Parameters
        ----------
        idx : TIndex
            Index of the time window.

        Returns
        -------
        NDArray[np.int64]
            Array of dataset indexes corresponding to the time window.
        """
        return get_dataset_indexes_timestep(
            self.data_start_time,
            self.data_end_time,
            self.period,
            idx,
            self.time_window_handler,
        )


# to avoid rounding issues
# The basic time precision is 1 millisecond.
# This should support all datasets (the small period expected is 1 second)
t_epsilon = np.timedelta64(1, "ms")


def get_dataset_indexes_timestep(
    data_start_time: NPDT64,
    data_end_time: NPDT64 | None,
    period: NPTDel64,
    idx: TIndex,
    tw_handler: TimeWindowHandler,
) -> tuple[NDArray[np.int64], DTRange]:
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

    Returns
    -------
    NDArray[np.int64]
        Array of dataset indexes corresponding to the time window.
    """

    # Function is separated from the class to allow testing without instantiating the class.
    dtr = tw_handler.window(idx)
    # If there is no or only marginal overlap with the dataset, return empty index ranges
    if (
        not data_start_time
        or not data_end_time
        or dtr.end < data_start_time
        or dtr.start > data_end_time
        or dtr.start < data_start_time
        or dtr.end > data_end_time
        or (data_end_time is not None and dtr.start > data_end_time)
    ):
        return (np.array([], dtype=np.int64), dtr)

    # relative time in dataset
    delta_t_start = dtr.start - data_start_time
    delta_t_end = dtr.end - data_start_time - t_epsilon
    assert isinstance(delta_t_start, timedelta64), "delta_t_start must be timedelta64"
    start_didx = delta_t_start // period
    end_didx = delta_t_end // period

    # adjust start_idx if not exactly on start time
    if (delta_t_start % period) > np.timedelta64(0, "s"):
        # empty window in between two timesteps
        if start_didx == end_didx:
            return (np.array([], dtype=np.int64), dtr)
        start_didx += 1

    end_didx = start_didx + int((dtr.end - dtr.start - t_epsilon) / period)

    return (np.arange(start_didx, end_didx + 1, dtype=np.int64), dtr)
