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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)


class TimeWindowHandler:
    """
    Handler for time windows and translation of indices to times
    """

    def __init__(self, t_start, t_end, t_window_len, t_window_step):
        self.zero_time = np.datetime64("1850-01-01T00:00")

        format_str = "%Y%m%d%H%M%S"
        self.t_start = np.datetime64(datetime.datetime.strptime(str(t_start), format_str))
        self.t_end = np.datetime64(datetime.datetime.strptime(str(t_end), format_str))
        self.t_window_len = np.timedelta64(t_window_len, "h")
        self.t_window_step = np.timedelta64(t_window_step, "h")

        assert self.t_start < self.t_end, "end datetime has to be in the past of start datetime"
        assert self.t_start > self.zero_time, "start datetime has to be >= 1850-01-01T00:00."

    def get_index_range(self) -> tuple[np.int64, np.int64]:
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

        idx_start = 0
        idx_end = (self.t_end - self.t_start) // self.t_window_step
        assert idx_start <= idx_end, "time window idxs invalid"

        return (idx_start, idx_end)

    def get_absolute_index(self, idx: int) -> tuple[np.int64, np.int64]:
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

        idx_start = (self.t_start + self.t_window_step * idx - self.zero_time).seconds
        idx_end = (idx_start + self.t_window_len - self.zero_time).seconds
        assert idx_start <= idx_end, "time window idxs invalid"

        return idx_start, idx_end

    def window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
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

        return (t_start_win, t_end_win)


@dataclass
class ReaderData:
    """
    Wrapper for return values from DataReader.get_source and DataReader.get_target
    """

    coords: np.array  # [np.float32]
    geoinfos: np.array  # [np.float32]
    data: np.array  # [np.float32]
    datetimes: np.array  # [np.datetime64]

    def __init__(self):
        self.coords = np.zeros([0, 0], dtype=np.float32)
        self.geoinfos = np.zeros([0, 0], dtype=np.float32)
        self.data = np.zeros([0, 0], dtype=np.float32)
        self.datetimes = np.zeros([0], dtype=np.datetime64)

    def is_empty(self):
        return self.data.shape[0] == 0


class DataReaderBase:
    "Base class for data readers"

    # @abstractmethod
    def __init__(
        self,
        start: int,
        end: int,
        t_window_len: int,
        t_window_step: int,
        filename: Path,
        stream_info: dict,
    ) -> None:
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
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        self.t_eps = np.timedelta64(1, "ms")

        self.t_window_len = np.timedelta64(t_window_len, "h")
        self.t_window_step = np.timedelta64(t_window_step, "h")

        self.time_window_handler = TimeWindowHandler(start, end, t_window_len, t_window_step)

        # variables that need to be set / properly initialized by child classes that provide
        # concrete implementation

        self.len = 0
        self.source_idx = []
        self.target_idx = []
        self.geoinfo_idx = []
        self.source_channels = []
        self.target_channels = []

        self.mean = np.zeros(0)
        self.stdev = np.ones(0)
        self.mean_geoinfo = np.zeros(0)
        self.stdev_geoinfo = np.ones(0)

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

        return self.len

    def get_source(self, idx: int) -> ReaderData:
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

    def get_target(self, idx: int) -> ReaderData:
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

    def _get(self, idx: int, channels_idx: np.array) -> ReaderData:
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

        return ReaderData()

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

    def normalize_coords(self, coords: np.array) -> np.array:
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

    def normalize_geoinfos(self, geoinfos: np.array) -> np.array:
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
        for i, ch in enumerate(self.geoinfo_idx):
            geoinfos[..., i] = (geoinfos[..., i] - self.mean_geoinfo[ch]) / self.stdev_geoinfo[ch]

        return geoinfos

    def normalize_source_channels(self, source: np.array) -> np.array:
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

    def normalize_target_channels(self, target: np.array) -> np.array:
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

    def denormalize_source_channels(self, source: np.array) -> np.array:
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

    def denormalize_target_channels(self, data: np.array) -> np.array:
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
    "Base class for data readers with regular time step"

    # @abstractmethod
    def __init__(
        self,
        start: int,
        end: int,
        t_window_len: int,
        t_window_step: int,
        filename: Path,
        stream_info: dict,
    ) -> None:
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
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        super().__init__(start, end, t_window_len, t_window_step, filename, stream_info)

        # variables need to be set by child classes
        self.frequency = np.datetime64(0, "s")
        self.ds_start_time = self.time_window_handler.zero_time
        self.ds_end_time = self.time_window_handler.zero_time

    def _get_dataset_idxs(self, idx) -> np.array:
        """
        Translate idx for time window to idxs into dataset

        Parameters
        ----------
        idx :
            index of temporal window

        Returns
        -------
            indices for dataset corresponding to idx
        """

        win_start, win_end = self.time_window_handler.window(idx)
        if win_start < self.ds_start_time or win_end > self.ds_end_time:
            return np.array([], dtype=np.int32)

        # relative time in dataset
        delta_t_start = win_start - self.ds_start_time

        # index to first step, taking potential misalignment of windows and steps into account

        s_idx = delta_t_start // self.frequency
        s_idx += 0 if (delta_t_start % self.frequency).seconds == 0 else 1
        # delta between first step and start of window
        delta_win_start = (s_idx * self.frequency) - delta_t_start
        # self.t_eps implements that windows are excluding the boundary time at the end
        e_idx = s_idx + ((self.t_window_len - delta_win_start - self.t_eps) // self.frequency)

        assert s_idx > 0 and s_idx < len(self.ds), "Invalid start index for dataset."
        assert e_idx > 0 and e_idx < len(self.ds), "Invalid end index for dataset."
        assert np.logical_and(
            self.dates[s_idx : e_idx + 1 : self.sub_sampling_per_window] >= win_start,
            self.dates[s_idx : e_idx + 1 : self.sub_sampling_per_window] < win_end,
        ).all(), "Incorrect indices for window."

        return np.arange(s_idx, e_idx + 1, self.sub_sampling_per_window)
