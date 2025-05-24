# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from pathlib import Path
import logging
from dataclasses import dataclass

import numpy as np

_logger = logging.getLogger(__name__)

@dataclass
class ReaderData : 
    coords : np.array #[np.float32]
    geoinfos: np.array #[np.float32]
    data: np.array #[np.float32]
    times: np.array #[np.datetime64]

    def __init__() :
        self.coords = np.array( [0,0], dtype=np.float32)
        self.geoinfos = np.array( [0,0], dtype=np.float32)
        self.data = np.array( [0,0], dtype=np.float32)
        self.times = np.array( [0], dtype=np.datetime64)

class DataReaderBase :
    "Base class for data readers"

    def __init__(
        self,
        start: int,
        end: int,
        len_hrs: int,
        step_hrs: int,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """
        Construct dataset based on anemoi dataset

        Parameters
        ----------
        start : int
            Start time
        end : int
            End time
        len_hrs : int
            length of data window
        step_hrs :
            delta hours between start times of windows
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        # an implementation needs to implemen/overwrite the following functionality
        # self.len
        # self._get
        # self.source_idx
        # self.mean
        # self.stdev

        self.len = 0
        self.source_idx = []
        self.target_idx = []
        self.geoinfo_idx = []
        self.source_channels = []
        self.target_channels = []

        self.mean = np.zeros( 0)
        self.stdev = np.ones( 0)
        self.mean_geoinfo = np.zeros( 0)
        self.stdev_geoinfo = np.ones( 0)

        self.len_hrs = len_hrs
        self.step_hrs = step_hrs

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

    def get_source(self, idx: int) -> ReaderData :
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
        return self._get(idx, self.source_idx)

    def get_target(self, idx: int) -> ReaderData :
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
        return self._get(idx, self.target_idx)

    def _get( self, idx: int, channels_idx: np.array ) -> ReaderData :
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
            source[..., i] = (source[..., i] - self.mean_geoinfo[ch]) / self.stdev_geoinfo[ch]
        
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

    def time_window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
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
        
        return (np.array([], dtype=np.datetime64), np.array([], dtype=np.datetime64))
