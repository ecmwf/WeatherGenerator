# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import zarr
import json

class IconDataset:
    """
    A data reader for ICON model output stored in zarr.

    Parameters
    ----------
    start : datetime | int
        Start time of the data period as datetime object or integer in "%Y%m%d%H%M" format
    end : datetime | int
        End time of the data period (inclusive) with same format as start
    len_hrs : int
        Length of temporal windows in days
    step_hrs : int
        (Currently unused) Intended step size between windows in hours
    filename : Path
        Path to Zarr dataset containing ICON output
    stream_info : dict[str, list[str]]
        Dictionary with "source" and "target" keys specifying channel subsets to use
        (e.g., {"source": ["temp_00"], "target": ["TRCH4_chemtr_00"]})

    Attributes
    ----------
    len_hrs : int
        Temporal window length in days
    mesh_size : int
        Number of nodes in the ICON mesh
    source_channels : list[str]
        Patterns of selected source channels
    target_channels : list[str]
        Patterns of selected target channels
    mean : np.ndarray
        Per-channel means for normalization (includes coordinates)
    stdev : np.ndarray
        Per-channel standard deviations for normalization (includes coordinates)
    properties : dict[str, list[str]]
        Dataset metadata including 'stream_id' from Zarr attributes

    """

    def __init__(
        self,
        start: datetime | int,
        end: datetime | int,
        len_hrs: int,
        step_hrs: int,
        filename: Path,
        stream_info: dict,
    ):
        self.len_hrs = len_hrs

        format_str = "%Y%m%d%H%M"
        if type(start) is not datetime:
            start = datetime.strptime(str(start), format_str)
        start = np.datetime64(start).astype("datetime64[D]")

        if type(end) is not datetime:
            end = datetime.strptime(str(end), format_str)
        end = np.datetime64(end).astype("datetime64[D]")

        # loading datafile
        self.filename = filename
        self.ds = zarr.open(filename, mode="r")
        self.mesh_size = self.ds.attrs["ncells"]

        # Loading stat file
        stats_filename = Path(filename).with_suffix('.json')
        with open(stats_filename, 'r') as stats_file:
            self.stats = json.load(stats_file)

        time_as_in_data_file = np.array(self.ds['time'], dtype='timedelta64[D]')+np.datetime64(self.ds['time'].attrs['units'].split('since ')[-1])

        start_ds = time_as_in_data_file[0]
        end_ds = time_as_in_data_file[-1]

        # asserting start and end times
        if start_ds > end or end_ds < start:
            # TODO: this should be set in the base class
            self.source_channels = []
            self.target_channels = []
            self.source_idx = np.array([])
            self.target_idx = np.array([])
            self.geoinfo_idx = []
            self.len = 0
            self.ds = None
            return

        self.start_idx = (start - start_ds).astype("timedelta64[D]").astype(int) * self.mesh_size
        self.end_idx = (
            (end - start_ds).astype("timedelta64[D]").astype(int) + 1
        ) * self.mesh_size - 1

        self.len = (self.end_idx - self.start_idx) // self.mesh_size

        assert self.end_idx > self.start_idx, (
            f"Abort: Final index of {self.end_idx} is the same of larger than start index {self.start_idx}"
        )

        len_data_entries = len(self.ds['time']) * self.mesh_size

        assert self.end_idx + len_hrs <= len_data_entries, (
            f"Abort: end_date must be set at least {len_hrs} before the last date in the dataset"
        )

        # variables 
        self.colnames = list(self.ds)
        self.cols_idx = np.array(list(np.arange(len(self.colnames))))
    
        # Ignore step_hrs, idk how it supposed to work
        # TODO, TODO, TODO:
        self.step_hrs = 1

        # time
        repeated_times = np.repeat(time_as_in_data_file, self.mesh_size).reshape(-1, 1)
        self.time = repeated_times 

        # coordinates
        coords_units = self.ds['clat'].attrs['units']

        if coords_units == 'radian':
            lat_as_in_data_file = np.rad2deg(self.ds["clat"][:].astype("f"))
            lon_as_in_data_file = np.rad2deg(self.ds["clon"][:].astype("f"))

        else:
            lat_as_in_data_file = self.ds["clat"][:].astype("f")
            lon_as_in_data_file = self.ds["clon"][:].astype("f")

        self.lat = np.tile(lat_as_in_data_file, len(time_as_in_data_file))
        self.lon = np.tile(lon_as_in_data_file, len(time_as_in_data_file))

        self.properties = {
            "stream_id":  0 
        }

        # stats
        stats_vars = self.stats['metadata']['variables']
        assert stats_vars == self.colnames, (
            f"Variables in normalization file {stats_vars} do not match dataset columns {self.colnames}"
        )

        self.mean = np.array(self.stats["statistics"]["mean"], dtype='d') 
        self.stdev = np.array(self.stats["statistics"]["std"], dtype='d') 

        # Channel selection and indexing
        source_channels = stream_info["source"] if "source" in stream_info else None
        if source_channels:
            self.source_channels, self.source_idx = self.select(source_channels)
        else:
            self.source_channels = self.colnames
            self.source_idx = self.cols_idx

        target_channels = stream_info["target"] if "target" in stream_info else None
        if target_channels:
            self.target_channels, self.target_idx = self.select(target_channels)
        else:
            self.target_channels = self.colnames
            self.target_idx = self.cols_idx

        # Check if standard deviations are strictly positive for selected channels
        selected_channel_indices = list(set(self.source_idx).union(set(self.target_idx)))
        non_positive_stds = np.where(self.stdev[selected_channel_indices] <= 0)[0]
        assert len(non_positive_stds) == 0, (
            f"Abort: Encountered non-positive standard deviations for selected columns {[self.colnames[selected_channel_indices][i] for i in non_positive_stds]}."
        )
        # TODO: define in base class
        self.geoinfo_idx = []


    def select(self, ch_filters: list[str]) -> tuple[list[str], np.array]:
        """
        Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.
        """

        mask = [np.array([f in c for f in ch_filters]).any() for c in self.colnames]

        selected_cols_idx = np.where(mask)[0]
        selected_colnames = [self.colnames[i] for i in selected_cols_idx]

        return selected_colnames, selected_cols_idx

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

    def _get(self, idx: int, channels: np.array) -> tuple:
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
        if self.ds is None:
            fp32 = np.float32
            return (
                np.array([], dtype=fp32),
                np.array([], dtype=fp32),
                np.array([], dtype=fp32),
                np.array([], dtype=fp32),
            )

        # indexing
        start_row = self.start_idx + idx * self.mesh_size
        end_row = start_row + self.len_hrs * self.mesh_size

        # data
        data_reshaped = [
            np.asarray(self.ds[ch_]).reshape(-1, 1)[start_row:end_row]
            for ch_ in channels
        ]
        data = np.concatenate(data_reshaped, axis=1) 

        lat = np.expand_dims(self.lat[start_row:end_row],1)
        lon = np.expand_dims(self.lon[start_row:end_row],1)

        latlon = np.concatenate([lat, lon], 1)

        # empty geoinfos
        geoinfos = np.zeros((data.shape[0], 0), dtype=data.dtype)
        datetimes = np.squeeze(self.time[start_row:end_row])

        return (latlon, geoinfos, data, datetimes)

    def get_source(self, idx: int) -> tuple[np.array, np.array, np.array, np.array]:
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
        return self._get(idx, self.source_channels)

    def get_target(self, idx: int) -> tuple[np.array, np.array, np.array, np.array]:
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
        return self._get(idx, self.target_channels)

    def get_source_size(self) -> int:
        """
        Get size of all columns, including coordinates and geoinfo, with source

        Parameters
        ----------
        None

        Returns
        -------
        size of coords
        """
        return 2 + len(self.geoinfo_idx) + len(self.source_idx) if self.ds else 0

    def get_target_size(self) -> int:
        """
        Get size of all columns, including coordinates and geoinfo, with source

        Parameters
        ----------
        None

        Returns
        -------
        size of coords
        """
        return 2 + len(self.geoinfo_idx) + len(self.target_idx) if self.ds else 0

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

    def normalize_coords(self, coords: torch.tensor) -> torch.tensor:
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

    def normalize_source_channels(self, source: torch.tensor) -> torch.tensor:
        """
        Normalize source channels

        Parameters
        ----------
        source :
            data to be normalized

        Returns
        -------
        Normalized data
        """
        assert source.shape[1] == len(self.source_idx)
        for i, ch in enumerate(self.source_idx):
            source[..., i] = (source[..., i] - self.mean[ch]) / self.stdev[ch]

        return source

    def normalize_target_channels(self, target: torch.tensor) -> torch.tensor:
        """
        Normalize target channels

        Parameters
        ----------
        target :
            data to be normalized

        Returns
        -------
        Normalized data
        """
        assert target.shape[1] == len(self.target_idx)
        for i, ch in enumerate(self.target_idx):
            target[..., i] = (target[..., i] - self.mean[ch]) / self.stdev[ch]

        return target

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
        start_row = self.start_idx + idx * self.mesh_size
        end_row = start_row + self.len_hrs * self.mesh_size
        
        return (self.time[start_row, 0], self.time[end_row, 0])

    def denormalize_target_channels(self, data: torch.tensor) -> torch.tensor:
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
        assert data.shape[-1] == len(self.target_idx), "incorrect number of channels"
        for i, ch in enumerate(self.target_idx):
            data[..., i] = (data[..., i] * self.stdev[ch]) + self.mean[ch]

        return data

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

    def normalize_geoinfos(self, geoinfos: torch.tensor) -> torch.tensor:
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

        assert geoinfos.shape[-1] == 0, "incorrect number of geoinfo channels"
        return geoinfos
