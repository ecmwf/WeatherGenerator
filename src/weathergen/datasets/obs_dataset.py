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

import numpy as np
import torch
import zarr


class ObsDataset:
    def __init__(
        self,
        start: int,
        end: int,
        len_hrs: int,
        step_hrs: int,
        filename: Path,
        stream_info: dict,
    ) -> None:
        self.filename = filename
        self.z = zarr.open(filename, mode="r")
        self.data = self.z["data"]
        self.dt = self.z["dates"]  # datetime only
        self.hrly_index = self.z["idx_197001010000_1"]
        self.colnames = self.data.attrs["colnames"]
        self.len_hrs = len_hrs
        self.step_hrs = step_hrs if step_hrs else len_hrs

        # self.selected_colnames = self.colnames
        # self.selected_cols_idx = np.arange(len(self.colnames))
        idx = 0
        for i, col in enumerate(reversed(self.colnames)):
            idx = i
            # if col[:9] == 'obsvalue_' :
            if not (col[:4] == "sin_" or col[:4] == "cos_"):
                break
        self.selected_colnames = self.colnames[: len(self.colnames) - idx]
        self.selected_cols_idx = np.arange(len(self.colnames))[: len(self.colnames) - idx]

        # Create index for samples
        self._setup_sample_index(start, end, self.len_hrs, self.step_hrs)
        # assert len(self.indices_start) == len(self.indices_end)

        self._load_properties()

        # TODO: re-implement selection of source and target channels

        self.source_idx = [i for i, col in enumerate(self.selected_colnames) if "obsvalue" in col]
        self.source_channels = [self.selected_colnames[i] for i in self.source_idx]

        self.target_idx = [i for i, col in enumerate(self.selected_colnames) if "obsvalue" in col]
        self.target_channels = [self.selected_colnames[i] for i in self.target_idx]

        for i, _ in enumerate(self.colnames):
            idx = i
            if self.colnames[i] == "lat" and self.colnames[i + 1] == "lon":
                break
        self.coords_idx = [i, i + 1]
        channels_idx = [i for i, col in enumerate(self.selected_colnames) if "obsvalue" in col]
        self.geoinfo_idx = list(range(i + 2, channels_idx[0]))

        self.mean = np.array(self.properties["means"])
        self.stdev = np.sqrt(np.array(self.properties["vars"]))

    def __getitem__(self, idx: int) -> tuple:
        start_row = self.indices_start[idx]
        end_row = self.indices_end[idx]

        data = self.data.oindex[start_row:end_row, self.selected_cols_idx]
        datetimes = self.dt[start_row:end_row][:, 0]

        return (data, datetimes)

    def __len__(self) -> int:
        return min(len(self.indices_start), len(self.indices_end))

    def select(self, cols_list: list[str]) -> None:
        """
        Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.
        """
        self.selected_colnames = cols_list
        self.selected_cols_idx = np.array([self.colnames.index(item) for item in cols_list])

    def time_window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
        """
        Returns a tuple of datetime objects describing the start and end times of the sample at position idx.
        """

        if idx < 0:
            idx = len(self) + idx

        time_start = self.start_dt + datetime.timedelta(hours=(int(idx * self.step_hrs)), seconds=1)
        time_end = min(
            self.start_dt + datetime.timedelta(hours=(int(idx * self.step_hrs + self.len_hrs))),
            self.end_dt,
        )

        return (np.datetime64(time_start), np.datetime64(time_end))

    def first_sample_with_data(self) -> int:
        """
        Returns the position of the first sample which contains data.
        """
        return (
            int(np.nonzero(self.indices_end)[0][0])
            if self.indices_end[-1] != self.indices_end[0]
            else None
        )

    def last_sample_with_data(self) -> int:
        """
        Returns the position of the last sample which contains data.
        """
        if self.indices_end[-1] == self.indices_end[0]:
            last_sample = None
        else:
            last_sample = int(
                np.where(np.diff(np.append(self.indices_end, self.indices_end[-1])) > 0)[0][-1] + 1
            )

        return last_sample

    def _setup_sample_index(self, start: int, end: int, len_hrs: int, step_hrs: int) -> None:
        """
        Dataset is divided into samples;
           - each n_hours long
           - sample 0 starts at start (yyyymmddhhmm)
           - index array has one entry for each sample; contains the index of the first row
           containing data for that sample
        """

        base_yyyymmddhhmm = 197001010000

        assert start > base_yyyymmddhhmm, (
            f"Abort: ObsDataset sample start (yyyymmddhhmm) must be greater than {base_yyyymmddhhmm}\n"
            f"       Current value: {start}"
        )

        # Derive new index based on hourly backbone index
        format_str = "%Y%m%d%H%M%S"
        base_dt = datetime.datetime.strptime(str(base_yyyymmddhhmm), format_str)
        self.start_dt = datetime.datetime.strptime(str(start), format_str)
        self.end_dt = datetime.datetime.strptime(str(end), format_str)

        # Calculate the number of hours between start of hourly base index and the requested sample index
        diff_in_hours_start = int((self.start_dt - base_dt).total_seconds() / 3600)
        diff_in_hours_end = int((self.end_dt - base_dt).total_seconds() / 3600)

        end_range_1 = min(diff_in_hours_end, self.hrly_index.shape[0] - 1)
        self.indices_start = self.hrly_index[diff_in_hours_start:end_range_1:step_hrs]

        end_range_2 = min(
            diff_in_hours_end + len_hrs, self.hrly_index.shape[0] - 1
        )  # handle beyond end of data range safely
        self.indices_end = (
            self.hrly_index[diff_in_hours_start + len_hrs : end_range_2 : step_hrs] - 1
        )
        # Handle situations where the requested dataset span goes beyond the hourly index stored in the zarr
        if diff_in_hours_end > (self.hrly_index.shape[0] - 1):
            if diff_in_hours_start > (self.hrly_index.shape[0] - 1):
                n = (diff_in_hours_end - diff_in_hours_start) // step_hrs
                self.indices_start = np.zeros(n, dtype=int)
                self.indices_end = np.zeros(n, dtype=int)
            else:
                self.indices_start = np.append(
                    self.indices_start,
                    np.ones(
                        (diff_in_hours_end - self.hrly_index.shape[0] - 1) // step_hrs, dtype=int
                    )
                    * self.indices_start[-1],
                )
                self.indices_end = np.append(
                    self.indices_end,
                    np.ones(
                        (diff_in_hours_end - self.hrly_index.shape[0] - 1) // step_hrs, dtype=int
                    )
                    * self.indices_end[-1],
                )

        # Prevent -1 in samples before the we have data
        self.indices_end = np.maximum(self.indices_end, 0)

        if self.indices_end.shape != self.indices_start.shape:
            self.indices_end = np.append(self.indices_end, self.indices_end[-1])

        # If end (yyyymmddhhmm) is not a multiple of len_hrs
        # truncate the last sample so that it doesn't go beyond the requested dataset end date
        self.indices_end = np.minimum(self.indices_end, self.hrly_index[end_range_1])

    def _load_properties(self) -> None:
        self.properties = {}

        self.properties["means"] = self.data.attrs["means"]
        self.properties["vars"] = self.data.attrs["vars"]
        # self.properties["data_idxs"] = self.data.attrs["data_idxs"]
        self.properties["obs_id"] = self.data.attrs["obs_id"]

    def _get(
        self, idx: int, channels_idx: np.array
    ) -> tuple[np.array, np.array, np.array, np.array]:
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

        start_row = self.indices_start[idx]
        end_row = self.indices_end[idx]

        latlon = self.data.oindex[start_row:end_row, self.coords_idx]
        geoinfos = (
            self.data.oindex[start_row:end_row, self.geoinfo_idx]
            if len(self.geoinfo_idx) > 0
            else np.zeros((latlon.shape[0], 0), np.float32)
        )
        data = self.data.oindex[start_row:end_row, channels_idx]
        datetimes = self.dt[start_row:end_row][:, 0]

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
        return self._get(idx, self.source_idx)

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
        return self._get(idx, self.target_idx)

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

    def normalize_geoinfos(self, geoinfos: torch.tensor) -> torch.tensor:
        """
        Normalize geoinfos

        Parameters
        ----------
        geoinfos :
            geoinfos to be normalized

        Returns
        -------
        Normalized geoinfos
        """

        assert geoinfos.shape[-1] == len(self.geoinfo_idx), "incorrect number of channels"
        for i, ch in enumerate(self.geoinfo_idx):
            geoinfos[..., i] = (geoinfos[..., i] - self.mean[ch]) / self.stdev[ch]

        return geoinfos

    def normalize_source_channels(self, source: torch.tensor) -> torch.tensor:
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
        assert source.shape[-1] == len(self.source_idx), "incorrect number of channels"
        for i, ch in enumerate(self.source_idx):
            source[..., i] = (source[..., i] - self.mean[ch]) / self.stdev[ch]

        return source

    def normalize_target_channels(self, target: torch.tensor) -> torch.tensor:
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
        assert target.shape[-1] == len(self.target_idx), "incorrect number of channels"
        for i, ch in enumerate(self.target_idx):
            target[..., i] = (target[..., i] - self.mean[ch]) / self.stdev[ch]

        return target

    def denormalize_source_channels(self, source: torch.tensor) -> torch.tensor:
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
        assert source.shape[-1] == len(self.source_idx), "incorrect number of channels"
        for i, ch in enumerate(self.source_idx):
            source[..., i] = (source[..., i] * self.stdev[ch]) + self.mean[ch]

        return source

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
