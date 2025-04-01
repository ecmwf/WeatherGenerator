# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime

import numpy as np
import zarr


class FesomDataset:
    def __init__(
        self,
        start: datetime | int,
        end: datetime | int,
        len_hrs: int,
        step_hrs: int,
        filename: str,
        stream_info: dict,
    ):
        self.len_hrs = len_hrs

        format_str = "%Y%m%d%H%M%S"
        if type(start) is int:
            start = datetime.strptime(str(start), format_str)
        start = np.datetime64(start).astype("datetime64[D]")

        if type(end) is int:
            end = datetime.strptime(str(end), format_str)
        end = np.datetime64(end).astype("datetime64[D]")

        self.filename = filename
        self.ds = zarr.open(filename, mode="r")
        self.mesh_size = self.ds.data.attrs["nod2"]

        self.time = self.ds["dates"]

        start_ds = self.time[0][0].astype("datetime64[D]")
        end_ds = self.time[-1][0].astype("datetime64[D]")

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

        self.colnames = list(self.ds.data.attrs["colnames"])
        self.cols_idx = list(np.arange(len(self.colnames)))
        self.lat_index = list(self.colnames).index("lat")
        self.lon_index = list(self.colnames).index("lon")
        self.colnames.remove("lat")
        self.colnames.remove("lon")
        self.cols_idx.remove(self.lat_index)
        self.cols_idx.remove(self.lon_index)
        self.cols_idx = np.array(self.cols_idx)

        # Ignore step_hrs, idk how it supposed to work
        # TODO, TODO, TODO:
        self.step_hrs = 1

        self.data = self.ds["data"]

        self.properties = {
            "stream_id": self.ds.data.attrs["obs_id"],
        }

        self.mean = np.concatenate((np.array([0, 0]), np.array(self.ds.data.attrs["means"])))
        self.stdev = np.sqrt(
            np.concatenate((np.array([0, 0]), np.array(self.ds.data.attrs["vars"])))
        )

        source_channels = stream_info["source"] if "source" in stream_info else None
        if source_channels:
            self.source_channels, self.source_idx = self.selec(source_channels)
        else:
            self.source_channels = self.colnames
            self.source_idx = self.cols_idx

        target_channels = stream_info["target"] if "target" in stream_info else None
        if target_channels:
            self.target_channels, self.target_idx = self.select(target_channels)
        else:
            self.target_channels = self.colnames
            self.target_idx = self.cols_idx

        # TODO: define in base class
        self.geoinfo_idx = []

    def select(self, ch_filters: list[str]) -> None:
        """
        Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.
        """

        mask = [np.array([f in c for f in ch_filters]).any() for c in self.colnames]

        selected_cols_idx = np.where(mask)[0]
        selected_colnames = [self.colnames[i] for i in selected_cols_idx]

        return selected_colnames, selected_cols_idx

    def __len__(self):
        return self.len

    def _get(self, idx: int, idx_channels: np.array) -> tuple:
        """ """
        if self.ds is None:
            fp32 = np.float32
            return (
                np.array([], dtype=fp32),
                np.array([], dtype=fp32),
                np.array([], dtype=fp32),
                np.array([], dtype=fp32),
            )

        start_row = self.start_idx + idx * self.mesh_size
        end_row = start_row + self.len_hrs * self.mesh_size
        data = self.data.oindex[start_row:end_row, idx_channels]

        lat = np.expand_dims(self.data.oindex[start_row:end_row, self.lat_index], 1)
        lon = np.expand_dims(self.data.oindex[start_row:end_row, self.lon_index], 1)

        latlon = np.concatenate([lat, lon], 1)
        # empty geoinfos
        geoinfos = np.zeros((data.shape[0], 0), dtype=data.dtype)
        datetimes = np.squeeze(self.time[start_row:end_row])

        return (latlon, geoinfos, data, datetimes)

    def get_source(self, idx: int) -> tuple:
        """ """

        return self._get(idx, self.source_idx)

    def get_target(self, idx: int) -> tuple:
        """ """

        return self._get(idx, self.target_idx)

    def get_source_size(self):
        """
        TODO
        """
        return 2 + len(self.geoinfo_idx) + len(self.source_idx) if self.ds else 0

    def get_source_num_channels(self):
        """
        TODO
        """
        return len(self.source_idx)

    def get_target_size(self):
        """
        TODO
        """
        return 2 + len(self.geoinfo_idx) + len(self.target_idx) if self.ds else 0

    def get_target_num_channels(self):
        """
        TODO
        """
        return len(self.target_idx)

    def get_geoinfo_size(self):
        """
        TODO
        """
        return len(self.geoinfo_idx)

    def normalize_coords(self, coords):
        """
        TODO
        """
        coords[..., 0] = np.sin(np.deg2rad(coords[..., 0]))
        coords[..., 1] = np.sin(0.5 * np.deg2rad(coords[..., 1]))

        return coords

    def normalize_geoinfos(self, geoinfos):
        """
        TODO
        """

        assert geoinfos.shape[-1] == 0
        return geoinfos

    def normalize_source_channels(self, source):
        """
        TODO
        """
        assert source.shape[1] == len(self.source_idx)
        for i, ch in enumerate(self.source_idx):
            source[..., i] = (source[..., i] - self.mean[ch]) / self.stdev[ch]

        return source

    def normalize_target_channels(self, target):
        """
        TODO
        """
        assert target.shape[1] == len(self.target_idx)
        for i, ch in enumerate(self.target_idx):
            target[..., i] = (target[..., i] - self.mean[ch]) / self.stdev[ch]

        return target

    def time_window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
        start_row = self.start_idx + idx * self.mesh_size
        end_row = start_row + self.len_hrs * self.mesh_size

        return (self.time[start_row, 0], self.time[end_row, 0])
