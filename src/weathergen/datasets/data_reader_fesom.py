# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import glob
import logging
from pathlib import Path
from typing import override

import dask
import dask.array as da
import numpy as np
import zarr

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    DTRange,
    NDArray,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    t_epsilon,
)

_logger = logging.getLogger(__name__)


class DataReaderFesom(DataReaderTimestep):
    """
    A dataset class for handling temporal windows of FESOM model output data stored in Zarr format.

    This class is optimized for use with multiple dataloader workers by implementing
    lazy initialization of file handles and efficient, batched data reads.
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        # Store configuration but DO NOT open files here
        self.filenames = sorted(glob.glob(str(filename) + "/*"))
        self._tw_handler = tw_handler
        self._stream_info = stream_info
        self.target_files = self.filenames

        self._src_lat_conv = False
        self._src_lon_conv = False
        self._trg_lat_conv = False
        self._trg_lon_conv = False

        if "target_file" in stream_info:
            self.target_files = sorted(glob.glob(str(stream_info["target_file"]) + "/*"))

        if len(self.filenames) == 0:
            self.init_empty()
            self._initialized = True
            return

        # Initialize data-dependent attributes to None. They will be set by _lazy_init.
        self.source_time: da.Array | None = None
        self.source_data: da.Array | None = None
        self.target_time: da.Array | None = None
        self.target_data: da.Array | None = None
        self.len = 0  # Default length is 0 until initialized
        self.source_channels = []
        self.source_idx = []
        self.target_channels = []
        self.target_idx = []
        self.geoinfo_channels = []
        self.geoinfo_idx = []
        self.properties = {}
        self.fake_specs = {}
        self.fake_target = False

        if len(self.filenames) == 0 or len(self.target_files) == 0:
            name = stream_info["name"]
            _logger.warning(
                f"{name} couldn't find any files matching {filename}. Stream is skipped."
            )
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            # No need to return, the length is 0, so it will be skipped.

        # We call super() last, after we know if the stream is valid or not.
        # We also pass dummy values, as the real ones will be set in _lazy_init.
        super().__init__(self._tw_handler, self._stream_info)

        # This flag ensures initialization happens only once per worker
        self._initialized = False
        # print(f"checking stream info {list(stream_info.keys())}")

    def _get_mesh_size(self, group: zarr.Group) -> int:
        if "n_points" in group.data.attrs:
            return group.data.attrs["n_points"]
        else:
            return group.data.attrs["nod2"]

    def _reorder_groups(self, colnames: list[str], groups: list[zarr.Group]) -> list[da.Array]:
        reordered_data_arrays: list[da.Array] = []

        for group in groups:
            local_colnames = group["data"].attrs["colnames"]

            # If the order is already correct, no need to do anything.
            if local_colnames == colnames:
                reordered_data_arrays.append(da.from_zarr(group["data"]))
            else:
                # Create the list of indices to re-shuffle the columns.
                reorder_indices = [local_colnames.index(name) for name in colnames]

                # Lazily re-index the dask array. This operation is not executed immediately.
                dask_array = da.from_zarr(group["data"])
                reordered_array = dask_array[:, reorder_indices]
                reordered_data_arrays.append(reordered_array)

        return reordered_data_arrays

    def _remove_lonlat(self, colnames: list[str]) -> list[str]:
        temp_colnames = list(colnames)
        temp_colnames.remove("lat")
        temp_colnames.remove("lon")
        return temp_colnames

    def _lazy_init(self) -> None:
        """
        Initializes the dataset object. This method is called once per worker process
        to ensure dask scheduler is not shared between them.
        """
        # pylint: disable=attribute-defined-outside-init

        if self._initialized:
            return

        _logger.info(f"Initialising {self._stream_info['name']}")

        # Each worker now opens its own file handles safely
        s_groups: list[zarr.Group] = [zarr.open_group(name, mode="r") for name in self.filenames]
        t_groups: list[zarr.Group] = [zarr.open_group(name, mode="r") for name in self.target_files]

        s_times: list[zarr.Array] = [group["dates"] for group in s_groups]
        t_times: list[zarr.Array] = [group["dates"] for group in t_groups]

        self.source_time = da.concatenate(s_times, axis=0)
        self.target_time = da.concatenate(t_times, axis=0)

        # Use the first group for metadata
        self.source_mesh_size = self._get_mesh_size(s_groups[0])
        self.target_mesh_size = self._get_mesh_size(t_groups[0])

        # Metadata reading is cheap, but let's do it with the rest of the init
        self.start_source = self.source_time[0][0].compute()
        self.end_source = self.source_time[-1][0].compute()

        if self.start_source > self._tw_handler.t_end or self.end_source < self._tw_handler.t_start:
            name = self._stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            self.init_empty()
            self._initialized = True
            return

        self.start_target = self.target_time[0][0].compute()
        self.end_target = self.target_time[-1][0].compute()

        if self.start_target > self._tw_handler.t_end or self.end_target < self._tw_handler.t_start:
            name = self._stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            self.init_empty()
            self._initialized = True
            return

        self.source_period = (
            self.source_time[self.source_mesh_size][0] - self.source_time[0][0]
        ).compute()
        self.target_period = (
            self.target_time[self.target_mesh_size][0] - self.target_time[0][0]
        ).compute()

        # Re-initialize the parent class with correct time info
        super().__init__(  # Initialise only for source as source-target split is not supported
            self._tw_handler,
            self._stream_info,
            self.start_source,
            self.end_source,
            self.source_period,
        )

        if (
            self._tw_handler.t_start > self.start_source
            and self._tw_handler.t_start > self.end_source
        ):
            self.source_start_idx = (
                (self._tw_handler.t_start - self.start_source) // self.source_period + 1
            ) * self.source_mesh_size
        else:
            self.source_start_idx = 0

        if (
            self._tw_handler.t_start > self.start_target
            and self._tw_handler.t_start > self.end_target
        ):
            self.target_start_idx = (
                (self._tw_handler.t_start - self.start_target) // self.target_period + 1
            ) * self.target_mesh_size
        else:
            self.target_start_idx = 0

        self.source_end_idx = (
            (self._tw_handler.t_end - self.start_source) // self.source_period + 1
        ) * self.source_mesh_size
        self.target_end_idx = (
            (self._tw_handler.t_end - self.start_target) // self.target_period + 1
        ) * self.target_mesh_size

        if self.source_end_idx > len(self.source_time):
            self.source_end_idx = len(self.source_time)
        if self.target_end_idx > len(self.target_time):
            self.target_end_idx = len(self.target_time)

        self.source_len = (self.source_end_idx - self.source_start_idx) // self.source_mesh_size
        self.target_len = (self.target_end_idx - self.target_start_idx) // self.target_mesh_size
        self.len = min(self.source_len, self.target_len)

        # Check for a valid length after calculations
        if self.len <= 0:
            self.init_empty()
            self._initialized = True
            return

        self.source_colnames: list[str] = list(s_groups[0].data.attrs["colnames"])
        self.target_colnames: list[str] = list(t_groups[0].data.attrs["colnames"])

        self.source_cols_idx = list(np.arange(len(self.source_colnames), dtype=int))
        self.target_cols_idx = list(np.arange(len(self.target_colnames), dtype=int))

        self.src_lat_index: int = self.source_colnames.index("lat")
        self.src_lon_index: int = self.source_colnames.index("lon")
        self.trg_lat_index: int = self.target_colnames.index("lat")
        self.trg_lon_index: int = self.target_colnames.index("lon")

        source_reorderd = self._reorder_groups(self.source_colnames, s_groups)
        target_reorderd = self._reorder_groups(self.target_colnames, t_groups)

        # Modify a copy, not the original list while iterating
        self.source_colnames = self._remove_lonlat(self.source_colnames)
        self.target_colnames = self._remove_lonlat(self.target_colnames)

        self.source_cols_idx.remove(self.src_lat_index)
        self.source_cols_idx.remove(self.src_lon_index)
        self.source_cols_idx = np.array(self.source_cols_idx)

        self.target_cols_idx.remove(self.trg_lat_index)
        self.target_cols_idx.remove(self.trg_lon_index)
        self.target_cols_idx = np.array(self.target_cols_idx)

        self.properties = {"stream_id": s_groups[0].data.attrs["obs_id"]}

        self.source_mean = np.concatenate(
            (np.array([0, 0]), np.array(s_groups[0].data.attrs["means"]))
        )
        self.source_stdev = np.sqrt(
            np.concatenate((np.array([1, 1]), np.array(s_groups[0].data.attrs["std"])))
        )
        self.source_stdev[self.source_stdev <= 1e-5] = 1.0

        self.target_mean = np.concatenate(
            (np.array([0, 0]), np.array(t_groups[0].data.attrs["means"]))
        )
        self.target_stdev = np.sqrt(
            np.concatenate((np.array([1, 1]), np.array(t_groups[0].data.attrs["std"])))
        )
        self.target_stdev[self.target_stdev <= 1e-5] = 1.0

        self.source = da.concatenate(source_reorderd, axis=0)
        self.target = da.concatenate(target_reorderd, axis=0)

        source_channels = self._stream_info.get("source")
        source_excl = self._stream_info.get("source_exclude")
        self.source_channels, self.source_idx = (
            self.select(self.source_colnames, self.source_cols_idx, source_channels, source_excl)
            if source_channels or source_excl
            else (self.source_colnames, self.source_cols_idx)
        )

        target_channels = self._stream_info.get("target")
        target_excl = self._stream_info.get("target_exclude")
        self.target_channels, self.target_idx = (
            self.select(self.target_colnames, self.target_cols_idx, target_channels, target_excl)
            if target_channels or target_excl
            else (self.target_colnames, self.target_cols_idx)
        )

        src_timestep_lats = self.source[: self.source_mesh_size, self.src_lat_index].compute()
        trg_timestep_lats = self.target[: self.target_mesh_size, self.trg_lat_index].compute()

        if np.any(src_timestep_lats > 90.0):
            _logger.warning(
                f"Latitude for stream '{self._stream_info['name']}' "
                f"source appears to be in a [0, 180] format. "
                f"It will be automatically converted to the required [-90, 90] format."
            )
            self._src_lat_conv = True

        if np.any(trg_timestep_lats > 90.0):
            _logger.warning(
                f"Latitude for stream '{self._stream_info['name']}' "
                f"target appears to be in a [0, 180] format. "
                f"It will be automatically converted to the required [-90, 90] format."
            )
            self._trg_lat_conv = True

        src_timestep_lons = self.source[: self.source_mesh_size, self.src_lon_index].compute()
        trg_timestep_lons = self.target[: self.target_mesh_size, self.trg_lon_index].compute()

        if np.any(src_timestep_lons > 180.0):
            _logger.warning(
                f"Longitude for stream '{self._stream_info['name']}' "
                f"source appears to be in a [0, 360] format. "
                f"It will be automatically converted to the required [-180, 180] format."
            )
            self._src_lon_conv = True

        if np.any(trg_timestep_lons > 180.0):
            _logger.warning(
                f"Longitude for stream '{self._stream_info['name']}' "
                f"target appears to be in a [0, 360] format."
                f"It will be automatically converted to the required [-180, 180] format."
            )
            self._trg_lon_conv = True

        self.geoinfo_channels = []
        self.geoinfo_idx = []

        self._initialized = True

    def select(
        self,
        colnames: list[str],
        cols_idx: NDArray,
        ch_filters: list[str] | None,
        excl: list[str] | None = None,
    ) -> tuple[list[str], NDArray]:
        if excl and ch_filters:
            mask = [
                any(f == c for f in ch_filters) and all(ex not in c for ex in excl)
                for c in colnames
            ]
        elif ch_filters:
            mask = [any(f == c for f in ch_filters) for c in colnames]
        elif excl:
            mask = [all(ex not in c for ex in excl) for c in colnames]
        else:
            assert False, "Cannot use select with both ch_filters and excl as None"

        selected_cols_idx = cols_idx[np.where(mask)[0]]
        selected_colnames = [colnames[i] for i in np.where(mask)[0]]
        return selected_colnames, selected_cols_idx

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0

    @override
    def length(self) -> int:
        # Make sure initialization has happened before returning length
        self._lazy_init()
        return self.len

    def _get_source_idxs(self, idx: TIndex) -> tuple[NDArray, DTRange]:
        """
        Get source dataset indexes for a given time window index, when the dataset is periodic.

        This function assumes state of a variable is persistent, thus if no data is found
        in the time window, last measurement is used before the beggining of the windows is used.

        Parameters
        ----------
        idx : TIndex
            Index of the time window.

        Returns
        -------
        NDArray[np.int64]
            Array of dataset indexes corresponding to the time window.
        """
        tw_handler = self.time_window_handler

        # Function is separated from the class to allow testing without instantiating the class.
        dtr = tw_handler.window(idx)
        # If there is no or only marginal overlap with the dataset, return empty index ranges
        if (
            not self.start_source
            or not self.end_source
            or dtr.end < self.start_source
            or dtr.start > self.end_source
            or dtr.start < self.start_source
            or dtr.end > self.end_source
            or (self.end_source is not None and dtr.start > self.end_source)
        ):
            return (np.array([], dtype=np.int64), dtr)

        # relative time in dataset
        delta_t_start = dtr.start - self.start_source
        delta_t_end = dtr.end - self.start_source - t_epsilon
        assert isinstance(delta_t_start, np.timedelta64), "delta_t_start must be timedelta64"
        start_didx = delta_t_start // self.source_period
        end_didx = delta_t_end // self.source_period

        # adjust start_idx if not exactly on start time
        if (delta_t_start % self.source_period) > np.timedelta64(0, "s"):
            # empty window in between two timesteps
            if start_didx == end_didx:
                return (np.array([start_didx], dtype=np.int64), dtr)
            start_didx += 1

        end_didx = start_didx + int((dtr.end - dtr.start - t_epsilon) / self.source_period)
        return (np.arange(start_didx, end_didx + 1, dtype=np.int64), dtr)

    def _get_target_idxs(self, idx: TIndex) -> tuple[NDArray, DTRange]:
        """
        Get target dataset indexes for a given time window index, when the dataset is periodic.

        This function assumes state of a variable is persistent, thus if no data is found
        in the time window, last measurement is used before the beggining of the windows is used.

        Parameters
        ----------
        idx : TIndex
            Index of the time window.

        Returns
        -------
        NDArray[np.int64]
            Array of dataset indexes corresponding to the time window.
        """
        tw_handler = self.time_window_handler

        # Function is separated from the class to allow testing without instantiating the class.
        dtr = tw_handler.window(idx)
        # If there is no or only marginal overlap with the dataset, return empty index ranges
        if (
            not self.start_target
            or not self.end_target
            or dtr.end < self.start_target
            or dtr.start > self.end_target
            or dtr.start < self.start_target
            or dtr.end > self.end_target
            or (self.end_target is not None and dtr.start > self.end_target)
        ):
            return (np.array([], dtype=np.int64), dtr)

        # relative time in dataset
        delta_t_start = dtr.start - self.start_target
        delta_t_end = dtr.end - self.start_target - t_epsilon
        assert isinstance(delta_t_start, np.timedelta64), "delta_t_start must be timedelta64"
        start_didx = delta_t_start // self.target_period
        end_didx = delta_t_end // self.target_period

        # adjust start_idx if not exactly on start time
        if (delta_t_start % self.target_period) > np.timedelta64(0, "s"):
            # empty window in between two timesteps
            if start_didx == end_didx:
                return (np.array([start_didx], dtype=np.int64), dtr)
            start_didx += 1

        end_didx = start_didx + int((dtr.end - dtr.start - t_epsilon) / self.target_period)
        return (np.arange(start_didx, end_didx + 1, dtype=np.int64), dtr)

    @override
    def get_source(self, idx: TIndex) -> ReaderData:
        self._lazy_init()
        (t_idxs, dtr) = self._get_source_idxs(idx)
        if self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(self.source_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        start_row = t_idxs[0] * self.source_mesh_size
        end_row = (t_idxs[-1] + 1) * self.source_mesh_size

        # Note: we read all columns from start_row to end_row once,
        # then select the ones we need. This is more efficient for Zarr.
        full_data_slice = self.source[start_row:end_row]
        datetimes_lazy = self.source_time[start_row:end_row]

        # Define the specific slices we need from the larger block
        data_lazy = full_data_slice[:, self.source_idx]
        lat_lazy = full_data_slice[:, self.src_lat_index]
        lon_lazy = full_data_slice[:, self.src_lon_index]

        # Dask optimizes this to a single (or few) efficient read operation(s).
        data, lat, lon, datetimes = dask.compute(
            data_lazy, lat_lazy, lon_lazy, datetimes_lazy, scheduler="single-threaded"
        )

        if self._src_lat_conv:
            lat = 90.0 - lat

        if self._src_lon_conv:
            lon = ((lon + 180.0) % 360.0) - 180.0

        coords = np.stack([lat, lon], axis=1)
        geoinfos = np.zeros((data.shape[0], 0), dtype=data.dtype)
        datetimes = np.squeeze(datetimes)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )

        return rd

    @override
    def get_target(self, idx: TIndex) -> ReaderData:
        self._lazy_init()
        (t_idxs, dtr) = self._get_target_idxs(idx)
        if self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(self.source_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        start_row = t_idxs[0] * self.target_mesh_size
        end_row = (t_idxs[-1] + 1) * self.target_mesh_size

        # Note: we read all columns from start_row to end_row once,
        # then select the ones we need. This is more efficient for Zarr.
        full_data_slice = self.target[start_row:end_row]
        datetimes_lazy = self.target_time[start_row:end_row]

        # Define the specific slices we need from the larger block
        data_lazy = full_data_slice[:, self.target_idx]
        lat_lazy = full_data_slice[:, self.trg_lat_index]
        lon_lazy = full_data_slice[:, self.trg_lon_index]

        # Dask optimizes this to a single (or few) efficient read operation(s).
        data, lat, lon, datetimes = dask.compute(
            data_lazy, lat_lazy, lon_lazy, datetimes_lazy, scheduler="single-threaded"
        )

        if self._trg_lat_conv:
            lat = 90.0 - lat

        if self._trg_lon_conv:
            lon = ((lon + 180.0) % 360.0) - 180.0

        coords = np.stack([lat, lon], axis=1)
        geoinfos = np.zeros((data.shape[0], 0), dtype=data.dtype)
        datetimes = np.squeeze(datetimes)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )

        return rd

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        return self.get_source(idx)

    @override
    def normalize_source_channels(self, source: NDArray) -> NDArray:
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
            source[..., i] = (source[..., i] - self.source_mean[ch]) / self.source_stdev[ch]

        return source

    @override
    def normalize_target_channels(self, target: NDArray) -> NDArray:
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
            target[..., i] = (target[..., i] - self.target_mean[ch]) / self.target_stdev[ch]

        return target

    @override
    def denormalize_source_channels(self, source: NDArray) -> NDArray:
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
            source[..., i] = (source[..., i] * self.source_stdev[ch]) + self.source_mean[ch]

        return source

    @override
    def denormalize_target_channels(self, data: NDArray) -> NDArray:
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
            data[..., i] = (data[..., i] * self.target_stdev[ch]) + self.target_mean[ch]

        return data
