# (C) Copyright 2024 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# for interactive debugging
import logging
import os
from pathlib import Path
from typing import override

import numpy as np
import xarray as xr
import zarr
from numpy.typing import NDArray

os.environ["ZARR_V3_EXPERIMENTAL_API"] = "1"  # doesn't seem to work

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)


class DataReaderSeviri(DataReaderTimestep):
    """Data reader for SEVIRI satellite data."""

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """Initialize the SEVIRI data reader."""

        self.fillvalue = np.nan
        np32 = np.float32

        # set sampling parameters
        self.stride_temporal = stream_info["temporal_stride"]  # downsample to six hourly timesteps
        self.stride_spatial = stream_info[
            "spatial_stride"
        ]  # use every 8th point to reduce memory usage on workers

        index_path = Path(stream_info["metadata"]) / stream_info["scene"]
        self.spatial_indices = np.load(index_path)["seviri_indices"]

        self._zarr_path = filename
        self._ds = None  # opened lazily

        # Open temporarily with xarray just for init metadata (time handling is easier)
        ds_xr = xr.open_zarr(filename, group="seviri")
        ds_xr["time"] = ds_xr["time"].astype("datetime64[ns]")
        ds_xr = ds_xr.sel(time=slice(stream_info["data_start_time"], stream_info["data_end_time"]))

        col_extent = ds_xr["longitude"].shape[0]
        lat_idx = self.spatial_indices // col_extent
        lon_idx = self.spatial_indices % col_extent

        # Cache spatial indices for zarr access
        self._lat_idx = np.array(lat_idx[:: self.stride_spatial])
        self._lon_idx = np.array(lon_idx[:: self.stride_spatial])

        # code.interact(local=locals())

        # Apply spatial subset
        ds_xr = ds_xr.isel(latitude=self._lat_idx, longitude=self._lon_idx)

        # Cache time values as numpy (avoid zarr access for time later)
        self._time_values = np.array(ds_xr.time.values)

        # Find time indices in the full zarr that correspond to our time selection
        ds_full = xr.open_zarr(filename, group="seviri")
        ds_full["time"] = ds_full["time"].astype("datetime64[ns]")
        full_times = ds_full.time.values
        start_time = ds_xr.time.min().values
        self._time_offset = int(np.searchsorted(full_times, start_time))

        # caches lats and lons
        lat_name = stream_info.get("latitude_name", "latitude")
        self.latitudes = _clip_lat(np.array(ds_xr[lat_name], dtype=np32))
        lon_name = stream_info.get("longitude_name", "longitude")
        self.longitudes = _clip_lon(np.array(ds_xr[lon_name], dtype=np32))

        # check if the data overlaps with the time window, otherwise initialises as empty datareader
        if tw_handler.t_start >= ds_xr.time.max() or tw_handler.t_end <= ds_xr.time.min():
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        if "frequency" in stream_info:
            assert False, "Frequency sub-sampling currently not supported"

        period = np.timedelta64(self.stride_temporal, "h")

        data_start_time = ds_xr.time[0].values
        data_end_time = ds_xr.time[-1].values

        assert data_start_time is not None and data_end_time is not None, (
            data_start_time,
            data_end_time,
        )

        # sets the time window handler and stream info in the base class
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time,
            data_end_time,
            period,
        )

        # If there is no overlap with the time range, no need to keep the dataset.
        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            self.init_empty()
            return
        else:
            self.len = len(ds_xr["time"]) // self.stride_temporal

        self.exclude = {"LWMASK", "LANDCOV", "_indices", "quality_flag"}
        self.channels_file = [k for k in ds_xr.keys()]

        self.geoinfo_channels = stream_info.get("geoinfos", [])
        self.geoinfo_idx = [self.channels_file.index(ch) for ch in self.geoinfo_channels]

        # cache geoinfos
        if len(self.geoinfo_channels) != 0:
            self.geoinfo_data = np.stack(
                [np.array(ds_xr[ch], dtype=np32) for ch in self.geoinfo_channels]
            )
            self._geoinfo_flat = self.geoinfo_data.transpose([1, 2, 0]).reshape(
                (-1, len(self.geoinfo_channels))
            )

        # select/filter requested target channels
        self.target_idx, self.target_channels = self.select_channels(ds_xr, "target")

        self.source_channels = stream_info.get("source", [])
        self.source_idx = [self.channels_file.index(ch) for ch in self.source_channels]

        ds_name = stream_info["name"]
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")

        self.properties = {
            "stream_id": 0,
        }

        self.mean, self.stdev = self._create_statistics()
        self.mean_geoinfo, self.stdev_geoinfo = (
            self.mean[self.geoinfo_idx],
            self.stdev[self.geoinfo_idx],
        )

        # Close xarray, force lazy zarr open in workers
        ds_xr.close()
        ds_full.close()
        self._ds = None

    def _open_ds(self):
        store = zarr.open(self._zarr_path, mode="r")
        return store["seviri"]

    @property
    def ds(self):
        if self._ds is None:
            self._ds = self._open_ds()
        return self._ds

    @ds.setter
    def ds(self, value):
        self._ds = value

    def _create_statistics(self):
        statistics = Path(self.stream_info["metadata"]) / "statistics_global.npz"
        df_stats = _assemble_statistics_from_npz(statistics)

        mean, stdev = [], []

        for ch in self.channels_file:
            if ch in self.exclude:
                mean.append(0.0)
                stdev.append(1.0)
            else:
                mean.append(df_stats[ch]["mean"])
                stdev.append(df_stats[ch]["std"])

        mean = np.array(mean)
        stdev = np.array(stdev)

        return mean, stdev

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self._ds = None
        self.len = 0

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for window (for either source or target, through public interface)
        """
        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self._ds is None and self.len == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        if len(t_idxs) == 0 or len(channels_idx) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"

        # Convert to actual zarr indices (accounting for time offset and stride)
        didx_start = self._time_offset + t_idxs[0] * self.stride_temporal
        didx_end = self._time_offset + t_idxs[-1] * self.stride_temporal + 1

        sel_channels = [self.channels_file[i] for i in channels_idx]

        # Access zarr directly with numpy advanced indexing
        data_list = []
        for ch in sel_channels:
            # zarr array: shape is (time, lat, lon)
            ch_data = self.ds[ch][didx_start : didx_end : self.stride_temporal, self._lat_idx, :][
                :, :, self._lon_idx
            ]
            data_list.append(ch_data)

        data = np.stack(data_list, axis=-1)  # shape: (n_times, n_lats, n_lons, n_channels)

        n_times = data.shape[0]
        n_lats = data.shape[1]
        n_lons = data.shape[2]
        n_spatial = n_lats * n_lons

        # flatten along time dimension
        data = data.reshape((n_times * n_spatial, len(channels_idx)))

        # prepare geoinfos
        if len(self.geoinfo_channels) != 0:
            geoinfos = np.tile(self._geoinfo_flat, (n_times, 1))
        else:
            geoinfos = np.zeros((n_spatial * n_times, 0), dtype=np.float32)

        # construct lat/lon coords
        lat2d, lon2d = np.meshgrid(
            self.latitudes,
            self.longitudes,
            indexing="ij",
        )
        lat_flat = lat2d.reshape(-1)
        lon_flat = lon2d.reshape(-1)

        # Tile spatial coordinates for each timestep
        coords = np.tile(np.column_stack((lat_flat, lon_flat)), (n_times, 1))

        # Use cached time values
        time_indices = slice(
            t_idxs[0] * self.stride_temporal,
            t_idxs[-1] * self.stride_temporal + 1,
            self.stride_temporal,
        )
        datetimes = np.repeat(self._time_values[time_indices], n_spatial)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd

    def select_channels(self, ds, ch_type: str) -> NDArray[np.int64]:
        """Select channels based on stream info for either source or target."""

        channels = self.stream_info.get(ch_type)
        assert channels is not None, f"{ch_type} channels need to be specified"

        is_empty = len(channels) == 0 if channels is not None else False
        if is_empty:
            stream_name = self.stream_info["name"]
            _logger.warning(f"No channel for {stream_name} for {ch_type}.")
            chs_idx = np.empty(shape=[0], dtype=int)
            channels = []
        else:
            chs_idx = np.sort([self.channels_file.index(ch) for ch in channels])
            channels = [self.channels_file[i] for i in chs_idx]

        return np.array(chs_idx), channels


def _clip_lat(lats: NDArray) -> NDArray[np.float32]:
    """Clip latitudes to the range [-90, 90] and ensure periodicity."""
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)


def _clip_lon(lons: NDArray) -> NDArray[np.float32]:
    """Clip longitudes to the range [-180, 180] and ensure periodicity."""
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)


def _assemble_statistics_from_npz(src: str | Path) -> dict[str, dict[str, float]]:
    """
    Loads statistics saved with `save_statistics_npz`.
    Returns:
        dict[var_name, dict[stat_name, value]]
    """
    out: dict[str, dict[str, float]] = {}

    # If it's path-like, normalize to Path; otherwise assume it's file-like
    if isinstance(src, (str | Path)):
        src = Path(src)

    with np.load(src, allow_pickle=True) as z:
        variables = list(z["variables"])
        stat_names = [k for k in z.files if k != "variables"]

        for i, var in enumerate(variables):
            out[str(var)] = {}
            for stat in stat_names:
                out[str(var)][stat] = np.asarray(z[stat][i]).item()

    return out
