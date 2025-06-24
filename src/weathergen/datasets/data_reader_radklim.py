# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
import json
from pathlib import Path
from typing import Any, override

import fsspec
import numpy as np
import xarray as xr
import zarr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    DType,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)


class RadklimKerchunkReader(DataReaderTimestep):
    """
    Reader for RADKLIM data accessed via a Kerchunk reference.

    The reader handles temporal subsetting, lazy loading, and normalization.
    """

    # Channels used for source and target data
    source_channels: list[str] = ["RR"]
    target_channels: list[str] = ["RR"]
    geoinfo_channels: list[str] = []

    # Channel indices
    source_idx: list[int] = [0]
    target_idx: list[int] = [0]
    geoinfo_idx: list[int] = []

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        stream_info: dict,
        *,
        chunks: dict[str, Any] | None = None,
    ) -> None:
        """
        Construct RADKLIM data reader from Kerchunk reference and stats JSON.

        Parameters
        ----------
        tw_handler : TimeWindowHandler
            Time window configuration
        stream_info : dict
            Dictionary with stream metadata including paths to reference and stats
        chunks : dict, optional
            Chunking options to apply to the xarray dataset

        Returns
        -------
        None
        """
        self._empty: bool = False

        ref_path = Path(stream_info["referece_path"])
        norm_path = Path(stream_info["stats_path"])

        # Load normalization statistics
        if not norm_path.exists():
            raise FileNotFoundError(f"normalisation JSON not found: {norm_path}")

        stats = json.loads(norm_path.read_text())
        self.mean = np.asarray(stats.get("mean", []), dtype=np.float32)
        self.stdev = np.asarray(stats.get("std", []), dtype=np.float32)
        self.mean_geoinfo = np.asarray(stats.get("mean_geoinfo", []), dtype=np.float32)
        self.stdev_geoinfo = np.asarray(stats.get("std_geoinfo", []), dtype=np.float32)

        if len(self.mean) != len(self.source_channels):
            raise ValueError("normalisation stats length does not match number of variables")

        # Load Kerchunk reference
        if not ref_path.exists():
            raise FileNotFoundError(f"Kerchunk reference JSON not found: {ref_path}")

        kerchunk_ref = json.loads(ref_path.read_text())
        fs = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper = fs.get_mapper("")

        # Consolidate metadata (may be a no-op depending on upstream)
        with contextlib.suppress(Exception):
            zarr.consolidate_metadata(mapper)

        ds_full = xr.open_dataset(mapper, engine="zarr", consolidated=True)
        times_full: NDArray[np.datetime64] = ds_full["time"].values
        if times_full.size == 0:
            super().__init__(tw_handler, stream_info, None, None, None)
            self.init_empty()
            return

        # Check time step regularity and define period
        deltas_sec = np.diff(times_full.astype("datetime64[s]"))
        unique_deltas = np.unique(deltas_sec)
        if unique_deltas.size != 1:
            raise ValueError("RADKLIM Kerchunk reference has irregular time steps")
        period = unique_deltas[0]

        data_start = times_full[0]
        data_end = times_full[-1]

        super().__init__(tw_handler, stream_info, data_start, data_end, period)

        # If there is no overlap with the time window, exit early
        if tw_handler.t_start >= data_end or tw_handler.t_end <= data_start:
            self.init_empty()
            return

        # Determine time index window for slicing
        self.start_idx = int(np.searchsorted(times_full, tw_handler.t_start, side="left"))
        self.end_idx = int(np.searchsorted(times_full, tw_handler.t_end, side="right"))

        # Subset and optionally rechunk the dataset
        subset = ds_full[self.source_channels].isel(time=slice(self.start_idx, self.end_idx))
        if chunks is not None:
            subset = subset.chunk(chunks)
        self.ds = subset

        # Load and flatten coordinate grid
        y1d = self.ds["y"].values.astype(np.float32)
        x1d = self.ds["x"].values.astype(np.float32)
        self.ny = len(y1d)
        self.nx = len(x1d)
        self.points_per_slice = self.ny * self.nx

        lat_var = self.ds["lat"]
        lon_var = self.ds["lon"]
        raw_lat = lat_var.isel(time=0).values if "time" in lat_var.dims else lat_var.values
        raw_lon = lon_var.isel(time=0).values if "time" in lon_var.dims else lon_var.values

        self.latitudes = _clip_lat(raw_lat)
        self.longitudes = _clip_lon(raw_lon)
        self._base_coords = np.column_stack(
            (self.latitudes.reshape(-1), self.longitudes.reshape(-1))
        ).astype(DType)

        self.num_steps_per_window = int(tw_handler.t_window_len / period)

    @override
    def init_empty(self) -> None:
        """
        Initialize empty dataset state.
        """
        self._empty = True
        super().init_empty()

    @override
    def length(self) -> int:
        """
        Total number of valid windows available for sampling.

        Returns
        -------
        int
            Number of available time windows
        """
        if self._empty:
            return 0
        nt = int(self.ds.sizes["time"])
        return max(0, nt - self.num_steps_per_window + 1)

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Extract a window of data, filtered by channels and validity.

        Parameters
        ----------
        idx : int
            Window index
        channels_idx : list of int
            Channel indices to extract

        Returns
        -------
        ReaderData
            Data window including coordinates, values, and timestamps
        """
        t_idxs_abs, dtr = self._get_dataset_idxs(idx)

        # Early return for empty or invalid request
        if self._empty or t_idxs_abs.size == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        # Convert global indices to local indices in dataset
        t_idxs_rel = t_idxs_abs - self.start_idx
        if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= self.ds.sizes["time"]):
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        # Slice window
        start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
        ds_win = self.ds.isel(time=slice(start, stop))

        # Stack into (time, y, x, var) format
        arr4 = (
            ds_win.to_array(dim="var")
            .transpose("time", "y", "x", "var")
            .values.astype(np.float32, copy=False)
        )
        nt, ny, nx, nvars = arr4.shape

        # Validate channel indices
        if not channels_idx:
            raise ValueError("channels_idx cannot be empty")
        if min(channels_idx) < 0 or max(channels_idx) >= nvars:
            raise IndexError("channels_idx out of bounds")
        if len(set(channels_idx)) != len(channels_idx):
            raise ValueError("channels_idx must be unique")

        # Flatten spatial/temporal dimensions
        flat_vars = arr4.reshape(-1, nvars)
        coords = np.tile(self._base_coords, (nt, 1))
        time_vals = ds_win["time"].values.astype("datetime64[ns]")
        times = np.repeat(time_vals, self.points_per_slice)

        # Apply nan filtering
        valid = ~np.any(np.isnan(flat_vars[:, channels_idx]), axis=1)
        if not np.any(valid):
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        coords_sel = coords[valid]
        data_sel = flat_vars[valid][:, channels_idx].astype(DType, copy=False)
        times_sel = times[valid]

        rdata = ReaderData(
            coords=coords_sel,
            geoinfos=np.zeros((coords_sel.shape[0], 0), dtype=DType),
            data=data_sel,
            datetimes=times_sel,
        )
        check_reader_data(rdata, dtr)
        return rdata


def _clip_lat(lats: NDArray[np.floating]) -> NDArray[np.float32]:
    """
    Clip latitudes to the range [-90, 90] and ensure periodicity.
    """
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)


def _clip_lon(lons: NDArray[np.floating]) -> NDArray[np.float32]:
    """
    Clip longitudes to the range [-180, 180] and ensure periodicity.
    """
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)
