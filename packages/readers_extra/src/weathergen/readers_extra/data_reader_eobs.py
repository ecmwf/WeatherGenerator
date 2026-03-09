# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import logging
from pathlib import Path
from typing import override

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)


class DataReaderEObs(DataReaderTimestep):
    """
    Data reader for gridded Zarr datasets with regular lat/lon structure.

    This reader handles datasets stored as Zarr with dimensions (time, latitude, longitude)
    and converts the gridded data to point-wise format required by the framework.

    The reader implements lazy initialization to work efficiently with multiple dataloader workers.
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        # Store configuration but DO NOT open files here (fork-safety for multiprocessing workers)
        self._filename = filename
        self._tw_handler = tw_handler
        self._stream_info = stream_info
        self._initialized = False

        # Call super() with placeholder period; will be overwritten after lazy init sets the real
        # data_start_time / data_end_time / period on this instance directly.
        super().__init__(tw_handler, stream_info)

        # Grid properties (populated during lazy init)
        self.latitudes: NDArray | None = None
        self.longitudes: NDArray | None = None
        self.n_lat: int = 0
        self.n_lon: int = 0
        self.n_points: int = 0

        # debug
        self.log_debug = True

        # Set empty defaults so the object is always in a valid state
        self.init_empty()
        self._lazy_init()

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _lazy_init(self) -> None:
        """
        Open the dataset and populate all metadata.  Called once per worker.
        """
        if self._initialized:
            return
        self._initialized = True  # set early so errors don't cause infinite recursion

        try:
            ds: xr.Dataset = xr.open_zarr(
                self._filename, consolidated=True, chunks=None, zarr_format=2
            )
        except Exception as e:
            name = self._stream_info["name"]
            _logger.error(f"Failed to open {name} at {self._filename}: {e}")
            return  # leave in empty state

        # ---- Time axis -------------------------------------------------------

        try:
            time_coord: NDArray = ds.coords["time"].values
        except KeyError:
            _logger.warning(f"Dataset for '{self._stream_info['name']}' does not contain 'time' coordinate; switch to 'time_centered'.")
            time_coord: NDArray = ds.coords["time_centered"].values

        data_start_time = np.datetime64(time_coord[0])
        data_end_time = np.datetime64(time_coord[-1])

        if (
            self._tw_handler.t_start >= data_end_time
            or self._tw_handler.t_end <= data_start_time
        ):
            name = self._stream_info["name"]
            _logger.warning(
                f"{name} is not supported over data loader window. Stream is skipped."
            )
            return  # leave in empty state

        if len(time_coord) > 1:
            period = np.timedelta64(time_coord[1] - time_coord[0])
        else:
            period = np.timedelta64(1, "D")

        if "frequency" in self._stream_info:
            # Reuse the same helper the base module uses (timedelta_to_str inverse).
            # The base module exposes no str_to_timedelta, so we parse manually.
            period = _str_to_timedelta(self._stream_info["frequency"])

        # Patch the instance attributes that DataReaderTimestep._get_dataset_idxs reads.
        # This is equivalent to what super().__init__(..., data_start_time, data_end_time, period)
        # would set, but without the side-effects of calling __init__ a second time.
        self.data_start_time = data_start_time
        self.data_end_time = data_end_time
        self.period = period

        # ---- Spatial grid ----------------------------------------------------
        try:
            self.latitudes = ds.coords["latitude"].values.astype(np.float32)
            self.longitudes = ds.coords["longitude"].values.astype(np.float32)
        except KeyError:
            _logger.warning(f"Dataset for '{self._stream_info['name']}' does not contain 'latitude'/'longitude' coordinates; switch to 'nav_lat'/'nav_lon'.")
            self.latitudes = ds.coords["nav_lat"].values.astype(np.float32)
            self.longitudes = ds.coords["nav_lon"].values.astype(np.float32)

        if np.any(self.latitudes < -90) or np.any(self.latitudes > 90):
            _logger.warning(
                f"Latitude values outside [-90, 90] in '{self._stream_info['name']}'; clipping."
            )
            self.latitudes = np.clip(self.latitudes, -90.0, 90.0)

        if np.any(self.longitudes < -180) or np.any(self.longitudes > 180):
            _logger.warning(
                f"Longitude values outside [-180, 180] in '{self._stream_info['name']}'; "
                "converting from [0, 360]."
            )
            self.longitudes = ((self.longitudes + 180.0) % 360.0 - 180.0).astype(np.float32)

        self.n_lat = len(self.latitudes)
        self.n_lon = len(self.longitudes)
        self.n_points = self.n_lat * self.n_lon

        # ---- Available variables (non-stat, time-varying) --------------------
        time_variants = {"time", "time_centered"}
        available_vars: list[str] = [
            var
            for var in ds.data_vars
            if not var.endswith("_mean")
            and not var.endswith("_std")
            and any(t in ds[var].dims for t in time_variants) #"time" in ds[var].dims
        ]

        # ---- Channel selection -----------------------------------------------
        # source_idx / target_idx are indices into available_vars (like Anemoi uses ds.variables).
        source_channels_filter = self._stream_info.get("source")
        source_exclude = self._stream_info.get("source_exclude", [])
        self.source_channels, self.source_idx = self._select_channels(
            available_vars, source_channels_filter, source_exclude
        )
        self.source_idx = list(self.source_idx)  # keep as list, consistent with base class
        _logger.info(f"{self._stream_info['name']} selected source channels: {self.source_channels} (indices: {self.source_idx})")

        target_channels_filter = self._stream_info.get("target")
        target_exclude = self._stream_info.get("target_exclude", [])
        self.target_channels, self.target_idx = self._select_channels(
            available_vars, target_channels_filter, target_exclude
        )
        self.target_idx = list(self.target_idx)
        _logger.info(f"{self._stream_info['name']} selected target channels: {self.target_channels} (indices: {self.target_idx})")

        self.geoinfo_channels = []
        self.geoinfo_idx = np.array([], dtype=np.int64)
        self.mean_geoinfo = np.zeros(0, dtype=np.float32)
        self.stdev_geoinfo = np.ones(0, dtype=np.float32)

        self.target_channel_weights = self.parse_target_channel_weights()

        # ---- Statistics ------------------------------------------------------
        # mean/stdev must be arrays of length == len(available_vars) so that
        # the base-class _normalize/_denormalize can index them with source_idx / target_idx.
        self.mean, self.stdev = self._load_statistics(available_vars, ds)

        # ---- Length (timesteps inside the window) ----------------------------
        time_mask = (
            (time_coord >= self._tw_handler.t_start)
            & (time_coord < self._tw_handler.t_end)
        )
        self.len = int(np.sum(time_mask))

        # ---- Keep reference to open dataset ----------------------------------
        self.ds = ds
        self.available_vars = available_vars

        ds_name = self._stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: grid shape: {self.n_lat} x {self.n_lon}")

        self.properties = {"stream_id": self._stream_info.get("stream_id", 0)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_channels(
        self,
        available_vars: list[str],
        include_filters: list[str] | None,
        exclude_filters: list[str] | None = None,
    ) -> tuple[list[str], NDArray[np.int64]]:
        """Return (channel_names, indices_into_available_vars)."""
        if exclude_filters is None:
            exclude_filters = []

        selected_names: list[str] = []
        selected_idxs: list[int] = []

        for i, var in enumerate(available_vars):
            if include_filters is not None:
                if not any(f in var or f == var for f in include_filters):
                    continue
            if any(f in var for f in exclude_filters):
                continue
            selected_names.append(var)
            selected_idxs.append(i)

        return selected_names, np.array(selected_idxs, dtype=np.int64)

    def _load_statistics(
        self, available_vars: list[str], ds: xr.Dataset
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Return mean and stdev arrays aligned with *available_vars* (not just selected channels).
        This matches the layout expected by DataReaderBase._normalize / _denormalize which index
        the arrays with source_idx / target_idx.
        """
        means = np.zeros(len(available_vars), dtype=np.float32)
        stds = np.ones(len(available_vars), dtype=np.float32)

        for i, ch in enumerate(available_vars):
            mean_var = f"{ch}_mean"
            std_var = f"{ch}_std"

            if mean_var in ds.data_vars:
                means[i] = float(ds[mean_var].values)
            else:
                _logger.warning(f"No pre-computed mean for {ch}, using 0.0.")

            if std_var in ds.data_vars:
                stds[i] = float(ds[std_var].values)
            else:
                _logger.warning(f"No pre-computed std for {ch}, using 1.0.")

        # Avoid division by zero
        stds[stds <= 1e-5] = 1.0
        return means, stds

    # ------------------------------------------------------------------
    # DataReaderBase / DataReaderTimestep overrides
    # ------------------------------------------------------------------

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self.len = 0
        self.n_points = 0
        self.available_vars = []

    @override
    def length(self) -> int:
        # self._lazy_init()
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for a time window.

        Parameters
        ----------
        idx : TIndex
            Index of temporal window
        channels_idx : list[int]
            Indices of channels to return, expressed as indices into *available_vars*
            (i.e. what base class stores as source_idx / target_idx).
        """
        self._lazy_init()

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            _logger.info(f"No valid time indices found for idx={idx}; returning empty data. (if self.ds is None or self.len == 0 or len(t_idxs) == 0:)")
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=0,
            )

        # Map channel indices → variable names
        selected_channels = [self.available_vars[i] for i in channels_idx]
        # selected_channels = self.available_vars

        data_arrays: list[NDArray] = []
        datetimes_list: list[np.datetime64] = []

        try:
            time_values = self.ds.coords["time"].values
        except KeyError:
            _logger.warning(f"Dataset for '{self._stream_info['name']}' does not contain 'time' coordinate; switch to 'time_centered'.")
            time_values = self.ds.coords["time_centered"].values

        n_time = len(time_values)

        if self.log_debug:
            _logger.info(
                f"Available vars: {self.available_vars}, requested channels: {selected_channels}"
                f"\n Fetching data for idx={idx} (dataset time range: {time_values[0]} to {time_values[-1]}), "
                f"\n Selected time indices: {t_idxs} -- len={len(t_idxs)}"
            )

        for t_idx in t_idxs:
            if t_idx < 0 or t_idx >= n_time:
                continue

            # (n_points, n_channels)
            try:
                timestep_data = np.stack(
                    [
                        self.ds[ch].isel(time=int(t_idx)).values.astype(np.float32).flatten()
                        for ch in selected_channels
                    ],
                    axis=1,
                )
            except ValueError:
                timestep_data = np.stack(
                    [
                        self.ds[ch].isel(time_centered=int(t_idx)).values.astype(np.float32).flatten()
                        for ch in selected_channels
                    ],
                    axis=1,
                )
                
            data_arrays.append(timestep_data)
            dt = np.datetime64(time_values[t_idx])
            datetimes_list.extend([dt] * self.n_points)

        if not data_arrays:
            _logger.info(f"No valid time indices found for idx={idx}; returning empty data. (if not data_arrays:)")
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=0,
            )

        # (n_timesteps * n_points, n_channels)
        data = np.vstack(data_arrays)


        # NOTE tmp
        n_valid_timesteps = len(data_arrays)  # use actual count, not len(t_idxs)

        # Coordinate grid — broadcast lat/lon into flat point list
        lon_grid, lat_grid = np.meshgrid(self.longitudes, self.latitudes)
        coords_single = np.stack(
            [lat_grid.flatten(), lon_grid.flatten()], axis=1
        ).astype(np.float32)
        
        # NOTE tmp: prima era len(t_idxs) ma se ci sono t_idxs invalidi (es. fuori range del dataset) allora data_arrays sarà più corto di len(t_idxs). Meglio usare len(data_arrays) che è il numero reale di timesteps validi che abbiamo effettivamente caricato.
        coords = np.tile(coords_single, (n_valid_timesteps, 1))

        geoinfos = np.zeros((len(data), 0), dtype=np.float32)
        datetimes = np.array(datetimes_list, dtype="datetime64[s]")

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        if self.log_debug:
            _logger.info(f"Constructed ReaderData with coords shape {coords.shape}, geoinfos shape {geoinfos.shape}, data shape {data.shape}, datetimes shape {datetimes.shape}")
            _logger.info(f"  Sample coords: {coords[:5]}, sample data: {data[:5]}, geoinfos: {geoinfos[:5]}, sample datetimes: {datetimes[:5]}")
            _logger.info(f"  Channels in data: {selected_channels}")

            _logger.info(f"  data type: {data.dtype}, coords type: {coords.dtype}, datetimes type: {datetimes.dtype}")

            coords_nan = np.isnan(coords).sum()
            coords_total = coords.size

            geoinfos_nan = np.isnan(geoinfos).sum()
            geoinfos_total = geoinfos.size

            data_nan = np.isnan(data).sum()
            data_total = data.size

            datetimes_nan = np.isnan(datetimes).sum()
            datetimes_total = datetimes.size

            _logger.info(f"NaN stats:")
            _logger.info(f"  {self._stream_info['name']} coords: {coords_nan}/{coords_total} ({coords_nan/coords_total:.4%})")
            _logger.info(f"  {self._stream_info['name']} geoinfos: {geoinfos_nan}/{geoinfos_total} ({geoinfos_nan/geoinfos_total:.4%})")
            _logger.info(f"  {self._stream_info['name']} data: {data_nan}/{data_total} ({data_nan/data_total:.4%})")
            _logger.info(f"  {self._stream_info['name']} datetimes: {datetimes_nan}/{datetimes_total} ({datetimes_nan/datetimes_total:.4%})")
            _logger.info(f"  {self._stream_info['name']} Channels in data: {selected_channels}")

        check_reader_data(rd, dtr)

        if self.log_debug:
            _logger.info(f"     {self._stream_info['name']} ReaderData check passed.")
            self.log_debug = False  # only log once per worker to avoid spamming

        return rd


# ---------------------------------------------------------------------------
# Module-level helper (replaces the non-existent str_to_timedelta import)
# ---------------------------------------------------------------------------

def _str_to_timedelta(s: str) -> np.timedelta64:
    """
    Parse simple frequency strings such as '6h', '1D', '30min' into np.timedelta64.
    Supported suffixes: h, H, D, min, T, s, S.
    """
    import re

    m = re.fullmatch(r"(\d+)\s*(h|H|D|min|T|s|S)", s.strip())
    if m is None:
        raise ValueError(f"Cannot parse frequency string: {s!r}")
    value = int(m.group(1))
    unit_map = {"h": "h", "H": "h", "D": "D", "min": "m", "T": "m", "s": "s", "S": "s"}
    return np.timedelta64(value, unit_map[m.group(2)])