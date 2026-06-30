# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import override

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (  # type: ignore
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)


class DataReaderGREP(DataReaderTimestep):
    """
    Wrapper Data reader for gridded Zarr datasets with regular lat/lon structure.

    This reader handles datasets stored as Zarr with dimensions (time, latitude, longitude)
        beta: handling  dimensions (time_centered, nav_lat, nav_lon)
    Converts the gridded data to ReaderData format.
    """

    def __init__(
        self, tw_handler: TimeWindowHandler, filename: Path, stream_info: dict, stage: str
    ) -> None:
        """
        Construct data reader for Zarr GREP dataset

        Parameters
        ----------
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream
        stage :
            training stage

        Returns
        -------
        None
        """

        # Store configuration but DO NOT open files here (fork-safety for multiprocessing workers)
        self._filename = filename
        self._tw_handler = tw_handler
        self._stream_info = stream_info
        self._stage = stage
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

    def _lazy_init(self) -> None:
        """
        Open the dataset and populate all metadata.  Called once per worker.
        """
        if self._initialized:
            return
        self._initialized = True

        kwargs = {}
        store = Path(self._filename)
        if (store / "zarr.json").exists():
            kwargs = {"zarr_format": 3}
        elif (store / ".zgroup").exists():
            kwargs = {"zarr_format": 2}
        else:
            raise ValueError(f"Cannot determine Zarr format for {self._filename}")

        ds = xr.open_zarr(
            self._filename,
            consolidated=True,
            chunks=None,
            **kwargs,
        )

        # ---- Time axis -------------------------------------------------------

        for coord_name in ["time", "time_centered", "time_counter"]:
            if coord_name in ds.coords:
                time_coord: NDArray = ds.coords[coord_name].values
                break
        else:
            raise KeyError("Nessuna coordinata temporale trovata")

        data_start_time = np.datetime64(time_coord[0])
        data_end_time = np.datetime64(time_coord[-1])

        if self._tw_handler.t_start >= data_end_time or self._tw_handler.t_end <= data_start_time:
            name = self._stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
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
        grid_defs = [
            ("latitude", "longitude", False),  # regular grid
            ("nav_lat", "nav_lon", True),  # curvilinear
            ("TLAT", "TLON", True),  # curvilinear (alt)
        ]

        for lat_key, lon_key, _is_curvi in grid_defs:
            if lat_key in ds.coords and lon_key in ds.coords:
                break
        else:
            raise KeyError("No recognized grid found")

        _logger.info(
            f"Dataset '{self._stream_info['name']}' uses "
            f"{'curvilinear' if _is_curvi else 'regular'} grid ({lat_key}/{lon_key})."
        )

        self._curvilinear = _is_curvi

        lat = ds.coords[lat_key].values.astype(np.float32)
        lon = ds.coords[lon_key].values.astype(np.float32)

        lat = np.clip(lat, -90.0, 90.0)
        lon = ((lon + 180.0) % 360.0 - 180.0).astype(np.float32)

        if _is_curvi:
            self._nav_lat_flat = lat.ravel()
            self._nav_lon_flat = lon.ravel()

            self.n_lat, self.n_lon = lat.shape
            self.n_points = self.n_lat * self.n_lon

            self.latitudes = np.unique(self._nav_lat_flat)
            self.longitudes = np.unique(self._nav_lon_flat)
        else:
            self.latitudes = lat
            self.longitudes = lon

            self.n_lat = len(lat)
            self.n_lon = len(lon)
            self.n_points = self.n_lat * self.n_lon

        # ---- Available variables (non-stat, time-varying) --------------------
        time_variants = {"time", "time_centered", "time_counter"}
        available_vars: list[str] = [
            var
            for var in ds.data_vars
            if not var.endswith("_mean")
            and not var.endswith("_std")
            and any(t in ds[var].dims for t in time_variants)  # "time" in ds[var].dims
        ]

        # ---- Channel selection -----------------------------------------------
        # source_idx / target_idx are indices into available_vars (like Anemoi uses ds.variables).
        source_channels_filter = self._stream_info.get("source")
        source_exclude = self._stream_info.get("source_exclude", [])
        self.source_channels, self.source_idx = self._select_channels(
            available_vars, source_channels_filter, source_exclude
        )
        self.source_idx = list(self.source_idx)  # keep as list, consistent with base class
        _logger.info(
            f"{self._stream_info['name']} selected source channels: "
            f"{self.source_channels} (indices: {self.source_idx})"
        )

        target_channels_filter = self._stream_info.get("target")
        target_exclude = self._stream_info.get("target_exclude", [])
        self.target_channels, self.target_idx = self._select_channels(
            available_vars, target_channels_filter, target_exclude
        )
        self.target_idx = list(self.target_idx)
        _logger.info(
            f"{self._stream_info['name']} selected target channels: "
            f"{self.target_channels} (indices: {self.target_idx})"
        )

        self.geoinfo_channels = []
        self.geoinfo_idx = np.array([], dtype=np.int64)
        self.mean_geoinfo = np.zeros(0, dtype=np.float32)
        self.stdev_geoinfo = np.ones(0, dtype=np.float32)

        self.target_channel_weights = self.parse_target_channel_weights()

        # Coordinate grid — handle both regular and curvilinear grids
        if self._curvilinear:
            self.coords_single = np.stack([self._nav_lat_flat, self._nav_lon_flat], axis=1).astype(
                np.float32
            )

            # ---- mask ---------------------------------------------------------
            mask = None

            # case 1: # Maschera i punti dove punti sono nan o 0
            if "time_counter" in ds.dims:
                data_mask = ds[self.source_channels[0]].isel(time_counter=0)
                mask = np.isnan(data_mask) | (data_mask == 0)
                mask = mask.values.ravel()

            # case 2: # Maschera i punti dove nav_lat == 0 AND nav_lon == 0
            elif "nav_lat" in ds.coords and "nav_lon" in ds.coords:
                mask = (self._nav_lat_flat == 0.0) & (self._nav_lon_flat == 0.0)

            # case 3: # Maschera i punti dove punti sono nan
            elif "TLAT" in ds.coords and "TLON" in ds.coords:
                data_mask = ds[self.source_channels[0]].isel(time=0)
                mask = np.isnan(data_mask).values.ravel()

            if mask is not None:
                self.coords_single[mask] = np.nan

        else:
            lon_grid, lat_grid = np.meshgrid(self.longitudes, self.latitudes)
            self.coords_single = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1).astype(
                np.float32
            )

        # ---- Statistics ------------------------------------------------------
        # mean/stdev must be arrays of length == len(available_vars) so that
        # the base-class _normalize/_denormalize can index them with source_idx / target_idx.
        self.mean, self.stdev = self._load_statistics(available_vars, ds)

        # ---- Length (timesteps inside the window) ----------------------------
        time_mask = (time_coord >= self._tw_handler.t_start) & (time_coord < self._tw_handler.t_end)
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
        # self._lazy_init()

        if not channels_idx:
            return ReaderData.empty(
                num_data_fields=0,
                num_geo_fields=0,
            )

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            _logger.info(
                f"No valid time indices found for idx={idx}; returning empty data. "
                "(if self.ds is None or self.len == 0 or len(t_idxs) == 0:)"
            )
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=0,
            )

        # Map channel indices → variable names
        selected_channels = [self.available_vars[i] for i in channels_idx]
        # selected_channels = self.available_vars

        data_arrays: list[NDArray] = []
        datetimes_list: list[np.datetime64] = []

        for coord in ["time", "time_centered", "time_counter"]:
            if coord in self.ds.coords:
                time_values = self.ds.coords[coord].values
                break
        else:
            raise KeyError(
                f"No time coordinate found. Available coordinates: {list(self.ds.coords)}"
            )

        n_time = len(time_values)

        if self.log_debug:
            _logger.info(
                f"Available vars: {self.available_vars}, requested channels: {selected_channels}"
                f"\n Fetching data for idx={idx} (dataset time range: "
                f"{time_values[0]} to {time_values[-1]}), "
                f"\n Selected time indices: {t_idxs} -- len={len(t_idxs)}"
            )

        for t_idx in t_idxs:
            if t_idx < 0 or t_idx >= n_time:
                continue

            # (n_points, n_channels)
            for time_dim in ["time", "time_centered", "time_counter"]:
                if time_dim in self.ds.dims:
                    break
            else:
                raise KeyError("No time dimension found")

            timestep_data = np.stack(
                [
                    self.ds[ch].isel({time_dim: int(t_idx)}).values.astype(np.float32).flatten()
                    for ch in selected_channels
                ],
                axis=1,
            )

            data_arrays.append(timestep_data)
            dt = np.datetime64(time_values[t_idx])
            datetimes_list.extend([dt] * self.n_points)

        if not data_arrays:
            _logger.info(f"No valid time indices found for idx={idx}; returning empty data.")
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=0,
            )

        # (n_timesteps * n_points, n_channels)
        data = np.vstack(data_arrays)

        # use actual count, not len(t_idxs)
        n_valid_timesteps = len(data_arrays)

        # NOTE tmp: prima era len(t_idxs) ma se ci sono t_idxs invalidi
        # (es. fuori range del dataset) allora data_arrays sarà più corto di len(t_idxs).
        # Meglio usare n_valid_timesteps che è il numero reale di timesteps validi
        # che abbiamo effettivamente caricato.
        coords = np.tile(self.coords_single, (n_valid_timesteps, 1))

        geoinfos = np.zeros((len(data), 0), dtype=np.float32)
        datetimes = np.array(datetimes_list, dtype="datetime64[s]")

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        if self.log_debug:
            _logger.info(
                f"Constructed ReaderData with coords shape {coords.shape}, "
                f"geoinfos shape {geoinfos.shape}, data shape {data.shape}, "
                f"datetimes shape {datetimes.shape}"
            )
            _logger.info(
                f"  Sample coords: {coords[1000:1005]}, sample data: {data[1000:1005]}, "
                f"geoinfos: {geoinfos[1000:1005]}, sample datetimes: {datetimes[1000:1005]}"
            )
            _logger.info(f"  Channels in data: {selected_channels}")
            _logger.info(
                f"  data type: {data.dtype}, coords type: {coords.dtype}, "
                f"datetimes type: {datetimes.dtype}"
            )
            self.log_debug = False  # only log once per worker to avoid spamming

        check_reader_data(rd, dtr)

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
