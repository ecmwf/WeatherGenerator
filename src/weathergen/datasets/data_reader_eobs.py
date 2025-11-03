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

import dask.array as da
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    DTRange,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
    str_to_timedelta,
)

_logger = logging.getLogger(__name__)

# TODO make this datareader works with multiple datasets in ZARR format
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
        """
        Construct data reader for gridded Zarr dataset.

        Parameters
        ----------
        tw_handler : TimeWindowHandler
            Handler for time windows
        filename : Path
            Path to the Zarr dataset
        stream_info : dict
            Stream configuration containing channel selection and other metadata

        Returns
        -------
        None
        """
        # Store configuration but DO NOT open files here
        self._filename = filename
        self._tw_handler = tw_handler
        self._stream_info = stream_info

        # Initialize data-dependent attributes to None
        self.ds: xr.Dataset | None = None
        self.len = 0
        self.source_channels = []
        self.source_idx = []
        self.target_channels = []
        self.target_idx = []
        self.geoinfo_channels = []
        self.geoinfo_idx = []
        self.properties = {}
        
        # Grid properties
        self.latitudes: NDArray | None = None
        self.longitudes: NDArray | None = None
        self.n_lat: int = 0
        self.n_lon: int = 0
        self.n_points: int = 0

        # Statistics
        self.mean: NDArray | None = None
        self.stdev: NDArray | None = None

        # Call super() with temporary values
        super().__init__(self._tw_handler, self._stream_info)

        # Flag to ensure initialization happens only once per worker
        self._initialized = False

    def _lazy_init(self) -> None:
        """
        Initialize the dataset. Called once per worker process to ensure
        proper handling of file handles across processes.
        """
        if self._initialized:
            return

        try:
            # Open the Zarr dataset with xarray
            self.ds = xr.open_zarr(
                self._filename,
                consolidated=False,
                chunks=None,
                zarr_format=3
                )
        except Exception as e:
            name = self._stream_info["name"]
            _logger.error(f"Failed to open {name} at {self._filename}: {e}")
            self.init_empty()
            self._initialized = True
            return

        # Extract time coordinate
        time_coord = self.ds.coords["time"].values
        data_start_time = np.datetime64(time_coord[0])
        data_end_time = np.datetime64(time_coord[-1])

        # Check if dataset overlaps with requested time window
        if (
            self._tw_handler.t_start >= data_end_time
            or self._tw_handler.t_end <= data_start_time
        ):
            name = self._stream_info["name"]
            _logger.warning(
                f"{name} is not supported over data loader window. Stream is skipped."
            )
            self.init_empty()
            self._initialized = True
            return

        # Determine the period/frequency
        if len(time_coord) > 1:
            period = np.timedelta64(time_coord[1] - time_coord[0])
        else:
            # Default to daily if only one timestep
            period = np.timedelta64(1, "D")

        # Handle frequency override from stream_info
        if "frequency" in self._stream_info:
            period = str_to_timedelta(self._stream_info["frequency"])

        # Re-initialize parent class with correct time info
        super().__init__(
            self._tw_handler,
            self._stream_info,
            data_start_time,
            data_end_time,
            period,
        )

        # Calculate valid time range indices
        time_mask = (time_coord >= self._tw_handler.t_start) & (
            time_coord < self._tw_handler.t_end
        )
        self.len = int(np.sum(time_mask))

        if self.len <= 0:
            self.init_empty()
            self._initialized = True
            return

        # Extract and validate spatial coordinates
        self.latitudes = self.ds.coords["latitude"].values.astype(np.float32)
        self.longitudes = self.ds.coords["longitude"].values.astype(np.float32)

        # Validate coordinate ranges
        if np.any(self.latitudes < -90) or np.any(self.latitudes > 90):
            _logger.warning(
                f"Latitude values outside valid range [-90, 90] in stream "
                f"'{self._stream_info['name']}'"
            )
            self.latitudes = np.clip(self.latitudes, -90.0, 90.0)

        if np.any(self.longitudes < -180) or np.any(self.longitudes > 180):
            _logger.warning(
                f"Longitude values outside valid range [-180, 180] in stream "
                f"'{self._stream_info['name']}'. Converting from [0, 360] format."
            )
            self.longitudes = ((self.longitudes + 180.0) % 360.0 - 180.0).astype(
                np.float32
            )

        self.n_lat = len(self.latitudes)
        self.n_lon = len(self.longitudes)
        self.n_points = self.n_lat * self.n_lon

        # Identify available data variables (exclude coordinate and statistics variables)
        available_vars = [
            var
            for var in self.ds.data_vars
            if not var.endswith("_mean")
            and not var.endswith("_std")
            and "time" in self.ds[var].dims
        ]

        # Select source channels
        source_channels_filter = self._stream_info.get("source")
        source_exclude = self._stream_info.get("source_exclude", [])
        self.source_channels, self.source_idx = self._select_channels(
            available_vars, source_channels_filter, source_exclude
        )

        # Select target channels
        target_channels_filter = self._stream_info.get("target")
        target_exclude = self._stream_info.get("target_exclude", [])
        self.target_channels, self.target_idx = self._select_channels(
            available_vars, target_channels_filter, target_exclude
        )

        # No geoinfo channels for gridded data
        self.geoinfo_channels = []
        self.geoinfo_idx = []

        # Get target channel weights
        self.target_channel_weights = self.parse_target_channel_weights()

        # Load or compute statistics
        all_channels = sorted(set(self.source_channels + self.target_channels))
        self._load_statistics(all_channels)

        # Log configuration
        ds_name = self._stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: grid shape: {self.n_lat} x {self.n_lon}")

        self.properties = {
            "stream_id": self._stream_info.get("id", 0),
        }

        self._initialized = True

    def _select_channels(
        self,
        available_vars: list[str],
        include_filters: list[str] | None,
        exclude_filters: list[str] | None = None,
    ) -> tuple[list[str], list[int]]:
        """
        Select channels based on include/exclude filters.

        Parameters
        ----------
        available_vars : list[str]
            List of available variable names
        include_filters : list[str] | None
            List of patterns to include (None means include all)
        exclude_filters : list[str] | None
            List of patterns to exclude

        Returns
        -------
        tuple[list[str], list[int]]
            Selected channel names and their indices
        """
        if exclude_filters is None:
            exclude_filters = []

        selected = []
        for var in available_vars:
            # Check inclusion
            if include_filters is not None:
                if not any(f in var or f == var for f in include_filters):
                    continue

            # Check exclusion
            if any(f in var for f in exclude_filters):
                continue

            selected.append(var)

        # Return channels and their indices in the original list
        indices = [available_vars.index(ch) for ch in selected]
        return selected, indices

    def _load_statistics(self, channels: list[str]) -> None:
        """
        Load or compute statistics (mean and standard deviation) for channels.

        Parameters
        ----------
        channels : list[str]
            List of channel names for which to load statistics
        """
        means = []
        stds = []

        for ch in channels:
            # Try to load pre-computed statistics
            mean_var = f"{ch}_mean"
            std_var = f"{ch}_std"

            if mean_var in self.ds.data_vars:
                mean = float(self.ds[mean_var].values)
            else:
                _logger.warning(
                    f"No pre-computed mean for {ch}, using 0.0. "
                    "Consider computing statistics offline."
                )
                mean = 0.0

            if std_var in self.ds.data_vars:
                std = float(self.ds[std_var].values)
            else:
                _logger.warning(
                    f"No pre-computed std for {ch}, using 1.0. "
                    "Consider computing statistics offline."
                )
                std = 1.0

            means.append(mean)
            stds.append(std)

        self.mean = np.array(means, dtype=np.float32)
        self.stdev = np.array(stds, dtype=np.float32)

        # Avoid division by zero
        self.stdev[self.stdev <= 1e-5] = 1.0

    @override
    def init_empty(self) -> None:
        """Initialize an empty reader."""
        super().init_empty()
        self.ds = None
        self.len = 0
        self.n_points = 0

    @override
    def length(self) -> int:
        """Return the length of the dataset."""
        self._lazy_init()
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
            Selection of channel indices

        Returns
        -------
        ReaderData
            Data structure containing coords, geoinfos, data, and datetimes
        """
        self._lazy_init()

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        # Get the actual channel names
        all_channels = sorted(set(self.source_channels + self.target_channels))
        selected_channels = [all_channels[i] for i in channels_idx]

        # Extract data for selected timesteps and channels
        data_arrays = []
        datetimes_list = []

        for t_idx in t_idxs:
            if t_idx < 0 or t_idx >= len(self.ds.coords["time"]):
                continue

            # Extract data for this timestep
            timestep_data = []
            for ch in selected_channels:
                # Load data using isel for efficient indexing
                var_data = self.ds[ch].isel(time=t_idx).values.astype(np.float32)
                # Flatten spatial dimensions (lat, lon) -> (n_points,)
                var_data_flat = var_data.flatten()
                timestep_data.append(var_data_flat)

            # Stack channels: (n_points, n_channels)
            timestep_data = np.stack(timestep_data, axis=1)
            data_arrays.append(timestep_data)

            # Get datetime for this timestep
            dt = np.datetime64(self.ds.coords["time"].values[t_idx])
            datetimes_list.extend([dt] * self.n_points)

        if len(data_arrays) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        # Concatenate all timesteps: (n_timesteps * n_points, n_channels)
        data = np.vstack(data_arrays)

        # Create coordinate grid
        lon_grid, lat_grid = np.meshgrid(self.longitudes, self.latitudes)
        coords_single = np.stack(
            [lat_grid.flatten(), lon_grid.flatten()], axis=1
        ).astype(np.float32)

        # Repeat coordinates for each timestep
        coords = np.tile(coords_single, (len(t_idxs), 1))

        # Empty geoinfos
        geoinfos = np.zeros((len(data), 0), dtype=np.float32)

        # Convert datetimes to numpy array
        datetimes = np.array(datetimes_list, dtype="datetime64[ns]")

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )

        check_reader_data(rd, dtr)

        return rd