# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
from pathlib import Path
from typing import override

import fsspec
import numpy as np
import xarray as xr
import zarr

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)

frequencies = {
    "3hrPt": np.timedelta64(10800000000000, "ns"),
    "day": np.timedelta64(86400000000000, "ns"),
    "fx": np.timedelta64(0, "ns"),
    "mon": np.timedelta64(2548800000000000, "ns"),
    "monC": np.timedelta64(2505600000000000, "ns"),
    "yr": np.timedelta64(31536000000000000, "ns"),
}


class DataReaderIconBase(DataReaderTimestep):
    "Wrapper for ICON data variables"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        stream_info: dict,
    ) -> None:
        """
        Parent class for ICON data variables

        Parameters
        ----------
        tw_handler : TimeWindowHandler
            Handles temporal slicing and mapping from time indices to datetime
        stream_info : dict
            Stream metadata
        """

        # Extract key metadata from stream_info
        lon_attribute = stream_info["attributes"]["lon"]
        lat_attribute = stream_info["attributes"]["lat"]
        mesh_attribute = stream_info["attributes"]["grid"]

        # Set mesh size based on spatial grid definition
        self.mesh_size = len(self.ds[mesh_attribute])

        # Time range in the dataset
        self.time = self.ds["time"].values
        start_ds = np.datetime64(self.time[0])
        end_ds = np.datetime64(self.time[-1])

        # Skip stream if it doesn't intersect with time window
        if start_ds > tw_handler.t_end or end_ds < tw_handler.t_start:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        # Compute temporal resolution if not already defined
        self.temporal_frequency = (
            self.time[1] - self.time[0]
            if self.temporal_frequency is None
            else self.temporal_frequency
        )

        # Initialize parent class with resolved time window
        super().__init__(
            tw_handler,
            stream_info,
            start_ds,
            end_ds,
            self.temporal_frequency,
        )

        # Compute absolute start/end indices in the dataset based on time window
        self.start_idx = (tw_handler.t_start - start_ds).astype("timedelta64[D]").astype(
            int
        ) * self.mesh_size
        self.end_idx = (
            (tw_handler.t_end - start_ds).astype("timedelta64[D]").astype(int) + 1
        ) * self.mesh_size - 1

        # Sanity check
        assert self.end_idx > self.start_idx, (
            f"Abort: Final index of {self.end_idx} is the same or smaller than start index {self.start_idx}"
        )

        # Number of time steps in selected range
        self.len = int((self.end_idx - self.start_idx) // self.mesh_size)

        # === Coordinates ===

        # Convert to degrees if stored in radians
        coords_units = self.ds[lat_attribute].attrs["units"]
        if coords_units == "radian":
            self.lat = np.rad2deg(self.ds[lat_attribute][:].astype("f"))
            self.lon = np.rad2deg(self.ds[lon_attribute][:].astype("f"))
        else:
            self.lat = self.ds[lat_attribute][:].astype("f")
            self.lon = self.ds[lon_attribute][:].astype("f")

        # Placeholder; currently unused
        self.step_hrs = 1

        # Stream metadata
        self.properties = {
            "stream_id": 0,
        }

        # === Normalization statistics ===

        # Ensure stats match dataset columns
        assert self.stats_vars == self.colnames, (
            f"Variables in normalization file {self.stats_vars} do not match dataset columns {self.colnames}"
        )

        # === Channel selection ===

        # Source channels
        source_channels = stream_info.get("source_channels")
        if source_channels:
            self.source_channels, self.source_idx = self.select(source_channels)
        else:
            self.source_channels = self.colnames
            self.source_idx = self.cols_idx

        # Target channels
        target_channels = stream_info.get("target_channels")
        if target_channels:
            self.target_channels, self.target_idx = self.select(target_channels)
        else:
            self.target_channels = self.colnames
            self.target_idx = self.cols_idx

        # Ensure all selected channels have valid standard deviations
        selected_channel_indices = list(set(self.source_idx).union(set(self.target_idx)))
        non_positive_stds = np.where(self.stdev[selected_channel_indices] <= 0)[0]
        assert len(non_positive_stds) == 0, (
            f"Abort: Encountered non-positive standard deviations for selected columns "
            f"{[self.colnames[selected_channel_indices][i] for i in non_positive_stds]}."
        )

        # === Geo-info channels (currently unused) ===
        self.geoinfo_channels = []
        self.geoinfo_idx = []

    def select(self, ch_filters: list[str]) -> (np.array, list[str]):
        """
        Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.

        Parameters
        ----------
        ch_filters: list[str]
            list of patterns to access

        Returns
        -------
        selected_colnames: np.array,
            Selected columns according to the patterns specified in ch_filters
        selected_cols_idx
            respective index of these patterns in the data array
        """
        mask = [np.array([f in c for f in ch_filters]).any() for c in self.colnames]

        selected_cols_idx = self.cols_idx[np.where(mask)[0]]
        selected_colnames = [self.colnames[int(i)] for i in np.where(mask)[0]]

        return selected_colnames, selected_cols_idx

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0

    @override
    def length(self) -> int:
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

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for temporal window

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

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # TODO: handle sub-sampling

        t_idxs_start = t_idxs[0]
        t_idxs_end = t_idxs[-1] + 1

        # datetime
        datetimes = np.asarray(self.time[t_idxs_start:t_idxs_end])

        # lat/lon coordinates + tiling to match time steps
        lat = self.lat.values[:, np.newaxis]
        lon = self.lon.values[:, np.newaxis]

        lat = np.tile(lat, len(datetimes))
        lon = np.tile(lon, len(datetimes))

        coords = np.concatenate([lat, lon], axis=1)

        # time coordinate repeated to match grid points
        datetimes = np.repeat(datetimes, self.mesh_size).reshape(-1, 1)
        datetimes = np.squeeze(datetimes)

        # expanding indexes for data
        start_row = t_idxs_start * self.mesh_size
        end_row = t_idxs_end * self.mesh_size

        # data
        channels = np.array(self.colnames)[channels_idx]

        data_reshaped = [
            np.asarray(self.ds[ch_]).reshape(-1, 1)[start_row:end_row] for ch_ in channels
        ]
        data = np.concatenate(data_reshaped, axis=1)

        # empty geoinfos
        geoinfos = np.zeros((data.shape[0], 0), dtype=data.dtype)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd


##########################
class DataReaderIcon(DataReaderIconBase):
    "Wrapper for ICON variables - This class reads Zarr format datasets"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        # Open Zarr dataset with Xarray
        self.ds = xr.open_zarr(filename, consolidated=True)

        # Column (variable) names and indices
        self.colnames = list(self.ds)
        self.cols_idx = np.array(list(np.arange(len(self.colnames))))

        # Will be inferred later based on the dataset’s time variable
        self.temporal_frequency = None

        # Load associated statistics file for normalization
        stats_filename = Path(filename).with_name(Path(filename).stem + "_stats.json")
        with open(stats_filename) as stats_file:
            self.stats = json.load(stats_file)

        # Extract variable list from stats metadata
        stats_vars_metadata = self.stats["metadata"]["variables"]
        self.stats_vars = [v for v in stats_vars_metadata if v not in {"clat", "clon", "time"}]

        # Load mean and standard deviation per variable
        self.mean = np.array(self.stats["statistics"]["mean"], dtype="d")
        self.stdev = np.array(self.stats["statistics"]["std"], dtype="d")

        # Delegate further initialization to the base class
        super().__init__(
            tw_handler,
            stream_info,
        )


##########################
class DataReaderIconCmip6(DataReaderIconBase):
    "Wrapper for ICON CMIP6 data variables - This class reads NetCDF4 using kerchunk"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        # Open the kerchunk-generated reference JSON
        ref_path = Path(filename)
        if not ref_path.exists():
            raise FileNotFoundError(f"Kerchunk reference JSON not found: {ref_path}")

        # Load JSON references and initialize a virtual file system
        kerchunk_ref = json.loads(ref_path.read_text())
        fs = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper = fs.get_mapper("")

        # Ensure metadata is consolidated for zarr-style access
        zarr.consolidate_metadata(mapper)

        # Open the dataset using Xarray with Zarr engine
        self.ds = xr.open_dataset(mapper, engine="zarr", consolidated=True)

        # Column (variable) names and indices
        self.colnames = stream_info["variables"]
        self.cols_idx = np.array(list(np.arange(len(self.colnames))))

        # Determine temporal frequency from dataset metadata
        frequency_attr = self.ds.attrs["frequency"]
        self.temporal_frequency = frequencies[frequency_attr]

        # Load associated statistics file for normalization
        stats_filename = Path(filename).with_name(Path(filename).stem + "_stats.json")
        with open(stats_filename) as stats_file:
            self.stats = json.load(stats_file)

        # Variables included in the stats
        self.stats_vars = list(self.stats)

        # Load mean and standard deviation per variable
        self.mean = np.array([self.stats[var]["mean"] for var in self.stats_vars], dtype=np.float64)
        self.stdev = np.array([self.stats[var]["std"] for var in self.stats_vars], dtype=np.float64)

        # Delegate further initialization to the base class
        super().__init__(
            tw_handler,
            stream_info,
        )
