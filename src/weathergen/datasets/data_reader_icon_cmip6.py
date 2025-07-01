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
import json
import fsspec
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
        '3hrPt': np.timedelta64(10800000000000, 'ns'),
        'day': np.timedelta64(86400000000000, 'ns'),
        'fx': np.timedelta64(0, 'ns'),
        'mon': np.timedelta64(2548800000000000, 'ns'),
        'monC': np.timedelta64(2505600000000000, 'ns'),
        'yr': np.timedelta64(31536000000000000, 'ns'),
}

class DataReaderIconCmip6(DataReaderTimestep):
    "Wrapper for ICON CMIP6 data variables"
    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:

        """
        Construct data reader for ICON CMIP6 data variables

        Parameters
        ----------
        filename  : Path
            filename (and path) of json kerchunk generated file
        stream_info : Omega object
            information about stream 

        Attributes
        ----------
        self.filename
        self.ds
        self.mesh_size
        self.colnames
        self.cols_idx
        self.stats
        self.time
        self.start_idx
        self.end_idx
        self.len
        self.lat
        self.lon
        self.step_hrs
        self.properties
        self.mean
        self.stdev
        self.source_channels
        self.source_idx
        self.target_channels
        self.target_idx
        self.geoinfo_channels
        self.geoinfo_idx

        Returns
        -------
        None
        """

        # retrieve variables names from stream file
        lon_attribute = stream_info["attributes"]["lon"]
        lat_attribute = stream_info["attributes"]["lat"]
        mesh_attribute = stream_info["attributes"]["grid"]
    
        self.filename = filename
        # Opening the dataset through kerchunk mapper
        ref_path = Path(self.filename)
        if not ref_path.exists():
            raise FileNotFoundError(f"Kerchunk reference JSON not found: {ref_path}")

        kerchunk_ref = json.loads(ref_path.read_text())
        fs = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper = fs.get_mapper("")
        # consolidate metadata – if the reference already has a .zmetadata this
        # is a no‑op; otherwise it creates an in‑memory view.
        # try:
        #     zarr.consolidate_metadata(mapper)
        # except Exception:
        #     pass
        zarr.consolidate_metadata(mapper)
        self.ds = xr.open_dataset(mapper, engine="zarr", consolidated=True)
        self.mesh_size = len(self.ds[mesh_attribute])

        # variables
        self.colnames = stream_info['variables']
        self.cols_idx = np.array(list(np.arange(len(self.colnames)))) 

        stats_filename = Path(filename).with_name(Path(filename).stem + "_stats.json")
        with open(stats_filename) as stats_file:
            self.stats = json.load(stats_file)

        stats_vars = list(self.stats)
        assert stats_vars == self.colnames, (
            f"Variables in normalization file {stats_vars} do not match dataset columns {self.colnames}"
        )

        # time
        self.time = self.ds["time"].values

        start_ds = self.time[0]
        end_ds = self.time[-1]

        if start_ds > tw_handler.t_end or end_ds < tw_handler.t_start:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        self.start_idx = (tw_handler.t_start - start_ds).astype("timedelta64[D]").astype(
            int
        ) * self.mesh_size
        self.end_idx = (
            (tw_handler.t_end - start_ds).astype("timedelta64[D]").astype(int) + 1
        ) * self.mesh_size - 1
        
        self.len = (self.end_idx - self.start_idx) // self.mesh_size

        assert self.end_idx > self.start_idx, (
            f"Abort: Final index of {self.end_idx} is the same of larger than start index {self.start_idx}"
        )

        frequency_attr = self.ds.attrs['frequency']
        period = frequencies[frequency_attr] 

        super().__init__(
            tw_handler,
            stream_info,
            start_ds,
            end_ds,
            period,
        )

        len_data_entries = len(self.ds["time"]) * self.mesh_size
        len_hrs = tw_handler.t_window_len
        assert self.end_idx + len_hrs <= len_data_entries, (
            f"Abort: end_date must be set at least {len_hrs} before the last date in the dataset"
        )

        # coordinates
        coords_units = self.ds[lat_attribute].attrs['units']

        if coords_units == "radian":
            self.lat = np.rad2deg(self.ds[lat_attribute][:].astype("f"))
            self.lon = np.rad2deg(self.ds[lon_attribute][:].astype("f"))

        else:
            self.lat = self.ds[lat_attribute][:].astype("f")
            self.lon = self.ds[lon_attribute][:].astype("f")

        # Ignore step_hrs, idk how it supposed to work
        # TODO, TODO, TODO:
        self.step_hrs = 1

        self.properties = {
            "stream_id": 0,
        }

        # stats
        self.mean = np.array(
            [self.stats[var]["mean"] for var in stats_vars],
            dtype=np.float64
        )
        self.stdev = np.array(
            [self.stats[var]["std"] for var in stats_vars],
            dtype=np.float64
        )

        source_channels = stream_info.get("source_channels")
        if source_channels:
            self.source_channels, self.source_idx = self.select(source_channels)
        else:
            self.source_channels = self.colnames
            self.source_idx = self.cols_idx

        target_channels = stream_info.get("target_channels")
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
        datetimes = self.time[t_idxs_start:t_idxs_end]
        # print("Before")
        # print(f"datetimes.shape = {datetimes.shape}", flush = True)

        # lat/lon coordinates + tiling to match time steps
        lat = self.lat.values[:, np.newaxis]
        lon = self.lon.values[:, np.newaxis]
        # print(f"lat.shape = {lat.shape}", flush = True)
        # print(f"lon.shape = {lon.shape}", flush = True)

        lat = np.tile(lat, len(datetimes))
        lon = np.tile(lon, len(datetimes))
        # print("\n\n")
        # print("After")
        # print(f"lat.shape = {lat.shape}", flush = True)
        # print(f"lon.shape = {lon.shape}", flush = True)

        coords = np.concatenate([lat, lon], axis=1)
        # print(f"coords.shape = {coords.shape}", flush = True)

        # time coordinate repeated to match grid points
        datetimes = np.repeat(datetimes, self.mesh_size).reshape(-1, 1)
        datetimes = np.squeeze(datetimes)
        # print(f"datetimes.shape = {datetimes.shape}", flush = True)

        # expanding indexes for data
        start_row = t_idxs_start * self.mesh_size
        end_row = t_idxs_end * self.mesh_size

        # data
        channels = np.array(self.colnames)[channels_idx]
        data_reshaped = [
            np.asarray(self.ds[ch_]).reshape(-1, 1)[start_row:end_row] for ch_ in channels
        ]
        data = np.concatenate(data_reshaped, axis=1)
        # print(f"len(data_reshaped) = {len(data_reshaped)}", flush = True)
        # print(f"data_reshaped.[0]shape = {data_reshaped[0].shape}", flush = True)
        # print("\n\n")

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
