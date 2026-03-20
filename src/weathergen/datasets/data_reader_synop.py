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

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
    _clip_lat,
    _clip_lon,
)

_logger = logging.getLogger(__name__)


class DataReaderSynop(DataReaderTimestep):
    "Wrapper for SYNOP datasets in NetCDF"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """
        Construct data reader for anemoi dataset

        Parameters
        ----------
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        np32 = np.float32

        # open  dataset to peak that it is compatible with requested parameters
        ds = xr.open_dataset(filename, engine="netcdf4")

        # If there is no overlap with the time range, the dataset will be empty
        if tw_handler.t_start >= ds.time.max() or tw_handler.t_end <= ds.time.min():
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self._init_empty()
            return

        if "frequency" in stream_info:
            assert False, "Frequency sub-sampling currently not supported"

        period = (ds.time[1] - ds.time[0]).values
        data_start_time = ds.time[0].values
        data_end_time = ds.time[-1].values
        assert data_start_time is not None and data_end_time is not None, (
            data_start_time,
            data_end_time,
        )
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time,
            data_end_time,
            period,
        )
        # If there is no overlap with the time range, no need to keep the dataset.
        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            self._init_empty()
            return
        else:
            self.ds = ds
            self.len = len(ds)

        self.fillvalue = self.stream_info.get("fillvalue", None)
        self.channels_file = list(ds.keys())

        # Resolve coordinates
        lat_name = stream_info.get("latitude_name", "latitude")
        lon_name = stream_info.get("longitude_name", "longitude")

        self.latitudes = _clip_lat(np.array(lat_name, dtype=np32))
        self.longitudes = _clip_lon(np.array(lon_name, dtype=np32))

        # Resolve geoinfos
        self.geoinfo_channels = stream_info.get("geoinfos", [])
        self.geoinfo_idx = [self.channels_file.index(ch) for ch in self.geoinfo_channels]
        geoinfo_data_list = []
        for ch in self.geoinfo_channels:
            geoinfo_data_list.append(np.array(ch, dtype=np32))

        if geoinfo_data_list:
            self.geoinfo_data = np.stack(geoinfo_data_list).transpose()
        else:
            self.geoinfo_data = np.zeros((len(self.latitudes), 0), dtype=np32)

        # select/filter requested source channels
        self.source_idx = self.select_channels(ds, "source")
        self.source_channels = [self.channels_file[i] for i in self.source_idx]

        # select/filter requested target channels
        self.target_idx = self.select_channels(ds, "target")
        self.target_channels = [self.channels_file[i] for i in self.target_idx]

        ds_name = stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: geoinfo channels: {self.geoinfo_channels}")

        self.properties = {
            "stream_id": 0,
        }

        # Load mean and stdev from data file if specified in stream config, otherwise compute
        self.mean, self.stdev = self._load_or_compute_mean_stdev()
        self.mean_geoinfo = self.mean[self.geoinfo_idx]
        self.stdev_geoinfo = self.stdev[self.geoinfo_idx]

    def _load_or_compute_mean_stdev(self) -> (np.array, np.array):
        """
        Load mean and stdev from data file if specified in stream config, otherwise compute.

        Returns: (np.array, np.array)
            Mean and standard deviation arrays for all channels
        """
        mean_key = self.stream_info.get("mean_key")
        stdev_key = self.stream_info.get("stdev_key")

        if mean_key and mean_key in self.ds.keys() and stdev_key and stdev_key in self.ds.keys():
            _logger.info(f"Loading mean from '{mean_key}' and stdev from '{stdev_key}'")
            mean = np.array(self.ds[mean_key], dtype=np.float64)
            stdev = np.array(self.ds[stdev_key], dtype=np.float64)
            
            # Validate that the loaded mean and stdev have the correct shape
            expected_len = len(self.channels_file)
            if len(mean) != expected_len or len(stdev) != expected_len:
                _logger.warning(
                    f"Pre-computed statistics have incorrect shape "
                    f"(mean: {len(mean)}, stdev: {len(stdev)}, expected: {expected_len}). "
                    f"Falling back to computation."
                )
                return self._compute_mean_stdev()

            _logger.info(f"Finished loading mean and stdev.")
            
            return mean, stdev

        # Fall back to computing mean and stdev
        return self._compute_mean_stdev()

    def _compute_mean_stdev(self) -> (np.array, np.array):
        _logger.info("Starting computation of mean and stdev.")

        mean, stdev = [], []

        for ch in self.channels_file:
            data = np.array(self.ds[ch], np.float64)
            if self.fillvalue is not None:
                mask = data == self.fillvalue
                data[mask] = np.nan
            mean += [np.nanmean(data.flatten())]
            stdev += [np.nanstd(data.flatten())]

        mean = np.array(mean)
        stdev = np.array(stdev)

        _logger.info("Finished computation of mean and stdev.")

        return mean, stdev

    @override
    def _init_empty(self) -> None:
        super()._init_empty()
        self.ds = None
        self.len = 0

    @override
    def length(self) -> int:
        """
        Length of dataset

        Return :
        Length
        """
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for window (for either source or target, through public interface)

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx),
            )

        assert t_idxs[0] >= 0, "index must be non-negative"
        didx_start = t_idxs[0]
        # End is inclusive
        didx_end = t_idxs[-1] + 1

        # extract number of time steps and collapse ensemble dimension
        # ds is a wrapper around zarr with get_coordinate_selection not being exposed since
        # subsetting is pushed to the ctor via frequency argument; this also ensures that no sub-
        # sampling is required here
        sel_channels = [self.channels_file[i] for i in channels_idx]
        data = self.ds[sel_channels].isel(time=slice(didx_start, didx_end)).to_array()

        # filter the spatial dimension and reorder to (time * spatial, var)
        dims = list(data.dims)
        ax_var = dims.index("variable")
        ax_time = dims.index("time")
        ax_spatial = next(i for i in range(len(dims)) if i not in (ax_var, ax_time))
        data = np.transpose(data.values, [ax_time, ax_spatial, ax_var])
        # flatten (time, spatial) into a single leading dimension
        data = data.reshape(-1, len(sel_channels))

        # set invalid values to NaN for MetNo nc files
        if self.fillvalue is not None:
            mask = data == self.fillvalue
            data[mask] = np.nan

        # construct lat/lon coords
        latlon = np.concatenate(
            [
                np.expand_dims(self.latitudes, 0),
                np.expand_dims(self.longitudes, 0),
            ],
            axis=0,
        ).transpose()

        # repeat len(t_idxs) times
        coords = np.vstack((latlon,) * len(t_idxs))
        geoinfos = np.vstack((self.geoinfo_data,) * len(t_idxs))

        # date time matching #data points of data
        datetimes = np.repeat(self.ds.time[didx_start:didx_end].values, len(data) // len(t_idxs))

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd

    def select_channels(self, ds, ch_type: str) -> NDArray[np.int64]:
        """
        Select source or target channels

        Parameters
        ----------
        ds0 :
            raw anemoi dataset with available channels
        ch_type :
            "source" or "target", i.e channel type to select

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes

        """

        channels = self.stream_info.get(ch_type)
        assert channels is not None, f"{ch_type} channels need to be specified"
        # sanity check
        is_empty = len(channels) == 0 if channels is not None else False
        if is_empty:
            stream_name = self.stream_info["name"]
            _logger.warning(f"No channel for {stream_name} for {ch_type}.")

        chs_idx = np.sort([self.channels_file.index(ch) for ch in channels])

        return np.array(chs_idx)