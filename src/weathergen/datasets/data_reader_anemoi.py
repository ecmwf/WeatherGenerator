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

import anemoi.datasets as anemoi_datasets
import numpy as np
from anemoi.datasets.data import MissingDateError
from anemoi.datasets.data.dataset import Dataset
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
    str_to_timedelta,
)

_logger = logging.getLogger(__name__)


class DataReaderAnemoi(DataReaderTimestep):
    "Wrapper for Anemoi datasets"

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

        # open  dataset to peak that it is compatible with requested parameters
        ds0: Dataset = anemoi_datasets.open_dataset(filename)
        # If there is no overlap with the time range, the dataset will be empty
        if tw_handler.t_start >= ds0.dates[-1] or tw_handler.t_end <= ds0.dates[0]:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        kwargs = {}
        if "frequency" in stream_info:
            kwargs["frequency"] = str_to_timedelta(stream_info["frequency"])
        if "subsampling_rate" in stream_info:
            name = stream_info["name"]
            _logger.warning(
                f"subsampling_rate specified for anemoi dataset for stream {name}. "
                + "Use frequency instead."
            )
        ds: Dataset = anemoi_datasets.open_dataset(
            ds0, **kwargs, start=tw_handler.t_start, end=tw_handler.t_end
        )

        period = np.timedelta64(ds.frequency)
        data_start_time = ds.dates[0]
        data_end_time = ds.dates[-1]
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
            self.init_empty()
            return
        else:
            self.ds = ds
            self.len = len(ds)

        # caches lats and lons
        self.latitudes = _clip_lat(ds.latitudes)
        self.longitudes = _clip_lon(ds.longitudes)

        # select/filter requested source channels
        self.source_idx = self.select_channels(ds0, "source")
        self.source_channels = [ds.variables[i] for i in self.source_idx]

        # select/filter requested target channels
        self.target_idx = self.select_channels(ds0, "target")
        self.target_channels = [ds.variables[i] for i in self.target_idx]

        # get target channel weights from stream config
        self.target_channel_weights = self.parse_target_channel_weights()

        # select/filter requested geoinfo channels (static/constant-in-time variables)
        self.geoinfo_idx = self.select_geoinfo_channels(ds0)
        self.geoinfo_channels = [ds.variables[i] for i in self.geoinfo_idx]

        # set geoinfo normalization statistics and cache geoinfo data
        if len(self.geoinfo_idx) > 0:
            self.mean_geoinfo = ds.statistics["mean"][self.geoinfo_idx]
            self.stdev_geoinfo = ds.statistics["stdev"][self.geoinfo_idx]
            # Cache geoinfo data once (constant in time, no need to read every epoch)
            # Read from first timestep and store for reuse
            geoinfo_data = ds[0:1][:, list(self.geoinfo_idx), 0]
            # Shape: (1, num_geoinfo, num_gridpoints) -> (num_gridpoints, num_geoinfo)
            self._cached_geoinfo = geoinfo_data[0].transpose().astype(np.float32)

            # Log diagnostic info for geoinfo statistics
            ds_name = stream_info["name"]
            for i, ch_idx in enumerate(self.geoinfo_idx):
                ch_name = ds.variables[ch_idx]
                mean_val = self.mean_geoinfo[i]
                stdev_val = self.stdev_geoinfo[i]
                nan_count = np.isnan(self._cached_geoinfo[:, i]).sum()
                if stdev_val == 0 or np.isclose(stdev_val, 0):
                    _logger.warning(
                        f"{ds_name}: geoinfo channel '{ch_name}' has stdev=0 "
                        "(constant field, will skip division in normalization)"
                    )
                if nan_count > 0:
                    _logger.warning(
                        f"{ds_name}: geoinfo channel '{ch_name}' has {nan_count} NaN values "
                        f"({100*nan_count/len(self._cached_geoinfo):.2f}% of grid points)"
                    )
                _logger.debug(
                    f"{ds_name}: geoinfo '{ch_name}' - mean={mean_val:.4f}, stdev={stdev_val:.4f}"
                )

            # Replace NaN values in cached geoinfo with 0 (after normalization this will be neutral)
            total_nans = np.isnan(self._cached_geoinfo).sum()
            if total_nans > 0:
                _logger.warning(
                    f"{ds_name}: Replacing {total_nans} total NaN values in geoinfo with 0"
                )
                self._cached_geoinfo = np.nan_to_num(self._cached_geoinfo, nan=0.0)
        else:
            self.mean_geoinfo = np.zeros(0)
            self.stdev_geoinfo = np.ones(0)
            self._cached_geoinfo = None

        ds_name = stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: geoinfo channels: {self.geoinfo_channels}")

        self.properties = {
            "stream_id": 0,
        }
        self.mean = ds.statistics["mean"]
        self.stdev = ds.statistics["stdev"]

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self.len = 0
        self._cached_geoinfo = None

    @override
    def length(self) -> int:
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
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"
        didx_start = t_idxs[0]
        # End is inclusive
        didx_end = t_idxs[-1] + 1

        # extract number of time steps and collapse ensemble dimension
        # ds is a wrapper around zarr with get_coordinate_selection not being exposed since
        # subsetting is pushed to the ctor via frequency argument; this also ensures that no sub-
        # sampling is required here
        try:
            data = self.ds[didx_start:didx_end][:, :, 0].astype(np.float32)
        except MissingDateError as e:
            _logger.debug(f"Date not present in anemoi dataset: {str(e)}. Skipping.")
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # extract channels
        data = (
            data[:, list(channels_idx)]
            .transpose([0, 2, 1])
            .reshape((data.shape[0] * data.shape[2], -1))
        )

        # construct lat/lon coords
        latlon = np.concatenate(
            [
                np.expand_dims(self.latitudes, 0),
                np.expand_dims(self.longitudes, 0),
            ],
            axis=0,
        ).transpose()
        # repeat latlon len(t_idxs) times
        coords = np.vstack((latlon,) * len(t_idxs))

        # use cached geoinfo data (no disk read needed - already loaded during init)
        if self._cached_geoinfo is not None:
            # repeat cached geoinfo for all timesteps
            geoinfos = np.vstack((self._cached_geoinfo,) * len(t_idxs))
        else:
            geoinfos = np.zeros((len(data), 0), dtype=data.dtype)

        # date time matching #data points of data
        # Assuming a fixed frequency for the dataset
        datetimes = np.repeat(self.ds.dates[didx_start:didx_end], len(data) // len(t_idxs))

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd

    def select_channels(self, ds0: anemoi_datasets, ch_type: str) -> NDArray[np.int64]:
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
        channels_exclude = self.stream_info.get(ch_type + "_exclude", [])
        # sanity check
        is_empty = len(channels) == 0 if channels is not None else False
        if is_empty:
            stream_name = self.stream_info["name"]
            _logger.warning(f"No channel for {stream_name} for {ch_type}.")

        chs_idx = np.sort(
            [
                ds0.name_to_index[k]
                for (k, v) in ds0.typed_variables.items()
                if (
                    not v.is_computed_forcing
                    and not v.is_constant_in_time
                    and (
                        np.array([f in k for f in channels]).any() if channels is not None else True
                    )
                    and not np.array([f in k for f in channels_exclude]).any()
                )
            ]
        )

        return np.array(chs_idx, dtype=np.int64)

    def select_geoinfo_channels(self, ds0: anemoi_datasets) -> NDArray[np.int64]:
        """
        Select geoinfo channels (static/constant-in-time variables)

        Parameters
        ----------
        ds0 :
            raw anemoi dataset with available channels

        Returns
        -------
        NDArray of channel indices for geoinfo variables

        """

        geoinfo_channels = self.stream_info.get("geoinfo", [])

        if not geoinfo_channels:
            return np.array([], dtype=np.int64)

        # Select channels that are constant in time and match the geoinfo list
        chs_idx = np.sort(
            [
                ds0.name_to_index[k]
                for (k, v) in ds0.typed_variables.items()
                if (
                    v.is_constant_in_time
                    and not v.is_computed_forcing
                    and np.array([f in k for f in geoinfo_channels]).any()
                )
            ]
        )

        if len(chs_idx) == 0 and len(geoinfo_channels) > 0:
            stream_name = self.stream_info["name"]
            _logger.warning(
                f"No matching geoinfo channels found for {stream_name}. "
                f"Requested: {geoinfo_channels}"
            )

        return np.array(chs_idx, dtype=np.int64)


def _clip_lat(lats: NDArray) -> NDArray[np.float32]:
    """
    Clip latitudes to the range [-90, 90] and ensure periodicity.
    """
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)


def _clip_lon(lons: NDArray) -> NDArray[np.float32]:
    """
    Clip longitudes to the range [-180, 180] and ensure periodicity.
    """
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)
