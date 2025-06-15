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

        # Determine source and target channels, filtering out forcings etc and using
        # specified source and target channels if specified
        source_channels = stream_info.get("source")
        self.source_idx = np.sort(
            [
                ds.name_to_index[k]
                for i, (k, v) in enumerate(ds0.typed_variables.items())
                if (
                    not v.is_computed_forcing
                    and not v.is_constant_in_time
                    and (
                        np.array([f in k for f in source_channels]).any()
                        if source_channels
                        else True
                    )
                )
            ]
        )
        target_channels = stream_info.get("target")
        self.target_idx = np.sort(
            [
                ds.name_to_index[k]
                for (k, v) in ds0.typed_variables.items()  # Only in ds0, not in Subset?
                if (
                    not v.is_computed_forcing
                    and not v.is_constant_in_time
                    and (
                        np.array([f in k for f in target_channels]).any()
                        if target_channels
                        else True
                    )
                )
            ]
        )
        self.source_channels = [ds.variables[i] for i in self.source_idx]
        self.target_channels = [ds.variables[i] for i in self.target_idx]
        self.geoinfo_channels = []
        self.geoinfo_idx = []

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
        self.len = 0

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

        didx_start = t_idxs[0]
        # End is inclusive
        didx_end = t_idxs[-1] + 1

        # extract number of time steps and collapse ensemble dimension
        # ds is a wrapper around zarr with get_coordinate_selection not being exposed since
        # subsetting is pushed to the ctor via frequency argument; this also ensures that no sub-
        # sampling is required here
        data = self.ds[didx_start:didx_end][:, :, 0].astype(np.float32)

        # extract channels
        data = (
            data[:, channels_idx].transpose([0, 2, 1]).reshape((data.shape[0] * data.shape[2], -1))
        )

        # construct lat/lon coords
        latlon = np.concatenate(
            [
                np.expand_dims(self.latitudes, 0),
                np.expand_dims(self.longitudes, 0),
            ],
            axis=0,
        ).transpose()
        coords = np.repeat(latlon, len(t_idxs), axis=0).reshape((-1, latlon.shape[1]))

        # empty geoinfos for anemoi
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
