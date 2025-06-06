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
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    # DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
    get_dataset_indexes_periodic,
)

_logger = logging.getLogger(__name__)


class DataReaderAnemoi(DataReaderBase):
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

        super().__init__(tw_handler)

        # open  dataset to peak that it is compatible with requested parameters
        ds = anemoi_datasets.open_dataset(filename)

        # check that start and end time are within the dataset time range
        ds_dt_start = ds.dates[0]
        ds_dt_end = ds.dates[-1]
        assert ds_dt_start is not None and ds_dt_end is not None, (ds_dt_start, ds_dt_end)
        # assert ds_dt_start <= tw_handler.t_start, (ds_dt_start, tw_handler)
        # assert ds_dt_end >= tw_handler.t_end, (ds_dt_end, tw_handler)

        # open dataset

        # caches lats and lons
        self.latitudes = _clip_lat(ds.latitudes)
        self.longitudes = _clip_lon(ds.longitudes)

        # Determine source and target channels, filtering out forcings etc and using
        # specified source and target channels if specified
        source_channels = stream_info.get("source")
        self.source_idx = np.sort(
            [
                ds.name_to_index[k]
                for i, (k, v) in enumerate(ds.typed_variables.items())
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
                for (k, v) in ds.typed_variables.items()
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

        _logger.info(f"Source channels: {self.source_channels}")
        _logger.info(f"Target channels: {self.target_channels}")

        self.properties = {  # TODO: unused?
            "stream_id": 0,
        }
        self.mean = ds.statistics["mean"]
        self.stdev = ds.statistics["stdev"]

        # set dataset to None when no overlap with time range
        if tw_handler.t_start >= ds_dt_end or tw_handler.t_end <= ds_dt_start:
            self.ds = None
        else:
            # TODO: specify frequency more flexibly and with finer granularity if necessary
            f = str(stream_info["frequency"]) + "h" if "frequency" in stream_info else ds.frequency
            self.ds = anemoi_datasets.open_dataset(
                ds, frequency=f, start=tw_handler.t_start, end=tw_handler.t_end
            )
            # self.dates = self.ds.dates
            self.len = len(self.ds)
            # self.ds_start_time = self.ds.start_date
            # self.ds_end_time = self.ds.end_date
            # self.sub_sampling_per_window = 1
            # self.frequency = self.ds.frequency

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

        # rdata = ReaderData()

        # TODO: we will always assume it return a range -> no need to build array
        t_idxs = get_dataset_indexes_periodic(
            self.ds.start_date,
            self.ds.end_date,
            np.timedelta64(self.ds.frequency),
            idx,
            self.time_window_handler,
        )
        # TODO: this would not work with sub-sampling
        didx_start = t_idxs[0]
        # End is inclusive
        didx_end = t_idxs[-1] + 1

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty()

        # extract number of time steps and collapse ensemble dimension
        # ds is a wrapper around zarr with get_coordinate_selection not being exposed since
        # subsetting is pushed to the ctor via frequency argument; this also ensures that no sub-
        # sampling is required here
        # TODO: should we convert to float32?
        data = self.ds[didx_start:didx_end][:, :, 0]

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
        check_reader_data(rd)
        return rd


def _clip_lat(lats: NDArray) -> NDArray[np.float32]:
    """
    Clip latitudes to the range [-90, 90] and ensure periodicity.
    """
    return (2 * np.clip(lats, -90, 90) - lats).astype(np.float32)


def _clip_lon(lons: NDArray) -> NDArray[np.float32]:
    """
    Clip longitudes to the range [-180, 180] and ensure periodicity.
    """
    return ((lons + 180) % 360 - 180).astype(np.float32)
