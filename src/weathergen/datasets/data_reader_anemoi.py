# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from pathlib import Path
from typing import override

import anemoi.datasets as anemoi_datasets
import numpy as np
from numpy.typing import ArrayLike

from weathergen.datasets.data_reader_base import DataReaderTimestep, ReaderData

_logger = logging.getLogger(__name__)


class DataReaderAnemoi(DataReaderTimestep):
    "Wrapper for Anemoi datasets"

    def __init__(
        self,
        start: int,
        end: int,
        t_window_len: int,
        t_window_step: int,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """
        Construct data reader for anemoi dataset

        Parameters
        ----------
        start : int
            Start time
        end : int
            End time
        t_window_len : int
            length of data window
        t_window_step :
            delta hours between start times of windows
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        super().__init__(start, end, t_window_len, t_window_step, filename, stream_info)

        # open  dataset to peak that it is compatible with requested parameters
        ds = anemoi_datasets.open_dataset(filename)

        # check that start and end time are within the dataset time range
        ds_dt_start = ds.dates[0]
        ds_dt_end = ds.dates[-1]

        format_str = "%Y%m%d%H%M%S"
        dt_start = datetime.datetime.strptime(str(start), format_str)
        dt_end = datetime.datetime.strptime(str(end), format_str)

        # open dataset

        # caches lats and lons
        self.latitudes = ds.latitudes.astype(np.float32)
        self.longitudes = ds.longitudes.astype(np.float32)

        # Ensures that coordinates remain into the interval [-90,90] for latitudes
        # and [-180, 180] for longitudes. Ensures that periodicity has been taken
        # into consideration for the specific intervals.
        self.latitudes = 2 * np.clip(self.latitudes, -90, 90) - self.latitudes

        self.longitudes = (self.longitudes + 180) % 360 - 180

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

        self.properties = {
            "stream_id": 0,
        }
        self.mean = ds.statistics["mean"]
        self.stdev = ds.statistics["stdev"]

        # set dataset to None when no overlap with time range
        if dt_start >= ds_dt_end or dt_end <= ds_dt_start:
            self.ds = None
        else:
            # TODO: specify frequency more flexibly and with finer granularity if necessary
            f = str(stream_info["frequency"]) + "h" if "frequency" in stream_info else ds.frequency
            self.ds = anemoi_datasets.open_dataset(ds, frequency=f, start=dt_start, end=dt_end)
            self.dates = self.ds.dates
            self.len = len(self.ds)
            self.ds_start_time = self.ds.start_date
            self.ds_end_time = self.ds.end_date
            self.sub_sampling_per_window = 1
            self.frequency = self.ds.frequency

    @override
    def _get(self, idx: int, channels_idx: ArrayLike) -> ReaderData:
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

        rdata = ReaderData()
        t_idxs = self._get_dataset_idxs(idx)

        if self.len == 0 or len(t_idxs) == 0:
            return rdata

        # extract number of time steps and collapse ensemble dimension
        # ds is a wrapper around zarr with get_coordinate_selection not being exposed since
        # subsetting is pushed to the ctor via frequency argument; this also ensures that no sub-
        # sampling is required here
        rdata.data = self.ds[t_idxs[0] : t_idxs[-1] + 1][:, :, 0]

        # extract channels
        rdata.data = (
            rdata.data[:, channels_idx]
            .transpose([0, 2, 1])
            .reshape((rdata.data.shape[0] * rdata.data.shape[2], -1))
        )

        # construct lat/lon coords
        latlon = np.concatenate(
            [
                np.expand_dims(self.latitudes, 0),
                np.expand_dims(self.longitudes, 0),
            ],
            axis=0,
        ).transpose()
        rdata.coords = np.repeat(latlon, len(t_idxs), axis=0).reshape((-1, latlon.shape[1]))

        # empty geoinfos for anemoi
        rdata.geoinfos = np.zeros((rdata.data.shape[0], 0), dtype=rdata.data.dtype)

        # date time matching #data points of data
        rdata.datetimes = np.repeat(
            np.expand_dims(self.ds.dates[t_idxs[0] : t_idxs[-1] + 1], 0),
            rdata.data.shape[0],
            axis=0,
        ).flatten()

        return rdata
