# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
import torch

import zarr
from zfdb import (
    ChunkAxisType,
    FdbSource,
    FdbZarrArray,
    FdbZarrGroup,
    FdbZarrMapping,
    Request,
)
from zfdb.datasources import make_lat_long_sources
from zfdb.datasources import make_dates_source
import pyfdb


class ZFDBDataset:
    "FDB->Zarr dataset"

    def __init__(
        self,
        start: int,
        end: int,
        len_hrs: int,
        step_hrs: int,
        filename: str,
        stream_info: dict,
    ) -> None:
        """
        Construct dataset

        Parameters
        ----------
        start : int
            Start time
        end : int
            End time
        len_hrs : int
            length of data window
        step_hrs :
            delta hours between start times of windows
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        # start and end time for dataset
        format_str = "%Y%m%d%H%M%S"
        dt_start = datetime.datetime.strptime(str(start), format_str)
        dt_end = datetime.datetime.strptime(str(end), format_str)
        # TODO: check that temporal support of dataset if within requested start/end datetime

        # get all channels that are needed
        # TODO: handle when source/target is not specified, ie query available channels in dataset
        channels = list(set( stream_info["source"]).union( set( stream_info["target"])))

        fdb = pyfdb.FDB()
        request = Request(
                request={
                    "date": np.arange(
                        np.datetime64("2020-01-01"),  # TODO: dt_start
                        np.datetime64("2020-01-03"),  # TODO: dt_end
                    ),
                    "time": ["00", "06", "12", "18"],
                    "class": "ea",
                    "domain": "g",
                    "expver": "0001",
                    "stream": "oper",
                    "type": "an",
                    "step": "0",
                    "levtype": "sfc",
                    "param": channels,
                },
                chunk_axis=ChunkAxisType.DateTime,
            )
        lat, lon = make_lat_long_sources(fdb, request[0])
        datetimes = make_dates_source( np.datetime64("2020-01-01"),
                                       np.datetime64("2020-01-03"),
                                       np.timedelta64( 6, "h") )

        mapping = FdbZarrMapping(
            FdbZarrGroup(
                children=[
                    FdbZarrArray(
                        name="data",
                        datasource=FdbSource(
                            request=[ request ]
                        ),
                    ),
                    FdbZarrArray(name="latitudes", datasource=lat),
                    FdbZarrArray(name="longitudes", datasource=lon),
                    FdbZarrArray(name="datetimes", datasource=datetimes),
                ]
            )
        )
        self.ds = zarr.open_group(mapping, mode="r")

        self.data = self.ds['data']
        self.lats = np.array( self.ds['latitudes'])
        self.lons = np.array( self.ds['longitudes'])
        self.datatimes = self.ds['datetimes']

        self.properties = {
            "stream_id": 0,
        }

        # TODO
        self.source_channels = stream_info["source"]
        self.target_channels = stream_info["target"]
        self.target_idx = [0, 1]
        self.source_idx = [0, 1]
        self.geoinfo_idx = []

        # TODO
        self.num_steps_per_window = 1

        # TODO: get mean and var 

    def __len__(self):
        """
        Length of dataset

        Parameters
        ----------
        None

        Returns
        -------
        length of dataset
        """
        if not self.ds:
            return 0

        return len(self.ds)

    def get_source(self, idx: int) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Get source data for idx

        Parameters
        ----------
        idx : int
            Index of temporal window

        Returns
        -------
        source data (coords, geoinfos, data, datetimes)
        """
        return self._get(idx, self.source_idx)

    def get_target(self, idx: int) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Get target data for idx

        Parameters
        ----------
        idx : int
            Index of temporal window

        Returns
        -------
        target data (coords, geoinfos, data, datetimes)
        """
        return self._get(idx, self.target_idx)

    def _get(
        self, idx: int, channels_idx: np.array
    ) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Get data for window

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

        if not self.ds:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        # extract number of time steps
        data = self.ds[idx : idx + self.num_steps_per_window]
        # extract channels
        data = (
            data[:, channels_idx]
        )

        # construct lat/lon coords
        latlon = np.concatenate(
            [
                np.expand_dims(self.latitudes, 0),
                np.expand_dims(self.longitudes, 0),
            ],
            0,
        ).transpose()
        latlon = np.repeat(latlon, self.num_steps_per_window, axis=0).reshape((-1, latlon.shape[1]))

        # empty geoinfos
        geoinfos = np.zeros((data.shape[0], 0), dtype=data.dtype)

        # date time matching #data points of data
        datetimes = np.repeat(
            np.expand_dims(self.datetimes[idx : idx + self.num_steps_per_window], 0),
            data.shape[0],
            axis=0,
        ).flatten()

        return (latlon, geoinfos, data, datetimes)

    def get_source_num_channels(self):
        """
        Get number of source channels

        Parameters
        ----------
        None

        Returns
        -------
        number of source channels
        """
        return len(self.source_idx)

    def get_target_num_channels(self):
        """
        Get number of target channels

        Parameters
        ----------
        None

        Returns
        -------
        number of target channels
        """
        return len(self.target_idx)

    def get_coords_size(self):
        """
        Get size of coords

        Parameters
        ----------
        None

        Returns
        -------
        size of coords
        """
        return 2

    def get_geoinfo_size(self):
        """
        Get size of geoinfos

        Parameters
        ----------
        None

        Returns
        -------
        size of geoinfos
        """
        return len(self.geoinfo_idx)

    def normalize_coords(self, coords):
        """
        Normalize coordinates

        Parameters
        ----------
        coords : torch.tensor
            coordinates to be normalized

        Returns
        -------
        Normalized coordinates
        """
        coords[..., 0] = np.sin(np.deg2rad(coords[..., 0]))
        coords[..., 1] = np.sin(0.5 * np.deg2rad(coords[..., 1]))

        return coords

    def normalize_geoinfos(self, geoinfos):
        """
        Normalize geoinfos

        Parameters
        ----------
        geoinfos : torch.tensor
            geoinfos to be normalized

        Returns
        -------
        Normalized geoinfo
        """

        assert geoinfos.shape[-1] == 0
        return geoinfos

    def normalize_source_channels(self, source):
        """
        Normalize source channels

        Parameters
        ----------
        data : torch.tensor
            data to be normalized

        Returns
        -------
        Normalized data
        """
        assert source.shape[-1] == len(self.source_idx)
        for i, ch in enumerate(self.source_idx):
            source[..., i] = (source[..., i] - self.mean[ch]) / self.stdev[ch]

        return source

    def normalize_target_channels(self, target):
        """
        Normalize target channels

        Parameters
        ----------
        data : torch.tensor
            data to be normalized

        Returns
        -------
        Normalized data
        """
        assert target.shape[-1] == len(self.target_idx)
        for i, ch in enumerate(self.target_idx):
            target[..., i] = (target[..., i] - self.mean[ch]) / self.stdev[ch]

        return target

    def denormalize_source_channels(self, source):
        """
        Denormalize source channels

        Parameters
        ----------
        data : torch.tensor
            data to be denormalized

        Returns
        -------
        Denormalized data
        """
        assert source.shape[-1] == len(self.source_idx)
        for i, ch in enumerate(self.source_idx):
            source[..., i] = (source[..., i] * self.stdev[ch]) + self.mean[ch]

        return source

    def denormalize_target_channels(self, data: torch.tensor):
        """
        Denormalize target channels

        Parameters
        ----------
        data : torch.tensor
            data to be denormalized (target or pred)

        Returns
        -------
        Denormalized data
        """
        assert data.shape[-1] == len(self.target_idx)
        for i, ch in enumerate(self.target_idx):
            data[..., i] = (data[..., i] * self.stdev[ch]) + self.mean[ch]

        return data

    def time_window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
        """
        Temporal window corresponding to index

        Parameters
        ----------
        idx : int
            index of temporal window

        Returns
        -------
            start and end of temporal window
        """
        if not self.ds:
            return (np.array([], dtype=np.datetime64), np.array([], dtype=np.datetime64))

        return (self.ds.dates[idx], self.ds.dates[idx])
