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
import zarr

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)


class DataReaderFesom(DataReaderTimestep):
    """
    A dataset class for handling temporal windows of FESOM model output data stored in Zarr format.

    Parameters
    ----------
    start : datetime | int
        Start time of the data period as datetime object or integer in "%Y%m%d%H%M" format
    end : datetime | int
        End time of the data period (inclusive) with same format as start
    len_hrs : int
        Length of temporal windows in days
    step_hrs : int
        (Currently unused) Intended step size between windows in hours
    filename : Path
        Path to Zarr dataset containing FESOM output
    stream_info : dict
        Dictionary with "source" and "target" keys specifying channel subsets to use
        (e.g., {"source": ["temp"], "target": ["salinity"]})

    Attributes
    ----------
    len_hrs : int
        Temporal window length in days
    mesh_size : int
        Number of nodes in the FESOM mesh
    source_channels : list[str]
        Names of selected source channels
    target_channels : list[str]
        Names of selected target channels
    mean : np.ndarray
        Per-channel means for normalization (includes coordinates)
    stdev : np.ndarray
        Per-channel standard deviations for normalization (includes coordinates)
    properties : dict
        Dataset metadata including 'stream_id' from Zarr attributes

    Notes
    -----
    - Automatically handles datetime conversion and alignment with dataset time axis
    - Returns empty data containers if requested period doesn't overlap with dataset
    - Implements coordinate normalization using sinusoidal projections
    - Provides channel-wise normalization/denormalization for source/target variables
    - Uses Zarr's orthogonal indexing for efficient data access
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        self.filename = filename
        self.ds = zarr.open(filename, mode="r")
        self.mesh_size = self.ds.data.attrs["nod2"]

        self.time = self.ds["dates"]

        # TODO: time conversion to datetime64 should happen here.
        start_ds = self.time[0][0]
        end_ds = self.time[-1][0]

        if start_ds > tw_handler.t_end or end_ds < tw_handler.t_start:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        # TODO: why the case to [D]?
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

        period = self.time[self.mesh_size][0] - self.time[0][0]

        super().__init__(
            tw_handler,
            stream_info,
            start_ds,
            end_ds,
            period,
        )

        self.colnames: list[str] = list(self.ds.data.attrs["colnames"])
        self.cols_idx = list(np.arange(len(self.colnames)))
        self.lat_index = list(self.colnames).index("lat")
        self.lon_index = list(self.colnames).index("lon")
        self.colnames.remove("lat")
        self.colnames.remove("lon")
        self.cols_idx.remove(self.lat_index)
        self.cols_idx.remove(self.lon_index)
        self.cols_idx = np.array(self.cols_idx)

        # Ignore step_hrs, idk how it supposed to work
        # TODO, TODO, TODO:
        self.step_hrs = 1

        self.data = self.ds["data"]

        self.properties = {
            "stream_id": self.ds.data.attrs["obs_id"],
        }

        self.mean = np.concatenate((np.array([0, 0]), np.array(self.ds.data.attrs["means"])))
        self.stdev = np.sqrt(
            np.concatenate((np.array([1, 1]), np.array(self.ds.data.attrs["vars"])))
        )

        source_channels = stream_info.get("source")
        if source_channels:
            self.source_channels, self.source_idx = self.select(source_channels)
        else:
            self.source_channels = self.colnames
            self.source_idx = self.cols_idx

        target_channels = stream_info.get("target")
        if target_channels:
            self.target_channels, self.target_idx = self.select(target_channels)
        else:
            self.target_channels = self.colnames
            self.target_idx = self.cols_idx

        self.geoinfo_channels = []
        self.geoinfo_idx = []

    def select(self, ch_filters: list[str]) -> None:
        """
        Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.
        """
        mask = [np.array([f in c for f in ch_filters]).any() for c in self.colnames]

        selected_cols_idx = self.cols_idx[np.where(mask)[0]]
        selected_colnames = [self.colnames[i] for i in np.where(mask)[0]]

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

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # TODO: handle sub-sampling
        start_row = t_idxs[0] * self.mesh_size
        end_row = (t_idxs[-1] + 1) * self.mesh_size
        data = self.data.oindex[start_row:end_row, channels_idx]

        lat = np.expand_dims(self.data.oindex[start_row:end_row, self.lat_index], 1)
        lon = np.expand_dims(self.data.oindex[start_row:end_row, self.lon_index], 1)

        coords = np.concatenate([lat, lon], 1)
        # empty geoinfos
        geoinfos = np.zeros((data.shape[0], 0), dtype=data.dtype)
        datetimes = np.squeeze(self.time[start_row:end_row])

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd
