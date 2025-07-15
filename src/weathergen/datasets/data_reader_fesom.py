# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import glob
import logging
from pathlib import Path
from typing import override, Optional

import dask
import dask.array as da
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

    This class is optimized for use with multiple dataloader workers by implementing
    lazy initialization of file handles and efficient, batched data reads.
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        # Store configuration but DO NOT open files here
        self.filenames = sorted(glob.glob(str(filename)))
        self._tw_handler = tw_handler
        self._stream_info = stream_info

        if len(self.filenames) == 0:
            self.init_empty()
            self._initialized = True
            return

        # Initialize data-dependent attributes to None. They will be set by _lazy_init.
        self.time: Optional[da.Array] = None
        self.data: Optional[da.Array] = None
        self.len = 0  # Default length is 0 until initialized
        self.source_channels = []
        self.source_idx = []
        self.target_channels = []
        self.target_idx = []
        self.geoinfo_channels = []
        self.geoinfo_idx = []
        self.properties = {}

        if len(self.filenames) == 0:
            name = stream_info["name"]
            _logger.warning(
                f"{name} couldn't find any files matching {filename}. Stream is skipped."
            )
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            # No need to return, the length is 0, so it will be skipped.

        # We call super() last, after we know if the stream is valid or not.
        # We also pass dummy values, as the real ones will be set in _lazy_init.
        super().__init__(self._tw_handler, self._stream_info)

        # This flag ensures initialization happens only once per worker
        self._initialized = False

    def _lazy_init(self) -> None:
        """
        Initializes the dataset object. This method is called once per worker process
        to ensure file handles are not shared across forked processes.
        """
        if self._initialized:
            return

        # Each worker now opens its own file handles safely
        groups: list[zarr.Group] = [zarr.open_group(name, mode="r") for name in self.filenames]
        times: list[zarr.Array] = [group["dates"] for group in groups]
        data: list[zarr.Array] = [group["data"] for group in groups]

        self.time = da.concatenate(times, axis=0)
        self.data = da.concatenate(data, axis=0)

        # Use the first group for metadata
        first_group = groups[0]
        if "nod2" in first_group.data.attrs:
            self.mesh_size = first_group.data.attrs["nod2"]
        else:
            self.mesh_size = first_group.data.attrs["n_points"]

        # Metadata reading is cheap, but let's do it with the rest of the init
        start_ds = self.time[0][0].compute()
        end_ds = self.time[-1][0].compute()

        if start_ds > self._tw_handler.t_end or end_ds < self._tw_handler.t_start:
            name = self._stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            self.init_empty()
            self._initialized = True
            return

        period = (self.time[self.mesh_size][0] - self.time[0][0]).compute()

        # Re-initialize the parent class with correct time info
        super().__init__(self._tw_handler, self._stream_info, start_ds, end_ds, period)

        if self._tw_handler.t_start > start_ds:
            self.start_idx = ((self._tw_handler.t_start - start_ds) // period + 1) * self.mesh_size
        else:
            self.start_idx = 0

        self.end_idx = ((self._tw_handler.t_end - start_ds) // period + 1) * self.mesh_size

        if self.end_idx > len(self.time):
            self.end_idx = len(self.time)

        self.len = (self.end_idx - self.start_idx) // self.mesh_size

        # Check for a valid length after calculations
        if self.len <= 0:
            self.init_empty()
            self._initialized = True
            return

        self.colnames: list[str] = list(first_group.data.attrs["colnames"])
        self.cols_idx = list(np.arange(len(self.colnames)))
        self.lat_index = self.colnames.index("lat")
        self.lon_index = self.colnames.index("lon")

        # Modify a copy, not the original list while iterating
        temp_colnames = list(self.colnames)
        temp_colnames.remove("lat")
        temp_colnames.remove("lon")
        self.colnames = temp_colnames

        self.cols_idx.remove(self.lat_index)
        self.cols_idx.remove(self.lon_index)
        self.cols_idx = np.array(self.cols_idx)

        self.step_hrs = 1  # TODO

        self.properties = {"stream_id": first_group.data.attrs["obs_id"]}

        self.mean = np.concatenate((np.array([0, 0]), np.array(first_group.data.attrs["means"])))
        self.stdev = np.sqrt(
            np.concatenate((np.array([1, 1]), np.array(first_group.data.attrs["std"])))
        )
        self.stdev[self.stdev == 0.0] = 1.0

        source_channels = self._stream_info.get("source")
        self.source_channels, self.source_idx = (
            self.select(source_channels) if source_channels else (self.colnames, self.cols_idx)
        )

        target_channels = self._stream_info.get("target")
        self.target_channels, self.target_idx = (
            self.select(target_channels) if target_channels else (self.colnames, self.cols_idx)
        )

        self.geoinfo_channels = []
        self.geoinfo_idx = []

        self._initialized = True

    def select(self, ch_filters: list[str]) -> tuple[list[str], np.ndarray]:
        mask = [any(f in c for f in ch_filters) for c in self.colnames]
        selected_cols_idx = self.cols_idx[np.where(mask)[0]]
        selected_colnames = [self.colnames[i] for i in np.where(mask)[0]]
        return selected_colnames, selected_cols_idx

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0

    @override
    def length(self) -> int:
        # Make sure initialization has happened before returning length
        self._lazy_init()
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        self._lazy_init()

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        start_row = t_idxs[0] * self.mesh_size
        end_row = (t_idxs[-1] + 1) * self.mesh_size

        _logger.info(f"Started loading data from : {start_row} {end_row}")

        # Note: we read all columns from start_row to end_row once,
        # then select the ones we need. This is more efficient for Zarr.
        full_data_slice = self.data[start_row:end_row]
        time_slice = self.time[start_row:end_row]

        # Define the specific slices we need from the larger block
        data_lazy = full_data_slice[:, channels_idx]
        lat_lazy = full_data_slice[:, self.lat_index]
        lon_lazy = full_data_slice[:, self.lon_index]
        datetimes_lazy = time_slice

        # Dask optimizes this to a single (or few) efficient read operation(s).
        data, lat, lon, datetimes = dask.compute(
            data_lazy, lat_lazy, lon_lazy, datetimes_lazy, scheduler="single-threaded"
        )

        coords = np.stack([lat, lon], axis=1)
        geoinfos = np.zeros((data.shape[0], 0), dtype=data.dtype)
        datetimes = np.squeeze(datetimes)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd
