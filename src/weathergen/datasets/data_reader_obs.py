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

import numpy as np
import zarr

from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    ReaderData,
    TimeWindowHandler,
    check_reader_data,
)

_logger = logging.getLogger(__name__)


class DataReaderObs(DataReaderBase):
    def __init__(self, tw_handler: TimeWindowHandler, filename: Path, stream_info: dict) -> None:
        super().__init__(tw_handler, stream_info)

        self.filename = filename
        self.z = zarr.open(filename, mode="r")
        self.data = self.z["data"]
        self.dt = self.z["dates"]  # datetime only
        self.hrly_index = self.z["idx_197001010000_1"]
        self.colnames = self.data.attrs["colnames"]

        # self.selected_colnames = self.colnames
        # self.selected_cols_idx = np.arange(len(self.colnames))
        idx = 0
        for i, col in enumerate(reversed(self.colnames)):
            idx = i
            # if col[:9] == 'obsvalue_' :
            if not (col[:4] == "sin_" or col[:4] == "cos_"):
                break
        self.selected_colnames = self.colnames[: len(self.colnames) - idx]
        self.selected_cols_idx = np.arange(len(self.colnames))[: len(self.colnames) - idx]

        # Create index for samples
        self._setup_sample_index()
        # assert len(self.indices_start) == len(self.indices_end)

        self._load_properties()

        channels_idx = [i for i, col in enumerate(self.selected_colnames) if "obsvalue" in col]
        self.data_offset = channels_idx[0]

        self.source_idx = [i for i, col in enumerate(self.selected_colnames) if "obsvalue" in col]
        self.source_idx = (np.array(self.source_idx) - channels_idx[0]).tolist()
        self.source_channels = [self.selected_colnames[i] for i in self.source_idx]

        self.target_idx = [i for i, col in enumerate(self.selected_colnames) if "obsvalue" in col]
        self.target_idx = (np.array(self.target_idx) - channels_idx[0]).tolist()
        self.target_channels = [self.selected_colnames[i] for i in self.target_idx]

        for i, _ in enumerate(self.colnames):
            idx = i
            if self.colnames[i] == "lat" and self.colnames[i + 1] == "lon":
                break
        self.coords_idx = [i, i + 1]
        self.geoinfo_idx = list(range(i + 2, channels_idx[0]))
        self.geoinfo_channels = [self.selected_colnames[i] for i in self.geoinfo_idx]

        self.mean = np.array(self.properties["means"])[channels_idx]
        self.stdev = np.sqrt(np.array(self.properties["vars"])[channels_idx])
        self.mean_geoinfo = np.array(self.properties["means"])[self.geoinfo_idx]
        self.stdev_geoinfo = np.sqrt(np.array(self.properties["vars"])[self.geoinfo_idx])

        self.len = min(len(self.indices_start), len(self.indices_end))

    @override
    def length(self) -> int:
        return self.len

    def select(self, cols: list[str] | None, cols_exclude: list[str] | None) -> None:
        """
        Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.
        """

        if cols is not None:
            self.selected_colnames = [
                c
                for c in self.colnames
                if (
                    np.array([c_sel in c for c_sel in cols]).any()
                    and not np.array([c_nsel in c for c_sel in cols_exclude]).any()
                )
            ]
        else:
            self.selected_colnames = self.colnames

        self.selected_cols_idx = np.array([self.colnames.index(c) for c in self.selected_colnames])

    def first_sample_with_data(self) -> int:
        """
        Returns the position of the first sample which contains data.
        """
        return (
            int(np.nonzero(self.indices_end)[0][0])
            if self.indices_end[-1] != self.indices_end[0]
            else None
        )

    def last_sample_with_data(self) -> int:
        """
        Returns the position of the last sample which contains data.
        """
        if self.indices_end[-1] == self.indices_end[0]:
            last_sample = None
        else:
            last_sample = int(
                np.where(np.diff(np.append(self.indices_end, self.indices_end[-1])) > 0)[0][-1] + 1
            )

        return last_sample

    def _setup_sample_index(self) -> None:
        """
        Dataset is divided into samples;
           - each n_hours long
           - sample 0 starts at start (yyyymmddhhmm)
           - index array has one entry for each sample; contains the index of the first row
           containing data for that sample
        """

        # TODO: generalize this
        assert self.time_window_handler.t_window_len.item().total_seconds() % 3600 == 0, (
            "t_window_len has to be full hour (currently {self.time_window_handler.t_window_len})"
        )
        len_hrs = int(self.time_window_handler.t_window_len.item().total_seconds()) // 3600
        assert self.time_window_handler.t_window_step.item().total_seconds() % 3600 == 0, (
            "t_window_step has to be full hour (currently {self.time_window_handler.t_window_len})"
        )
        step_hrs = int(self.time_window_handler.t_window_step.item().total_seconds()) // 3600

        # TODO: move to ctor
        base_yyyymmddhhmm = 197001010000

        # assert start > base_yyyymmddhhmm, (
        #     f"Abort: ObsDataset sample start (yyyymmddhhmm) must be greater than {base_yyyymmddhhmm}\n"
        #     f"       Current value: {start}"
        # )

        # Derive new index based on hourly backbone index
        format_str = "%Y%m%d%H%M%S"
        base_dt = datetime.datetime.strptime(str(base_yyyymmddhhmm), format_str)
        self.start_dt = self.time_window_handler.t_start.item()
        self.end_dt = self.time_window_handler.t_end.item()

        # Calculate the number of hours between start of hourly base index and the requested sample index
        diff_in_hours_start = int((self.start_dt - base_dt).total_seconds() / 3600)
        diff_in_hours_end = int((self.end_dt - base_dt).total_seconds() / 3600)

        end_range_1 = min(diff_in_hours_end, self.hrly_index.shape[0] - 1)
        self.indices_start = self.hrly_index[diff_in_hours_start:end_range_1:step_hrs]

        end_range_2 = min(
            diff_in_hours_end + len_hrs, self.hrly_index.shape[0] - 1
        )  # handle beyond end of data range safely
        self.indices_end = (
            self.hrly_index[diff_in_hours_start + len_hrs : end_range_2 : step_hrs] - 1
        )
        # Handle situations where the requested dataset span goes beyond the hourly index stored in the zarr
        if diff_in_hours_end > (self.hrly_index.shape[0] - 1):
            if diff_in_hours_start > (self.hrly_index.shape[0] - 1):
                n = (diff_in_hours_end - diff_in_hours_start) // step_hrs
                self.indices_start = np.zeros(n, dtype=int)
                self.indices_end = np.zeros(n, dtype=int)
            else:
                self.indices_start = np.append(
                    self.indices_start,
                    np.ones(
                        (diff_in_hours_end - self.hrly_index.shape[0] - 1) // step_hrs, dtype=int
                    )
                    * self.indices_start[-1],
                )
                self.indices_end = np.append(
                    self.indices_end,
                    np.ones(
                        (diff_in_hours_end - self.hrly_index.shape[0] - 1) // step_hrs, dtype=int
                    )
                    * self.indices_end[-1],
                )

        # Prevent -1 in samples before the we have data
        self.indices_end = np.maximum(self.indices_end, 0)

        if self.indices_end.shape != self.indices_start.shape:
            self.indices_end = np.append(self.indices_end, self.indices_end[-1])

        # If end (yyyymmddhhmm) is not a multiple of len_hrs
        # truncate the last sample so that it doesn't go beyond the requested dataset end date
        self.indices_end = np.minimum(self.indices_end, self.hrly_index[end_range_1])

    def _load_properties(self) -> None:
        self.properties = {}

        self.properties["means"] = self.data.attrs["means"]
        self.properties["vars"] = self.data.attrs["vars"]
        # self.properties["data_idxs"] = self.data.attrs["data_idxs"]
        self.properties["obs_id"] = self.data.attrs["obs_id"]

    @override
    def _get(self, idx: int, channels_idx: list[int]) -> ReaderData:
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
        ReaderDatas (coords, geoinfos, data, datetimes)
        """

        start_row = self.indices_start[idx]
        end_row = self.indices_end[idx]

        coords = self.data.oindex[start_row:end_row, self.coords_idx]
        geoinfos = (
            self.data.oindex[start_row:end_row, self.geoinfo_idx]
            if len(self.geoinfo_idx) > 0
            else np.zeros((coords.shape[0], 0), np.float32)
        )

        channels_idx = np.array(channels_idx)
        data = self.data.oindex[start_row:end_row, self.data_offset + channels_idx]
        datetimes = self.dt[start_row:end_row][:, 0]

        rdata = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )

        dtr = self.time_window_handler.window(idx)
        check_reader_data(rdata, dtr)

        return rdata
