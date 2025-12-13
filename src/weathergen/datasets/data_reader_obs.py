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
        base_dt_raw = stream_info.get("base_datetime", "1970-01-01T00:00:00")
        format_str = "%Y-%m-%dT%H:%M:%S"

        # Use python datetime for strict parsing/validation and string formatting
        dt_obj = datetime.datetime.strptime(str(base_dt_raw), format_str)

        # Store as numpy datetime64 for stable arithmetic
        self.base_datetime = np.datetime64(dt_obj)

        # To read idx convert to a string, format e.g.: 197001010000
        base_date_str = dt_obj.strftime("%Y%m%d%H%M")
        self.hrly_index = self.z[f"idx_{base_date_str}_1"]
        self.colnames = self.data.attrs["colnames"]

        data_colnames = [col for col in self.colnames if "obsvalue" in col]
        data_idx = [i for i, col in enumerate(self.colnames) if "obsvalue" in col]

        # determine source / target channels and corresponding idx using include and exclude lists

        s_chs = stream_info.get("source")
        s_chs_exclude = stream_info.get("source_exclude", [])

        t_chs = stream_info.get("target")
        t_chs_exclude = stream_info.get("target_exclude", [])

        # source_n_empty = len(s_chs) > 0 if s_chs is not None else True
        # assert source_n_empty, "source is empty; at least one channels must be present."
        # target_n_empty = len(t_chs) > 0 if t_chs is not None else True
        # assert target_n_empty, "target is empty; at least one channels must be present."

        self.source_channels = self.select_channels(data_colnames, s_chs, s_chs_exclude)
        self.source_idx = [self.colnames.index(c) for c in self.source_channels]
        self.source_idx = np.array(self.source_idx, dtype=np.int64)

        self.target_channels = self.select_channels(data_colnames, t_chs, t_chs_exclude)
        self.target_idx = [self.colnames.index(c) for c in self.target_channels]
        self.target_idx = np.array(self.target_idx, dtype=np.int64)

        # determine idx for coords and geoinfos
        self.coords_idx = [self.colnames.index("lat"), self.colnames.index("lon")]
        self.geoinfo_idx = list(range(self.coords_idx[-1] + 1, data_idx[0]))
        self.geoinfo_channels = [self.colnames[i] for i in self.geoinfo_idx]

        # load additional properties (mean, var)
        self._load_properties()
        self.mean = np.array(self.properties["means"])  # [data_idx]
        self.stdev = np.sqrt(np.array(self.properties["vars"]))  # [data_idx])
        self.mean_geoinfo = np.array(self.properties["means"])[self.geoinfo_idx]
        self.stdev_geoinfo = np.sqrt(np.array(self.properties["vars"])[self.geoinfo_idx])

        # Create index for samples
        self._setup_sample_index()

        self.len = min(len(self.indices_start), len(self.indices_end))

    @override
    def length(self) -> int:
        return self.len

    def select_channels(
        self, colnames: list[str], cols_select: list[str] | None, cols_exclude: list[str] | None
    ) -> list[str]:
        """
        Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.
        """
        selected_colnames = [
            c
            for c in colnames
            if (
                np.array([c_sel in c for c_sel in cols_select]).any()
                if cols_select is not None
                else True and not np.array([c_nsel in c for c_nsel in cols_exclude]).any()
            )
        ]

        return selected_colnames

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
        t_len = self.time_window_handler.t_window_len
        len_seconds = t_len / np.timedelta64(1, "s")
        assert len_seconds % 3600 == 0, (
            f"t_window_len has to be full hour (now {self.time_window_handler.t_window_len})"
        )
        len_hrs = int(len_seconds) // 3600

        t_step = self.time_window_handler.t_window_step
        step_seconds = t_step / np.timedelta64(1, "s")
        assert step_seconds % 3600 == 0, (
            f"t_window_step has to be full hour (now {self.time_window_handler.t_window_step})"
        )
        step_hrs = int(step_seconds) // 3600

        self.start_dt = self.time_window_handler.t_start
        self.end_dt = self.time_window_handler.t_end

        # Calculate the number of hours between start of hourly base index
        diff_in_hours_start = int((self.start_dt - self.base_datetime) / np.timedelta64(1, "h"))
        diff_in_hours_end = int((self.end_dt - self.base_datetime) / np.timedelta64(1, "h"))

        end_range_1 = min(diff_in_hours_end, self.hrly_index.shape[0] - 1)
        self.indices_start = self.hrly_index[diff_in_hours_start:end_range_1:step_hrs]

        end_range_2 = min(
            diff_in_hours_end + len_hrs, self.hrly_index.shape[0] - 1
        )  # handle beyond end of data range safely
        self.indices_end = (
            self.hrly_index[diff_in_hours_start + len_hrs : end_range_2 : step_hrs] - 1
        )
        ## Handle situations where the requested dataset span
        #  goes beyond the hourly index stored in the zarr
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
                        # add (len_hrs + 1) since above we also have diff_in_hours_start + len_hrs
                        (diff_in_hours_end - self.hrly_index.shape[0] + (len_hrs + 1)) // step_hrs,
                        dtype=int,
                    )
                    * self.indices_end[-1],
                )

        # Prevent -1 in samples before we have data
        self.indices_end = np.maximum(self.indices_end, 0)

        # If end (yyyymmddhhmm) is not a multiple of len_hrs
        # truncate the last sample so that it doesn't go beyond the requested dataset end date
        self.indices_end = np.minimum(self.indices_end, self.hrly_index[end_range_1])

    def _load_properties(self) -> None:
        self.properties = {}

        self.properties["means"] = self.data.attrs["means"]
        self.properties["vars"] = self.data.attrs["vars"]

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

        if len(channels_idx) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        start_row = self.indices_start[idx - 1]
        end_row = self.indices_end[idx]

        coords = self.data.oindex[start_row:end_row, self.coords_idx]
        geoinfos = (
            self.data.oindex[start_row:end_row, self.geoinfo_idx]
            if len(self.geoinfo_idx) > 0
            else np.zeros((coords.shape[0], 0), np.float32)
        )

        data = self.data.oindex[start_row:end_row, channels_idx]
        datetimes = self.dt[start_row:end_row][:, 0]

        # indices_start, indices_end above work with [t_start, t_end] and violate
        # our convention [t_start, t_end) where endpoint is excluded
        # compute mask to enforce it
        t_win = self.time_window_handler.window(idx)
        t_mask = np.logical_and(datetimes >= t_win.start, datetimes < t_win.end)

        rdata = ReaderData(
            coords=coords[t_mask],
            geoinfos=geoinfos[t_mask],
            data=data[t_mask],
            datetimes=datetimes[t_mask],
        )

        dtr = self.time_window_handler.window(idx)
        check_reader_data(rdata, dtr)

        return rdata
