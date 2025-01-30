# (C) Copyright 2024 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
import zarr
import code


class ObsDataset():

    def __init__(
        self,
        filename: str,
        start: int,
        end: int,
        len_hrs: int,
        step_hrs: int = None,
        normalize: bool = True,
        select: list[str] = None,
    ) -> None:

        self.normalize = normalize
        self.filename = filename
        self.z = zarr.open( filename, mode="r")
        self.data = self.z["data"]
        self.dt = self.z["dates"]  # datetime only
        self.hrly_index = self.z["idx_197001010000_1"]
        self.colnames = self.data.attrs["colnames"]
        self.len_hrs = len_hrs
        self.step_hrs = step_hrs if step_hrs else len_hrs

        # self.selected_colnames = self.colnames
        # self.selected_cols_idx = np.arange(len(self.colnames))
        for i, col in enumerate( reversed( self.colnames)) :
            # if col[:9] == 'obsvalue_' :
            if not (col[:4] == 'sin_' or col[:4] == 'cos_') :
                break
        self.selected_colnames = self.colnames[ : len(self.colnames)-i ]
        self.selected_cols_idx = np.arange(len(self.colnames))[  : len(self.colnames)-i ]
        
        # Create index for samples
        self._setup_sample_index(start, end, self.len_hrs, self.step_hrs)
        # assert len(self.indices_start) == len(self.indices_end)

        self._load_properties()

        if select:
            self.select(select)

    def __getitem__( self, idx: int) -> tuple :

        start_row = self.indices_start[idx]
        end_row = self.indices_end[idx]

        data = self.data.oindex[start_row:end_row, self.selected_cols_idx]
        datetimes = self.dt[start_row:end_row][:,0]

        return (data, datetimes)

    def __len__(self) -> int:

        return min( len(self.indices_start), len(self.indices_end))

    def select(self, cols_list: list[str]) -> None:
        """
        Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.
        """
        self.selected_colnames = cols_list
        self.selected_cols_idx = np.array(
            [self.colnames.index(item) for item in cols_list]
        )

    def time_window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
        """
        Returns a tuple of datetime objects describing the start and end times of the sample at position idx.
        """

        if idx < 0:
            idx = len(self) + idx

        time_start = self.start_dt + datetime.timedelta(
            hours=( int(idx * self.step_hrs)), seconds=1
        )
        time_end = min(
            self.start_dt
            + datetime.timedelta(hours=( int(idx * self.step_hrs + self.len_hrs))),
            self.end_dt,
        )

        return (np.datetime64(time_start), np.datetime64(time_end))

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
                np.where(
                    np.diff(np.append(self.indices_end, self.indices_end[-1])) > 0
                )[0][-1]
                + 1
            )

        return last_sample

    def _setup_sample_index(
        self, start: int, end: int, len_hrs: int, step_hrs: int
    ) -> None:
        """
        Dataset is divided into samples;
           - each n_hours long
           - sample 0 starts at start (yyyymmddhhmm)
           - index array has one entry for each sample; contains the index of the first row
           containing data for that sample
        """

        base_yyyymmddhhmm = 197001010000

        assert start > base_yyyymmddhhmm, (
            f"Abort: ObsDataset sample start (yyyymmddhhmm) must be greater than {base_yyyymmddhhmm}\n"
            f"       Current value: {start}"
        )

        # Derive new index based on hourly backbone index
        format_str = "%Y%m%d%H%M%S"
        base_dt = datetime.datetime.strptime(str(base_yyyymmddhhmm), format_str)
        self.start_dt = datetime.datetime.strptime(str(start), format_str)
        self.end_dt = datetime.datetime.strptime(str(end), format_str)

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
                        (diff_in_hours_end - self.hrly_index.shape[0] - 1) // step_hrs,
                        dtype=int
                    )
                    * self.indices_start[-1],
                )
                self.indices_end = np.append(
                    self.indices_end,
                    np.ones(
                        (diff_in_hours_end - self.hrly_index.shape[0] - 1) // step_hrs,
                        dtype=int
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

####################################################################################################
if __name__ == "__main__":

    zarrpath = config.zarrpath
    zarrpath = '/lus/h2resw01/fws4/lb/project/ai-ml/observations/zarr/v0.2'

    # # polar orbiting satellites
    # d1 = ObsDataset( zarrpath, '34001', 201301010000, 202112310000, 24)
    # d2 = ObsDataset( zarrpath, '34002', 201301010000, 202112310000, 24)
    # d3 = ObsDataset( zarrpath, '1009', 201301010000, 202112310000, 24)
    # d4 = ObsDataset( zarrpath, '11002', 201301010000, 202112310000, 24)
    # d5 = ObsDataset( zarrpath, '1001', 201301010000, 202112310000, 24)
    # d6 = ObsDataset( zarrpath, '1004', 201301010000, 202112310000, 24)
    # d7 = ObsDataset( zarrpath, '1007', 201301010000, 202112310000, 24)

    # # geostationary satellites
    # d1 = ObsDataset( zarrpath, '4023', 201301010000, 202112310000, 6,
    #                  )

    # conventional obs
    d1 = ObsDataset( zarrpath + '/16002.zarr', 201301010000, 202112310000, 24)
    d2 = ObsDataset( zarrpath + '/16045.zarr', 201301010000, 202112310000, 24)
    d3 = ObsDataset( zarrpath + '/bufr_ship_synop_ofb_ea_0001.zarr', 201301010000, 202112310000, 24)
    d4 = ObsDataset( zarrpath + '/bufr_land_synop_ofb_ea_0001.zarr', 201301010000, 202112310000, 24)

    d = d1
    code.interact( local=locals())

    sample = d[0]
    print(sample.shape)
