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
from anemoi.datasets import open_dataset


class AnemoiDataset:
    "Wrapper for Anemoi dataset"

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
        assert len_hrs == step_hrs, "Currently only step_hrs=len_hrs is supported"

        # open  dataset to peak that it is compatible with requested parameters
        self.ds = open_dataset(filename)

        # check that start and end time are within the dataset time range

        ds_dt_start = self.ds.dates[0]
        ds_dt_end = self.ds.dates[-1]

        format_str = "%Y%m%d%H%M%S"
        dt_start = datetime.datetime.strptime(str(start), format_str)
        dt_end = datetime.datetime.strptime(str(end), format_str)

        # open dataset

        # caches lats and lons
        self.latitudes = self.ds.latitudes.astype(np.float32)
        self.longitudes = self.ds.longitudes.astype(np.float32)

        # find physical fields (i.e. filter out auxiliary information to facilitate prediction)
        self.fields_idx = np.sort(
            [
                self.ds.name_to_index[k]
                for i, (k, v) in enumerate(self.ds.typed_variables.items())
                if not v.is_computed_forcing and not v.is_constant_in_time
            ]
        )
        # TODO: use complement of self.fields_idx as geoinfo
        self.fields = [self.ds.variables[i] for i in self.fields_idx]
        self.colnames = ["lat", "lon"] + self.fields
        self.selected_colnames = self.colnames

        self.properties = {
            "obs_id": 0,
            "means": self.ds.statistics["mean"],
            "vars": np.square(self.ds.statistics["stdev"]),
        }

        # set dataset to None when no overlap with time range
        if dt_start >= ds_dt_end or dt_end <= ds_dt_start:
            self.ds = None
            return

        self.ds = open_dataset(self.ds, frequency=str(step_hrs) + "h", start=dt_start, end=dt_end)

    def __len__(self):
        "Length of dataset"

        if not self.ds:
            return 0

        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple:
        "Get (data,datetime) for given index"

        if not self.ds:
            return (np.array([], dtype=np.float32), np.array([], dtype=np.float32))

        # prepend lat and lon to data; squeeze out ensemble dimension (for the moment)
        data = np.concatenate(
            [
                np.expand_dims(self.latitudes, 0),
                np.expand_dims(self.longitudes, 0),
                self.ds[idx].squeeze(),
            ],
            0,
        ).transpose()

        # date time matching #data points of data
        datetimes = np.full(data.shape[0], self.ds.dates[idx])

        return (data, datetimes)

    def time_window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
        if not self.ds:
            return (np.array([], dtype=np.datetime64), np.array([], dtype=np.datetime64))

        return (self.ds.dates[idx], self.ds.dates[idx])
