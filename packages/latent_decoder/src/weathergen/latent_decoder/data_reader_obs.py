# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A small reader for WeatherGenerator observation zarr stores, configured by a stream entry as
in config/streams/ and returning the fields of weathergen.datasets.data_reader_base.ReaderData.
Users with other data replace this module; the rest only needs `get_target(dtr)` and the
normalisation."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import zarr

from weathergen.latent_decoder.latents import DTRange


@dataclass
class ReaderData:
    coords: npt.NDArray  # [n, 2] lat, lon
    geoinfos: npt.NDArray  # [n, g]
    data: npt.NDArray  # [n, c], NaN where a report lacks a value
    datetimes: npt.NDArray  # [n]


class DataReaderObs:
    """Reads `filenames[0]` of the stream: `data` [row, column] with the column names, means and
    vars in its attributes, `dates` [row, 1], and an hourly row index `idx_197001010000_1` whose
    bin h holds the reports in (h, h + 1]. Channels are the stream's `target` and
    `geoinfo_channels`, given by their full column names."""

    base_datetime = np.datetime64("1970-01-01T00", "h")

    def __init__(self, stream_info: dict):
        self.filename = str(stream_info["filenames"][0])
        self.target_channels = list(stream_info["target"])
        self.geoinfo_channels = list(stream_info["geoinfo_channels"])
        z = zarr.open_group(self.filename, mode="r")
        colnames = list(z["data"].attrs["colnames"])
        self.coords_idx = [colnames.index("lat"), colnames.index("lon")]
        self.geoinfo_idx = [colnames.index(c) for c in self.geoinfo_channels]
        self.target_idx = [colnames.index(c) for c in self.target_channels]
        # geoinfos are not normalised here: the stored stats of stalt are spoilt by missing values
        self.mean = np.array(z["data"].attrs["means"])[self.target_idx].astype(np.float32)
        self.stdev = np.sqrt(np.array(z["data"].attrs["vars"])[self.target_idx]).astype(np.float32)
        self._z = None  # opened in each data-loader worker

    def get_target(self, dtr: DTRange) -> ReaderData:
        """Reports with dtr.start <= time < dtr.end. Reports without a position or without any
        of the target channels are dropped."""
        if self._z is None:
            self._z = zarr.open_group(self.filename, mode="r")
        z = self._z
        idx = z["idx_197001010000_1"]
        h0 = int((np.datetime64(dtr.start, "h") - self.base_datetime) / np.timedelta64(1, "h"))
        h1 = int(np.ceil((dtr.end - self.base_datetime) / np.timedelta64(1, "h")))
        # one bin either side: bins are (h, h + 1], and the dates decide in the end
        a, b = int(idx[max(h0 - 1, 0)]), int(idx[min(h1 + 1, idx.shape[0] - 1)])
        datetimes = z["dates"][a:b, 0]
        in_window = (datetimes >= dtr.start) & (datetimes < dtr.end)
        rows, datetimes = z["data"][a:b][in_window], datetimes[in_window]
        # some missing values are stored as the float32 maximum rather than NaN
        rows[np.abs(rows) > 1e30] = np.nan
        coords, data = rows[:, self.coords_idx], rows[:, self.target_idx]
        keep = np.isfinite(coords).all(axis=1) & np.isfinite(data).any(axis=1)
        return ReaderData(
            coords[keep], rows[keep][:, self.geoinfo_idx], data[keep], datetimes[keep]
        )

    def normalize_target_channels(self, data):
        return (data - self.mean) / self.stdev

    def denormalize_target_channels(self, data):
        return data * self.stdev + self.mean
