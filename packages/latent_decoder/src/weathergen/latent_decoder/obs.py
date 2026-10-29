"""Observations in the WeatherGenerator observation zarr format: `data` [row, column] with the
column names in its attributes, `dates` [row, 1], and an hourly row index `idx_197001010000_1`
whose bin h holds the reports in (h, h + 1]. Users with other data replace this module; the
rest only needs `window(start, end)`."""

import numpy as np
import zarr

SYNOP = [
    "obsvalue_t2m_0",
    "obsvalue_td2m_0",
    "obsvalue_u10m_0",
    "obsvalue_v10m_0",
    "obsvalue_pmsl_0",
    "obsvalue_ps_0",
]
GEOINFOS = [
    "stalt",
    "lsm",
    "cos_sza",
    "cos_local_time",
    "sin_local_time",
    "cos_julian_day",
    "sin_julian_day",
]
EPOCH = np.datetime64("1970-01-01T00", "h")


class ObsStore:
    def __init__(self, path, channels=SYNOP, geoinfos=GEOINFOS):
        self.path, self.channels, self.geoinfos = str(path), list(channels), list(geoinfos)
        self._z, self._cols = None, None

    def _store(self):
        # opened lazily, so that each data-loader worker opens its own
        if self._z is None:
            z = zarr.open_group(self.path, mode="r")
            names = list(z["data"].attrs["colnames"])
            self._cols = [names.index(c) for c in ["lat", "lon", *self.geoinfos, *self.channels]]
            self._z = z
        return self._z

    def window(self, start, end) -> dict:
        """Reports with start <= time < end: times, lat, lon, geo [n, g], values [n, c] (NaN where
        a report lacks a value). Reports without a position or any of the channels are dropped."""
        z = self._store()
        idx = z["idx_197001010000_1"]
        h0 = int((np.datetime64(start, "h") - EPOCH) / np.timedelta64(1, "h"))
        h1 = int(np.ceil((np.datetime64(end, "s") - EPOCH) / np.timedelta64(1, "h")))
        # one bin either side: bins are (h, h + 1], and the dates decide in the end
        a, b = int(idx[max(h0 - 1, 0)]), int(idx[min(h1 + 1, idx.shape[0] - 1)])
        times = z["dates"][a:b, 0]
        keep = (times >= np.datetime64(start, "ns")) & (times < np.datetime64(end, "ns"))
        data = z["data"][a:b][keep][:, self._cols]
        # some missing values are stored as the float32 maximum rather than NaN
        data[np.abs(data) > 1e30] = np.nan
        g = len(self.geoinfos)
        values = data[:, 2 + g :]
        has_any = np.isfinite(values).any(axis=1) & np.isfinite(data[:, :2]).all(axis=1)
        data, times = data[has_any], times[keep][has_any]
        return {
            "times": times,
            "lat": data[:, 0],
            "lon": data[:, 1],
            "geo": data[:, 2 : 2 + g],
            "values": data[:, 2 + g :],
        }
