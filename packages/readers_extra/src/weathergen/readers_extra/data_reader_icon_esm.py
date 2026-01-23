# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
from pathlib import Path
from typing import override

import fsspec
import numpy as np
import xarray as xr
import zarr

from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)

frequencies = {
    "3hrPt": np.timedelta64(10800000000000, "ns"),
    "day": np.timedelta64(86400000000000, "ns"),
    "fx": np.timedelta64(0, "ns"),
    "mon": np.timedelta64(2548800000000000, "ns"),
    "monC": np.timedelta64(2505600000000000, "ns"),
    "yr": np.timedelta64(31536000000000000, "ns"),
}


class DataReaderIconEsm(DataReaderTimestep):
    "Wrapper for ICON data channels"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        # Open the kerchunk-generated reference JSON
        ref_path = Path(filename)
        if not ref_path.exists():
            raise FileNotFoundError(f"Kerchunk reference JSON not found: {ref_path}")

        # Load JSON references and initialize a virtual file system
        kerchunk_ref = json.loads(ref_path.read_text())
        fs = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper = fs.get_mapper("")

        # Ensure metadata is consolidated for zarr-style access
        zarr.consolidate_metadata(mapper)

        # Open the dataset using Xarray with Zarr engine
        self.ds = xr.open_dataset(mapper, engine="zarr", consolidated=True, chunks={"time": 1})

        # get pressure levels
        self.plev = stream_info["plev"]
        self.depth = stream_info["depth"]
        self.lev = stream_info["lev"]
        # self.levels = stream_info["pressure_levels"]

        # Column (variable) names and indices
        self.colnames, self.cols_idx = self.get_cols(stream_info["channels"])

        # Determine temporal frequency from dataset metadata
        frequency_attr = self.ds.attrs["frequency"]
        self.temporal_frequency = frequencies[frequency_attr]

        # Load associated statistics file for normalization
        stats_filename = Path(filename).with_name(Path(filename).stem + "_stats.json")
        with open(stats_filename) as stats_file:
            self.stats = json.load(stats_file)

        # channels included in the stats
        self.stats_vars = list(self.stats)

        # Load mean and standard deviation per variable
        self.mean = np.array([self.stats[var]["mean"] for var in self.stats_vars], dtype=np.float64)
        self.stdev = np.array([self.stats[var]["std"] for var in self.stats_vars], dtype=np.float64)

        # Set mesh size based on spatial grid definition
        self.mesh_size = len(self.ds["i"])

        # Time range in the dataset
        self.time = self.ds["time"].values
        start_ds = np.datetime64(self.time[0])
        end_ds = np.datetime64(self.time[-1])

        # Skip stream if it doesn't intersect with time window
        if start_ds > tw_handler.t_end or end_ds < tw_handler.t_start:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        # Compute temporal resolution if not already defined
        self.temporal_frequency = (
            self.time[1] - self.time[0]
            if self.temporal_frequency is None
            else self.temporal_frequency
        )

        # Initialize parent class with resolved time window
        super().__init__(
            tw_handler,
            stream_info,
            start_ds,
            end_ds,
            self.temporal_frequency,
        )

        # Compute absolute start/end indices in the dataset based on time window
        self.start_idx = (tw_handler.t_start - start_ds).astype("timedelta64[D]").astype(
            int
        ) * self.mesh_size
        self.end_idx = (
            (tw_handler.t_end - start_ds).astype("timedelta64[D]").astype(int) + 1
        ) * self.mesh_size - 1

        # Sanity check
        assert self.end_idx > self.start_idx, (
            f"Abort: Final index of {self.end_idx} is the same or smaller than "
            f"start index {self.start_idx}"
        )

        # Number of time steps in selected range
        self.len = int((self.end_idx - self.start_idx) // self.mesh_size)

        # === Coordinates ===

        # Convert to degrees if stored in radians
        coords_units = self.ds["latitude"].attrs["units"]
        if coords_units == "radian":
            self.lat = np.rad2deg(self.ds["latitude"][:].astype("f"))
            self.lon = np.rad2deg(self.ds["longitude"][:].astype("f"))
        else:
            self.lat = self.ds["latitude"][:].astype("f")
            self.lon = self.ds["longitude"][:].astype("f")

        # Extract coordinates and pressure level
        self.lat = _clip_lat(self.lat)
        self.lon = _clip_lon(self.lon)

        # Placeholder; currently unused
        self.step_hrs = 1

        # Stream metadata
        self.properties = {
            "stream_id": 0,
        }

        # === Normalization statistics ===

        # Ensure stats match dataset columns
        assert self.stats_vars == self.colnames, (
            f"In {stream_info['name']} stream, channels in normalization file {self.stats_vars} "
            f"do not match dataset columns {self.colnames}"
        )

        # === Channel selection ===
        self.source_channels, self.source_idx = self.select("source")
        self.target_channels, self.target_idx = self.select("target")

        # Ensure all selected channels have valid standard deviations
        selected_channel_indices = list(set(self.source_idx).union(set(self.target_idx)))
        non_positive_stds = np.where(self.stdev[selected_channel_indices] <= 0)[0]
        if len(non_positive_stds) != 0:
            bad_vars = [self.colnames[selected_channel_indices[i]] for i in non_positive_stds]
            raise ValueError(
                f"Abort: Encountered non-positive standard deviations for selected columns "
                f"{bad_vars}."
            )

        # === Geo-info channels (currently unused) ===
        self.geoinfo_channels = []
        self.geoinfo_idx = []

    def select(self, ch_type: str) -> tuple[list[str], np.typing.NDArray]:
        """
        Select channels constrained by allowed pressure levels and optional excludes.
        ch_type: "source" or "target" (for *_exclude key in stream_info)
        """
        channels_exclude = self.stream_info.get(f"{ch_type}_exclude", [])

        new_colnames: list[str] = []
        for ch in self.colnames:
            ch_parts = ch.split("_")
            if len(ch_parts) == 2:
                ch_base = ch_parts[0]
                ch_num = ch_parts[1]
                coords_list = list(self.ds[ch_base].coords)
                if ch_base not in channels_exclude:
                    if (
                        ("plev" in coords_list and ch_num in self.plev)
                        or ("depth" in coords_list and ch_num in self.depth)
                        or ("lev" in coords_list and ch_num in self.lev)
                    ):
                        new_colnames.append(ch)
                else:
                    continue
            else:
                if ch not in channels_exclude:
                    new_colnames.append(ch)

        mask = [c in new_colnames for c in self.colnames]
        selected_cols_idx = self.cols_idx[np.where(mask)]
        selected_colnames = [self.colnames[int(i)] for i in np.where(mask)[0]]

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

    def get_cols(self, channels: list[str]) -> tuple[list[str], list[int]]:
        """
        TBD
        """
        colnames = []
        for ch in channels:
            coords_list = list(self.ds[ch].coords)
            if "plev" in coords_list:
                plev_dim = self.ds[ch].plev.ndim
                if plev_dim == 2:
                    plev_all = self.ds[ch]["plev"][0, :].values
                    for plev_ in plev_all:
                        plev_str = f"{plev_:.0f}"
                        colnames.append(f"{ch}_{plev_str}")
                else:
                    colnames.append(f"{ch}")
            elif "depth" in coords_list:
                depth_dim = self.ds[ch].depth.ndim
                if depth_dim == 2:
                    depth_all = self.ds[ch]["depth"][0, :].values
                    for depth_ in depth_all:
                        depth_str = f"{depth_:.4f}"
                        colnames.append(f"{ch}_{depth_str}")
                else:
                    colnames.append(f"{ch}")
            elif "lev" in coords_list:
                lev_dim = self.ds[ch].lev.ndim
                if lev_dim == 2:
                    lev_all = self.ds[ch]["lev"][0, :].values
                    for lev_ in lev_all:
                        lev_str = f"{lev_:.1f}"
                        colnames.append(f"{ch}_{lev_str}")
                else:
                    colnames.append(f"{ch}")
            else:
                colnames.append(f"{ch}")
        cols_idx = np.array(list(np.arange(len(colnames))))

        return colnames, cols_idx

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for temporal window

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : list[int]
            Selection of channels

        Returns
        -------
        ReaderData
        """
        (t_idxs, dtr) = self._get_dataset_idxs(idx)
        # dtr is a time window object it has the attributes t_start_win and t_end_win

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # Select channels
        channels = np.array(self.colnames)[channels_idx]

        start_ts = dtr.start
        end_ts = dtr.end - np.timedelta64(1, "h")
        data_arr = []
        try:
            data_per_channel = []
            datetimes = []
            coords = []
            for ch in channels:
                ch_parts = ch.split("_")
                if len(ch_parts) == 2:
                    ch_base = ch_parts[0]
                    ch_num = ch_parts[1]
                    coords_list = list(self.ds[ch_base].coords)
                    if "plev" in coords_list and ch_parts[1] in self.plev:
                        plev_all = self.ds[ch_base]["plev"][0].values
                        da = self.ds[ch_base].assign_coords(plev=("plev", plev_all))
                        da = da.sel(plev=ch_num, time=slice(start_ts, end_ts))
                    elif "depth" in coords_list and ch_parts[1] in self.depth:
                        depth_all = self.ds[ch_base]["depth"][0].values
                        da = self.ds[ch_base].assign_coords(depth=("depth", depth_all))
                        da = da.sel(depth=ch_num, time=slice(start_ts, end_ts))
                    elif "lev" in coords_list and ch_parts[1] in self.lev:
                        lev_all = self.ds[ch_base]["lev"][0].values
                        da = self.ds[ch_base].assign_coords(lev=("lev", lev_all))
                        da = da.sel(lev=ch_num, time=slice(start_ts, end_ts))
                    else:
                        _logger.warning(
                            f"Channel {ch} with part {ch_parts[1]} not found in dataset. Skipping."
                        )
                        continue
                else:
                    da = self.ds[ch].sel(time=slice(start_ts, end_ts))
                data_arr = da.compute(scheduler="synchronous")

                if not data_per_channel:
                    # datetimes
                    datetimes = np.repeat(data_arr.time.values, self.mesh_size).reshape(-1, 1)
                    datetimes = np.squeeze(datetimes)

                    # coords
                    n_times = len(data_arr.time)
                    lat = np.tile(data_arr.latitude.values[:, np.newaxis], (n_times, 1))
                    lon = np.tile(data_arr.longitude.values[:, np.newaxis], (n_times, 1))

                    coords = np.concatenate([lat, lon], axis=1)

                # data
                data_per_channel.append(np.asarray(data_arr.data.reshape(-1, 1)))

            data = np.concatenate(data_per_channel, axis=1)
        except Exception as e:
            _logger.debug(f"Date not present in ICON dataset: {str(e)}. Skipping.")
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # Empty geoinfos
        geoinfos = np.zeros((data.shape[0], 0), dtype=np.float32)

        rd = ReaderData(
            coords=coords.astype(np.float32),
            geoinfos=geoinfos,
            data=data.astype(np.float32),
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)
        _logger.info("[DATA LOADED]", flush=True)
        return rd
