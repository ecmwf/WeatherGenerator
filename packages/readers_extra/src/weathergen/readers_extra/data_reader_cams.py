import json
import logging
from pathlib import Path
from typing import override

import numpy as np
import xarray as xr

from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

import os, time
from typing import Sequence


############################################################################

_logger = logging.getLogger(__name__)


class DataReaderCams(DataReaderTimestep):
    "Wrapper for CAMs data variables"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """
        Parameters
        ----------
        tw_handler : TimeWindowHandler
            Handles temporal slicing and mapping from time indices to datetime
        filename :
            filename (and path) of dataset
        stream_info : dict
            Stream metadata
        """

        # ======= Reading the Dataset ================
        # open groups
        ds_surface = xr.open_zarr(filename, group="surface", chunks={"time": 24})
        ds_profiles = xr.open_zarr(filename, group="profiles", chunks={"time": 24})

        # merge along variables
        self.ds = xr.merge([ds_surface, ds_profiles])
        
        # Column (variable) names and indices
        self.colnames = stream_info["variables"] # list(self.ds)
        self.cols_idx = np.array(list(np.arange(len(self.colnames))))

        # Load associated statistics file for normalization
        stats_filename = Path(filename).with_name(Path(filename).stem + "_stats.json")
        with open(stats_filename) as stats_file:
            self.stats = json.load(stats_file)

        # Variables included in the stats
        self.stats_vars = list(self.stats)

        # Load mean and standard deviation per variable
        self.mean = np.array([self.stats[var]["mean"] for var in self.stats_vars], dtype=np.float64)
        self.stdev = np.array([self.stats[var]["std"] for var in self.stats_vars], dtype=np.float64)

        # Extract coordinates and pressure level
        self.lat =  _clip_lat(self.ds["latitude"].values)
        self.lon =  _clip_lon(self.ds["longitude"].values)
        self.levels = stream_info["pressure_levels"]

        # Time range in the dataset
        self.time = self.ds["time"].values
        start_ds = np.datetime64(self.time[0])
        end_ds = np.datetime64(self.time[-1])
        self.temporal_frequency = self.time[1] - self.time[0]

        if start_ds > tw_handler.t_end or end_ds < tw_handler.t_start:
            # print("inside skipping stream")
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        # Initialize parent class with resolved time window
        super().__init__(
            tw_handler,
            stream_info,
            start_ds,
            end_ds,
            self.temporal_frequency,
        )

        # Compute absolute start/end indices in the dataset based on time window
        self.start_idx = (tw_handler.t_start - start_ds).astype("timedelta64[ns]").astype(int)
        self.end_idx = (tw_handler.t_end - start_ds).astype("timedelta64[ns]").astype(int) + 1

        # Number of time steps in selected range
        self.len = self.end_idx - self.start_idx + 1

        # Placeholder; currently unused
        self.step_hrs = 1

        # Stream metadata
        self.properties = {
            "stream_id": 0,
        }

        # === Normalization statistics ===

        # Ensure stats match dataset columns
        assert self.stats_vars == self.colnames, (
            f"Variables in normalization file {self.stats_vars} do not match "
            f"dataset columns {self.colnames}"
        )

        # === Channel selection ===
        source_channels = stream_info.get("source")
        target_channels = stream_info.get("target")

        self.source_channels, self.source_idx = self.select("source", source_channels)
        self.target_channels, self.target_idx = self.select("target", target_channels)


        # Ensure all selected channels have valid standard deviations
        selected_channel_indices = list(set(self.source_idx).union(set(self.target_idx)))
        non_positive_stds = np.where(self.stdev[selected_channel_indices] <= 0)[0]
        assert len(non_positive_stds) == 0, (
            f"Abort: Encountered non-positive standard deviations for selected columns "
            f"{[self.colnames[selected_channel_indices][i] for i in non_positive_stds]}."
        )

        # === Geo-info channels (currently unused) ===
        self.geoinfo_channels = []
        self.geoinfo_idx = []



    def select(self, ch_type: str, ch_list: list[str]) -> tuple[list[str], np.typing.NDArray]:
        """
        Select channels constrained by allowed pressure levels and optional excludes.
        ch_type: "source" or "target" (for *_exclude key in stream_info)
        """
        channels_exclude = self.stream_info.get(f"{ch_type}_exclude", [])

        new_colnames: list[str] = []
        ch_list_loop = ch_list if ch_list else self.colnames
        for ch in ch_list_loop:
            if ch not in channels_exclude:
                ch_parts = ch.split("_")
                # Only include channels that are either surface variables or valid pressure 
                # level variables
                if len(ch_parts) != 2 or ch_parts[1] in self.levels:
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

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for temporal window

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """
        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"
        t0 = t_idxs[0]
        t1 = t_idxs[-1] + 1  # end is exclusive
        T = t1 - t0

        nlat = len(self.lat)
        nlon = len(self.lon)
        # channels to read
        channels = np.array(self.colnames)[channels_idx].tolist()

        # --- read & shape data to match anemoi path: (T, C, G) -> (T, G, C) -> (T*G, C)
        data_per_channel = []
        try:
            for ch in channels:
                ch_parts = ch.split("_")
                # retrieving profile channels
                if len(ch_parts) == 2 and ch_parts[1] in self.levels :
                    ch_ = ch_parts[0]
                    level=int(ch_parts[1])
                    data_lazy = self.ds[ch_].sel(isobaricInhPa=level)[t0:t1, :, :].astype("float32")
                # retrieving surface channels
                else:
                    data_lazy = self.ds[ch][t0:t1, :, :].astype("float32")

                data = data_lazy.compute(scheduler='synchronous').values
                data_per_channel.append(data.reshape(T, nlat * nlon))  # (T, G) 

        except Exception as e:
            _logger.debug(f"Date not present in CAMS dataset: {str(e)}. Skipping.")
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # stack channels to (T, C, G)
        data_TCG = np.stack(data_per_channel, axis=1)  # (T, C, G)
        # move channels to last and flatten time: (T, G, C) -> (T*G, C)
        data = np.transpose(data_TCG, (0, 2, 1)).reshape(T * (nlat * nlon), len(channels)).astype(np.float32)

        # --- coords: build flattened [lat, lon] once, then repeat for each time
        lon2d, lat2d = np.meshgrid(np.asarray(self.lon), np.asarray(self.lat))  # shapes (nlat, nlon)
        G = lon2d.size
        latlon_flat = np.column_stack([lat2d.ravel(order="C"), lon2d.ravel(order="C")])  # (G, 2); LAT first, LON second
        coords = np.vstack([latlon_flat] * T)  # (T*G, 2)

        # --- datetimes: repeat each timestamp for all grid points
        datetimes = np.repeat(self.time[t0:t1], G)

        # --- empty geoinfos (match anemoi)
        geoinfos = np.zeros((data.shape[0], 0), dtype=np.float32)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)
        return rd