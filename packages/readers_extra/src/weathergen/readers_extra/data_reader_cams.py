import json
import logging
from pathlib import Path
from typing import override

import numpy as np
import torch
import xarray as xr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

type DType = np.float32  # The type for the data in the datasets.

epsilon = 1e-4
log_epsilon = np.log(epsilon)

# Coefficients for the transformation: c1*min(x,2.5) + c2*log_term
c1 = 0.5  # Weight for linear clipped term
c2 = 0.5  # Weight for logarithmic term


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
        ds_surface = xr.open_zarr(
            filename, group="surface", chunks={"time": 24}, decode_timedelta=False
        )
        ds_profiles = xr.open_zarr(
            filename, group="profiles", chunks={"time": 24}, decode_timedelta=False
        )

        # merge along variables
        self.ds = xr.merge([ds_surface, ds_profiles])

        self.stream_info = stream_info
        # Column (variable) names and indices
        self.colnames = stream_info["variables"]  # list(self.ds)
        self.cols_idx = np.array(list(np.arange(len(self.colnames))))

        # Load associated statistics file for normalization
        stats_filename = Path(filename).with_name(
            Path(filename).stem + "_clipped_log_norm_stats_new.json"
        )
        with open(stats_filename) as stats_file:
            self.stats = json.load(stats_file)

        # Variables included in the stats
        self.stats_vars = list(self.stats)

        # Load mean, standard deviation, and max per variable
        self.mean = np.array([self.stats[var]["mean"] for var in self.stats_vars], dtype=np.float64)
        self.stdev = np.array([self.stats[var]["std"] for var in self.stats_vars], dtype=np.float64)
        self.max = np.array([self.stats[var]["max"] for var in self.stats_vars], dtype=np.float64)

        # Extract coordinates and pressure level
        self.lat = _clip_lat(self.ds["latitude"].values)
        self.lon = _clip_lon(self.ds["longitude"].values)
        self.levels = stream_info["pressure_levels"]

        # Time range in the dataset
        self.time = self.ds["time"].values
        start_ds = np.datetime64(self.time[0])
        end_ds = np.datetime64(self.time[-1])
        self.temporal_frequency = self.time[1] - self.time[0]
        # native spacing in hours, allow stream_info to override
        default_step = int(self.temporal_frequency / np.timedelta64(1, "h"))
        self.step_hrs = stream_info.get("step_hrs", default_step)

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

        # Stream metadata
        self.properties = {
            "stream_id": 0,
            "time_window_len_hours": self.step_hrs,
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

        # Ensure all selected channels have valid max values
        selected_channel_indices = list(set(self.source_idx).union(set(self.target_idx)))
        non_positive_maxs = np.where(self.max[selected_channel_indices] <= 0)[0]
        assert len(non_positive_maxs) == 0, (
            f"Abort: Encountered non-positive max values for selected columns "
            f"{[self.colnames[selected_channel_indices][i] for i in non_positive_maxs]}."
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
                if ch in self.colnames:
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
        Extract data for a temporal window and specific channels from CAMS dataset.

        Parameters
        ----------
        idx : TIndex
            Temporal index or range specifying which timesteps to retrieve
        channels_idx : list[int]
            Indices of channels/variables to extract from the dataset

        Returns
        -------
        ReaderData
            Structured data containing coordinates, metadata, variable data, and timestamps
        """
        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        # Return empty data if dataset is unavailable, no valid time indices, or no channels
        if self.ds is None or self.len == 0 or len(t_idxs) == 0 or len(channels_idx) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"

        # ------------------------------------------------------------------
        # apply step‑hour filter before touching the zarr store; this keeps
        # the number of time slices that are actually loaded to a minimum.
        if self.step_hrs > 1:
            times_window = self.time[t_idxs]
            hours = (
                (times_window.astype("datetime64[h]") - times_window.astype("datetime64[D]"))
                / np.timedelta64(1, "h")
            ).astype(int)
            mask = (hours % self.step_hrs) == 0
            t_idxs = t_idxs[mask]

        if len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        t_n = len(t_idxs)
        # Grid dimensions
        nlat = len(self.lat)
        nlon = len(self.lon)

        # Map channel indices to variable names
        channels = np.array(self.colnames)[channels_idx].tolist()

        # Extract data for each channel, handling surface vs. profile variables differently
        data_per_channel = []
        try:
            for ch in channels:
                ch_parts = ch.split("_")

                # Profile variables: extract specific pressure level (e.g., "temperature_850")
                if len(ch_parts) == 2 and ch_parts[1] in self.levels:
                    variable_name = ch_parts[0]
                    pressure_level = int(ch_parts[1])
                    data_lazy = (
                        self.ds[variable_name]
                        .sel(isobaricInhPa=pressure_level)
                        .isel(time=t_idxs)
                        .astype("float32")
                    )
                # Surface variables: extract directly (e.g., "surface_pressure")
                else:
                    data_lazy = self.ds[ch].isel(time=t_idxs).astype("float32")

                # Compute and flatten spatial dimensions: (time, lat, lon) -> (time, grid_points)
                data = data_lazy.compute(scheduler="synchronous").values
                data_per_channel.append(data.reshape(t_n, nlat * nlon))

        except Exception as e:
            _logger.info(f"Date not present in CAMS dataset: {str(e)}. Skipping.")
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        if len(data_per_channel) == 0:
            _logger.info("No channel data loaded for CAMS dataset. Skipping.")
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # Reorganize data from per-channel list to unified array
        # Stack: list of (time, grid) -> (time, channels, grid)
        data_stacked = np.stack(data_per_channel, axis=1)

        # Transpose and flatten: (time, channels, grid) -> (time, grid, channels) ->
        # (time*grid, channels)
        # Final shape matches expected format: each row is a (lat, lon, time) sample with all
        # channel values
        data = (
            np.transpose(data_stacked, (0, 2, 1))
            .reshape(t_n * (nlat * nlon), len(channels))
            .astype(np.float32)
        )

        # Create coordinate array: repeat lat/lon grid for each timestep
        lon2d, lat2d = np.meshgrid(np.asarray(self.lon), np.asarray(self.lat))
        total_grid = lon2d.size  # Total grid points

        # Flatten spatial coordinates and tile for all timesteps
        latlon_flat = np.column_stack(
            [lat2d.ravel(order="C"), lon2d.ravel(order="C")]
        )  # (grid_points, 2)
        coords = np.vstack([latlon_flat] * t_n)  # (time*grid_points, 2)

        # Create datetime array: repeat each timestamp for all spatial grid points
        datetimes = np.repeat(self.time[t_idxs], total_grid)

        # Empty geo-information array (placeholder for compatibility)
        geoinfos = np.zeros((data.shape[0], 0), dtype=np.float32)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)
        return rd

    @override
    def normalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        """
        Normalize source channels using two-step process:
        Step 1: Normalize by scale (max): x_scaled = x / scale_v
        Step 2: Apply transformation: c1 * min(x_scaled, 2.5) + \
            c2 * (log(max(x_scaled, 10^-4)) - log(10^-4)) / (-log(10^-4))

        Parameters
        ----------
        source :
            data to be normalized

        Returns
        -------
        Normalized data
        """

        if source.shape[-1] != len(self.source_idx):
            raise ValueError(
                f"incorrect number of source channels: expected {len(self.source_idx)}, "
                f"got {source.shape[-1]}"
            )

        for i, ch_idx in enumerate(self.source_idx):
            x = source[..., i]
            scale_v = self.max[ch_idx]

            # Step 1: Normalize by scale
            x_scaled = x / scale_v

            # Step 2: Apply transformation (Equation B9)
            if torch.is_tensor(x_scaled):
                linear_term = c1 * torch.clamp(x_scaled, max=2.5)
                clipped_data = torch.clamp(x_scaled, min=epsilon)
                log_term = c2 * (torch.log(clipped_data) - log_epsilon) / (-log_epsilon)
            else:
                linear_term = c1 * np.minimum(x_scaled, 2.5)
                clipped_data = np.maximum(x_scaled, epsilon)
                log_term = c2 * (np.log(clipped_data) - log_epsilon) / (-log_epsilon)
            normalized = linear_term + log_term
            source[..., i] = normalized

        return source

    @override
    def normalize_target_channels(self, target: NDArray[DType]) -> NDArray[DType]:
        """
        Normalize target channels using two-step process:
        Step 1: Normalize by scale (max): x_scaled = x / scale_v
        Step 2: Apply transformation: c1 * min(x_scaled, 2.5) + \
            c2 * (log(max(x_scaled, 10^-4)) - log(10^-4)) / (-log(10^-4))

        Parameters
        ----------
        target :
            data to be normalized

        Returns
        -------
        Normalized data
        """

        if target.shape[-1] != len(self.target_idx):
            raise ValueError(
                f"incorrect number of target channels: expected {len(self.target_idx)}, "
                f"got {target.shape[-1]}"
            )

        for i, ch_idx in enumerate(self.target_idx):
            x = target[..., i]
            scale_v = self.max[ch_idx]

            # Step 1: Normalize by scale
            x_scaled = x / scale_v

            # Step 2: Apply transformation (Equation B9)
            if torch.is_tensor(x_scaled):
                linear_term = c1 * torch.clamp(x_scaled, max=2.5)
                clipped_data = torch.clamp(x_scaled, min=epsilon)
                log_term = c2 * (torch.log(clipped_data) - log_epsilon) / (-log_epsilon)
            else:
                linear_term = c1 * np.minimum(x_scaled, 2.5)
                clipped_data = np.maximum(x_scaled, epsilon)
                log_term = c2 * (np.log(clipped_data) - log_epsilon) / (-log_epsilon)
            normalized = linear_term + log_term
            target[..., i] = normalized

        return target

    @override
    def denormalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        """
        Denormalize source channels by reversing the two-step process:
        Step 1: Reverse transformation to get x_scaled
        Step 2: Unscale: x = x_scaled * scale_v
        Uses iterative Newton-Raphson method to approximate the inverse transformation.

        Parameters
        ----------
        source :
            data to be denormalized

        Returns
        -------
        Denormalized data
        """

        if source.shape[-1] != len(self.source_idx):
            raise ValueError(
                f"incorrect number of source channels: expected {len(self.source_idx)}, "
                f"got {source.shape[-1]}"
            )

        for i, ch_idx in enumerate(self.source_idx):
            y = source[..., i]
            scale_v = self.max[ch_idx]

            # Step 1: Reverse transformation to get x_scaled
            # Use iterative method to find x_scaled such that: y = c1*min(x_scaled,2.5) + \
            #    c2*(log(max(x_scaled,ε))-log(ε))/(-log(ε))
            if torch.is_tensor(y):
                # Initial guess: assume log term dominates
                x_scaled = torch.exp(y / c2 * (-log_epsilon) + log_epsilon)
                # Iterative refinement (5 iterations should suffice)
                for _ in range(5):
                    linear_term = c1 * torch.clamp(x_scaled, max=2.5)
                    clipped = torch.clamp(x_scaled, min=epsilon)
                    log_term = c2 * (torch.log(clipped) - log_epsilon) / (-log_epsilon)
                    y_pred = linear_term + log_term
                    error = y - y_pred
                    x_scaled = x_scaled + 0.1 * error * x_scaled  # Scaled update
                    x_scaled = torch.clamp(x_scaled, min=epsilon)  # Keep positive

                # Step 2: Unscale
                denormalized = x_scaled * scale_v
                source[..., i] = denormalized

            else:
                # Initial guess: assume log term dominates
                x_scaled = np.exp(y / c2 * (-log_epsilon) + log_epsilon)
                # Iterative refinement
                for _ in range(5):
                    linear_term = c1 * np.minimum(x_scaled, 2.5)
                    clipped = np.maximum(x_scaled, epsilon)
                    log_term = c2 * (np.log(clipped) - log_epsilon) / (-log_epsilon)
                    y_pred = linear_term + log_term
                    error = y - y_pred
                    x_scaled = x_scaled + 0.1 * error * x_scaled
                    x_scaled = np.maximum(x_scaled, epsilon)

                # Step 2: Unscale
                denormalized = x_scaled * scale_v
                source[..., i] = denormalized

        return source

    @override
    def denormalize_target_channels(self, data: NDArray[DType]) -> NDArray[DType]:
        """
        Denormalize target channels by reversing the two-step process:
        Step 1: Reverse transformation to get x_scaled
        Step 2: Unscale: x = x_scaled * scale_v
        Uses iterative Newton-Raphson method to approximate the inverse transformation.

        Parameters
        ----------
        data :
            data to be denormalized

        Returns
        -------
        Denormalized data
        """
        if data.shape[-1] != len(self.target_idx):
            raise ValueError(
                f"incorrect number of target channels: expected {len(self.target_idx)}, "
                f"got {data.shape[-1]}"
            )

        for i, ch_idx in enumerate(self.target_idx):
            y = data[..., i]
            scale_v = self.max[ch_idx]

            # Step 1: Reverse transformation to get x_scaled
            # Use iterative method to find x_scaled such that: y = c1*min(x_scaled,2.5) + \
            #    c2*(log(max(x_scaled,ε))-log(ε))/(-log(ε))
            if torch.is_tensor(y):
                # Initial guess: assume log term dominates
                x_scaled = torch.exp(y / c2 * (-log_epsilon) + log_epsilon)
                # Iterative refinement (5 iterations should suffice)
                for _ in range(5):
                    linear_term = c1 * torch.clamp(x_scaled, max=2.5)
                    clipped = torch.clamp(x_scaled, min=epsilon)
                    log_term = c2 * (torch.log(clipped) - log_epsilon) / (-log_epsilon)
                    y_pred = linear_term + log_term
                    error = y - y_pred
                    x_scaled = x_scaled + 0.1 * error * x_scaled  # Scaled update
                    x_scaled = torch.clamp(x_scaled, min=epsilon)  # Keep positive

                # Step 2: Unscale
                denormalized = x_scaled * scale_v
                data[..., i] = denormalized

            else:
                # Initial guess: assume log term dominates
                x_scaled = np.exp(y / c2 * (-log_epsilon) + log_epsilon)
                # Iterative refinement
                for _ in range(5):
                    linear_term = c1 * np.minimum(x_scaled, 2.5)
                    clipped = np.maximum(x_scaled, epsilon)
                    log_term = c2 * (np.log(clipped) - log_epsilon) / (-log_epsilon)
                    y_pred = linear_term + log_term
                    error = y - y_pred
                    x_scaled = x_scaled + 0.1 * error * x_scaled
                    x_scaled = np.maximum(x_scaled, epsilon)

                # Step 2: Unscale
                denormalized = x_scaled * scale_v
                data[..., i] = denormalized

        return data
