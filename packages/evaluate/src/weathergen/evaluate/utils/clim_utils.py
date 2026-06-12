# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from tqdm import tqdm

from weathergen.evaluate.utils.derived_channels import scale_z_channels

_logger = logging.getLogger(__name__)


def match_climatology_time(target_datetime: pd.Timestamp, clim_data: xr.Dataset) -> int | None:
    """
    Find matching climatology time index for target datetime.

    Parameters
    ----------
    target_datetime : pd.Timestamp
        Target datetime to match
    clim_data : xr.Dataset
        Climatology dataset with time dimension

    Returns
    -------
    int or None
        Matching time index, or None if no match found
    """

    # Convert numpy datetime64 to pandas datetime if needed
    if isinstance(target_datetime, np.datetime64):
        target_datetime = pd.to_datetime(target_datetime)

    target_doy = target_datetime.dayofyear
    target_hour = target_datetime.hour

    # EFFICIENT TIME MATCHING using vectorized operations
    clim_times = pd.to_datetime(clim_data.time.values)
    clim_doys = clim_times.dayofyear
    clim_hours = clim_times.hour

    time_matches = (clim_doys == target_doy) & (clim_hours == target_hour)
    matching_indices = np.where(time_matches)[0]

    # To Do: leap years and other edge cases
    if len(matching_indices) == 0:
        _logger.warning(
            f"No matching climatology time found for {target_datetime} (DOY: {target_doy}, "
            f"Hour: {target_hour})"
            f"Please check that climatology data and stream input data filenames match."
        )
        return None
    else:
        # Use first match if multiple exist
        if len(matching_indices) > 1:
            _logger.debug(f"Found {len(matching_indices)} matching times, using first one")
        return matching_indices[0]


def build_climatology_indexer(clim_lats: np.typing.NDArray, clim_lons: np.typing.NDArray):
    """
    Build a fast KDTree indexer for climatology coordinates.
    Returns a function that maps (target_lats, target_lons) -> climatology indices.
    """
    # Normalize climatology longitudes once
    clim_lons = np.where(clim_lons >= 180, clim_lons - 360, clim_lons)

    # Build KDTree on climatology coordinates
    clim_coords = np.column_stack((clim_lats, clim_lons))
    tree = cKDTree(clim_coords)

    def indexer(
        target_lats: np.typing.NDArray, target_lons: np.typing.NDArray, tol: float = 1e-5
    ) -> np.typing.NDArray:
        target_coords = np.column_stack((target_lats, target_lons))
        dist, idx = tree.query(target_coords, distance_upper_bound=tol)

        # Mark unmatched points as -1
        idx[~np.isfinite(dist)] = -1
        return idx.astype(np.int32)

    return indexer


def align_clim_data(
    target_output: dict,
    clim_data: xr.Dataset,
) -> dict:
    """
    Align climatology data with target data structure.

    Supports two climatology formats:

    - Legacy format (no ``statistic`` dimension): the aligned DataArray for each
      forecast step has the same shape as the target.
    - New format (with ``statistic`` dimension): the aligned DataArray keeps the
      full ``statistic`` dimension (e.g. ``['mean', 'q20', ..., 'q80']``), so that
      each score function can select the statistics it needs.

    Returns
    -------
    dict
        Dictionary mapping forecast step -> aligned climatology DataArray.
    """
    has_statistic_dim = clim_data is not None and "statistic" in clim_data.dims
    all_stats: list[str] | None = list(clim_data.statistic.values) if has_statistic_dim else None

    # Create empty climatology arrays for each forecast step
    aligned_clim: dict = {}

    for fstep, target_da in target_output.items():
        if has_statistic_dim:
            all_stat_coords: dict = {"statistic": all_stats}
            for dim in target_da.dims:
                if dim in target_da.coords and target_da.coords[dim].dims == (dim,):
                    all_stat_coords[dim] = target_da.coords[dim].values
            aligned_clim[fstep] = xr.DataArray(
                np.full(
                    (len(all_stats),) + target_da.shape,
                    np.nan,
                    dtype=np.float32,
                ),
                dims=["statistic"] + list(target_da.dims),
                coords=all_stat_coords,
            )
        else:
            aligned_clim[fstep] = xr.DataArray(
                np.full_like(target_da.values, np.nan),
                coords=target_da.coords,
                dims=target_da.dims,
            )

    # Cache for previously computed indices
    cached_target_lats = None
    cached_target_lons = None
    cached_clim_indices = None

    if clim_data is None:
        return aligned_clim

    # Build KDTree indexer once
    clim_lats = clim_data.latitude.values
    clim_lons = clim_data.longitude.values
    clim_indexer = build_climatology_indexer(clim_lats, clim_lons)

    for fstep, target_data in target_output.items():
        samples = np.unique(target_data.sample.values)
        has_sample_dim = "sample" in target_data.dims

        for sample in tqdm(samples, f"Aligning climatology for forecast step {fstep}"):
            sel_key = "sample" if has_sample_dim else "ipoint"
            sel_val = sample if has_sample_dim else (target_data.sample.values == sample)
            sel_mask = {sel_key: sel_val}

            timestamp = np.unique(target_data.sel(sel_mask).valid_time.values)[0]
            matching_time_idx = match_climatology_time(timestamp, clim_data)

            if matching_time_idx is None:
                continue

            if has_statistic_dim:
                # Keep the full statistic dimension; score functions select what they need.
                prepared_clim_data = (
                    clim_data.data.isel(time=matching_time_idx)
                    .sel(channels=target_data.channel.values)
                    .transpose("statistic", "grid_points", "channels")
                )
            else:
                prepared_clim_data = (
                    clim_data.data.isel(time=matching_time_idx)
                    .sel(channels=target_data.channel.values)
                    .transpose("grid_points", "channels")  # dimensions specific to anemoi
                )

            target_lats = target_data.loc[sel_mask].lat.values
            target_lons = target_data.loc[sel_mask].lon.values
            if (
                cached_clim_indices is not None
                and np.array_equal(target_lats, cached_target_lats)
                and np.array_equal(target_lons, cached_target_lons)
            ):
                clim_indices = cached_clim_indices
            else:
                clim_indices = clim_indexer(target_lats, target_lons)
                unmatched_mask = clim_indices == -1
                if np.any(unmatched_mask):
                    n_unmatched = np.sum(unmatched_mask)
                    raise ValueError(
                        f"Found {n_unmatched} target coordinates with no matching climatology "
                        f"coordinates. This will cause incorrect ACC calculations. "
                        f"Check coordinate alignment between target and climatology data."
                    )
                cached_clim_indices = clim_indices
                cached_target_lats = target_lats
                cached_target_lons = target_lons

            clim_values = prepared_clim_data.isel(grid_points=clim_indices).values
            try:
                aligned_clim[fstep].loc[sel_mask] = clim_values
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Failed to align climatology data with target data. "
                    f"This error typically occurs when the number of points per sample varies "
                    f"between samples. "
                    f"ACC/RPS/RPSS are currently only supported for forecasting data with constant "
                    f"points per sample. "
                    f"Please ensure all samples have the same spatial coverage and grid points. "
                    f"Original error: {e}"
                ) from e

    return aligned_clim


def get_climatology(reader, da_tars, stream: str) -> dict | None:
    """
    Load climatology data if specified in the evaluation configuration.

    Parameters
    ----------
    reader : WeatherGenReader
        Reader object to access data and configurations
    da_tars : dict
        Dictionary of target data arrays keyed by forecast step
    stream : str
        Name of the data stream

    Returns
    -------
    dict | None
        Dictionary mapping forecast step -> aligned climatology DataArray, or ``None``
        if no climatology path is configured.  For the new climatology format the
        DataArrays carry a leading ``statistic`` dimension; for the legacy format they
        do not.  Score functions select the statistics they need.
    """
    clim_data_path = reader.get_climatology_filename(stream)

    if clim_data_path is not None:
        clim_data = xr.open_dataset(clim_data_path)
        _logger.info("Aligning climatological data with target structure...")
        aligned = align_clim_data(da_tars, clim_data)
        return {fstep: scale_z_channels(da, stream) for fstep, da in aligned.items()}

    return None


def needs_climatology(metrics_dict: dict) -> bool:
    """
    Check if any of the specified metrics require climatology data.

    Parameters
    ----------
    metrics : dict
        Dictionary mapping metric names to their parameters.

    Returns
    -------
    bool
        True if any metric requires climatology, False otherwise
    """
    metrics = [m for metrics in metrics_dict.values() for m in metrics.keys()]
    req_clim = ["acc", "rps", "rpss"]
    return any(m in req_clim for m in metrics)
