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
from tqdm import tqdm
from scipy.spatial import cKDTree
_logger = logging.getLogger(__name__)

def match_climatology_time(
    target_datetime: pd.Timestamp, clim_data: xr.Dataset
) -> int | None:
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
            f"No matching climatology time found for {target_datetime} (DOY: {target_doy}, Hour: {target_hour})"
        )
        return None
    else:
        # Use first match if multiple exist
        if len(matching_indices) > 1:
            _logger.debug(
                f"Found {len(matching_indices)} matching times, using first one"
            )
        return matching_indices[0]

def build_climatology_indexer(clim_lats: np.ndarray, clim_lons: np.ndarray):
    """
    Build a fast KDTree indexer for climatology coordinates.
    Returns a function that maps (target_lats, target_lons) -> climatology indices.
    """
    # Normalize climatology longitudes once
    clim_lons = np.where(clim_lons >= 180, clim_lons - 360, clim_lons)

    # Build KDTree on climatology coordinates
    clim_coords = np.column_stack((clim_lats, clim_lons))
    tree = cKDTree(clim_coords)

    def indexer(target_lats: np.ndarray, target_lons: np.ndarray, tol: float = 1e-5) -> np.ndarray:
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
    Align climatology data with target data structure (fast KDTree version).
    """
    aligned_clim_data = {
        fstep: xr.full_like(data, np.nan) for fstep, data in target_output.items()
    }

    if clim_data is None:
        return aligned_clim_data

    # Build KDTree indexer once
    clim_lats = clim_data.latitude.values
    clim_lons = clim_data.longitude.values
    clim_indexer = build_climatology_indexer(clim_lats, clim_lons)

    for fstep, target_data in target_output.items():
        # Match climatology time
        matching_time_idx = match_climatology_time(
            target_data.valid_time.values[0], clim_data
        )
        prepared_clim = (
            clim_data.data.isel(time=matching_time_idx)
            .sel(channels=target_data.channel.values)
            .transpose("grid_points", "channels")
        )

        # use KDTree to find indices
        target_lats = target_data.lat.values
        target_lons = target_data.lon.values
        clim_indices = clim_indexer(target_lats, target_lons)

        # Fail if unmatched points exist
        if np.any(clim_indices == -1):
            n_unmatched = np.sum(clim_indices == -1)
            raise ValueError(
                f"Found {n_unmatched} target coordinates with no matching climatology coordinates. "
                "This will cause incorrect ACC calculations. "
                "Check coordinate alignment between target and climatology data."
            )

        clim_values = prepared_clim.isel(grid_points=clim_indices).values

        try:
            aligned_clim_data[fstep].data[:] = clim_values
        except (ValueError, IndexError) as e:
            raise ValueError(
                "Failed to align climatology data with target data for ACC calculation. "
                "This usually occurs when the number of points per sample varies. "
                "ACC metric only supports forecasting data with constant points per sample. "
                f"Original error: {e}"
            ) from e

    return aligned_clim_data

def get_climatology(reader, da_tars, stream: str) -> xr.Dataset | None:
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
    xr.Dataset or None
        Climatology dataset if available, otherwise None
    """
    # Get climatology data path from configuration
    clim_data_path = reader.get_climatology_filename(stream)
  
    aligned_clim_data = None   

    if clim_data_path:
        clim_data = xr.open_dataset(clim_data_path)
        _logger.info("Aligning climatological data with target structure...")
        aligned_clim_data = align_clim_data(da_tars, clim_data)

    return aligned_clim_data