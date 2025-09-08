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


def find_climatology_indices(
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    clim_lats: np.ndarray,
    clim_lons: np.ndarray,
) -> np.ndarray:
    """
    Function to find climatology indices matching target coordinates with tolerance.

    This function performs 2D coordinate matching between target coordinates and
    climatology coordinates using approximate matching with 1e-10 tolerance.

    Parameters
    ----------
    target_lats : np.ndarray
        Target latitude coordinates (1D array)
    target_lons : np.ndarray
        Target longitude coordinates (1D array)
    clim_lats : np.ndarray
        Climatology latitude coordinates (1D array)
    clim_lons : np.ndarray
        Climatology longitude coordinates (1D array)

    Returns
    -------
    np.ndarray
        Array of climatology indices for each target coordinate.
        Shape: (len(target_lats),). Contains -1 for coordinates with no match.
    """

    # Convert to numpy arrays if needed
    target_lats = np.asarray(target_lats)
    target_lons = np.asarray(target_lons)
    clim_lats = np.asarray(clim_lats)
    clim_lons = np.asarray(clim_lons)

    # Hardcode conversion of clim_lons from 0-360 to -180-180
    clim_lons = clim_lons - 180

    # Create result array initialized with -1 (no match)
    result_indices = np.full(len(target_lats), -1, dtype=np.int32)

    # Approximate matching with 1e-10 tolerance
    tolerance = 1e-5
    for i, (target_lat, target_lon) in enumerate(
        zip(target_lats, target_lons, strict=True)
    ):
        # Find approximate matches using tolerance-based comparison
        lat_match = np.abs(clim_lats - target_lat) <= tolerance
        lon_match = np.abs(clim_lons - target_lon) <= tolerance
        coord_match = lat_match & lon_match

        match_indices = np.where(coord_match)[0]
        if len(match_indices) > 0:
            result_indices[i] = match_indices[0]  # Use first match

    return result_indices
