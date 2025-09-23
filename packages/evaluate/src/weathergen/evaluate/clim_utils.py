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
    clim_lons[clim_lons >= 180] = clim_lons[clim_lons >= 180] - 360

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


def align_clim_data(
    target_output: dict,
    clim_data: xr.Dataset,
) -> dict:
    """
    Align climatology data with target data structure.
    """
    # create empty climatology data for each forecast step
    aligned_clim_data = {}
    for fstep, _ in target_output.items():
        aligned_clim_data[fstep] = xr.DataArray(
            np.full_like(
                target_output[fstep].values,
                np.nan,  # Create array with same shape filled with NaNs
            ),
            coords=target_output[fstep].coords,  # Use the same coordinates as target
            dims=target_output[fstep].dims,  # Use the same dimensions as target
        )

    if clim_data is None:
        return aligned_clim_data
    else:
        for fstep, target_data in target_output.items():
            samples = np.unique(target_data.sample.values)
            # Prepare climatology data for each sample
            matching_time_idx = match_climatology_time(
                target_data.valid_time.values[0], clim_data
            )
            prepared_clim_data = (
                clim_data.data.isel(
                    time=matching_time_idx,
                )
                .sel(
                    channels=target_data.channel.values,
                )
                .transpose("grid_points", "channels")
            )

            for sample in tqdm(samples):
                if len(samples) > 1:
                    sample_mask = target_data.sample.values == sample
                    target_lats = target_data.loc[{"ipoint": sample_mask}].lat.values
                    target_lons = target_data.loc[{"ipoint": sample_mask}].lon.values
                else:
                    target_lats = target_data.lat.values
                    target_lons = target_data.lon.values
                clim_lats = prepared_clim_data.latitude.values
                clim_lons = prepared_clim_data.longitude.values
                clim_indices = find_climatology_indices(
                    target_lats, target_lons, clim_lats, clim_lons
                )

                # Check for unmatched coordinates
                unmatched_mask = clim_indices == -1
                if np.any(unmatched_mask):
                    n_unmatched = np.sum(unmatched_mask)
                    raise ValueError(
                        f"Found {n_unmatched} target coordinates with no matching climatology coordinates. "
                        f"This will cause incorrect ACC calculations. "
                        f"Check coordinate alignment between target and climatology data."
                    )

                clim_values = prepared_clim_data.isel(grid_points=clim_indices).values
                try:
                    if len(samples) > 1:
                        aligned_clim_data[fstep].loc[{"ipoint": sample_mask}] = (
                            clim_values
                        )
                    else:
                        aligned_clim_data[fstep] = clim_values
                except (ValueError, IndexError) as e:
                    raise ValueError(
                        f"Failed to align climatology data with target data for ACC calculation. "
                        f"This error typically occurs when the number of points per sample varies between samples. "
                        f"ACC metric is currently only supported for forecasting data with constant points per sample. "
                        f"Please ensure all samples have the same spatial coverage and grid points. "
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
    stream_dict = reader.eval_cfg["streams"][stream]
    inference_cfg = reader.get_inference_config()
    # This searches for the climatology filename in the stream configuration
    clim_fn = next(
        (
            item.get("climatology_filename")
            for item in inference_cfg["streams"]
            if item.get("name") == stream
        ),
        None,
    )

    if stream_dict.get("needs_climatology", False):
        # Check if climatology path is specified in the eval configuration
        if "climatology_path" in stream_dict:
            clim_data_path = stream_dict["climatology_path"]
            clim_data = xr.open_dataset(clim_data_path)
            _logger.info("Aligning climatological data with target structure...")
            aligned_clim_data = align_clim_data(da_tars, clim_data)
        # Otherwise check if a general aux data path and clim fn is specified in the inference configuration
        elif "data_path_aux" in inference_cfg and clim_fn is not None:
            clim_data_path = inference_cfg["data_path_aux"]
            clim_data_path = clim_data_path + clim_fn
            clim_data = xr.open_dataset(clim_data_path)
            _logger.info("Aligning climatological data with target structure...")
            aligned_clim_data = align_clim_data(da_tars, clim_data)
        else:
            _logger.warning(
                f"No climatology path specified for stream {stream}. Setting climatology to NaN. "
                "Add 'climatology_path' to evaluation config to keep metrics like ACC."
            )
            aligned_clim_data = None
    else:
        aligned_clim_data = None

    return aligned_clim_data