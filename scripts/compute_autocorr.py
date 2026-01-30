#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Compute spatial and temporal autocorrelation scales for atmospheric variables.

This script computes per-variable autocorrelation scales from Zarr/Anemoi datasets
and outputs a YAML configuration file for use in variable-specific masking.

Usage:
    python compute_autocorr.py --dataset /path/to/data.zarr --output config/autocorr.yml
    python compute_autocorr.py --dataset /path/to/data.zarr --channels z_500 t_850 q_700

The output YAML can be included in stream configs under `channel_masking.autocorr`.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import anemoi.datasets as anemoi_datasets
import numpy as np
import yaml
from numpy.typing import NDArray

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Earth radius in km
EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points on Earth.

    Parameters
    ----------
    lat1, lon1 : float
        Coordinates of first point in degrees.
    lat2, lon2 : float
        Coordinates of second point in degrees.

    Returns
    -------
    float
        Distance in kilometers.
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c


def compute_spatial_autocorr(
    data: NDArray[np.float32],
    lats: NDArray[np.float32],
    lons: NDArray[np.float32],
    max_lag_km: float = 3000.0,
    n_bins: int = 30,
    n_sample_pairs: int = 50000,
    seed: int = 42,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimate spatial correlation length scale via distance-binned correlation.

    Uses random sampling of point pairs to estimate correlation as a function
    of distance, then fits an exponential decay to find the e-folding scale.

    Parameters
    ----------
    data : NDArray
        Data array of shape [n_times, n_points] or [n_points].
    lats : NDArray
        Latitudes of shape [n_points].
    lons : NDArray
        Longitudes of shape [n_points].
    max_lag_km : float
        Maximum distance lag to consider in km.
    n_bins : int
        Number of distance bins.
    n_sample_pairs : int
        Number of random point pairs to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    l_corr_km : float
        Estimated spatial correlation length in km.
    bin_centers : NDArray
        Center of each distance bin in km.
    correlations : NDArray
        Correlation coefficient for each bin.
    """
    rng = np.random.default_rng(seed)

    # Handle time dimension
    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_times, n_points = data.shape

    # Normalize data (per timestep)
    data_mean = np.nanmean(data, axis=1, keepdims=True)
    data_std = np.nanstd(data, axis=1, keepdims=True)
    data_std = np.where(data_std < 1e-10, 1.0, data_std)
    data_norm = (data - data_mean) / data_std

    # Sample random point pairs
    idx1 = rng.integers(0, n_points, size=n_sample_pairs)
    idx2 = rng.integers(0, n_points, size=n_sample_pairs)

    # Compute distances
    distances = np.array(
        [haversine_distance_km(lats[i], lons[i], lats[j], lons[j]) for i, j in zip(idx1, idx2)]
    )

    # Compute correlations (averaged over time)
    correlations_raw = np.nanmean(data_norm[:, idx1] * data_norm[:, idx2], axis=0)

    # Bin by distance
    bin_edges = np.linspace(0, max_lag_km, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_correlations = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_correlations[i] = np.nanmean(correlations_raw[mask])
            bin_counts[i] = mask.sum()

    # Fit exponential decay: corr(d) = exp(-d / L_corr)
    # Use only bins with sufficient samples and positive correlation
    valid = (bin_counts > 10) & (bin_correlations > 0.05)
    if valid.sum() < 3:
        logger.warning("Insufficient valid bins for fitting. Using fallback estimate.")
        # Fallback: find distance where correlation drops below 1/e
        above_threshold = bin_correlations > (1 / np.e)
        if above_threshold.any():
            l_corr_km = bin_centers[above_threshold][-1]
        else:
            l_corr_km = bin_centers[0] if len(bin_centers) > 0 else 500.0
    else:
        # Linear regression on log-correlation vs distance
        log_corr = np.log(bin_correlations[valid])
        d_valid = bin_centers[valid]
        # corr = exp(-d/L) => log(corr) = -d/L => slope = -1/L
        slope, _ = np.polyfit(d_valid, log_corr, 1)
        if slope < 0:
            l_corr_km = -1.0 / slope
        else:
            # Positive slope indicates issue, use fallback
            l_corr_km = 500.0

    return float(l_corr_km), bin_centers, bin_correlations


def compute_temporal_autocorr(
    data: NDArray[np.float32],
    period_hours: float,
    max_lag_hours: float = 168.0,
    n_sample_points: int = 1000,
    seed: int = 42,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimate temporal correlation scale via lag-based correlation.

    Parameters
    ----------
    data : NDArray
        Data array of shape [n_times, n_points].
    period_hours : float
        Time step between samples in hours.
    max_lag_hours : float
        Maximum lag to consider in hours.
    n_sample_points : int
        Number of spatial points to sample for averaging.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    t_corr_hours : float
        Estimated temporal correlation scale in hours.
    lag_hours : NDArray
        Lag values in hours.
    correlations : NDArray
        Correlation coefficient for each lag.
    """
    rng = np.random.default_rng(seed)

    n_times, n_points = data.shape
    max_lag_steps = min(int(max_lag_hours / period_hours), n_times - 1)

    # Sample spatial points
    sample_idx = rng.choice(n_points, size=min(n_sample_points, n_points), replace=False)
    data_sample = data[:, sample_idx]

    # Normalize
    data_mean = np.nanmean(data_sample, axis=0, keepdims=True)
    data_std = np.nanstd(data_sample, axis=0, keepdims=True)
    data_std = np.where(data_std < 1e-10, 1.0, data_std)
    data_norm = (data_sample - data_mean) / data_std

    # Compute lag correlations
    lag_hours = np.arange(0, max_lag_steps + 1) * period_hours
    correlations = np.zeros(max_lag_steps + 1)

    for lag in range(max_lag_steps + 1):
        if lag == 0:
            correlations[lag] = 1.0
        else:
            corr = np.nanmean(data_norm[:-lag] * data_norm[lag:])
            correlations[lag] = corr

    # Find e-folding time (where correlation drops below 1/e)
    threshold = 1.0 / np.e
    below_threshold = correlations < threshold
    if below_threshold.any():
        first_below = np.argmax(below_threshold)
        # Interpolate for more precise estimate
        if first_below > 0:
            t_corr_hours = lag_hours[first_below - 1] + (lag_hours[first_below] - lag_hours[first_below - 1]) * (
                correlations[first_below - 1] - threshold
            ) / (correlations[first_below - 1] - correlations[first_below] + 1e-10)
        else:
            t_corr_hours = lag_hours[0]
    else:
        # Correlation never drops below threshold within window
        t_corr_hours = max_lag_hours

    return float(t_corr_hours), lag_hours, correlations


def analyze_variable(
    ds,
    var_idx: int,
    var_name: str,
    n_time_samples: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Analyze a single variable to compute autocorrelation scales.

    Parameters
    ----------
    ds : anemoi Dataset
        The opened dataset.
    var_idx : int
        Index of the variable in the dataset.
    var_name : str
        Name of the variable.
    n_time_samples : int
        Number of time steps to sample for analysis.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with autocorrelation results.
    """
    rng = np.random.default_rng(seed)

    n_times = len(ds.dates)
    # ds.frequency can be int (seconds) or timedelta
    freq = ds.frequency
    if hasattr(freq, 'total_seconds'):
        period_hours = freq.total_seconds() / 3600
    else:
        period_hours = freq / 3600  # Assume seconds

    # Sample time indices (spread across the dataset)
    if n_times <= n_time_samples:
        t_indices = np.arange(n_times)
    else:
        t_indices = np.sort(rng.choice(n_times, size=n_time_samples, replace=False))

    logger.info(f"  Loading {len(t_indices)} time samples for {var_name}...")

    # Load data for this variable
    # Shape: ds[time] -> [n_vars, n_ens, n_points]
    data_list = []
    for t_idx in t_indices:
        try:
            sample = ds[int(t_idx)]  # [n_vars, n_ens, n_points]
            data_list.append(sample[var_idx, 0, :])  # [n_points]
        except Exception as e:
            logger.warning(f"  Failed to load time index {t_idx}: {e}")
            continue

    if len(data_list) < 10:
        logger.warning(f"  Insufficient data for {var_name}. Using defaults.")
        return {
            "space_km": 500,
            "time_h": 24,
            "warning": "insufficient_data",
        }

    data = np.stack(data_list, axis=0)  # [n_times, n_points]

    # Get coordinates
    lats = ds.latitudes
    lons = ds.longitudes

    # Compute spatial autocorrelation
    logger.info(f"  Computing spatial autocorrelation for {var_name}...")
    l_corr_km, _, _ = compute_spatial_autocorr(data, lats, lons, seed=seed)

    # Compute temporal autocorrelation (need contiguous time for this)
    logger.info(f"  Computing temporal autocorrelation for {var_name}...")
    # For temporal, use a contiguous window
    window_size = min(200, n_times)
    start_idx = rng.integers(0, max(1, n_times - window_size))
    t_contiguous = np.arange(start_idx, min(start_idx + window_size, n_times))

    data_temporal = []
    for t_idx in t_contiguous:
        try:
            sample = ds[int(t_idx)]
            data_temporal.append(sample[var_idx, 0, :])  # [n_points]
        except Exception:
            break

    if len(data_temporal) >= 20:
        data_temporal = np.stack(data_temporal, axis=0)
        t_corr_hours, _, _ = compute_temporal_autocorr(data_temporal, period_hours, seed=seed)
    else:
        logger.warning(f"  Insufficient contiguous data for temporal autocorr. Using estimate.")
        t_corr_hours = 24.0  # Default fallback

    # Round to reasonable precision
    l_corr_km = round(l_corr_km / 10) * 10  # Round to nearest 10 km
    t_corr_hours = round(t_corr_hours)  # Round to nearest hour

    return {
        "space_km": int(l_corr_km),
        "time_h": int(t_corr_hours),
    }


def correlation_length_to_hl_mask(
    l_corr_km: float,
    multiplier: float = 1.0,
    hl_min: int = 1,
    hl_max: int = 5,
) -> int:
    """
    Map spatial correlation length to HEALPix masking level.

    Finds the finest HEALPix level where the cell size is still larger
    than the correlation length scaled by the multiplier.

    Parameters
    ----------
    l_corr_km : float
        Spatial correlation length in km (e-folding distance).
    multiplier : float
        Scale factor applied to correlation length (default: 1.0).
    hl_min : int
        Minimum HEALPix level (prevents hemisphere-scale masks).
    hl_max : int
        Maximum HEALPix level (prevents sub-grid masks).

    Returns
    -------
    int
        HEALPix level for masking.
    """
    target_size = l_corr_km * multiplier

    def cell_size_km(h: int) -> float:
        n_cells = 12 * (4**h)
        cell_area_km2 = (4 * np.pi * EARTH_RADIUS_KM**2) / n_cells
        return np.sqrt(cell_area_km2)

    # Find finest level (highest number) where cell_size > target_size
    for hl in range(hl_max, hl_min - 1, -1):
        if cell_size_km(hl) > target_size:
            return hl

    return hl_min


def main():
    parser = argparse.ArgumentParser(
        description="Compute autocorrelation scales for variable-specific masking."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to Zarr/Anemoi dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="autocorr_config.yml",
        help="Output YAML file path.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="*",
        default=None,
        help="Specific channels to analyze. If not provided, analyzes all.",
    )
    parser.add_argument(
        "--n-time-samples",
        type=int,
        default=100,
        help="Number of time samples for spatial autocorrelation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--correlation-multiplier",
        type=float,
        default=1.5,
        help="Multiplier for L_corr to L_mask mapping.",
    )

    args = parser.parse_args()

    logger.info(f"Opening dataset: {args.dataset}")
    ds = anemoi_datasets.open_dataset(args.dataset)

    # Handle frequency which can be int (seconds) or timedelta
    freq = ds.frequency
    if hasattr(freq, 'total_seconds'):
        freq_hours = freq.total_seconds() / 3600
    else:
        freq_hours = freq / 3600

    logger.info(f"Dataset info:")
    logger.info(f"  Variables: {len(ds.variables)}")
    logger.info(f"  Time steps: {len(ds.dates)}")
    logger.info(f"  Frequency: {freq_hours}h")
    logger.info(f"  Grid points: {len(ds.latitudes)}")

    # Determine which variables to analyze
    if args.channels:
        var_names = args.channels
        var_indices = []
        for name in var_names:
            if name in ds.name_to_index:
                var_indices.append(ds.name_to_index[name])
            else:
                logger.warning(f"Variable '{name}' not found in dataset. Skipping.")
    else:
        # Analyze all non-computed, non-constant variables
        var_names = []
        var_indices = []
        for name, info in ds.typed_variables.items():
            if not info.is_computed_forcing and not info.is_constant_in_time:
                var_names.append(name)
                var_indices.append(ds.name_to_index[name])

    logger.info(f"Analyzing {len(var_names)} variables...")

    # Analyze each variable
    results = {}
    for var_name, var_idx in zip(var_names, var_indices):
        logger.info(f"Analyzing: {var_name}")
        try:
            result = analyze_variable(
                ds,
                var_idx,
                var_name,
                n_time_samples=args.n_time_samples,
                seed=args.seed,
            )
            # Add recommended hl_mask
            result["hl_mask_recommended"] = correlation_length_to_hl_mask(
                result["space_km"],
                multiplier=args.correlation_multiplier,
            )
            results[var_name] = result
            logger.info(
                f"  Result: L_corr={result['space_km']} km, "
                f"T_corr={result['time_h']} h, "
                f"hl_mask={result['hl_mask_recommended']}"
            )
        except Exception as e:
            logger.error(f"  Failed to analyze {var_name}: {e}")
            results[var_name] = {
                "space_km": 500,
                "time_h": 24,
                "error": str(e),
            }

    # Build output config
    output_config = {
        "# Autocorrelation configuration for variable-specific masking": None,
        "# Generated by compute_autocorr.py": None,
        "channel_masking": {
            "enabled": True,
            "autocorr": results,
            "mapping": {
                "correlation_multiplier": args.correlation_multiplier,
                "hl_mask_min": 1,
                "hl_mask_max": 5,
                "time_block_min": 1,
                "time_block_max": 8,
            },
            "temporal": {
                "strategy": "tube",
                "obs_strategy": "tube",
            },
            "default": {
                "space_km": 500,
                "time_h": 24,
                "hl_mask": 3,
                "keep_rate": 0.6,
                "time_block": 1,
            },
        },
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Custom YAML representer to handle None values (comments)
    def represent_none(dumper, _):
        return dumper.represent_scalar("tag:yaml.org,2002:null", "")

    yaml.add_representer(type(None), represent_none)

    with open(output_path, "w") as f:
        # Write header comment
        f.write("# Autocorrelation configuration for variable-specific masking\n")
        f.write(f"# Generated from: {args.dataset}\n")
        f.write(f"# Correlation multiplier: {args.correlation_multiplier}\n\n")
        yaml.dump(
            {"channel_masking": output_config["channel_masking"]},
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    logger.info(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
