#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compute spatial autocorrelation per variable and suggest per-stream masking configs.

This script analyses a dataset to determine the spatial correlation length of
each variable, maps that to an appropriate HEALPix masking level (``hl_mask``),
and groups variables by similar correlation scale.  The output is a summary
table plus YAML snippets ready to be used as ``masking_override`` blocks in
stream config files.

Example usage:

    uv run python packages/science/compute_spatial_autocorrelation.py \\
        --dataset /path/to/data.zarr \\
        --type anemoi \\ or obs, or less supported options below
        --channels z_500 z_850 t_500 t_850 q_700 tp \\ defaults to all vars
        --n-time-samples 100 \\
        --n-sample-pairs 100000 \\
        --correlation-multiplier 0.5 \\
        
        then see further optional args below for controlling the output.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0


@dataclass
class DatasetInfo:
    """Minimal container for data loaded from a dataset."""

    lats: NDArray[np.float64] | None  # [n_points] radians
    lons: NDArray[np.float64] | None  # [n_points] radians
    data: dict[str, NDArray]  # var_name -> [n_times, n_points]
    period_hours: float | None = None
    lats_ragged: list[NDArray[np.float64]] | None = None  # per-time [n_points_t]
    lons_ragged: list[NDArray[np.float64]] | None = None  # per-time [n_points_t]
    data_ragged: dict[str, list[NDArray]] | None = None  # var_name -> list[n_points_t]


@dataclass
class VarResult:
    """Autocorrelation analysis result for a single variable."""

    name: str
    l_corr_km: float
    hl_mask: int
    bin_centers_km: NDArray = field(repr=False)
    bin_correlations: NDArray = field(repr=False)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_anemoi(
    path: str | Path,
    n_time_samples: int,
    channels: list[str] | None,
    seed: int,
) -> DatasetInfo:
    """Load data from an anemoi-format zarr dataset."""
    import anemoi.datasets as anemoi_datasets

    ds = anemoi_datasets.open_dataset(path)
    rng = np.random.default_rng(seed)

    all_vars = list(ds.variables)
    if channels is None:
        channels = all_vars

    # Map channel names to indices
    var_indices = {}
    for ch in channels:
        if ch not in all_vars:
            logger.warning(f"Channel '{ch}' not found in dataset, skipping. Available: {all_vars}")
            continue
        var_indices[ch] = all_vars.index(ch)

    if not var_indices:
        raise ValueError(f"No valid channels found. Available: {all_vars}")

    n_times_total = len(ds)
    n_samples = min(n_time_samples, n_times_total)
    time_indices = rng.choice(n_times_total, size=n_samples, replace=False)
    time_indices.sort()

    lats = np.deg2rad(ds.latitudes)
    lons = np.deg2rad(ds.longitudes)

    data = {}
    for ch, idx in var_indices.items():
        values = []
        for ti in time_indices:
            row = ds[int(ti)]  # common shapes: [n_vars, n_points], [n_vars, 1, n_points]
            row = np.asarray(row)
            if row.ndim == 1:
                values.append(row)
                continue

            var_axis = None
            for axis, dim in enumerate(row.shape):
                if dim == len(all_vars):
                    var_axis = axis
                    break

            if var_axis is None:
                raise ValueError(
                    "Could not locate variable axis in anemoi sample with shape "
                    f"{row.shape} (n_vars={len(all_vars)})"
                )

            var_slice = np.take(row, idx, axis=var_axis)
            var_slice = np.squeeze(var_slice)
            if var_slice.ndim > 1:
                var_slice = var_slice.reshape(-1)
            values.append(var_slice)

        data[ch] = np.stack(values, axis=0)

    period = None
    if hasattr(ds, "frequency"):
        freq = np.timedelta64(ds.frequency)
        period = freq / np.timedelta64(1, "h")

    return DatasetInfo(lats=lats, lons=lons, data=data, period_hours=period)


def load_zarr_columnar(
    path: str | Path,
    lat_col: str,
    lon_col: str,
    data_cols: list[str] | None,
    n_time_samples: int,
    seed: int,
    max_points_per_time: int | None = 50_000,
) -> DatasetInfo:
    """Load from a zarr store with named lat/lon columns."""
    import zarr

    store = zarr.open(path, mode="r")
    rng = np.random.default_rng(seed)

    if lat_col in store and lon_col in store:
        lats = np.deg2rad(np.asarray(store[lat_col]))
        lons = np.deg2rad(np.asarray(store[lon_col]))

        if data_cols is None:
            skip = {lat_col, lon_col, "time", "datetime", "date"}
            data_cols = [k for k in store.keys() if k not in skip]

        # Determine time dimension
        first_arr = np.asarray(store[data_cols[0]])
        if first_arr.ndim == 1:
            # No time dimension
            data = {col: np.asarray(store[col])[np.newaxis, :] for col in data_cols}
        else:
            n_times_total = first_arr.shape[0]
            n_samples = min(n_time_samples, n_times_total)
            time_indices = rng.choice(n_times_total, size=n_samples, replace=False)
            time_indices.sort()
            data = {col: np.asarray(store[col])[time_indices] for col in data_cols}

        return DatasetInfo(lats=lats, lons=lons, data=data)

    # Observation-style zarr with a single data table and column metadata
    if "data" not in store:
        raise ValueError("Zarr store does not contain lat/lon arrays or a 'data' table.")

    data_arr = store["data"]
    colnames = list(data_arr.attrs.get("colnames", []))
    if not colnames:
        raise ValueError("Zarr 'data' array missing 'colnames' metadata.")

    if lat_col in colnames:
        lat_idx = colnames.index(lat_col)
    else:
        lat_idx = int(data_arr.attrs.get("lat_idx", [None])[0])
    if lon_col in colnames:
        lon_idx = colnames.index(lon_col)
    else:
        lon_idx = int(data_arr.attrs.get("lon_idx", [None])[0])

    if lat_idx is None or lon_idx is None:
        raise ValueError("Could not determine lat/lon column indices for observation table.")

    if data_cols is None:
        data_idxs = data_arr.attrs.get("data_idxs")
        if data_idxs is None:
            skip = {lat_col, lon_col, "time", "datetime", "date"}
            data_idxs = [i for i, name in enumerate(colnames) if name not in skip]
        data_cols = [colnames[i] for i in data_idxs]
        data_indices = list(data_idxs)
    else:
        missing = [c for c in data_cols if c not in colnames]
        if missing:
            raise ValueError(f"Requested columns not found in data table: {missing}")
        data_indices = [colnames.index(c) for c in data_cols]

    idx_key = next((k for k in store.keys() if k.startswith("idx_")), None)
    if idx_key is None:
        raise ValueError("Observation zarr missing time index array (idx_*).")

    idx = np.asarray(store[idx_key])
    n_rows = data_arr.shape[0]
    n_times_total = len(idx)
    end_idx = np.concatenate([idx[1:], np.array([n_rows], dtype=idx.dtype)])
    counts = end_idx - idx
    valid_times = np.where(counts >= 2)[0]
    if len(valid_times) == 0:
        raise ValueError("No valid time slices found in observation zarr table.")

    n_samples = min(n_time_samples, len(valid_times))
    time_indices = rng.choice(valid_times, size=n_samples, replace=False)
    time_indices.sort()

    lats_list: list[NDArray[np.float64]] = []
    lons_list: list[NDArray[np.float64]] = []
    data_list: dict[str, list[NDArray]] = {col: [] for col in data_cols}

    for ti in time_indices:
        start = int(idx[ti])
        end = int(end_idx[ti])
        if end <= start:
            continue
        rows = np.asarray(data_arr[start:end])
        if rows.ndim != 2:
            raise ValueError(f"Observation rows expected 2D, got {rows.shape}")

        lats = np.deg2rad(rows[:, lat_idx])
        lons = np.deg2rad(rows[:, lon_idx])

        if max_points_per_time is not None and len(lats) > max_points_per_time:
            sample = rng.choice(len(lats), size=max_points_per_time, replace=False)
            lats = lats[sample]
            lons = lons[sample]
            rows = rows[sample]

        lats_list.append(lats)
        lons_list.append(lons)
        for col, cidx in zip(data_cols, data_indices, strict=False):
            data_list[col].append(rows[:, cidx])

    if not lats_list:
        raise ValueError("No valid time slices found in observation zarr table.")

    return DatasetInfo(
        lats=None,
        lons=None,
        data={},
        lats_ragged=lats_list,
        lons_ragged=lons_list,
        data_ragged=data_list,
    )


def load_xarray(
    path: str | Path,
    lat_var: str,
    lon_var: str,
    data_vars: list[str] | None,
    n_time_samples: int,
    seed: int,
) -> DatasetInfo:
    """Load from a netCDF/xarray-compatible file."""
    import xarray as xr

    ds = xr.open_dataset(path)
    rng = np.random.default_rng(seed)

    lats_raw = ds[lat_var].values
    lons_raw = ds[lon_var].values

    if data_vars is None:
        skip = {lat_var, lon_var}
        data_vars = [v for v in ds.data_vars if v not in skip]

    # Handle gridded data: flatten spatial dims
    sample_var = ds[data_vars[0]]
    dims = sample_var.dims

    # Find time dimension
    time_dim = None
    for d in dims:
        if "time" in d.lower():
            time_dim = d
            break

    if time_dim is not None:
        n_times_total = ds.sizes[time_dim]
        n_samples = min(n_time_samples, n_times_total)
        time_indices = rng.choice(n_times_total, size=n_samples, replace=False)
        time_indices.sort()
        ds_sub = ds.isel({time_dim: time_indices})
    else:
        ds_sub = ds

    # If lat/lon are 1D coordinate arrays, create a meshgrid
    if lats_raw.ndim == 1 and lons_raw.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons_raw, lats_raw)
        lats_flat = np.deg2rad(lat_grid.ravel())
        lons_flat = np.deg2rad(lon_grid.ravel())
    else:
        lats_flat = np.deg2rad(lats_raw.ravel())
        lons_flat = np.deg2rad(lons_raw.ravel())

    data = {}
    for var in data_vars:
        arr = ds_sub[var].values
        # Flatten spatial dims, keep time
        if time_dim is not None:
            spatial_size = np.prod(arr.shape[1:])
            data[var] = arr.reshape(arr.shape[0], spatial_size)
        else:
            data[var] = arr.ravel()[np.newaxis, :]

    return DatasetInfo(lats=lats_flat, lons=lons_flat, data=data)


# ---------------------------------------------------------------------------
# Anomaly / detrending helpers
# ---------------------------------------------------------------------------


def _standardize_structured(data: NDArray) -> NDArray:
    """Per-point temporal standardization: remove time-mean, divide by time-std.

    This removes the climatological spatial pattern (latitude gradients, land-sea
    contrast, orographic effects) so that the autocorrelation reflects the
    correlation structure of *weather anomalies* rather than the smooth background
    climate.  Without this, fields with strong gradients (tp, q, 2t) show
    artificially long correlation lengths.
    """
    time_mean = np.nanmean(data, axis=0)  # [n_points]
    anomalies = data - time_mean[None, :]
    time_std = np.nanstd(data, axis=0)  # [n_points]
    time_std = np.where(time_std < 1e-12, 1.0, time_std)
    return anomalies / time_std[None, :]


def _standardize_ragged(data_list: list[NDArray]) -> list[NDArray]:
    """Per-snapshot spatial standardization for ragged unstructured data.

    Since each time slice may have different observation locations, we cannot
    compute a per-point temporal mean.  Instead, we remove the spatial mean and
    normalise by the spatial std within each snapshot.  This removes the gross
    large-scale gradient for each time step.
    """
    result: list[NDArray] = []
    for values in data_list:
        values = np.asarray(values, dtype=np.float64)
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        if std_val < 1e-12:
            result.append(values - mean_val)
        else:
            result.append((values - mean_val) / std_val)
    return result


# ---------------------------------------------------------------------------
# Spatial autocorrelation
# ---------------------------------------------------------------------------


def haversine_km(lat1: NDArray, lon1: NDArray, lat2: NDArray, lon2: NDArray) -> NDArray:
    """Vectorized haversine distance in km. Inputs in radians."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_spatial_autocorr(
    data: NDArray,
    lats: NDArray,
    lons: NDArray,
    max_lag_km: float = 3000.0,
    n_bins: int = 50,
    n_sample_pairs: int = 100_000,
    seed: int = 42,
) -> tuple[float, NDArray, NDArray]:
    """Estimate the spatial correlation length of a variable on a fixed grid.

    The algorithm randomly samples pairs of grid points at different
    distances and computes the Pearson correlation of the variable's
    anomaly values as a function of great-circle (haversine) distance.
    Specifically:

    1. Draw ``n_sample_pairs`` random (time, point_i, point_j) triples.
    2. Compute the haversine distance for each pair and discard pairs
       beyond ``max_lag_km``.
    3. Bin remaining pairs by distance into ``n_bins`` equal-width bins.
    4. For each bin, compute the Pearson correlation:
       ρ(d) = Cov(X_i, X_j) / Var(X),  where the variance is global.
    5. Fit an exponential decay ρ(d) ≈ exp(-d / L_corr) to the binned
       correlations to estimate the correlation length L_corr in km.

    Parameters
    ----------
    data : array [n_times, n_points]
    lats, lons : arrays [n_points], in radians
    max_lag_km : maximum lag distance for binning
    n_bins : number of distance bins
    n_sample_pairs : number of random point pairs to sample
    seed : RNG seed

    Returns
    -------
    l_corr_km : estimated correlation length in km
    bin_centers : distance bin centers in km
    bin_corr : binned correlation values
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim != 2:
        raise ValueError(
            f"Expected data with shape [n_times, n_points], got array with shape {data.shape}"
        )

    n_points_expected = len(lats)
    if data.shape[1] != n_points_expected and data.shape[0] == n_points_expected:
        data = data.T

    if data.shape[1] != n_points_expected:
        raise ValueError(
            "Data spatial dimension does not match lat/lon length: "
            f"data.shape={data.shape}, n_points={n_points_expected}"
        )

    n_times, n_points = data.shape

    # Sample random pairs of (time, point_i, point_j)
    time_indices = rng.integers(0, n_times, size=n_sample_pairs)
    idx_i = rng.integers(0, n_points, size=n_sample_pairs)
    idx_j = rng.integers(0, n_points, size=n_sample_pairs)

    # Remove self-pairs
    valid = idx_i != idx_j
    time_indices = time_indices[valid]
    idx_i = idx_i[valid]
    idx_j = idx_j[valid]

    # Compute distances
    distances = haversine_km(lats[idx_i], lons[idx_i], lats[idx_j], lons[idx_j])

    # Filter by max distance
    in_range = distances <= max_lag_km
    distances = distances[in_range]
    time_indices = time_indices[in_range]
    idx_i = idx_i[in_range]
    idx_j = idx_j[in_range]

    if len(distances) < 100:
        raise ValueError(
            "Too few valid point pairs for autocorrelation estimation "
            f"({len(distances)} pairs). The dataset may be too small or too sparse."
        )

    # Get values for all pairs
    vals_i = data[time_indices, idx_i]
    vals_j = data[time_indices, idx_j]

    # Remove pairs with NaN
    nan_mask = np.isnan(vals_i) | np.isnan(vals_j)
    if nan_mask.any():
        keep = ~nan_mask
        distances = distances[keep]
        vals_i = vals_i[keep]
        vals_j = vals_j[keep]

    if len(distances) < 100:
        raise ValueError(
            "Too few non-NaN point pairs for autocorrelation estimation "
            f"({len(distances)} pairs). The variable may contain too many NaNs."
        )

    # Bin by distance and compute correlation per bin
    bin_edges = np.linspace(0, max_lag_km, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.digitize(distances, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute per-bin Pearson correlation via E[XY] - E[X]E[Y]
    global_mean = np.nanmean(np.concatenate([vals_i, vals_j]))
    global_var = np.nanvar(np.concatenate([vals_i, vals_j]))

    if global_var < 1e-12:
        raise ValueError(
            "Near-zero variance in sampled data — the variable is effectively constant "
            "and its correlation length is undefined."
        )

    bin_corr = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_indices == b
        count = mask.sum()
        bin_counts[b] = count
        if count < 30:
            continue
        vi = vals_i[mask]
        vj = vals_j[mask]
        cov = np.mean((vi - global_mean) * (vj - global_mean))
        bin_corr[b] = cov / global_var

    # Fit exponential decay: corr(d) = exp(-d/L)
    l_corr_km = _fit_correlation_length(bin_centers, bin_corr, bin_counts)

    return l_corr_km, bin_centers, bin_corr


def compute_spatial_autocorr_unstructured(
    data_list: list[NDArray],
    lats_list: list[NDArray],
    lons_list: list[NDArray],
    max_lag_km: float = 3000.0,
    n_bins: int = 50,
    n_sample_pairs: int = 100_000,
    seed: int = 42,
) -> tuple[float, NDArray, NDArray]:
    """Compute spatial autocorrelation for ragged (unstructured) per-time observations.

    Unlike ``compute_spatial_autocorr``, this function handles datasets where
    each time step can have a different number of observation points at
    different locations (e.g. SYNOP, radiosonde).  The algorithm:

    1. Weight-sample time steps proportionally to the number of possible
       pairs, so denser time steps contribute more.
    2. For each sampled time step, draw random point pairs from that
       snapshot and compute haversine distances.
    3. Bin all (distance, value_i, value_j) tuples into distance bins and
       compute the Pearson correlation per bin, identical to the structured
       version.
    4. Fit an exponential decay to estimate the correlation length L_corr.
    """
    rng = np.random.default_rng(seed)
    if not (len(data_list) == len(lats_list) == len(lons_list)):
        raise ValueError("Ragged data, lat, and lon lists must have the same length.")

    n_times = len(data_list)
    if n_times == 0:
        raise ValueError("No time samples available for autocorrelation estimation.")

    sizes = np.array([len(lats) for lats in lats_list], dtype=int)
    valid_times = np.where(sizes >= 2)[0]
    if len(valid_times) == 0:
        raise ValueError(
            "All time slices have fewer than 2 observation points — "
            "cannot compute pairwise autocorrelation."
        )

    weights = sizes[valid_times] * (sizes[valid_times] - 1)
    if weights.sum() == 0:
        raise ValueError("Too few valid observation pairs for autocorrelation estimation.")

    time_samples = rng.choice(
        valid_times, size=n_sample_pairs, replace=True, p=weights / weights.sum()
    )

    distances_list: list[NDArray] = []
    vals_i_list: list[NDArray] = []
    vals_j_list: list[NDArray] = []

    for t in np.unique(time_samples):
        count = int(np.sum(time_samples == t))
        lats = lats_list[t]
        lons = lons_list[t]
        vals = np.asarray(data_list[t])
        if vals.ndim != 1:
            vals = vals.reshape(-1)
        n_points = len(vals)
        if n_points < 2:
            continue

        idx_i = rng.integers(0, n_points, size=count)
        idx_j = rng.integers(0, n_points, size=count)
        same = idx_i == idx_j
        while same.any():
            idx_j[same] = rng.integers(0, n_points, size=int(same.sum()))
            same = idx_i == idx_j

        distances = haversine_km(lats[idx_i], lons[idx_i], lats[idx_j], lons[idx_j])
        distances_list.append(distances)
        vals_i_list.append(vals[idx_i])
        vals_j_list.append(vals[idx_j])

    if not distances_list:
        raise ValueError(
            "No valid observation pairs were generated — "
            "the dataset may be too sparse for autocorrelation estimation."
        )

    distances = np.concatenate(distances_list)
    vals_i = np.concatenate(vals_i_list)
    vals_j = np.concatenate(vals_j_list)

    in_range = distances <= max_lag_km
    distances = distances[in_range]
    vals_i = vals_i[in_range]
    vals_j = vals_j[in_range]

    if len(distances) < 100:
        raise ValueError(
            "Too few valid point pairs for autocorrelation estimation "
            f"({len(distances)} pairs). The observation dataset may be too sparse."
        )

    nan_mask = np.isnan(vals_i) | np.isnan(vals_j)
    if nan_mask.any():
        keep = ~nan_mask
        distances = distances[keep]
        vals_i = vals_i[keep]
        vals_j = vals_j[keep]

    if len(distances) < 100:
        raise ValueError(
            "Too few non-NaN observation pairs for autocorrelation estimation "
            f"({len(distances)} pairs). The variable may contain too many NaNs."
        )

    bin_edges = np.linspace(0, max_lag_km, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.digitize(distances, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    global_mean = np.nanmean(np.concatenate([vals_i, vals_j]))
    global_var = np.nanvar(np.concatenate([vals_i, vals_j]))

    if global_var < 1e-12:
        raise ValueError(
            "Near-zero variance in sampled observation data — the variable is effectively "
            "constant and its correlation length is undefined."
        )

    bin_corr = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        bin_counts[b] = count
        if count < 30:
            continue
        vi = vals_i[mask]
        vj = vals_j[mask]
        cov = np.mean((vi - global_mean) * (vj - global_mean))
        bin_corr[b] = cov / global_var

    l_corr_km = _fit_correlation_length(bin_centers, bin_corr, bin_counts)

    return l_corr_km, bin_centers, bin_corr


def _fit_correlation_length(bin_centers: NDArray, bin_corr: NDArray, bin_counts: NDArray) -> float:
    """Fit correlation length from binned correlation data.

    The correlation length L_corr characterises the spatial scale over which
    a variable's anomalies are significantly correlated.  Because different
    atmospheric variables have very different correlation structures (e.g.
    smooth geopotential vs. noisy precipitation), no single estimator is
    universally reliable.  We therefore try three methods in order of
    decreasing robustness:

    Strategy (in priority order):

    1. **1/e threshold crossing** – the most model-free estimator.  We find
       the distance at which the binned correlation drops below 1/e ≈ 0.37.
       This works well for any monotonically decaying correlation function,
       including non-exponential shapes common in precipitation and wind
       fields.  It is preferred because it makes no parametric assumptions.

    2. **Integrated correlation scale** – L_eff = ∫ max(ρ(r), 0) dr.  When
       the correlation function never cleanly crosses the 1/e level (e.g.
       it starts below 1/e due to noise, or has a plateau), this integral
       measure provides a robust single-number summary.  It is less
       sensitive to bin noise than a threshold crossing but can under-
       estimate L_corr if the function has a long positive tail.

    3. **Log-linear (exponential) fit** – weighted least-squares regression
       of log(ρ) vs. distance, i.e. fitting ρ(d) = exp(-d/L).  This is
       used as a last resort because the exponential model can overestimate
       L_corr when the true correlation function has a steep initial drop
       followed by a slow tail (common for moisture variables).

    If all three methods fail (e.g. too few valid bins), a ``ValueError``
    is raised rather than returning a potentially misleading default.
    """
    min_bin_count = 30

    valid = (~np.isnan(bin_corr)) & (bin_counts >= min_bin_count) & (bin_corr > 0.01)
    valid_centers = bin_centers[valid]
    valid_corr = bin_corr[valid]

    if len(valid_centers) < 3:
        raise ValueError(
            "Too few valid distance bins for correlation length estimation "
            f"({len(valid_centers)} valid bins). The data may be too noisy or too sparse."
        )

    # --- Method 1: 1/e threshold crossing (most robust) ---
    threshold = 1.0 / np.e
    for i in range(len(valid_corr) - 1):
        if valid_corr[i] >= threshold > valid_corr[i + 1]:
            frac = (valid_corr[i] - threshold) / (valid_corr[i] - valid_corr[i + 1])
            l_corr = valid_centers[i] + frac * (valid_centers[i + 1] - valid_centers[i])
            if 10.0 < l_corr < 20000.0:
                return l_corr

    # --- Method 2: Integrated correlation scale ---
    # L_eff = ∫ max(ρ(r), 0) dr  (trapezoidal integration over valid bins)
    positive_corr = np.maximum(valid_corr, 0.0)
    if len(valid_centers) >= 2:
        l_eff = float(np.trapezoid(positive_corr, valid_centers))
        if 10.0 < l_eff < 20000.0:
            return l_eff

    # --- Method 3: Log-linear (exponential) fit ---
    log_corr = np.log(valid_corr)
    try:
        weights = np.sqrt(bin_counts[valid].astype(float))
        coeffs = np.polyfit(valid_centers, log_corr, 1, w=weights)
        slope = coeffs[0]
        if slope < -1e-8:
            l_corr = -1.0 / slope
            if 10.0 < l_corr < 20000.0:
                return l_corr
    except (np.linalg.LinAlgError, ValueError):
        pass

    raise ValueError(
        "All three correlation length estimation methods failed "
        "(1/e crossing, integrated scale, log-linear fit). "
        "The correlation function may be too noisy or non-monotonic."
    )


# ---------------------------------------------------------------------------
# hl_mask mapping and grouping
# ---------------------------------------------------------------------------


def correlation_length_to_hl_mask(
    l_corr_km: float,
    healpix_level: int,
    multiplier: float = 1.5,
) -> int:
    """Map a correlation length to the appropriate HEALPix masking level.

    Finds the finest HEALPix level where the cell size exceeds
    ``l_corr_km * multiplier``.

    Parameters
    ----------
    l_corr_km : spatial correlation length in km
    healpix_level : the training grid HEALPix level
    multiplier : how much larger mask blocks should be vs. correlation length

    Returns
    -------
    hl_mask : integer HEALPix level for masking (0 to healpix_level)
    """
    target_km = l_corr_km * multiplier
    # HEALPix cell size at level l: approx 4000 / 2^l km
    # (12 base pixels, area = 4*pi*R^2/Npix, side ~ sqrt(area))
    for hl in range(healpix_level, -1, -1):
        n_pix = 12 * 4**hl
        cell_area_km2 = (4 * np.pi * EARTH_RADIUS_KM**2) / n_pix
        cell_size_km = np.sqrt(cell_area_km2)
        if cell_size_km >= target_km:
            return hl
    return 0


def group_by_hl_mask(var_results: dict[str, VarResult]) -> dict[int, list[str]]:
    """Group variables by their recommended hl_mask level.

    Returns
    -------
    dict mapping hl_mask -> list of variable names
    """
    groups: dict[int, list[str]] = {}
    for name, result in var_results.items():
        groups.setdefault(result.hl_mask, []).append(name)
    return dict(sorted(groups.items()))


def group_by_hl_mask_for_multiplier(
    var_results: dict[str, VarResult],
    healpix_level: int,
    multiplier: float,
) -> dict[int, list[str]]:
    """Group variables by hl_mask for a given correlation multiplier."""
    groups: dict[int, list[str]] = {}
    for name, result in var_results.items():
        hl = correlation_length_to_hl_mask(result.l_corr_km, healpix_level, multiplier)
        groups.setdefault(hl, []).append(name)
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

# Approximate cell sizes for display
_HL_CELL_SIZES = {
    0: "~4000",
    1: "~2000",
    2: "~1000",
    3: "~500",
    4: "~250",
    5: "~125",
}

_HL_CORR_RANGES = {
    0: "2000+ km",
    1: "1000-2000 km",
    2: "500-1000 km",
    3: "250-500 km",
    4: "100-250 km",
    5: "<100 km",
}


def format_results_table(var_results: dict[str, VarResult]) -> str:
    """Format a human-readable results table."""
    lines = []
    lines.append(f"{'Variable':<20s}  {'L_corr (km)':>12s}  {'hl_mask':>7s}")
    lines.append("-" * 20 + "  " + "-" * 12 + "  " + "-" * 7)
    for name, r in sorted(var_results.items(), key=lambda x: -x[1].l_corr_km):
        lines.append(f"{name:<20s}  {r.l_corr_km:>12.0f}  {r.hl_mask:>7d}")
    return "\n".join(lines)


def format_groupings(groups: dict[int, list[str]]) -> str:
    """Format stream grouping suggestions."""
    lines = []
    for hl, vars_list in sorted(groups.items()):
        corr_range = _HL_CORR_RANGES.get(hl, "unknown")
        vars_str = ", ".join(vars_list)
        lines.append(f"Stream group hl_mask={hl} (L_corr {corr_range}): {vars_str}")
    return "\n".join(lines)


def generate_yaml_snippets(groups: dict[int, list[str]]) -> str:
    """Generate YAML masking_override snippets for each group."""
    lines = []
    for hl, vars_list in sorted(groups.items()):
        vars_str = ", ".join(vars_list)
        lines.append(f"# Stream group for: {vars_str}")
        lines.append(f"# Recommended hl_mask: {hl}")
        lines.append("masking_override:")
        lines.append("  model_input:")
        lines.append("    masking_strategy_config:")
        lines.append(f"      hl_mask: {hl}")
        lines.append("  target_input:")
        lines.append("    masking_strategy_config:")
        lines.append(f"      hl_mask: {max(0, hl)}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def analyse_dataset(
    dataset_path: str | Path,
    dataset_type: str,
    channels: list[str] | None = None,
    n_time_samples: int = 100,
    n_sample_pairs: int = 100_000,
    correlation_multiplier: float = 1.5,
    healpix_level: int = 5,
    seed: int = 42,
    lat_col: str = "lat",
    lon_col: str = "lon",
    lat_var: str = "latitude",
    lon_var: str = "longitude",
    max_points_per_time: int | None = 50_000,
    detrend: bool = True,
) -> tuple[dict[str, VarResult], dict[int, list[str]]]:
    """Run the full analysis pipeline.

    Parameters
    ----------
    detrend : bool
        If True (default), remove the climatological spatial pattern before
        computing autocorrelation.  This prevents large-scale gradients
        (latitude, orography) from inflating correlation lengths.

    Returns
    -------
    var_results : per-variable analysis results
    groups : hl_mask -> list of variable names
    """
    # Load data
    # NOTE: We use lightweight standalone loaders instead of the training
    # DataReaders (DataReaderAnemoi / DataReaderObs).  The analysis needs
    # per-variable [n_times, n_points] arrays, whereas the readers return
    # all channels flattened into [n_times*n_points, n_channels] which would
    # need to be reshaped back.  Reusing them would also require constructing
    # a TimeWindowHandler to samples times.
    logger.info(f"Loading dataset from {dataset_path} (type={dataset_type})")
    if dataset_type == "anemoi":
        ds_info = load_anemoi(dataset_path, n_time_samples, channels, seed)
    elif dataset_type in ("fesom", "obs"):
        ds_info = load_zarr_columnar(
            dataset_path,
            lat_col,
            lon_col,
            channels,
            n_time_samples,
            seed,
            max_points_per_time=max_points_per_time,
        )
    elif dataset_type in ("xarray", "cams", "eobs", "iconart", "iconesm"):
        ds_info = load_xarray(dataset_path, lat_var, lon_var, channels, n_time_samples, seed)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    if ds_info.data_ragged is not None:
        n_times = len(ds_info.lats_ragged or [])
        avg_points = int(np.mean([len(x) for x in ds_info.lats_ragged or [0]]))
        logger.info(
            f"Loaded {len(ds_info.data_ragged)} variables, {n_times} time samples, "
            f"avg {avg_points} points/time (ragged)"
        )
    else:
        logger.info(
            f"Loaded {len(ds_info.data)} variables, "
            f"{next(iter(ds_info.data.values())).shape[0]} time samples, "
            f"{len(ds_info.lats)} spatial points"
        )

    # Standardize to anomalies if requested
    if detrend:
        logger.info("Detrending: computing anomaly autocorrelation (climatology removed)")
        if ds_info.data_ragged is not None:
            for var_name in ds_info.data_ragged:
                ds_info.data_ragged[var_name] = _standardize_ragged(ds_info.data_ragged[var_name])
        else:
            for var_name in ds_info.data:
                ds_info.data[var_name] = _standardize_structured(ds_info.data[var_name])
    else:
        logger.info("No detrending: computing raw-field autocorrelation")

    # Compute autocorrelation per variable
    var_results: dict[str, VarResult] = {}
    if ds_info.data_ragged is not None:
        assert ds_info.lats_ragged is not None
        assert ds_info.lons_ragged is not None
        data_items = ds_info.data_ragged.items()
    else:
        data_items = ds_info.data.items()

    for var_name, var_data in data_items:
        logger.info(f"Computing autocorrelation for '{var_name}'...")
        if ds_info.data_ragged is not None:
            l_corr, bin_centers, bin_corr = compute_spatial_autocorr_unstructured(
                var_data,
                ds_info.lats_ragged,
                ds_info.lons_ragged,
                n_sample_pairs=n_sample_pairs,
                seed=seed,
            )
        else:
            l_corr, bin_centers, bin_corr = compute_spatial_autocorr(
                var_data,
                ds_info.lats,
                ds_info.lons,
                n_sample_pairs=n_sample_pairs,
                seed=seed,
            )
        hl = correlation_length_to_hl_mask(l_corr, healpix_level, correlation_multiplier)
        var_results[var_name] = VarResult(
            name=var_name,
            l_corr_km=l_corr,
            hl_mask=hl,
            bin_centers_km=bin_centers,
            bin_correlations=bin_corr,
        )
        logger.info(f"  {var_name}: L_corr={l_corr:.0f} km -> hl_mask={hl}")

    groups = group_by_hl_mask(var_results)
    return var_results, groups


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute spatial autocorrelation per variable and suggest masking configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument(
        "--type",
        required=True,
        choices=["anemoi", "fesom", "obs", "xarray", "cams", "eobs", "iconart", "iconesm"],
        help="Dataset type",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="Variables to analyse. If omitted, all variables in the dataset are analysed.",
    )
    parser.add_argument("--n-time-samples", type=int, default=100, help="Timesteps to sample")
    parser.add_argument("--n-sample-pairs", type=int, default=100_000, help="Point pairs to sample")
    parser.add_argument(
        "--correlation-multiplier",
        type=float,
        default=1.5,
        help="Multiplier for mapping L_corr -> mask block size",
    )
    parser.add_argument(
        "--correlation-multipliers",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of multipliers to print separate suggestions",
    )
    parser.add_argument("--healpix-level", type=int, default=5, help="Training grid HEALPix level")
    parser.add_argument("--output", default=None, help="Output YAML file path")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    # Extra args for non-anemoi types
    parser.add_argument("--lat-col", default="lat", help="Latitude column name (zarr)")
    parser.add_argument("--lon-col", default="lon", help="Longitude column name (zarr)")
    parser.add_argument("--lat-var", default="latitude", help="Latitude variable name (xarray)")
    parser.add_argument("--lon-var", default="longitude", help="Longitude variable name (xarray)")
    parser.add_argument(
        "--max-points-per-time",
        type=int,
        default=50_000,
        help="Max points per time slice for unstructured observations",
    )
    parser.add_argument(
        "--no-detrend",
        action="store_true",
        help="Disable anomaly standardization (use raw fields for autocorrelation)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    var_results, groups = analyse_dataset(
        dataset_path=args.dataset,
        dataset_type=args.type,
        channels=args.channels,
        n_time_samples=args.n_time_samples,
        n_sample_pairs=args.n_sample_pairs,
        correlation_multiplier=args.correlation_multiplier,
        healpix_level=args.healpix_level,
        seed=args.seed,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        lat_var=args.lat_var,
        lon_var=args.lon_var,
        max_points_per_time=args.max_points_per_time,
        detrend=not args.no_detrend,
    )

    # Output results to stdout
    def _write(msg: str = "") -> None:
        import sys

        sys.stdout.write(msg + "\n")

    mode = "anomaly" if not args.no_detrend else "raw"
    _write("\n" + "=" * 50)
    _write(f"Per-variable autocorrelation analysis (mode={mode})")
    _write("=" * 50)
    _write(format_results_table(var_results))
    _write()
    # When --correlation-multipliers is given (e.g. 1.0 1.5 2.0), we print
    # separate grouping tables and YAML snippets for each multiplier value so
    # the user can compare how aggressively variables are grouped.  Otherwise
    # we use the single --correlation-multiplier value.
    multipliers = args.correlation_multipliers or [args.correlation_multiplier]
    yaml_sections: list[str] = []

    for multiplier in multipliers:
        if multiplier == args.correlation_multiplier:
            groups_for_multiplier = groups
        else:
            groups_for_multiplier = group_by_hl_mask_for_multiplier(
                var_results, args.healpix_level, multiplier
            )

        _write("=" * 50)
        _write(f"Suggested stream groupings (multiplier={multiplier:g})")
        _write("=" * 50)
        _write(format_groupings(groups_for_multiplier))
        _write()

        yaml_output = generate_yaml_snippets(groups_for_multiplier)
        _write("=" * 50)
        _write(f"YAML masking_override snippets (multiplier={multiplier:g})")
        _write("=" * 50)
        _write(yaml_output)
        yaml_sections.append(f"# Multiplier: {multiplier:g}\n{yaml_output.strip()}")

    if args.output:
        Path(args.output).write_text("\n\n".join(yaml_sections) + "\n")
        _write(f"\nYAML written to {args.output}")


if __name__ == "__main__":
    main()
