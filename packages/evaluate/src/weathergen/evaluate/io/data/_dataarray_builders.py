# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""DataArray construction helpers for WeatherGenZarrReader.get_data_raw.

These functions were formerly @staticmethod methods on WeatherGenZarrReader.
Extracted here so that the reader module stays focused on I/O orchestration.
"""

import numpy as np
import xarray as xr
from numpy.typing import NDArray


def build_gridded_dataarrays(
    tars_list: list[NDArray],
    preds_list: list[NDArray],
    samples: list[int],
    read_channels: list[str],
    lat: NDArray,
    lon: NDArray,
    per_sample_valid_times: list[np.datetime64],
    source_interval_starts: NDArray,
    forecast_step_val: int,
    ensemble: list[str],
    all_ens: list[str] | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build DataArrays for gridded data by stacking samples along a new axis.

    All samples share the same grid, so np.stack works directly.

    Parameters
    ----------
    tars_list : list[np.ndarray]
        Per-sample target arrays, shape (n_ipoints, n_channels).
    preds_list : list[np.ndarray]
        Per-sample prediction arrays, shape (n_ipoints, n_channels[, n_ens]).
    samples : list[int]
        Sample indices.
    read_channels : list[str]
        Channel names.
    lat : np.ndarray
        Latitude array (full grid; sliced to n_ipoints internally).
    lon : np.ndarray
        Longitude array (full grid; sliced to n_ipoints internally).
    per_sample_valid_times : list[np.datetime64]
        One valid_time per sample.  Each sample represents a different
        forecast initialisation, so valid_time differs across samples
        even for the same forecast step.
    source_interval_starts : np.ndarray
        Per-sample source interval start times, shape (n_samples,).
    forecast_step_val : int
        Forecast step value to assign as coordinate.
    ensemble : list[str]
        Requested ensemble members or ["mean"].
    all_ens : list[str] | None
        All ensemble member names from zarr (needed for index mapping).

    Returns
    -------
    da_tar, da_pred : xr.DataArray
    """
    n_samples = len(samples)
    n_ipoints = tars_list[0].shape[0]
    sub_lat = lat[:n_ipoints]
    sub_lon = lon[:n_ipoints]

    tars_stacked = np.stack(tars_list, axis=0)  # (n_samples, n_ipoints, n_channels)
    preds_stacked = np.stack(preds_list, axis=0)  # (n_samples, n_ipoints, n_channels[, n_ens])

    # valid_time must be 2D (sample, ipoint) to match the shape produced by
    # get_data() → _force_consistent_grids → xr.concat(dim="sample").
    # _add_lead_time_coord computes lead_time = valid_time - source_interval_start
    # and needs both arrays to broadcast as (sample, ipoint).
    # Each sample has its OWN valid_time (different initialisation dates),
    # so we build a 2D array where row i is filled with sample i's time.
    vt_col = np.array(per_sample_valid_times, dtype="datetime64[ns]")  # (n_samples,)
    valid_time_2d = np.broadcast_to(
        vt_col[:, np.newaxis],  # (n_samples, 1)
        (n_samples, n_ipoints),
    ).copy()  # copy: broadcast arrays are read-only

    base_coords = {
        "sample": samples,
        "ipoint": np.arange(n_ipoints),
        "channel": read_channels,
        "lat": ("ipoint", sub_lat),
        "lon": ("ipoint", sub_lon),
        "valid_time": (("sample", "ipoint"), valid_time_2d),
        "source_interval_start": ("sample", source_interval_starts.copy()),
        "forecast_step": forecast_step_val,
    }

    da_tar = xr.DataArray(
        tars_stacked,
        dims=["sample", "ipoint", "channel"],
        coords=base_coords,
    )

    da_pred = build_pred_dataarray(
        preds_stacked,
        base_coords,
        ensemble,
        all_ens,
    )

    return da_tar, da_pred


def build_scatter_dataarrays(
    tars_list: list[NDArray],
    preds_list: list[NDArray],
    samples: list[int],
    read_channels: list[str],
    per_sample_valid_times: list[np.datetime64],
    source_interval_starts: NDArray,
    forecast_step_val: int,
    ensemble: list[str],
    all_ens: list[str] | None,
    per_sample_coords: list[NDArray | None],
    coords_fallback: NDArray,
    per_sample_obs_times: list[NDArray] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build DataArrays for non-gridded (scatter) data.

    Samples may have different ipoint counts, so we concatenate along
    the ipoint dimension — matching the get_data() behavior for scatter data.

    Parameters
    ----------
    tars_list : list[np.ndarray]
        Per-sample target arrays.
    preds_list : list[np.ndarray]
        Per-sample prediction arrays.
    samples : list[int]
        Sample indices.
    read_channels : list[str]
        Channel names.
    per_sample_valid_times : list[np.datetime64]
        One representative valid_time per sample (used as fallback when
        per-observation times are not available).
    source_interval_starts : np.ndarray
        Per-sample source interval start times.
    forecast_step_val : int
        Forecast step value to assign as coordinate.
    ensemble : list[str]
        Requested ensemble members or ["mean"].
    all_ens : list[str] | None
        All ensemble member names from zarr (needed for index mapping).
    per_sample_coords : list[np.ndarray | None]
        Per-sample coordinate arrays read from zarr (shape (n_ip, 2) each).
        Falls back to coords_fallback when None.
    coords_fallback : np.ndarray
        Reference coords from sample 0, used as fallback.
    per_sample_obs_times : list[np.ndarray] | None
        Per-sample arrays of observation times, shape (n_ip,) each.
        When provided, each observation gets its actual timestamp;
        otherwise the single per_sample_valid_times value is broadcast.

    Returns
    -------
    da_tar, da_pred : xr.DataArray
    """
    per_sample_tars = []
    per_sample_preds = []

    for si, sample in enumerate(samples):
        n_ip = tars_list[si].shape[0]
        tar_data = tars_list[si]  # (n_ip, n_channels)
        pred_data = preds_list[si]  # (n_ip, n_channels[, n_ens])

        # Use per-sample coords if available, otherwise fall back to reference
        sc = per_sample_coords[si] if si < len(per_sample_coords) else None
        if sc is not None and len(sc) >= n_ip:
            sample_lat = sc[:n_ip, 0].astype(np.float64)
            sample_lon = sc[:n_ip, 1].astype(np.float64)
        elif coords_fallback is not None and n_ip <= len(coords_fallback):
            sample_lat = coords_fallback[:n_ip, 0].astype(np.float64)
            sample_lon = coords_fallback[:n_ip, 1].astype(np.float64)
        else:
            sample_lat = np.full(n_ip, np.nan)
            sample_lon = np.full(n_ip, np.nan)

        vt_arr = (
            per_sample_obs_times[si][:n_ip].astype("datetime64[ns]")
            if per_sample_obs_times is not None and si < len(per_sample_obs_times)
            else np.full(n_ip, per_sample_valid_times[si], dtype="datetime64[ns]")
        )
        si_start = source_interval_starts[si]

        sample_coords = {
            "ipoint": np.arange(n_ip),
            "channel": read_channels,
            "lat": ("ipoint", sample_lat),
            "lon": ("ipoint", sample_lon),
            "valid_time": ("ipoint", vt_arr),
            "source_interval_start": si_start,
            "forecast_step": forecast_step_val,
            "sample": sample,
        }

        da_t = xr.DataArray(
            tar_data,
            dims=["ipoint", "channel"],
            coords=sample_coords,
        )
        per_sample_tars.append(da_t)

        # Handle ensemble for predictions
        if pred_data.ndim == 3:
            if ensemble == ["mean"]:
                pred_data = pred_data.mean(axis=-1)
                pred_coords = dict(sample_coords)
                da_p = xr.DataArray(
                    pred_data,
                    dims=["ipoint", "channel"],
                    coords=pred_coords,
                )
            else:
                ens_idxs = (
                    [all_ens.index(e) for e in ensemble]
                    if all_ens
                    else list(range(pred_data.shape[-1]))
                )
                pred_data = pred_data[:, :, ens_idxs]
                pred_coords = dict(sample_coords)
                pred_coords["ens"] = ensemble
                da_p = xr.DataArray(
                    pred_data,
                    dims=["ipoint", "channel", "ens"],
                    coords=pred_coords,
                )
        else:
            da_p = xr.DataArray(
                pred_data,
                dims=["ipoint", "channel"],
                coords=sample_coords,
            )
        per_sample_preds.append(da_p)

    # Concatenate along ipoint (like get_data() does for non-gridded)
    da_tar = xr.concat(per_sample_tars, dim="ipoint", coords="different", compat="equals")
    da_pred = xr.concat(per_sample_preds, dim="ipoint", coords="different", compat="equals")

    return da_tar, da_pred


def build_pred_dataarray(
    preds_stacked: NDArray,
    base_coords: dict,
    ensemble: list[str],
    all_ens: list[str] | None,
) -> xr.DataArray:
    """Build prediction DataArray, handling ensemble dimension.

    Parameters
    ----------
    preds_stacked : np.ndarray
        Shape (n_samples, n_ipoints, n_channels[, n_ens]).
    base_coords : dict
        Coordinate dict (without ens).
    ensemble : list[str]
        Requested ensemble members or ["mean"].
    all_ens : list[str] | None
        All ensemble member names from zarr (needed for index mapping).
    """
    if preds_stacked.ndim == 4:
        if ensemble == ["mean"]:
            # Average over ensemble axis, drop ens coordinate
            # (matches get_data() which does .mean("ens").squeeze())
            preds_stacked = preds_stacked.mean(axis=-1)
            return xr.DataArray(
                preds_stacked,
                dims=["sample", "ipoint", "channel"],
                coords=base_coords,
            )
        else:
            # Select requested ensemble members by index
            if all_ens is not None:
                ens_idxs = [all_ens.index(e) for e in ensemble]
                preds_stacked = preds_stacked[:, :, :, ens_idxs]
            pred_coords = dict(base_coords)
            pred_coords["ens"] = ensemble
            return xr.DataArray(
                preds_stacked,
                dims=["sample", "ipoint", "channel", "ens"],
                coords=pred_coords,
            )
    else:
        # No ensemble dim
        return xr.DataArray(
            preds_stacked,
            dims=["sample", "ipoint", "channel"],
            coords=base_coords,
        )
