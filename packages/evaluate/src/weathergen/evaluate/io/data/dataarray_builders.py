# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""DataArray construction helpers for WeatherGenZarrReader.get_data.

These functions were formerly @staticmethod methods on WeatherGenZarrReader.
Extracted here so that the reader module stays focused on I/O orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass

import earthkit.regrid as ekr
import numpy as np
import xarray as xr
from earthkit.regrid.gridspec import GridSpec as EkGridSpec
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class EnsembleSelect:
    """Pre-resolved ensemble selection.

    Use :meth:`mean` for the ensemble-mean sentinel, or :meth:`from_names`
    to resolve requested member names against the full list stored in zarr.
    """

    labels: list[str]
    indices: list[int]
    is_mean: bool = False

    # ------ factories ------

    @classmethod
    def mean(cls) -> EnsembleSelect:
        """Sentinel: average over the ensemble axis and drop it."""
        return cls(labels=[], indices=[], is_mean=True)

    @classmethod
    def from_names(
        cls,
        requested: list[str],
        all_ens: list[str] | None,
    ) -> EnsembleSelect:
        """Resolve *requested* member names into concrete indices.

        Parameters
        ----------
        requested : list[str]
            Requested ensemble members (e.g. ``["ens0", "ens2"]``).
            Pass ``["mean"]`` to get the mean sentinel.
        all_ens : list[str] | None
            All ensemble member names from the zarr store.

        Returns
        -------
        EnsembleSelect
        """
        if requested == ["mean"]:
            return cls.mean()
        if all_ens is not None:
            indices = [all_ens.index(e) for e in requested]
        else:
            indices = list(range(len(requested)))
        return cls(labels=requested, indices=indices)


def build_gridded_dataarrays(
    tars_list: list[NDArray],
    preds_list: list[NDArray],
    samples: list[int],
    read_channels: list[str],
    lat: NDArray,
    lon: NDArray,
    per_sample_valid_times: list[np.datetime64],
    init_times: NDArray,
    forecast_step_val: int,
    ens_select: EnsembleSelect,
    regrid_opts: dict,
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
    init_times : np.ndarray
        Per-sample source interval start times, shape (n_samples,).
    forecast_step_val : int
        Forecast step value to assign as coordinate.
    ens_select : EnsembleSelect
        Pre-resolved ensemble selection (from :meth:`EnsembleSelect.from_names`).
        ``EnsembleSelect.mean()`` → mean; otherwise selects members.
    regrid_opts : dict
        Regrid each sample from its original grid to a regular
        lat/lon grid before stacking.  Must contain 'target_grid' (e.g. [1.5, 1.5]).
        Optionally 'original_grid' to skip auto-detection.

    Returns
    -------
    da_tar, da_pred : xr.DataArray
    """
    # Regrid each sample individually (correct n_ipoints per sub-step)
    if regrid_opts:
        tars_list, preds_list, lat, lon = regrid_dataarrays(tars_list, preds_list, regrid_opts)

    n_samples = len(samples)
    n_ipoints = tars_list[0].shape[0]
    sub_lat = lat[:n_ipoints]
    sub_lon = lon[:n_ipoints]

    tars_stacked = np.stack(tars_list, axis=0)  # (n_samples, n_ipoints, n_channels)
    preds_stacked = np.stack(preds_list, axis=0)  # (n_samples, n_ipoints, n_channels[, n_ens])

    # valid_time must be 2D (sample, ipoint) to match the shape produced by
    # get_data() → _force_consistent_grids → xr.concat(dim="sample").
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
        "init_times": ("sample", init_times.copy()),
        "forecast_step": forecast_step_val,
    }

    da_tar = _build_dataarray(tars_stacked, base_coords)

    da_pred = _build_dataarray(
        preds_stacked,
        base_coords,
        ens_select,
    )

    return da_tar, da_pred


def build_scatter_dataarrays(
    tars_list: list[NDArray],
    preds_list: list[NDArray],
    samples: list[int],
    read_channels: list[str],
    per_sample_valid_times: list[np.datetime64],
    init_times: NDArray,
    forecast_step_val: int,
    ens_select: EnsembleSelect,
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
    init_times : np.ndarray
        Per-sample source interval start times.
    forecast_step_val : int
        Forecast step value to assign as coordinate.
    ens_select : EnsembleSelect
        Pre-resolved ensemble selection (from :meth:`EnsembleSelect.from_names`).
        ``EnsembleSelect.mean()`` → mean; otherwise selects members.
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
            sample_lat = sc[:n_ip, 0]
            sample_lon = sc[:n_ip, 1]
        elif coords_fallback is not None and n_ip <= len(coords_fallback):
            sample_lat = coords_fallback[:n_ip, 0]
            sample_lon = coords_fallback[:n_ip, 1]
        else:
            sample_lat = np.full(n_ip, np.nan)
            sample_lon = np.full(n_ip, np.nan)

        vt_arr = (
            per_sample_obs_times[si][:n_ip].astype("datetime64[ns]")
            if per_sample_obs_times is not None and si < len(per_sample_obs_times)
            else np.full(n_ip, per_sample_valid_times[si], dtype="datetime64[ns]")
        )

        sample_coords = {
            "ipoint": np.arange(n_ip),
            "channel": read_channels,
            "lat": ("ipoint", sample_lat),
            "lon": ("ipoint", sample_lon),
            "valid_time": ("ipoint", vt_arr),
            "init_times": init_times[si],
            "forecast_step": forecast_step_val,
            "sample": sample,
        }

        scatter_dims = ["ipoint", "channel"]

        da_t = _build_dataarray(
            tar_data,
            sample_coords,
            base_dims=scatter_dims,
        )
        per_sample_tars.append(da_t)

        da_p = _build_dataarray(
            pred_data,
            sample_coords,
            ens_select,
            base_dims=scatter_dims,
        )
        per_sample_preds.append(da_p)

    # Concatenate along ipoint (like get_data() does for non-gridded)
    da_tar = xr.concat(per_sample_tars, dim="ipoint", coords="different", compat="equals")
    da_pred = xr.concat(per_sample_preds, dim="ipoint", coords="different", compat="equals")

    return da_tar, da_pred


def _build_dataarray(
    data: NDArray,
    base_coords: dict,
    ens_select: EnsembleSelect | None = None,
    base_dims: list[str] | None = None,
) -> xr.DataArray:
    """Build a DataArray, resolving an optional ensemble dimension.

    Works for both targets (no ensemble) and predictions (with or without
    ensemble).  When the trailing axis is not an ensemble dimension the
    *ens_select* argument is harmlessly ignored, so callers can omit it
    for targets.

    Parameters
    ----------
    data : np.ndarray
        Array whose last axis is optionally an ensemble dimension.
        Typical shapes: ``(n_samples, n_ipoints, n_channels[, n_ens])``
        for gridded data or ``(n_ipoints, n_channels[, n_ens])`` for a
        single scatter sample.
    base_coords : dict
        Coordinate dict (without ``ens``).
    ens_select : EnsembleSelect | None
        ``None`` or ``EnsembleSelect.mean()`` → average over the ensemble
        axis.  ``EnsembleSelect.from_names(...)`` → select members.
    base_dims : list[str] | None
        Dimension names for the non-ensemble axes.  Defaults to
        ``["sample", "ipoint", "channel"]`` (gridded / stacked case).
    """
    if base_dims is None:
        base_dims = ["sample", "ipoint", "channel"]

    dims = list(base_dims)
    coords = dict(base_coords)
    n_base = len(base_dims)

    if data.ndim == n_base + 1:
        if ens_select is None or ens_select.is_mean:
            # Average over ensemble axis, drop ens coordinate
            data = data.mean(axis=-1)
        else:
            idx = tuple([slice(None)] * n_base + [ens_select.indices])
            data = data[idx]
            dims.append("ens")
            coords["ens"] = ens_select.labels

    return xr.DataArray(data, dims=dims, coords=coords)


# ---------------------------------------------------------------------------
# Numpy-level regridding (called inside each worker)
# ---------------------------------------------------------------------------


# Grid point counts for known ECMWF grids
def get_grid_name(n_ipoints: int) -> str:
    """Get the grid name corresponding to a given number of grid points.

    Parameters
    ----------
    n_ipoints : int
        The number of grid points in the input data.

    Returns
    -------
    str
        The name of the grid corresponding to the given number of grid points.
    """
    known_grids = {
        542080: "N320",
        40320: "O96",
    }
    return known_grids.get(n_ipoints)


def _detect_grid(n_ipoints: int, regrid_opts: dict) -> str:
    """Resolve the original grid name from n_ipoints or explicit config."""
    original_grid = regrid_opts.get("original_grid") if isinstance(regrid_opts, dict) else None
    if original_grid is not None:
        return original_grid
    grid = get_grid_name(n_ipoints)
    if grid is None:
        raise ValueError(
            f"Cannot auto-detect grid type: {n_ipoints} grid points does not match "
            f"any known grid. Supported: N320 (542080 pts), O96 (40320 pts). "
            f"Pass 'original_grid' explicitly in the regrid config."
        )
    return grid


def _regrid_field(field_1d: NDArray, in_grid: dict, out_grid: dict) -> NDArray:
    """Regrid a single 1D field and return the flattened result."""

    result_2d = ekr.interpolate(field_1d, in_grid, out_grid)
    return result_2d.ravel()


def _regrid_array(data: NDArray, regrid_opts: dict) -> NDArray:
    """Regrid a numpy array from a reduced Gaussian grid to a regular lat/lon grid.

    Parameters
    ----------
    data : NDArray
        Input array of shape ``(n_ipoints, n_channels)`` or
        ``(n_ipoints, n_channels, n_ens)``.
    regrid_opts : dict
        Must contain 'target_grid' (e.g. [1.5, 1.5]).  Optionally
        'original_grid' to skip auto-detection.

    Returns
    -------
    NDArray
        Regridded array of shape ``(n_lat * n_lon, n_channels[, n_ens])``.
    """
    n_ipoints = data.shape[0]
    original_grid = regrid_opts.get("original_grid")
    if original_grid is None:
        original_grid = _detect_grid(n_ipoints, regrid_opts)
    target_grid = regrid_opts.get("target_grid", [1.5, 1.5])
    if not isinstance(target_grid, str):
        target_grid = list(target_grid)  # earthkit.regrid requires a plain list, not numpy array

    in_grid = {"grid": original_grid}
    out_grid = {"grid": target_grid}

    if data.ndim == 2:
        # (n_ipoints, n_channels)
        n_channels = data.shape[1]
        cols = [_regrid_field(data[:, ch], in_grid, out_grid) for ch in range(n_channels)]
        out = np.column_stack(cols)
    elif data.ndim == 3:
        # (n_ipoints, n_channels, n_ens)
        n_channels, n_ens = data.shape[1], data.shape[2]
        slices = [
            [_regrid_field(data[:, ch, e], in_grid, out_grid) for e in range(n_ens)]
            for ch in range(n_channels)
        ]
        out = np.stack([np.column_stack(s) for s in slices], axis=1)
    else:
        raise ValueError(f"Unexpected data shape for regridding: {data.shape}")

    return out


def regrid_dataarrays(tars_list, preds_list, regrid_opts):
    """Regrid each sample in tars_list and preds_list according to regrid_opts."""

    tars_list = [_regrid_array(t, regrid_opts) for t in tars_list]
    preds_list = [_regrid_array(p, regrid_opts) for p in preds_list]

    target_grid = (
        regrid_opts.get("target_grid", [1.5, 1.5]) if isinstance(regrid_opts, dict) else [1.5, 1.5]
    )

    # TODO: improve this. Now it works only for regular lat-lon grids
    out_spec = {}
    out_spec["grid"] = list(target_grid)
    # Only keep keys relevant to the output grid spec
    gs = EkGridSpec.from_dict(out_spec)
    ymax, xmin, ymin, xmax = gs["area"]
    dy, dx = gs["grid"]
    n_lat_out = round((ymax - ymin) / dy) + 1
    n_lon_out = round((xmax - xmin) / dx) + 1

    lat_1d = np.linspace(ymin, ymax, n_lat_out)
    lon_1d = np.linspace(xmin, xmax, n_lon_out)
    lat_grid, lon_grid = np.meshgrid(lat_1d, lon_1d, indexing="ij")
    lat = lat_grid.ravel()
    lon = lon_grid.ravel()

    return tars_list, preds_list, lat, lon
