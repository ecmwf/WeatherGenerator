# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Array / DataArray utility functions: range computation, coordinate helpers."""

import fnmatch

import numpy as np
import omegaconf as oc
import xarray as xr


def calc_val(x: xr.DataArray, bound: str) -> list[float]:
    """Return per-variable max or min values across ipoints.

    Parameters
    ----------
    x : xr.DataArray
        DataArray with an ``ipoint`` dimension.
    bound : str
        ``"max"`` or ``"min"``.
    """
    if "ipoint" not in x.dims or x.sizes["ipoint"] == 0:
        return np.array([np.nan])
    if bound == "max":
        return x.max(dim="ipoint").values
    elif bound == "min":
        return x.min(dim="ipoint").values
    else:
        raise ValueError("bound must be either 'max' or 'min'")


def calc_bounds(data_tars, data_preds, var, bound):
    """Compute bound values across all fsteps for both targets and predictions.

    Parameters
    ----------
    data_tars : dict
        Forecast step → target DataArray.
    data_preds : dict
        Forecast step → prediction DataArray.
    var : str
        Channel / variable name.
    bound : str
        ``"max"`` or ``"min"``.

    Returns
    -------
    list
        Per-fstep bound values.
    """
    list_bound = []
    for da_tars, da_preds in zip(data_tars.values(), data_preds.values(), strict=False):
        list_bound.extend(
            (
                calc_val(da_tars.where(da_tars.channel == var, drop=True), bound),
                calc_val(da_preds.where(da_preds.channel == var, drop=True), bound),
            )
        )
    return list_bound


def common_ranges(
    data_tars: dict,
    data_preds: dict,
    plot_chs: list[str],
    global_plotting_opts_stream: oc.DictConfig,
) -> oc.DictConfig:
    """Calculate common colour ranges per variable across all fsteps.

    Parameters
    ----------
    data_tars : dict
        Forecast step → target DataArray.
    data_preds : dict
        Forecast step → prediction DataArray.
    plot_chs : list[str]
        Variables to include.
    global_plotting_opts_stream : oc.DictConfig
        Existing per-stream plotting config (used as base; may already contain ranges).

    Returns
    -------
    oc.DictConfig
        Updated config with ``vmin`` / ``vmax`` for every variable in *plot_chs*.
    """
    maps_config = global_plotting_opts_stream.copy()
    for var in plot_chs:
        if var not in maps_config:
            maps_config[var] = {}
        # override empty bounds with matching glob bounds from config
        for key, value in maps_config.items():
            if not isinstance(value, oc.DictConfig | dict):
                continue
            k = str(key)
            if any(c in k for c in "*?[]") and fnmatch.fnmatch(var, k):
                for bound in ("vmin", "vmax"):
                    if isinstance(value.get(bound), int | float):
                        maps_config[var].setdefault(bound, value[bound])
        # if vmax still missing, compute bound from data
        if not isinstance(maps_config[var].get("vmax"), (int | float)):
            list_max = calc_bounds(data_tars, data_preds, var, "max")
            list_max = np.concatenate([arr.flatten() for arr in list_max])
            list_max = list_max[np.isfinite(list_max)]
            if list_max.size > 0:
                maps_config[var].update({"vmax": float(list_max.max())})
        # if vmin still missing, compute bound from data
        if not isinstance(maps_config[var].get("vmin"), (int | float)):
            list_min = calc_bounds(data_tars, data_preds, var, "min")
            list_min = np.concatenate([arr.flatten() for arr in list_min])
            list_min = list_min[np.isfinite(list_min)]
            if list_min.size > 0:
                maps_config[var].update({"vmin": float(list_min.min())})
    return maps_config


def bias_ranges(
    data_tars: dict,
    data_preds: dict,
    plot_chs: list[str],
    global_plotting_opts_stream: oc.DictConfig,
) -> oc.DictConfig:
    """Calculate symmetric bias colour ranges (preds − tars) per variable.

    Parameters
    ----------
    data_tars : dict
        Forecast step → target DataArray.
    data_preds : dict
        Forecast step → prediction DataArray.
    plot_chs : list[str]
        Variables to include.
    global_plotting_opts_stream : oc.DictConfig
        Existing per-stream plotting config used as base.

    Returns
    -------
    oc.DictConfig
        Per-variable symmetric ranges (``vmin = -abs_max``, ``vmax = abs_max``).
    """
    import logging

    _logger = logging.getLogger(__name__)

    bias_config = global_plotting_opts_stream.copy()
    for var in plot_chs:
        bias_vals = [
            (p - t).sel(channel=var).values
            for t, p in zip(data_tars.values(), data_preds.values(), strict=False)
        ]
        all_bias = np.concatenate([v.flatten() for v in bias_vals])
        all_bias = all_bias[np.isfinite(all_bias)]
        if all_bias.size == 0:
            continue
        abs_p = float(np.percentile(np.abs(all_bias), 98))
        bias_config.update({var: {"vmax": abs_p, "vmin": -abs_p}})
        _logger.info(f"bias_ranges: {var} → vmin={-abs_p:.4f}, vmax={abs_p:.4f}")
    return bias_config


def scalar_coord_to_dim(da: xr.DataArray, name: str, axis: int = -1) -> xr.DataArray:
    """Promote a scalar coordinate to a dimension in *da*.

    If *name* is already a dimension, *da* is returned unchanged.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray.
    name : str
        Coordinate name to promote.
    axis : int
        Axis along which to insert the new dimension (default ``-1``).

    Returns
    -------
    xr.DataArray
        DataArray with *name* as a dimension (size 1).
    """
    if name in da.dims:
        return da
    if name in da.coords and da.coords[name].ndim == 0:
        val = da.coords[name].item()
        da = da.drop_vars(name)
        da = da.expand_dims({name: [val]}, axis=axis)
    return da
