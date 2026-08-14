# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Array / DataArray utility functions: range computation, coordinate helpers."""

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
        if var in maps_config:
            if not isinstance(maps_config[var].get("vmax"), (int | float)):
                list_max = calc_bounds(data_tars, data_preds, var, "max")
                list_max = np.concatenate([arr.flatten() for arr in list_max]).tolist()
                maps_config[var].update({"vmax": float(max(list_max))})
            if not isinstance(maps_config[var].get("vmin"), (int | float)):
                list_min = calc_bounds(data_tars, data_preds, var, "min")
                list_min = np.concatenate([arr.flatten() for arr in list_min]).tolist()
                maps_config[var].update({"vmin": float(min(list_min))})
        else:
            list_max = calc_bounds(data_tars, data_preds, var, "max")
            list_max = np.concatenate([arr.flatten() for arr in list_max]).tolist()
            list_min = calc_bounds(data_tars, data_preds, var, "min")
            list_min = np.concatenate([arr.flatten() for arr in list_min]).tolist()
            maps_config.update({var: {"vmax": float(max(list_max)), "vmin": float(min(list_min))}})
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
    bias_config = global_plotting_opts_stream.copy()
    for var in plot_chs:
        bias_vals = [
            (p - t).sel(channel=var).values
            for t, p in zip(data_tars.values(), data_preds.values(), strict=False)
        ]
        abs_max = float(
            max(abs(np.concatenate(bias_vals).max()), abs(np.concatenate(bias_vals).min()))
        )
        bias_config.update({var: {"vmax": abs_max, "vmin": -abs_max}})
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
