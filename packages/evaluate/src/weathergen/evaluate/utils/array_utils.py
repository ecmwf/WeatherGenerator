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


def colour_bounds(data_tars: dict, data_preds: dict, plot_chs: list[str]) -> dict[str, dict]:
    """The data bounds the map colour ranges are built from, per variable.

    Parameters
    ----------
    data_tars : dict
        Forecast step → target DataArray.
    data_preds : dict
        Forecast step → prediction DataArray.
    plot_chs : list[str]
        Variables to include.

    Returns
    -------
    dict[str, dict]
        ``{var: {"vmin", "vmax", "bias_vmax", "std_vmin", "std_vmax"}}`` over all the
        forecast steps and samples given: min / max of targets and predictions, the
        absolute maximum of prediction − target, and the min / max of the ensemble
        standard deviation (the two ``std_`` keys only when the predictions have an
        ``ens`` dimension).  Pure numbers, so the bounds of chunks of the data combine
        exactly (``merge_bounds``).
    """
    has_ens = any("ens" in da.dims for da in data_preds.values())
    std_preds = {fs: da.std(dim="ens") for fs, da in data_preds.items()} if has_ens else {}
    bounds = {}
    for var in plot_chs:
        bias_vals = np.concatenate(
            [
                (p - t).sel(channel=var).values.ravel()
                for t, p in zip(data_tars.values(), data_preds.values(), strict=False)
            ]
        )
        bounds[var] = {
            "vmin": _extreme(data_tars, data_preds, var, "min"),
            "vmax": _extreme(data_tars, data_preds, var, "max"),
            "bias_vmax": float(max(abs(bias_vals.max()), abs(bias_vals.min()))),
        }
        if has_ens:
            bounds[var]["std_vmin"] = _extreme(std_preds, std_preds, var, "min")
            bounds[var]["std_vmax"] = _extreme(std_preds, std_preds, var, "max")
    return bounds


def _extreme(data_tars: dict, data_preds: dict, var: str, bound: str) -> float:
    """``calc_bounds`` reduced to one number, NaNs ignored."""
    values = np.concatenate([np.ravel(a) for a in calc_bounds(data_tars, data_preds, var, bound)])
    return float(np.nanmax(values) if bound == "max" else np.nanmin(values))


def configured_bound(stream_opts: oc.DictConfig | dict, var: str, key: str) -> float | None:
    """*key* (``vmin``, ``bias_vmax``, ...) of *var* in the stream's plotting options, or None.

    An entry under the variable's own name wins over an entry under a glob pattern that
    matches it (``"tp_*"``).
    """
    entry = stream_opts.get(var)
    if isinstance(entry, oc.DictConfig | dict) and isinstance(entry.get(key), int | float):
        return entry[key]
    for pattern, entry in stream_opts.items():
        pattern = str(pattern)
        if not any(c in pattern for c in "*?[]") or not fnmatch.fnmatch(var, pattern):
            continue
        if isinstance(entry, oc.DictConfig | dict) and isinstance(entry.get(key), int | float):
            return entry[key]
    return None


def common_ranges(
    bounds: dict[str, dict],
    plot_chs: list[str],
    global_plotting_opts_stream: oc.DictConfig,
    prefix: str = "",
) -> oc.DictConfig:
    """Colour range config per variable: ``vmin`` / ``vmax`` from the config, else from *bounds*.

    Parameters
    ----------
    bounds : dict[str, dict]
        ``colour_bounds(...)`` of the data; only consulted for the bounds the config lacks.
    plot_chs : list[str]
        Variables to include.
    global_plotting_opts_stream : oc.DictConfig
        Existing per-stream plotting config (used as base; may already contain ranges).
    prefix : str
        ``"std_"`` for the ensemble-std maps: their bounds are the ``std_vmin`` /
        ``std_vmax`` of the config and of *bounds*, written as ``vmin`` / ``vmax``.

    Returns
    -------
    oc.DictConfig
        Updated config with ``vmin`` / ``vmax`` for every variable in *plot_chs*.
    """
    maps_config = global_plotting_opts_stream.copy()
    for var in plot_chs:
        if var not in maps_config:
            maps_config[var] = {}
        for bound in ("vmin", "vmax"):
            value = configured_bound(global_plotting_opts_stream, var, prefix + bound)
            maps_config[var][bound] = bounds[var][prefix + bound] if value is None else value
    return maps_config


def bias_ranges(
    bounds: dict[str, dict],
    plot_chs: list[str],
    global_plotting_opts_stream: oc.DictConfig,
) -> oc.DictConfig:
    """Symmetric bias colour ranges (preds − tars) per variable.

    Parameters
    ----------
    bounds : dict[str, dict]
        ``colour_bounds(...)`` of the data; its ``bias_vmax`` is used unless the config has one.
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
        abs_max = configured_bound(global_plotting_opts_stream, var, "bias_vmax")
        if abs_max is None:
            abs_max = bounds[var]["bias_vmax"]
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
