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


def data_colour_ranges(data_tars: dict, data_preds: dict, plot_chs: list[str]) -> dict[str, dict]:
    """The colour ranges of the data given, per channel.

    Parameters
    ----------
    data_tars : dict
        Forecast step → target DataArray.
    data_preds : dict
        Forecast step → prediction DataArray.
    plot_chs : list[str]
        Channels to include.

    Returns
    -------
    dict[str, dict]
        ``{channel: {"vmin", "vmax", "bias_vmax", "std_vmin", "std_vmax"}}`` over all the
        forecast steps and samples given: min / max of targets and predictions, the
        absolute maximum of prediction − target, and the min / max of the ensemble
        standard deviation (the two ``std_`` keys only when the predictions have an
        ``ens`` dimension).  Pure numbers, so the ranges of chunks of the data combine
        exactly into those of the whole (``merge_colour_ranges``).
    """
    has_ens = any("ens" in da.dims for da in data_preds.values())
    std_preds = {fs: da.std(dim="ens") for fs, da in data_preds.items()} if has_ens else {}
    ranges = {}
    for var in plot_chs:
        bias_vals = np.concatenate(
            [
                (p - t).sel(channel=var).values.ravel()
                for t, p in zip(data_tars.values(), data_preds.values(), strict=False)
            ]
        )
        ranges[var] = {
            "vmin": _extreme(data_tars, data_preds, var, "min"),
            "vmax": _extreme(data_tars, data_preds, var, "max"),
            "bias_vmax": float(max(abs(bias_vals.max()), abs(bias_vals.min()))),
        }
        if has_ens:
            ranges[var]["std_vmin"] = _extreme(std_preds, std_preds, var, "min")
            ranges[var]["std_vmax"] = _extreme(std_preds, std_preds, var, "max")
    return ranges


def _extreme(data_tars: dict, data_preds: dict, var: str, bound: str) -> float:
    """``calc_bounds`` reduced to one number, NaNs ignored."""
    values = np.concatenate([np.ravel(a) for a in calc_bounds(data_tars, data_preds, var, bound)])
    return float(np.nanmax(values) if bound == "max" else np.nanmin(values))


def merge_colour_ranges(chunks: list[dict[str, dict]]) -> dict[str, dict]:
    """The ``data_colour_ranges`` of the union of the chunks: min of the minima, max of the maxima.

    Exact, since every value is a min or a max over samples and points.  Channels and keys
    missing from some chunks are taken from the ones that have them.
    """
    merged: dict[str, dict] = {}
    for ranges in chunks:
        for channel, by_key in ranges.items():
            into = merged.setdefault(channel, {})
            for key, value in by_key.items():
                reduce = min if key.endswith("vmin") else max
                into[key] = value if key not in into else reduce(into[key], value)
    return merged


def configured_colour_range(
    stream_opts: oc.DictConfig | dict, channel: str, key: str
) -> float | None:
    """*key* (``vmin``, ``bias_vmax``, ...) of *channel* in the stream's plotting options, or None.

    An entry under the channel's own name wins over an entry under a glob pattern that
    matches it (``"tp_*"``).
    """
    entry = stream_opts.get(channel)
    if isinstance(entry, oc.DictConfig | dict) and isinstance(entry.get(key), int | float):
        return entry[key]
    for pattern, entry in stream_opts.items():
        pattern = str(pattern)
        if not any(c in pattern for c in "*?[]") or not fnmatch.fnmatch(channel, pattern):
            continue
        if isinstance(entry, oc.DictConfig | dict) and isinstance(entry.get(key), int | float):
            return entry[key]
    return None


def colour_ranges_configured(
    stream_opts: oc.DictConfig | dict, channels: list[str], has_ens: bool
) -> bool:
    """Whether the stream's plotting options fix every colour range of *channels*.

    Then the data need not be read for them.  The ensemble-std ranges count only when the
    data has an ``ens`` dimension (*has_ens*).
    """
    keys = ("vmin", "vmax", "bias_vmax") + (("std_vmin", "std_vmax") if has_ens else ())
    return all(
        configured_colour_range(stream_opts, ch, k) is not None for ch in channels for k in keys
    )


def plot_colour_ranges(
    da_tars: dict, da_preds: dict, channels: list[str], stream_opts: oc.DictConfig
) -> tuple[dict, dict, dict]:
    """Colour ranges of the maps, the bias maps and the ensemble-std maps, per channel.

    From *stream_opts* (the stream's entry of the global plotting options: ``vmin``,
    ``vmax``, ``bias_vmax``, ``std_vmin``, ``std_vmax`` per channel, or per glob pattern);
    what it lacks is computed over all forecast steps and samples of the data given.  A
    chunked run puts the ranges of the whole (run, stream) into *stream_opts*, so the
    data is not looked at here.
    """
    has_ens = any("ens" in da.dims for da in da_preds.values())
    if colour_ranges_configured(stream_opts, channels, has_ens):
        data_ranges: dict = {}
    else:
        data_ranges = data_colour_ranges(da_tars, da_preds, channels)
    range_args = (data_ranges, channels, stream_opts)
    maps_config_dict = oc.OmegaConf.to_container(common_ranges(*range_args), resolve=True)
    bias_config_dict = oc.OmegaConf.to_container(bias_ranges(*range_args), resolve=True)
    if not has_ens:
        return maps_config_dict, bias_config_dict, maps_config_dict

    std_config_dict = oc.OmegaConf.to_container(
        common_ranges(*range_args, prefix="std_"), resolve=True
    )
    for cfg in std_config_dict.values():
        if isinstance(cfg, dict):
            cfg["colormap"] = "YlOrRd"
    return maps_config_dict, bias_config_dict, std_config_dict


def common_ranges(
    data_ranges: dict[str, dict],
    plot_chs: list[str],
    global_plotting_opts_stream: oc.DictConfig,
    prefix: str = "",
) -> oc.DictConfig:
    """Colour range config per channel: ``vmin`` / ``vmax`` from the config, else from the data.

    Parameters
    ----------
    data_ranges : dict[str, dict]
        ``data_colour_ranges(...)``; only consulted for the ranges the config lacks.
    plot_chs : list[str]
        Channels to include.
    global_plotting_opts_stream : oc.DictConfig
        Existing per-stream plotting config (used as base; may already contain ranges).
    prefix : str
        ``"std_"`` for the ensemble-std maps: their ranges are the ``std_vmin`` /
        ``std_vmax`` of the config and of *data_ranges*, written as ``vmin`` / ``vmax``.

    Returns
    -------
    oc.DictConfig
        Updated config with ``vmin`` / ``vmax`` for every channel in *plot_chs*.
    """
    maps_config = global_plotting_opts_stream.copy()
    for var in plot_chs:
        if var not in maps_config:
            maps_config[var] = {}
        for bound in ("vmin", "vmax"):
            value = configured_colour_range(global_plotting_opts_stream, var, prefix + bound)
            maps_config[var][bound] = data_ranges[var][prefix + bound] if value is None else value
    return maps_config


def bias_ranges(
    data_ranges: dict[str, dict],
    plot_chs: list[str],
    global_plotting_opts_stream: oc.DictConfig,
) -> oc.DictConfig:
    """Symmetric bias colour ranges (preds − tars) per channel.

    Parameters
    ----------
    data_ranges : dict[str, dict]
        ``data_colour_ranges(...)``; its ``bias_vmax`` is used unless the config has one.
    plot_chs : list[str]
        Channels to include.
    global_plotting_opts_stream : oc.DictConfig
        Existing per-stream plotting config used as base.

    Returns
    -------
    oc.DictConfig
        Per-channel symmetric ranges (``vmin = -abs_max``, ``vmax = abs_max``).
    """
    bias_config = global_plotting_opts_stream.copy()
    for var in plot_chs:
        abs_max = configured_colour_range(global_plotting_opts_stream, var, "bias_vmax")
        if abs_max is None:
            abs_max = data_ranges[var]["bias_vmax"]
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
