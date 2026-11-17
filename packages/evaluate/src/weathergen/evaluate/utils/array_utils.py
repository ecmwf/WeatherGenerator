# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Array / DataArray utility functions: range computation, coordinate helpers."""

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


def select_samples(da: xr.DataArray, samples: list) -> xr.DataArray:
    """The *samples* of a DataArray: along the ``sample`` dim of gridded data, or the points
    tagged with them when ``sample`` is a coordinate on ``ipoint`` (scatter data)."""
    if "sample" in da.dims:
        return da.sel(sample=samples)
    return da.where(da.sample.isin(samples), drop=True)


def select_ensemble_data(da: xr.DataArray, ens):
    """Return ensemble member, mean or std view of a DataArray."""
    if "ens" not in da.dims:
        return da

    if ens == "mean":
        return da.mean(dim="ens")

    if ens == "std":
        return da.std(dim="ens")

    return da.sel(ens=ens)
