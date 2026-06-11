# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Post-processing helpers for evaluation DataArrays
(channel selection, derived channels, lead-time).
"""

import logging

import numpy as np
import xarray as xr
import earthkit.regrid as ekr

from weathergen.evaluate.utils.derived_channels import DeriveChannels

_logger = logging.getLogger(__name__)


def select_channels(
    da_tar: xr.DataArray, da_pred: xr.DataArray, stream, channels, stream_cfg
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Preprocess the data by scaling z channels if needed and adding lead_time coordinate.

    Parameters
    ----------
    da_tar :
        Input DataArray to preprocess.
    da_pred :
        Input DataArray to preprocess.
    stream:
        Stream name, used to determine if z channels need to be scaled.
    channels:
        List of channels to select.
    stream_cfg:
        Stream configuration dictionary, used to determine if derived channels need to be computed.
    Returns
    -------
        Data arrays with selected channels and added derived channels if applicable.
    """
    # Ensure channel is a dimension, not a scalar coordinate (can happen after squeeze)
    if "channel" not in da_tar.dims:
        da_tar = da_tar.expand_dims("channel")
    if "channel" not in da_pred.dims:
        da_pred = da_pred.expand_dims("channel")

    assert da_pred.channel.values.tolist() == da_tar.channel.values.tolist(), (
        "Channels in prediction and target do not match."
    )

    all_channels = da_tar.channel.values.tolist()

    if set(channels) != set(all_channels):
        _logger.debug(
            f"Restricting targets and predictions to channels {channels} for stream {stream}..."
        )

        dc = DeriveChannels(
            all_channels,
            channels,
            stream_cfg,
        )

        da_tar, da_pred, channels = dc.get_derived_channels(da_tar, da_pred)

        # Verify that requested channels are available
        all_channels = da_tar.channel.values.tolist()
        missing_channels = set(channels) - set(all_channels)
        if missing_channels:
            _logger.warning(
                f"Skipping channels {missing_channels} for stream {stream}. "
                f"Not found in available channels."
            )
            channels = [ch for ch in channels if ch in all_channels]

        da_tar = da_tar.sel(channel=channels)
        da_pred = da_pred.sel(channel=channels)

    return da_tar, da_pred


def add_lead_time_coord(da: xr.DataArray, sample_dim="sample") -> xr.DataArray:
    """
    Add lead_time coordinate computed as:
    valid_time - init_times

    lead_time has dims (sample, ipoint) and dtype timedelta64[ns].

    Parameters
    ----------
    da :
        Input DataArray
    sample_dim :
        The name of the sample dimension (default is "sample") which should be kept.
        Collapse over the others.
    Returns
    -------
        Returns a DataArray with the lead_time coordinate added.

    NB. Need to be used AFTER splitting by valid_time and stacking by sample,
    so that all valid_times within a sample are the same and we can assign a
    single lead_time per sample.

    """
    vt = da["valid_time"].values
    it = da["init_times"].values
    # Compute lead_time: valid_time - init_times

    if vt.ndim > 1:
        it_expanded = it[:, np.newaxis] if it.ndim == 1 else it
        lead_time_values = vt - it_expanded
        # Get unique lead_time per sample, verify consistency
        lead_times = [
            np.unique(lead_time_values[i][~np.isnat(lead_time_values[i])])
            for i in range(lead_time_values.shape[0])
        ]
        if any(len(lt) != 1 for lt in lead_times):
            raise ValueError(
                "Inconsistent lead_time values within samples for "
                f"forecast_step {da.forecast_step.values}"
            )
        lead_time_per_sample = np.array([lt[0] for lt in lead_times])
    else:
        lead_time_values = vt - it
        lead_time_per_sample = np.unique(lead_time_values[~np.isnat(lead_time_values)])

    # Verify all samples have same lead_time for this forecast_step
    unique_lead = np.unique(lead_time_per_sample)
    if len(unique_lead) != 1:
        raise ValueError(
            "Multiple lead_time values across samples for "
            f"forecast_step {da.forecast_step.values}: {unique_lead}"
        )

    da = da.assign_coords(lead_time=unique_lead[0])
    return da

def regrid(da: xr.DataArray, regrid_opts: dict | None = None) -> xr.DataArray:
    """
    Regrid the input DataArray to a common grid defined by regrid_opts.

    The original grid type is automatically detected from the number of spatial
    points (ipoint dimension). Currently supported grids:
      - N320 (reduced Gaussian): 542,080 points
      - O96  (octahedral reduced Gaussian): 40,320 points

    Parameters
    ----------
    da :
        Input DataArray to regrid. Must have an 'ipoint' dimension.
    regrid_opts :
        Dictionary containing regridding options.
        Supported keys:
          - target_grid: target grid spec for earthkit (default: [1.5, 1.5])
        If None, no regridding is performed and the original DataArray is returned.

    Returns
    -------
        Regridded DataArray.

    Raises
    ------
    ValueError
        If the number of grid points does not match a known grid type.
    """
    if regrid_opts is None:
        return da

    # Auto-detect original grid from number of spatial points
    if regrid_opts.get("original_grid", None) is None:
        n_ipoints = da.sizes.get("ipoint", len(da.coords.get("ipoint", [])))

        # TODO: Consider more robust grid detection (e.g. from coordinates) if possible, to avoid ambiguity in case of unknown grids with same number of points.
        known_grids = {
            542080: "N320",
            40320: "O96",
        }

        original_grid = known_grids.get(n_ipoints)
        if original_grid is None:
            raise ValueError(
                f"Cannot auto-detect grid type: {n_ipoints} grid points does not match "
                f"any known grid. Please pass the grid type explicitly in the config as 'original_grid'."
            )

    target_grid = regrid_opts.get("target_grid", [1.5, 1.5])

    _logger.info(f"Regridding from {original_grid} ({n_ipoints} pts) to target grid {target_grid}...")

    return ekr.interpolate(da, {"grid": original_grid}, {"grid": target_grid})
