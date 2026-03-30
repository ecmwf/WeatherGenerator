# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""xarray post-processing helpers for channel selection, scaling, and time splitting."""

import logging

import numpy as np
import xarray as xr
from tqdm import tqdm

from weathergen.evaluate.utils.derived_channels import DeriveChannels

_logger = logging.getLogger(__name__)


def _select_channels(
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


def _scale_z_channels(data: xr.DataArray, stream: str) -> xr.DataArray:
    """
    Check scale all channels.

    Parameters
    ----------
    data :
        Input dataset
    stream :
        Stream name.
    Returns
    -------
        Returns a Dataset where channels have been scaled if needed
    """
    if stream is None or not str(stream).startswith("ERA5"):
        return data

    channels_z = [ch for ch in np.atleast_1d(data.channel.values) if str(ch).startswith("z_")]
    factor = 9.80665

    if channels_z:
        channels = data.channel.astype(str)
        mask = channels.str.startswith("z_")
        data = data.where(~mask, data / factor)
    return data


def _split_by_valid_time(arrays: list[xr.DataArray]) -> list[xr.DataArray]:
    """
    Split arrays by valid_time and stack by sample, creating separate
    arrays for each unique lead_time.

    Lead_time is calculated as: valid_time - source_interval_start

    Parameters
    ----------
    arrays : list[xr.DataArray]
        List of DataArrays, each containing multiple valid_times per sample

    Returns
    -------
    list[xr.DataArray]
        List of DataArrays, one per unique lead_time, with samples
        stacked along 'sample' dimension
    """
    # Pre-compute all lead times and build index in single pass
    lead_time_groups = {}  # lead_time -> list of (arr_idx, ipoint_indices)

    unique_valid_times = [np.unique(da.valid_time.values) for da in arrays]

    if len(unique_valid_times) == len(arrays) and all(len(uvt) == 1 for uvt in unique_valid_times):
        _logger.debug(
            "All arrays have a single unique valid_time. Skipping splitting by valid_time."
        )
        arrays = _force_consistent_grids(arrays)

        return [arrays]

    for arr_idx, da in tqdm(enumerate(arrays), total=len(arrays), desc="Splitting by valid time"):
        vt = da.valid_time.values
        sis = da.source_interval_start.values

        # Calculate lead_time once
        if vt.ndim > 1:
            lead_times = vt - (sis[:, np.newaxis] if sis.ndim == 1 else sis)
            # Flatten and get unique lead times with their ipoint indices
            valid_mask = ~np.isnat(lead_times)
            for i in range(lead_times.shape[0]):
                row_leads = lead_times[i][valid_mask[i]]
                row_ipoints = np.where(valid_mask[i])[0]
                for lead, ipoint in zip(row_leads, row_ipoints, strict=False):
                    lead_time_groups.setdefault(lead, []).append((arr_idx, i, ipoint))
        else:
            lead_times = vt - sis
            valid_mask = ~np.isnat(lead_times)
            valid_leads = lead_times[valid_mask]
            valid_ipoints = np.where(valid_mask)[0]
            for lead, ipoint in zip(valid_leads, valid_ipoints, strict=False):
                lead_time_groups.setdefault(lead, []).append((arr_idx, 0, ipoint))

    # Get reference grid from first array for alignment
    ref_lat = arrays[0].lat.values
    ref_lon = arrays[0].lon.values
    ref_sort_idx = np.lexsort((ref_lon, ref_lat))
    ref_lat_sorted = ref_lat[ref_sort_idx]
    ref_lon_sorted = ref_lon[ref_sort_idx]

    # Process each lead time
    sorted_leads = sorted(lead_time_groups.keys())
    out = []

    for forecast_step, lead in enumerate(sorted_leads, start=1):
        # Group by array index to minimize selections
        array_groups = {}
        for arr_idx, sample_idx, ipoint in lead_time_groups[lead]:
            array_groups.setdefault(arr_idx, {}).setdefault(sample_idx, []).append(ipoint)

        per_sample = []
        for arr_idx, sample_dict in array_groups.items():
            da = arrays[arr_idx]

            for sample_idx, ipoint_list in sample_dict.items():
                # Single selection operation
                ipoint_arr = np.array(ipoint_list)
                da_subset = da.isel(ipoint=ipoint_arr)

                # Align to reference grid
                sort_idx = np.lexsort((da_subset.lon.values, da_subset.lat.values))
                da_subset = da_subset.isel(ipoint=sort_idx).assign_coords(
                    ipoint=np.arange(len(ipoint_arr)),
                    lat=("ipoint", ref_lat_sorted[: len(ipoint_arr)]),
                    lon=("ipoint", ref_lon_sorted[: len(ipoint_arr)]),
                )

                # Ensure sample dimension
                if "sample" not in da_subset.dims:
                    sample_val = da.sample.values.item() if da.sample.ndim == 0 else sample_idx
                    da_subset = da_subset.expand_dims(sample=[sample_val])

                per_sample.append(da_subset)

        if per_sample:
            # Single concat operation
            combined = xr.concat(per_sample, dim="sample", coords="different", compat="equals")
            combined = combined.assign_coords(
                ipoint=np.arange(combined.sizes["ipoint"]), forecast_step=forecast_step
            )
            out.append(combined)

    return out


def _add_lead_time_coord(da: xr.DataArray, sample_dim="sample") -> xr.DataArray:
    """
    Add lead_time coordinate computed as:
    valid_time - source_interval_start

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
    sis = da["source_interval_start"].values
    # Compute lead_time: valid_time - source_interval_start

    if vt.ndim > 1:
        sis_expanded = sis[:, np.newaxis] if sis.ndim == 1 else sis
        lead_time_values = vt - sis_expanded
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
        lead_time_values = vt - sis
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


def _force_consistent_grids(ref: list[xr.DataArray]) -> xr.DataArray:
    """
    Force all samples to share the same ipoint order.

    This function aligns the spatial ordering (lat/lon/ipoint) of all samples
    to that of the first sample, ensuring consistent spatial coordinates for
    subsequent concatenation. It is essential for regular-grid (gridded) data
    where spatial order matters but may differ across samples.

    Parameters
    ----------
    ref: list[xr.DataArray]
        List of xarray DataArrays, each representing one sample. Must have at least one element.

    Returns
    -------
    xr.DataArray
        A concatenated DataArray across the 'sample' dimension, where each sample's ipoint indices
        have been reordered to match the sorted lat/lon order of the first sample.

    Notes
    -----
    - All input DataArrays must share identical lat/lon values
        (though possibly in different orders).
    - Enforces consistent ipoint indexing after alignment (0..N-1).
    - Preserves and aligns all other coordinates and data variables.
    """
    assert len(ref) > 0, "_force_consistent_grids requires at least one input DataArray in 'ref'."

    # Pick first sample as reference
    ref_lat = ref[0].lat
    ref_lon = ref[0].lon

    sort_idx = np.lexsort((ref_lon.values, ref_lat.values))
    npoints = sort_idx.size
    aligned = []
    samples = []
    for i, a in enumerate(ref):
        a_sorted = a.isel(ipoint=sort_idx)
        samples.append(a_sorted.sample.values)
        a_sorted = a_sorted.assign_coords(
            ipoint=np.arange(npoints),
            lat=("ipoint", ref_lat.values[sort_idx]),
            lon=("ipoint", ref_lon.values[sort_idx]),
        )

        if "sample" not in a_sorted.dims:
            a_sorted = a_sorted.expand_dims(sample=[i])

        aligned.append(a_sorted)

    return xr.concat(aligned, dim="sample", coords="different", compat="equals")
