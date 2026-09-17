# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Module-level loky worker functions and direct zarr I/O helpers.

These are module-level (not class-bound) so they are pickable and can be
dispatched to loky / ProcessPoolExecutor workers.
"""

import contextlib
import logging

import numpy as np
import zarr
from numpy.typing import NDArray

from weathergen.evaluate.utils.derived_channels import is_derivable_channel

_logger = logging.getLogger(__name__)


def _compute_early_channel_selection(
    read_channels: list[str],
    requested_channels: list[str],
    stream_cfg: dict,
) -> tuple[list[int] | None, list[str]]:
    """Compute channel indices for early selection in _read_sample.

    When the stream does NOT use derived channels, we can select only the
    requested channels at the numpy level inside each worker, avoiding
    the transfer and stacking of unrequested channels.

    Parameters
    ----------
    read_channels : list[str]
        Full list of channels available in the zarr store.
    requested_channels : list[str]
        Channels the user requested for evaluation/plotting.
    stream_cfg : dict
        Stream configuration (checked for ``derive_channels`` key).

    Returns
    -------
    channel_idxs : list[int] | None
        Indices into ``read_channels`` to select, or ``None`` to read all.
    effective_channels : list[str]
        Channel names that will be returned (subset or full list).
    """
    # If derived channels are configured, or any requested channel is
    # auto-derivable (e.g. 10ff), we must keep ALL channels so that the
    # derivation logic in _select_channels has its source data.
    if "derive_channels" in stream_cfg:
        return None, read_channels

    if any(is_derivable_channel(ch) for ch in requested_channels):
        return None, read_channels

    # Find the intersection: requested channels that exist in the zarr store
    available_set = set(read_channels)
    needed = [ch for ch in requested_channels if ch in available_set]

    # If all channels are requested anyway, or the intersection is empty,
    # skip early selection.
    if not needed or len(needed) == len(read_channels):
        return None, read_channels

    # Build index list preserving zarr order for stable indexing
    chan_to_idx = {ch: i for i, ch in enumerate(read_channels)}
    idxs = sorted(chan_to_idx[ch] for ch in needed)
    effective = [read_channels[i] for i in idxs]

    _logger.debug(
        f"Early channel selection: {len(effective)}/{len(read_channels)} channels "
        f"({', '.join(effective[:5])}{'...' if len(effective) > 5 else ''})"
    )
    return idxs, effective


def _read_sample(
    zarr_path: str,
    sample: int,
    stream: str,
    fsteps: list[int],
    channel_idxs: list[int],
    is_zip: bool,
    read_coords: bool = False,
    is_gridded: bool = True,
) -> tuple[list[NDArray], list[NDArray], list[NDArray], dict]:
    """
    Read all forecast steps for one sample via direct zarr array access.

    Bypasses ZarrIO / OutputDataset / as_xarray / dask for maximum speed.
    Each worker opens its own zarr store handle (safe for both ZipStore and
    LocalStore).

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store (.zarr directory or .zip file).
    sample : int
        Sample index to read.
    stream : str
        Stream name (e.g. "ERA5").
    fsteps : list[int]
        Forecast steps to read.
    channel_idxs : list[int]
        Pre-computed indices into the channel axis (select only needed channels).
    is_zip : bool
        Whether the store is a ZipStore (.zip).
    read_coords : bool
        If True, also read per-sample coords (needed for scatter/non-gridded data).
    is_gridded : bool
        If True, split by unique valid_times to create sub-steps (gridded data
        with multiple forecast sub-steps per fstep).  If False (scatter/obs data),
        keep all observations in a single array per fstep — each observation has
        its own time and splitting would create one array per observation.

    Returns
    -------
    preds_all : list[np.ndarray]
        Per-fstep prediction arrays, shape (ipoints, channels[, ens]).
        For sub-steps, multiple arrays per fstep (one per unique valid_time).
    targets_all : list[np.ndarray]
        Per-fstep target arrays, shape (ipoints, channels).
    times_all : list[np.ndarray]
        Per-fstep time arrays (unique times per entry).
    meta : dict
        Metadata: {"source_interval": ..., "n_substeps": list[int],
                    "coords": list[np.ndarray | None]} where coords has
                    one entry per fstep (each may be None or shape (n_ip, 2)).
    """
    if is_zip:
        store = zarr.storage.ZipStore(zarr_path, mode="r")
        ds = zarr.open_group(store=store, mode="r")
    else:
        store = zarr.storage.LocalStore(zarr_path)
        ds = zarr.open_group(store=store, mode="r")

    preds_all, targets_all, times_all = [], [], []
    n_substeps = []  # track how many sub-steps per fstep
    source_interval = None

    # Recover the source interval once per sample. Older outputs store source
    # timestamps under ``.../0/source/times``; newer ones persist the same
    # metadata in the target/prediction group attrs.
    try:
        source_times = np.asarray(ds[f"{sample}/{stream}/0/source/times"])
        source_window_start = str(np.min(source_times))
        source_window_end = str(np.max(source_times))
        source_interval = {"start": source_window_start, "end": source_window_end}
    except (KeyError, AttributeError):
        source_interval = {}
        for fs in fsteps:
            base = f"{sample}/{stream}/{fs}"
            for group_name in ("target", "prediction"):
                try:
                    group = ds[f"{base}/{group_name}"]
                except KeyError:
                    continue

                source_interval_attr = group.attrs.get("source_interval")
                if source_interval_attr:
                    source_interval = dict(source_interval_attr)
                    break
            if source_interval:
                break

    for fs in fsteps:
        base = f"{sample}/{stream}/{fs}"

        # Direct array access — bypasses OutputDataset/as_xarray/dask entirely
        pred_data = np.asarray(ds[f"{base}/prediction/data"])
        target_data = np.asarray(ds[f"{base}/target/data"])
        times_data = np.asarray(ds[f"{base}/prediction/times"])

        # Select channels by index
        if channel_idxs is not None:
            pred_data = (
                pred_data[:, channel_idxs] if pred_data.ndim == 2 else pred_data[:, channel_idxs, :]
            )
            target_data = target_data[:, channel_idxs]

        # Handle sub-steps (gridded data with multiple valid_times per fstep).
        # For scatter/observation data each observation has its own timestamp,
        # so splitting by unique time would create one tiny array per obs —
        # thousands of them — causing the assembly code to hang.
        unique_times = np.unique(times_data)
        if is_gridded and len(unique_times) > 1:
            count = 0
            for ut in unique_times:
                mask = times_data == ut
                preds_all.append(pred_data[mask])
                targets_all.append(target_data[mask])
                count += 1
            times_all.append(unique_times)
            n_substeps.append(count)
        else:
            preds_all.append(pred_data)
            targets_all.append(target_data)
            # For scatter data, keep the full per-observation times array
            # so the DataArray builder can assign per-ipoint valid_time.
            # For gridded data with 1 unique time, unique_times suffices.
            if not is_gridded:
                times_all.append(times_data)
            else:
                times_all.append(unique_times)
            n_substeps.append(1)

    # Optionally read per-fstep coordinates (for scatter / non-gridded data).
    # Each fstep can have a different number of observations, so we read
    # the coordinate array for every fstep rather than just the first one.
    per_fstep_coords: list[NDArray | None] = []
    if read_coords and fsteps:
        for fs in fsteps:
            try:
                base_c = f"{sample}/{stream}/{fs}"
                per_fstep_coords.append(np.asarray(ds[f"{base_c}/prediction/coords"]))
            except (KeyError, AttributeError):
                per_fstep_coords.append(None)
    else:
        per_fstep_coords = [None] * len(fsteps)

    with contextlib.suppress(Exception):
        store.close()

    meta = {
        "source_interval": source_interval,
        "n_substeps": n_substeps,
        "coords": per_fstep_coords,
    }
    return preds_all, targets_all, times_all, meta


def _read_coords_and_meta(
    zarr_path: str,
    stream: str,
    fstep: int,
    is_zip: bool,
) -> tuple[NDArray, list[str], NDArray]:
    """
    Read coordinates and channel names from the zarr store (once).

    Returns
    -------
    coords : np.ndarray, shape (ipoints, 2) — lat, lon
    channels : list[str] — all channel names from zarr
    times_ref : np.ndarray — reference times from sample 0
    """
    if is_zip:
        store = zarr.storage.ZipStore(zarr_path, mode="r")
        ds = zarr.open_group(store=store, mode="r")
    else:
        store = zarr.storage.LocalStore(zarr_path)
        ds = zarr.open_group(store=store, mode="r")

    base = f"0/{stream}/{fstep}"
    coords = np.asarray(ds[f"{base}/prediction/coords"])
    times_ref = np.asarray(ds[f"{base}/prediction/times"])

    # Read channel names from group attributes
    pred_group = ds[f"{base}/prediction"]
    channels = list(pred_group.attrs.get("channels", []))

    with contextlib.suppress(Exception):
        store.close()

    return coords, channels, times_ref
