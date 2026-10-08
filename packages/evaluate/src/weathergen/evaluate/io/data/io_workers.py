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
import threading
from collections import OrderedDict

import anemoi.datasets
import numpy as np
import zarr
from numpy.typing import NDArray

from weathergen.evaluate.utils.derived_channels import is_derivable_channel

_logger = logging.getLogger(__name__)

# Module-level cache so that threading workers reuse the same anemoi dataset
# handle across tasks dispatched to the same process.
_anemoi_cache: dict[str, tuple] = {}
_anemoi_lock = threading.Lock()

# Bounded LRU cache of already-read (and channel-selected) anemoi date slices,
# keyed by (filename, date_idx).  Consecutive forecast steps share valid times,
# so caching avoids re-reading the same full-grid slice (~hundreds of MB each)
# many times.  The bound keeps peak memory in check.
_ANEMOI_SLICE_CACHE_MAXSIZE = 8
_anemoi_slice_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
_anemoi_slice_lock = threading.Lock()


def _get_anemoi_date_slice(
    ds,
    filename: str,
    date_idx: int,
    target_idx: list[int],
) -> np.ndarray:
    """Return the ``(n_gridpoints, n_target_channels)`` slice for one date.

    Uses a bounded module-level LRU cache so repeated requests for the same
    anemoi date (across overlapping forecast steps) avoid re-reading the
    full grid from disk.
    """
    key = (filename, date_idx)
    with _anemoi_slice_lock:
        cached = _anemoi_slice_cache.get(key)
        if cached is not None:
            _anemoi_slice_cache.move_to_end(key)
            return cached

    # Read outside the lock (I/O bound); tolerate a duplicate concurrent read.
    # ds[idx] → (n_variables, n_ens, n_gridpoints)
    data_slice = np.asarray(ds[date_idx])[target_idx, 0, :].T.astype(np.float32)

    with _anemoi_slice_lock:
        _anemoi_slice_cache[key] = data_slice
        _anemoi_slice_cache.move_to_end(key)
        while len(_anemoi_slice_cache) > _ANEMOI_SLICE_CACHE_MAXSIZE:
            _anemoi_slice_cache.popitem(last=False)
    return data_slice


def _open_anemoi_dataset(anemoi_cfg: dict) -> tuple:
    """Open the anemoi dataset once and precompute target channel indices.

    Uses a module-level cache keyed by filename so that threading workers
    reuse the same handle. A lock ensures only one thread opens the dataset.

    Returns ``(ds, target_idx, ds_dates, ds_lats, ds_lons)`` for reuse across
    forecast steps.  ``ds_lats``/``ds_lons`` are the dataset grid coordinates
    (longitudes normalised to ``[-180, 180)``) used for nearest-neighbour
    matching to the prediction grid points.
    """
    filename = anemoi_cfg["filename"]
    if filename in _anemoi_cache:
        return _anemoi_cache[filename]

    with _anemoi_lock:
        # Double-check after acquiring lock
        if filename in _anemoi_cache:
            return _anemoi_cache[filename]

        target_channels = anemoi_cfg["channels"]

        _logger.info(f"Opening anemoi dataset {filename} (lazy)")
        # Pre-select only the target variables so each read materialises just
        # those channels (not all ~113), cutting per-read memory ~4x.  After
        # ``select`` the variables are returned in the requested order, so the
        # target index is simply 0..len(channels)-1.
        try:
            ds = anemoi.datasets.open_dataset(filename, select=list(target_channels))
            # ``select`` reduces the variable axis; re-index against the
            # post-select variable order (do not assume it preserves the
            # requested order).
            target_idx = [ds.variables.index(ch) for ch in target_channels]
        except Exception as exc:
            _logger.warning(
                f"Anemoi 'select' failed ({type(exc).__name__}: {exc}); "
                f"falling back to full-variable read."
            )
            ds = anemoi.datasets.open_dataset(filename)
            target_idx = [ds.variables.index(ch) for ch in target_channels]
        ds_dates = ds.dates.astype("datetime64[s]")
        ds_lats = np.asarray(ds.latitudes, dtype=np.float64)
        # Normalise dataset longitudes to [-180, 180) to match prediction coords.
        ds_lons = ((np.asarray(ds.longitudes, dtype=np.float64) + 180.0) % 360.0) - 180.0
        result = (ds, target_idx, ds_dates, ds_lats, ds_lons)
        _anemoi_cache[filename] = result
        return result


# Cache of prediction-coord → anemoi grid-index mappings, keyed by
# (filename, coords bytes) so repeated fsteps reuse the nearest-neighbour result.
_anemoi_index_cache: dict[tuple, np.ndarray] = {}


def _match_anemoi_indices(
    ds_lats: np.ndarray,
    ds_lons: np.ndarray,
    pred_coords: np.ndarray,
    filename: str,
) -> np.ndarray:
    """Find the nearest anemoi grid index for each prediction coordinate.

    Parameters
    ----------
    ds_lats, ds_lons : np.ndarray
        Anemoi grid latitudes and longitudes (longitudes in [-180, 180)).
    pred_coords : np.ndarray, shape (n_pred, 2)
        Prediction (lat, lon) coordinates.
    filename : str
        Anemoi dataset filename (for caching).

    Returns
    -------
    np.ndarray, shape (n_pred,)
        Index into the anemoi grid for each prediction point.
    """
    key = (filename, pred_coords.tobytes())
    cached = _anemoi_index_cache.get(key)
    if cached is not None:
        return cached

    pred_lat = pred_coords[:, 0].astype(np.float64)
    # Normalise prediction longitudes to [-180, 180) as well.
    pred_lon = ((pred_coords[:, 1].astype(np.float64) + 180.0) % 360.0) - 180.0

    # Great-circle-ish nearest neighbour via chord distance on the unit sphere.
    deg2rad = np.pi / 180.0
    ds_lat_r = ds_lats * deg2rad
    ds_lon_r = ds_lons * deg2rad
    ds_xyz = np.stack(
        [
            np.cos(ds_lat_r) * np.cos(ds_lon_r),
            np.cos(ds_lat_r) * np.sin(ds_lon_r),
            np.sin(ds_lat_r),
        ],
        axis=1,
    )

    pred_lat_r = pred_lat * deg2rad
    pred_lon_r = pred_lon * deg2rad
    pred_xyz = np.stack(
        [
            np.cos(pred_lat_r) * np.cos(pred_lon_r),
            np.cos(pred_lat_r) * np.sin(pred_lon_r),
            np.sin(pred_lat_r),
        ],
        axis=1,
    )

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(ds_xyz)
        _, indices = tree.query(pred_xyz, k=1)
        indices = np.asarray(indices, dtype=np.int64)
    except ImportError:
        # Fallback: brute-force nearest neighbour (fine for small n_pred).
        indices = np.empty(pred_xyz.shape[0], dtype=np.int64)
        for i in range(pred_xyz.shape[0]):
            d = np.sum((ds_xyz - pred_xyz[i]) ** 2, axis=1)
            indices[i] = int(np.argmin(d))

    _anemoi_index_cache[key] = indices
    return indices


def _read_anemoi_target(
    ds,
    target_idx: list[int],
    ds_dates: np.ndarray,
    times: np.ndarray,
    channel_idxs: list[int] | None,
    grid_indices: np.ndarray | None = None,
    filename: str | None = None,
) -> np.ndarray:
    """Read target data from a pre-opened anemoi dataset for the given valid times.

    Parameters
    ----------
    ds : anemoi dataset handle
        Already-opened dataset (from ``_open_anemoi_dataset``).
    target_idx : list[int]
        Indices of val_target_channels within ``ds.variables``.
    ds_dates : np.ndarray
        ``ds.dates`` cast to ``datetime64[s]`` (precomputed).
    times : np.ndarray
        Valid times (datetime64) to extract from the dataset.
    channel_idxs : list[int] | None
        Channel indices to select (applied after reading).
    grid_indices : np.ndarray | None
        If provided, select only these grid-point indices from the full
        anemoi grid so the target matches the prediction grid points.

    Returns
    -------
    np.ndarray
        Target data array, shape ``(n_points, n_channels)``, float32.
    """
    unique_times = np.unique(times)

    # Handle empty times (some fsteps have no data)
    if len(unique_times) == 0:
        n_ch = len(target_idx)
        if channel_idxs is not None:
            n_ch = len(channel_idxs)
        return np.empty((0, n_ch), dtype=np.float32)

    # Read one time-slice at a time (typically ≤6 per fstep)
    if grid_indices is not None:
        # Row-aligned assembly: each prediction point has its own valid time
        # (``times``) and its own anemoi grid index (``grid_indices``).  Build
        # a target with exactly one row per prediction point.
        n_pred = grid_indices.shape[0]
        n_ch_full = len(target_idx)
        target_data = np.empty((n_pred, n_ch_full), dtype=np.float32)

        # Resolve the anemoi date index for each unique valid time.
        ut_to_idx: dict = {}
        for ut in unique_times:
            ut_s = np.datetime64(ut, "s")
            matches = np.where(ds_dates == ut_s)[0]
            if len(matches) == 0:
                raise ValueError(
                    f"Time {ut} not found in anemoi dataset. "
                    f"Available range: {ds_dates[0]} .. {ds_dates[-1]}"
                )
            ut_to_idx[ut] = int(matches[0])

        # Batch-read a contiguous span of dates in one call.  The anemoi zarr
        # stores one date per chunk, so a slice read decompresses all chunks
        # in parallel — much faster than many single-date reads.  Only batch
        # when the span is compact to avoid pulling unrelated dates.
        idxs = sorted(ut_to_idx.values())
        span = idxs[-1] - idxs[0] + 1
        date_slices: dict = {}
        if len(idxs) > 1 and span <= 2 * len(idxs):
            lo = idxs[0]
            block = np.asarray(ds[lo : idxs[-1] + 1])  # (span, n_var, n_ens, n_grid)
            for ut, idx in ut_to_idx.items():
                date_slices[ut] = block[idx - lo][target_idx, 0, :].T.astype(np.float32)
        else:
            for ut, idx in ut_to_idx.items():
                date_slices[ut] = _get_anemoi_date_slice(ds, filename, idx, target_idx)

        for ut in unique_times:
            data_slice = date_slices[ut]
            # Rows of the prediction that share this valid time.
            row_mask = times == ut
            target_data[row_mask] = data_slice[grid_indices[row_mask]]
    else:
        all_data = []
        for ut in unique_times:
            ut_s = np.datetime64(ut, "s")
            matches = np.where(ds_dates == ut_s)[0]

            if len(matches) == 0:
                raise ValueError(
                    f"Time {ut} not found in anemoi dataset. "
                    f"Available range: {ds_dates[0]} .. {ds_dates[-1]}"
                )

            idx = int(matches[0])
            # Cached full-grid slice for this date (channel-selected).
            data_slice = _get_anemoi_date_slice(ds, filename, idx, target_idx)
            all_data.append(data_slice)

        # For gridded data every grid point shares the same unique time,
        # so each slice already has the right shape.
        if len(all_data) == 1:
            target_data = all_data[0]
        else:
            target_data = np.concatenate(all_data, axis=0)

    # Apply the same early channel selection as the zarr path
    if channel_idxs is not None:
        target_data = target_data[:, channel_idxs]

    return target_data.astype(np.float32)


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
    regrid_opts: dict | None = None,
    anemoi_target_cfg: dict | None = None,
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

    # Open anemoi dataset once for all fsteps (if configured)
    anemoi_handle = None
    if anemoi_target_cfg is not None:
        anemoi_handle = _open_anemoi_dataset(anemoi_target_cfg)

    for fi, fs in enumerate(fsteps):
        base = f"{sample}/{stream}/{fs}"

        # Direct array access — bypasses OutputDataset/as_xarray/dask entirely
        pred_data = np.asarray(ds[f"{base}/prediction/data"])
        times_data = np.asarray(ds[f"{base}/prediction/times"])

        if anemoi_handle is not None:
            # Read target from anemoi dataset
            anemoi_ds, target_idx, ds_dates, ds_lats, ds_lons = anemoi_handle
            # Match the anemoi full grid to the prediction grid points via
            # nearest-neighbour on lat/lon so the target aligns with the
            # (possibly sparse) prediction output.
            pred_coords = np.asarray(ds[f"{base}/prediction/coords"])
            grid_indices = _match_anemoi_indices(
                ds_lats, ds_lons, pred_coords, anemoi_target_cfg["filename"]
            )
            if fi == 0:
                _logger.info(
                    f"Sample {sample} fstep {fs}: reading target from anemoi "
                    f"(times shape={times_data.shape}, unique={len(np.unique(times_data))}, "
                    f"pred_points={pred_coords.shape[0]})"
                )
            target_data = _read_anemoi_target(
                anemoi_ds,
                target_idx,
                ds_dates,
                times_data,
                channel_idxs,
                grid_indices,
                filename=anemoi_target_cfg["filename"],
            )
        else:
            target_data = np.asarray(ds[f"{base}/target/data"])
            if channel_idxs is not None:
                target_data = target_data[:, channel_idxs]

        # Select channels by index for prediction
        if channel_idxs is not None:
            pred_data = (
                pred_data[:, channel_idxs] if pred_data.ndim == 2 else pred_data[:, channel_idxs, :]
            )
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
