# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Orchestration logic for the fast raw-zarr I/O path.

The two implementations (LocalStore fstep-serial, ZipStore all-fsteps-at-once)
and all their shared sub-routines live here so that WeatherGenZarrReader stays
focused on the public API and caching.
"""

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from numpy.typing import NDArray

from weathergen.evaluate.io.data.dataarray_builders import (
    build_gridded_dataarrays,
    build_scatter_dataarrays,
)
from weathergen.evaluate.io.data.raw_io_workers import (
    _compute_early_channel_selection,
    _read_coords_and_meta,
    _read_sample_raw,
)
from weathergen.evaluate.io.data.xarray_utils import (
    _add_lead_time_coord,
    _scale_z_channels,
    _select_channels,
)
from weathergen.evaluate.io.io_reader import ReaderOutput

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared state passed from WeatherGenZarrReader into the impl functions.
# Using a dataclass avoids threading `self` through module-level functions
# while keeping the call sites readable.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RawIOState:
    """Resolved I/O parameters for one get_data_raw call."""

    run_id: str
    zarr_path: str
    is_zip: bool
    stream: str
    stream_cfg: dict
    fsteps: list[int]
    samples: list[int]
    channels: list[str]
    ensemble: list[str]
    all_ens: list[str] | None
    is_gridded: bool
    channel_idxs: list[int] | None
    read_channels: list[str]
    coords_raw: NDArray
    lat: NDArray
    lon: NDArray
    n_workers: int
    backend: str = "loky"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_raw_io_state(
    run_id: str,
    fname_zarr: Path,
    stream: str,
    stream_cfg: dict,
    all_channels: list[str],
    is_gridded: bool,
    fsteps: list[int],
    samples: list[int],
    channels: list[str],
    ensemble: list[str],
    n_io_workers: int,
    all_ens: list[str] | None,
) -> RawIOState:
    """Resolve all I/O parameters that are shared between the two impl paths."""
    zarr_path = str(fname_zarr)
    is_zip = zarr_path.endswith(".zip")

    # ---- Read coordinates and channel names from zarr (once) ----
    coords_raw, zarr_channels, _ = _read_coords_and_meta(zarr_path, stream, fsteps[0], is_zip)
    read_channels: list[str] = zarr_channels if zarr_channels else all_channels
    channel_idxs: list[int] | None = None if zarr_channels else list(range(len(all_channels)))

    # ---- Early channel selection (skip unrequested channels at numpy level) ----
    channel_idxs, read_channels = _compute_early_channel_selection(
        read_channels, channels, stream_cfg
    )

    lat = coords_raw[:, 0].astype(np.float64)
    lon = coords_raw[:, 1].astype(np.float64)
    n_workers = min(n_io_workers, len(samples))

    return RawIOState(
        run_id=run_id,
        zarr_path=zarr_path,
        is_zip=is_zip,
        stream=stream,
        stream_cfg=stream_cfg,
        fsteps=fsteps,
        samples=samples,
        channels=channels,
        ensemble=ensemble,
        all_ens=all_ens,
        is_gridded=is_gridded,
        channel_idxs=channel_idxs,
        read_channels=read_channels,
        coords_raw=coords_raw,
        lat=lat,
        lon=lon,
        n_workers=n_workers,
    )


def _parallel_read(
    zarr_path: str,
    samples: list[int],
    stream: str,
    fsteps_arg: list[int],
    channel_idxs: list[int] | None,
    is_zip: bool,
    need_coords: bool,
    is_gridded: bool,
    n_workers: int,
    backend: str,
    label: str,
) -> list:
    """Dispatch _read_sample_raw over samples, with parallel→sequential fallback."""
    kwargs = dict(
        zarr_path=zarr_path,
        stream=stream,
        fsteps=fsteps_arg,
        channel_idxs=channel_idxs,
        is_zip=is_zip,
        read_coords=need_coords,
        is_gridded=is_gridded,
    )

    if n_workers > 1:
        try:
            return Parallel(n_jobs=n_workers, backend=backend, verbose=5)(
                delayed(_read_sample_raw)(sample=s, **kwargs) for s in samples
            )
        except (RuntimeError, OSError) as err:
            _logger.warning(f"{label}: parallel pool failed ({err}). Falling back to sequential.")
            with contextlib.suppress(Exception):
                get_reusable_executor().shutdown(wait=True)

    return [_read_sample_raw(sample=s, **kwargs) for s in samples]


def _extract_source_interval_starts(results: list, samples: list[int]) -> NDArray:
    """Build a (n_samples,) datetime64[ns] array of source_interval_start values."""
    si_list = []
    for i in range(len(samples)):
        si = results[i][3].get("source_interval", {})
        start_str = si.get("start", None)
        si_list.append(
            np.datetime64(start_str, "ns") if start_str is not None else np.datetime64("NaT", "ns")
        )
    return np.array(si_list)


def _assemble_substep(
    state: RawIOState,
    results: list,
    tars_list: list[NDArray],
    preds_list: list[NDArray],
    per_sample_valid_times: list,
    source_interval_starts: NDArray,
    forecast_step_val: int,
    fstep_idx: int,  # index into results[i][2] for scatter obs_times
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build and post-process (select, scale, add lead_time) one sub-step's DataArrays."""
    if state.is_gridded:
        da_tar, da_pred = build_gridded_dataarrays(
            tars_list,
            preds_list,
            state.samples,
            state.read_channels,
            state.lat,
            state.lon,
            per_sample_valid_times,
            source_interval_starts,
            forecast_step_val,
            state.ensemble,
            state.all_ens,
        )
    else:
        per_sample_coords = [results[i][3].get("coords", None) for i in range(len(state.samples))]
        per_sample_obs_times = [results[i][2][fstep_idx] for i in range(len(state.samples))]
        da_tar, da_pred = build_scatter_dataarrays(
            tars_list,
            preds_list,
            state.samples,
            state.read_channels,
            per_sample_valid_times,
            source_interval_starts,
            forecast_step_val,
            state.ensemble,
            state.all_ens,
            per_sample_coords,
            state.coords_raw,
            per_sample_obs_times=per_sample_obs_times,
        )

    da_tar, da_pred = _select_channels(
        da_tar, da_pred, state.stream, state.channels, state.stream_cfg
    )

    if state.is_gridded:
        da_tar = _add_lead_time_coord(da_tar)
        da_pred = _add_lead_time_coord(da_pred)
        da_pred = _scale_z_channels(da_pred, state.stream)
        da_tar = _scale_z_channels(da_tar, state.stream)

    return da_tar, da_pred


def _collect_substep_valid_times(
    results: list, n_sub: int, sub_idx: int, fstep_idx: int
) -> list:
    """Extract per-sample valid_times for one sub-step."""
    per_sample_valid_times = []
    for i in range(len(results)):
        time_entry = results[i][2][fstep_idx]
        if n_sub > 1 and sub_idx < len(time_entry):
            per_sample_valid_times.append(np.datetime64(time_entry[sub_idx], "ns"))
        elif len(time_entry) > 0:
            per_sample_valid_times.append(np.datetime64(time_entry[0], "ns"))
        else:
            per_sample_valid_times.append(np.datetime64("NaT", "ns"))
    return per_sample_valid_times


def _store_substep(
    da_tar: xr.DataArray,
    da_pred: xr.DataArray,
    n_sub: int,
    fs: int,
    fstep_counter: int,
    da_tars_dict: dict,
    da_preds_dict: dict,
) -> int:
    """Store one sub-step into the output dicts; return the next fstep_counter."""
    if n_sub > 1:
        da_tar = da_tar.assign_coords(forecast_step=fstep_counter)
        da_pred = da_pred.assign_coords(forecast_step=fstep_counter)
        da_tars_dict[fstep_counter] = da_tar
        da_preds_dict[fstep_counter] = da_pred
        return fstep_counter + 1
    else:
        da_tars_dict[int(fs)] = da_tar
        da_preds_dict[int(fs)] = da_pred
        return fstep_counter


# ---------------------------------------------------------------------------
# LocalStore implementation: fstep-serial, samples-parallel
# ---------------------------------------------------------------------------


def get_data_raw_impl(state: RawIOState) -> ReaderOutput:
    """LocalStore fast-path: one fstep at a time, all samples in parallel.

    Processes one forecast step at a time to keep peak memory bounded at
    ``n_samples × 1 × n_ipoints × n_channels × 4 bytes``.
    """
    _logger.info(
        f"RUN {state.run_id} - {state.stream}: Loading {len(state.samples)} samples × "
        f"{len(state.fsteps)} fsteps via raw zarr I/O "
        f"(workers={state.n_workers}, backend={state.backend})..."
    )

    da_tars_dict: dict = {}
    da_preds_dict: dict = {}
    fstep_counter = 1
    source_interval_starts: NDArray | None = None
    n_workers = state.n_workers

    for fi, fs in enumerate(state.fsteps):
        _logger.info(
            f"RUN {state.run_id} - {state.stream}: "
            f"Reading fstep {fs} ({fi + 1}/{len(state.fsteps)})..."
        )

        results = _parallel_read(
            zarr_path=state.zarr_path,
            samples=state.samples,
            stream=state.stream,
            fsteps_arg=[fs],
            channel_idxs=state.channel_idxs,
            is_zip=state.is_zip,
            need_coords=not state.is_gridded,
            is_gridded=state.is_gridded,
            n_workers=n_workers,
            backend=state.backend,
            label=f"RUN {state.run_id} - {state.stream} fstep {fs}",
        )
        # If _parallel_read fell back to sequential, honour that for the rest
        if n_workers > 1 and len(results) == len(state.samples):
            # check if it actually ran sequentially (no exception path exposed,
            # but n_workers may have been reset inside — we can't detect that here,
            # so we just trust the results list length is correct).
            pass

        if source_interval_starts is None:
            source_interval_starts = _extract_source_interval_starts(results, state.samples)

        n_sub = results[0][3]["n_substeps"][0]

        for sub_idx in range(n_sub):
            tars_list = [results[i][1][sub_idx] for i in range(len(state.samples))]
            preds_list = [results[i][0][sub_idx] for i in range(len(state.samples))]
            per_sample_valid_times = _collect_substep_valid_times(results, n_sub, sub_idx, 0)

            fs_val = fs if n_sub == 1 else fstep_counter
            da_tar, da_pred = _assemble_substep(
                state, results, tars_list, preds_list,
                per_sample_valid_times, source_interval_starts, fs_val, 0
            )
            del tars_list, preds_list
            fstep_counter = _store_substep(
                da_tar, da_pred, n_sub, fs, fstep_counter, da_tars_dict, da_preds_dict
            )

        del results

    if n_workers > 1:
        get_reusable_executor().shutdown(wait=True)

    _logger.info(
        f"RUN {state.run_id} - {state.stream}: Raw I/O complete. "
        f"{len(da_tars_dict)} forecast entries loaded."
    )
    return ReaderOutput(target=da_tars_dict, prediction=da_preds_dict)


# ---------------------------------------------------------------------------
# ZipStore implementation: all fsteps per sample in one dispatch
# ---------------------------------------------------------------------------


def get_data_raw_zip_impl(state: RawIOState) -> ReaderOutput:
    """ZipStore fast-path: all fsteps per sample dispatched in one Parallel call.

    Amortises the ZipStore central-directory parse cost across all fsteps
    per sample by reading them all inside a single worker call.
    """
    _logger.info(
        f"RUN {state.run_id} - {state.stream}: Loading {len(state.samples)} samples × "
        f"{len(state.fsteps)} fsteps via ZipStore-parallel raw zarr I/O "
        f"(workers={state.n_workers}, backend={state.backend})..."
    )

    results = _parallel_read(
        zarr_path=state.zarr_path,
        samples=state.samples,
        stream=state.stream,
        fsteps_arg=state.fsteps,
        channel_idxs=state.channel_idxs,
        is_zip=state.is_zip,
        need_coords=not state.is_gridded,
        is_gridded=state.is_gridded,
        n_workers=state.n_workers,
        backend=state.backend,
        label=f"RUN {state.run_id} - {state.stream} (ZipStore)",
    )

    source_interval_starts = _extract_source_interval_starts(results, state.samples)
    n_substeps_per_fstep = results[0][3]["n_substeps"]

    # Build flat-index offsets: each fstep may produce n_sub entries in results lists
    offsets = []
    off = 0
    for ns in n_substeps_per_fstep:
        offsets.append(off)
        off += ns

    da_tars_dict: dict = {}
    da_preds_dict: dict = {}
    fstep_counter = 1

    for fi, fs in enumerate(state.fsteps):
        n_sub = n_substeps_per_fstep[fi]
        base_off = offsets[fi]

        for sub_idx in range(n_sub):
            list_idx = base_off + sub_idx
            tars_list = [results[i][1][list_idx] for i in range(len(state.samples))]
            preds_list = [results[i][0][list_idx] for i in range(len(state.samples))]
            per_sample_valid_times = _collect_substep_valid_times(results, n_sub, sub_idx, fi)

            fs_val = fs if n_sub == 1 else fstep_counter
            da_tar, da_pred = _assemble_substep(
                state, results, tars_list, preds_list,
                per_sample_valid_times, source_interval_starts, fs_val, fi
            )
            del tars_list, preds_list
            fstep_counter = _store_substep(
                da_tar, da_pred, n_sub, fs, fstep_counter, da_tars_dict, da_preds_dict
            )

    del results

    if state.n_workers > 1:
        get_reusable_executor().shutdown(wait=True)

    _logger.info(
        f"RUN {state.run_id} - {state.stream}: ZipStore-parallel raw I/O complete. "
        f"{len(da_tars_dict)} forecast entries loaded."
    )
    return ReaderOutput(target=da_tars_dict, prediction=da_preds_dict)
