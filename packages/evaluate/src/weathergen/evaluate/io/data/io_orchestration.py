# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Orchestration logic for the zarr I/O path.

The two implementations (LocalStore fstep-serial, ZipStore all-fsteps-at-once)
and all their shared sub-routines live here so that WeatherGenZarrReader stays
focused on the public API and caching.
"""

import contextlib
import logging
import os
import resource
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from numpy.typing import NDArray

from weathergen.evaluate.io.data.dataarray_builders import (
    EnsembleSelect,
    build_gridded_dataarrays,
    build_scatter_dataarrays,
)
from weathergen.evaluate.io.data.dataarray_postprocessing import (
    _add_lead_time_coord,
    _select_channels,
)
from weathergen.evaluate.io.data.io_workers import (
    _compute_early_channel_selection,
    _read_coords_and_meta,
    _read_sample,
)
from weathergen.evaluate.io.io_reader import ReaderOutput
from weathergen.evaluate.utils.derived_channels import scale_z_channels

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared state passed from WeatherGenZarrReader into the impl functions.
# Using a dataclass avoids threading `self` through module-level functions
# while keeping the call sites readable.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IOState:
    """Resolved I/O parameters for one get_data call."""

    run_id: str
    zarr_path: str
    is_zip: bool
    stream: str
    stream_cfg: dict
    fsteps: list[int]
    samples: list[int]
    channels: list[str]
    ensemble: list[str]
    ens_select: EnsembleSelect
    is_gridded: bool
    channel_idxs: list[int] | None
    read_channels: list[str]
    coords: NDArray
    lat: NDArray
    lon: NDArray
    n_workers: int
    backend: str = "loky"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def get_num_workers(*, check_process_headroom: bool = False, max_workers: int | None = None) -> int:
    """Determine safe number of parallel workers.

    Parameters
    ----------
    check_process_headroom : bool
        When *True* (useful for ``loky`` / process-based backends), also
        verify that the user has enough ``RLIMIT_NPROC`` headroom before
        returning > 1.  If headroom is dangerously low the function
        returns 1 regardless of the CPU-based estimate.
    max_workers : int | None
        Optional hard cap for max workers.  When set from the eval config
        (``max_workers`` key in the YAML), it overrides the default of 36.

    Auto-detection priority:
    1. ``$SLURM_CPUS_PER_TASK`` — CPUs allocated to this task (preferred).
    2. ``$SLURM_CPUS_ON_NODE`` — CPUs available on the node.
    3. ``os.cpu_count()`` — fallback outside Slurm.

    The detected CPU count is capped at *max_workers* (default 36).
    """
    _max_workers = max_workers if max_workers is not None else 36

    # Prefer Slurm-aware CPU counts — they reflect the actual allocation,
    # not the full node (which os.cpu_count() returns).
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus is not None:
        try:
            n = max(1, int(slurm_cpus))
            n = min(n, _max_workers)
            _logger.info(f"Auto-detected {slurm_cpus} Slurm CPUs. Using n_workers={n}.")
        except ValueError:
            slurm_cpus = None  # fall through

    if slurm_cpus is None:
        cpu_count = os.cpu_count() or 16
        n = max(1, min(cpu_count, _max_workers))
        _logger.info(f"No Slurm environment detected (cpu_count={cpu_count}). Using n_workers={n}.")

    # --- Optional process-headroom guard (for loky / process backends) ---
    if check_process_headroom and n > 1:
        n = _apply_process_headroom(n)

    return n


def _apply_process_headroom(n: int) -> int:
    """Reduce *n* to 1 when the user's RLIMIT_NPROC headroom is dangerously low."""
    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NPROC)
        if soft_limit == resource.RLIM_INFINITY:
            soft_limit = 65536

        result = subprocess.run(
            ["ps", "-u", str(os.getuid()), "--no-headers", "-o", "pid"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        user_procs = len(result.stdout.strip().splitlines()) if result.returncode == 0 else 0

        available = soft_limit - user_procs
        if available < 64:
            _logger.info(
                f"Low process headroom ({available}/{soft_limit} slots free). Forcing n_workers=1."
            )
            return 1

        capped = min(n, available // 8)
        if capped < n:
            _logger.info(
                f"Process headroom {available}/{soft_limit} free. "
                f"Capping n_workers from {n} to {capped}."
            )
        return max(1, capped)

    except Exception as exc:
        _logger.debug(f"Could not check process headroom ({exc}). Keeping n_workers={n}.")
        return n


# Generic parallel dispatch with fallback


def dispatch_parallel(
    calls: list,
    *,
    n_workers: int,
    backend: str = "loky",
    desc: str = "",
    verbose: int = 2,
) -> list:
    """Run *calls* with ``joblib.Parallel``, falling back to sequential on error.

    Parameters
    ----------
    calls
        Pre-built ``delayed(fn)(*args, **kwargs)`` objects.
    n_workers
        Requested parallelism.  Automatically capped to ``len(calls)``.
    backend
        Joblib backend (``"loky"``, ``"threading"``, …).
    desc
        Description shown in the ``tqdm`` progress bar when running sequentially.
    verbose
        Joblib verbosity level (only used when ``n_workers > 1``).

    Returns
    -------
    list
        Collected results, one per call, in the same order as *calls*.

    Notes
    -----
    * ``Parallel(n_jobs=1)`` already runs sequentially, so the only reason we
      keep a try/except path is that **loky** can fail at pool creation time
      (RLIMIT_NPROC exhausted, sandbox issues, etc.).
    * When *n_workers* ≤ 1 **and** the backend is ``"loky"`` we also skip the
      ``Parallel`` call to avoid any pool-creation overhead.
    """
    n_tasks = len(calls)
    if n_tasks == 0:
        return []

    effective = min(n_workers, n_tasks)

    # skip Parallel entirely when sequential loky to avoid pool-creation overhead
    if effective <= 1 and backend == "loky":
        results = [c[0](*c[1], **c[2]) for c in calls]

    # parallel: try, then fall back to sequential on pool-creation failure.
    else:
        try:
            results = Parallel(n_jobs=effective, backend=backend, verbose=verbose)(calls)
            if backend == "loky":
                with contextlib.suppress(Exception):
                    get_reusable_executor().shutdown(wait=True)
        except Exception as exc:
            _logger.warning(
                f"{desc}: parallel pool failed ({type(exc).__name__}: {exc}). "
                f"Falling back to sequential."
            )
            if backend == "loky":
                with contextlib.suppress(Exception):
                    get_reusable_executor().shutdown(wait=True)
            results = [c[0](*c[1], **c[2]) for c in calls]

    return results


def _build_io_state(
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
    ens_select: EnsembleSelect,
) -> IOState:
    """Resolve all I/O parameters that are shared between the two impl paths."""
    zarr_path = str(fname_zarr)
    is_zip = zarr_path.endswith(".zip")

    # ---- Read coordinates and channel names from zarr (once) ----
    coords, zarr_channels, _ = _read_coords_and_meta(zarr_path, stream, fsteps[0], is_zip)
    read_channels: list[str] = zarr_channels if zarr_channels else all_channels
    channel_idxs: list[int] | None = None if zarr_channels else list(range(len(all_channels)))

    # ---- Early channel selection (skip unrequested channels at numpy level) ----
    channel_idxs, read_channels = _compute_early_channel_selection(
        read_channels, channels, stream_cfg
    )

    lat = coords[:, 0]
    lon = coords[:, 1]

    return IOState(
        run_id=run_id,
        zarr_path=zarr_path,
        is_zip=is_zip,
        stream=stream,
        stream_cfg=stream_cfg,
        fsteps=fsteps,
        samples=samples,
        channels=channels,
        ensemble=ensemble,
        ens_select=ens_select,
        is_gridded=is_gridded,
        channel_idxs=channel_idxs,
        read_channels=read_channels,
        coords=coords,
        lat=lat,
        lon=lon,
        n_workers=n_io_workers,
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
) -> tuple[list, bool]:
    """Dispatch _read_sample over samples, with parallel→sequential fallback.

    Returns
    -------
    tuple[list, bool]
        ``(results, fell_back)`` — the per-sample results and whether
        the dispatch fell back from parallel to sequential execution.
    """
    kwargs = dict(
        zarr_path=zarr_path,
        stream=stream,
        fsteps=fsteps_arg,
        channel_idxs=channel_idxs,
        is_zip=is_zip,
        read_coords=need_coords,
        is_gridded=is_gridded,
    )

    calls = [delayed(_read_sample)(sample=s, **kwargs) for s in samples]
    effective = min(n_workers, len(calls))

    if effective <= 1:
        results = [c[0](*c[1], **c[2]) for c in calls]
        return results, False

    try:
        results = Parallel(n_jobs=effective, backend=backend, verbose=5)(calls)
        with contextlib.suppress(Exception):
            get_reusable_executor().shutdown(wait=True)
        return results, False
    except Exception as exc:
        _logger.warning(
            f"{label}: parallel pool failed ({type(exc).__name__}: {exc}). "
            f"Falling back to sequential."
        )
        with contextlib.suppress(Exception):
            get_reusable_executor().shutdown(wait=True)
        results = [c[0](*c[1], **c[2]) for c in calls]
        return results, True


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
    state: IOState,
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
            state.ens_select,
        )
    else:
        # meta["coords"] is a list[NDArray | None] with one entry per fstep.
        # Extract the coords for the current fstep_idx from each sample's result.
        all_coords_lists = [results[i][3].get("coords", []) for i in range(len(state.samples))]
        per_sample_coords = [
            (cl[fstep_idx] if cl and fstep_idx < len(cl) else None) for cl in all_coords_lists
        ]
        per_sample_obs_times = [results[i][2][fstep_idx] for i in range(len(state.samples))]
        da_tar, da_pred = build_scatter_dataarrays(
            tars_list,
            preds_list,
            state.samples,
            state.read_channels,
            per_sample_valid_times,
            source_interval_starts,
            forecast_step_val,
            state.ens_select,
            per_sample_coords,
            state.coords,
            per_sample_obs_times=per_sample_obs_times,
        )

    da_tar, da_pred = _select_channels(
        da_tar, da_pred, state.stream, state.channels, state.stream_cfg
    )

    if state.is_gridded:
        da_tar = _add_lead_time_coord(da_tar)
        da_pred = _add_lead_time_coord(da_pred)
        da_pred = scale_z_channels(da_pred, state.stream)
        da_tar = scale_z_channels(da_tar, state.stream)

    return da_tar, da_pred


def _collect_substep_valid_times(results: list, n_sub: int, sub_idx: int, fstep_idx: int) -> list:
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


def get_data_dirstore(state: IOState) -> ReaderOutput:
    """LocalStore fast-path: one fstep at a time, all samples in parallel.

    Processes one forecast step at a time to keep peak memory bounded at
    ``n_samples × 1 × n_ipoints × n_channels × 4 bytes``.
    """
    _logger.info(
        f"RUN {state.run_id} - {state.stream}: Loading {len(state.samples)} samples × "
        f"{len(state.fsteps)} fsteps via zarr I/O "
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

        results, fell_back = _parallel_read(
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
        if fell_back:
            n_workers = 1

        if source_interval_starts is None:
            source_interval_starts = _extract_source_interval_starts(results, state.samples)

        n_sub = results[0][3]["n_substeps"][0]

        for sub_idx in range(n_sub):
            tars_list = [results[i][1][sub_idx] for i in range(len(state.samples))]
            preds_list = [results[i][0][sub_idx] for i in range(len(state.samples))]
            per_sample_valid_times = _collect_substep_valid_times(results, n_sub, sub_idx, 0)

            fs_val = fs if n_sub == 1 else fstep_counter
            da_tar, da_pred = _assemble_substep(
                state,
                results,
                tars_list,
                preds_list,
                per_sample_valid_times,
                source_interval_starts,
                fs_val,
                0,
            )
            del tars_list, preds_list
            fstep_counter = _store_substep(
                da_tar, da_pred, n_sub, fs, fstep_counter, da_tars_dict, da_preds_dict
            )

        del results

    if n_workers > 1:
        get_reusable_executor().shutdown(wait=True)

    _logger.info(
        f"RUN {state.run_id} - {state.stream}: I/O complete. "
        f"{len(da_tars_dict)} forecast entries loaded."
    )
    return ReaderOutput(target=da_tars_dict, prediction=da_preds_dict)


# ---------------------------------------------------------------------------
# ZipStore implementation: all fsteps per sample in one dispatch
# ---------------------------------------------------------------------------


def get_data_zipstore(state: IOState) -> ReaderOutput:
    """ZipStore fast-path: dispatch *all* (sample, fstep) pairs in parallel.

    Each worker opens its own ZipStore handle, so the zip central-directory
    is parsed once per worker (not once per task) thanks to loky worker
    reuse.  With ``samples × fsteps`` tasks the pool utilises all available
    workers instead of being capped at ``len(samples)``.
    """
    n_total = len(state.samples) * len(state.fsteps)
    _logger.info(
        f"RUN {state.run_id} - {state.stream}: Loading {len(state.samples)} samples × "
        f"{len(state.fsteps)} fsteps = {n_total} items via ZipStore-parallel zarr I/O "
        f"(workers={state.n_workers}, backend={state.backend})..."
    )

    # --- Dispatch every (sample, fstep) pair as a separate task -----------
    kwargs = dict(
        zarr_path=state.zarr_path,
        stream=state.stream,
        channel_idxs=state.channel_idxs,
        is_zip=state.is_zip,
        read_coords=not state.is_gridded,
        is_gridded=state.is_gridded,
    )
    calls = [
        delayed(_read_sample)(sample=s, fsteps=[fs], **kwargs)
        for s in state.samples
        for fs in state.fsteps
    ]
    flat_results = dispatch_parallel(
        calls,
        n_workers=state.n_workers,
        backend=state.backend,
        desc=f"RUN {state.run_id} - {state.stream} (ZipStore)",
        verbose=5,
    )

    # --- Re-group: flat_results[sample_idx * n_fsteps + fstep_idx] --------
    n_fsteps = len(state.fsteps)
    # Gather per-sample results in the same shape as get_data_dirstore expects
    source_interval_starts = _extract_source_interval_starts(
        [flat_results[si * n_fsteps] for si in range(len(state.samples))],
        state.samples,
    )

    da_tars_dict: dict = {}
    da_preds_dict: dict = {}
    fstep_counter = 1

    for fi, fs in enumerate(state.fsteps):
        # Each (sample, fstep) result has n_substeps for that single fstep
        n_sub = flat_results[fi][3]["n_substeps"][0]  # from first sample

        for sub_idx in range(n_sub):
            tars_list = [
                flat_results[si * n_fsteps + fi][1][sub_idx] for si in range(len(state.samples))
            ]
            preds_list = [
                flat_results[si * n_fsteps + fi][0][sub_idx] for si in range(len(state.samples))
            ]
            per_sample_valid_times = []
            for si in range(len(state.samples)):
                res = flat_results[si * n_fsteps + fi]
                time_entry = res[2][0]  # single fstep → index 0
                if n_sub > 1 and sub_idx < len(time_entry):
                    per_sample_valid_times.append(np.datetime64(time_entry[sub_idx], "ns"))
                elif len(time_entry) > 0:
                    per_sample_valid_times.append(np.datetime64(time_entry[0], "ns"))
                else:
                    per_sample_valid_times.append(np.datetime64("NaT", "ns"))

            fs_val = fs if n_sub == 1 else fstep_counter
            # Build a per-sample results list compatible with _assemble_substep
            per_sample_results = [
                flat_results[si * n_fsteps + fi] for si in range(len(state.samples))
            ]
            da_tar, da_pred = _assemble_substep(
                state,
                per_sample_results,
                tars_list,
                preds_list,
                per_sample_valid_times,
                source_interval_starts,
                fs_val,
                0,  # fstep_idx is always 0 since each result has a single fstep
            )
            del tars_list, preds_list
            fstep_counter = _store_substep(
                da_tar, da_pred, n_sub, fs, fstep_counter, da_tars_dict, da_preds_dict
            )

    del flat_results

    if state.n_workers > 1:
        get_reusable_executor().shutdown(wait=True)

    _logger.info(
        f"RUN {state.run_id} - {state.stream}: ZipStore-parallel I/O complete. "
        f"{len(da_tars_dict)} forecast entries loaded."
    )
    return ReaderOutput(target=da_tars_dict, prediction=da_preds_dict)
