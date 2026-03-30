# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Orchestration logic for WeatherGenZarrReader.get_data (ZarrIO-based path).

Phases extracted from the method body so WeatherGenZarrReader stays thin.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import xarray as xr
from tqdm import tqdm

from weathergen.evaluate.io.data._raw_io_workers import (
    _init_worker_zio,
    _load_single_sample,
    _load_single_sample_own_context,
)
from weathergen.evaluate.io.data._xarray_utils import (
    _add_lead_time_coord,
    _scale_z_channels,
    _select_channels,
    _split_by_valid_time,
)
from weathergen.evaluate.io.io_reader import ReaderOutput

_logger = logging.getLogger(__name__)


def _dispatch_reads(
    zio,
    fname_zarr: Path,
    samples: list[int],
    fsteps: list[int],
    ensemble: list[str],
    is_gridded: bool,
    effective_threads: int,
    run_id: str,
    stream: str,
) -> tuple[dict[int, list], int]:
    """Phase 1: dispatch all (sample, fstep) reads using a thread/process pool.

    Returns
    -------
    results_by_fstep : dict[int, list]
        Mapping fstep → list of (sample, target, pred, valid_times) tuples.
    effective_threads : int
        Final worker count (may drop to 0 if pool errored out).
    """
    n_total = len(samples) * len(fsteps)
    is_zip_store = hasattr(zio, "_store") and "zip" in type(zio._store).__name__.lower()

    _logger.info(
        f"RUN {run_id} - {stream}: Loading {len(samples)} samples × "
        f"{len(fsteps)} fsteps = {n_total} items "
        f"(workers={effective_threads}, zip={is_zip_store})..."
    )

    results_by_fstep: dict[int, list] = {f: [] for f in fsteps}

    if effective_threads > 1:
        if is_zip_store:
            pool_cls = ProcessPoolExecutor
            pool_kwargs = {
                "max_workers": effective_threads,
                "initializer": _init_worker_zio,
                "initargs": (fname_zarr,),
            }
            submit_args = lambda s, f: (  # noqa: E731
                _load_single_sample_own_context, fname_zarr, s, stream, f, ensemble, is_gridded
            )
        else:
            pool_cls = ThreadPoolExecutor
            pool_kwargs = {"max_workers": effective_threads}
            submit_args = lambda s, f: (  # noqa: E731
                _load_single_sample, zio, s, stream, f, ensemble, is_gridded
            )

        try:
            with pool_cls(**pool_kwargs) as executor:
                futures = {
                    executor.submit(*submit_args(s, f)): (f, s)
                    for f in fsteps
                    for s in samples
                }
                for future in tqdm(
                    as_completed(futures),
                    total=n_total,
                    desc=f"Loading {run_id} - {stream}",
                ):
                    result = future.result()
                    if result is not None:
                        fstep_r, sample_r, target, pred, vt = result
                        results_by_fstep[fstep_r].append((sample_r, target, pred, vt))
        except (RuntimeError, OSError) as pool_err:
            _logger.warning(
                f"Parallel pool failed ({pool_err}). Falling back to sequential loading."
            )
            effective_threads = 0  # signal caller to run sequential

    if effective_threads <= 1:
        for f in fsteps:
            for s in tqdm(samples, desc=f"Loading {run_id} - {stream} - fstep {f}"):
                result = _load_single_sample(zio, s, stream, f, ensemble, is_gridded)
                if result is None:
                    continue
                _, sample_r, target, pred, vt = result
                results_by_fstep[f].append((sample_r, target, pred, vt))

    return results_by_fstep, effective_threads


def _reassemble_fsteps(
    results_by_fstep: dict[int, list],
    fsteps: list[int],
    is_gridded: bool,
    run_id: str,
    stream: str,
) -> tuple[list, list, list]:
    """Phase 2: sort per-fstep results and split/concat into DataArrays.

    Returns
    -------
    fsteps_final : list
        Per-fstep label (valid_times list for substeps, or int fstep).
    da_tars : list[xr.DataArray | list[xr.DataArray]]
    da_preds : list[xr.DataArray | list[xr.DataArray]]
    """
    da_tars, da_preds, fsteps_final = [], [], []

    for fstep in fsteps:
        per_fstep = results_by_fstep[fstep]
        if not per_fstep:
            _logger.info(f"[{run_id} - {stream}] No valid data for fstep {fstep}.")
            continue

        per_fstep.sort(key=lambda x: x[0])

        da_tars_fs = [r[1] for r in per_fstep]
        da_preds_fs = [r[2] for r in per_fstep]
        valid_times_fs = [r[3] for r in per_fstep if r[3] is not None]
        fsteps_final.append(valid_times_fs if valid_times_fs else fstep)

        _logger.debug(
            f"Concatenating targets and predictions for stream {stream}, "
            f"forecast_step {fstep}..."
        )

        if is_gridded:
            da_preds_fs = _split_by_valid_time(da_preds_fs)
            da_tars_fs = _split_by_valid_time(da_tars_fs)
        else:
            da_tars_fs = xr.concat(da_tars_fs, dim="ipoint", coords="different", compat="equals")
            da_preds_fs = xr.concat(
                da_preds_fs, dim="ipoint", coords="different", compat="equals"
            )

        da_tars.append(da_tars_fs)
        da_preds.append(da_preds_fs)

    return fsteps_final, da_tars, da_preds


def _apply_postprocessing(
    fsteps_final: list,
    da_tars: list,
    da_preds: list,
    stream: str,
    channels: list[str],
    stream_cfg: dict,
    is_gridded: bool,
) -> ReaderOutput:
    """Phase 3: channel selection, scaling, and lead_time coord assignment."""
    da_tars_dict: dict = {}
    da_preds_dict: dict = {}
    i = 1

    for fstep, da_t, da_p in zip(fsteps_final, da_tars, da_preds, strict=True):
        with_substeps = isinstance(da_t, list)
        items = zip(da_t, da_p, strict=True) if with_substeps else [(da_t, da_p)]

        for t, p in items:
            t, p = _select_channels(t, p, stream, channels, stream_cfg)

            if is_gridded:
                t = _add_lead_time_coord(t)
                p = _add_lead_time_coord(p)
                p = _scale_z_channels(p, stream)
                t = _scale_z_channels(t, stream)

            if with_substeps:
                t = t.assign_coords(forecast_step=i)
                p = p.assign_coords(forecast_step=i)
                da_tars_dict[i] = t
                da_preds_dict[i] = p
                i += 1
            else:
                da_tars_dict[int(fstep)] = t
                da_preds_dict[int(fstep)] = p

    return ReaderOutput(target=da_tars_dict, prediction=da_preds_dict)
