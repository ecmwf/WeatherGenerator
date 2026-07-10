#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
"""
era5_to_cepdiag.py — Extract ERA5 ground truth from an anemoi zarr and stage
it for CEPDIAG verification.

For each forecast init date found in the WeatherGenerator CEPDIAG staging
directory, reads the ERA5 anemoi zarr, computes weekly means over the same
time windows as the forecast files, regrids from the O96 reduced Gaussian
grid to a regular 1°×1° lat/lon grid, and writes one NetCDF file per
(init_date, param):

    {expid}_era5_0_{param}_{YYYYMMDD}.nc

placed in the same staging directory as the WG forecast files:

    {results_dir}/{run_id}/cepdiag/eval/stage/

File format (matches zarr_to_cepdiag.py output exactly)
────────────────────────────────────────────────────────
  Dimensions : (time, latitude, longitude)
  time       : valid datetime of each weekly mean (first day of the window)
               e.g. [init+0d, init+7d, init+14d, init+21d]
  latitude   : regular, south→north, step = lat_res  (e.g. -89.5 … 89.5)
  longitude  : regular, west→east,   step = lon_res  (e.g. -179.5 … 179.5)

CEPDIAG naming convention for verifying analysis
─────────────────────────────────────────────────
  {expid}_{verana}_{member}_{param}_{YYYYMMDD}.nc
  e.g.  af25nepk_era5_0_t2m_20230801.nc

ERA5 is deterministic → member = 0 always.
No [va_era5] section is needed in cepdiag_mofc.conf; CEPDIAG recognises
'era5' as a special analysis type and defaults to member 0 automatically.

ERA5 zarr path
──────────────
Resolved automatically from models/{run_id}/*.json:
  data_path_anemoi + streams[*].filenames  (picks the aifs-ea-an-oper file)
Override with --era5-zarr if needed.

Usage
─────
  python scripts/era5_to_cepdiag.py --run-id af25nepk
  python scripts/era5_to_cepdiag.py \\
      --run-id af25nepk \\
      --params t2m,mslp,z500,z200,t850,u850 \\
      --overwrite

The ERA5 temporal sampling matches the model's own time step (e.g. 6-hourly).
This is read automatically from models/{run_id}/*.json and can be overridden
with --time-step (e.g. --time-step 6).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

# Re-use regridder, CHANNEL_MAP and write_cepdiag_nc from zarr_to_cepdiag.py
sys.path.insert(0, str(Path(__file__).parent))
from zarr_to_cepdiag import (
    CHANNEL_MAP,
    DEFAULT_PARAMS,
    ReducedGaussianRegridder,
    write_cepdiag_nc,
)

_logger = logging.getLogger(__name__)

ERA5_VERANA  = "era5"   # matches verana = era5 in cepdiag_mofc.conf
ERA5_MEMBER  = 0        # ERA5 is deterministic
HOURS_PER_WEEK = 7 * 24  # 168 hourly ERA5 steps per 7-day window

# ERA5 zarr time base (1h frequency, hourly steps from 1979-01-01T00)
_ERA5_T0 = np.datetime64("1979-01-01T00:00:00", "h")


# ──────────────────────────────────────────────────────────────────────────────
# Model JSON helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_model_json(models_dir: Path, run_id: str) -> Path:
    """Return path to the model JSON for *run_id*."""
    # Layout: models/{run_id}/*.json
    subdir = models_dir / run_id
    if subdir.is_dir():
        matches = sorted(subdir.glob("*.json"))
        if matches:
            return matches[0]
    # Layout: models/model_{run_id}_*.json
    matches = sorted(models_dir.glob(f"*{run_id}*.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No model JSON found for run-id '{run_id}' under {models_dir}"
    )


def _get_model_time_step_hours(json_path: Path) -> int:
    """
    Read the forecast time step from the model JSON and return it in whole hours.

    Looks for training_config.forecast.time_step (e.g. '06:00:00').
    Defaults to 6 if the key is absent or cannot be parsed.
    """
    cfg = json.loads(json_path.read_text())
    ts_str = (
        cfg.get("training_config", {})
           .get("forecast", {})
           .get("time_step", "06:00:00")
    )
    try:
        if ":" in ts_str:                       # 'HH:MM:SS'
            hours = int(ts_str.split(":")[0])
        elif ts_str.lower().endswith("h"):       # '6h'
            hours = int(ts_str[:-1])
        else:
            hours = int(ts_str)
        if hours <= 0:
            raise ValueError("non-positive")
    except (ValueError, AttributeError):
        _logger.warning(
            "Cannot parse time_step %r from %s; defaulting to 6h.",
            ts_str, json_path.name,
        )
        hours = 6
    return hours


def _open_era5_zarr(json_path: Path) -> zarr.Group:
    """
    Open the ERA5 anemoi zarr referenced in the model JSON.

    Strategy:
    1. Prefer filenames matching '*ea-an-oper*' (ERA5 analysis).
    2. Fall back to any zarr whose 'variables' attribute contains '2t'.
    """
    cfg = json.loads(json_path.read_text())
    data_root = Path(cfg["data_path_anemoi"])
    streams = cfg.get("streams", [])

    ea_candidates: list[Path] = []
    other_candidates: list[Path] = []
    for stream in streams:
        for fname in stream.get("filenames", []):
            p = data_root / fname
            if "ea-an-oper" in fname.lower() or "era5" in fname.lower():
                ea_candidates.append(p)
            else:
                other_candidates.append(p)

    for p in ea_candidates + other_candidates:
        if not p.exists():
            _logger.debug("Zarr not found, skipping: %s", p)
            continue
        z = zarr.open(str(p), mode="r")
        if "2t" in list(z.attrs.get("variables", [])):
            _logger.info("ERA5 zarr: %s", p)
            return z

    raise RuntimeError(
        f"Could not find a zarr containing variable '2t' for run-id "
        f"'{json_path}'.  Use --era5-zarr to specify the path manually."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Init-date and step discovery from existing forecast files
# ──────────────────────────────────────────────────────────────────────────────

def _scan_forecast_files(stage_dir: Path, expid: str) -> tuple[list[str], int]:
    """
    Scan the CEPDIAG stage directory for existing WG ensemble forecast files.

    Returns
    -------
    init_dates : sorted list of 'YYYYMMDD' strings
    n_weeks    : number of weekly timesteps (from the first file's time dim)
    """
    dates: set[str] = set()
    n_weeks = 4  # fallback

    for f in stage_dir.glob(f"{expid}_ens_0_*_????????.nc"):
        parts = f.stem.split("_")
        date_str = parts[-1]
        if len(date_str) == 8 and date_str.isdigit():
            dates.add(date_str)
            if n_weeks == 4:  # read from first match only
                try:
                    ds = xr.open_dataset(f)
                    n_weeks = len(ds.time)
                    ds.close()
                except Exception:
                    pass

    if not dates:
        raise FileNotFoundError(
            f"No WG forecast files ({expid}_ens_0_*_YYYYMMDD.nc) found in "
            f"{stage_dir}.  Run zarr_to_cepdiag.py first."
        )
    return sorted(dates), n_weeks


# ──────────────────────────────────────────────────────────────────────────────
# ERA5 weekly-mean computation
# ──────────────────────────────────────────────────────────────────────────────

def _era5_time_index(init_date_str: str) -> int:
    """Return the ERA5 zarr index (1h steps from 1979-01-01T00) for init_date."""
    init_dt = np.datetime64(
        f"{init_date_str[:4]}-{init_date_str[4:6]}-{init_date_str[6:8]}T00:00:00",
        "h",
    )
    idx = int((init_dt - _ERA5_T0) / np.timedelta64(1, "h"))
    if idx < 0:
        raise ValueError(f"Init date {init_date_str} is before ERA5 start (1979-01-01)")
    return idx


def _compute_weekly_means(
    z: zarr.Group,
    var_idx: int,
    init_date_str: str,
    n_weeks: int,
    era5_stride: int = 1,
) -> tuple[np.ndarray, list[np.datetime64]]:
    """
    Read *n_weeks* × 7 days of ERA5 data for *var_idx* starting at
    *init_date_str* and return weekly mean fields plus week-start timestamps.

    *era5_stride* controls which hourly ERA5 snapshots enter each mean:
    stride=1 → all 168 hourly values per week
    stride=6 → every 6th hour (28 values/week, matching a 6h model time step)

    HOURS_PER_WEEK (168) must be divisible by *era5_stride*.

    Returns
    -------
    weekly : (n_weeks, n_cells)  float32
    times  : list of n_weeks np.datetime64[ns] (start-of-window timestamps)
    """
    if HOURS_PER_WEEK % era5_stride != 0:
        raise ValueError(
            f"era5_stride={era5_stride} does not evenly divide "
            f"HOURS_PER_WEEK={HOURS_PER_WEEK}"
        )
    idx0 = _era5_time_index(init_date_str)
    n_hours = n_weeks * HOURS_PER_WEEK

    # Verify we don't read past the end of the zarr
    zarr_len = z["data"].shape[0]
    if idx0 + n_hours > zarr_len:
        available = (zarr_len - idx0) // HOURS_PER_WEEK
        _logger.warning(
            "ERA5 zarr ends before %d weeks after %s; "
            "only %d complete week(s) available.",
            n_weeks, init_date_str, available,
        )
        n_weeks = available
        n_hours = n_weeks * HOURS_PER_WEEK
        if n_weeks == 0:
            raise RuntimeError(
                f"No complete weeks available in ERA5 zarr after {init_date_str}"
            )

    # Read contiguous block then stride-select to match the model time step.
    # z['data'] layout: (N_time, N_vars, 1, N_cells)
    raw_all = z["data"][idx0 : idx0 + n_hours, var_idx, 0, :]  # (H, C) float32
    raw = raw_all[::era5_stride]  # (n_weeks * steps_per_week, C)

    steps_per_week = HOURS_PER_WEEK // era5_stride
    weekly = raw.reshape(n_weeks, steps_per_week, raw.shape[-1]).mean(
        axis=1
    ).astype(np.float32)  # (n_weeks, n_cells)

    init_h = np.datetime64(
        f"{init_date_str[:4]}-{init_date_str[4:6]}-{init_date_str[6:8]}T00:00:00",
        "h",
    )
    times = [
        np.datetime64(init_h + np.timedelta64(w * HOURS_PER_WEEK, "h"), "ns")
        for w in range(n_weeks)
    ]
    return weekly, times


# ──────────────────────────────────────────────────────────────────────────────
# Per-parameter processing
# ──────────────────────────────────────────────────────────────────────────────

def _process_param(
    z: zarr.Group,
    regridder: ReducedGaussianRegridder,
    var_idx: int,
    cepdiag_param: str,
    scale: float,
    offset: float,
    init_date_str: str,
    n_weeks: int,
    era5_stride: int,
    out_dir: Path,
    expid: str,
    lats: np.ndarray,
    lons: np.ndarray,
    overwrite: bool,
) -> bool:
    """Process one (param, init_date) pair.  Returns True if a file was written."""
    outfile = (
        out_dir
        / f"{expid}_{ERA5_VERANA}_{ERA5_MEMBER}_{cepdiag_param}_{init_date_str}.nc"
    )
    if outfile.exists() and not overwrite:
        _logger.debug("Skipping existing %s", outfile.name)
        return False

    _logger.info("  Writing %s", outfile.name)

    weekly, times = _compute_weekly_means(z, var_idx, init_date_str, n_weeks, era5_stride)

    if scale != 1.0 or offset != 0.0:
        weekly = weekly * np.float32(scale) + np.float32(offset)

    # Regrid each weekly mean: (n_weeks, n_cells) → (n_weeks, n_lats, n_lons)
    regridded = np.empty((len(times), *regridder.tgt_shape), dtype=np.float32)
    for w in range(len(times)):
        regridded[w] = regridder.regrid(weekly[w])

    init_dt = np.datetime64(
        f"{init_date_str[:4]}-{init_date_str[4:6]}-{init_date_str[6:8]}T00:00:00",
        "ns",
    )
    write_cepdiag_nc(
        outfile    = outfile,
        param      = cepdiag_param,
        member     = ERA5_MEMBER,
        init_time  = init_dt,
        week_times = times,
        field      = regridded,
        lats       = lats,
        lons       = lons,
    )
    return True


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--run-id", required=True,
        help="WeatherGenerator run ID (e.g. af25nepk)",
    )
    ap.add_argument(
        "--results-dir",
        help="Root of results directory (default: <repo>/results)",
    )
    ap.add_argument(
        "--models-dir",
        help="Root of models directory  (default: <repo>/models)",
    )
    ap.add_argument(
        "--era5-zarr",
        help="Override: explicit path to ERA5 anemoi zarr directory",
    )
    ap.add_argument(
        "--params",
        default=",".join(DEFAULT_PARAMS),
        help="Comma-separated CEPDIAG param names to stage "
             f"(default: {','.join(DEFAULT_PARAMS)})",
    )
    ap.add_argument(
        "--n-weeks", type=int, default=None,
        help="Number of weekly means per init date "
             "(auto-detected from existing forecast files)",
    )
    ap.add_argument(
        "--lon-res", type=float, default=1.0,
        help="Output longitude resolution in degrees (default: 1.0)",
    )
    ap.add_argument(
        "--lat-res", type=float, default=1.0,
        help="Output latitude resolution in degrees (default: 1.0)",
    )
    ap.add_argument(
        "--time-step", type=int, default=None, metavar="HOURS",
        help="ERA5 sampling interval in hours (default: read from model JSON; "
             "e.g. 6 for a 6-hourly model)",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files",
    )
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── paths ──────────────────────────────────────────────────────────────
    repo_root    = Path(__file__).resolve().parents[1]
    results_root = Path(args.results_dir) if args.results_dir else repo_root / "results"
    models_root  = Path(args.models_dir)  if args.models_dir  else repo_root / "models"

    stage_dir = results_root / args.run_id / "cepdiag" / "eval" / "stage"
    if not stage_dir.is_dir():
        ap.error(
            f"CEPDIAG stage directory not found: {stage_dir}\n"
            "Run zarr_to_cepdiag.py first."
        )

    expid = args.run_id

    # ── discover init dates and n_weeks from existing forecast files ───────
    init_dates, n_weeks_detected = _scan_forecast_files(stage_dir, expid)
    n_weeks = args.n_weeks if args.n_weeks is not None else n_weeks_detected
    _logger.info(
        "Init date(s): %s  |  weekly steps per date: %d",
        ", ".join(init_dates), n_weeks,
    )

    # ── open ERA5 zarr and resolve sampling stride ────────────────────────
    json_path: Path | None = None
    if args.era5_zarr:
        era5_z = zarr.open(args.era5_zarr, mode="r")
        _logger.info("ERA5 zarr (manual): %s", args.era5_zarr)
    else:
        json_path = _find_model_json(models_root, expid)
        _logger.info("Model JSON: %s", json_path.name)
        era5_z = _open_era5_zarr(json_path)

    if args.time_step is not None:
        era5_stride = args.time_step
    elif json_path is not None:
        era5_stride = _get_model_time_step_hours(json_path)
    else:
        era5_stride = 1  # --era5-zarr without --time-step: use all hourly data
    if HOURS_PER_WEEK % era5_stride != 0:
        ap.error(
            f"--time-step {era5_stride} does not evenly divide "
            f"HOURS_PER_WEEK={HOURS_PER_WEEK} (168)"
        )
    steps_per_week = HOURS_PER_WEEK // era5_stride
    _logger.info(
        "ERA5 sampling: every %dh → %d samples/week", era5_stride, steps_per_week
    )

    era5_vars = list(era5_z.attrs["variables"])
    era5_freq = era5_z.attrs.get("frequency", "1h")
    if era5_freq not in ("1h", "1H"):
        _logger.warning(
            "ERA5 zarr frequency is %s, not 1h.  "
            "Weekly-mean computation assumes 1-hourly data.",
            era5_freq,
        )
    _logger.info(
        "ERA5 zarr: %d variables, %d timesteps (%s)",
        len(era5_vars), era5_z["data"].shape[0], era5_freq,
    )

    # ── target params ──────────────────────────────────────────────────────
    # Build reverse map: cepdiag_param → (wg_channel_name, scale, offset)
    cepdiag_to_wg = {v[0]: (k, v[1], v[2]) for k, v in CHANNEL_MAP.items()}
    params_todo: list[tuple[str, str, float, float]] = []  # (cepdiag, wg, scale, off)
    for p in [s.strip() for s in args.params.split(",")]:
        if p not in cepdiag_to_wg:
            _logger.warning("Unknown CEPDIAG param %r — skipping", p)
            continue
        wg_ch, scale, offset = cepdiag_to_wg[p]
        if wg_ch not in era5_vars:
            _logger.warning(
                "Variable %r not in ERA5 zarr (needed for %s) — skipping",
                wg_ch, p,
            )
            continue
        params_todo.append((p, wg_ch, scale, offset))

    if not params_todo:
        _logger.error("No valid params to process.  Check --params.")
        return

    # ── target grid ────────────────────────────────────────────────────────
    lats = np.arange(-90.0 + args.lat_res / 2, 90.0, args.lat_res)
    lons = np.arange(-180.0 + args.lon_res / 2, 180.0, args.lon_res)

    # ── build regridder ────────────────────────────────────────────────────
    src_lats = era5_z["latitudes"][:]
    src_lons = era5_z["longitudes"][:]
    src_coords = np.column_stack([src_lats, src_lons])
    regridder = ReducedGaussianRegridder(src_coords, lats, lons)

    # ── main loop: param × init_date ───────────────────────────────────────
    n_written = 0
    for cepdiag_p, wg_ch, scale, offset in params_todo:
        var_idx = era5_vars.index(wg_ch)
        _logger.info(
            "Parameter %s  (ERA5 var %r, idx=%d)", cepdiag_p, wg_ch, var_idx
        )
        for init_date in init_dates:
            written = _process_param(
                z             = era5_z,
                regridder     = regridder,
                var_idx       = var_idx,
                cepdiag_param = cepdiag_p,
                scale         = scale,
                offset        = offset,
                init_date_str = init_date,
                n_weeks       = n_weeks,
                era5_stride   = era5_stride,
                out_dir       = stage_dir,
                expid         = expid,
                lats          = lats,
                lons          = lons,
                overwrite     = args.overwrite,
            )
            if written:
                n_written += 1

    _logger.info(
        "Done.  %d ERA5 analysis file(s) written to %s", n_written, stage_dir
    )


if __name__ == "__main__":
    main()
