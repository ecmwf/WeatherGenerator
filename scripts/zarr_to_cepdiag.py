#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
"""
zarr_to_cepdiag.py — Convert WeatherGenerator Zarr output to CEPDIAG staged NetCDF.

Reads validation_chkpt*_rank*.zip files from a WeatherGenerator results directory,
regrids from the model's reduced Gaussian grid to a regular lat/lon grid,
computes weekly means, and writes per-variable CEPDIAG-compatible NetCDF files.

Output layout
─────────────
  {results_dir}/{run_id}/cepdiag/
    eval/
      stage/                          ← CEPDIAG stagedir (datadir/stage)
        {run_id}_ens_{m}_{param}_{date}.nc
      metrics/                        ← CEPDIAG writes metric output here
      plots/                          ← CEPDIAG writes plots here
    cepdiag_mofc.conf                 ← ready-to-use CEPDIAG config template

NetCDF file format (per variable, per member, per start date)
─────────────────────────────────────────────────────────────
  Dimensions : (time, latitude, longitude)
  time       : valid datetime of each weekly mean (first day of the averaging window)
  latitude   : regular, south→north, step = lat_res  (e.g., -89.5 … 89.5)
  longitude  : regular, west→east,   step = lon_res  (e.g., -179.5 … 179.5)
  nstepmean  : 1  (weekly means are pre-computed; set nstepmean=1 in cepdiag.conf)

Memory strategy
───────────────
  Data is processed channel-by-channel, week-by-week, to bound peak memory use
  to O(n_steps_per_week × n_cells) ≈ 28 × 40320 × 4 bytes ≈ 4.5 MB per field.
  The full step loop is traversed once per channel to avoid loading all 83
  channels simultaneously.

Usage
─────
  # Basic (uses results/ relative to WeatherGenerator root):
  python scripts/zarr_to_cepdiag.py --run-id af25nepk

  # Explicit paths and options:
  python scripts/zarr_to_cepdiag.py \\
      --run-id af25nepk \\
      --results-dir /path/to/results \\
      --params t2m,z500,t850,mslp,u850,u200 \\
      --lon-res 1.0 --lat-res 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
import zarr.storage
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

_logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# WeatherGenerator channel → (CEPDIAG param id, unit scale, unit offset)
#
# CEPDIAG's paramdefs.xml defines sunits (source/staging units).  These match
# ERA5/IFS native units from MARS, which are also what WeatherGenerator uses.
# No unit conversion is therefore needed for any of the standard fields here.
# ──────────────────────────────────────────────────────────────────────────────
CHANNEL_MAP: dict[str, tuple[str, float, float]] = {
    # Surface fields
    "2t":    ("t2m",   1.0, 0.0),   # near-surface temperature          [K]
    "msl":   ("mslp",  1.0, 0.0),   # mean sea-level pressure            [Pa]
    "skt":   ("tskin", 1.0, 0.0),   # skin temperature                   [K]
    "sp":    ("sp",    1.0, 0.0),   # surface pressure                   [Pa]
    "tcc":   ("tcc",   1.0, 0.0),   # total cloud cover                  [1]
    "10u":   ("u10m",  1.0, 0.0),   # 10 m eastward wind                 [m/s]
    "10v":   ("v10m",  1.0, 0.0),   # 10 m northward wind                [m/s]
    # Pressure-level fields – geopotential (m²/s², CEPDIAG sunits = m²/s²)
    "z_500": ("z500",  1.0, 0.0),   # 500 hPa geopotential               [m²/s²]
    "z_200": ("z200",  1.0, 0.0),   # 200 hPa geopotential               [m²/s²]
    "z_50":  ("z50",   1.0, 0.0),   # 50 hPa geopotential                [m²/s²]
    # Pressure-level fields – temperature                                 [K]
    "t_850": ("t850",  1.0, 0.0),
    "t_500": ("t500",  1.0, 0.0),
    "t_200": ("t200",  1.0, 0.0),
    # Pressure-level fields – u-wind                                      [m/s]
    "u_850": ("u850",  1.0, 0.0),
    "u_200": ("u200",  1.0, 0.0),
    "u_100": ("u100",  1.0, 0.0),
    "u_50":  ("u50",   1.0, 0.0),
}

# Default CEPDIAG-relevant params (subset of CHANNEL_MAP keys)
# DEFAULT_PARAMS = ["t2m", "mslp", "z500", "z200", "t850", "u850", "u200", "tcc"]
DEFAULT_PARAMS = ["t2m", "mslp", "z500"]

HOURS_PER_STEP = 6
STEPS_PER_DAY  = 24 // HOURS_PER_STEP   # = 4
DAYS_PER_WEEK  = 7
STEPS_PER_WEEK = DAYS_PER_WEEK * STEPS_PER_DAY  # = 28


# ──────────────────────────────────────────────────────────────────────────────
# Zarr helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_json_from_zip(zip_path: Path, inner_path: str) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(inner_path))


def _open_store(zip_path: Path) -> tuple[zarr.storage.ZipStore, zarr.Group]:
    store = zarr.storage.ZipStore(zip_path, mode="r")
    root  = zarr.open(store, mode="r")
    return store, root


def _valid_steps(root: zarr.Group, prefix: str) -> list[int]:
    """Return sorted list of forecast steps that contain prediction data."""
    steps = []
    for key in root[prefix].keys():
        try:
            step = int(key)
        except ValueError:
            continue
        t = root[f"{prefix}/{key}/target/times"][:]
        if len(t) > 0:
            steps.append(step)
    return sorted(steps)


# ──────────────────────────────────────────────────────────────────────────────
# Regridder — build once, reuse across all fields
# ──────────────────────────────────────────────────────────────────────────────

class ReducedGaussianRegridder:
    """Regrid from a reduced Gaussian (unstructured) grid to a regular lat/lon grid.

    The Delaunay triangulation is built once on construction and reused for all
    subsequent fields.  Date-line wrapping is handled by tiling the source points
    at ±360° before triangulation.
    """

    def __init__(self, src_coords: np.ndarray, tgt_lats: np.ndarray, tgt_lons: np.ndarray):
        self.tgt_shape = (len(tgt_lats), len(tgt_lons))
        n_src = len(src_coords)

        # Tile source points at lon ± 360° to close the date-line gap.
        lats_ext = np.tile(src_coords[:, 0], 3)
        lons_ext = np.concatenate([
            src_coords[:, 1] - 360.0,
            src_coords[:, 1],
            src_coords[:, 1] + 360.0,
        ])
        src_pts_ext = np.column_stack([lats_ext, lons_ext])

        tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lons, tgt_lats)
        self._tgt_pts = np.column_stack([tgt_lat2d.ravel(), tgt_lon2d.ravel()])

        _logger.info("Building Delaunay triangulation for %d source points …", n_src)
        tri = Delaunay(src_pts_ext)
        _logger.info("Triangulation complete.")

        # Build interpolator once; values will be updated in-place per field.
        # Passing the pre-built Delaunay object avoids rebuilding on each call.
        dummy = np.zeros(len(lats_ext))
        self._interp = LinearNDInterpolator(tri, dummy, fill_value=np.nan)

    def regrid(self, src_field: np.ndarray) -> np.ndarray:
        """Regrid a 1-D field (n_cells,) → (n_lats, n_lons)."""
        vals_ext = np.tile(src_field.astype(np.float64), 3)
        # Update values in-place — reuses triangulation without rebuilding.
        self._interp.values = vals_ext[:, np.newaxis]
        return self._interp(self._tgt_pts).reshape(self.tgt_shape)


# ──────────────────────────────────────────────────────────────────────────────
# Per-channel streaming conversion (memory-efficient)
# ──────────────────────────────────────────────────────────────────────────────

def stream_weekly_means_one_channel(
    root:      zarr.Group,
    prefix:    str,                 # e.g. "0/ERA5"
    steps:     list[int],           # valid steps sorted
    ch_idx:    int,                 # channel index within prediction/data
    n_members: int,
) -> tuple[np.ndarray, list[np.datetime64]]:
    """Streaming computation of weekly means for a single channel.

    Reads STEPS_PER_WEEK steps at a time, accumulates the mean in-place,
    and discards each step after adding it.  Peak extra memory per call:
    O(STEPS_PER_WEEK × n_cells × n_members).

    Returns
    -------
    weekly : (n_weeks, n_members, n_cells)  float32
    times  : list of n_weeks np.datetime64 (first valid time of each window)
    """
    n_total   = len(steps)
    n_weeks   = n_total // STEPS_PER_WEEK

    if n_weeks == 0:
        raise ValueError(
            f"Only {n_total} steps available ({n_total * HOURS_PER_STEP}h); "
            f"need at least {STEPS_PER_WEEK} for one 7-day mean."
        )
    if n_total < 6 * STEPS_PER_WEEK:
        _logger.warning(
            "%d/%d complete weekly means available. "
            "Run forecasts to ≥42 days for full S2S range.",
            n_weeks, 6,
        )

    # Peek at n_cells from step 0
    n_cells = root[f"{prefix}/{steps[0]}/prediction/data"].shape[0]

    weekly = np.zeros((n_weeks, n_members, n_cells), dtype=np.float64)
    times: list[np.datetime64] = []

    for w in range(n_weeks):
        s0 = w * STEPS_PER_WEEK
        t0 = root[f"{prefix}/{steps[s0]}/target/times"][0]
        times.append(np.datetime64(t0, "ns"))

        for s_rel in range(STEPS_PER_WEEK):
            step = steps[s0 + s_rel]
            # data shape: (n_cells, n_channels, n_members)
            # Read only the slice for our channel to minimise I/O
            step_data = root[f"{prefix}/{step}/prediction/data"][:, ch_idx, :]
            # step_data: (n_cells, n_members)
            weekly[w] += step_data.T          # (n_members, n_cells)

        weekly[w] /= STEPS_PER_WEEK

    return weekly.astype(np.float32), times


# ──────────────────────────────────────────────────────────────────────────────
# NetCDF output
# ──────────────────────────────────────────────────────────────────────────────

def write_cepdiag_nc(
    outfile:    Path,
    param:      str,
    member:     int,
    init_time:  np.datetime64,
    week_times: list[np.datetime64],
    field:      np.ndarray,         # (n_weeks, n_lats, n_lons)
    lats:       np.ndarray,
    lons:       np.ndarray,
) -> None:
    """Write one CEPDIAG staged NetCDF file.

    Layout expected by cepdiag/cep/metrics/metric.py:
      - variable named by CEPDIAG param id
      - dimensions: (time, latitude, longitude)
      - time : valid datetime of each weekly mean (first day of the window)
    """
    da = xr.DataArray(
        data=field.astype(np.float32),
        dims=["time", "latitude", "longitude"],
        coords={
            "time":      np.array(week_times, dtype="datetime64[ns]"),
            "latitude":  lats,
            "longitude": lons,
        },
        name=param,
        attrs={"long_name": param},
    )
    da.coords["latitude"].attrs["units"]  = "degrees_north"
    da.coords["longitude"].attrs["units"] = "degrees_east"

    ds = da.to_dataset()
    ds.attrs["forecast_reference_time"] = str(init_time)
    ds.attrs["ensemble_member"]         = member
    ds.attrs["Conventions"]             = "CF-1.8"

    ds.to_netcdf(outfile, encoding={param: {"zlib": True, "complevel": 4, "dtype": "float32"}})


# ──────────────────────────────────────────────────────────────────────────────
# CEPDIAG config template
# ──────────────────────────────────────────────────────────────────────────────

_CONF_TEMPLATE = """\
# CEPDIAG configuration for WeatherGenerator run-id: {run_id}
# Auto-generated by zarr_to_cepdiag.py
#
# Notes
# -----
# nstepmean = 1  : weekly means are pre-computed by zarr_to_cepdiag.py
# steps          : one value per week (days from init); matches number of
#                  timesteps in the staged NetCDF files
# verana         : point to a locally staged ERA5 analysis (see CEPDIAG docs)
# fcsystem = mofc: extended-range (monthly forecast) system
#
[setup]
suitename          = {run_id}
user               = {user}
sourcedir          = {cepdiag_dir}
ecfhomeroot        = /path/to/ecflow
# For standalone use (no ECflow), rundir/py must resolve to cep/.
# Create a symlink: ln -sf {cepdiag_dir}/cep {cepdiag_dir}/py
rundir             = {cepdiag_dir}
datadir            = {eval_dir}
obsdir             = /path/to/cepobs

[ecflow_options]
toplevel_suite = tests

[staging]
fcsystem   = mofc
params     = {params}
steps      = {steps}
stepunit   = days
nstepmean  = 1
lonlatres  = {lon_res},{lat_res}
verana     = era5

[ecflow_maps]
metrics    = crps,bias,amrmse,ser
mapview    = glob,nh,sh

{fc_section}

{ds_section}
"""

_FC_SECTION_NOHC = """\
[fc_{run_id}]
group      = {run_id}
enssize    = {n_members}
hcmode     = false
dates      = {dates_csv}"""

_FC_SECTION_HC = """\
# hcfromyear/hctoyear: hindcast period; firstdate = reference (actual forecast) date
[fc_{run_id}]
group      = {run_id}
enssize    = {n_members}
hcmode     = true
hcfromyear = {hcfromyear}
hctoyear   = {hctoyear}
firstdate  = {ref_date}
lastdate   = {ref_date}
datestep   = 1d

# Optional: add IFS as reference — stage 0001_ens_m_param_YYYYMMDD.nc from MARS
# [fc_0001]
# group      = ifs
# enssize    = 11
# hcmode     = true
# hcfromyear = {hcfromyear}
# hctoyear   = {hctoyear}
# firstdate  = {ref_date}
# lastdate   = {ref_date}
# datestep   = 1d"""

_DS_SECTION_NOHC = """\
[ds_all]
dates      = {dates_csv}"""

_DS_SECTION_HC = """\
[ds_all]
firstdate  = {hc_first_date}
lastdate   = {hc_last_date}
datestep   = 1y"""


def write_conf_template(
    conf_path:  Path,
    run_id:     str,
    eval_dir:   Path,
    cepdiag_src: Path,              # absolute path to the cepdiag repo (sourcedir/rundir)
    params:     list[str],
    n_members:  int,
    n_weeks:    int,
    lon_res:    float,
    lat_res:    float,
    init_dates: list[np.datetime64],
    hcmode:     bool = False,
    hcfromyear: int | None = None,
    hctoyear:   int | None = None,
    ref_date:   str | None = None,   # YYYYMMDD reference (actual forecast) date
) -> None:
    import getpass

    cepdiag_params = [
        CHANNEL_MAP[ch][0]
        for ch in CHANNEL_MAP
        if CHANNEL_MAP[ch][0] in params
    ]
    step_labels = ",".join(str(w * DAYS_PER_WEEK) for w in range(n_weeks))

    def _ymd(dt64: np.datetime64) -> str:
        return str(dt64.astype("datetime64[D]")).replace("-", "")

    sorted_dates = sorted(init_dates)
    first_date = _ymd(sorted_dates[0])
    last_date  = _ymd(sorted_dates[-1])

    if hcmode:
        if hcfromyear is None or hctoyear is None or ref_date is None:
            raise ValueError("hcmode requires hcfromyear, hctoyear, and ref_date")
        # Derive hindcast date range from ref_date's month/day
        mm_dd = ref_date[4:]   # e.g. '0801'
        fc_section = _FC_SECTION_HC.format(
            run_id     = run_id,
            n_members  = n_members,
            hcfromyear = hcfromyear,
            hctoyear   = hctoyear,
            ref_date   = ref_date,
        )
        ds_section = _DS_SECTION_HC.format(
            hc_first_date = f"{hcfromyear}{mm_dd}",
            hc_last_date  = f"{hctoyear}{mm_dd}",
        )
    else:
        dates_csv = ",".join(_ymd(d) for d in sorted_dates)
        fc_section = _FC_SECTION_NOHC.format(
            run_id     = run_id,
            n_members  = n_members,
            dates_csv  = dates_csv,
        )
        ds_section = _DS_SECTION_NOHC.format(
            dates_csv  = dates_csv,
        )

    content = _CONF_TEMPLATE.format(
        run_id      = run_id,
        user        = getpass.getuser(),
        eval_dir    = eval_dir,
        cepdiag_dir = cepdiag_src,
        params      = ",".join(cepdiag_params),
        steps       = step_labels,
        lon_res     = lon_res,
        lat_res     = lat_res,
        fc_section  = fc_section,
        ds_section  = ds_section,
    )
    conf_path.write_text(content)
    _logger.info("CEPDIAG config template written to %s", conf_path)


# ──────────────────────────────────────────────────────────────────────────────
# Processing one (sample, stream) group
# ──────────────────────────────────────────────────────────────────────────────

def process_sample(
    zip_path:   Path,
    root:       zarr.Group,
    sample:     int,
    stream:     str,
    out_dir:    Path,
    run_id:     str,
    wg_channels: list[str],         # WeatherGenerator channel names to convert
    regridder:  ReducedGaussianRegridder,
    tgt_lats:   np.ndarray,
    tgt_lons:   np.ndarray,
    overwrite:  bool,
) -> dict | None:
    """Process one (sample, stream) and write its staged NetCDF files.

    Returns a metadata dict or None if the sample is skipped.
    """
    prefix = f"{sample}/{stream}"

    # ── metadata ───────────────────────────────────────────────────────────
    try:
        pred_attrs = _read_json_from_zip(zip_path, f"{prefix}/0/prediction/zarr.json")
        src_attrs  = _read_json_from_zip(zip_path, f"{prefix}/0/source/zarr.json")
    except KeyError as exc:
        _logger.warning("Cannot read metadata for sample=%d stream=%s: %s", sample, stream, exc)
        return None

    channels   = pred_attrs["attributes"]["channels"]
    init_time  = np.datetime64(src_attrs["attributes"]["source_interval"]["start"], "ns")
    date_str   = str(init_time.astype("datetime64[D]")).replace("-", "")
    channel_idx = {ch: i for i, ch in enumerate(channels)}

    # ── valid steps ────────────────────────────────────────────────────────
    steps = _valid_steps(root, prefix)
    if not steps:
        _logger.warning("No valid steps for sample=%d stream=%s – skipping", sample, stream)
        return None

    n_members = root[f"{prefix}/{steps[0]}/prediction/data"].shape[2]
    n_weeks   = len(steps) // STEPS_PER_WEEK
    if n_weeks == 0:
        _logger.warning(
            "sample=%d stream=%s: only %d steps (%dh) – need ≥%d for one weekly mean",
            sample, stream, len(steps), len(steps) * HOURS_PER_STEP, STEPS_PER_WEEK,
        )
        return None

    _logger.info(
        "  sample=%d stream=%s  init=%s  steps=%d  weeks=%d  members=%d",
        sample, stream, date_str, len(steps), n_weeks, n_members,
    )

    # ── convert each requested channel ────────────────────────────────────
    for wg_ch in wg_channels:
        if wg_ch not in channel_idx:
            _logger.debug("Channel %s absent in %s – skipping", wg_ch, zip_path.name)
            continue

        cepdiag_param, scale, offset = CHANNEL_MAP[wg_ch]
        ch_idx = channel_idx[wg_ch]

        # Check whether all output files already exist (skip expensive read)
        outfiles = [
            out_dir / f"{run_id}_ens_{m}_{cepdiag_param}_{date_str}.nc"
            for m in range(n_members)
        ]
        if not overwrite and all(f.exists() for f in outfiles):
            _logger.debug("All member files for %s/%s exist – skipping", date_str, cepdiag_param)
            continue

        # Stream weekly means for this channel (memory-efficient)
        weekly, week_times = stream_weekly_means_one_channel(
            root, prefix, steps, ch_idx, n_members
        )
        # weekly: (n_weeks, n_members, n_cells)

        # Apply unit transform
        if scale != 1.0 or offset != 0.0:
            weekly = weekly * scale + offset

        # Regrid and write per member
        for mem in range(n_members):
            outfile = outfiles[mem]
            if outfile.exists() and not overwrite:
                continue

            field_ll = np.empty((n_weeks, *regridder.tgt_shape), dtype=np.float32)
            for w in range(n_weeks):
                field_ll[w] = regridder.regrid(weekly[w, mem, :])

            write_cepdiag_nc(
                outfile    = outfile,
                param      = cepdiag_param,
                member     = mem,
                init_time  = init_time,
                week_times = week_times,
                field      = field_ll,
                lats       = tgt_lats,
                lons       = tgt_lons,
            )
            _logger.info("    Written: %s", outfile.name)

    return {"init_time": init_time, "n_members": n_members, "n_weeks": n_weeks}


# ──────────────────────────────────────────────────────────────────────────────
# Processing one ZIP file
# ──────────────────────────────────────────────────────────────────────────────

def process_zip(
    zip_path:    Path,
    out_dir:     Path,
    run_id:      str,
    wg_channels: list[str],
    regridder:   ReducedGaussianRegridder,
    tgt_lats:    np.ndarray,
    tgt_lons:    np.ndarray,
    overwrite:   bool,
) -> dict[str, dict]:
    _logger.info("Processing %s …", zip_path.name)
    _, root = _open_store(zip_path)

    meta_out: dict[str, dict] = {}
    for sample_key in sorted(root.keys(), key=int):
        sample = int(sample_key)
        for stream in sorted(root[sample_key].keys()):
            meta = process_sample(
                zip_path    = zip_path,
                root        = root,
                sample      = sample,
                stream      = stream,
                out_dir     = out_dir,
                run_id      = run_id,
                wg_channels = wg_channels,
                regridder   = regridder,
                tgt_lats    = tgt_lats,
                tgt_lons    = tgt_lons,
                overwrite   = overwrite,
            )
            if meta is not None:
                date_str = str(meta["init_time"].astype("datetime64[D]")).replace("-", "")
                meta_out[date_str] = meta

    return meta_out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-id", "-r", required=True,
                   help="WeatherGenerator run-id (e.g. af25nepk)")
    p.add_argument("--results-dir", default=None,
                   help="Root results directory (default: <repo>/results/)")
    p.add_argument(
        "--params", default=None,
        help=(
            "Comma-separated CEPDIAG param IDs to convert "
            f"(default: {','.join(DEFAULT_PARAMS)})"
        ),
    )
    p.add_argument("--lon-res", type=float, default=1.0,
                   help="Target longitude resolution in degrees (default: 1.0)")
    p.add_argument("--lat-res", type=float, default=1.0,
                   help="Target latitude resolution in degrees (default: 1.0)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-write existing NetCDF files (default: skip)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # Hindcast mode (reforecast comparison)
    hc = p.add_argument_group("Hindcast / reforecast mode (hcmode = true)")
    hc.add_argument("--hcmode", action="store_true",
                    help="Generate conf with hcmode=true for multi-year reforecast runs")
    hc.add_argument("--hcfromyear", type=int, default=None, metavar="YEAR",
                    help="First hindcast year (e.g. 2005)")
    hc.add_argument("--hctoyear",   type=int, default=None, metavar="YEAR",
                    help="Last hindcast year (e.g. 2024)")
    hc.add_argument("--ref-date",   default=None, metavar="YYYYMMDD",
                    help="Reference (actual forecast) date for hcmode, e.g. 20250801")
    return p


def _cepdiag_to_wg_channels(param_ids: list[str]) -> list[str]:
    """Map CEPDIAG param IDs back to WeatherGenerator channel names."""
    cepdiag_to_wg = {v[0]: k for k, v in CHANNEL_MAP.items()}
    channels = []
    for pid in param_ids:
        if pid in cepdiag_to_wg:
            channels.append(cepdiag_to_wg[pid])
        else:
            _logger.warning("Unknown CEPDIAG param id '%s' – ignoring", pid)
    return channels


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── paths ──────────────────────────────────────────────────────────────
    repo_root    = Path(__file__).resolve().parent.parent
    results_root = Path(args.results_dir) if args.results_dir else repo_root / "results"
    run_dir      = results_root / args.run_id
    cepdiag_dir  = run_dir / "cepdiag"
    eval_dir     = cepdiag_dir / "eval"
    forecast_dir = eval_dir / "stage"   # = conf.stagedir (datadir + '/stage')

    forecast_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── channels to convert ────────────────────────────────────────────────
    param_ids   = args.params.split(",") if args.params else DEFAULT_PARAMS
    wg_channels = _cepdiag_to_wg_channels(param_ids)
    if not wg_channels:
        raise SystemExit("No valid channels to convert.  Check --params.")
    _logger.info("Params to convert: %s", ", ".join(param_ids))

    # ── target grid ────────────────────────────────────────────────────────
    tgt_lats = np.arange(-90.0 + args.lat_res / 2, 90.0, args.lat_res)
    tgt_lons = np.arange(-180.0 + args.lon_res / 2, 180.0, args.lon_res)
    _logger.info("Target grid: %d lats × %d lons (%.1f° × %.1f°)",
                 len(tgt_lats), len(tgt_lons), args.lat_res, args.lon_res)

    # ── ZIP files ──────────────────────────────────────────────────────────
    zip_files = sorted(run_dir.glob("validation_chkpt*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No validation_chkpt*.zip found in {run_dir}")
    _logger.info("Found %d ZIP file(s)", len(zip_files))

    # ── build regridder from first ZIP ────────────────────────────────────
    _logger.info("Loading source grid from %s …", zip_files[0].name)
    _, root0 = _open_store(zip_files[0])
    src_coords = root0["0/ERA5/0/source/coords"][:]
    regridder  = ReducedGaussianRegridder(src_coords, tgt_lats, tgt_lons)

    # ── process all ZIPs ───────────────────────────────────────────────────
    all_meta: dict[str, dict] = {}
    for zip_path in zip_files:
        meta = process_zip(
            zip_path    = zip_path,
            out_dir     = forecast_dir,
            run_id      = args.run_id,
            wg_channels = wg_channels,
            regridder   = regridder,
            tgt_lats    = tgt_lats,
            tgt_lons    = tgt_lons,
            overwrite   = args.overwrite,
        )
        all_meta.update(meta)

    if not all_meta:
        _logger.error("No output written – check input data.")
        return

    # ── CEPDIAG config template ────────────────────────────────────────────
    last = list(all_meta.values())[-1]
    if args.hcmode and (args.hcfromyear is None or args.hctoyear is None or args.ref_date is None):
        raise SystemExit("--hcmode requires --hcfromyear, --hctoyear, and --ref-date")

    write_conf_template(
        conf_path   = cepdiag_dir / "cepdiag_mofc.conf",
        run_id      = args.run_id,
        eval_dir    = eval_dir,
        cepdiag_src = Path("~/repos/cepdiag").expanduser(),
        params      = param_ids,
        n_members   = last["n_members"],
        n_weeks     = last["n_weeks"],
        lon_res     = args.lon_res,
        lat_res     = args.lat_res,
        init_dates  = [v["init_time"] for v in all_meta.values()],
        hcmode      = args.hcmode,
        hcfromyear  = args.hcfromyear,
        hctoyear    = args.hctoyear,
        ref_date    = args.ref_date,
    )

    _logger.info(
        "Done.  %d init date(s) processed.\n"
        "  Stage dir : %s\n"
        "  Eval dir  : %s\n"
        "  Config    : %s",
        len(all_meta), forecast_dir, eval_dir, cepdiag_dir / "cepdiag_mofc.conf",
    )


if __name__ == "__main__":
    main()
