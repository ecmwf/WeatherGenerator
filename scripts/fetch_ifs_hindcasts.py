#!/usr/bin/env python3
"""
Fetch IFS extended-range hindcasts (MOFC / ENFH stream) from the ECMWF Web API
and stage them as CEPDIAG-compatible NetCDF files.

Prerequisites
-------------
1.  Register at https://accounts.ecmwf.int  (free)
2.  Get your API key from https://api.ecmwf.int/v1/key/
3.  Create ~/.ecmwfapirc:
        {
            "url"   : "https://api.ecmwf.int/v1",
            "key"   : "YOUR_KEY_HERE",
            "email" : "your@email.com"
        }
4.  pip install ecmwf-api-client cfgrib xarray eccodes
    (package name: ecmwfapi — import as: from ecmwfapi import ECMWFDataServer)

Usage
-----
  # Retrieve all hindcast years for a single ref date and 4 params
  python3 scripts/fetch_ifs_hindcasts.py \\
      --run-id ww9atcoz \\
      --ref-date 20250801 \\
      --hcfromyear 2005 --hctoyear 2024 \\
      --params t2m,mslp,z500,t850

  # Also retrieve the actual 2025 forecast (not a hindcast)
  python3 scripts/fetch_ifs_hindcasts.py \\
      --run-id ww9atcoz \\
      --ref-date 20250801 \\
      --params t2m,mslp,z500,t850 \\
      --forecast-only

IFS data note
-------------
Hindcasts (hdate mode):
    class=od  stream=enfh (≤47r3) or eefh (≥48r1, from 2023-06-27)
    date = ref-date  (the "real" forecast date)
    hdate = 20050801/20060801/.../20240801
    type = cf (member 0)  or  pf (members 1-10)

The script computes 6-hourly → weekly means matching the WG conventions
(same as era5_to_cepdiag.py) so the staged IFS files are directly
comparable in CEPDIAG.

Output files land in the same stage directory as the WG forecasts:
    results/<run_id>/cepdiag/eval/stage/0001_ens_<m>_<param>_<YYYYMMDD>.nc
Uncomment [fc_0001] in cepdiag_mofc.conf to activate the IFS experiment.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# Parameter definitions:  cepdiag_id → (grib_code, level, nc_varname, src_units, target_units)
# ---------------------------------------------------------------------------
PARAM_DEFS: dict[str, dict] = {
    "t2m":  {"code": "167.128", "level": "sfc",  "nname": "t2m", "src": "K",         "tgt": "K"},
    "mslp": {"code": "151.128", "level": "sfc",  "nname": "msl", "src": "Pa",         "tgt": "Pa"},
    "z500": {"code": "129.128", "level": "500",   "nname": "z",   "src": "m^2/s^2",   "tgt": "m^2/s^2"},
    "z200": {"code": "129.128", "level": "200",   "nname": "z",   "src": "m^2/s^2",   "tgt": "m^2/s^2"},
    "t850": {"code": "130.128", "level": "850",   "nname": "t",   "src": "K",         "tgt": "K"},
    "t500": {"code": "130.128", "level": "500",   "nname": "t",   "src": "K",         "tgt": "K"},
}

# IFS mofc: control + 10 perturbed members
IFS_EXPID   = "0001"
IFS_ENSSIZE = 11   # members 0-10
# IFS extended-range: 6-hourly steps for 6 weeks = 1008 h
IFS_ATMFREQ    = 6                     # hours between IFS output steps
IFS_N_WEEKS    = 6
HOURS_PER_WEEK = 7 * 24               # 168

# IFS cycle that introduced the EEFH stream (2023-06-27, cycle 48r1)
_CYCLE_48R1 = datetime(2023, 6, 27)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ifs_stream(ref_date_str: str, hdate: bool) -> str:
    """Return the correct IFS stream keyword (ENFH vs EEFH)."""
    ref_dt = datetime.strptime(ref_date_str, "%Y%m%d")
    if ref_dt >= _CYCLE_48R1:
        return "eefh" if hdate else "eefo"
    return "enfh" if hdate else "enfo"


def _hdate_list(ref_date_str: str, fromyear: int, toyear: int) -> list[str]:
    """Return list of hindcast dates e.g. ['20050801','20060801',...]."""
    ref_dt = datetime.strptime(ref_date_str, "%Y%m%d")
    return [f"{y:04d}{ref_dt.month:02d}{ref_dt.day:02d}" for y in range(fromyear, toyear + 1)]


def _weekly_means_from_grib(
    grib_path: Path,
    varname: str,
    n_weeks: int,
    atmfreq: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single-member GRIB file, group 6-hourly steps into weekly means.

    Returns
    -------
    lats  : (nlat,) degrees_north
    lons  : (nlon,) degrees_east
    data  : (n_weeks, nlat, nlon) float32
    """
    import cfgrib
    import xarray as xr

    ds = xr.open_dataset(str(grib_path), engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    # find the data variable
    var = None
    for v in ds.data_vars:
        var = v
        break
    da = ds[var]  # dims: (step, latitude, longitude) or similar

    steps_per_week = HOURS_PER_WEEK // atmfreq   # 28 for 6h
    total_steps    = n_weeks * steps_per_week
    # da.values shape: (total_steps, nlat, nlon)
    arr = da.values[:total_steps].reshape(n_weeks, steps_per_week,
                                          da.shape[-2], da.shape[-1])
    weekly = arr.mean(axis=1).astype(np.float32)  # (n_weeks, nlat, nlon)
    lats = da.latitude.values
    lons = da.longitude.values
    return lats, lons, weekly


def _write_nc(
    out_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
    varname: str,
    n_weeks: int,
    init_date_str: str,
):
    """Write a CEPDIAG-compatible NetCDF file (same layout as era5_to_cepdiag)."""
    import netCDF4 as nc4

    with nc4.Dataset(str(out_path), "w", format="NETCDF4") as ds:
        ds.createDimension("time",      n_weeks)
        ds.createDimension("latitude",  len(lats))
        ds.createDimension("longitude", len(lons))
        ds.Conventions    = "CF-1.8"
        ds.init_date      = init_date_str
        ds.source         = "IFS MOFC hindcast via ECMWF Web API"

        lat_v = ds.createVariable("latitude",  "f4", ("latitude",))
        lat_v.units = "degrees_north"
        lat_v[:] = lats

        lon_v = ds.createVariable("longitude", "f4", ("longitude",))
        lon_v.units = "degrees_east"
        lon_v[:] = lons

        t_v = ds.createVariable("time", "i4", ("time",))
        t_v.units = "days since init"
        t_v[:] = [7 * (w + 1) for w in range(n_weeks)]

        v = ds.createVariable(varname, "f4", ("time", "latitude", "longitude"),
                              fill_value=9.96921e+36, zlib=True, complevel=4)
        v[:] = data


# ---------------------------------------------------------------------------
# Core retrieval logic
# ---------------------------------------------------------------------------

def _fetch_member(
    server,
    ref_date: str,
    hdates: list[str],    # empty → forecast mode (no hdate)
    exptype: str,          # "cf" or "pf"
    member: int | None,    # None for cf
    param_code: str,
    level: str,
    steps: list[int],
    tmpdir: Path,
) -> Path:
    """Submit one MARS request and return path to downloaded GRIB file."""
    is_hdate = bool(hdates)
    stream = _ifs_stream(ref_date, hdate=is_hdate)
    target = str(tmpdir / f"ifs_{exptype}_{member}_{param_code}_{level}.grb")

    req: dict = {
        "class":  "od",
        "expver": "1",
        "stream": stream,
        "type":   exptype,
        "date":   ref_date,
        "time":   "0000",
        "step":   "/".join(str(s) for s in steps),
        "param":  param_code,
        "grid":   "1.0/1.0",   # 1°×1° for CEPDIAG compatibility
    }
    if level != "sfc":
        req["levtype"]   = "pl"
        req["levelist"]  = level
    else:
        req["levtype"] = "sfc"

    if is_hdate:
        req["hdate"] = "/".join(hdates)

    if exptype == "pf":
        req["number"] = str(member)

    _log.info("Submitting MARS request: stream=%s date=%s %s members=%s",
              stream, ref_date, f"hdate={hdates[0]}..{hdates[-1]}" if is_hdate else "", member)
    server.execute(req, target)
    return Path(target)


def retrieve_and_stage(
    run_id: str,
    results_dir: Path,
    ref_date: str,
    hcfromyear: int | None,
    hctoyear: int | None,
    params: list[str],
    members: list[int],
    n_weeks: int,
    forecast_only: bool,
    overwrite: bool,
) -> None:
    from ecmwfapi import ECMWFService

    stage_dir = results_dir / run_id / "cepdiag" / "eval" / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)

    server = ECMWFService("mars")

    if forecast_only or hcfromyear is None:
        date_list = [ref_date]
        hdates = []
    else:
        date_list = _hdate_list(ref_date, hcfromyear, hctoyear)
        hdates    = date_list   # all hindcast dates in one request

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        for param_id in params:
            if param_id not in PARAM_DEFS:
                _log.warning("Unknown param '%s' — skipping", param_id)
                continue
            pdef = PARAM_DEFS[param_id]
            code, level, ncname = pdef["code"], pdef["level"], pdef["nname"]

            for member in members:
                exptype = "cf" if member == 0 else "pf"
                mem_arg = None if member == 0 else member

                # Check if all output files already exist
                if not forecast_only and hdates:
                    missing = [d for d in date_list
                               if not (stage_dir / f"{IFS_EXPID}_ens_{member}_{param_id}_{d}.nc").exists()]
                    if not missing and not overwrite:
                        _log.info("All %d files for %s m%d already exist — skipping", len(date_list), param_id, member)
                        continue
                    target_dates = date_list  # all at once
                else:
                    target_dates = date_list

                # --- fetch ---
                steps = list(range(IFS_ATMFREQ, n_weeks * HOURS_PER_WEEK + 1, IFS_ATMFREQ))
                grib_path = _fetch_member(
                    server, ref_date,
                    hdates if not forecast_only else [],
                    exptype, mem_arg, code, level, steps, tmpdir,
                )

                # --- decode GRIB, split by init date, write NC ---
                _log.info("Decoding and staging %s member %d ...", param_id, member)
                import cfgrib, xarray as xr
                ds = xr.open_dataset(str(grib_path), engine="cfgrib",
                                     backend_kwargs={"indexpath": ""})
                da_var = ds[list(ds.data_vars)[0]]  # main data array

                # For hindcasts GRIB has dim 'hdate'; for forecasts it has 'valid_time'
                if "hdate" in da_var.dims or "hdate" in da_var.coords:
                    date_coord = da_var["hdate"].values
                    for i, hd in enumerate(date_coord):
                        hd_str = str(hd).replace("-", "")[:8]
                        out = stage_dir / f"{IFS_EXPID}_ens_{member}_{param_id}_{hd_str}.nc"
                        if out.exists() and not overwrite:
                            continue
                        arr = da_var.isel(hdate=i).values  # (step, lat, lon)
                        steps_pw = HOURS_PER_WEEK // IFS_ATMFREQ
                        weekly = arr[:n_weeks * steps_pw].reshape(
                            n_weeks, steps_pw, arr.shape[-2], arr.shape[-1]).mean(axis=1).astype(np.float32)
                        _write_nc(out, da_var.latitude.values, da_var.longitude.values,
                                  weekly, param_id, n_weeks, hd_str)
                        _log.info("  wrote %s", out.name)
                else:
                    # single forecast date
                    out = stage_dir / f"{IFS_EXPID}_ens_{member}_{param_id}_{ref_date}.nc"
                    if out.exists() and not overwrite:
                        _log.info("  %s exists — skipping", out.name)
                        continue
                    arr = da_var.values  # (step, lat, lon)
                    steps_pw = HOURS_PER_WEEK // IFS_ATMFREQ
                    weekly = arr[:n_weeks * steps_pw].reshape(
                        n_weeks, steps_pw, arr.shape[-2], arr.shape[-1]).mean(axis=1).astype(np.float32)
                    _write_nc(out, da_var.latitude.values, da_var.longitude.values,
                              weekly, param_id, n_weeks, ref_date)
                    _log.info("  wrote %s", out.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-id",  "-r", required=True,
                   help="WeatherGenerator run ID (sets the output stage/ path)")
    p.add_argument("--results-dir", default=None,
                   help="Root results directory (default: <repo>/results/)")
    p.add_argument("--ref-date", required=True, metavar="YYYYMMDD",
                   help="Reference forecast date, e.g. 20250801")
    p.add_argument("--hcfromyear", type=int, default=None, metavar="YEAR",
                   help="First hindcast year (e.g. 2005). Omit for forecast-only.")
    p.add_argument("--hctoyear",   type=int, default=None, metavar="YEAR",
                   help="Last hindcast year (e.g. 2024). Omit for forecast-only.")
    p.add_argument("--params", default="t2m,mslp,z500,t850",
                   help=f"Comma-separated param IDs (default: t2m,mslp,z500,t850). "
                        f"Available: {','.join(PARAM_DEFS)}")
    p.add_argument("--members", default="0-10",
                   help="Member range or list, e.g. '0-10' or '0,1,2' (default: 0-10)")
    p.add_argument("--n-weeks", type=int, default=IFS_N_WEEKS,
                   help=f"Number of forecast weeks (default: {IFS_N_WEEKS})")
    p.add_argument("--forecast-only", action="store_true",
                   help="Only retrieve the actual ref-date forecast (no hindcasts)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download/overwrite existing files")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def _parse_members(s: str) -> list[int]:
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    repo_root   = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else repo_root / "results"
    params      = [p.strip() for p in args.params.split(",")]
    members     = _parse_members(args.members)

    _log.info("IFS hindcast retrieval")
    _log.info("  ref-date     : %s", args.ref_date)
    if not args.forecast_only and args.hcfromyear:
        ndates = args.hctoyear - args.hcfromyear + 1
        _log.info("  hindcast years: %d-%d  (%d dates)", args.hcfromyear, args.hctoyear, ndates)
    _log.info("  params       : %s", params)
    _log.info("  members      : %s", members)
    _log.info("  output dir   : %s", results_dir / args.run_id / "cepdiag/eval/stage")

    retrieve_and_stage(
        run_id       = args.run_id,
        results_dir  = results_dir,
        ref_date     = args.ref_date,
        hcfromyear   = args.hcfromyear,
        hctoyear     = args.hctoyear,
        params       = params,
        members      = members,
        n_weeks      = args.n_weeks,
        forecast_only= args.forecast_only,
        overwrite    = args.overwrite,
    )
    _log.info("Done.")


if __name__ == "__main__":
    main()
