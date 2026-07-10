#!/usr/bin/env python3
"""
Fetch IFS extended-range (re)forecasts from MARS using the ECMWF proprietary
MARS binary and stage them as CEPDIAG-compatible NetCDF files.

Use this script on ECMWF HPC (Atos/Bologna) where `mars` is available.
For external machines use fetch_ifs_hindcasts.py (ECMWF Web API).

Usage
-----
  # Hindcast mode: 20 years × all members × 4 params in 4 MARS requests
  python3 scripts/fetch_ifs_hindcasts_mars.py \\
      --run-id ww9atcoz \\
      --ref-date 20250801 \\
      --hcfromyear 2005 --hctoyear 2024 \\
      --params t2m,mslp,z500,t850

  # Forecast only (no hdate):
  python3 scripts/fetch_ifs_hindcasts_mars.py \\
      --run-id ww9atcoz \\
      --ref-date 20250801 \\
      --forecast-only

  # Dry-run: print MARS requests without executing
  python3 scripts/fetch_ifs_hindcasts_mars.py ... --dry-run

Batching strategy
-----------------
Params are grouped by level type (surface / pressure level).  Within each
group, the control forecast (type=cf, member 0) and all perturbed members
(type=pf, number=1/2/.../N) are retrieved in separate requests.  This reduces
the total number of MARS requests to 2 × n_levtype_groups.

All hindcast years are fetched in one shot via hdate = YYYYMMDD/YYYYMMDD/...

IFS stream selection
--------------------
  stream = eefo / eefh  (≥48r1, ref-date ≥ 2023-06-27)
  stream = enfo / enfh  (<48r1)

Output
------
  results/<run_id>/cepdiag/eval/stage/0001_ens_<m>_<param>_<YYYYMMDD>.nc
Uncomment [fc_0001] in cepdiag_mofc.conf to activate the IFS experiment.
After retrieval, rsync/scp the stage/ directory to your analysis machine.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Parameter definitions: cepdiag_id → grib_code, levtype, level, nc varname
# ---------------------------------------------------------------------------
PARAM_DEFS: dict[str, dict] = {
    "t2m":  {"code": "167.128", "levtype": "sfc", "level": None, "nname": "t2m"},
    "mslp": {"code": "151.128", "levtype": "sfc", "level": None, "nname": "msl"},
    "z500": {"code": "129.128", "levtype": "pl",  "level": 500,  "nname": "z"},
    "z200": {"code": "129.128", "levtype": "pl",  "level": 200,  "nname": "z"},
    "t850": {"code": "130.128", "levtype": "pl",  "level": 850,  "nname": "t"},
    "t500": {"code": "130.128", "levtype": "pl",  "level": 500,  "nname": "t"},
}

IFS_EXPID      = "0001"
IFS_ATMFREQ    = 6                  # hours between IFS output steps
IFS_N_WEEKS    = 6
HOURS_PER_WEEK = 7 * 24             # 168

# IFS cycle 48r1 introduced the EEFH/EEFO streams (2023-06-27)
_CYCLE_48R1 = datetime(2023, 6, 27)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MARS request helpers
# ---------------------------------------------------------------------------

def _ifs_stream(ref_date_str: str, hdate: bool) -> str:
    ref_dt = datetime.strptime(ref_date_str, "%Y%m%d")
    if ref_dt >= _CYCLE_48R1:
        return "eefh" if hdate else "eefo"
    return "enfh" if hdate else "enfo"


def _hdate_list(ref_date_str: str, fromyear: int, toyear: int) -> list[str]:
    ref_dt = datetime.strptime(ref_date_str, "%Y%m%d")
    return [f"{y:04d}{ref_dt.month:02d}{ref_dt.day:02d}"
            for y in range(fromyear, toyear + 1)]


def _mars_request_text(req: dict) -> str:
    """
    Render a dict as a MARS request string.
    List values are joined with '/'; file paths are quoted.

    Example output:
        retrieve,
        class = od,
        stream = eefh,
        target = "/scratch/.../output.grb"
    """
    lines = ["retrieve"]
    for key, val in req.items():
        if isinstance(val, list):
            mars_val = "/".join(str(v) for v in val)
        else:
            mars_val = str(val)
        # paths containing '/' must be quoted in MARS syntax
        if "/" in mars_val and key == "target":
            mars_val = f'"{mars_val}"'
        lines.append(f"{key} = {mars_val}")
    return ",\n".join(lines) + "\n"


def _run_mars(request_text: str, request_file: Path, dry_run: bool) -> None:
    request_file.write_text(request_text)
    if dry_run:
        _log.info("DRY-RUN — MARS request:\n%s", request_text)
    else:
        _log.debug("MARS request:\n%s", request_text)
        with open(request_file) as fh:
            subprocess.check_call("mars", stdin=fh)


# ---------------------------------------------------------------------------
# GRIB → NetCDF staging
# ---------------------------------------------------------------------------

def _write_nc(
    out_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
    param_id: str,
    n_weeks: int,
    init_date_str: str,
) -> None:
    import netCDF4 as nc4

    with nc4.Dataset(str(out_path), "w", format="NETCDF4") as ds:
        ds.createDimension("time",      n_weeks)
        ds.createDimension("latitude",  len(lats))
        ds.createDimension("longitude", len(lons))
        ds.Conventions = "CF-1.8"
        ds.init_date   = init_date_str
        ds.source      = "IFS MOFC via MARS binary"

        lv = ds.createVariable("latitude",  "f4", ("latitude",))
        lv.units = "degrees_north"
        lv[:] = lats

        lov = ds.createVariable("longitude", "f4", ("longitude",))
        lov.units = "degrees_east"
        lov[:] = lons

        tv = ds.createVariable("time", "i4", ("time",))
        tv.units = "days since init"
        tv[:] = [7 * (w + 1) for w in range(n_weeks)]

        v = ds.createVariable(param_id, "f4", ("time", "latitude", "longitude"),
                              fill_value=9.96921e+36, zlib=True, complevel=4)
        v[:] = data


def _weekly_mean(arr: np.ndarray, n_weeks: int) -> np.ndarray:
    """Collapse (step, lat, lon) → (n_weeks, lat, lon) via weekly means."""
    steps_pw = HOURS_PER_WEEK // IFS_ATMFREQ   # 28
    arr = arr[: n_weeks * steps_pw]             # trim to requested weeks
    return arr.reshape(n_weeks, steps_pw, arr.shape[-2], arr.shape[-1]) \
               .mean(axis=1).astype(np.float32)


def _decode_and_stage(
    grib_path: Path,
    stage_dir: Path,
    params_in_grib: list[str],      # cepdiag param ids contained in this file
    cf_or_pf: str,                  # "cf" or "pf"
    pf_members: list[int],          # [1..N], ignored for cf
    is_hdate: bool,
    ref_date: str,
    n_weeks: int,
    overwrite: bool,
) -> None:
    """Decode a GRIB file and write one NC file per param × member × init-date."""
    import cfgrib
    import xarray as xr

    for param_id in params_in_grib:
        pd      = PARAM_DEFS[param_id]
        levtype = pd["levtype"]
        level   = pd["level"]
        nname   = pd["nname"]

        fby: dict = {}
        if levtype == "sfc":
            fby["typeOfLevel"] = "surface"
        else:
            fby["typeOfLevel"] = "isobaricInhPa"
            fby["level"]       = level

        # cfgrib may split a multi-member or multi-hdate GRIB into several
        # datasets; open_datasets() handles that cleanly
        datasets = cfgrib.open_datasets(str(grib_path),
                                        filter_by_keys=fby,
                                        backend_kwargs={"indexpath": ""})
        if not datasets:
            _log.warning("No GRIB messages matched for %s (levtype=%s level=%s)",
                         param_id, levtype, level)
            continue

        # Merge all sub-datasets (e.g. per-member split)
        ds = xr.merge(datasets)

        if nname not in ds:
            _log.warning("Variable '%s' not found in decoded GRIB for %s", nname, param_id)
            continue

        da = ds[nname]   # dims include: step, [number], [hdate/valid_time], latitude, longitude

        members = [0] if cf_or_pf == "cf" else pf_members

        # Determine init-date coordinate
        if is_hdate:
            hdate_coord = da.coords.get("hdate", None)
            if hdate_coord is None:
                # Some cfgrib versions store it as 'time' together with step
                _log.error("Cannot find hdate coordinate in %s for %s", grib_path.name, param_id)
                continue
            date_values = [str(v).replace("-", "")[:8] for v in hdate_coord.values.ravel()]
        else:
            date_values = [ref_date]

        for m_idx, member in enumerate(members):
            # Select member dimension if present
            if "number" in da.dims:
                da_m = da.sel(number=member)
            else:
                da_m = da   # cf: no number dim

            for d_idx, date_str in enumerate(date_values):
                out = stage_dir / f"{IFS_EXPID}_ens_{member}_{param_id}_{date_str}.nc"
                if out.exists() and not overwrite:
                    _log.info("  %s exists — skipping", out.name)
                    continue

                # Select hdate slice
                if is_hdate and "hdate" in da_m.dims:
                    arr = da_m.isel(hdate=d_idx).values   # (step, lat, lon)
                else:
                    arr = da_m.values                      # (step, lat, lon)

                weekly = _weekly_mean(arr, n_weeks)
                _write_nc(out, da_m.latitude.values, da_m.longitude.values,
                          weekly, param_id, n_weeks, date_str)
                _log.info("  wrote %s", out.name)


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve_and_stage(
    run_id:       str,
    results_dir:  Path,
    ref_date:     str,
    hcfromyear:   int | None,
    hctoyear:     int | None,
    params:       list[str],
    cf_members:   list[int],    # must be [0] or []
    pf_members:   list[int],    # e.g. [1..10]
    n_weeks:      int,
    forecast_only: bool,
    overwrite:    bool,
    dry_run:      bool,
    work_dir:     Path | None,
) -> None:
    stage_dir = results_dir / run_id / "cepdiag" / "eval" / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)

    is_hdate = not forecast_only and hcfromyear is not None
    stream   = _ifs_stream(ref_date, hdate=is_hdate)
    hdates   = _hdate_list(ref_date, hcfromyear, hctoyear) if is_hdate else []
    steps    = list(range(IFS_ATMFREQ, n_weeks * HOURS_PER_WEEK + 1, IFS_ATMFREQ))

    _log.info("MARS stream: %s  |  hdate mode: %s  |  steps: %d..%d  (%d total)",
              stream, is_hdate,
              steps[0], steps[-1], len(steps))

    # -----------------------------------------------------------------------
    # Group params for efficient batching:
    #   sfc params (t2m, mslp, ...) → ONE request with param=167.128/151.128
    #   pl params per grib_code     → one request per code (each has its own levels)
    #     e.g. z: levelist=200/500  |  t: levelist=500/850
    # -----------------------------------------------------------------------
    # key: ("sfc", None)  or  ("pl", grib_code)
    # val: [(param_id, level, grib_code), ...]
    groups: dict[tuple[str, str | None], list[tuple[str, int | None, str]]] = defaultdict(list)
    for p in params:
        if p not in PARAM_DEFS:
            _log.warning("Unknown param '%s' — skipping", p)
            continue
        pd = PARAM_DEFS[p]
        if pd["levtype"] == "sfc":
            groups[("sfc", None)].append((p, None, pd["code"]))
        else:
            groups[("pl", pd["code"])].append((p, pd["level"], pd["code"]))

    _log.info("Retrieval groups:")
    for (lt, code), entries in groups.items():
        ids = [x[0] for x in entries]
        codes = sorted({x[2] for x in entries})
        _log.info("  levtype=%-4s  param=%-21s  → %s",
                  lt, "/".join(codes), ids)

    with tempfile.TemporaryDirectory(dir=work_dir) as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        for (levtype, _code_key), entries in groups.items():
            param_ids   = [e[0] for e in entries]
            levels      = sorted({e[1] for e in entries if e[1] is not None})
            grib_codes  = sorted({e[2] for e in entries})

            # Build the shared part of the request
            base_req: dict = {
                "class":  "od",
                "expver": "1",
                "stream": stream,
                "date":   ref_date,
                "time":   "0000",
                "step":   steps,
                "param":  grib_codes if len(grib_codes) > 1 else grib_codes[0],
                "grid":   "1.0/1.0",
            }
            if levtype == "pl":
                base_req["levtype"]  = "pl"
                base_req["levelist"] = levels
            else:
                base_req["levtype"] = "sfc"

            if is_hdate:
                base_req["hdate"] = hdates

            # --- control forecast (type=cf, member 0) ---
            if cf_members:
                cf_tag    = f"{'_'.join(grib_codes).replace('.','_')}"
                cf_target = tmpdir / f"ifs_cf_{levtype}_{cf_tag}.grb"
                cf_req = {**base_req, "type": "cf", "target": str(cf_target)}
                req_file = tmpdir / "marsrequest_cf.req"
                _log.info("Submitting cf request: levtype=%s param=%s", levtype, "/".join(grib_codes))
                _run_mars(_mars_request_text(cf_req), req_file, dry_run)
                if not dry_run:
                    _decode_and_stage(cf_target, stage_dir, param_ids,
                                      "cf", [], is_hdate, ref_date, n_weeks, overwrite)

            # --- perturbed forecast (type=pf, members 1-N) ---
            if pf_members:
                pf_tag    = f"{'_'.join(grib_codes).replace('.','_')}"
                pf_target = tmpdir / f"ifs_pf_{levtype}_{pf_tag}.grb"
                pf_req = {**base_req, "type": "pf",
                          "number": pf_members, "target": str(pf_target)}
                req_file = tmpdir / "marsrequest_pf.req"
                _log.info("Submitting pf request: levtype=%s param=%s  number=%d..%d",
                          levtype, "/".join(grib_codes), pf_members[0], pf_members[-1])
                _run_mars(_mars_request_text(pf_req), req_file, dry_run)
                if not dry_run:
                    _decode_and_stage(pf_target, stage_dir, param_ids,
                                      "pf", pf_members, is_hdate, ref_date, n_weeks, overwrite)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-id", "-r", required=True,
                   help="WeatherGenerator run ID")
    p.add_argument("--results-dir", default=None,
                   help="Root results directory (default: <repo>/results/)")
    p.add_argument("--ref-date", required=True, metavar="YYYYMMDD",
                   help="Reference forecast date, e.g. 20250801")
    p.add_argument("--hcfromyear", type=int, default=None, metavar="YEAR",
                   help="First hindcast year. Omit for forecast-only.")
    p.add_argument("--hctoyear",   type=int, default=None, metavar="YEAR",
                   help="Last hindcast year. Omit for forecast-only.")
    p.add_argument("--params", default="t2m,mslp,z500,t850",
                   help=f"Comma-separated param IDs (default: t2m,mslp,z500,t850). "
                        f"Available: {','.join(PARAM_DEFS)}")
    p.add_argument("--members", default="0-10",
                   help="Member range or list, e.g. '0-10' or '1-5' (default: 0-10)")
    p.add_argument("--n-weeks", type=int, default=IFS_N_WEEKS,
                   help=f"Number of forecast weeks (default: {IFS_N_WEEKS})")
    p.add_argument("--forecast-only", action="store_true",
                   help="Retrieve the actual ref-date forecast (no hindcasts)")
    p.add_argument("--work-dir", default=None,
                   help="Directory for temporary GRIB files (default: system tmpdir). "
                        "Use $SCRATCH on ECMWF HPC.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download/overwrite existing NC files")
    p.add_argument("--dry-run", action="store_true",
                   help="Print MARS requests without executing them")
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
    all_members = _parse_members(args.members)
    cf_members  = [m for m in all_members if m == 0]
    pf_members  = [m for m in all_members if m != 0]
    params      = [p.strip() for p in args.params.split(",")]
    work_dir    = Path(args.work_dir) if args.work_dir else None

    _log.info("IFS hindcast retrieval (MARS binary)")
    _log.info("  ref-date     : %s", args.ref_date)
    if not args.forecast_only and args.hcfromyear:
        n = args.hctoyear - args.hcfromyear + 1
        _log.info("  hindcast years: %d-%d  (%d dates)", args.hcfromyear, args.hctoyear, n)
    _log.info("  params       : %s", params)
    _log.info("  cf members   : %s", cf_members)
    _log.info("  pf members   : %s", pf_members)
    _log.info("  n_weeks      : %d  (%d steps)", args.n_weeks,
              args.n_weeks * HOURS_PER_WEEK // IFS_ATMFREQ)
    _log.info("  output dir   : %s", results_dir / args.run_id / "cepdiag/eval/stage")
    if args.dry_run:
        _log.info("  *** DRY-RUN mode — no MARS calls will be made ***")

    retrieve_and_stage(
        run_id        = args.run_id,
        results_dir   = results_dir,
        ref_date      = args.ref_date,
        hcfromyear    = args.hcfromyear,
        hctoyear      = args.hctoyear,
        params        = params,
        cf_members    = cf_members,
        pf_members    = pf_members,
        n_weeks       = args.n_weeks,
        forecast_only = args.forecast_only,
        overwrite     = args.overwrite,
        dry_run       = args.dry_run,
        work_dir      = work_dir,
    )
    _log.info("Done.")


if __name__ == "__main__":
    main()
