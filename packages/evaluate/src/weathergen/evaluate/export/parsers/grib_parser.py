"""GRIB export for WeatherGenerator ICON-stream zarr output.

Converts a WeatherGenerator validation zarr zip (ICON stream) to per-lead-time
GRIB files that evalml's load_forecast_data_from_grib can read directly.

Output layout::

    {output_dir}/{YYYYmmddHHMM}/
        {YYYYmmddHHMM}_{step:03d}.grib   (one per hourly lead time)

TOT_PREC_1H is written as cumulative-from-start precipitation; evalml diffs it.

One-time setup before evalml can read the output::

    mkdir -p ~/.local/share/eckit/geo/grid/icon
    curl -fL https://sites.ecmwf.int/repository/eckit/geo/grid/icon-ch/icon-ch1-c.ek \\
        -o ~/.local/share/eckit/geo/grid/icon/\\
17643da2574959b644d254a3cd6e2bc0-b0699f374c63d05028c18c12f80a48f4.ek

    DEFS=evalml/.venv/share/eccodes-cosmo-resources/definitions
    export ECCODES_DEFINITION_PATH=$(realpath $DEFS)
"""

import logging
import time as _time
from datetime import datetime
from pathlib import Path

import eccodes
import numpy as np
import numpy.typing as npt
import zarr

_logger = logging.getLogger(__name__)

# ── Variable definitions ──────────────────────────────────────────────────────

SURFACE_CHANNELS: dict[str, dict] = {
    "T_2M": {"template_key": "heightAboveGround", "shortName": "2t", "level": 2},
    "TD_2M": {"template_key": "heightAboveGround", "shortName": "2d", "level": 2},
    "U_10M": {"template_key": "heightAboveGround", "shortName": "10u", "level": 10},
    "V_10M": {"template_key": "heightAboveGround", "shortName": "10v", "level": 10},
    "PMSL": {"template_key": "meanSea", "shortName": "prmsl", "level": 0},
    "PS": {"template_key": "surface", "shortName": "sp", "level": 0},
    "TOT_PREC_1H": {"template_key": "TOT_PREC", "shortName": "tp", "level": 0, "accumulated": True},
    "TOT_PREC": {"template_key": "TOT_PREC", "shortName": "tp", "level": 0, "accumulated": True},
}

PLEVEL_CHANNELS: dict[str, str] = {
    "T": "t",
    "QV": "q",
    "U": "u",
    "V": "v",
    "FI": "z",
}

TEMPLATE_FILES = {
    "heightAboveGround": "icon-ch1-typeOfLevel=heightAboveGround.grib",
    "surface": "icon-ch1-typeOfLevel=surface.grib",
    "meanSea": "icon-ch1-typeOfLevel=meanSea.grib",
    "TOT_PREC": "icon-ch1-shortName=TOT_PREC.grib",
    "isobaricInhPa": "icon-ch1-typeOfLevel=isobaricInhPa.grib",
}

# ── Template loading ──────────────────────────────────────────────────────────


def load_templates(template_dir: Path) -> dict[str, object]:
    """Read one GRIB message from each template file; return {key: eccodes_handle}."""
    templates = {}
    for key, filename in TEMPLATE_FILES.items():
        path = template_dir / filename
        if not path.exists():
            _logger.warning("Template not found, skipping %s: %s", key, path)
            continue
        with open(path, "rb") as f:
            msg = eccodes.codes_grib_new_from_file(f)
        templates[key] = msg
        _logger.info("Loaded template: %s (%s)", key, filename)
    return templates


def release_templates(templates: dict) -> None:
    for msg in templates.values():
        eccodes.codes_release(msg)


# ── GRIB writing ──────────────────────────────────────────────────────────────


def _set_time(msg, init_dt: datetime, step_h: int, accumulated: bool = False) -> None:
    eccodes.codes_set(msg, "dataDate", int(init_dt.strftime("%Y%m%d")))
    eccodes.codes_set(msg, "dataTime", int(init_dt.strftime("%H%M")))
    eccodes.codes_set(msg, "stepUnits", 1)
    if accumulated:
        eccodes.codes_set(msg, "startStep", 0)
    eccodes.codes_set(msg, "endStep", step_h)


def append_grib_field(
    out_path: Path,
    template_msg,
    values: npt.NDArray[np.float32],
    init_dt: datetime,
    step_h: int,
    short_name: str,
    level: int,
    accumulated: bool = False,
) -> None:
    msg = eccodes.codes_clone(template_msg)
    try:
        eccodes.codes_set(msg, "shortName", short_name)
        eccodes.codes_set(msg, "level", level)
        _set_time(msg, init_dt, step_h, accumulated=accumulated)
        eccodes.codes_set_values(msg, values.astype(float))
        with open(out_path, "ab") as f:
            eccodes.codes_write(msg, f)
    finally:
        eccodes.codes_release(msg)


# ── Zarr reading ──────────────────────────────────────────────────────────────


def _open_zarr(zarr_path: Path):
    if zarr_path.suffix == ".zip":
        store = zarr.storage.ZipStore(str(zarr_path), mode="r")
        return zarr.open(store)
    return zarr.open(str(zarr_path), mode="r")


def _iter_timesteps(stream_group):
    """Yield (valid_time, data_slice) in chronological order, one hourly step at a time."""
    index: list[tuple[str, str, np.datetime64]] = []
    for fstep_key in sorted(stream_group.keys(), key=int):
        fstep = stream_group[fstep_key]
        src = fstep.get("prediction") or fstep.get("target")
        raw_times = src["times"][:]
        for t in np.unique(raw_times):
            index.append((str(t), fstep_key, t))

    index.sort(key=lambda x: x[0])

    current_fstep_key = None
    current_raw_times = None
    current_raw_data = None

    for _t_str, fstep_key, t in index:
        if fstep_key != current_fstep_key:
            fstep = stream_group[fstep_key]
            src = fstep.get("prediction") or fstep.get("target")
            current_raw_times = src["times"][:]
            current_raw_data = src["data"][:]
            if current_raw_data.ndim == 3:
                current_raw_data = current_raw_data[..., 0]
            current_fstep_key = fstep_key

        mask = current_raw_times == t
        yield np.datetime64(t), current_raw_data[mask]


def _init_time(stream_group) -> datetime:
    attrs = (
        stream_group[sorted(stream_group.keys(), key=int)[0]]
        .get("prediction", stream_group[sorted(stream_group.keys(), key=int)[0]].get("target"))
        .attrs
    )
    end_str = attrs["source_interval"]["end"]
    t = np.datetime64(end_str).astype("datetime64[ms]").astype(datetime)
    return t.replace(tzinfo=None)


# ── Per-sample conversion ─────────────────────────────────────────────────────


def _convert_sample(
    stream_group,
    output_dir: Path,
    templates: dict,
    variables: list[str],
    include_pressure_levels: bool,
) -> None:
    """Convert one forecast sample to a directory of GRIB files."""
    init_dt = _init_time(stream_group)
    _logger.info("─" * 60)
    _logger.info("Init time : %s", init_dt.strftime("%Y-%m-%dT%H:%M"))

    first_key = sorted(stream_group.keys(), key=int)[0]
    src = stream_group[first_key].get("prediction") or stream_group[first_key].get("target")
    channels: list[str] = list(src.attrs["channels"])
    n_fsteps = sum(1 for _ in stream_group.keys())

    n_steps_total = sum(
        len(np.unique(stream_group[k].get("prediction", stream_group[k].get("target"))["times"][:]))
        for k in stream_group.keys()
    )

    sfc_vars_out = [v for v in variables if v in SURFACE_CHANNELS and v in channels]
    prec_channel = next((c for c in ("TOT_PREC_1H", "TOT_PREC") if c in channels), None)
    want_precip = prec_channel is not None and any(
        v in variables for v in ("TOT_PREC_1H", "TOT_PREC")
    )
    cumulative_prec: npt.NDArray[np.float32] | None = None

    plevel_vars_out: list[str] = []
    if include_pressure_levels:
        for ch in channels:
            parts = ch.rsplit("_", 1)
            if len(parts) == 2 and parts[0] in PLEVEL_CHANNELS:
                try:
                    int(parts[1])
                    plevel_vars_out.append(ch)
                except ValueError:
                    pass

    _logger.info("Fsteps    : %d  →  %d hourly steps total", n_fsteps, n_steps_total)
    _logger.info("Channels  : %d in zarr", len(channels))
    _logger.info("Sfc vars  : %s", ", ".join(sfc_vars_out) if sfc_vars_out else "(none)")
    if include_pressure_levels:
        _logger.info(
            "PL vars   : %d (e.g. %s)", len(plevel_vars_out), ", ".join(plevel_vars_out[:4])
        )
    if want_precip:
        _logger.info("Precip    : %s → TOT_PREC (cumulative-from-start)", prec_channel)

    sample_dir = output_dir / init_dt.strftime("%Y%m%d%H%M")
    sample_dir.mkdir(parents=True, exist_ok=True)
    _logger.info("Output    : %s", sample_dir)

    t0 = _time.monotonic()
    current_fstep_logged = None
    n_written = 0

    for step_h, (valid_t, data) in enumerate(_iter_timesteps(stream_group)):
        out_file = sample_dir / f"{init_dt.strftime('%Y%m%d%H%M')}_{step_h:03d}.grib"
        if out_file.exists():
            out_file.unlink()

        fstep_block = step_h // 6 + 1
        if fstep_block != current_fstep_logged:
            elapsed = _time.monotonic() - t0
            _logger.info(
                "  fstep %2d / %d  (step %03d, valid %s)  [%.0fs elapsed]",
                fstep_block,
                n_fsteps,
                step_h,
                np.datetime_as_string(valid_t, unit="h"),
                elapsed,
            )
            current_fstep_logged = fstep_block

        if want_precip and prec_channel in channels:
            hourly = data[:, channels.index(prec_channel)]
            cumulative_prec = hourly if cumulative_prec is None else cumulative_prec + hourly

        for var in variables:
            if var not in SURFACE_CHANNELS:
                continue
            cfg = SURFACE_CHANNELS[var]
            tpl_key = cfg["template_key"]
            if tpl_key not in templates:
                continue
            if cfg.get("accumulated"):
                if cumulative_prec is None:
                    continue
                values = cumulative_prec
            else:
                if var not in channels:
                    continue
                values = data[:, channels.index(var)]
            append_grib_field(
                out_path=out_file,
                template_msg=templates[tpl_key],
                values=values,
                init_dt=init_dt,
                step_h=step_h,
                short_name=cfg["shortName"],
                level=cfg["level"],
                accumulated=cfg.get("accumulated", False),
            )

        if include_pressure_levels and "isobaricInhPa" in templates:
            for ch in plevel_vars_out:
                parts = ch.rsplit("_", 1)
                append_grib_field(
                    out_path=out_file,
                    template_msg=templates["isobaricInhPa"],
                    values=data[:, channels.index(ch)],
                    init_dt=init_dt,
                    step_h=step_h,
                    short_name=PLEVEL_CHANNELS[parts[0]],
                    level=int(parts[1]),
                )

        n_written += 1

    elapsed_total = _time.monotonic() - t0
    total_mb = sum(f.stat().st_size for f in sample_dir.glob("*.grib")) / 1024**2
    _logger.info(
        "Done: %d GRIB files, %.0f MB total, %.1f s  →  %s",
        n_written,
        total_mb,
        elapsed_total,
        sample_dir,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def convert_zarr_to_grib(
    zarr_path: Path,
    output_dir: Path,
    templates_dir: Path,
    stream: str = "ICON",
    samples: list[int] | None = None,
    variables: list[str] | None = None,
    include_pressure_levels: bool = False,
) -> None:
    """Convert a WeatherGenerator zarr zip to evalml-compatible GRIB files.

    Parameters
    ----------
    zarr_path:
        Path to the WeatherGenerator zarr zip (or directory).
    output_dir:
        Root output directory; one subdirectory per init time is created inside.
    templates_dir:
        Path to evalml GRIB templates directory (resources/inference/templates/).
    stream:
        Stream name inside the zarr to convert (default: "ICON").
    samples:
        Sample IDs to convert. None converts all samples.
    variables:
        Surface variable names to export. None exports the default set.
    include_pressure_levels:
        Also export pressure-level variables (T, QV, U, V, FI).
    """
    if variables is None:
        variables = ["T_2M", "TD_2M", "U_10M", "V_10M", "TOT_PREC_1H", "PMSL", "PS"]

    _logger.info("Opening zarr: %s", zarr_path)
    root = _open_zarr(zarr_path)

    sample_ids = samples if samples is not None else [int(k) for k in root.keys()]
    _logger.info("Converting %d samples, stream=%s", len(sample_ids), stream)

    templates = load_templates(templates_dir)
    if not templates:
        raise RuntimeError(f"No templates loaded from {templates_dir}. Check the path.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for sid in sample_ids:
        key = str(sid)
        if key not in root:
            _logger.warning("Sample %d not found in zarr, skipping", sid)
            continue
        if stream not in root[key]:
            _logger.warning("Stream %s not found in sample %d, skipping", stream, sid)
            continue
        _convert_sample(
            stream_group=root[key][stream],
            output_dir=output_dir,
            templates=templates,
            variables=variables,
            include_pressure_levels=include_pressure_levels,
        )

    release_templates(templates)
    _logger.info("Done. Output written to %s", output_dir)
