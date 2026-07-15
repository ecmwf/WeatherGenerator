# pylint: disable=bad-builtin

import contextlib
import logging
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf

from weathergen.evaluate.export.cf_utils import CfParser

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

"""
Usage:

uv run export --run-id z8mi61v4 --stream ICON
--output-dir ./out
--format evalml --type prediction
--quaver-template-folder /path/to/evalml/resources/inference/templates
--quaver-template-grid-type icon-ch1

evalML expects one GRIB file per lead time per forecast initialisation:

    {output_dir}/{YYYYmmddHHMM}/{YYYYmmddHHMM}_{step:03d}.grib

Each file holds every requested variable at that lead time. Total precipitation
is written cumulative-from-start (evalML diffs it back to hourly values).

This parser mirrors scripts/zarr_to_grib.py: it clones the per-typeOfLevel GRIB
templates shipped with evalML and fills in metadata + values with eccodes,
keeping the data in the native (ICON) grid order.
"""

# WeatherGenerator surface channel -> GRIB template + shortName + level.
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

# Pressure-level channel prefix -> GRIB shortName (names like T_850, QV_500, FI_250).
PLEVEL_CHANNELS: dict[str, str] = {"T": "t", "QV": "q", "U": "u", "V": "v", "FI": "z"}

# typeOfLevel key -> template filename, formatted with the grid type (e.g. icon-ch1).
TEMPLATE_FILES: dict[str, str] = {
    "heightAboveGround": "{grid}-typeOfLevel=heightAboveGround.grib",
    "surface": "{grid}-typeOfLevel=surface.grib",
    "meanSea": "{grid}-typeOfLevel=meanSea.grib",
    "TOT_PREC": "{grid}-shortName=TOT_PREC.grib",
    "isobaricInhPa": "{grid}-typeOfLevel=isobaricInhPa.grib",
}


class EvalmlParser(CfParser):
    """evalML GRIB exporter (one file per lead time, per init time)."""

    def __init__(self, config: OmegaConf, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if not getattr(self, "quaver_template_folder", None):
            raise ValueError("Template folder must be provided for evalml format.")
        if not getattr(self, "quaver_template_grid_type", None):
            raise ValueError("Template grid type (e.g. icon-ch1) must be provided for evalml.")
        if not getattr(self, "channels", None):
            raise ValueError("Channels must be provided for evalml format.")

        super().__init__(config, **kwargs)

        self.templates = self._load_templates()
        if not self.templates:
            raise RuntimeError(
                f"No templates loaded from {self.quaver_template_folder} "
                f"for grid '{self.quaver_template_grid_type}'."
            )

    def _load_templates(self) -> dict[str, object]:
        """Read one GRIB message from each template file; return {template_key: handle}."""
        folder = Path(self.quaver_template_folder)
        grid = self.quaver_template_grid_type
        templates = {}
        for key, pattern in TEMPLATE_FILES.items():
            path = folder / pattern.format(grid=grid)
            if not path.exists():
                _logger.warning("Template not found, skipping %s: %s", key, path)
                continue
            with open(path, "rb") as f:
                templates[key] = eccodes.codes_grib_new_from_file(f)
            _logger.info("Loaded template: %s (%s)", key, path.name)
        return templates

    def process_sample(
        self,
        fstep_iterator_results: iter,
        ref_time: np.datetime64,
        source_interval_start: np.datetime64 = None,
        source_interval_end: np.datetime64 = None,
    ):
        """Write one GRIB file per lead time into a per-init-time directory."""
        init_dt = pd.Timestamp(source_interval_end)
        stamp = init_dt.strftime("%Y%m%d%H%M")
        sample_dir = Path(self.output_dir) / stamp
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Collect every hourly sub-step across all fsteps, keyed by valid_time,
        # so they can be written (and precipitation accumulated) in time order.
        subs: dict[np.datetime64, xr.DataArray] = {}
        for result in fstep_iterator_results:
            if result is None:
                continue
            if not isinstance(result, xr.DataArray):
                result = result.as_xarray().squeeze()
            result = result.sel(channel=self.channels)
            for vt in np.unique(result.valid_time.values):
                mask = result.valid_time.values == vt
                subs[vt] = result.isel(ipoint=mask)

        cumulative: dict[str, np.typing.NDArray] = {}
        for vt in sorted(subs):
            da_sub = subs[vt]
            step_h = int((vt - np.datetime64(source_interval_end)) / np.timedelta64(1, "h"))
            out_file = sample_dir / f"{stamp}_{step_h:03d}.grib"
            if out_file.exists():
                out_file.unlink()

            n_msgs = 0
            for var in self.channels:
                spec = self._channel_spec(var)
                if spec is None:
                    continue
                template_key, short_name, level, accumulated = spec
                if template_key not in self.templates:
                    _logger.warning("No template '%s' for %s, skipping", template_key, var)
                    continue

                values = self.scale_data(da_sub.sel(channel=var), var).values
                if accumulated:
                    cumulative[var] = values if var not in cumulative else cumulative[var] + values
                    values = cumulative[var]

                self._write_field(
                    out_file, template_key, values, init_dt, step_h, short_name, level, accumulated
                )
                n_msgs += 1

            _logger.info(
                "  step %03d  valid=%s  msgs=%d", step_h, np.datetime_as_string(vt, "h"), n_msgs
            )

        _logger.info(f"Saved sample data to {self.output_format} in {sample_dir}.")

    @staticmethod
    def _channel_spec(var: str) -> tuple[str, str, int, bool] | None:
        """Return (template_key, shortName, level, accumulated) for a channel, or None."""
        if var in SURFACE_CHANNELS:
            cfg = SURFACE_CHANNELS[var]
            return (
                cfg["template_key"],
                cfg["shortName"],
                cfg["level"],
                cfg.get("accumulated", False),
            )
        parts = var.rsplit("_", 1)
        if len(parts) == 2 and parts[0] in PLEVEL_CHANNELS and parts[1].isdigit():
            return "isobaricInhPa", PLEVEL_CHANNELS[parts[0]], int(parts[1]), False
        _logger.warning("No evalml mapping for channel '%s', skipping", var)
        return None

    def _write_field(
        self,
        out_path: Path,
        template_key: str,
        values: np.typing.NDArray,
        init_dt: pd.Timestamp,
        step_h: int,
        short_name: str,
        level: int,
        accumulated: bool,
    ) -> None:
        """Clone the template, set metadata + values, and append the message to out_path."""
        msg = eccodes.codes_clone(self.templates[template_key])
        try:
            eccodes.codes_set(msg, "shortName", short_name)
            eccodes.codes_set(msg, "level", level)
            eccodes.codes_set(msg, "dataDate", int(init_dt.strftime("%Y%m%d")))
            eccodes.codes_set(msg, "dataTime", int(init_dt.strftime("%H%M")))
            eccodes.codes_set(msg, "stepUnits", 1)  # hours
            if accumulated:
                eccodes.codes_set(msg, "startStep", 0)
            eccodes.codes_set(msg, "endStep", step_h)
            eccodes.codes_set_values(msg, np.asarray(values, dtype=float))
            with open(out_path, "ab") as f:
                eccodes.codes_write(msg, f)
        finally:
            eccodes.codes_release(msg)

    def __del__(self):
        for msg in getattr(self, "templates", {}).values():
            with contextlib.suppress(Exception):  # best-effort cleanup
                eccodes.codes_release(msg)
