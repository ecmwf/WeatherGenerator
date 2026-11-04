"""Native ICON GRIB2 export of physical (already denormalized) COSMO output.

The reference Zarr supplies coordinates and interval metadata only, never predictions
or normalization statistics. Local products may have unknown ecCodes short names;
the raw parameter, table and fixed-surface keys define their identity.
"""

import logging
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import zarr

from weathergen.evaluate.export.cf_utils import CfParser

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

SURFACE = (1, None, None)
ABSENT = (255, None, None)
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
PRESSURE_PARAMETERS = {
    "z": (0, 3, 4),
    "q": (0, 1, 0),
    "t": (0, 0, 0),
    "u": (0, 2, 2),
    "v": (0, 2, 3),
    "w": (0, 2, 9),
}
FORCINGS = frozenset(
    (
        "cos_julian_day",
        "cos_latitude",
        "cos_local_time",
        "cos_longitude",
        "insolation",
        "sin_julian_day",
        "sin_latitude",
        "sin_local_time",
        "sin_longitude",
    )
)


@dataclass(frozen=True)
class FieldSpec:
    parameter: tuple
    first: tuple = SURFACE
    second: tuple = ABSENT
    processing: int | None = None
    duration: int = 0  # minutes; gust duration comes from recorded metadata
    factor: float = 1.0
    level_type: str = "sfc"


FIELD_SPECS = {
    "100u": FieldSpec((0, 2, 2), (103, 0, 100)),
    "10u": FieldSpec((0, 2, 2), (103, 0, 10)),
    "100v": FieldSpec((0, 2, 3), (103, 0, 100)),
    "10v": FieldSpec((0, 2, 3), (103, 0, 10)),
    "2t": FieldSpec((0, 0, 0), (103, 0, 2)),
    "2d": FieldSpec((0, 0, 6), (103, 0, 2)),
    "skt": FieldSpec((0, 0, 17)),
    "z": FieldSpec((0, 3, 4)),
    "sp": FieldSpec((0, 3, 0)),
    "msl": FieldSpec((0, 3, 1), (101, None, None)),
    "CAPE_ML": FieldSpec((0, 7, 6), (192, None, None)),
    "CAPE_MU": FieldSpec((0, 7, 6), (193, None, None)),
    "CIN_ML": FieldSpec((0, 7, 7), (192, None, None)),
    "CIN_MU": FieldSpec((0, 7, 7), (193, None, None)),
    "LPI": FieldSpec((0, 17, 192)),
    "hcc": FieldSpec((0, 6, 22), (100, -2, 0), (100, -2, 400)),
    "mcc": FieldSpec((0, 6, 22), (100, -2, 400), (100, -2, 800)),
    "lcc": FieldSpec((0, 6, 22), (100, -2, 800), SURFACE),
    "tcc": FieldSpec((0, 6, 1)),
    "lsm": FieldSpec((2, 0, 0)),
    "PLCOV": FieldSpec((2, 0, 4)),
    "SKC": FieldSpec((2, 0, 199)),
    "SOILTYP": FieldSpec((2, 3, 196)),
    "sdor": FieldSpec((0, 3, 20)),
    "anor": FieldSpec((0, 3, 21)),
    "slor": FieldSpec((0, 3, 22)),
    "isor": FieldSpec((0, 3, 24)),
    "W_SO": FieldSpec((2, 3, 20), (106, 2, 1), (106, 2, 3)),
    "tp": FieldSpec((0, 1, 52), processing=1, duration=60),
    "tp_6h": FieldSpec((0, 1, 52), processing=1, duration=360),
    "ro": FieldSpec((2, 0, 5), (106, 2, 0), processing=1, duration=60),
    "ro_6h": FieldSpec((2, 0, 5), (106, 2, 0), processing=1, duration=360),
    "ssrd": FieldSpec((0, 4, 7), processing=0, duration=60, factor=1 / 3600),
    "ssrd_6h": FieldSpec((0, 4, 7), processing=0, duration=360, factor=1 / 21600),
    "strd": FieldSpec((0, 5, 3), processing=0, duration=60, factor=1 / 3600),
    "strd_6h": FieldSpec((0, 5, 3), processing=0, duration=360, factor=1 / 21600),
    "VMAX_10M": FieldSpec((0, 2, 22), (103, 0, 10), processing=2),
}


def _field_spec(name, metadata):
    if name in FIELD_SPECS:
        spec = FIELD_SPECS[name]
        if name == "VMAX_10M":
            period = metadata.get(name, {}).get("period", [])
            if len(period) != 2:
                raise ValueError("VMAX_10M requires a two-ended variables_metadata.period")
            minutes = []
            for value in period:
                match = re.fullmatch(r"(\d+)(m|h)", str(value))
                if not match:
                    raise ValueError(f"Invalid VMAX_10M period: {period}")
                minutes.append(int(match[1]) * (60 if match[2] == "h" else 1))
            if minutes[1] <= minutes[0]:
                raise ValueError(f"Invalid VMAX_10M period: {period}")
            spec = replace(spec, duration=minutes[1] - minutes[0])
        return spec
    match = re.fullmatch(r"(z|q|t|u|v|w)_(\d+)", name)
    if match and int(match[2]) in LEVELS:
        return FieldSpec(PRESSURE_PARAMETERS[match[1]], (100, -2, int(match[2])), level_type="pl")
    raise ValueError(f"Unmapped COSMO physical channel: {name}")


def _identity(spec):
    return (*spec.parameter, *spec.first, *spec.second, spec.processing)


def _prototype(name, spec):
    if spec.level_type == "pl" or name == "z":
        prefix = name.split("_")[0]
        parameter = (0, 2, 8) if prefix == "w" else PRESSURE_PARAMETERS[prefix]
        return "pl", _identity(FieldSpec(parameter, (100, -2, 700 if prefix == "w" else 500)))
    if name in ("100u", "100v"):
        spec = FIELD_SPECS["10" + name[-1]]
    elif name in ("PLCOV", "SKC", "sdor", "anor", "slor", "isor"):
        spec = FIELD_SPECS["lsm"]
    elif name == "skt":
        spec = FieldSpec((0, 0, 0))
    elif name in ("ro", "ro_6h", "tp_6h"):
        spec = FIELD_SPECS["tp"]
    elif name in ("ssrd", "ssrd_6h", "strd_6h"):
        spec = FIELD_SPECS["strd"]
    return "sfc", _identity(spec)


def _coordinates(lat, lon):
    # Promote before wrapping: float32 + 180 would round already-normalized longitudes.
    lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    if (
        lat.ndim != 1
        or lon.shape != lat.shape
        or not np.isfinite(lat).all()
        or not np.isfinite(lon).all()
    ):
        raise ValueError("COSMO coordinates must be finite latitude/longitude point arrays")
    return np.column_stack(
        (
            (2 * np.clip(lat, -90, 90) - lat).astype(np.float32),
            ((lon + 180) % 360 - 180).astype(np.float32),
        )
    )


class CosmoParser(CfParser):
    def __init__(self, config, **kwargs):
        try:
            import eccodes
        except ImportError as exc:
            raise ImportError(
                "ecCodes is required only for COSMO export. Install conda-forge "
                "python-eccodes/eccodes in a separate Pixi environment."
            ) from exc
        self.ec = eccodes
        kwargs.setdefault("fstep_hours", 1)
        super().__init__(config, **kwargs)
        self.templates = {}
        self.files = {}
        self.created_paths = []
        self.message_count = 0
        self.failed = False
        self._closed = False
        if kwargs.get("regrid_degree") is not None or kwargs.get("regrid_type") is not None:
            raise ValueError("COSMO export requires the native grid")
        for key in ("cosmo_dataset", "cosmo_surface_template", "cosmo_pressure_template"):
            if not kwargs.get(key):
                raise ValueError(f"{key} is required for COSMO export")
        dataset = zarr.open_group(str(self.cosmo_dataset), mode="r")
        metadata = dataset.attrs.get("variables_metadata", {})
        self.native = _coordinates(dataset["latitudes"][:], dataset["longitudes"][:])
        order = np.lexsort((self.native[:, 1], self.native[:, 0]))
        self.native_order = order
        self.native_sorted = self.native[order]
        if len(order) == 0 or np.any(
            np.all(self.native_sorted[1:] == self.native_sorted[:-1], axis=1)
        ):
            raise ValueError("Reference grid contains duplicate coordinates or is empty")
        requested = list(self.channels)
        if len(set(requested)) != len(requested):
            raise ValueError("COSMO channel selection contains duplicates")
        excluded = [
            name
            for name in requested
            if name in FORCINGS and metadata.get(name, {}).get("computed_forcing")
        ]
        self.channels = [name for name in requested if name not in excluded]
        if excluded:
            _logger.info("Excluded computed_forcing channels: %s", ", ".join(excluded))
        self.specs = {name: _field_spec(name, metadata) for name in self.channels}
        if not self.specs:
            raise ValueError("COSMO selection contains no physical channels")
        try:
            self._load_templates()
            paths = {
                level: Path(self.output_dir)
                / (
                    f"{self.data_type}_{self.stream}_{level}_{self.run_id}_rank{self.rank_label}.grib"
                )
                for level in sorted({spec.level_type for spec in self.specs.values()})
            }
            for path in paths.values():
                if path.exists() or path.is_symlink():
                    raise FileExistsError(f"Refusing to overwrite COSMO output: {path}")
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            for level, path in paths.items():
                self.files[level] = path.open("xb")
                self.created_paths.append(path)
        except BaseException:
            self.close()
            raise

    def _message_identity(self, handle):
        keys = (
            "discipline",
            "parameterCategory",
            "parameterNumber",
            "typeOfFirstFixedSurface",
            "scaleFactorOfFirstFixedSurface",
            "scaledValueOfFirstFixedSurface",
            "typeOfSecondFixedSurface",
            "scaleFactorOfSecondFixedSurface",
            "scaledValueOfSecondFixedSurface",
            "typeOfStatisticalProcessing",
        )
        return tuple(
            None
            if not self.ec.codes_is_defined(handle, key) or self.ec.codes_is_missing(handle, key)
            else self.ec.codes_get_long(handle, key)
            for key in keys
        )

    def _grid_identity(self, handle):
        e = self.ec
        if e.codes_get_long(handle, "gridDefinitionTemplateNumber") != 101:
            raise ValueError("COSMO templates require grid template 3.101 (unstructured_grid)")
        if e.codes_get_long(handle, "numberOfDataPoints") != len(self.native):
            raise ValueError("Template grid point count differs from reference coordinates")
        return tuple(
            e.codes_get(handle, key)
            for key in (
                "uuidOfHGrid",
                "numberOfGridUsed",
                "numberOfGridInReference",
                "numberOfDataPoints",
                "centre",
                "subCentre",
                "tablesVersion",
                "localTablesVersion",
            )
        )

    def _load_templates(self):
        e = self.ec
        prototypes = {name: _prototype(name, spec) for name, spec in self.specs.items()}
        needed = set(prototypes.values())
        retained = {}
        grid = None
        try:
            for level_type, path in (
                ("sfc", self.cosmo_surface_template),
                ("pl", self.cosmo_pressure_template),
            ):
                count = 0
                with open(path, "rb") as source:
                    while (handle := e.codes_grib_new_from_file(source)) is not None:
                        try:
                            count += 1
                            identity = self._grid_identity(handle)
                            if grid is not None and identity != grid:
                                raise ValueError(
                                    "COSMO template grid/centre/table identities do not match"
                                )
                            grid = identity
                            key = (level_type, self._message_identity(handle))
                            if key in needed:
                                if key in retained:
                                    raise ValueError(f"Ambiguous COSMO prototype: {key}")
                                retained[key] = e.codes_clone(handle)
                        finally:
                            e.codes_release(handle)
                if not count:
                    raise ValueError(f"Empty COSMO template file: {path}")
            missing = needed - retained.keys()
            if missing:
                raise ValueError(f"Missing complete COSMO prototypes: {sorted(missing, key=str)}")
            for name, spec in self.specs.items():
                handle = e.codes_clone(retained[prototypes[name]])
                self.templates[name] = handle
                for key, value in zip(
                    ("discipline", "parameterCategory", "parameterNumber"),
                    spec.parameter,
                    strict=True,
                ):
                    e.codes_set(handle, key, value)
                for which, surface in (("First", spec.first), ("Second", spec.second)):
                    for prefix, value in zip(
                        ("typeOf", "scaleFactorOf", "scaledValueOf"), surface, strict=True
                    ):
                        key = f"{prefix}{which}FixedSurface"
                        if value is None:
                            e.codes_set_missing(handle, key)
                        else:
                            e.codes_set(handle, key, value)
        finally:
            for handle in retained.values():
                e.codes_release(handle)

    def _native_permutation(self, lat, lon):
        pairs = _coordinates(lat, lon)
        if pairs.shape != self.native.shape:
            raise ValueError("Each valid time must contain exactly the native grid point count")
        if np.array_equal(pairs, self.native):
            return None
        cached = getattr(self, "_permutation", None)
        if cached is not None and np.array_equal(pairs[cached], self.native):
            return cached
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        if not np.array_equal(pairs[order], self.native_sorted):
            raise ValueError(
                "Coordinates contain duplicate, missing, extra or unequal native cells"
            )
        permutation = np.empty(len(order), dtype=np.intp)
        permutation[self.native_order] = order
        self._permutation = permutation
        return permutation

    @staticmethod
    def _minute_time(value):
        value = np.datetime64(value, "ns")
        if np.isnat(value) or value != value.astype("datetime64[m]"):
            raise ValueError("COSMO timestamps must be valid and minute-aligned")
        return value

    def _encode(self, name, values, ref_time, valid_time, member, members):
        e = self.ec
        spec = self.specs[name]
        ref_time, valid_time = self._minute_time(ref_time), self._minute_time(valid_time)
        end = int((valid_time - ref_time) / np.timedelta64(1, "m"))
        if end < 0:
            raise ValueError("COSMO valid time precedes forecast reference time")
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(self.native),):
            raise ValueError("Field values must have exactly the native point count")
        if np.isinf(values).any():
            raise ValueError(f"Infinite values in COSMO channel {name}")
        if spec.factor != 1:
            values = values * spec.factor
        missing = np.isnan(values)
        handle = e.codes_clone(self.templates[name])
        try:
            ensemble = members > 1
            statistical = spec.processing is not None
            pdt = (11 if ensemble else 8) if statistical else (1 if ensemble else 0)
            e.codes_set(handle, "productDefinitionTemplateNumber", pdt)
            for key, value in {
                "typeOfProcessedData": 1,
                "typeOfGeneratingProcess": 4 if ensemble else 2,
                "generatingProcessIdentifier": 255,
                "significanceOfReferenceTime": 1,
            }.items():
                e.codes_set(handle, key, value)
            if ensemble:
                e.codes_set(handle, "numberOfForecastsInEnsemble", members)
                e.codes_set(handle, "perturbationNumber", member)
                e.codes_set(handle, "typeOfEnsembleForecast", 255)
            stamp = ref_time.astype("datetime64[m]").astype(object)
            e.codes_set(handle, "dataDate", int(stamp.strftime("%Y%m%d")))
            e.codes_set(handle, "dataTime", stamp.hour * 100 + stamp.minute)
            e.codes_set(handle, "second", 0)
            e.codes_set(handle, "stepUnits", "m")
            if statistical:
                e.codes_set(handle, "numberOfTimeRanges", 1)
                e.codes_set(handle, "typeOfStatisticalProcessing", spec.processing)
                e.codes_set(handle, "typeOfTimeIncrement", 2)
                e.codes_set(handle, "numberOfMissingInStatisticalProcess", 0)
                e.codes_set(handle, "indicatorOfUnitForTimeIncrement", 0)
                e.codes_set(handle, "timeIncrement", 0)
                e.codes_set(handle, "startStep", end - spec.duration)
                e.codes_set(handle, "endStep", end)
                e.codes_set(handle, "indicatorOfUnitForTimeRange", 0)
                e.codes_set(handle, "lengthOfTimeRange", spec.duration)
            else:
                e.codes_set(handle, "forecastTime", end)
            stamp = valid_time.astype("datetime64[m]").astype(object)
            if statistical:
                for key, value in zip(
                    ("year", "month", "day", "hour", "minute", "second"),
                    (stamp.year, stamp.month, stamp.day, stamp.hour, stamp.minute, 0),
                    strict=True,
                ):
                    e.codes_set(handle, f"{key}OfEndOfOverallTimeInterval", value)
            e.codes_set(handle, "bitmapPresent", int(missing.any()))
            if missing.any():
                finite = values[~missing]
                minimum = float(finite.min()) if finite.size else 0.0
                sentinel = minimum - max(abs(minimum), 1.0)
                e.codes_set(handle, "missingValue", sentinel)
                values = np.where(missing, sentinel, values)
            e.codes_set(handle, "packingType", "grid_ccsds")
            e.codes_set(handle, "bitsPerValue", 16)
            e.codes_set(handle, "decimalScaleFactor", 0)
            e.codes_set_values(handle, values)
            if (
                e.codes_get_long(handle, "validityDate") != int(stamp.strftime("%Y%m%d"))
                or e.codes_get_long(handle, "validityTime") != stamp.hour * 100 + stamp.minute
            ):
                raise ValueError("Encoded COSMO validity differs from the intended timestamp")
            return e.codes_get_message(handle)
        finally:
            e.codes_release(handle)

    def process_sample(
        self,
        fstep_iterator_results,
        ref_time,
        source_interval_start=None,
        source_interval_end=None,
        **kwargs,
    ):
        if self._closed:
            raise ValueError("COSMO parser is closed")
        try:
            for result in fstep_iterator_results:
                if not hasattr(result, "dims"):
                    result = result.as_xarray()
                for dim in ("sample", "stream", "forecast_step"):
                    if dim in result.dims:
                        if result.sizes[dim] != 1:
                            raise ValueError(f"COSMO expects a singleton {dim} axis")
                        result = result.isel({dim: 0}, drop=True)
                if "ens" in result.dims:
                    if "mem" in result.dims:
                        raise ValueError("Both ens and mem axes supplied")
                    result = result.rename(ens="mem")
                if not {"ipoint", "channel"} <= set(result.dims) or set(result.dims) - {
                    "ipoint",
                    "channel",
                    "mem",
                }:
                    raise ValueError(
                        "COSMO fields require ipoint/channel and optional mem dimensions"
                    )
                channels = list(result.channel.values)
                if len(set(channels)) != len(channels) or set(self.channels) - set(channels):
                    raise ValueError("Missing or duplicate prediction channels")
                for key in ("valid_time", "lat", "lon"):
                    if result[key].dims != ("ipoint",):
                        raise ValueError(f"COSMO {key} must be indexed by ipoint")
                times = np.asarray(result.valid_time.values, dtype="datetime64[ns]")
                if not len(times) or np.isnat(times).any():
                    raise ValueError("COSMO valid times must be nonempty and non-NaT")
                members = result.sizes.get("mem", 1)
                if not 1 <= members <= 255:
                    raise ValueError("COSMO requires 1–255 members")
                for valid_time in np.unique(times):
                    sub = result.isel(ipoint=times == valid_time)
                    permutation = self._native_permutation(sub.lat.values, sub.lon.values)
                    for name in self.channels:
                        field = sub.sel(channel=name)
                        for member in range(members):
                            values = (
                                field.isel(mem=member).values
                                if "mem" in field.dims
                                else field.values
                            )
                            if permutation is not None:
                                values = values[permutation]
                            message = self._encode(
                                name, values, ref_time, valid_time, member, members
                            )
                            self.files[self.specs[name].level_type].write(message)
                            self.message_count += 1
        except BaseException:
            self.failed = True
            self.close()
            raise

    def close(self):
        """Release all resources; discard only this instance's incomplete output."""
        if self._closed:
            return
        self._closed = True
        self.failed |= sys.exc_info()[0] is not None
        errors = []
        for stream in self.files.values():
            try:
                stream.close()
            except OSError as exc:
                errors.append(exc)
                self.failed = True
        self.files.clear()
        for handle in self.templates.values():
            try:
                self.ec.codes_release(handle)
            except Exception as exc:
                errors.append(exc)
                self.failed = True
        self.templates.clear()
        if self.failed:
            for path in self.created_paths:
                try:
                    path.unlink()
                except OSError as exc:
                    errors.append(exc)
        else:
            _logger.info("Exported %d COSMO messages to %s", self.message_count, self.created_paths)
        if errors:
            raise errors[0]
