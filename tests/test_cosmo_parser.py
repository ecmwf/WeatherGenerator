"""CPU COSMO checks; real CLI artifacts are checked only with COSMO_* opt-in."""

import builtins
import importlib
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from omegaconf import OmegaConf

SURFACE = [
    "100u", "100v", "10u", "10v", "2t", "2d", "skt", "z", "sp", "msl",
    "CAPE_ML", "CAPE_MU", "CIN_ML", "CIN_MU", "LPI", "hcc", "mcc", "lcc",
    "tcc", "lsm", "PLCOV", "SKC", "SOILTYP", "sdor", "anor", "slor", "isor",
    "W_SO", "tp", "tp_6h", "ro", "ro_6h", "ssrd", "ssrd_6h", "strd", "strd_6h", "VMAX_10M",
]
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
PHYSICAL = SURFACE + [f"{prefix}_{level}" for prefix in "zqtuvw" for level in LEVELS]
FORCINGS = [
    "cos_julian_day", "cos_latitude", "cos_local_time", "cos_longitude", "insolation",
    "sin_julian_day", "sin_latitude", "sin_local_time", "sin_longitude",
]
REF = np.datetime64("2005-01-01T12:00", "ns")
LAT = np.array([47.3, 46.1, 47.0, 46.8], dtype=np.float32)
LON = np.array([8.4, 7.2, 9.1, 6.5], dtype=np.float32)
UUID = "17643da2574959b644d254a3cd6e2bc0"


def expected_identity(channel):
    """Independent physical identities, not imported from the encoder's table.

    Fixed surfaces are decoded to physical SI values so equivalent GRIB scale
    representations do not weaken the checks of actual pressure/soil bounds.
    """
    first, second, processing, minutes, divisor = (1, None), (255, None), None, 0, 1
    pressure = re.fullmatch(r"([zqtuvw])_(\d+)", channel)
    if pressure:
        prefix, level = pressure.groups()
        assert int(level) in LEVELS
        parameter = {
            "z": (0, 3, 4),
            "q": (0, 1, 0),
            "t": (0, 0, 0),
            "u": (0, 2, 2),
            "v": (0, 2, 3),
            "w": (0, 2, 9),
        }[prefix]
        first = (100, int(level) * 100)
    elif channel in ("100u", "100v", "10u", "10v", "2t", "2d"):
        parameter = {"u": (0, 2, 2), "v": (0, 2, 3), "t": (0, 0, 0), "d": (0, 0, 6)}[channel[-1]]
        first = (103, int(channel[:-1]))
    elif channel.startswith(("CAPE_", "CIN_")):
        parameter = (0, 7, 6 if channel.startswith("CAPE") else 7)
        first = (192 if channel.endswith("ML") else 193, None)
    elif channel in ("hcc", "mcc", "lcc"):
        parameter = (0, 6, 22)
        first = (100, {"hcc": 0, "mcc": 40000, "lcc": 80000}[channel])
        second = (1, None) if channel == "lcc" else (100, 40000 if channel == "hcc" else 80000)
    elif channel == "W_SO":
        parameter, first, second = (2, 3, 20), (106, 0.01), (106, 0.03)
    elif channel == "VMAX_10M":
        parameter, first, processing, minutes = (0, 2, 22), (103, 10), 2, 70
    elif channel.split("_")[0] in ("tp", "ro", "ssrd", "strd"):
        base = channel.split("_")[0]
        minutes = 360 if channel.endswith("_6h") else 60
        parameter = {"tp": (0, 1, 52), "ro": (2, 0, 5), "ssrd": (0, 4, 7), "strd": (0, 5, 3)}[base]
        processing = 1 if base in ("tp", "ro") else 0
        if base == "ro":
            first = (106, 0)
        if processing == 0:
            divisor = minutes * 60
    else:
        parameter = {
            "skt": (0, 0, 17),
            "z": (0, 3, 4),
            "sp": (0, 3, 0),
            "msl": (0, 3, 1),
            "LPI": (0, 17, 192),
            "tcc": (0, 6, 1),
            "lsm": (2, 0, 0),
            "PLCOV": (2, 0, 4),
            "SKC": (2, 0, 199),
            "SOILTYP": (2, 3, 196),
            "sdor": (0, 3, 20),
            "anor": (0, 3, 21),
            "slor": (0, 3, 22),
            "isor": (0, 3, 24),
        }[channel]
        if channel == "msl":
            first = (101, None)
    return (parameter, first, second, processing, minutes), divisor


@pytest.fixture
def ec():
    return pytest.importorskip("eccodes")


def messages(ec, paths):
    for path in paths:
        with Path(path).open("rb") as stream:
            while (handle := ec.codes_grib_new_from_file(stream)) is not None:
                try:
                    yield handle
                finally:
                    ec.codes_release(handle)


def surface(ec, handle, which):
    kind = ec.codes_get_long(handle, f"typeOf{which}FixedSurface")
    scale_key, value_key = f"scaleFactorOf{which}FixedSurface", f"scaledValueOf{which}FixedSurface"
    if ec.codes_is_missing(handle, value_key):
        assert ec.codes_is_missing(handle, scale_key)
        return kind, None
    return kind, ec.codes_get_long(handle, value_key) * 10.0 ** -ec.codes_get_long(
        handle, scale_key
    )


def identity(ec, handle):
    parameter = tuple(
        ec.codes_get_long(handle, key)
        for key in ("discipline", "parameterCategory", "parameterNumber")
    )
    pdt = ec.codes_get_long(handle, "productDefinitionTemplateNumber")
    ec.codes_set(handle, "stepUnits", "m")
    processing = (
        ec.codes_get_long(handle, "typeOfStatisticalProcessing") if pdt in (8, 11) else None
    )
    minutes = ec.codes_get_long(handle, "endStep") - ec.codes_get_long(handle, "startStep")
    return (
        parameter,
        surface(ec, handle, "First"),
        surface(ec, handle, "Second"),
        processing,
        minutes,
    )


def assert_values(ec, handle, expected):
    expected = np.asarray(expected, dtype=np.float64)
    actual = ec.codes_get_values(handle)
    assert actual.shape == expected.shape
    missing = np.isnan(expected)
    if missing.any():
        assert ec.codes_get_long(handle, "bitmapPresent") == 1
        np.testing.assert_array_equal(ec.codes_get_array(handle, "bitmap") == 0, missing)
    finite = ~missing
    roundoff = np.finfo(np.float32).eps * np.maximum(1, np.abs(expected[finite]))
    bound = float(ec.codes_get_double(handle, "packingError")) + roundoff
    assert np.all(np.abs(actual[finite] - expected[finite]) <= bound), (
        np.max(np.abs(actual[finite] - expected[finite])),
        np.max(bound),
    )


def assert_time(ec, handle, step, duration=0):
    assert ec.codes_get_long(handle, "dataDate") == 20050101
    assert ec.codes_get_long(handle, "dataTime") == 1200
    assert ec.codes_get_long(handle, "validityDate") == 20050101
    assert ec.codes_get_long(handle, "validityTime") == 1200 + step * 100
    ec.codes_set(handle, "stepUnits", "m")
    assert ec.codes_get_long(handle, "endStep") == step * 60
    assert ec.codes_get_long(handle, "startStep") == step * 60 - duration
    assert ec.codes_get_long(handle, "typeOfProcessedData") == 1
    assert ec.codes_get_long(handle, "generatingProcessIdentifier") == 255


def write_templates(ec, path, channels, uuid=UUID):
    with path.open("wb") as stream:
        for channel in channels:
            handle = ec.codes_grib_new_from_samples("GRIB2")
            try:
                spec, _ = expected_identity(channel)
                parameter, first, second, processing, minutes = spec
                if channel == "skt":
                    parameter = (0, 0, 0)  # T_G is the prototype, not the output identity.
                if channel == "w_700":
                    parameter = (0, 2, 8)  # Prototype omega must become geometric w.
                for key, value in {
                    "centre": 215,
                    "subCentre": 255,
                    "tablesVersion": 19,
                    "localTablesVersion": 1,
                    "gridDefinitionTemplateNumber": 101,
                    "numberOfDataPoints": 4,
                    "numberOfGridUsed": 1,
                    "numberOfGridInReference": 1,
                    "uuidOfHGrid": uuid,
                    "productDefinitionTemplateNumber": 11 if processing is not None else 1,
                    "dataDate": 20250101,
                    "dataTime": 0,
                    "numberOfForecastsInEnsemble": 11,
                    "generatingProcessIdentifier": 141,
                }.items():
                    ec.codes_set(handle, key, value)
                for key, value in zip(
                    ("discipline", "parameterCategory", "parameterNumber"), parameter, strict=True
                ):
                    ec.codes_set(handle, key, value)
                for which, (kind, value) in (("First", first), ("Second", second)):
                    ec.codes_set(handle, f"typeOf{which}FixedSurface", kind)
                    if value is None:
                        ec.codes_set_missing(handle, f"scaleFactorOf{which}FixedSurface")
                        ec.codes_set_missing(handle, f"scaledValueOf{which}FixedSurface")
                    else:
                        scale = -2 if kind == 100 else 2 if kind == 106 else 0
                        ec.codes_set(handle, f"scaleFactorOf{which}FixedSurface", scale)
                        ec.codes_set(
                            handle, f"scaledValueOf{which}FixedSurface", round(value * 10.0**scale)
                        )
                if processing is not None:
                    ec.codes_set(handle, "typeOfStatisticalProcessing", processing)
                    ec.codes_set(handle, "stepUnits", "m")
                    ec.codes_set(handle, "startStep", 0)
                    ec.codes_set(handle, "endStep", minutes)
                ec.codes_set(handle, "packingType", "grid_ccsds")
                ec.codes_set(handle, "bitsPerValue", 16)
                ec.codes_set_values(handle, np.arange(4, dtype=float))
                ec.codes_write(handle, stream)
            finally:
                ec.codes_release(handle)


@pytest.fixture
def inputs(tmp_path, ec):
    dataset = tmp_path / "reference.zarr"
    root = zarr.open_group(str(dataset), mode="w", zarr_format=2)
    root.create_array("latitudes", data=LAT)
    root.create_array("longitudes", data=LON)
    root.attrs.update(
        {
            "variables": PHYSICAL + FORCINGS,
            "variables_metadata": {
                **{name: {"computed_forcing": True} for name in FORCINGS},
                "VMAX_10M": {"period": ["650m", "12h"]},
            },
        }
    )
    sfc, pl = tmp_path / "surface.grib", tmp_path / "pressure.grib"
    # Keep only one representative for derived one/six-hour products.
    prototypes = [
        name
        for name in SURFACE
        if not name.endswith("_6h")
        and name
        not in ("100u", "100v", "z", "ro", "ssrd", "PLCOV", "SKC", "sdor", "anor", "slor", "isor")
    ]
    write_templates(ec, sfc, prototypes)
    write_templates(ec, pl, [f"{p}_{700 if p == 'w' else 500}" for p in "zqtuvw"])
    return dict(
        cosmo_dataset=str(dataset),
        cosmo_surface_template=str(sfc),
        cosmo_pressure_template=str(pl),
        output_dir=str(tmp_path / "out"),
        data_type="prediction",
        stream="MCH",
        run_id="unit",
        rank_label="0000",
        output_format="cosmo",
        grid_type="unstructured_grid",
    )


def parser(inputs, channels):
    from weathergen.evaluate.export.parsers.cosmo_parser import CosmoParser

    return CosmoParser(OmegaConf.create({}), channels=channels, **inputs)


def array(channels, values=None, step=1, order=None):
    if values is None:
        values = np.arange(4 * len(channels), dtype=np.float32).reshape(4, len(channels)) + 270
    values = np.asarray(values)
    order = np.arange(4) if order is None else np.asarray(order)
    dims = ["ipoint", "channel"] + (["mem"] if values.ndim == 3 else [])
    return xr.DataArray(
        values[order],
        dims=dims,
        coords={
            "channel": channels,
            "lat": ("ipoint", LAT[order]),
            "lon": ("ipoint", LON[order]),
            "valid_time": ("ipoint", np.full(4, REF + np.timedelta64(step, "h"))),
        },
    )


def outputs(inputs):
    return sorted(Path(inputs["output_dir"]).glob("*.grib"))


def test_native_order_and_reference_time(inputs, ec):
    channels = ["2t", "t_850"]
    values = np.array([[281, 251], [282, 252], [283, 253], [284, 254]], dtype=np.float32)
    encoder = parser(inputs, channels)
    encoder.process_sample(
        iter(
            [
                array(channels, values, order=[2, 0, 3, 1]),
                array(channels, values + 1, step=2, order=[3, 2, 1, 0]),
            ]
        ),
        REF,
        source_interval_start=REF - np.timedelta64(1, "h"),
    )
    encoder.close()
    encoder.close()
    assert {p.name for p in outputs(inputs)} == {
        "prediction_MCH_sfc_unit_rank0000.grib",
        "prediction_MCH_pl_unit_rank0000.grib",
    }
    seen = set()
    for handle in messages(ec, outputs(inputs)):
        channel = "2t" if surface(ec, handle, "First")[0] == 103 else "t_850"
        step = (ec.codes_get_long(handle, "validityTime") - 1200) // 100
        assert_time(ec, handle, step)
        assert identity(ec, handle) == expected_identity(channel)[0]
        assert ec.codes_get_string(handle, "uuidOfHGrid") == UUID
        assert_values(ec, handle, values[:, channels.index(channel)] + step - 1)
        assert (channel, step) not in seen
        seen.add((channel, step))
    assert seen == {(c, s) for c in channels for s in (1, 2)}


@pytest.mark.parametrize("bad", ["duplicate", "different", "nan_coordinate", "nat", "infinity"])
def test_invalid_sample_cleans_partial_output(inputs, bad):
    encoder = parser(inputs, ["2t"])
    good = array(["2t"])
    broken = good.copy(deep=True)
    if bad == "duplicate":
        broken.coords["lat"].values[1] = LAT[0]
        broken.coords["lon"].values[1] = LON[0]
    elif bad == "different":
        broken.coords["lon"].values[1] += 0.25
    elif bad == "nan_coordinate":
        broken.coords["lat"].values[0] = np.nan
    elif bad == "nat":
        broken.coords["valid_time"].values[0] = np.datetime64("NaT")
    else:
        broken.values[0, 0] = np.inf
    try:
        with pytest.raises(ValueError):
            encoder.process_sample(iter([good, broken]), REF)
    finally:
        encoder.close()
    assert outputs(inputs) == []


def test_nan_bitmap(inputs, ec):
    values = np.array([[271], [np.nan], [-2], [274]], dtype=np.float32)
    encoder = parser(inputs, ["2t"])
    encoder.process_sample(iter([array(["2t"], values)]), REF)
    encoder.close()
    handles = 0
    for handle in messages(ec, outputs(inputs)):
        assert_values(ec, handle, values[:, 0])
        handles += 1
    assert handles == 1


@pytest.mark.parametrize("members", [None, 1, 2])
@pytest.mark.parametrize("channel", ["2t", "tp"])
def test_member_identity(inputs, ec, members, channel):
    values = np.arange(4, dtype=np.float32)[:, None]
    if members is not None:
        values = values[:, :, None] + np.arange(members)[None, None, :] * 100
    encoder = parser(inputs, [channel])
    encoder.process_sample(iter([array([channel], values)]), REF)
    encoder.close()
    seen = []
    for handle in messages(ec, outputs(inputs)):
        ensemble = members == 2
        statistical = channel == "tp"
        assert ec.codes_get_long(handle, "productDefinitionTemplateNumber") == (
            8 if statistical else 0
        ) + ensemble * (3 if statistical else 1)
        assert ec.codes_get_long(handle, "typeOfGeneratingProcess") == (4 if ensemble else 2)
        member = ec.codes_get_long(handle, "perturbationNumber") if ensemble else 0
        if ensemble:
            assert ec.codes_get_long(handle, "numberOfForecastsInEnsemble") == 2
            assert ec.codes_get_long(handle, "typeOfEnsembleForecast") == 255
        assert_values(ec, handle, np.arange(4) + member * 100)
        seen.append(member)
    assert sorted(seen) == list(range(members or 1))


def test_all_physical_fields_and_exclusions(inputs, ec, caplog):
    caplog.set_level("INFO")
    assert len(PHYSICAL) == 115
    values = np.tile(np.array([-2.0, 0.0, 17.25, 84.0])[:, None], (1, 124))
    for channel in ("ssrd", "strd"):
        values[:, (PHYSICAL + FORCINGS).index(channel)] = [-3600, 0, 7200, 18000]
        values[:, (PHYSICAL + FORCINGS).index(channel + "_6h")] = [-21600, 0, 43200, 108000]
    encoder = parser(inputs, PHYSICAL + FORCINGS)
    encoder.process_sample(
        iter([array(PHYSICAL + FORCINGS, values), array(PHYSICAL + FORCINGS, values + 1, step=2)]),
        REF,
    )
    encoder.close()
    lookup = {expected_identity(c)[0]: c for c in PHYSICAL}
    assert len(lookup) == 115
    seen = set()
    for handle in messages(ec, outputs(inputs)):
        decoded = identity(ec, handle)
        channel = lookup[decoded]
        step = (ec.codes_get_long(handle, "validityTime") - 1200) // 100
        assert_time(ec, handle, step, decoded[-1])
        _, divisor = expected_identity(channel)
        assert_values(ec, handle, (values[:, PHYSICAL.index(channel)] + step - 1) / divisor)
        assert (channel, step) not in seen
        seen.add((channel, step))
    assert seen == {(c, s) for c in PHYSICAL for s in (1, 2)}
    for name in FORCINGS:
        assert name in caplog.text


@pytest.mark.parametrize("channels", [["not_a_field"], ["t_123"], FORCINGS, []])
def test_invalid_channel_selection(inputs, channels):
    with pytest.raises(ValueError):
        parser(inputs, channels)
    assert outputs(inputs) == []


def test_wrong_template_grid(inputs, ec):
    write_templates(ec, Path(inputs["cosmo_pressure_template"]), ["t_500"], uuid="0" * 32)
    with pytest.raises(ValueError):
        parser(inputs, ["2t", "t_850"])
    assert outputs(inputs) == []


def test_conflicting_output_preserved(inputs):
    out = Path(inputs["output_dir"])
    out.mkdir()
    existing = out / "prediction_MCH_pl_unit_rank0000.grib"
    existing.write_bytes(b"previous successful output")
    with pytest.raises((FileExistsError, ValueError)):
        parser(inputs, ["2t", "t_850"])
    assert existing.read_bytes() == b"previous successful output"
    assert outputs(inputs) == [existing]


def test_source_absent_step_two(tmp_path):
    from weathergen.common.io import ItemKey, OutputDataset, OutputItem, TimeRange, zarrio_writer
    from weathergen.evaluate.export.export_core import get_source_info

    store = tmp_path / "step2.zarr"
    key = ItemKey(sample=0, forecast_step=2, stream="MCH")
    prediction = OutputDataset(
        name="prediction",
        item_key=key,
        source_interval=TimeRange(REF, REF + np.timedelta64(1, "h")),
        data=np.zeros((4, 1, 1), dtype=np.float32),
        times=np.full(4, REF + np.timedelta64(2, "h")),
        coords=np.column_stack((LAT, LON)),
        geoinfo=np.empty((4, 0)),
        channels=["2t"],
        geoinfo_channels=[],
    )
    with zarrio_writer(store) as writer:
        writer.write_zarr(OutputItem(key, forecast_offset=1, prediction=prediction))
    starts, ends = get_source_info(store, "MCH", [0], fstep_hours=1)
    assert starts == ends == [REF]


def test_optional_imports(monkeypatch):
    from weathergen.evaluate.export import parser_factory

    original_import = builtins.__import__
    imported = []

    def no_cosmo(name, *args, **kwargs):
        imported.append(name)
        if name.endswith("cosmo_parser"):
            raise AssertionError("Non-COSMO selection imported COSMO")
        return original_import(name, *args, **kwargs)

    module_name = "weathergen.evaluate.export.parsers.netcdf_parser"
    module = types.ModuleType(module_name)
    sentinel = object()
    module.NetcdfParser = lambda *args, **kwargs: sentinel
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(builtins, "__import__", no_cosmo)
    importlib.reload(parser_factory)
    factory = parser_factory.CfParserFactory
    assert factory.get_parser(OmegaConf.create({}), output_format="netcdf") is sentinel
    with pytest.raises(ValueError, match="format"):
        factory.get_parser(OmegaConf.create({}), output_format="unknown")
    assert not any(name.endswith("cosmo_parser") for name in imported)


def test_missing_eccodes_is_actionable(monkeypatch, tmp_path):
    module = importlib.import_module("weathergen.evaluate.export.parsers.cosmo_parser")
    original_import = builtins.__import__

    def no_eccodes(name, *args, **kwargs):
        if name == "eccodes" or name.startswith("eccodes."):
            raise ImportError("deliberately unavailable optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_eccodes)
    with pytest.raises(ImportError) as error:
        module.CosmoParser(
            OmegaConf.create({}),
            channels=["2t"],
            output_dir=str(tmp_path),
            cosmo_dataset="unused",
            cosmo_surface_template="unused",
            cosmo_pressure_template="unused",
            data_type="prediction",
            stream="MCH",
            run_id="test",
            rank_label="0",
            output_format="cosmo",
            grid_type="unstructured_grid",
        )
    message = str(error.value).lower()
    assert all(word in message for word in ("cosmo", "pixi", "conda-forge", "python-eccodes"))


def test_export_help_and_stream_names(capsys, tmp_path):
    from weathergen.evaluate.export.export_inference import parse_args

    with pytest.raises(SystemExit) as error:
        parse_args(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "cosmo" in help_text
    for option in ("--cosmo-dataset", "--cosmo-surface-template", "--cosmo-pressure-template"):
        assert option in help_text
    args = parse_args(
        [
            "--run-id",
            "test",
            "--output-dir",
            str(tmp_path),
            "--format",
            "netcdf",
            "--stream",
            "MY_CUSTOM_STREAM",
        ]
    )
    assert args.stream == "MY_CUSTOM_STREAM"
    with pytest.raises(SystemExit):
        parse_args(["--run-id", "test", "--output-dir", str(tmp_path), "--format", "cosmo"])


def test_dataset_roundtrip(ec):
    names = (
        "COSMO_SMOKE_DIR",
        "COSMO_DATASET",
        "COSMO_SURFACE_TEMPLATE",
        "COSMO_PRESSURE_TEMPLATE",
    )
    if not all(os.environ.get(name) for name in names):
        pytest.skip("Opt in with " + ", ".join(names))
    work = Path(os.environ["COSMO_SMOKE_DIR"])
    dataset = zarr.open_group(os.environ["COSMO_DATASET"], mode="r")
    variables = list(dataset.attrs["variables"])
    assert set(variables) == set(PHYSICAL + FORCINGS)
    grid_keys = (
        "centre",
        "subCentre",
        "tablesVersion",
        "localTablesVersion",
        "gridDefinitionTemplateNumber",
        "numberOfDataPoints",
        "numberOfGridUsed",
        "numberOfGridInReference",
        "uuidOfHGrid",
    )
    template_grid = None
    for template in names[2:]:
        reader = messages(ec, [os.environ[template]])
        try:
            handle = next(reader)
            grid = tuple(ec.codes_get(handle, key) for key in grid_keys)
            if template_grid is None:
                template_grid = grid
            assert grid == template_grid
        finally:
            reader.close()
    lookup = {expected_identity(channel)[0]: channel for channel in PHYSICAL}
    assert len(lookup) == 115
    cases = [
        ("small_grib", "prediction", ["2t", "t_850"]),
        ("full_grib", "prediction", PHYSICAL),
        ("full_grib", "target", PHYSICAL),
    ]
    totals = {}
    for directory, data_type, channels in cases:
        paths = sorted((work / directory).glob(f"{data_type}_MCH_*_cosmo_smoke_rank*.grib"))
        assert len(paths) == 2, paths
        seen = set()
        level_counts = {"sfc": 0, "pl": 0}
        # Stream GRIB twice to bound memory to one source time slice, not two
        # 124-channel million-point fields plus decoded output copies.
        for step in (1, 2):
            values = np.asarray(dataset["data"][step, :, 0, :])
            for path in paths:
                for handle in messages(ec, [path]):
                    valid = ec.codes_get_long(handle, "validityTime")
                    assert valid in (1300, 1400)
                    if valid != 1200 + step * 100:
                        continue
                    spec = identity(ec, handle)
                    channel = lookup[spec]
                    assert channel in channels
                    assert (channel, step) not in seen
                    seen.add((channel, step))
                    level = "pl" if re.fullmatch(r"[zqtuvw]_\d+", channel) else "sfc"
                    assert f"_{level}_" in path.name
                    level_counts[level] += 1
                    assert tuple(ec.codes_get(handle, key) for key in grid_keys) == template_grid
                    assert_time(ec, handle, step, spec[-1])
                    assert ec.codes_get_long(handle, "productDefinitionTemplateNumber") == (
                        0 if spec[3] is None else 8
                    )
                    assert ec.codes_get_long(handle, "typeOfGeneratingProcess") == 2
                    assert ec.codes_get_string(handle, "packingType") == "grid_ccsds"
                    _, divisor = expected_identity(channel)
                    expected = values[variables.index(channel)].astype(np.float64) / divisor
                    # ecCodes stores constant CCSDS fields in the reference value alone.
                    bits = ec.codes_get_long(handle, "bitsPerValue")
                    if bits == 0:
                        assert np.all(expected == expected[0])
                    else:
                        assert bits == 16
                    assert_values(ec, handle, expected)
            del values
        assert seen == {(channel, step) for channel in channels for step in (1, 2)}
        assert level_counts == (
            {"sfc": 2, "pl": 2} if directory == "small_grib" else {"sfc": 74, "pl": 156}
        )
        totals[directory, data_type] = len(seen)
    assert totals["small_grib", "prediction"] == 4
    assert totals["full_grib", "prediction"] + totals["full_grib", "target"] == 460


def test_named_singleton_dimensions(inputs, ec):
    encoder = parser(inputs, ["2t"])
    data = array(["2t"]).expand_dims(sample=[0], stream=["MCH"], forecast_step=[1], ens=[0])
    encoder.process_sample(iter([data]), REF)
    encoder.close()
    count = 0
    for handle in messages(ec, outputs(inputs)):
        assert_values(ec, handle, np.arange(4) + 270)
        assert ec.codes_get_long(handle, "productDefinitionTemplateNumber") == 0
        count += 1
    assert count == 1


def test_encoding_failure_cleanup(inputs, ec, monkeypatch):
    encoder = parser(inputs, ["2t", "t_850"])
    encoder.process_sample(iter([array(["2t", "t_850"])]), REF)
    unrelated = Path(inputs["output_dir"]) / "previous.grib"
    unrelated.write_bytes(b"keep")

    def fail_write(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(ec, "codes_get_message", fail_write)
    try:
        with pytest.raises(OSError, match="injected"):
            encoder.process_sample(iter([array(["2t", "t_850"], step=2)]), REF)
    finally:
        encoder.close()
        encoder.close()
    assert outputs(inputs) == [unrelated]
    assert unrelated.read_bytes() == b"keep"


def test_export_closes_parser_on_worker_failure(monkeypatch, tmp_path):
    from weathergen.evaluate.export import export_core

    closed = []
    fake_parser = types.SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(export_core.CfParserFactory, "get_parser", lambda **kwargs: fake_parser)
    monkeypatch.setattr(
        export_core,
        "get_model_results",
        lambda *args: [tmp_path / "validation_chkpt00000_rank0000.zarr"],
    )
    monkeypatch.setattr(export_core, "get_fsteps", lambda *args: [1])
    monkeypatch.setattr(export_core, "get_streams", lambda *args: ["MCH"])
    monkeypatch.setattr(export_core, "get_channels", lambda *args: ["2t"])
    monkeypatch.setattr(export_core, "get_samples", lambda *args: [0])
    monkeypatch.setattr(export_core, "get_source_info", lambda *args, **kwargs: ([REF], [REF]))

    class BrokenPool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def imap_unordered(self, *args, **kwargs):
            raise OSError("worker read failed")

    monkeypatch.setattr(export_core, "Pool", BrokenPool)
    with pytest.raises(OSError, match="worker read failed"):
        export_core.export_model_outputs(
            "prediction",
            OmegaConf.create({}),
            run_id="test",
            samples=[0],
            fsteps=[1],
            stream="MCH",
            channels=["2t"],
            n_processes=1,
            epoch=0,
            rank="all",
            results_path=str(tmp_path),
            output_format="cosmo",
            fstep_hours=1,
        )
    assert closed == [True]


def test_ambiguous_template_rejected(inputs):
    path = Path(inputs["cosmo_surface_template"])
    original = path.read_bytes()
    path.write_bytes(original + original)
    with pytest.raises(ValueError):
        parser(inputs, ["2t"])
    assert outputs(inputs) == []


@pytest.mark.parametrize(
    "option,value", [("--regrid-degree", "1"), ("--regrid-type", "regular_ll")]
)
def test_cosmo_regrid_rejected(inputs, option, value):
    from weathergen.evaluate.export.export_inference import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--run-id",
                "test",
                "--output-dir",
                inputs["output_dir"],
                "--format",
                "cosmo",
                "--stream",
                "MCH",
                "--cosmo-dataset",
                inputs["cosmo_dataset"],
                "--cosmo-surface-template",
                inputs["cosmo_surface_template"],
                "--cosmo-pressure-template",
                inputs["cosmo_pressure_template"],
                option,
                value,
            ]
        )
