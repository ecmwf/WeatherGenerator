import logging
from pathlib import Path

import numpy as np
import xarray as xr

from weathergen.evaluate.plotting.timeseries import Timeseries

STREAM = "ERA5"
REGION = "global"
CHANNEL = "z_500"
SAMPLE = 0


def _make_step(step: int) -> xr.DataArray:
    """Small synthetic data for one forecast step: 5 points, 2 samples, 1 channel."""
    valid_time = np.datetime64("2026-01-01T00") + np.timedelta64(6 * step, "h")
    n_points = 5
    return xr.DataArray(
        np.random.default_rng(step).random((n_points, 2, 1)),
        dims=("ipoint", "sample", "channel"),
        coords={
            "ipoint": np.arange(n_points),
            "sample": [0, 1],
            "channel": [CHANNEL],
            "valid_time": ("ipoint", [valid_time] * n_points),
        },
    )


def _plot(n_steps: int, out_dir: Path) -> Path:
    preds = {f"step{s}": _make_step(s) for s in range(n_steps)}
    tars = {f"step{s}": _make_step(s) for s in range(n_steps)}
    Timeseries(preds, tars).plot_single_timeseries(str(out_dir), CHANNEL, SAMPLE, STREAM, REGION)
    name = f"timeseries_{REGION}_{CHANNEL}_sample_{SAMPLE}.png"
    return out_dir / "plots" / STREAM / "timeseries" / name


def test_single_forecast_step_saves_plot_and_warns(tmp_path, caplog):
    """One forecast step used to crash with a TypeError (issue #2985); now it warns and plots."""
    with caplog.at_level(logging.WARNING):
        plot_path = _plot(1, tmp_path)

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert any("one forecast step" in record.getMessage().lower() for record in caplog.records)


def test_two_forecast_steps_saves_plot_without_warning(tmp_path, caplog):
    """Several forecast steps keep working and do not trigger the single-step warning."""
    with caplog.at_level(logging.WARNING):
        plot_path = _plot(2, tmp_path)

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert not any("one forecast step" in record.getMessage().lower() for record in caplog.records)
