import numpy as np
import pytest

from weathergen.common.io import ItemKey, OutputDataset, OutputItem, TimeRange, ZarrIO


def _make_item(sample: int = 0, forecast_step: int = 0, stream: str = "ERA5") -> OutputItem:
    """Build a small synthetic output item containing a single prediction dataset."""
    key = ItemKey(sample, forecast_step, stream)
    n_points = 4
    prediction = OutputDataset(
        name="prediction",
        item_key=key,
        source_interval=TimeRange("2020-01-01T00:00", "2020-01-01T06:00"),
        data=np.zeros((n_points, 2, 1), dtype=np.float32),
        times=np.full(n_points, np.datetime64("2020-01-01T06:00", "ns")),
        coords=np.zeros((n_points, 2), dtype=np.float32),
        geoinfo=np.zeros((n_points, 0), dtype=np.float32),
        channels=["u", "v"],
        geoinfo_channels=[],
    )
    return OutputItem(key=key, forecast_offset=0, prediction=prediction)


def test_write_zarr_refuses_to_overwrite_existing_item(tmp_path):
    store_path = tmp_path / "output.zarr"
    with ZarrIO(store_path, read_only=False) as zio:
        zio.write_zarr(_make_item())

        with pytest.raises(ValueError, match="already exists") as exc_info:
            zio.write_zarr(_make_item())

    # The message should say which item collided and where the store lives.
    assert "0/ERA5/0" in str(exc_info.value)
    assert str(store_path) in str(exc_info.value)


def test_write_zarr_refuses_to_overwrite_item_from_previous_run(tmp_path):
    store_path = tmp_path / "output.zarr"
    with ZarrIO(store_path, read_only=False) as zio:
        zio.write_zarr(_make_item())

    # Re-opening an existing store (e.g. inference re-run with the same run_id)
    with ZarrIO(store_path, read_only=False) as zio:
        with pytest.raises(ValueError, match="already exists"):
            zio.write_zarr(_make_item())


def test_write_zarr_accepts_distinct_items(tmp_path):
    with ZarrIO(tmp_path / "output.zarr", read_only=False) as zio:
        zio.write_zarr(_make_item(sample=0, forecast_step=0))
        zio.write_zarr(_make_item(sample=0, forecast_step=1))
        zio.write_zarr(_make_item(sample=1, forecast_step=0))
