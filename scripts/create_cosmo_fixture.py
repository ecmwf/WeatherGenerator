#!/usr/bin/env python3
"""Write physical dataset values as synthetic WeatherGenerator output, not a model forecast.

Example: create_cosmo_fixture.py --dataset input.zarr --output results/fixture.zarr
Use --channels 2t t_850 --permute for the small native-grid restoration check.
"""

import argparse
from pathlib import Path

import numpy as np
import zarr

from weathergen.common.io import ItemKey, OutputDataset, OutputItem, TimeRange, zarrio_writer


def _write_step(writer, dataset, step, channels, indices, coords, dates, interval, permute):
    # Read the complete time chunk once, not once per requested channel.
    values = np.asarray(dataset["data"][step, :, 0, :])
    if indices is not None:
        values = values[indices]
    values = values.T
    if permute:
        values = values[::-1]

    key = ItemKey(sample=0, forecast_step=step, stream="MCH")
    metadata = {
        "item_key": key,
        "source_interval": interval,
        "times": np.full(coords.shape[0], dates[step], dtype="datetime64[ns]"),
        "coords": coords,
        "geoinfo": np.empty((coords.shape[0], 0), dtype=np.float32),
        "channels": channels,
        "geoinfo_channels": [],
    }
    if step == 0:
        item = OutputItem(
            key=key,
            forecast_offset=1,
            source=OutputDataset(name="source", data=values, **metadata),
        )
    else:
        item = OutputItem(
            key=key,
            forecast_offset=1,
            target=OutputDataset(name="target", data=values, **metadata),
            prediction=OutputDataset(name="prediction", data=values[..., None], **metadata),
        )
    writer.write_zarr(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Reference input Zarr dataset")
    parser.add_argument(
        "--output", type=Path, required=True, help="New WeatherGenerator output store"
    )
    parser.add_argument("--channels", nargs="+", help="Dataset channels (default: all variables)")
    parser.add_argument(
        "--permute", action="store_true", help="Reverse both point values and coordinates"
    )
    args = parser.parse_args()

    if args.output.exists() or args.output.is_symlink():
        parser.error(f"Output path already exists: {args.output}")

    dataset = zarr.open_group(str(args.dataset), mode="r")
    available = list(dataset.attrs["variables"])
    channels = available if args.channels is None else args.channels
    missing = sorted(set(channels) - set(available))
    if missing:
        parser.error(f"Unknown dataset channels: {', '.join(missing)}")
    if not channels or len(set(channels)) != len(channels):
        parser.error("Channels must be nonempty and unique")
    indices = None if channels == available else [available.index(name) for name in channels]

    shape = dataset["data"].shape
    if len(shape) != 4 or shape[0] < 3 or shape[1] != len(available) or shape[2] != 1:
        parser.error("Dataset data must have shape (at least 3 times, variables, 1, points)")
    dates = np.asarray(dataset["dates"][:3], dtype="datetime64[ns]")
    if (
        dates.shape != (3,)
        or np.isnat(dates).any()
        or not np.all(np.diff(dates) == np.timedelta64(1, "h"))
    ):
        parser.error("Dataset must start with three valid consecutive hourly timestamps")
    interval = TimeRange(dates[0], dates[1])

    raw_lat = np.asarray(dataset["latitudes"][:])
    raw_lon = np.asarray(dataset["longitudes"][:])
    lat = (2 * np.clip(raw_lat, -90, 90) - raw_lat).astype(np.float32)
    lon = ((raw_lon + 180) % 360 - 180).astype(np.float32)
    if lat.shape != (shape[3],) or lon.shape != (shape[3],):
        parser.error("Dataset coordinates must contain one latitude/longitude per point")
    coords = np.column_stack((lat, lon))
    if args.permute:
        coords = coords[::-1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zarrio_writer(args.output) as writer:
        for step in range(3):
            _write_step(
                writer, dataset, step, channels, indices, coords, dates, interval, args.permute
            )


if __name__ == "__main__":
    main()
