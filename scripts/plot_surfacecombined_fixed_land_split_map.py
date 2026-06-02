#!/usr/bin/env python3
"""Plot one train day and one validation day from a fixed-land SurfaceCombined split."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import zarr

INDEX_NAME = "idx_197001010000_1"
BASE_DATETIME = np.datetime64("1970-01-01T00:00:00", "ns")
HOUR = np.timedelta64(1, "h")

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train and validation station coordinates for one day each."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--train-date", required=True, help="Day to plot from the train set.")
    parser.add_argument(
        "--validation-date", required=True, help="Day to plot from the validation set."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--marker-size", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def hour_offset(value: np.datetime64) -> int:
    return int((value - BASE_DATETIME) / HOUR)


def coarse_row_bounds_for_window(
    index: zarr.Array, start: np.datetime64, end: np.datetime64
) -> tuple[int, int]:
    start_hour = hour_offset(start)
    end_hour = hour_offset(end)
    coarse_start_hour = max(start_hour - 1, 0)
    return int(index[coarse_start_hour]), int(index[end_hour])


def require_columns(colnames: list[str], names: list[str]) -> dict[str, int]:
    return {name: colnames.index(name) for name in names}


def unique_coords_for_day(
    path: Path, day_start: np.datetime64, day_end: np.datetime64
) -> npt.NDArray[np.float32]:
    group = zarr.open_group(path, mode="r")
    data = group["data"]
    dates = group["dates"]
    index = group[INDEX_NAME]
    columns = require_columns(list(data.attrs["colnames"]), ["lat", "lon"])
    row_start, row_end = coarse_row_bounds_for_window(index, day_start, day_end)
    data_chunk = data[row_start:row_end]
    date_chunk = dates[row_start:row_end][:, 0]
    mask = (date_chunk >= day_start) & (date_chunk < day_end)
    coords = data_chunk[mask][:, [columns["lat"], columns["lon"]]].astype(np.float32, copy=False)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.unique(coords, axis=0)


def plot_points(
    train_coords: npt.NDArray[np.float32],
    validation_coords: npt.NDArray[np.float32],
    train_date: str,
    validation_date: str,
    output: Path,
    title: str,
    marker_size: float,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(15, 8), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#dbe7f0", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f4efe3", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.35, edgecolor="#4a4a4a")
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.15, edgecolor="#8a8a8a")
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=0.25,
        color="#6e7d86",
        alpha=0.35,
        linestyle="--",
        draw_labels=False,
    )

    if len(train_coords):
        ax.scatter(
            train_coords[:, 1],
            train_coords[:, 0],
            s=marker_size,
            color="#2b6cb0",
            alpha=0.75,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            label=f"Train {train_date} ({len(train_coords):,} unique coords)",
            zorder=3,
        )
    if len(validation_coords):
        ax.scatter(
            validation_coords[:, 1],
            validation_coords[:, 0],
            s=marker_size,
            color="#d97706",
            alpha=0.8,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            label=f"Validation {validation_date} ({len(validation_coords):,} unique coords)",
            zorder=4,
        )

    ax.legend(loc="lower left", frameon=True, framealpha=0.92, facecolor="white")
    ax.set_title(title or "SurfaceCombined Fixed-Land Spatial Split", fontsize=14, pad=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_day = np.datetime64(args.train_date, "ns")
    validation_day = np.datetime64(args.validation_date, "ns")
    train_coords = unique_coords_for_day(args.train, train_day, train_day + np.timedelta64(1, "D"))
    validation_coords = unique_coords_for_day(
        args.validation,
        validation_day,
        validation_day + np.timedelta64(1, "D"),
    )
    plot_points(
        train_coords,
        validation_coords,
        args.train_date,
        args.validation_date,
        args.output,
        args.title,
        args.marker_size,
        args.dpi,
    )
    LOGGER.info("Saved plot to %s", args.output)
    LOGGER.info("Train unique coords on %s: %s", args.train_date, len(train_coords))
    LOGGER.info(
        "Validation unique coords on %s: %s",
        args.validation_date,
        len(validation_coords),
    )


if __name__ == "__main__":
    main()