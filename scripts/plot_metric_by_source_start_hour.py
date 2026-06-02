#!/usr/bin/env -S uv run python

"""Plot evaluation metrics against source window start hour.

This script intentionally reuses WeatherGenerator's evaluation reader and
score implementation so that metrics such as RMSE are computed through the
same tested path as the evaluation package.

Example
-------
uv run python scripts/plot_metric_by_source_start_hour.py \
  --run-ids f09zumgl wevz13ig \
  --labels baseline all_inputs \
  --stream ERA5 \
  --channels 10u 2t \
  --output-dir plots/source_start_hour_compare
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr
from omegaconf import OmegaConf

from weathergen.common.io import zarrio_reader
from weathergen.evaluate.io.wegen_reader import WeatherGenZarrReader
from weathergen.evaluate.scores.score import VerifiedData, get_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)
METRIC_CHOICES = ["rmse", "mse", "mae", "bias"]
ALL_CHANNELS_LABEL = "all_channels"


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    label: str


@dataclass(frozen=True, slots=True)
class ScoreRow:
    run_id: str
    label: str
    stream: str
    channel: str
    forecast_step: int
    lead_time_hours: int | None
    source_start_hour: int
    sample_count: int
    metric: str
    value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot WeatherGenerator metrics by source window start hour."
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        required=True,
        help="Inference run IDs with validation zarr/zip outputs.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels matching --run-ids. Defaults to the run IDs.",
    )
    parser.add_argument("--stream", required=True, help="Stream name to analyse, e.g. ERA5.")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Optional channel subset. Defaults to the common channels across all runs.",
    )
    parser.add_argument(
        "--average-channels",
        action="store_true",
        help=(
            "Aggregate the channel dimension inside the metric calculation and plot a single "
            "curve over all selected channels. For RMSE this computes one RMSE across the "
            "selected channels, not the arithmetic mean of per-channel RMSE values."
        ),
    )
    parser.add_argument(
        "--forecast-steps",
        nargs="+",
        type=int,
        default=None,
        help="Optional forecast step subset. Defaults to the common forecast steps across runs.",
    )
    parser.add_argument(
        "--metric",
        default="rmse",
        choices=METRIC_CHOICES,
        help="Metric to plot. Defaults to rmse.",
    )
    parser.add_argument(
        "--ensemble",
        nargs="+",
        default=["mean"],
        help="Ensemble selection passed to the reader. Defaults to mean.",
    )
    parser.add_argument(
        "--mini-epoch",
        type=int,
        default=0,
        help="Checkpoint index used in the validation output filename. Defaults to 0.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Validation output rank suffix. Defaults to 0.",
    )
    parser.add_argument(
        "--model-base-dir",
        type=Path,
        default=None,
        help="Optional model base directory override for loading run configs.",
    )
    parser.add_argument(
        "--results-base-dir",
        type=Path,
        default=None,
        help="Optional results base directory override for locating validation outputs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional cap for zarr I/O workers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots") / "source_start_hour",
        help="Directory where plots and CSV summaries are written.",
    )
    return parser.parse_args()


def build_run_specs(run_ids: list[str], labels: list[str] | None) -> list[RunSpec]:
    if labels is None:
        return [RunSpec(run_id=run_id, label=run_id) for run_id in run_ids]
    if len(labels) != len(run_ids):
        raise ValueError("--labels must have the same length as --run-ids.")
    return [RunSpec(run_id=run_id, label=label) for run_id, label in zip(run_ids, labels, strict=True)]


def build_reader(
    run_id: str,
    stream: str,
    mini_epoch: int,
    rank: int,
    model_base_dir: Path | None,
    results_base_dir: Path | None,
    max_workers: int | None,
) -> WeatherGenZarrReader:
    eval_cfg = OmegaConf.create(
        {
            "mini_epoch": mini_epoch,
            "rank": rank,
            "max_workers": max_workers,
            "model_base_dir": str(model_base_dir) if model_base_dir is not None else None,
            "results_base_dir": str(results_base_dir) if results_base_dir is not None else None,
            "streams": {stream: {}},
        }
    )
    return WeatherGenZarrReader(eval_cfg, run_id)


def resolve_common_channels(
    readers: list[WeatherGenZarrReader], stream: str, requested_channels: list[str] | None
) -> list[str]:
    channels_by_run = [reader.get_channels(stream) for reader in readers]
    common_channels = set(channels_by_run[0])
    for channels in channels_by_run[1:]:
        common_channels &= set(channels)

    if requested_channels is None:
        resolved = [channel for channel in channels_by_run[0] if channel in common_channels]
    else:
        missing = [channel for channel in requested_channels if channel not in common_channels]
        if missing:
            raise ValueError(
                f"Requested channels are not available in all runs for stream {stream}: {missing}"
            )
        resolved = requested_channels

    if not resolved:
        raise ValueError(f"No common channels found for stream {stream}.")
    return resolved


def resolve_common_fsteps(
    readers: list[WeatherGenZarrReader], requested_fsteps: list[int] | None
) -> list[int]:
    fsteps_by_run = [sorted(int(fstep) for fstep in reader.get_forecast_steps()) for reader in readers]
    common_fsteps = set(fsteps_by_run[0])
    for fsteps in fsteps_by_run[1:]:
        common_fsteps &= set(fsteps)

    if requested_fsteps is None:
        resolved = [fstep for fstep in fsteps_by_run[0] if fstep in common_fsteps]
    else:
        missing = [fstep for fstep in requested_fsteps if fstep not in common_fsteps]
        if missing:
            raise ValueError(f"Requested forecast steps are not available in all runs: {missing}")
        resolved = requested_fsteps

    if not resolved:
        raise ValueError("No common forecast steps found across the requested runs.")
    return resolved


def compute_hourly_metric(
    pred: xr.DataArray,
    gt: xr.DataArray,
    source_interval_start_by_sample: dict[int, np.datetime64],
    metric: str,
    average_channels: bool,
) -> tuple[xr.DataArray, dict[int, int]]:
    sample_ids = [int(sample) for sample in pred.coords["sample"].values.tolist()]
    missing_samples = [sample for sample in sample_ids if sample not in source_interval_start_by_sample]
    if missing_samples:
        raise ValueError(
            "Missing stored source_interval_start values for samples: "
            f"{missing_samples[:5]}"
        )

    source_interval_start_values = np.array(
        [source_interval_start_by_sample[sample] for sample in sample_ids],
        dtype="datetime64[ns]",
    )
    source_start_hour_values = source_interval_start_values.astype("datetime64[h]").astype(int) % 24
    unique_hours, counts = np.unique(source_start_hour_values, return_counts=True)
    count_by_hour = {int(hour): int(count) for hour, count in zip(unique_hours, counts, strict=True)}

    source_start_hour = xr.DataArray(
        source_start_hour_values,
        dims=("sample",),
        coords={"sample": pred.coords["sample"].values},
    )
    source_interval_start = xr.DataArray(
        source_interval_start_values,
        dims=("sample",),
        coords={"sample": pred.coords["sample"].values},
    )
    pred_with_hour = pred.assign_coords(
        source_interval_start=source_interval_start,
        source_start_hour=source_start_hour,
    )
    gt_with_hour = gt.assign_coords(
        source_interval_start=source_interval_start,
        source_start_hour=source_start_hour,
    )

    group_by_coord = "source_start_hour" if len(unique_hours) > 1 else None
    agg_dims = ["sample", "ipoint"]
    if average_channels:
        agg_dims.append("channel")
    score = get_score(
        VerifiedData(pred_with_hour, gt_with_hour, None, None, None),
        metric,
        agg_dims=agg_dims,
        group_by_coord=group_by_coord,
        compute=True,
    )

    if group_by_coord is None:
        score = score.expand_dims({"source_start_hour": [int(unique_hours[0])]})

    return score, count_by_hour


def get_lead_time_hours(score: xr.DataArray) -> int | None:
    if "lead_time" not in score.coords:
        return None
    return int(score.coords["lead_time"].values.astype("timedelta64[h]").astype(int))


def load_source_interval_start_by_sample(
    zarr_path: Path,
    samples: list[int],
    stream: str,
    forecast_step: int,
) -> dict[int, np.datetime64]:
    source_interval_start_by_sample: dict[int, np.datetime64] = {}
    with zarrio_reader(zarr_path) as zio:
        for sample in samples:
            data = zio.get_data(sample, stream, forecast_step)
            target = data.target.as_xarray()
            source_interval_start_by_sample[sample] = np.datetime64(
                target.coords["source_interval_start"].values[0]
            )

    return source_interval_start_by_sample


def compute_rows_for_run(
    run_spec: RunSpec,
    reader: WeatherGenZarrReader,
    stream: str,
    channels: list[str],
    forecast_steps: list[int],
    metric: str,
    ensemble: list[str],
    average_channels: bool,
) -> list[ScoreRow]:
    samples = sorted(int(sample) for sample in reader.get_samples())
    output = reader.get_data(
        stream,
        samples=samples,
        fsteps=forecast_steps,
        channels=channels,
        ensemble=ensemble,
    )

    rows: list[ScoreRow] = []
    for forecast_step in sorted(output.prediction):
        source_interval_start_by_sample = load_source_interval_start_by_sample(
            reader.fname_zarr,
            samples,
            stream,
            int(forecast_step),
        )
        score, count_by_hour = compute_hourly_metric(
            output.prediction[forecast_step],
            output.target[forecast_step],
            source_interval_start_by_sample,
            metric,
            average_channels,
        )
        lead_time_hours = get_lead_time_hours(score)
        score_hours = [int(hour) for hour in score.coords["source_start_hour"].values.tolist()]
        score_channels = (
            [ALL_CHANNELS_LABEL]
            if average_channels
            else [str(channel) for channel in score.coords["channel"].values.tolist()]
        )

        for channel in score_channels:
            values = score if average_channels else score.sel(channel=channel)
            for hour in score_hours:
                rows.append(
                    ScoreRow(
                        run_id=run_spec.run_id,
                        label=run_spec.label,
                        stream=stream,
                        channel=channel,
                        forecast_step=int(forecast_step),
                        lead_time_hours=lead_time_hours,
                        source_start_hour=hour,
                        sample_count=count_by_hour[hour],
                        metric=metric,
                        value=float(values.sel(source_start_hour=hour).item()),
                    )
                )

    return rows


def write_summary_csv(rows: list[ScoreRow], output_dir: Path, metric: str, stream: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{metric}_{stream}_by_source_start_hour.csv"
    fieldnames = [
        "run_id",
        "label",
        "stream",
        "channel",
        "forecast_step",
        "lead_time_hours",
        "source_start_hour",
        "sample_count",
        "metric",
        "value",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return csv_path


def plot_rows(rows: list[ScoreRow], run_specs: list[RunSpec], output_dir: Path, metric: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_order = {run_spec.run_id: index for index, run_spec in enumerate(run_specs)}

    grouped: dict[tuple[str, int], list[ScoreRow]] = {}
    for row in rows:
        grouped.setdefault((row.channel, row.forecast_step), []).append(row)

    for (channel, forecast_step), group_rows in sorted(grouped.items()):
        plt.figure(figsize=(10, 6), dpi=150)
        lead_time_hours = next((row.lead_time_hours for row in group_rows if row.lead_time_hours is not None), None)

        runs_in_group = sorted(
            {row.run_id for row in group_rows},
            key=lambda run_id: run_order[run_id],
        )
        for run_id in runs_in_group:
            run_rows = sorted(
                [row for row in group_rows if row.run_id == run_id],
                key=lambda row: row.source_start_hour,
            )
            plt.plot(
                [row.source_start_hour for row in run_rows],
                [row.value for row in run_rows],
                marker="o",
                linewidth=2,
                label=run_rows[0].label,
            )

        channel_title = "all selected channels" if channel == ALL_CHANNELS_LABEL else channel
        title = (
            f"{metric.upper()} vs source start hour | {channel_title} | forecast step {forecast_step}"
        )
        if lead_time_hours is not None:
            title += f" | lead {lead_time_hours}h"

        plt.title(title)
        plt.xlabel("source window start hour [UTC]")
        plt.ylabel(metric.upper())
        plt.xlim(-0.5, 23.5)
        plt.xticks(range(24))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = output_dir / f"{metric}_{channel}_fstep_{forecast_step}_by_source_start_hour.png"
        LOGGER.info("Saving %s", plot_path)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    run_specs = build_run_specs(args.run_ids, args.labels)
    readers = [
        build_reader(
            run_spec.run_id,
            args.stream,
            args.mini_epoch,
            args.rank,
            args.model_base_dir,
            args.results_base_dir,
            args.max_workers,
        )
        for run_spec in run_specs
    ]

    channels = resolve_common_channels(readers, args.stream, args.channels)
    forecast_steps = resolve_common_fsteps(readers, args.forecast_steps)
    LOGGER.info("Using channels: %s", channels)
    LOGGER.info("Using forecast steps: %s", forecast_steps)

    rows: list[ScoreRow] = []
    for run_spec, reader in zip(run_specs, readers, strict=True):
        LOGGER.info("Processing run %s", run_spec.run_id)
        rows.extend(
            compute_rows_for_run(
                run_spec,
                reader,
                args.stream,
                channels,
                forecast_steps,
                args.metric,
                args.ensemble,
                args.average_channels,
            )
        )

    csv_path = write_summary_csv(rows, args.output_dir, args.metric, args.stream)
    LOGGER.info("Saved summary CSV to %s", csv_path)
    plot_rows(rows, run_specs, args.output_dir, args.metric)


if __name__ == "__main__":
    main()