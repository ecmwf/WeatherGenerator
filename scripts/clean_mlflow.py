#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "weathergen-metrics",
# ]
# [tool.uv.sources]
# weathergen-metrics = { path = "../packages/metrics" }
# ///

import argparse
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from mlflow.entities import Metric, Run
from mlflow.tracking import MlflowClient

from weathergen.metrics.mlflow_utils import get_experiment_id, setup_mlflow

_logger = logging.getLogger("atmo_training")


@dataclass
class LastMetric:
    """Last metric to keep for each metric"""

    value: float
    timestamp_ms: int
    step: int
    # The number of occurences of this metric.
    metric_count: int


def latest_metric_point(client: MlflowClient, run_id: str, key: str) -> LastMetric | None:
    """
    Return (value, timestamp_ms, step) for the latest metric point by timestamp.
    If timestamps tie, prefer larger step.
    """
    history: list[Metric] = client.get_metric_history(run_id, key)
    if not history:
        return None

    best = None  # (timestamp, step, value)
    for m in history:
        cand = (m.timestamp, m.step, m.value)
        if best is None:
            best = cand
        else:
            # primary: timestamp, secondary: step
            if cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand

    ts, step, val = best
    return LastMetric(val, ts, step, len(history))


_BLACKLISTED_SUFFIXES = [s for i in range(2, 100) for s in [f".{i}", f".step{i}", f".step_{i}"]]


def clone_run_latest_metrics(
    client: MlflowClient,
    run_id: str,
    dry_run: bool = False,
    delete_original: bool = True,
) -> str | None:
    src = client.get_run(run_id)
    if src.info.lifecycle_stage == "deleted":
        _logger.warning(f"Run {run_id} already deleted")
        return None

    exp_id = src.info.experiment_id
    src_tags: dict[str, str] = dict(src.data.tags or {})
    src_params: dict[str, str] = dict(src.data.params or {})

    # Copy latest metrics per key, based on metric history timestamps
    metric_keys: list[str] = list((src.data.metrics or {}).keys())
    seen_blacklisted: list[str] = []
    latest_metrics: dict[str, LastMetric] = {}
    for key in metric_keys:
        if any(key.endswith(s) for s in _BLACKLISTED_SUFFIXES):
            seen_blacklisted.append(key)
            continue
        latest = latest_metric_point(client, run_id, key)
        if latest is not None:
            latest_metrics[key] = latest

    run_name = src.info.run_name
    if len(seen_blacklisted) == 0 and (
        len(latest_metrics) == 0 or max([m.metric_count for m in latest_metrics.values()]) == 1
    ):
        _logger.info(f"{run_id}:{run_name} is clean: tags={src_tags}")
        return None
    _logger.info(f"{run_id}:{run_name} tags={src_tags} metrics={latest_metrics}")

    # Create destination run with same start_time and tags.
    if dry_run:
        dst_run_id = "DRY_RUN_NO_RUN_CREATED"
    else:
        dst = client.create_run(
            experiment_id=exp_id, start_time=src.info.start_time, tags=src_tags, run_name=run_name
        )
        dst_run_id = dst.info.run_id

        # Copy params
        for k, v in src_params.items():
            # Mlflow params are strings; keep as-is.
            client.log_param(dst_run_id, k, v)

        for key, m in latest_metrics.items():
            client.log_metric(dst_run_id, key, m.value, timestamp=m.timestamp_ms, step=m.step)

        # Preserve termination info (end_time + status) if present
        # src.info.status is a string like "FINISHED" / "FAILED" / "KILLED" / "RUNNING"
        if src.info.end_time is not None:
            client.set_terminated(
                dst_run_id,
                status=src.info.status,
                end_time=src.info.end_time,
            )
        else:
            # If source is not terminated, keep destination as RUNNING; do nothing.
            pass

    # Delete original run (MLflow delete_run is typically a soft delete)
    if delete_original and not dry_run:
        _logger.info(f"{run_id}: delete")
        client.delete_run(run_id)

    return dst_run_id


def _fetch_runs(client: MlflowClient, max_runs: int, exp_id: str, before_ts_ms: int) -> list[Run]:
    res: list[Run] = []
    batch = client.search_runs(
        [exp_id],
        filter_string=f"attributes.start_time <= '{before_ts_ms}'",
        order_by=["attributes.start_time DESC"],
        max_results=max_runs,
    )
    _logger.info(f"Read {len(batch.to_list())} runs")
    res += batch.to_list()
    while batch is not None and batch.token is not None:
        batch = client.search_runs(
            [exp_id],
            filter_string=f"attributes.start_time <= '{before_ts_ms}'",
            order_by=["attributes.start_time DESC"],
            max_results=max_runs,
            page_token=batch.token,
        )
        res += batch.to_list()
        _logger.info(f"Read {len(batch.to_list())} runs")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""" Cleans up old MLFlow runs to remove the metrics history.

- takes all the child runs in mlflow before a cutoff dates (ex: train_xxxx before 2025-1-1)

- for each of these runs, makes a copy in mlflow that keepn only the last metric point.
Tags, configs, params are preserved otherwise.

- deletes the old run.
"""
    )
    parser.add_argument(
        "--before",
        required=True,
        help="YYYY-MM-DD date (UTC) used as an exclusive cutoff."
        " All the runs before will be cleaned up.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not create new runs or delete originals.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keeps the original run after cloning latest metrics. Leads to duplicate runs.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=1000,
        help="Maximum number of runs to scan.",
    )
    args = parser.parse_args()
    try:
        before_dt = datetime.strptime(args.before, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        parser.error("--before must be in YYYY-MM-DD format")
    before_ts_ms = int(before_dt.timestamp() * 1000)
    logging.basicConfig(level=logging.INFO)
    _logger.info("Setting up MLFlow")
    client = setup_mlflow(None)
    exp_id = get_experiment_id(None)
    _logger.info(f"Exp id: {exp_id}")
    exp = client.get_experiment(exp_id)
    _logger.info(f"before timestamp: {before_ts_ms}")
    runs = _fetch_runs(client, args.max_runs, exp_id, before_ts_ms)
    _logger.info(f"Found {len(runs)} runs")
    for idx, run in enumerate(runs):
        run_id = run.info.run_id
        run_tags = run.data.tags or {}
        if run.info.start_time is not None and run.info.start_time >= before_ts_ms:
            _logger.info(f"{run_id} ({idx} / {len(runs)}): start_time >= cutoff, skip")
            continue
        if run_tags.get("mlflow.parentRunId") is None:
            # Do not delete the parent runs, just the children.
            # The parent runs do not have info
            _logger.info(f"{run_id}: parent, skip")
        else:
            _logger.info(f"{run_id} ({idx} / {len(runs)}): processing")
            clone_run_latest_metrics(
                client, run_id, dry_run=args.dry_run, delete_original=not args.keep_original
            )
