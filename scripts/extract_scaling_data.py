#!/usr/bin/env uv run python
"""Extract strong scaling data from WeatherGenerator runs. Outputs parquet with run_id, num_nodes, training_time, overall_time_seconds, loss_avg_mean."""

import argparse
import re
import sys
from pathlib import Path

import polars as pl


def extract_num_nodes(err_log_path: Path) -> int | None:
    if not err_log_path.exists():
        return None
    try:
        content = err_log_path.read_text()
        # Case-insensitive match for "Number of Nodes:" with flexible whitespace
        match = re.search(r"number\s+of\s+nodes\s*:\s*(\d+)", content, re.IGNORECASE)
        return int(match.group(1)) if match else None
    except Exception:
        return None


def extract_metrics_from_run_id(run_id: str, shared_work_dir: Path) -> dict | None:
    """Extract metrics from NDJSON file with startup and training lines.
    
    Format:
    - Line 1: startup_time_seconds
    - Line 2+: loss_avg_mean, LossPhysical.loss_avg, etc.
    """
    metrics_path = shared_work_dir / "results" / run_id / f"{run_id}_train_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        df = pl.read_ndjson(metrics_path)
        if len(df) == 0:
            return None
        
        # Extract startup_time from first row (startup line)
        startup_time = None
        if "startup_time_seconds" in df.columns:
            startup_time = df.select(pl.col("startup_time_seconds").first()).item()
        
        # Extract loss_avg_mean from last non-NaN training row
        loss_avg_mean = None
        if "loss_avg_mean" in df.columns:
            loss_avg_mean = df.select(pl.col("loss_avg_mean").drop_nulls().last()).item()
        
        # Extract training for mini-epoch from last non-NaN row
        overall_training_time = None
        if "elapsed_time_mini_epoch" in df.columns:
            overall_training_time = df.select(pl.col("elapsed_time_mini_epoch").drop_nulls().last()).item()

        # Extract overall_time from last non-NaN row
        overall_time = None
        if "overall_time_seconds" in df.columns:
            overall_time = df.select(pl.col("overall_time_seconds").drop_nulls().last()).item()
        
        return {
            "overall_time_seconds": overall_time,
            "startup_time_seconds": startup_time,
            "training_time": overall_training_time,
            "loss_avg_mean": loss_avg_mean,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract strong scaling data from WeatherGenerator runs")
    parser.add_argument("--run-ids", nargs="+", help="List of run-ids to process")
    parser.add_argument("--logs-base-dir", type=Path, default=Path("logs"), help="Base directory for run logs (default: logs relative to current dir)")
    parser.add_argument("--shared-work-dir", type=Path, default=Path("/e/scratch/weatherai/shared_work"), help="Base directory for shared work/results")
    parser.add_argument("--output", type=Path, default=Path("scaling_data.parquet"), help="Output parquet file path")

    args = parser.parse_args()

    run_ids = args.run_ids
    if not run_ids:
        sys.exit("Error: No run-ids provided")

    results = []
    for run_id in run_ids:
        # Look for weathergen.*.err files (e.g., weathergen.part1.388004.err)
        log_dir = args.logs_base_dir / run_id
        err_files = list(log_dir.glob("weathergen.*.err")) if log_dir.exists() else []
        num_nodes = extract_num_nodes(err_files[0]) if err_files else None
        metrics = extract_metrics_from_run_id(run_id, args.shared_work_dir)
        if metrics is None:
            continue
        row = {
            "run_id": run_id,
            "num_nodes": num_nodes,
            "training_time": metrics.get("training_time"),
            "overall_time_seconds": metrics["overall_time_seconds"],
            "loss_avg_mean": metrics.get("loss_avg_mean"),
        }
        results.append(row)

    if not results:
        sys.exit("No data extracted")

    df = pl.DataFrame(results)
    if "num_nodes" in df.columns:
        df = df.sort("num_nodes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)


if __name__ == "__main__":
    main()
