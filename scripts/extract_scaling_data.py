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
        match = re.search(r"Number of Nodes:\s*(\d+)", content)
        return int(match.group(1)) if match else None
    except Exception:
        return None


def extract_metrics_from_run_id(run_id: str, shared_work_dir: Path) -> dict | None:
    metrics_path = shared_work_dir / "results" / run_id / f"{run_id}_train_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        df = pl.read_ndjson(metrics_path)
        if len(df) == 0:
            return None
        final_row = df.tail(1)
        overall_time = final_row.get_column("overall_time_seconds").item() if "overall_time_seconds" in final_row.columns else None
        if overall_time is None:
            return None
        startup_time = final_row.get_column("startup_time_seconds").item() if "startup_time_seconds" in final_row.columns else None
        loss_avg = final_row.get_column("loss_avg_mean").item() if "loss_avg_mean" in final_row.columns else None
        return {
            "overall_time_seconds": overall_time,
            "startup_time_seconds": startup_time,
            "training_time": overall_time - startup_time if startup_time else None,
            "loss_avg_mean": loss_avg,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract strong scaling data from WeatherGenerator runs")
    parser.add_argument("--run-ids", nargs="+", help="List of run-ids to process")
    parser.add_argument("--run-id-file", type=Path, help="File containing run-ids (one per line)")
    parser.add_argument("--logs-base-dir", type=Path, default=Path("/e/scratch/weatherai/logs"), help="Base directory for run logs")
    parser.add_argument("--shared-work-dir", type=Path, default=Path("/e/scratch/weatherai/shared_work"), help="Base directory for shared work/results")
    parser.add_argument("--output", type=Path, default=Path("scaling_data.parquet"), help="Output parquet file path")

    args = parser.parse_args()

    if args.run_ids and args.run_id_file:
        sys.exit("Error: Cannot specify both --run-ids and --run-id-file")
    elif args.run_ids:
        run_ids = args.run_ids
    elif args.run_id_file:
        if not args.run_id_file.exists():
            sys.exit(f"Error: Run-id file not found: {args.run_id_file}")
        run_ids = [line.strip() for line in args.run_id_file.read_text().splitlines() if line.strip()]
    else:
        sys.exit("Error: Must specify either --run-ids or --run-id-file")

    if not run_ids:
        sys.exit("Error: No run-ids provided")

    results = []
    for run_id in run_ids:
        log_pattern = args.logs_base_dir / run_id / "weathermen.*.err"
        err_files = list(log_pattern.parent.glob("weathermen.*.err"))
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
