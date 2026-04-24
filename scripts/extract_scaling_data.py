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

        return {
            "startup_time_seconds": startup_time,
            "training_time": overall_training_time,
            "loss_avg_mean": loss_avg_mean,
        }
    except Exception:
        return None


def extract_detailed_metrics(run_id: str, shared_work_dir: Path, output_path: Path) -> int:
    """Extract detailed metrics pairing timing rows with preceding loss rows.
    
    For each row containing elapsed_training_time_seconds, pair it with the 
    preceding row containing loss metrics. Returns the number of detailed entries extracted.
    """
    metrics_path = shared_work_dir / "results" / run_id / f"{run_id}_train_metrics.json"
    if not metrics_path.exists():
        return 0
    
    try:
        df = pl.read_ndjson(metrics_path)
        if len(df) == 0:
            return 0
        
        # Find rows with elapsed_training_time_seconds (timing rows)
        timing_mask = pl.col("elapsed_training_time_seconds").is_not_null()
        timing_indices = df.with_row_index().filter(timing_mask).get_column("index").to_list()
        
        if len(timing_indices) == 0:
            return 0
        
        # Get all row indices with loss data
        loss_mask = pl.col("loss_avg_mean").is_not_null()
        loss_rows_df = df.with_row_index().filter(loss_mask)
        
        detailed_records = []
        
        for timing_idx in timing_indices:
            # Find the last loss row before this timing row
            loss_rows_before = loss_rows_df.filter(pl.col("index") < timing_idx)
            
            if len(loss_rows_before) == 0:
                continue
            
            # Get the last loss row before timing
            loss_row = loss_rows_before.sort("index").tail(1)
            timing_row = df.with_row_index().filter(pl.col("index") == timing_idx).drop("index")
            
            # Drop index column from loss_row for merging
            loss_row = loss_row.drop("index")
            
            # Merge loss and timing data
            merged = loss_row.join(timing_row, how="cross")
            detailed_records.append(merged)
        
        if len(detailed_records) == 0:
            return 0
        
        # Combine all records
        detailed_df = pl.concat(detailed_records)
        
        # Write to output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_df.write_ndjson(output_path)
        
        return len(detailed_records)
        
    except Exception as e:
        print(f"Error extracting detailed metrics for {run_id}: {e}")
        return 0


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
    detailed_files_created = []
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
            "startup_time_seconds": metrics.get("startup_time_seconds"),
            "training_time": metrics.get("training_time"),
            "loss_avg_mean": metrics.get("loss_avg_mean"),
        }
        results.append(row)
        
        # Extract detailed metrics for this run
        # Create output file with "detailed" suffix before the extension
        output_stem = args.output.stem
        output_suffix = args.output.suffix
        detailed_output = args.output.with_name(f"{output_stem}_detailed{output_suffix}")
        count = extract_detailed_metrics(run_id, args.shared_work_dir, detailed_output)
        if count > 0:
            detailed_files_created.append((detailed_output, count))
            print(f"Extracted {count} detailed metric entries for {run_id}")

    if not results:
        sys.exit("No data extracted")

    df = pl.DataFrame(results)
    if "num_nodes" in df.columns:
        df = df.sort("num_nodes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    df.write_csv(args.output.with_suffix(".csv"))
    
    print(f"\nSummary:")
    print(f"  - Extracted {len(results)} run summaries to {args.output}")
    if detailed_files_created:
        total_detailed = sum(count for _, count in detailed_files_created)
        print(f"  - Extracted {total_detailed} detailed metric entries to {detailed_files_created[0][0]}")


if __name__ == "__main__":
    main()
