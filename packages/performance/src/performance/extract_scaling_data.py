#!/usr/bin/env uv run python
"""Extract strong scaling data from WeatherGenerator runs. Outputs parquet with run_id, num_nodes, training_time, overall_time_seconds, loss_avg_mean."""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def extract_num_nodes_from_output(run_id: str, logs_base_dir: Path) -> int | None:
    """Extract num_nodes from output.*.txt file in the run directory.

    Looks for 'nNodes' pattern in output files.
    """
    run_log_dir = logs_base_dir / run_id
    if not run_log_dir.exists():
        return None

    # Look for output.*.txt files
    output_files = list(run_log_dir.glob("output.*.txt"))
    if not output_files:
        # Fallback to err files if no output files found
        output_files = list(run_log_dir.glob("weathergen.*.err"))

    for output_file in output_files:
        try:
            content = output_file.read_text()
            # Look for nNodes pattern: "nNodes 128" (space-separated, as in NCCL logs)
            match = re.search(r"nNodes\s+(\d+)", content, re.IGNORECASE)
            if match:
                return int(match.group(1))
        except Exception:
            continue

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
        df = pd.read_json(metrics_path, lines=True)
        if len(df) == 0:
            return None

        # Extract startup_time from first row (startup line)
        startup_time = None
        if "startup_time_seconds" in df.columns:
            val = df["startup_time_seconds"].dropna()
            startup_time = val.iloc[0] if len(val) > 0 else None

        # Extract loss_avg_mean from last non-NaN training row
        loss_avg_mean = None
        if "loss_avg_mean" in df.columns:
            val = df["loss_avg_mean"].dropna()
            loss_avg_mean = val.iloc[-1] if len(val) > 0 else None

        # Extract training time for mini-epoch from last non-NaN row
        overall_training_time = None
        if "elapsed_time_mini_epoch" in df.columns:
            val = df["elapsed_time_mini_epoch"].dropna()
            overall_training_time = val.iloc[-1] if len(val) > 0 else None

        return {
            "startup_time_seconds": startup_time,
            "training_time": overall_training_time,
            "loss_avg_mean": loss_avg_mean,
        }
    except Exception:
        return None


def extract_detailed_metrics(
    run_id: str, shared_work_dir: Path, num_nodes: int | None = None
) -> list[pd.DataFrame]:
    """Extract detailed metrics pairing timing rows with preceding loss rows.

    For each row containing elapsed_training_time_seconds, pair it with the
    preceding row containing loss metrics. Returns a list of DataFrames.
    """
    metrics_path = shared_work_dir / "results" / run_id / f"{run_id}_train_metrics.json"
    if not metrics_path.exists():
        return []

    try:
        df = pd.read_json(metrics_path, lines=True)
        if len(df) == 0:
            return []

        # Find rows with elapsed_training_time_seconds (timing rows)
        if "elapsed_training_time_seconds" not in df.columns:
            return []
        timing_indices = df.index[df["elapsed_training_time_seconds"].notna()].tolist()

        if not timing_indices:
            return []

        # Find rows with loss data
        if "loss_avg_mean" not in df.columns:
            return []
        loss_indices = set(df.index[df["loss_avg_mean"].notna()].tolist())

        timing_cols = [
            "elapsed_training_time_seconds",
            "total_num_samples",
            "average_samples_per_second",
        ]

        detailed_records = []

        for timing_idx in timing_indices:
            # Find the last loss row before this timing row
            loss_rows_before = [i for i in loss_indices if i < timing_idx]
            if not loss_rows_before:
                continue

            last_loss_idx = max(loss_rows_before)

            # Build record dict from loss row + timing row
            record = {"run_id": run_id}
            if num_nodes is not None:
                record["num_nodes"] = num_nodes

            record["loss_avg_mean"] = df.at[last_loss_idx, "loss_avg_mean"]

            for col in timing_cols:
                if col in df.columns:
                    record[col] = df.at[timing_idx, col]

            detailed_records.append(pd.DataFrame([record]))

        return detailed_records

    except Exception as e:
        print(f"Error extracting detailed metrics for {run_id}: {e}")
        import traceback

        traceback.print_exc()
        return []


def parse_run_ids(run_ids_str: list[str]) -> list[tuple[int | None, str]]:
    """Parse run-ids argument which can be:
    1. A list of run-ids (old format): ["run1", "run2"] -> [(None, "run1"), (None, "run2")]
    2. A dict mapping num_nodes to run-ids (new format): "{1: run1, 4: run2}" -> [(1, "run1"), (4, "run2")]

    Returns list of (num_nodes, run_id) tuples.
    """
    if len(run_ids_str) == 1:
        # Check if it looks like a dict: "{key: value, ...}"
        stripped = run_ids_str[0].strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            # Parse as dict format: {num_nodes: run_id, ...}
            import ast

            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict):
                    # Convert string keys to int if needed
                    result = []
                    for k, v in parsed.items():
                        key = int(k) if isinstance(k, str) and k.isdigit() else k
                        result.append((key, str(v)))
                    return result
            except (ValueError, SyntaxError):
                pass

        # Single run-id or comma-separated list
        run_ids = [r.strip() for r in run_ids_str[0].split(",") if r.strip()]
        return [(None, run_id) for run_id in run_ids]

    # Multiple arguments - treat as list of run-ids
    return [(None, run_id) for run_id in run_ids_str]


def extract_num_nodes_from_output(run_id: str, logs_base_dir: Path) -> int | None:
    """Extract num_nodes from output.*.txt file in the run directory.

    Looks for 'nNodes' pattern in output files.
    """
    run_log_dir = logs_base_dir / run_id
    if not run_log_dir.exists():
        return None

    # Look for output.*.txt files
    output_files = list(run_log_dir.glob("output.*.txt"))
    if not output_files:
        # Fallback to err files if no output files found
        output_files = list(run_log_dir.glob("weathergen.*.err"))

    for output_file in output_files:
        try:
            content = output_file.read_text()
            # Look for nNodes pattern: "nNodes 128" (space-separated, as in NCCL logs)
            match = re.search(r"nNodes\s+(\d+)", content, re.IGNORECASE)
            if match:
                return int(match.group(1))
        except Exception:
            continue

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract strong scaling data from WeatherGenerator runs. "
        "Run-ids can be provided as a list (--run-ids run1 run2) or as a dict mapping num_nodes to run-ids "
        "(--run-ids '{1: run1, 4: run2}'). If num_nodes is not provided in the dict, it will be extracted from output.*.txt files."
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        help="Run-ids to process. Can be: (1) list: run1 run2 run3, or (2) dict: '{1: run1, 4: run2, 8: run3}'",
    )
    parser.add_argument(
        "--logs-base-dir",
        type=Path,
        default=Path("logs"),
        help="Base directory for run logs (default: logs relative to current dir)",
    )
    parser.add_argument(
        "--shared-work-dir",
        type=Path,
        default=Path("/e/scratch/weatherai/shared_work"),
        help="Base directory for shared work/results",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("scaling_data.parquet"), help="Output parquet file path"
    )

    args = parser.parse_args()

    run_id_mapping = parse_run_ids(args.run_ids)
    if not run_id_mapping:
        sys.exit("Error: No run-ids provided")

    results = []
    all_detailed_records = []

    for num_nodes, run_id in run_id_mapping:
        # If num_nodes not provided, extract from output.*.txt file
        if num_nodes is None:
            num_nodes = extract_num_nodes_from_output(run_id, args.logs_base_dir)

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

        detailed_records = extract_detailed_metrics(run_id, args.shared_work_dir, num_nodes)
        if detailed_records:
            all_detailed_records.extend(detailed_records)
            print(
                f"Extracted {len(detailed_records)} detailed metric entries for {run_id} ({num_nodes} nodes)"
            )

    if not results:
        sys.exit("No data extracted")

    df = pd.DataFrame(results)
    if "num_nodes" in df.columns:
        df = df.sort_values("num_nodes").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    df.to_csv(args.output.with_suffix(".csv"), index=False)

    # Write detailed metrics if any were collected
    if all_detailed_records:
        detailed_df = pd.concat(all_detailed_records, ignore_index=True)

        desired_cols = [
            "run_id",
            "num_nodes",
            "elapsed_training_time_seconds",
            "total_num_samples",
            "average_samples_per_second",
            "loss_avg_mean",
        ]
        available_cols = [c for c in desired_cols if c in detailed_df.columns]
        detailed_df = detailed_df[available_cols]

        output_stem = args.output.stem
        detailed_output = args.output.with_name(f"{output_stem}_detailed{args.output.suffix}")

        detailed_df.to_parquet(detailed_output, index=False)
        detailed_df.to_csv(detailed_output.with_suffix(".csv"), index=False)

    print("\nSummary:")
    print(f"  - Extracted {len(results)} run summaries to {args.output}")
    if all_detailed_records:
        print(
            f"  - Extracted {len(all_detailed_records)} detailed metric entries to {detailed_output}"
        )


if __name__ == "__main__":
    main()
