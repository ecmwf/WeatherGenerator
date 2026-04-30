#!/usr/bin/env uv run python
"""Generate scaling plots from parquet/ndjson data using matplotlib only.

Two entrypoints:
- standard: plots run-level metrics vs num_nodes
- detailed: plots sample-level metrics vs total_num_samples
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
VALID_IMAGE_SUFFIXES = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def resolve_input_path(path: Path) -> Path:
    """Resolve relative input paths against cwd first, then the script directory."""
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    script_candidate = SCRIPT_DIR / path
    if script_candidate.exists():
        return script_candidate

    return cwd_candidate


def resolve_output_path(path: Path) -> Path:
    """Ensure the output path uses a supported image suffix."""
    if path.suffix.lower() in VALID_IMAGE_SUFFIXES:
        return path
    return path.with_suffix(".png")


def read_table(path: Path) -> pl.DataFrame:
    """Read parquet or ndjson automatically."""
    try:
        print("Read as parquet")
        return pl.read_parquet(path)
    except Exception:
        print("Read as NDJSON")
        return pl.read_ndjson(path)


def color_map_for_nodes(node_counts: list) -> dict:
    return {node: PALETTE[i % len(PALETTE)] for i, node in enumerate(node_counts)}


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path = resolve_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_scaling_table(df: pl.DataFrame, input_path: Path, show_run_ids: bool = False) -> None:
    """Generate a PNG table image with scaling metrics from the parquet file.
    
    Columns: num_nodes, training_time, ideal_time, efficiency (optionally run_id)
    """
    # Check if required columns exist
    if "num_nodes" not in df.columns or "training_time" not in df.columns:
        print("Warning: Required columns (num_nodes, training_time) not found in data")
        return
    
    # Filter out rows with null values in required columns
    df_filtered = df.filter(
        pl.col("num_nodes").is_not_null() & pl.col("training_time").is_not_null()
    ).sort("num_nodes")
    
    if len(df_filtered) == 0:
        print("No valid data for scaling table")
        return
    
    # Get the 1-node training time for ideal time calculation
    one_node_data = df_filtered.filter(pl.col("num_nodes") == 1)
    if one_node_data.height == 0:
        print("Warning: No 1-node data found for ideal time calculation")
        return
    
    t1 = one_node_data["training_time"].item()
    
    # Derive scaling type from input filename
    input_name_lower = input_path.name.lower()
    if "weak" in input_name_lower:
        scaling_type = "Weak"
    elif "strong" in input_name_lower:
        scaling_type = "Strong"
    else:
        scaling_type = "Strong"  # Default to strong
    
    # Build table data with proper formatting
    has_run_id = "run_id" in df_filtered.columns
    col_names = ["# Nodes", "Training Time (seconds)", "Ideal Time (seconds)", "Efficiency"]
    if show_run_ids and has_run_id:
        col_names.insert(0, "run_id")
    
    table_data = []
    for row in df_filtered.iter_rows(named=True):
        num_nodes = row["num_nodes"]
        training_time = row["training_time"]
        
        if num_nodes == 1:
            ideal_time = "-"
            efficiency = "-"
        else:
            if scaling_type == "Strong":
                # Strong scaling: ideal time = t1 / num_nodes
                ideal_val = t1 / num_nodes
                efficiency_val = ideal_val / training_time
            else:
                # Weak scaling: ideal time = t1 (same work per node)
                ideal_val = t1
                efficiency_val = min(1.0, t1 / training_time)
            
            ideal_time = f"{ideal_val:.2f}"
            efficiency = f"{efficiency_val:.2f}"
        
        row_data = []
        if show_run_ids and has_run_id:
            row_data.append(str(row.get("run_id", "")))
        row_data.extend([
            str(num_nodes),
            f"{training_time:.2f}",
            ideal_time,
            efficiency
        ])
        table_data.append(row_data)
    
    # Generate output filename: input_stem_table.csv
    output_path = input_path.with_name(input_path.stem + "_table.csv")
    
    # Build DataFrame for CSV output
    df_table_data = {}
    for i, col in enumerate(col_names):
        df_table_data[col] = [row[i] for row in table_data]
    
    df_table = pl.DataFrame(df_table_data)
    
    # Write to CSV
    df_table.write_csv(output_path)
    print(f"Saved scaling table: {output_path}")


def plot_standard_scaling(
    df: pl.DataFrame,
    output_path: Path,
    scaling_type: str,
    metrics: list[str],
    x_scale: str,
    y_scale: str,
    y_metric: str,
    show_run_ids: bool = False,
) -> None:
    """Plot run-level scaling data vs num_nodes."""
    metric_labels = {
        "training_time": "Training Time (seconds)",
        "loss_avg_mean": "Average Loss",
        "normalized_throughput": "Speedup",
        "efficiency": "Scaling Efficiency",
    }

    valid_metrics = [m for m in metrics if m in df.columns and df.filter(pl.col(m).is_not_null()).height > 0]
    if not valid_metrics:
        print("No valid metrics to plot")
        return

    fig, axes = plt.subplots(len(valid_metrics), 1, figsize=(12, 6 * len(valid_metrics)), squeeze=False)

    for idx, metric in enumerate(valid_metrics):
        ax = axes[idx][0]
        df_plot = df.filter(pl.col(metric).is_not_null()).sort("num_nodes")
        node_counts = df_plot["num_nodes"].unique().to_list() if "num_nodes" in df_plot.columns else []
        colors = color_map_for_nodes(node_counts)

        # Handle normalized_throughput and efficiency metrics
        if y_metric == "normalized_throughput" and metric == "training_time":
            # Calculate normalized throughput: T1 / T
            one_node_data = df.filter(pl.col("num_nodes") == 1)
            if one_node_data.height > 0:
                t1 = one_node_data["training_time"].item()
                # Create a new dataframe with normalized throughput
                df_plot = df_plot.with_columns(
                    (t1 / pl.col("training_time")).alias("normalized_throughput")
                )
                plot_y = df_plot["normalized_throughput"]
            else:
                print("Warning: No 1-node data found for normalized throughput calculation")
                continue
        elif y_metric == "efficiency" and metric == "training_time":
            # Calculate efficiency based on scaling type
            one_node_data = df.filter(pl.col("num_nodes") == 1)
            if one_node_data.height > 0:
                t1 = one_node_data["training_time"].item()
                if scaling_type == "strong":
                    # Strong scaling: efficiency = (t1 / num_nodes) / training_time
                    df_plot = df_plot.with_columns(
                        ((t1 / pl.col("num_nodes")) / pl.col("training_time")).alias("efficiency")
                    )
                else:
                    # Weak scaling: efficiency = min(1.0, t1 / training_time)
                    df_plot = df_plot.with_columns(
                        pl.min_horizontal(pl.lit(1.0), t1 / pl.col("training_time")).alias("efficiency")
                    )
                plot_y = df_plot["efficiency"]
            else:
                print("Warning: No 1-node data found for efficiency calculation")
                continue
        else:
            plot_y = df_plot[metric]

        ax.plot(
            df_plot["num_nodes"],
            plot_y,
            "o-",
            color="steelblue",
            markersize=8,
        )

        if show_run_ids:
            for x, y, label in zip(df_plot["num_nodes"], plot_y.to_list(), df_plot["run_id"]):
                ax.text(x, y, label, ha="center", va="bottom", fontsize=8)

        if metric == "training_time" and y_metric in ("time", "normalized_throughput", "efficiency") and "training_time" in df.columns:
            one_node_data = df.filter(pl.col("num_nodes") == 1)
            if one_node_data.height > 0:
                t1 = one_node_data["training_time"].item()
                nodes = df_plot["num_nodes"].to_list()
                if y_metric == "efficiency":
                    # For efficiency, optimal is always 1.0 (100% efficiency)
                    optimal_y = [1.0 for _ in nodes]
                elif scaling_type == "weak":
                    if y_metric == "normalized_throughput":
                        # For normalized throughput, optimal is 1.0 (no speedup loss)
                        optimal_y = [1.0 for _ in nodes]
                    else:
                        optimal_y = [t1 for _ in nodes]
                elif scaling_type == "strong":
                    if y_metric == "normalized_throughput":
                        # For normalized throughput, optimal is n (linear speedup)
                        optimal_y = [float(n) for n in nodes]
                    else:
                        optimal_y = [t1 / n for n in nodes]
                else:
                    raise ValueError(f"Invalid scaling type: {scaling_type}")
                ax.plot(nodes, optimal_y, "r--", linewidth=1, label="Optimal scaling")

                # Show per-point efficiency loss as a vertical line and factor label.
                # Use plot_y (normalized throughput if applicable) instead of df_plot[metric]
                for x, y, y_opt in zip(nodes, plot_y.to_list(), optimal_y):
                    if y_opt == 0:
                        continue
                    factor = y / y_opt
                    ax.vlines(x, y_opt, y, colors="gray", linestyles=":", linewidth=1, alpha=0.7)
                    y_mid = (y + y_opt) / 2
                    ax.annotate(
                        f"{factor:.2f}",
                        xy=(x, y_mid),
                        xytext=(4, 0),
                        textcoords="offset points",
                        fontsize=14,
                        fontweight="bold",
                        color="dimgray",
                        va="center",
                    )
                ax.legend()

        ax.set_xscale(x_scale)
        if y_scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel("Number of Nodes", fontsize=16)
        if y_metric == "normalized_throughput" and metric == "training_time":
            ax.set_ylabel("Speedup", fontsize=16)
        elif y_metric == "efficiency" and metric == "training_time":
            ax.set_ylabel("Scaling Efficiency", fontsize=16)
        else:
            ax.set_ylabel(metric_labels.get(metric, metric), fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_detailed_scaling(
    df: pl.DataFrame,
    output_path: Path,
    x_scale: str,
    y_scale: str,
) -> None:
    """Plot sample-level detailed scaling data vs total_num_samples."""
    required_cols = ["total_num_samples", "elapsed_training_time_seconds", "loss_avg_mean", "num_nodes"]
    if not all(col in df.columns for col in required_cols):
        print("Detailed metrics not available in this dataset")
        print(f"Available columns: {df.columns}")
        return

    df_plot = df.filter(
        pl.col("total_num_samples").is_not_null()
        & (pl.col("total_num_samples") > 0)
        & pl.col("elapsed_training_time_seconds").is_not_null()
        & pl.col("loss_avg_mean").is_not_null()
        & pl.col("num_nodes").is_not_null()
    ).sort("num_nodes", "total_num_samples")

    if len(df_plot) == 0:
        print("No valid data for detailed scaling plots")
        return

    node_counts = sorted(df_plot["num_nodes"].unique().to_list())
    colors = color_map_for_nodes(node_counts)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    for node_count in node_counts:
        df_node = df_plot.filter(pl.col("num_nodes") == node_count).sort("total_num_samples")
        ax.plot(
            df_node["total_num_samples"],
            df_node["elapsed_training_time_seconds"],
            "o-",
            color=colors[node_count],
            markersize=6,
            label=f"{node_count} nodes",
        )
    ax.set_xscale(x_scale)
    if y_scale == "log":
        ax.set_yscale("log")
    ax.set_ylabel("Elapsed Training Time (seconds)", fontsize=16)
    ax.set_title("Elapsed Training Time vs Samples", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Node Count")

    ax = axes[1]
    for node_count in node_counts:
        df_node = df_plot.filter(pl.col("num_nodes") == node_count).sort("total_num_samples")
        ax.plot(
            df_node["total_num_samples"],
            df_node["loss_avg_mean"],
            "o-",
            color=colors[node_count],
            markersize=6,
            label=f"{node_count} nodes",
        )
    ax.set_xscale(x_scale)
    if y_scale == "log":
        ax.set_yscale("log")
    ax.set_xlabel("Total Number of Samples", fontsize=16)
    ax.set_ylabel("Average Loss", fontsize=16)
    ax.set_title("Loss vs Samples", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Node Count")

    fig.suptitle("Detailed Scaling Analysis", fontsize=16)
    plt.tight_layout()
    save_figure(fig, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate scaling plots from parquet or NDJSON data")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    standard = subparsers.add_parser("standard", help="Plot run-level scaling metrics vs num_nodes")
    standard.add_argument("--type", required=True, choices=["strong", "weak"], help="Scaling type")
    standard.add_argument("--input", type=Path, default=Path("scaling_data.parquet"), help="Input parquet/ndjson file")
    standard.add_argument("--output", type=Path, default=None, help="Output image path")
    standard.add_argument("--y-scale", choices=["linear", "log"], default="linear", help="Y-axis scale")
    standard.add_argument("--x-scale", choices=["linear", "log"], default="log", help="X-axis scale")
    standard.add_argument("--y-metric", choices=["time", "normalized_throughput", "efficiency"], default="normalized_throughput", help="Y-axis metric: 'time' for time-to-solution, 'normalized_throughput' for T1/T, or 'efficiency' for scaling efficiency")
    standard.add_argument("--show-run-ids", action="store_true", help="Show run_id labels on the plot and in the output table")

    loss_only = subparsers.add_parser("loss", help="Plot loss metrics vs num_nodes (separate from throughput)")
    loss_only.add_argument("--type", required=True, choices=["strong", "weak"], help="Scaling type")
    loss_only.add_argument("--input", type=Path, default=Path("scaling_data.parquet"), help="Input parquet/ndjson file")
    loss_only.add_argument("--output", type=Path, default=None, help="Output image path")
    loss_only.add_argument("--y-scale", choices=["linear", "log"], default="log", help="Y-axis scale")
    loss_only.add_argument("--x-scale", choices=["linear", "log"], default="log", help="X-axis scale")
    loss_only.add_argument("--show-run-ids", action="store_true", help="Show run_id labels on the plot and in the output table")

    detailed = subparsers.add_parser("detailed", help="Plot sample-level detailed scaling metrics")
    detailed.add_argument("--input", type=Path, default=Path("scaling_data_detailed.parquet"), help="Input detailed parquet/ndjson file")
    detailed.add_argument("--output", type=Path, default=None, help="Output image path")
    detailed.add_argument("--y-scale", choices=["linear", "log"], default="log", help="Y-axis scale")
    detailed.add_argument("--x-scale", choices=["linear", "log"], default="log", help="X-axis scale")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "standard":
        input_path = resolve_input_path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return

        output_path = args.output or input_path.with_suffix(".png")

        print(f"Loading data from: {input_path}")
        try:
            df = read_table(input_path)
        except Exception as e:
            print("Error: Could not read input file as parquet or NDJSON")
            print(str(e))
            return
        print(f"Loaded {len(df)} rows")
        # Standard mode: only plot training_time with normalized throughput or time
        metrics_to_plot = ["training_time"]
        plot_standard_scaling(df, output_path, args.type, metrics_to_plot, args.x_scale, args.y_scale, args.y_metric, args.show_run_ids)
        # Generate scaling table
        generate_scaling_table(df, input_path, show_run_ids=args.show_run_ids)
        return

    if args.mode == "loss":
        input_path = resolve_input_path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return

        output_path = args.output or input_path.with_suffix(".loss.png")

        print(f"Loading data from: {input_path}")
        try:
            df = read_table(input_path)
        except Exception as e:
            print("Error: Could not read input file as parquet or NDJSON")
            print(str(e))
            return
        print(f"Loaded {len(df)} rows")
        # Loss mode: only plot loss_avg_mean
        metrics_to_plot = ["loss_avg_mean"]
        plot_standard_scaling(df, output_path, args.type, metrics_to_plot, args.x_scale, args.y_scale, "time", args.show_run_ids)
        # Generate scaling table
        generate_scaling_table(df, input_path, show_run_ids=args.show_run_ids)
        return

    if args.mode == "detailed":
        input_path = resolve_input_path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return

        output_path = args.output or input_path.with_suffix(".png")

        print(f"Loading detailed data from: {input_path}")
        try:
            df = read_table(input_path)
        except Exception as e:
            print("Error: Could not read detailed file as parquet or NDJSON")
            print(str(e))
            return
        print(f"Loaded {len(df)} detailed rows")
        plot_detailed_scaling(df, output_path, args.x_scale, args.y_scale)
        return

    raise ValueError(f"Unknown mode: {args.mode}")



if __name__ == "__main__":
    main()
