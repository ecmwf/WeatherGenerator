#!/usr/bin/env uv run python
"""Generate strong scaling plots from parquet data. Single file with subplots per metric."""

import argparse
import os
from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_scaling_plots(df: pl.DataFrame, output_path: Path, metrics: list[str]):
    """Create a single plot with subplots for each metric."""

    # Count valid metrics
    valid_metrics = [m for m in metrics if m in df.columns and df.filter(pl.col(m).is_not_null()).height > 0]
    if not valid_metrics:
        print("No valid metrics to plot")
        return

    n_metrics = len(valid_metrics)

    fig = make_subplots(
        rows=n_metrics, cols=1,
        subplot_titles=valid_metrics,
        vertical_spacing=0.1,
    )

    for idx, metric in enumerate(valid_metrics):
        df_plot = df.filter(pl.col(metric).is_not_null()).sort("num_nodes")

        # Add scatter trace with lines and text labels
        fig.add_trace(
            go.Scatter(
                x=df_plot["num_nodes"],
                y=df_plot[metric],
                mode="lines+markers+text",
                text=df_plot["run_id"],
                textposition="top center",
                name=metric,
                showlegend=False,
                marker=dict(size=10, color="steelblue"),
                line=dict(width=2),
            ),
            row=idx + 1, col=1,
        )

        # Add optimal scaling reference line for training_time
        if metric == "training_time" and "training_time" in df.columns:
            # Find the 1-node training time
            one_node_data = df.filter(pl.col("num_nodes") == 1)
            if one_node_data.height > 0:
                t1 = one_node_data["training_time"].item()
                # Create optimal scaling line: t1 / n for each n
                nodes = df_plot["num_nodes"].to_list()
                optimal_y = [t1 / n for n in nodes]
                fig.add_trace(
                    go.Scatter(
                        x=nodes,
                        y=optimal_y,
                        mode="lines",
                        name="Optimal scaling",
                        line=dict(width=1, color="red", dash="dash"),
                        showlegend=True,
                    ),
                    row=idx + 1, col=1,
                )

        fig.update_xaxes(title_text="Number of Nodes (log scale)", type="log", row=idx + 1, col=1)
        fig.update_yaxes(title_text=metric, row=idx + 1, col=1)

    fig.update_layout(
        height=400 * n_metrics,
        title_text="Scaling Analysis",
        title_x=0.5,
        template="plotly_white",
    )

    fig.write_html(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate scaling plots from parquet data")
    parser.add_argument("--input", type=Path, default=Path("scaling_data.parquet"), help="Input parquet file")
    parser.add_argument("--output_dir", type=Path, default=Path("scaling_plots"), help="Output directory for HTML files")
    parser.add_argument("--output_file_name", type=Path, default=Path("scaling_plots.html"), help="Output HTML file name")
    parser.add_argument("--metrics", nargs="+", default=["training_time", "overall_time_seconds", "loss_avg_mean"], help="Metrics to plot")
    parser.add_argument("--generate-dummy", action="store_true", help="Generate dummy test data")

    args = parser.parse_args()

    if args.generate_dummy:
        print("Generating dummy test data...")
        dummy_data = {
            "run_id": ["run_1node", "run_2node", "run_4node", "run_8node", "run_16node"],
            "num_nodes": [1, 2, 4, 8, 16],
            "training_time": [1000, 520, 270, 140, 75],
            "overall_time_seconds": [1100, 580, 310, 165, 90],
            "loss_avg_mean": [0.45, 0.44, 0.44, 0.43, 0.43],
        }
        df = pl.DataFrame(dummy_data)
        args.input.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(args.input)
        print(f"Created dummy data: {args.input}")

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Use --generate-dummy to create test data")
        return

    print(f"Loading data from: {args.input}")
    df = pl.read_parquet(args.input)
    print(f"Loaded {len(df)} rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    create_scaling_plots(df, os.path.join(args.output_dir, args.output_file_name), args.metrics)


if __name__ == "__main__":
    main()
