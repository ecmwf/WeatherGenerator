#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "weathergen-evaluate",
#   "weathergen-common",
# ]
# [tool.uv.sources]
# weathergen-evaluate = { path = "../../../../../packages/evaluate" }
# ///

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from weathergen.src.weathergen.utils.config import load_config, set_paths, _REPO_ROOT
from weathergen.evaluate.utils import (
    calc_scores_per_stream,
    metric_list_to_json,
    plot_data,
    plot_summary,
    retrieve_metric_from_json,
)

_logger = logging.getLogger(__name__)

_DEFAULT_PLOT_DIR = _REPO_ROOT / "plots"


def evaluate() -> None:
    parser = argparse.ArgumentParser(
        description="Fast evaluation of WeatherGenerator runs."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration yaml file for plotting. e.g. config/plottig_config.yaml",
    )

    args = parser.parse_args()

    # configure logging
    logging.basicConfig(level=logging.INFO)

    # load configuration
    cfg = OmegaConf.load(args.config)

    runs = cfg.run_ids

    _logger.info(f"Detected {len(runs)} runs")

    # Directory to store the summary plots
    private_paths = cfg.get("private_paths", None)
    summary_dir = Path(
        cfg.get("summary_dir", _DEFAULT_PLOT_DIR)
    )  # base directory where summary plots will be stored

    metrics = cfg.evaluation.metrics
    regions = cfg.evaluation.get("regions", ["global"])

    # to get a structure like: scores_dict[metric][region][stream][run_id] = plot
    scores_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for run_id, run in runs.items():
        _logger.info(f"RUN {run_id}: Getting data...")

        # Allow for run ID specific directories 
        # If results_base_dir is not provided, default paths are used 
        results_base_dir = run.get("results_base_dir", None)

        if results_base_dir is None:
            cf_run = load_config(private_paths, run_id, run["epoch"])
            cf_run = set_paths(cf_run)
            results_dir = cf_run["run_path"]
            logging.info(f"Results directory obtained automatically: {results_dir}")
        else:
            results_dir = Path(results_base_dir) / run_id
            logging.info(f"Results directory parsed: {results_dir}")

        runplot_base_dir = Path(
            run.get("runplot_base_dir", results_base_dir.parent)
        )  # base directory where map plots and histograms will be stored
        metric_base_dir = Path(
            run.get("metric_base_dir", results_base_dir)
        )  # base directory where score files will be stored

        streams = run["streams"].keys()
        metric_dir = metric_base_dir / "evaluation"

        for stream in streams:
            _logger.info(f"RUN {run_id}: Processing stream {stream}...")

            stream_dict = run["streams"][stream]

            if stream_dict.get("plotting"):
                _logger.info(f"RUN {run_id}: Plotting stream {stream}...")
                _ = plot_data(
                    cfg, results_dir, runplot_base_dir, stream, stream_dict
                )

            if stream_dict.get("evaluation"):
                # Create output directory if it does not exist
                metric_dir.mkdir(parents=True, exist_ok=True)

                _logger.info(f"Retrieve or compute scores for {run_id} - {stream}...")

                for region in regions:
                    metrics_to_compute = []

                    for metric in metrics:
                        try:
                            metric_data = retrieve_metric_from_json(
                                metric_base_dir,
                                run_id,
                                stream,
                                region,
                                metric,
                                run.epoch,
                            )
                            scores_dict[metric][region][stream][run_id] = metric_data
                        except (FileNotFoundError, KeyError, ValueError):
                            metrics_to_compute.append(metric)

                    if metrics_to_compute:
                        all_metrics, points_per_sample = calc_scores_per_stream(
                            cfg, results_dir, stream, region, metrics_to_compute
                        )

                        metric_list_to_json(
                            [all_metrics],
                            [points_per_sample],
                            [stream],
                            region,
                            metric_base_dir,
                            run_id,
                            run.epoch,
                        )

                    for metric in metrics_to_compute:
                        scores_dict[metric][region][stream][run_id] = all_metrics.sel(
                            {"metric": metric}
                        )
    # plot summary
    if scores_dict and cfg.summary_plots:
        _logger.info("Started creating summary plots..")
        plot_summary(cfg, scores_dict, summary_dir, print_summary=cfg.print_summary)


if __name__ == "__main__":
    evaluate()
