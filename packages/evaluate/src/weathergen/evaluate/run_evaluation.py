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
import sys
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from weathergen.common.config import _REPO_ROOT
from weathergen.evaluate.io_reader import WeatherGenReader
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
    # By default, arguments from the command line are read.
    evaluate_from_args(sys.argv[1:])


def evaluate_from_args(argl: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Fast evaluation of WeatherGenerator runs.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration yaml file for plotting. e.g. config/plottig_config.yaml",
    )

    args = parser.parse_args(argl)
    if args.config:
        config = Path(args.config)
    else:
        _logger.info(
            "No config file provided, using the default template config (please edit accordingly)"
        )
        config = Path(_REPO_ROOT / "config" / "evaluate" / "eval_config.yml")
    evaluate_from_config(OmegaConf.load(config))


def evaluate_from_config(cfg):
    # configure logging
    logging.basicConfig(level=logging.INFO)

    # load configuration

    runs = cfg.run_ids

    _logger.info(f"Detected {len(runs)} runs")

    # Directory to store the summary plots
    private_paths = cfg.get("private_paths", None)
    summary_dir = Path(
        cfg.evaluation.get("summary_dir", _DEFAULT_PLOT_DIR)
    )  # base directory where summary plots will be stored

    metrics = cfg.evaluation.metrics
    regions = cfg.evaluation.get("regions", ["global"])
    plot_score_maps = cfg.evaluation.get("plot_score_maps", False)

    global_plotting_opts = cfg.get("global_plotting_options", {})

    # to get a structure like: scores_dict[metric][region][stream][run_id] = plot
    scores_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for run_id, run in runs.items():
        _logger.info(f"RUN {run_id}: Getting data...")

        reader = WeatherGenReader(run, run_id, private_paths)

        for stream in reader.streams:
            _logger.info(f"RUN {run_id}: Processing stream {stream}...")

            stream_dict = reader.get_stream(stream)
            if not stream_dict:
                _logger.info(
                    f"Stream {stream} does not exist in source data or config file is empty. Skipping."
                )
                continue

            if stream_dict.get("plotting"):
                _logger.info(f"RUN {run_id}: Plotting stream {stream}...")
                _ = plot_data(reader, stream, global_plotting_opts)

            if stream_dict.get("evaluation"):
                _logger.info(f"Retrieve or compute scores for {run_id} - {stream}...")

                for region in regions:
                    metrics_to_compute = []

                    for metric in metrics:
                        
                        metric_data = retrieve_metric_from_json(
                            reader,
                            stream,
                            region,
                            metric,
                        )

                        if metric_data is None or plot_score_maps:
                            metrics_to_compute.append(metric)
                            continue
                    
                        available_data = reader.check_availability(
                            stream, metric_data, mode="evaluation"
                        )

                        if not available_data.score_availability:
                            metrics_to_compute.append(metric)
                        else:
                            # simply select the chosen eval channels, samples, fsteps here...
                            scores_dict[metric][region][stream][run_id] = metric_data.sel(
                                sample=available_data.samples,
                                channel=available_data.channels,
                                forecast_step=available_data.fsteps,
                            )

                    if metrics_to_compute:
                        all_metrics, points_per_sample = calc_scores_per_stream(
                            reader, stream, region, metrics_to_compute, plot_score_maps
                        )
                    
                        metric_list_to_json(
                            reader,
                            [all_metrics],
                            [points_per_sample],
                            [stream],
                            region,
                        )

                    for metric in metrics_to_compute:
                        scores_dict[metric][region][stream][run_id] = all_metrics.sel(
                            {"metric": metric}
                        )

    # plot summary
    if scores_dict and cfg.evaluation.get("summary_plots", True):
        _logger.info("Started creating summary plots..")
        plot_summary(cfg, scores_dict, summary_dir)


if __name__ == "__main__":
    evaluate()
