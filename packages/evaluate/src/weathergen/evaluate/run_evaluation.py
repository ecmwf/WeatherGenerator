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

from weathergen.evaluate.utils import (
    calc_scores_per_stream,
    check_availability,
    metric_list_to_json,
    plot_data,
    plot_summary,
    retrieve_metric_from_json,
)
from weathergen.utils.config import _REPO_ROOT, load_config, load_model_config

_logger = logging.getLogger(__name__)

_DEFAULT_PLOT_DIR = _REPO_ROOT / "plots"


def evaluate() -> None:
    # By default, arguments from the command line are read.
    evaluate_from_args(sys.argv[1:])


def evaluate_from_args(argl: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Fast evaluation of WeatherGenerator runs."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration yaml file for plotting. e.g. config/plottig_config.yaml",
    )

    args = parser.parse_args(argl)
    evaluate_from_config(OmegaConf.load(args.config))


def evaluate_from_config(cfg):
    # configure logging
    logging.basicConfig(level=logging.INFO)

    # load configuration

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

        # Load model configuration and set (run-id specific) directories
        # If results_base_dir and model_base_dir are not provided, default paths are used
        model_base_dir = run.get("model_base_dir", None)

        if private_paths:
            _logger.info(
                f"Loading config for run {run_id} from private paths: {private_paths}"
            )
            run_cfg = load_config(private_paths, run_id, run["epoch"])
        else:
            _logger.info(
                f"Loading config for run {run_id} from model directory: {model_base_dir}"
            )
            run_cfg = load_model_config(run_id, run["epoch"], model_base_dir)

        results_base_dir = run.get(
            "results_base_dir", None
        )  # base directory where results will be stored
        if not results_base_dir:
            results_base_dir = Path(run_cfg["run_path"])
            logging.info(
                f"Results directory obtained from model config: {results_base_dir}"
            )
        else:
            logging.info(f"Results directory parsed: {results_base_dir}")

        runplot_base_dir = Path(
            run.get("runplot_base_dir", results_base_dir)
        )  # base directory where map plots and histograms will be stored
        metrics_base_dir = Path(
            run.get("metrics_base_dir", results_base_dir)
        )  # base directory where score files will be stored

        results_dir, runplot_dir = (
            Path(results_base_dir) / run_id,
            Path(runplot_base_dir) / run_id,
        )
        # for backward compatibility allow metric_dir to be specified in the run config
        metrics_dir = Path(
            run.get("metrics_dir", metrics_base_dir / run_id / "evaluation")
        )

        streams = run["streams"].keys()

        for stream in streams:
            _logger.info(f"RUN {run_id}: Processing stream {stream}...")

            stream_dict = run["streams"][stream]

            if stream_dict.get("plotting"):
                _logger.info(f"RUN {run_id}: Plotting stream {stream}...")
                _ = plot_data(cfg, run_cfg, results_dir, runplot_dir, stream)

            if stream_dict.get("evaluation"):
                _logger.info(f"Retrieve or compute scores for {run_id} - {stream}...")

                for region in regions:
                    metrics_to_compute = []

                    for metric in metrics:
                        try:
                            metric_data = retrieve_metric_from_json(
                                metrics_dir,
                                run_id,
                                stream,
                                region,
                                metric,
                                run.epoch,
                            )
                            checked, (channels, fsteps, samples) = check_availability(
                                cfg, stream, results_dir, metric_data, mode = "evaluation"
                            )
                            if not checked:
                                metrics_to_compute.append(metric)
                            else:
                                # simply select the chosen eval channels, samples, fsteps here...
                                scores_dict[metric][region][stream][run_id] = (
                                    metric_data.sel(
                                        sample=samples,
                                        channel=channels,
                                        forecast_step=fsteps,
                                    )
                                )
                        except (FileNotFoundError, KeyError):
                            metrics_to_compute.append(metric)

                    if metrics_to_compute:

                        all_metrics, points_per_sample = calc_scores_per_stream(
                            cfg, run_id, results_dir, stream, region, metrics_to_compute
                        )

                        metric_list_to_json(
                            [all_metrics],
                            [points_per_sample],
                            [stream],
                            region,
                            metrics_dir,
                            run_id,
                            run.epoch,
                        )

                    for metric in metrics_to_compute:
                        scores_dict[metric][region][stream][run_id] = all_metrics.sel(
                            {"metric": metric}
                        )

    # plot summary
    summary_plots = cfg.get("summary_plots", True)
    if scores_dict and summary_plots:
        _logger.info("Started creating summary plots..")
        plot_summary(cfg, scores_dict, summary_dir, print_summary=cfg.print_summary)


if __name__ == "__main__":
    evaluate()
