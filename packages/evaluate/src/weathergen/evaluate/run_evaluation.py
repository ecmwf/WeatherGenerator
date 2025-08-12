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
    metric_list_to_json,
    plot_data,
    plot_summary,
    retrieve_metric_from_json,
)

_logger = logging.getLogger(__name__)


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

    # configure logging
    logging.basicConfig(level=logging.INFO)

    # load configuration
    cfg = OmegaConf.load(args.config)

    runs = cfg.run_ids

    _logger.info(f"Detected {len(runs)} runs")

    assert cfg.get("output_scores_dir"), (
        "Please provide a path to the directory where the score files are stored or will be saved."
    )
    out_scores_dir = Path(cfg.output_scores_dir)
    out_scores_dir.mkdir(parents=True, exist_ok=True)

    metrics = cfg.evaluation.metrics
    regions = cfg.evaluation.get("regions", ["global"])

    # to get a structure like: scores_dict[metric][region][stream][run_id] = plot
    scores_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for run_id, run in runs.items():
        _logger.info(f"RUN {run_id}: Getting data...")

        streams = run["streams"].keys()

        for stream in streams:
            _logger.info(f"RUN {run_id}: Processing stream {stream}...")

            stream_dict = run["streams"][stream]

            if stream_dict.get("plotting"):
                _logger.info(f"RUN {run_id}: Plotting stream {stream}...")
                plot_data(cfg, run_id, stream, stream_dict)

            if stream_dict.get("evaluation"):
                _logger.info(f"Retrieve or compute scores for {run_id} - {stream}...")

                for region in regions:
                    metrics_to_compute = []

                    for metric in metrics:
                        try:
                            metric_data = retrieve_metric_from_json(
                                out_scores_dir,
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
                            cfg, run_id, stream, region, metrics_to_compute
                        )

                        metric_list_to_json(
                            [all_metrics],
                            [points_per_sample],
                            [stream],
                            region,
                            out_scores_dir,
                            run_id,
                            run.epoch,
                        )

                    for metric in metrics_to_compute:
                        scores_dict[metric][stream][run_id] = all_metrics.sel(
                            {"metric": metric}
                        )
    # plot summary
    if scores_dict and cfg.summary_plots:
        _logger.info("Started creating summary plots..")
        plot_summary(cfg, scores_dict, print_summary=cfg.print_summary)


if __name__ == "__main__":
    evaluate()
