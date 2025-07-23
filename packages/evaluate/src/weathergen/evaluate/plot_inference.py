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
from plotter import Plotter
from utils import (
    calc_scores_per_stream,
    metric_list_to_json,
    plot_data,
    plot_summary,
    retrieve_metric_from_json,
)

_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast evaluation of weather generator runs."
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
    models = cfg.model_ids

    _logger.info(f"Detected {len(models)} models")

    assert cfg.get("output_scores_dir"), (
        "Please provide a path to the directory where the score files are stored or will be saved."
    )
    out_scores_dir = Path(cfg.output_scores_dir)
    out_scores_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(cfg.results_dir)
    metrics = cfg.evaluation.metrics

    # to get a structure like: scores_dict[metric][stream][model_id] = plot
    scores_dict = defaultdict(lambda: defaultdict(dict))

    for model_id, model in models.items():
        plotter = Plotter(cfg, model_id)
        _logger.info(f"MODEL {model_id}: Getting data...")

        streams = model["streams"].keys()

        for stream in streams:
            _logger.info(f"MODEL {model_id}: Processing stream {stream}...")

            stream_dict = model["streams"][stream]

            _logger.info(f"MODEL {model_id}: Plotting stream {stream}...")
            plots = plot_data(cfg, model_id, stream, stream_dict)

            if stream_dict.evaluation:
                _logger.info(f"Retrieve or compute scores for {model_id} - {stream}...")

                metrics_to_compute = []
                for metric in metrics:
                    try:
                        metric_data = retrieve_metric_from_json(
                            out_scores_dir,
                            model_id,
                            stream,
                            metric,
                            model.epoch,
                            model.rank,
                        )
                        scores_dict[metric][stream][model_id] = metric_data
                    except (FileNotFoundError, KeyError, ValueError):
                        metrics_to_compute.append(metric)
                if metrics_to_compute:
                    all_metrics, points_per_sample = calc_scores_per_stream(
                        cfg, model_id, stream, metrics_to_compute
                    )

                    metric_list_to_json(
                        [all_metrics],
                        [points_per_sample],
                        [stream],
                        out_scores_dir,
                        model_id,
                        model.epoch,
                        model.rank,
                    )

                    all_metrics = all_metrics.compute()

                    for metric in metrics_to_compute:
                        scores_dict[metric][stream][model_id] = all_metrics.sel(
                            {"metric": metric}
                        )


# plot summary
if cfg.summary_plots:
    _logger.info("Started creating summary plots..")
    plot_summary(cfg, scores_dict, print_summary = cfg.print_summary)
