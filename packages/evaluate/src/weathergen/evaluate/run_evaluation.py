#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "weathergen-evaluate",
#   "weathergen-common",
#   "weathergen-metrics",
# ]
# [tool.uv.sources]
# weathergen-evaluate = { path = "../../../../../packages/evaluate" }
# ///

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from xarray import DataArray

import mlflow
from omegaconf import OmegaConf

from weathergen.common.config import _REPO_ROOT
from weathergen.common.platform_env import get_platform_env
from weathergen.evaluate.io_reader import WeatherGenReader
from weathergen.evaluate.plot_utils import collect_channels
from weathergen.evaluate.utils import (
    calc_scores_per_stream,
    metric_list_to_json,
    plot_data,
    plot_summary,
    retrieve_metric_from_json,
)
from weathergen.metrics.mlflow_utils import (
    MlFlowUpload,
    get_or_create_mlflow_parent_run,
    log_scores,
    setup_mlflow,
)

_logger = logging.getLogger(__name__)

_DEFAULT_PLOT_DIR = _REPO_ROOT / "plots"

_platform_env = get_platform_env()


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
    parser.add_argument(
        "--push-metrics",
        required=False,
        action="store_true",
        help="(optional) Upload scores to MLFlow.",
    )

    args = parser.parse_args(argl)
    if args.config:
        config = Path(args.config)
    else:
        _logger.info(
            "No config file provided, using the default template config (please edit accordingly)"
        )
        config = Path(_REPO_ROOT / "config" / "evaluate" / "eval_config.yml")
    mlflow_client = None
    if args.push_metrics:
        # logging.basicConfig(level=logging.INFO)
        hpc_conf = _platform_env.get_hpc_config()
        assert hpc_conf is not None
        private_home = Path(hpc_conf)
        private_cf = OmegaConf.load(private_home)
        mlflow_client = setup_mlflow(private_cf)
        _logger.info(f"MLFlow client set up: {mlflow_client}")

    evaluate_from_config(OmegaConf.load(config), mlflow_client)


def evaluate_from_config(cfg, mlflow_client):
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
                        try:
                            metric_data = retrieve_metric_from_json(
                                reader,
                                stream,
                                region,
                                metric,
                            )

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
                        except (FileNotFoundError, KeyError):
                            metrics_to_compute.append(metric)

                    if metrics_to_compute:
                        all_metrics, points_per_sample = calc_scores_per_stream(
                            reader, stream, region, metrics_to_compute
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

    if mlflow_client:
        # Reorder scores_dict to push to MLFlow per run_id:
        # Create a new defaultdict with the target structure: [run_id][metric][region][stream]
        reordered_dict: dict[str, dict[str, dict[str, dict[str, DataArray]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        # Iterate through the original dictionary to get all keys and the final value
        for metric, regions_dict in scores_dict.items():
            for region, streams_dict in regions_dict.items():
                for stream, runs_dict in streams_dict.items():
                    for run_id, final_dict in runs_dict.items():
                        # Assign the final_dict to the new structure using the reordered keys
                        reordered_dict[run_id][metric][region][stream] = final_dict

        channels_set = collect_channels(scores_dict, metric, region, runs)

        for run_id, run in runs.items():
            reader = WeatherGenReader(run, run_id, private_paths)
            from_run_id = reader.inference_cfg["from_run_id"]
            parent_run = get_or_create_mlflow_parent_run(mlflow_client, from_run_id)
            _logger.info(f"MLFlow parent run: {parent_run}")
            phase = "eval"
            with mlflow.start_run(run_id=parent_run.info.run_id):
                with mlflow.start_run(
                    run_name=f"{phase}_{from_run_id}_{run_id}",
                    parent_run_id=parent_run.info.run_id,
                    nested=True,
                ) as run:
                    mlflow.set_tags(MlFlowUpload.run_tags(run_id, phase, from_run_id))
                    log_scores(
                        reordered_dict[run_id],
                        mlflow_client,
                        run.info.run_id,
                        channels_set,
                    )

    # plot summary
    if scores_dict and cfg.evaluation.get("summary_plots", True):
        _logger.info("Started creating summary plots..")
        plot_summary(cfg, scores_dict, summary_dir)


if __name__ == "__main__":
    evaluate()
