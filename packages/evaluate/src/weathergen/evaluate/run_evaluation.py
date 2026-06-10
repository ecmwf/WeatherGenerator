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

# Standard library
import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import mlflow
from mlflow.client import MlflowClient
from omegaconf import DictConfig, OmegaConf, open_dict

# Local application / package
from weathergen.common.logger import init_loggers
from weathergen.common.paths import _REPO_ROOT
from weathergen.common.platform_env import get_platform_env
from weathergen.evaluate.io.csv_reader import CsvReader
from weathergen.evaluate.io.merge_reader import WeatherGenMergeReader
from weathergen.evaluate.io.wegen_reader import (
    WeatherGenJsonReader,
    WeatherGenReader,
    WeatherGenZarrReader,
)
from weathergen.evaluate.plotting.plot_orchestration import (
    plot_data,
    plot_summary,
    plot_timeseries_summary,
    run_score_map_pipeline,
    run_score_timeseries_pipeline,
)
from weathergen.evaluate.plotting.plot_utils import collect_channels
from weathergen.evaluate.scores.score_orchestration import (
    calc_scores_per_stream,
    metric_list_to_json,
)
from weathergen.evaluate.utils.dict_utils import merge, parse_metric_params, triple_nested_dict
from weathergen.metrics.mlflow_utils import (
    MlFlowUpload,
    get_or_create_mlflow_parent_run,
    log_scores,
    setup_mlflow,
)

_DEFAULT_PLOT_DIR = _REPO_ROOT / "plots"

_logger = logging.getLogger(__name__)
_platform_env = get_platform_env()


#################################################################


def evaluate() -> None:
    """entry point for evaluation script."""
    evaluate_from_args(sys.argv[1:])


def evaluate_from_args(argl: list[str]) -> None:
    """
    Wrapper of evaluate_from_config.

    Parameters
    ----------
    argl:
       List of arguments passed from terminal
    """
    # configure logging
    init_loggers()
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
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        help=(
            "Overwrite individual config options."
            " Individual items should be of the form: parent_obj.nested_obj=value."
            " NOTE: cannot be used for run_ids (use --run-ids instead)."
        ),
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=None,
        help=(
            "Filter run_ids from the config to only these."
            " E.g. --run-ids wu4wy9os fy6fgscn so67dku1"
        ),
    )

    args = parser.parse_args(argl)
    if args.config:
        config = Path(args.config)
    else:
        _logger.info(
            "No config file provided, using the default template config (please edit accordingly)"
        )
        config = Path(_REPO_ROOT / "config" / "evaluate" / "eval_config.yml")
    mlflow_client: MlflowClient | None = None
    if args.push_metrics:
        hpc_conf = _platform_env.get_hpc_config()
        assert hpc_conf is not None
        private_home = Path(hpc_conf)
        private_cf = OmegaConf.load(private_home)
        assert isinstance(private_cf, DictConfig)
        mlflow_client = setup_mlflow(private_cf)
        _logger.info(f"MLFlow client set up: {mlflow_client}")

    cf = OmegaConf.load(config)
    assert isinstance(cf, DictConfig)

    # Disable struct flag so that --options and --run-ids can freely modify keys.
    OmegaConf.set_struct(cf, False)

    if args.options:
        # Filter out any run_ids= items — those must use --run-ids instead.
        cli_items = [item for item in args.options if not item.startswith("run_ids=")]
        if len(cli_items) != len(args.options):
            _logger.warning(
                "run_ids= in --options is not supported (it's a dict, not a list). "
                "Use --run-ids instead. Ignoring run_ids= items."
            )
        if cli_items:
            cli_overwrite = OmegaConf.from_cli(cli_items)
            cf = OmegaConf.merge(cf, cli_overwrite)
            _logger.info(f"Applied --options overwrites: {cli_items}")

    if args.run_ids:
        existing = cf.get("run_ids", {})
        cf.run_ids = {k: existing.get(k, {}) for k in args.run_ids}
        _logger.info(f"Overwritten run_ids to: {args.run_ids}")

    evaluate_from_config(cf, mlflow_client)


def get_reader(
    reader_type: str,
    run: dict,
    run_id: str,
    private_paths: dict[str, str],
    region: str | None = None,
    metric: dict[str, object] | None = None,
):
    if reader_type == "zarr":
        reader = WeatherGenZarrReader(run, run_id, private_paths)
    elif reader_type == "csv":
        reader = CsvReader(run, run_id, private_paths)
    elif reader_type == "json":
        reader = WeatherGenJsonReader(run, run_id, private_paths, region, metric)
    elif reader_type == "merge":
        reader = WeatherGenMergeReader(run, run_id, private_paths)
    elif reader_type == "jsonmerge":
        reader = WeatherGenMergeReader(
            run, run_id, private_paths, region, metric, reader_type="json"
        )
    else:
        raise ValueError(f"Unknown reader type: {reader_type}")
    return reader


def _process_stream(
    run_id: str,
    run: dict,
    stream: str,
    private_paths: dict[str, str],
    global_plotting_opts: dict[str, object],
    regions: list[str],
    metrics: dict[str, object],
    plot_score_options: dict[str, object],
) -> tuple[str, str, dict[str, dict[str, dict[str, float]]]]:
    """
    Worker function for a single stream of a single run.
    Returns a dictionary with the scores instead of modifying shared dict.
    Parameters
    ----------

    run_id:
        Run identification string.
    run:
        Configuration dictionary for the given run.
    stream:
        String to be processed
    private_paths:
        List of private paths to be used to retrieve directories
    global_plotting_opts:
        Dictionary containing all common plotting options
    regions:
        List of regions to be processed.
    metrics:
        Dict of metrics to be processed and their parameters.
    plot_score_options:
        Dictionary containing all common score calculation options.
    """
    type_ = run.get("type", "zarr")
    reader = get_reader(type_, run, run_id, private_paths, regions, metrics)

    stream_dict = reader.get_stream(stream)
    if not stream_dict:
        _logger.info(f"Stream {stream} not found for run {run_id}. Skipping.")
        return run_id, stream, {}, {}

    needs_plotting = stream_dict.get("plotting") and type_ == "zarr"
    needs_scoring = stream_dict.get("evaluation", False)

    output_data = None
    if (needs_plotting or needs_scoring) and type_ == "zarr":
        available_data = reader.check_availability(stream, mode="evaluation")

        output_data = None
        if available_data.score_availability:
            output_data = reader.get_data(
                stream,
                fsteps=available_data.fsteps,
                samples=available_data.samples,
                channels=available_data.channels,
                ensemble=available_data.ensemble,
            )

            _logger.info(f"RUN {run_id} - {stream}: Data loaded successfully.")

    # Plotting (pass pre-loaded data)
    if needs_plotting:
        plot_data(reader, stream, global_plotting_opts, output_data=output_data)

    # Scoring per stream
    if not needs_scoring:
        return run_id, stream, {}, {}

    plot_score_maps = plot_score_options.get("plot_score_maps", False) and type_ == "zarr"
    plot_score_timeseries = (
        plot_score_options.get("plot_score_timeseries", False) and type_ == "zarr"
    )

    stream_loaded_scores, recomputable_metrics = reader.load_scores(stream, regions, metrics)
    scores_dict = stream_loaded_scores
    if recomputable_metrics:
        metrics_to_compute = recomputable_metrics
        regions_to_compute = list(set(recomputable_metrics.keys()))
    elif plot_score_maps or plot_score_timeseries:
        metrics_to_compute = {r: metrics for r in regions}
        regions_to_compute = regions
    else:
        return run_id, stream, scores_dict, {}

    stream_computed_scores = calc_scores_per_stream(
        reader,
        stream,
        regions_to_compute,
        metrics_to_compute,
        output_data=output_data,
    )
    metric_list_to_json(reader, stream, stream_computed_scores, regions_to_compute)
    scores_dict = merge(stream_loaded_scores, stream_computed_scores)

    if plot_score_maps:
        run_score_map_pipeline(
            reader,
            stream,
            regions_to_compute,
            metrics_to_compute,
            output_data=output_data,
            global_plotting_options=global_plotting_opts,
            plot_score_options=plot_score_options,
        )

    if plot_score_timeseries:
        ts_scores = run_score_timeseries_pipeline(
            reader,
            stream,
            regions_to_compute,
            metrics_to_compute,
            output_data=output_data,
            global_plotting_options=global_plotting_opts,
        )
    else:
        ts_scores = {}

    return run_id, stream, scores_dict, ts_scores


def evaluate_from_config(cfg: dict, mlflow_client: MlflowClient | None) -> None:
    """
    Main function that controls evaluation plotting and scoring.
    Parameters
    ----------
    cfg:
        Configuration input stored as dictionary.
    mlflow_client:
        Optional MLFlow client for uploading scores.
    """
    with open_dict(cfg):
        cfg.evaluation.metrics = parse_metric_params(cfg.evaluation.metrics)
    runs = cfg.run_ids
    _logger.info(f"Detected {len(runs)} runs")
    private_paths = cfg.get("private_paths")
    summary_dir = Path(cfg.evaluation.get("summary_dir", _DEFAULT_PLOT_DIR))
    metrics = cfg.evaluation.metrics

    plot_score_options = {
        "plot_score_maps": cfg.evaluation.get("plot_score_maps", False),
        "plot_score_animations": cfg.evaluation.get("plot_score_animations", False),
        "plot_score_timeseries": cfg.evaluation.get("plot_score_timeseries", False),
    }

    global_plotting_opts = cfg.get("global_plotting_options", {})
    default_streams = cfg.get("default_streams", {})
    max_workers = cfg.get("max_workers")  # global hard cap for parallel workers

    tasks = []

    # Build tasks per stream — avoid constructing heavyweight readers here;
    # _process_stream will create its own reader when it actually needs one.
    for run_id, run in runs.items():
        if "streams" not in run:
            run["streams"] = default_streams

        # Propagate top-level max_workers into each run dict so that readers
        # and orchestration code can pick it up via eval_cfg.get("max_workers").
        if max_workers is not None and "max_workers" not in run:
            run["max_workers"] = max_workers

        for stream in run.get("streams", {}):
            regions = cfg.evaluation.get(
                "regions", run.get("streams", {}).get(stream, {}).get("regions", ["global"])
            )
            tasks.append(
                {
                    "run_id": run_id,
                    "run": run,
                    "stream": stream,
                    "private_paths": private_paths,
                    "global_plotting_opts": global_plotting_opts,
                    "regions": regions,
                    "metrics": metrics,
                    "plot_score_options": plot_score_options,
                }
            )

    scores_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # timeseries_scores[metric][region][stream][run_id][fstep] = xr.DataArray
    timeseries_scores: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    results = [_process_stream(**task) for task in tasks]

    for run_id, stream, stream_scores, ts_scores in results:
        for metric, regions_dict in stream_scores.items():
            for region, streams_dict in regions_dict.items():
                for stream_key, runs_dict in streams_dict.items():
                    scores_dict[metric][region][stream_key].update(runs_dict)

        # Accumulate timeseries scores: ts_scores[metric][region][fstep]
        for metric, region_dict in ts_scores.items():
            for region, fstep_dict in region_dict.items():
                timeseries_scores[metric][region][stream][run_id].update(fstep_dict)

    # MLFlow logging
    if mlflow_client:
        reordered_dict = defaultdict(triple_nested_dict)
        for metric, regions_dict in scores_dict.items():
            for region, streams_dict in regions_dict.items():
                for stream, runs_dict in streams_dict.items():
                    for run_id, data in runs_dict.items():
                        reordered_dict[run_id][metric][region][stream] = data

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
                ) as mlflow_run:
                    mlflow.set_tags(MlFlowUpload.run_tags(run_id, phase, from_run_id))
                    log_scores(
                        reordered_dict[run_id],
                        mlflow_client,
                        mlflow_run.info.run_id,
                        channels_set,
                    )

    # summary plots
    if scores_dict:
        plot_summary(cfg, scores_dict, summary_dir)
    if timeseries_scores:
        plot_timeseries_summary(cfg, timeseries_scores, summary_dir)


if __name__ == "__main__":
    evaluate()
