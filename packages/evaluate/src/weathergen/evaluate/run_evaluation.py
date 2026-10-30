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
from weathergen.evaluate.plotting.plot_orchestration import plot_evaluation, plot_stream_data
from weathergen.evaluate.plotting.plot_utils import collect_channels
from weathergen.evaluate.plotting.score_maps import plot_score_maps
from weathergen.evaluate.scores.init_hour import compute_scores_by_init_hour
from weathergen.evaluate.scores.score_orchestration import (
    MetricResult,
    calc_scores_per_stream,
    metric_list_to_json,
    results_to_dict,
)
from weathergen.evaluate.utils.clim_utils import get_climatology, needs_climatology
from weathergen.evaluate.utils.config_compat import get_plot_score_options, parse_plot_config
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
    if reader_type in ("zarr", "anemoi-target"):
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
    write_scores: bool = True,
) -> tuple[str, str, dict[str, dict[str, MetricResult]]]:
    """
    Worker function for a single stream of a single run.

    Returns ``(run_id, stream, results)`` with ``results[region][metric] -> MetricResult``:
    the scores loaded from the JSON files, overridden by the ones computed here.  Empty
    when the stream is not scored.
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
    write_scores:
        Write the computed scores to the JSON score files.  Off on the workers of a
        distributed run, where the master writes the merged scores.
    """
    type_ = run.get("type", "zarr")
    reader = get_reader(type_, run, run_id, private_paths, regions, metrics)

    stream_dict = reader.get_stream(stream)
    if not stream_dict:
        _logger.info(f"Stream {stream} not found for run {run_id}. Skipping.")
        return run_id, stream, {}

    # Data plots, score maps and init-hour scores need the data: readers that load it only
    reads_data = type_ in ("zarr", "anemoi-target")
    needs_plotting = stream_dict.get("plotting") and reads_data
    needs_scoring = stream_dict.get("evaluation", False)

    # --- Determine scoring state before loading any data ---
    for option, name in (("plot_score_maps", "score_map"), ("plot_score_init_hour", "init_hour")):
        if plot_score_options.get(option, False) and not reads_data:
            _logger.warning(f"RUN {run_id} - {stream}: '{name}' not supported for type '{type_}'.")
    do_score_maps = plot_score_options.get("plot_score_maps", False) and reads_data
    plot_score_init_hour = plot_score_options.get("plot_score_init_hour", False) and reads_data

    recomputable_metrics = {}
    stream_loaded_scores: dict = {}
    needs_score_recomputation = False

    if needs_scoring:
        stream_loaded_scores, recomputable_metrics = reader.load_scores(stream, regions, metrics)
        needs_score_recomputation = (
            do_score_maps or plot_score_init_hour or bool(recomputable_metrics)
        ) and reads_data

    # --- Load data only when necessary ---
    output_data = None
    if needs_score_recomputation:
        available_data = reader.check_availability(stream, mode="evaluation")
        if available_data.score_availability:
            output_data = reader.get_data(
                stream,
                fsteps=available_data.fsteps,
                samples=available_data.samples,
                channels=available_data.channels,
                ensemble=available_data.ensemble,
            )
            _logger.info(f"RUN {run_id} - {stream}: Data loaded successfully.")

    # Field plots: pass pre-loaded data if available, otherwise plot_data loads its own subset
    if needs_plotting:
        plot_stream_data(reader, stream, global_plotting_opts, output_data=output_data)

    # Scoring per stream
    if not needs_scoring:
        return run_id, stream, {}

    # cached scores as MetricResults; the ones computed below override them
    results: dict[str, dict[str, MetricResult]] = {}
    for metric, by_region in stream_loaded_scores.items():
        for region, by_stream in by_region.items():
            for stream_key, by_run in by_stream.items():
                for run_key, score in by_run.items():
                    results.setdefault(region, {})[metric] = MetricResult(
                        metric, region, stream_key, run_key, score
                    )

    if recomputable_metrics:
        metrics_to_compute = recomputable_metrics
        regions_to_compute = list(set(recomputable_metrics.keys()))
    elif do_score_maps or plot_score_init_hour:
        metrics_to_compute = {r: metrics for r in regions}
        regions_to_compute = regions
    else:
        return run_id, stream, results

    # The climatology is aligned once and shared by both scoring calls.
    climatology = (
        get_climatology(reader, output_data.target, stream)
        if output_data is not None and needs_climatology(metrics_to_compute)
        else None
    )
    # Per-sample scores plus, for the score maps, the sample-aggregated scores.
    computed = calc_scores_per_stream(
        reader,
        stream,
        regions_to_compute,
        metrics_to_compute,
        output_data=output_data,
        climatology=climatology,
        compute_spatial=do_score_maps,
    )
    if plot_score_init_hour and output_data is not None:
        compute_scores_by_init_hour(
            reader, stream, computed, output_data, metrics_to_compute, climatology
        )
    if write_scores:
        metric_list_to_json(reader, stream, computed, regions_to_compute)

    # Drawn here, where the spatial scores of this (run, stream) are complete in a serial
    # run; a distributed run draws them on the master from the merged results.
    if do_score_maps:
        plot_score_maps(
            reader,
            stream,
            computed,
            global_plotting_options=global_plotting_opts,
            plot_score_options=plot_score_options,
        )

    for region, by_metric in computed.items():
        results.setdefault(region, {}).update(by_metric)
    return run_id, stream, results


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
    tasks = build_stream_tasks(cfg)
    results = [_process_stream(**task) for task in tasks]
    finalise(cfg, *accumulate_stream_results(results), mlflow_client)


def build_stream_tasks(cfg: dict) -> list[dict]:
    """Parse the config and build one ``_process_stream`` task per (run, stream).

    Readers are not built here; ``_process_stream`` builds its own.

    Parameters
    ----------
    cfg:
        Configuration input stored as dictionary; its metrics and plot options are
        normalised in place.

    Returns
    -------
    list[dict]
        The keyword arguments of ``_process_stream``, one dict per (run, stream).
    """
    with open_dict(cfg):
        cfg.evaluation.metrics = parse_metric_params(cfg.evaluation.metrics)
    runs = cfg.run_ids
    _logger.info(f"Detected {len(runs)} runs")
    private_paths = cfg.get("private_paths")
    metrics = cfg.evaluation.metrics

    # backward-compatibility with old way of specifying plotting options (bools) instead of lists:
    # TODO: remove this in a few weeks once all users moved to the new style.
    with open_dict(cfg):
        parse_plot_config(cfg)

    plot_score_options = get_plot_score_options(cfg.evaluation)

    global_plotting_opts = cfg.get("global_plotting_options", {})
    default_streams = cfg.get("default_streams", {})
    max_workers = cfg.get("max_workers")  # global hard cap for parallel workers

    tasks = []
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
    return tasks


def accumulate_stream_results(
    stream_results: list[tuple[str, str, dict[str, dict[str, MetricResult]]]],
) -> tuple[dict, dict]:
    """Nest the ``_process_stream`` results of all (run, stream) for the cross-run steps.

    Parameters
    ----------
    stream_results:
        ``(run_id, stream, results[region][metric] -> MetricResult)`` per (run, stream).

    Returns
    -------
    tuple[dict, dict]
        ``scores_dict[metric][region][stream][run_id] -> MetricResult.score`` and
        ``init_hour_scores`` with the same nesting for ``MetricResult.by_init_hour``.
    """
    scores_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # init_hour_scores[metric][region][stream][run_id] = xr.DataArray(init_hour, forecast_step, ...)
    init_hour_scores: dict = {}
    for _run_id, _stream, results in stream_results:
        for metric, regions_dict in results_to_dict(results, "score").items():
            for region, streams_dict in regions_dict.items():
                for stream_key, runs_dict in streams_dict.items():
                    scores_dict[metric][region][stream_key].update(runs_dict)

        merge(init_hour_scores, results_to_dict(results, "by_init_hour"))
    return scores_dict, init_hour_scores


def finalise(
    cfg: dict, scores_dict: dict, init_hour_scores: dict, mlflow_client: MlflowClient | None
) -> None:
    """Steps across runs, once all (run, stream) are done: MLflow upload and score plots.

    Parameters
    ----------
    cfg:
        Configuration input stored as dictionary.
    scores_dict, init_hour_scores:
        Output of ``accumulate_stream_results``.
    mlflow_client:
        Optional MLFlow client for uploading scores.
    """
    runs = cfg.run_ids
    private_paths = cfg.get("private_paths")
    summary_dir = Path(cfg.evaluation.get("summary_dir", _DEFAULT_PLOT_DIR))

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

    # verification plots across runs
    if scores_dict:
        plot_evaluation(cfg, scores_dict, init_hour_scores, summary_dir)


if __name__ == "__main__":
    evaluate()
