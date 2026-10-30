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
from weathergen.evaluate.distributed.chunking import (
    build_plotting_chunk_tasks,
    build_scoring_chunk_tasks,
    parse_distributed_config,
    windows,
)
from weathergen.evaluate.distributed.merge import merge_bounds, merge_results
from weathergen.evaluate.io.csv_reader import CsvReader
from weathergen.evaluate.io.merge_reader import WeatherGenMergeReader
from weathergen.evaluate.io.wegen_reader import (
    WeatherGenJsonReader,
    WeatherGenReader,
    WeatherGenZarrReader,
)
from weathergen.evaluate.plotting.animations import build_animations
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
from weathergen.evaluate.utils.array_utils import colour_bounds, configured_bound
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
    chunk: bool = False,
) -> tuple[str, str, dict[str, dict[str, MetricResult]], dict[int, dict]]:
    """
    Worker function for a single stream of a single run, or for one chunk of it.

    Returns ``(run_id, stream, results, bounds)`` with ``results[region][metric] ->
    MetricResult``: the scores loaded from the JSON files, overridden by the ones computed
    here, empty when the stream is not scored; and, for a chunk that loaded data, the
    ``colour_bounds`` of that data per forecast step (empty otherwise), from which the
    chunked run sets the colour ranges of the whole (run, stream).
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
    chunk:
        The call is one chunk of samples × forecast steps of the (run, stream)
        (``distributed.chunking``): the ``metrics`` given are computed for all ``regions``
        without consulting the JSON score files, nothing is written to them, and the score
        maps and videos are left to the end of the whole (run, stream).
    """
    type_ = run.get("type", "zarr")
    reader = get_reader(type_, run, run_id, private_paths, regions, metrics)

    stream_dict = reader.get_stream(stream)
    if not stream_dict:
        _logger.info(f"Stream {stream} not found for run {run_id}. Skipping.")
        return run_id, stream, {}, {}

    # Data plots, score maps and init-hour scores need the data: readers that load it only
    reads_data = type_ in ("zarr", "anemoi-target")
    needs_plotting = stream_dict.get("plotting") and reads_data
    needs_scoring = stream_dict.get("evaluation", False)

    do_score_maps = plot_score_options.get("plot_score_maps", False) and reads_data
    plot_score_init_hour = plot_score_options.get("plot_score_init_hour", False) and reads_data

    # --- Determine scoring state before loading any data ---
    results: dict[str, dict[str, MetricResult]] = {}
    metrics_to_compute: dict = {}
    recomputable_metrics: dict = {}
    if needs_scoring and chunk:
        metrics_to_compute = {r: metrics for r in regions}
    elif needs_scoring:
        results, metrics_to_compute, recomputable_metrics = cached_scores_and_metrics_to_compute(
            reader, stream, regions, metrics, plot_score_options, reads_data
        )

    # --- Load data only when necessary ---
    output_data = None
    if metrics_to_compute and reads_data:
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

    # the colour bounds of this chunk's data, per forecast step: the chunked run reduces
    # them over the chunks and the plotting forecast steps (_run_stream_in_chunks)
    bounds: dict[int, dict] = {}
    if chunk and output_data is not None:
        for fstep, target in output_data.target.items():
            prediction = output_data.prediction[fstep]
            bounds[fstep] = colour_bounds(
                {fstep: target}, {fstep: prediction}, [str(c) for c in target.channel.values]
            )

    # Field plots: pass pre-loaded data if available, otherwise plot_data loads its own subset
    if needs_plotting:
        plot_stream_data(reader, stream, global_plotting_opts, output_data=output_data)

    # Scoring per stream
    computed: dict[str, dict[str, MetricResult]] = {}
    if metrics_to_compute:
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
            list(metrics_to_compute),
            metrics_to_compute,
            output_data=output_data,
            climatology=climatology,
            compute_spatial=do_score_maps,
        )
        if plot_score_init_hour and output_data is not None:
            compute_scores_by_init_hour(
                reader, stream, computed, output_data, metrics_to_compute, climatology
            )
        write_missing_scores(reader, stream, computed, recomputable_metrics)

    # Score maps and videos need the whole (run, stream): a serial call is the whole, so it
    # draws them here; a chunk is not, so _run_stream_in_chunks draws them after all chunks.
    if not chunk:
        if do_score_maps and computed:
            plot_score_maps(reader, stream, computed, global_plotting_options=global_plotting_opts)
        if reads_data:
            build_animations(reader, stream, global_plotting_opts, plot_score_options)

    for region, by_metric in computed.items():
        results.setdefault(region, {}).update(by_metric)
    return run_id, stream, results, bounds


def cached_scores_and_metrics_to_compute(
    reader,
    stream: str,
    regions: list[str],
    metrics: dict[str, object],
    plot_score_options: dict[str, object],
    reads_data: bool,
) -> tuple[dict[str, dict[str, MetricResult]], dict, dict]:
    """What is already in the JSON score files of a scored (run, stream), and what to compute.

    Parameters
    ----------
    reader:
        Reader of the run.
    stream:
        Stream name.
    regions, metrics:
        Regions and ``{metric: parameters}`` requested.
    plot_score_options:
        ``get_plot_score_options(...)``: score maps and init-hour scores need the data, so
        they force a recomputation of every metric.
    reads_data:
        Whether the reader loads data (zarr readers); the two options above are warned
        about and ignored otherwise.

    Returns
    -------
    tuple[dict, dict, dict]
        ``cached[region][metric] -> MetricResult`` (the scores found in the files),
        ``metrics_to_compute[region][metric] -> parameters`` (the missing ones, or all of
        them when the score maps / init-hour scores are requested) and
        ``recomputable_metrics[region][metric] -> parameters`` (the missing ones only: what
        has to be written back).
    """
    for option, name in (("plot_score_maps", "score_map"), ("plot_score_init_hour", "init_hour")):
        if plot_score_options.get(option, False) and not reads_data:
            _logger.warning(
                f"RUN {reader.run_id} - {stream}: '{name}' not supported for this run type."
            )
    needs_data = reads_data and (
        plot_score_options.get("plot_score_maps", False)
        or plot_score_options.get("plot_score_init_hour", False)
    )
    stream_loaded_scores, recomputable_metrics = reader.load_scores(stream, regions, metrics)

    # cached scores as MetricResults; the ones computed later override them
    cached: dict[str, dict[str, MetricResult]] = {}
    for metric, by_region in stream_loaded_scores.items():
        for region, by_stream in by_region.items():
            for stream_key, by_run in by_stream.items():
                for run_key, score in by_run.items():
                    cached.setdefault(region, {})[metric] = MetricResult(
                        metric, region, stream_key, run_key, score
                    )

    if recomputable_metrics:
        metrics_to_compute = recomputable_metrics
    elif needs_data:
        metrics_to_compute = {r: metrics for r in regions}
    else:
        metrics_to_compute = {}
    return cached, metrics_to_compute, recomputable_metrics


def write_missing_scores(
    reader,
    stream: str,
    computed: dict[str, dict[str, MetricResult]],
    recomputable_metrics: dict,
) -> None:
    """Write to the JSON score files the computed scores that were missing from them.

    The others were recomputed for the score maps / init-hour scores and are the cached
    ones again, so they are not written (and not reported as replaced).
    """
    missing = {
        region: {m: r for m, r in by_metric.items() if m in recomputable_metrics.get(region, {})}
        for region, by_metric in computed.items()
    }
    missing = {region: by_metric for region, by_metric in missing.items() if by_metric}
    if missing:
        metric_list_to_json(reader, stream, missing, list(missing))


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
    if parse_distributed_config(cfg)[0]:
        run_chunked(cfg, mlflow_client)
        return
    tasks = build_stream_tasks(cfg)
    results = [_process_stream(**task) for task in tasks]
    finalise(cfg, *accumulate_stream_results(results), mlflow_client)


def run_chunked(cfg: dict, mlflow_client: MlflowClient | None) -> None:
    """``evaluate_from_config`` with every (run, stream) cut into chunks (``distributed``).

    The chunk tasks run one after the other in this process; an MPI run drives the same
    tasks from ``distributed.runner``.
    """
    _, sample_size, fstep_size = parse_distributed_config(cfg)
    stream_results = [
        _run_stream_in_chunks(task, sample_size, fstep_size) for task in build_stream_tasks(cfg)
    ]
    finalise(cfg, *accumulate_stream_results(stream_results), mlflow_client)


def _run_stream_in_chunks(
    task: dict, sample_size: int | None, fstep_size: int | None
) -> tuple[str, str, dict[str, dict[str, MetricResult]], dict]:
    """``_process_stream`` of one (run, stream), as chunk tasks.

    Scoring chunk tasks first (their results merged, the missing scores written), then the
    plotting chunk tasks with the colour ranges of the whole (run, stream), then the score
    maps and videos.  A run whose reader does not load data has nothing to chunk and goes
    through ``_process_stream`` as it is.

    The colour ranges are those a serial run uses: the bounds of the evaluation samples over
    the plotting forecast steps when the stream was scored here (returned by the scoring
    chunk tasks), else the bounds of the plotting subset, loaded window by window; a bound
    the config fixes is never computed.
    """
    run_id, run, stream = task["run_id"], task["run"], task["stream"]
    type_ = run.get("type", "zarr")
    if type_ not in ("zarr", "anemoi-target"):
        return _process_stream(**task)
    reader = get_reader(type_, run, run_id, task["private_paths"], task["regions"], task["metrics"])
    stream_cfg = reader.get_stream(stream)
    if not stream_cfg:
        return _process_stream(**task)
    global_plotting_opts = task["global_plotting_opts"]
    plot_score_options = dict(task["plot_score_options"])

    results: dict[str, dict[str, MetricResult]] = {}
    computed: dict[str, dict[str, MetricResult]] = {}
    chunk_bounds: list[dict[int, dict]] = []  # per scoring chunk, per forecast step
    if stream_cfg.get("evaluation"):
        available = reader.check_availability(stream, mode="evaluation")
        if len(windows(available.samples, sample_size)) > 1:
            # means over the samples: not merged across sample chunks (merge_results)
            for option, name in (
                ("plot_score_maps", "score maps"),
                ("plot_score_init_hour", "init-hour scores"),
            ):
                if plot_score_options.get(option, False):
                    _logger.warning(
                        f"RUN {run_id} - {stream}: {name} are skipped when the samples are chunked."
                    )
                    plot_score_options[option] = False
        task = task | {"plot_score_options": plot_score_options}
        results, metrics_to_compute, recomputable_metrics = cached_scores_and_metrics_to_compute(
            reader, stream, task["regions"], task["metrics"], plot_score_options, True
        )
        if metrics_to_compute:
            tasks, responsible = build_scoring_chunk_tasks(
                task,
                available.samples,
                available.fsteps,
                metrics_to_compute,
                sample_size,
                fstep_size,
            )
            outputs = [
                _process_stream(**_log_chunk(t, i, len(tasks), responsible[i]))
                for i, t in enumerate(tasks)
            ]
            computed = merge_results([o[2] for o in outputs], fsteps=responsible)
            chunk_bounds = [o[3] for o in outputs]
            write_missing_scores(reader, stream, computed, recomputable_metrics)
            for region, by_metric in computed.items():
                results.setdefault(region, {}).update(by_metric)

    if stream_cfg.get("plotting"):
        available = reader.check_availability(stream, mode="plotting")
        bounds = merge_bounds(
            [by_fstep[f] for by_fstep in chunk_bounds for f in available.fsteps if f in by_fstep]
        )
        if not bounds and _bounds_missing(task, stream, available):
            # not scored here: the bounds of the plotting subset, one window at a time
            for fsteps in windows(available.fsteps, fstep_size):
                for samples in windows(available.samples, sample_size):
                    data = reader.get_data(
                        stream,
                        fsteps=fsteps,
                        samples=samples,
                        channels=available.channels,
                        ensemble=available.ensemble,
                    )
                    chunk_bounds.append(
                        colour_bounds(data.target, data.prediction, available.channels)
                    )
            bounds = merge_bounds(chunk_bounds)
        tasks = build_plotting_chunk_tasks(
            task,
            available.samples,
            available.fsteps,
            available.channels,
            sample_size,
            fstep_size,
            bounds,
        )
        for i, t in enumerate(tasks):
            _process_stream(**_log_chunk(t, i, len(tasks)))

    if plot_score_options.get("plot_score_maps", False) and computed:
        plot_score_maps(reader, stream, computed, global_plotting_options=global_plotting_opts)
    build_animations(reader, stream, global_plotting_opts, plot_score_options)
    return run_id, stream, results, {}


def _bounds_missing(task: dict, stream: str, available) -> bool:
    """Whether the config leaves a colour bound of the stream's plotted channels to the data."""
    stream_opts = task["global_plotting_opts"].get(stream) or {}
    # the ens dimension is dropped on loading only for an ensemble reduced to [mean] / [std]
    has_ens = list(available.ensemble) not in (["mean"], ["std"])
    keys = ("vmin", "vmax", "bias_vmax") + (("std_vmin", "std_vmax") if has_ens else ())
    return any(
        configured_bound(stream_opts, ch, k) is None for ch in available.channels for k in keys
    )


def _log_chunk(task: dict, index: int, count: int, responsible: list[int] | None = None) -> dict:
    """Log which chunk of the (run, stream) *task* is, and return it.

    *responsible* are the forecast steps a scoring chunk answers for; the steps it is scored
    on beyond them are the overlap that ``froct`` / ``troct`` need.
    """
    mode = "evaluation" if "evaluation" in task["run"].streams[task["stream"]] else "plotting"
    window = task["run"].streams[task["stream"]][mode]
    fsteps = list(window.forecast_step)
    extra = [f for f in fsteps if responsible is not None and f not in responsible]
    _logger.info(
        f"RUN {task['run_id']} - {task['stream']}: {mode} chunk {index + 1}/{count}: "
        f"samples {list(window.sample)}, forecast steps {[f for f in fsteps if f not in extra]}"
        + (f" (+ step {extra[0]} for froct/troct)" if extra else "")
    )
    return task


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
    stream_results: list[tuple[str, str, dict[str, dict[str, MetricResult]], dict]],
) -> tuple[dict, dict]:
    """Nest the ``_process_stream`` results of all (run, stream) for the cross-run steps.

    Parameters
    ----------
    stream_results:
        ``(run_id, stream, results[region][metric] -> MetricResult, bounds)`` per
        (run, stream), as ``_process_stream`` returns them.

    Returns
    -------
    tuple[dict, dict]
        ``scores_dict[metric][region][stream][run_id] -> MetricResult.score`` and
        ``init_hour_scores`` with the same nesting for ``MetricResult.by_init_hour``.
    """
    scores_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # init_hour_scores[metric][region][stream][run_id] = xr.DataArray(init_hour, forecast_step, ...)
    init_hour_scores: dict = {}
    for _run_id, _stream, results, _bounds in stream_results:
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
