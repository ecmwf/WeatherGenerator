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
import multiprocessing as mp
import sys
from collections import defaultdict
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path

# Third-party
import mlflow
from mlflow.client import MlflowClient
from omegaconf import DictConfig, OmegaConf

# Local application / package
from weathergen.common.config import _REPO_ROOT
from weathergen.common.logger import init_loggers
from weathergen.common.platform_env import get_platform_env
from weathergen.evaluate.io.csv_reader import CsvReader
from weathergen.evaluate.io.wegen_reader import WeatherGenReader
from weathergen.evaluate.plotting.plot_utils import collect_channels
from weathergen.evaluate.utils.utils import (
    calc_scores_per_stream,
    plot_data,
    plot_summary,
    triple_nested_dict,
)
from weathergen.metrics.mlflow_utils import (
    MlFlowUpload,
    get_or_create_mlflow_parent_run,
    log_scores,
    setup_mlflow,
)

_DEFAULT_PLOT_DIR = _REPO_ROOT / "plots"

_logger = logging.getLogger(__name__)
_platform_env = get_platform_env()


def setup_main_logger(log_file: str | None, log_queue: mp.Queue) -> QueueListener:
    """Set up main process logger with QueueListener

    Parameters
    ----------
        log_file: str
            Name of
    """

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s: %(message)s")
    )

    handlers: list[logging.Handler] = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s: %(message)s")
        )
        handlers.append(file_handler)

    listener = QueueListener(log_queue, *handlers)
    listener.start()
    return listener


def setup_worker_logger(log_queue: mp.Queue) -> logging.Logger:
    """"""
    qh = QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(qh)
    return logger


#################################################################


def evaluate() -> None:
    """entry point for evaluation script."""
    # By default, arguments from the command line are read.
    log_queue: mp.Queue = mp.Queue()
    listener = setup_main_logger("evaluation.log", log_queue)
    try:
        evaluate_from_args(sys.argv[1:], log_queue)
    finally:
        listener.stop()
        log_queue.close()
        log_queue.join_thread()


def evaluate_from_args(argl: list[str], log_queue: mp.Queue) -> None:
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
    evaluate_from_config(cf, mlflow_client, log_queue)


def _process_stream_wrapper(
    args: dict[str, object],
) -> tuple[str, str, dict[str, dict[str, dict[str, float]]]]:
    return _process_stream(**args)


def _process_stream(
    run_id: str,
    run: dict,
    stream: str,
    private_paths: dict[str, str],
    global_plotting_opts: dict[str, object],
    regions: list[str],
    metrics: list[str],
    plot_score_maps: bool,
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
        List of metrics to be processed.
    plot_score_maps:
        Bool to define if the score maps need to be plotted or not.

    """
    try:
        type_ = run.get("type", "zarr")
        reader = (
            WeatherGenReader(run, run_id, private_paths)
            if type_ == "zarr"
            else CsvReader(run, run_id, private_paths)
        )

        stream_dict = reader.get_stream(stream)
        if not stream_dict:
            return run_id, stream, {}

        # Parallel plotting
        if stream_dict.get("plotting"):
            plot_data(reader, stream, global_plotting_opts)

        # Scoring per stream
        if not stream_dict.get("evaluation"):
            return run_id, stream, {}

        stream_scores = calc_scores_per_stream(reader, stream, regions, metrics, plot_score_maps)

        return run_id, stream, stream_scores

    except Exception as e:
        _logger.error(f"Error processing {run_id} - {stream}: {e}")
        return run_id, stream, {}


# Weird typing error from python: mp.Queue is seen as a method with a "|" operator => this fai
def evaluate_from_config(
    cfg: dict, mlflow_client: MlflowClient | None, log_queue: "mp.Queue | None"
) -> None:
    """
    Main function that controls evaluation plotting and scoring.
    Parameters
    ----------
    cfg:
        Configuration input stored as dictionary.
    """
    runs = cfg.run_ids
    _logger.info(f"Detected {len(runs)} runs")
    private_paths = cfg.get("private_paths")
    summary_dir = Path(cfg.evaluation.get("summary_dir", _DEFAULT_PLOT_DIR))
    metrics = cfg.evaluation.metrics
    regions = cfg.evaluation.get("regions", ["global"])
    plot_score_maps = cfg.evaluation.get("plot_score_maps", False)
    global_plotting_opts = cfg.get("global_plotting_options", {})
    use_parallel = cfg.evaluation.get("num_processes", 0)
    if use_parallel == "auto":
        num_processes = mp.cpu_count()
    elif isinstance(use_parallel, int):
        if use_parallel > 0:
            num_processes = min(use_parallel, mp.cpu_count())
        else:
            # Using the main process only
            num_processes = 0
    else:
        raise ValueError("parallel option must be 'auto' or an non-negative integer")

    if num_processes > 1:
        _logger.info("Using %d processes for evaluation", num_processes)
    else:
        _logger.info("Using main process for evaluation")

    scores_dict = defaultdict(triple_nested_dict)  # metric -> region -> stream -> run
    tasks = []

    # Build tasks per stream
    for run_id, run in runs.items():
        type_ = run.get("type", "zarr")
        reader = (
            WeatherGenReader(run, run_id, private_paths)
            if type_ == "zarr"
            else CsvReader(run, run_id, private_paths)
        )
        for stream in reader.streams:
            tasks.append(
                {
                    "run_id": run_id,
                    "run": run,
                    "stream": stream,
                    "private_paths": private_paths,
                    "global_plotting_opts": global_plotting_opts,
                    "regions": regions,
                    "metrics": metrics,
                    "plot_score_maps": plot_score_maps,
                }
            )

    scores_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    if num_processes == 0:
        if log_queue is not None:
            setup_worker_logger(log_queue)
        results = [_process_stream(**task) for task in tasks]
    else:
        with mp.Pool(
            processes=num_processes,
            initializer=setup_worker_logger,
            initargs=(log_queue,),
        ) as pool:
            results = pool.map(
                _process_stream_wrapper,
                tasks,
            )

    for _, stream, stream_scores in results:
        for metric, regions_dict in stream_scores.items():
            for region, streams_dict in regions_dict.items():
                for stream, runs_dict in streams_dict.items():
                    scores_dict[metric][region][stream].update(runs_dict)

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
    if scores_dict and cfg.evaluation.get("summary_plots", True):
        _logger.info("Started creating summary plots...")
        plot_summary(cfg, scores_dict, summary_dir)


if __name__ == "__main__":
    evaluate()
