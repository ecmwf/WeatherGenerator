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

import mlflow
from mlflow.client import MlflowClient
from omegaconf import OmegaConf
from logging.handlers import QueueHandler, QueueListener
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from weathergen.common.config import _REPO_ROOT
from weathergen.common.logger import init_loggers
from weathergen.common.platform_env import get_platform_env
from weathergen.evaluate.io_reader import CsvReader, WeatherGenReader
from weathergen.evaluate.plot_utils import collect_channels
from weathergen.evaluate.utils import (
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
LOG_QUEUE: mp.Queue()  = mp.Queue()
_logger = logging.getLogger(__name__)
_platform_env = get_platform_env()

def setup_main_logger(log_file: str | None =None):
    """Set up main process logger with QueueListener
    
    Parameters
    ----------
        log_file: str
            Name of
    """
    global LOG_QUEUE
    LOG_QUEUE = mp.Queue()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(processName)s] %(levelname)s: %(message)s'))

    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(processName)s] %(levelname)s: %(message)s'))
        handlers.append(file_handler)

    listener = QueueListener(LOG_QUEUE, *handlers)
    listener.start()
    return listener


def setup_worker_logger():
    """Worker logger uses global LOG_QUEUE"""
    global LOG_QUEUE
    qh = QueueHandler(LOG_QUEUE)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # remove other handlers
    logger.addHandler(qh)
    return logger

#################################################################

def evaluate() -> None:
    # By default, arguments from the command line are read.
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
        mlflow_client = setup_mlflow(private_cf)
        _logger.info(f"MLFlow client set up: {mlflow_client}")

    evaluate_from_config(OmegaConf.load(config), mlflow_client)

def run_parallel(
    tasks: iter,
    fn: callable,
    parallel: bool = True
) -> object:
    """
    Execute a function over a list of argument-tuples either in parallel or serially.

    Parameters
    ----------
    tasks : iterable
        An iterable of argument-tuples. Each tuple is expanded into fn(*args).
    fn : callable
        The function to execute for each task. Must be top-level if parallel=True.
    parallel : bool, optional
        If True, execution uses ProcessPoolExecutor for parallelism.
        If False, tasks run serially in the current process.

    Returns
    -------
        A generator yielding fn(*args) results for each task.

    Notes
    -----
    - This helper abstracts out parallel vs non-parallel execution.
    - When parallel=True, fn and all arguments must be picklable.
    - Tasks are yielded as soon as they complete (not in input order).
    """
    if not parallel:
        # Serial execution
        for t in tasks:
            yield fn(*t)
        return

    # Parallel execution
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(fn, *t): t for t in tasks}
        for fut in as_completed(futures):
            yield fut.result()


def _process_stream(run_id: str, run: dict, stream: str, private_paths: list[str], global_plotting_opts: dict[str], regions: list[str], metrics: list[str], plot_score_maps: bool):
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
        reader = WeatherGenReader(run, run_id, private_paths) if type_ == "zarr" else CsvReader(run, run_id, private_paths)

        stream_dict = reader.get_stream(stream)
        if not stream_dict:
            return run_id, stream, {}

        # Parallel plotting 
        if stream_dict.get("plotting"):
            plot_data(reader, stream, global_plotting_opts)

        # Scoring per stream
        if not stream_dict.get("evaluation"):
            return run_id, stream, {}

        stream_scores = calc_scores_per_stream(
            reader, stream, regions, metrics, plot_score_maps
        )

        return run_id, stream, stream_scores

    except Exception as e:
        _logger.error(f"Error processing {run_id} - {stream}: {e}")
        return run_id, stream, {}

def evaluate_from_config(cfg: dict, mlflow_client=None):
    """
    Main function that controls evaluation plotting and scoring. 
    Parameters
    ----------
    cfg:
        Configuration input stored as dictionary. 
    """
    runs = cfg.run_ids
    _logger.info(f"Detected {len(runs)} runs")
    private_paths = cfg.get("private_paths", None)
    summary_dir = Path(cfg.evaluation.get("summary_dir", _DEFAULT_PLOT_DIR))
    metrics = cfg.evaluation.metrics
    regions = cfg.evaluation.get("regions", ["global"])
    plot_score_maps = cfg.evaluation.get("plot_score_maps", False)
    global_plotting_opts = cfg.get("global_plotting_options", {})
    use_parallel = cfg.evaluation.get("parallel", True)

    listener = setup_main_logger("evaluation.log")
    _logger.info("Started main logging listener")

    scores_dict = defaultdict(triple_nested_dict)  # metric -> region -> stream -> run
    tasks = []

    # Build tasks per stream
    for run_id, run in runs.items():
        type_ = run.get("type", "zarr")
        reader = WeatherGenReader(run, run_id, private_paths) if type_ == "zarr" else CsvReader(run, run_id, private_paths)
        for stream in reader.streams:
            tasks.append((run_id, run, stream, private_paths, global_plotting_opts, regions, metrics, plot_score_maps))

    scores_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for run_id, stream, stream_scores in run_parallel(tasks, _process_stream, parallel=use_parallel):
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

    listener.stop()

if __name__ == "__main__":
    listener = setup_main_logger("evaluation.log")
    evaluate()
    listener.stop()
