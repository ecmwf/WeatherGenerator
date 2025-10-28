import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from dask.distributed import Client
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
    # configure logging. Put the date in unix timestamp format for easy sorting
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(
        level=logging.INFO,
        format="%(relativeCreated)6d ms - %(name)s - %(levelname)s - %(message)s",
    )

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

    _logger.info("Setting up Dask client for evaluation...")
    client: Client | None = None
    if cfg.get("dask", None) is not None:
        num_workers = cfg.get("dask.num_workers", None)
        threads_per_worker = cfg.get("dask.threads_per_worker", None)
        client = Client(
            processes=True, n_workers=num_workers, threads_per_worker=threads_per_worker
        )
        _logger.info(f"Dask client info: {client}")
    else:
        client = Client(processes=True, n_workers=1, threads_per_worker=1)
        _logger.info("No dask configuration found in config file, running with a single worker.")

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
                            reader, stream, region, metrics_to_compute, client=client
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
