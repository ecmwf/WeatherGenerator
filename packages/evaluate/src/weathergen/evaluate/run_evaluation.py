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
from weathergen.common.io import ZarrIO
from omegaconf import OmegaConf

from weathergen.evaluate.utils import (
    calc_scores_per_stream,
    metric_list_to_json,
    plot_data,
    plot_summary,
    retrieve_metric_from_json,
    peek_tar_channels
)
from weathergen.utils.config import _REPO_ROOT, load_config, set_paths

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

    # configure logging
    logging.basicConfig(level=logging.INFO)

    # load configuration
    cfg = OmegaConf.load(args.config)

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

        # Allow for run ID specific directories
        # If results_base_dir is not provided, default paths are used
        results_base_dir = run.get("results_base_dir", None)

        if results_base_dir is None:
            cf_run = load_config(private_paths, run_id, run["epoch"])
            cf_run = set_paths(cf_run)
            results_base_dir = Path(cf_run["run_path"])

            logging.info(
                f"Results directory obtained automatically: {results_base_dir}"
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
                _ = plot_data(cfg, results_dir, runplot_dir, stream, stream_dict)

            if stream_dict.get("evaluation"):
                _logger.info(f"Retrieve or compute scores for {run_id} - {stream}...")

                for region in regions:

                    metrics_to_compute = []

                    for metric in metrics:
                        metric_data = retrieve_metric_from_json(
                            metrics_dir,
                            run_id,
                            stream,
                            region,
                            metric,
                            run.epoch,
                        )
                        # try:
                        checked, (channels, fsteps, samples) =  _check_metric(metric, stream, run, metric_data, stream_dict, results_dir)
                        if not checked:
                            metrics_to_compute.append(metric)
                        else:
                            #simply select the chosen eval channels, samples, fsteps here...
                            # for sample in samples:
                            scores_dict[metric][region][stream][run_id] = (
                                metric_data.sel(channel=list(channels), forecast_step=list(fsteps))
                            )
                                # scores_dict[metric][region][stream][run_id][sample] = (
                                #     metric_data.sel(channel=list(channels), forecast_step=list(fsteps))
                                # )
                        _logger.info(samples)
                        _logger.info(metric_data)
                        # except (FileNotFoundError, KeyError, ValueError):
                        #     metrics_to_compute.append(metric)
                    

                    if metrics_to_compute:
                        all_metrics, points_per_sample = calc_scores_per_stream(
                            cfg, results_dir, stream, region, metrics_to_compute
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

                        _logger.info('recomputed...')
                        _logger.info(all_metrics)

                    for metric in metrics_to_compute:
                        scores_dict[metric][region][stream][run_id] = all_metrics.sel(
                            {"metric": metric}
                        )

    # plot summary

    if scores_dict and cfg.summary_plots:
        _logger.info("Started creating summary plots..")
        plot_summary(cfg, scores_dict, summary_dir, print_summary=cfg.print_summary)

def _check_metric(metric: str, stream: str, run: str, metric_data: dict, stream_dict: dict, results_dir: Path):

    channels = stream_dict.get("channels")
    fsteps = stream_dict["evaluation"].get(
                    "forecast_step"
                )
    samples = stream_dict["evaluation"].get(
                    "sample"
                )
    _logger.info(samples)
    channels = None if channels == "all" or channels is None else set(channels)
    fsteps = None if fsteps == "all" or fsteps is None else set(fsteps)
    samples = None if samples == "all" or samples is None else set(samples)
    
    available_channels =  set(metric_data["channel"].values.ravel())
    available_fsteps = set(metric_data["forecast_step"].values.ravel())
    available_samples = set(metric_data.coords["sample"].values.ravel())

    fname_zarr = results_dir.joinpath(
        f"validation_epoch{run['epoch']:05d}_rank{run['rank']:04d}.zarr"
    )

    if not fname_zarr.exists() or not fname_zarr.is_dir():
        _logger.error(f"Zarr file {fname_zarr} does not exist or is not a directory.")
        raise FileNotFoundError(
            f"Zarr file {fname_zarr} does not exist or is not a directory."
        )

    with ZarrIO(fname_zarr) as zio:
        zio_fsteps =  zio.forecast_steps
        zio_samples = zio.samples
        zio_channels = peek_tar_channels(zio, stream, zio_fsteps[0])
        zio_fsteps = set([int(fstep) for fstep in zio_fsteps])
        zio_samples = set([int(sample) for sample in zio_samples])
        zio_channels = set(zio_channels)

    _logger.info(
    f"Requested: \n"
    f"fsteps: {fsteps} \n"
    f"samples: {samples} \n"
    f"channels: {channels}"
    )

    _logger.info(
    f"Available: \n"
    f"available_fsteps: {available_fsteps} \n"
    f"available_samples: {available_samples} \n"
    f"available_channels: {available_channels}"
    )

    _logger.info(
    f"Zarr: \n"
    f"zio_fsteps: {zio_fsteps} \n"
    f"zio_samples: {zio_samples} \n"
    f"zio_channels: {zio_channels}"
    )

    #available must equal zio, otherwise recompute
    if channels is None:
        channels = available_channels
        if zio_channels != available_channels:
            _logger.info('Requested all channels, but previous config was a strict subset of channels in Zarr. Recomputing.')
            return False, (channels, fsteps, samples)
    if fsteps is None:
        fsteps = available_fsteps
        if zio_fsteps != available_fsteps:
            _logger.info('Requested all fsteps, but previous config was a strict subset of fsteps in Zarr. Recomputing.')
            return False, (channels, fsteps, samples)
    if samples is None:
        samples = available_samples
        if zio_samples != available_samples:
            _logger.info('Requested all samples, but previous config was a strict subset of samples in Zarr. Recomputing.')
            return False, (channels, fsteps, samples)

    #config must be a subset of zio, otherwise error
    if not channels <= zio_channels:
        raise ValueError(f'Requested channels that do not exist in the Zarr file. Channels must be a subset of {zio_channels}')
    if not fsteps <= zio_fsteps:
        raise ValueError(f'Requested fsteps that do not exist in the Zarr file. Fsteps must be a subset of {zio_fsteps}')
    if not samples <= zio_samples:
        raise ValueError(f'Requested samples that do not exist in the Zarr file. Samples must be a subset of {zio_samples}')

    #config must be subset of available, otherwise recompute
    if not channels <= available_channels:
        _logger.info(f'Channel(s) {channels - available_channels} are not a availble from previous evaluation. Recomputing.')
        return False, (channels, fsteps, samples)
    if not fsteps <= available_fsteps:
        _logger.info(f'Fstep(s) {fsteps - available_fsteps} are not a availble from previous evaluation. Recomputing.')
        return False, (channels, fsteps, samples)
    if not samples <= available_samples:
        _logger.info(f'Sample(s) {samples - available_samples} are not a availble from previous evaluation. Recomputing.')
        return False, (channels, fsteps, samples) 

    #return True if all checks pass – i.e. no need to recompute
    _logger.info("All checks passed – No need to recompute...")
    return True, (channels, fsteps, samples)

if __name__ == "__main__":
    evaluate()
