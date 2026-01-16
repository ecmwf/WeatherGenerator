# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Integration test for the Weather Generator with multiple streams and observations.
This test must run on a GPU machine.
It performs training and inference with multiple data sources including gridded and obs data.

Command:
uv run pytest ./integration_tests/small_multi_stream_test.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import omegaconf

from weathergen.evaluate.io.wegen_reader import (
    WeatherGenJSONReader,
)
from weathergen.evaluate.run_evaluation import evaluate_from_config
from weathergen.run_train import inference_from_args
from weathergen.utils.metrics import get_train_metrics_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    logger.addHandler(h)

logger.propagate = False

# TODO: define WEATHERGEN_HOME properly to avoid partial paths

streams = ["ERA5", "SurfaceCombined", "NPPATMS"]

############## INFERENCE #################


def infer_multi_stream(run_id):
    """Run inference for multi-stream model."""
    logger.info("run multi-stream inference")
    new_run_id = run_id + "_inf"  # TODO: better naming
    inference_from_args(
        [
            "-start",
            "2021-10-10",
            "-end",
            "2022-10-11",
            "--samples",
            "10",
            "--options",
            "forecast_offset=0",
            "zarr_store=zip",
        ]
        + ["--from_run_id", run_id, "--run_id", new_run_id, "--streams_output"]
        + streams
        + [
            "--config",
            "./config/evaluate/latent_space_eval_config.yaml",
        ]
    )
    return new_run_id


############## EVALUATION #################


def get_evaluation_config(run_id, verbose=False):
    """Create evaluation configuration for multiple streams."""
    cfg = omegaconf.OmegaConf.create(
        {
            "global_plotting_options": {
                "image_format": "png",
                "dpi_val": 300,
            },
            "evaluation": {
                "regions": ["global"],
                "metrics": ["rmse", "froct"],
                "summary_plots": True,
                "summary_dir": f"./results/{run_id}/plots/summary/",
                "print_summary": False,
                "verbose": verbose,
            },
            "run_ids": {
                run_id: {
                    "streams": {
                        "ERA5": {
                            "channels": ["q_850", "z_500", "2t", "10u", "10v", "msl"],
                            "evaluation": {"forecast_steps": "all", "sample": "all"},
                            "plotting": {
                                "sample": [0, 1],
                                "forecast_step": [0],
                                "plot_maps": True,
                                "plot_histograms": True,
                                "plot_animations": False,
                            },
                        },
                        "SurfaceCombined": {
                            "channels": ["obsvalue_t2m_0"],
                            "evaluation": {"forecast_steps": "all", "sample": "all"},
                            "plotting": {
                                "sample": [0, 1],
                                "forecast_step": [0],
                                "plot_maps": True,
                                "plot_histograms": True,
                                "plot_animations": False,
                            },
                        },
                        "NPPATMS": {
                            "channels": ["obsvalue_rawbt_1"],
                            "evaluation": {"forecast_steps": "all", "sample": "all"},
                            "plotting": {
                                "sample": [0, 1],
                                "forecast_step": [0],
                                "plot_maps": True,
                                "plot_histograms": True,
                                "plot_animations": False,
                            },
                        },
                    },
                    "label": "Multi-Stream Test",
                    "mini_epoch": 0,
                    "rank": 0,
                }
            },
        }
    )
    return cfg


def evaluate_multi_stream_results(run_id, verbose=False):
    """Run evaluation for multiple streams."""

    logger.info("run multi-stream evaluation")
    cfg = get_evaluation_config(run_id, verbose=verbose)
    try:
        evaluate_from_config(cfg, None, None)
    except FileNotFoundError as e:
        logger.error(f"Error during evaluation: {e}")


############## PRINT FUNCTIONS #################


def print_losses(run_id, stage="val"):
    """Print validation losses for specified streams."""
    logger.info(f"{stage.capitalize()} Losses for run_id: {run_id}")
    metrics = load_metrics(run_id)

    losses = {}

    for stream_name in streams:
        loss = next(
            (
                metric.get(f"LossPhysical.{stream_name}.mse.avg")
                for metric in reversed(metrics)
                if metric.get("stage") == stage
            ),
            None,
        )

        losses[stream_name] = loss
    stage_label = "Train" if stage == "train" else "Validation"
    # TODO: understand why logger is not working
    logger.info(
        f"{stage_label} losses – " + ", ".join(f"{k}: {v:.4f}" for k, v in losses.items()) + "\n"
    )


def print_evaluation_results(run_id, verbose=False):
    """Print evaluation results for specified streams."""

    eval_cfg = get_evaluation_config(run_id, verbose=verbose)
    scores = load_scores(eval_cfg, run_id)

    metrics = list(eval_cfg.evaluation.get("metrics"))
    regions = list(eval_cfg.evaluation.get("regions"))
    for stream_name in streams:
        stream_scores = scores[stream_name]

        for metric in metrics:
            logger.info("------------------------------------------")
            for region in regions:
                da = stream_scores[metric][region][stream_name][run_id]
                logger.info(f"\nEvaluation scores for {region} {stream_name} {metric}:")

                mean_da = da.mean(dim=["sample", "forecast_step", "ens"])
                logger.info(mean_da.to_dataframe(name=f"{metric} {region} {stream_name}"))


############## HELPERS #################


def load_metrics(run_id):
    """Helper function to load metrics"""

    file_path = get_train_metrics_path(base_path=Path("./results"), run_id=run_id)

    if not file_path.is_file():
        raise FileNotFoundError(f"Metrics file not found for run_id: {run_id}")
    with open(file_path) as f:
        json_str = f.readlines()
    return json.loads("[" + "".join([s.replace("\n", ",") for s in json_str])[:-1] + "]")


def load_scores(eval_cfg, run_id):
    """Helper function to load metrics"""

    run_cfg = eval_cfg.run_ids[run_id]

    metrics = list(eval_cfg.evaluation.get("metrics"))
    regions = list(eval_cfg.evaluation.get("regions"))

    reader = WeatherGenJSONReader(run_cfg, run_id, None, regions, metrics)

    scores = {}

    for stream_name in streams:
        stream_loaded_scores, _ = reader.load_scores(
            stream_name,
            regions,
            metrics,
        )

        scores[stream_name] = stream_loaded_scores

    return scores


############## MAIN #################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-stream latent space evaluation")
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run identifier for the model to evaluate"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output", default=False
    )
    args = parser.parse_args()

    run_id = args.run_id
    verbose = args.verbose

    infer_run_id = infer_multi_stream(run_id)

    # Evaluate results
    evaluate_multi_stream_results(infer_run_id, verbose=verbose)
    logger.info("\n\nFinal Results Summary: \n")
    logger.info("TRAINING & INFERENCE LOSSES: \n")
    print_losses(run_id, stage="train")
    print_losses(infer_run_id, stage="val")
    logger.info("EVALUATION: \n")
    print_evaluation_results(infer_run_id, verbose=verbose)
