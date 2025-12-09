"""
Small test for the Weather Generator.
This test must run on a GPU machine.
It performs a training and inference of the Weather Generator model.

Command:
uv run pytest  ./integration_tests/small1.py
"""

import json
import logging
import os
import shutil
from pathlib import Path
import omegaconf
import pytest

from weathergen.evaluate.run_evaluation import evaluate_from_config
from weathergen.run_train import inference_from_args, train_with_args
from weathergen.utils.metrics import get_train_metrics_path

logger = logging.getLogger(__name__)

# Read from git the current commit hash and take the first 5 characters:
try:
    from git import Repo

    repo = Repo(search_parent_directories=False)
    commit_hash = repo.head.object.hexsha[:5]
    logger.info(f"Current commit hash: {commit_hash}")
except Exception as e:
    commit_hash = "unknown"
    logger.warning(f"Could not get commit hash: {e}")

WEATHERGEN_HOME = Path(__file__).parent.parent


@pytest.fixture()
def setup(test_run_id):
    logger.info(f"setup fixture with {test_run_id}")
    shutil.rmtree(WEATHERGEN_HOME / "results" / test_run_id, ignore_errors=True)
    shutil.rmtree(WEATHERGEN_HOME / "models" / test_run_id, ignore_errors=True)
    yield
    logger.info("end fixture")


@pytest.mark.parametrize("test_run_id", ["test_small1_" + commit_hash])
def test_train(setup, test_run_id):
    logger.info(f"test_train with run_id {test_run_id} {WEATHERGEN_HOME}")

    train_with_args(
        f"--config={WEATHERGEN_HOME}/integration_tests/small1.yaml".split()
        + [
            "--run_id",
            test_run_id,
        ],
        f"{WEATHERGEN_HOME}/config/streams/streams_test/",
    )

    infer_with_missing(test_run_id)
    evaluate_results(test_run_id)
    assert_missing_metrics_file(test_run_id)
    assert_train_loss_below_threshold(test_run_id)
    assert_val_loss_below_threshold(test_run_id)
    logger.info("end test_train")


def infer(run_id):
    logger.info("run inference")
    inference_from_args(
        ["-start", "2022-10-10", "-end", "2022-10-11", "--samples", "10", "--mini_epoch", "0"]
        + [
            "--from_run_id",
            run_id,
            "--run_id",
            run_id,
            "--config",
            f"{WEATHERGEN_HOME}/integration_tests/small1.yaml",
        ]
    )


def infer_with_missing(run_id):
    logger.info("run inference")
    inference_from_args(
        ["-start", "2022-10-10", "-end", "2022-10-11", "--samples", "10", "--mini_epoch", "0"]
        + [
            "--from_run_id",
            run_id,
            "--run_id",
            run_id,
            "--config",
            f"{WEATHERGEN_HOME}/integration_tests/small1.yaml",
        ]
    )


def evaluate_results(run_id):
    logger.info("run evaluation")
    cfg = omegaconf.OmegaConf.create(
        {
            "global_plotting_options": {
                "image_format": "png",
                "dpi_val": 300,
            },
            "evaluation": {
                "metrics": ["rmse", "l1", "mse"],
                "verbose": True,
                "summary_plots": True,
                "summary_dir": "./plots/",
                "print_summary": True,
            },
            "run_ids": {
                run_id: {  # would be nice if this could be done with option
                    "streams": {
                        "ERA5": {
                            "results_base_dir": "./results/",
                            "channels": ["t_850"],  # "all" indicator would be nice
                            "evaluation": {"forecast_steps": "all", "sample": "all"},
                            "plotting": {
                                "sample": [0, 1],
                                "forecast_step": [0],
                                "plot_maps": True,
                                "plot_histograms": True,
                                "plot_animations": True,
                            },
                        }
                    },
                    "label": "MTM ERA5",
                    "mini_epoch": 0,
                    "rank": 0,
                }
            },
        }
    )
    # Not passing the mlflow client for tests.
    evaluate_from_config(cfg, None, None )


def load_metrics(run_id):
    """Helper function to load metrics"""
    file_path = get_train_metrics_path(base_path=WEATHERGEN_HOME / "results", run_id=run_id)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found for run_id: {run_id}")
    with open(file_path) as f:
        json_str = f.readlines()
    return json.loads("[" + r"".join([s.replace("\n", ",") for s in json_str])[:-1] + "]")


def assert_missing_metrics_file(run_id):
    """Test that a missing metrics file raises FileNotFoundError."""
    file_path = get_train_metrics_path(base_path=WEATHERGEN_HOME / "results", run_id=run_id)
    assert os.path.exists(file_path), f"Metrics file does not exist for run_id: {run_id}"
    metrics = load_metrics(run_id)
    logger.info(f"Loaded metrics for run_id: {run_id}: {metrics}")
    assert metrics is not None, f"Failed to load metrics for run_id: {run_id}"


def assert_train_loss_below_threshold(run_id):
    """Test that the 'stream.ERA5.loss_mse.loss_avg' metric is below a threshold."""
    metrics = load_metrics(run_id)
    loss_metric = next(
        (
            metric.get("loss.LossPhysical.ERA5.mse.loss_avg", None)
            for metric in reversed(metrics)
            if metric.get("stage") == "train"
        ),
        None,
    )
    assert loss_metric is not None, (
        "'loss.LossPhysical.ERA5.mse.loss_avg' metric is missing in metrics file"
    )
    # Check that the loss does not explode in a single mini_epoch
    # This is meant to be a quick test, not a convergence test
    target = 0.25
    assert loss_metric < target, (
        f"'loss.LossPhysical.ERA5.mse.loss_avg' is {loss_metric}, expected to be below {target}"
    )


def assert_val_loss_below_threshold(run_id):
    """Test that the 'stream.ERA5.loss_mse.loss_avg' metric is below a threshold."""
    metrics = load_metrics(run_id)
    loss_metric = next(
        (
            metric.get("loss.LossPhysical.ERA5.mse.loss_avg", None)
            for metric in reversed(metrics)
            if metric.get("stage") == "val"
        ),
        None,
    )
    assert loss_metric is not None, (
        "'stream.ERA5.loss_mse.loss_avg' metric is missing in metrics file"
    )
    # Check that the loss does not explode in a single mini_epoch
    # This is meant to be a quick test, not a convergence test
    assert loss_metric < 0.25, (
        f"'loss.LossPhysical.ERA5.mse.loss_avg' is {loss_metric}, expected to be below 0.25"
    )
