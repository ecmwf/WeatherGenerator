"""
Small test for the Weather Generator.
This test must run on a GPU machine. It performs a training and evaluation of the Weather Generator model.

Command:
uv run pytest  ./integration_tests/small1.py
"""

import json
import logging
import os
import shutil
from pathlib import Path

import pytest

from weathergen import evaluate_from_args, train_with_args


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




@pytest.fixture()
def setup(test_run_id):
    logger.info(f"setup fixture with {test_run_id}")
    shutil.rmtree(Path("./results") / test_run_id, ignore_errors=True)
    shutil.rmtree(Path("./models") / test_run_id, ignore_errors=True)
    yield
    print("end fixture")


@pytest.mark.parametrize("test_run_id", ["test_small1_" + commit_hash])
def test_train(setup, test_run_id):
    logger.info(f"test_train with run_id {test_run_id}")

    train_with_args(
        "--config=integration_tests/small1.yaml".split()
        + [
            "--run_id",
            test_run_id,
        ],
        "./config/streams/streams_test/",
    )

    evaluate_from_args(
        "-start 2022-10-10 -end 2022-10-11 --samples 10 --same_run_id --epoch 0".split()
        + [
            "--run_id",
            test_run_id,
            "--config",
            "integration_tests/small1.yaml",
        ]
    )
    assert_missing_metrics_file(test_run_id)
    assert_train_loss_below_threshold(test_run_id)
    assert_val_loss_below_threshold(test_run_id)
    logger.info("end test_train")


def load_metrics(run_id):
    """Helper function to load metrics"""
    file_path = f"./results/{run_id}/metrics.json"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found for run_id: {run_id}")
    json_str = open(file_path).readlines()
    return json.loads("[" + r"".join([s.replace("\n", ",") for s in json_str])[:-1] + "]")


def assert_missing_metrics_file(run_id):
    """Test that a missing metrics file raises FileNotFoundError."""
    file_path = f"./results/{run_id}/metrics.json"
    assert os.path.exists(file_path), f"Metrics file does not exist for run_id: {run_id}"
    metrics = load_metrics(run_id)
    logger.info(f"Loaded metrics for run_id: {run_id}: {metrics}")
    assert metrics is not None, f"Failed to load metrics for run_id: {run_id}"


def assert_train_loss_below_threshold(run_id):
    """Test that the 'stream.era5.loss_mse.loss_avg' metric is below a threshold."""
    metrics = load_metrics(run_id)
    loss_metric = next(
        (
            metric.get("stream.era5.loss_mse.loss_avg", None)
            for metric in reversed(metrics)
            if metric.get("stage") == "train"
        ),
        None,
    )
    assert loss_metric is not None, (
        "'stream.era5.loss_mse.loss_avg' metric is missing in metrics file"
    )
    assert loss_metric < 0.25, (
        f"'stream.era5.loss_mse.loss_avg' is {loss_metric}, expected to be below 0.25"
    )


def assert_val_loss_below_threshold(run_id):
    """Test that the 'stream.era5.loss_mse.loss_avg' metric is below a threshold."""
    metrics = load_metrics(run_id)
    loss_metric = next(
        (
            metric.get("stream.era5.loss_mse.loss_avg", None)
            for metric in reversed(metrics)
            if metric.get("stage") == "val"
        ),
        None,
    )
    assert loss_metric is not None, (
        "'stream.era5.loss_mse.loss_avg' metric is missing in metrics file"
    )
    assert loss_metric < 0.25, (
        f"'stream.era5.loss_mse.loss_avg' is {loss_metric}, expected to be below 0.25"
    )
