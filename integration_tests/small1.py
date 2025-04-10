import json
import logging
import os
import shutil
from pathlib import Path

import pytest

from weathergen import evaluate_from_args, train_with_args

logger = logging.getLogger(__name__)

run_id = "testsmall1"


@pytest.fixture()
def setup():
    logger.info("setup fixture")
    print("start fixture")
    p = Path("./results") / run_id
    shutil.rmtree(p, ignore_errors=True)
    yield
    print("end fixture")


def test_train(setup):
    logger.info("test_train")

    if False:
        train_with_args(
            "--run_id=testsmall1 ".split()
            + [
                "--private_config",
                "../WeatherGenerator-private/hpc/hpc2020/config/paths.yml",
                "--config",
                "integration_tests/small1.yaml",
                "--streams_directory",
                "./config/streams/streams_test/",
            ]
        )
        evaluate_from_args(
            "-start 2022-10-10 -end 2022-10-11 --samples 10 --same_run_id --epoch 1".split()
            + [
                "--run_id",
                run_id,
                "--private_config",
                "../WeatherGenerator-private/hpc/hpc2020/config/paths.yml",
            ]
        )
    assert_missing_metrics_file(run_id)
    assert_train_loss_below_threshold(run_id)
    assert_val_loss_below_threshold(run_id)
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
