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

import pytest

import weathergen.common.io as io
import weathergen.utils.config as config
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


    logger.info("run inference")
    inference_from_args(
        ["-start", "2022-10-10", "-end", "2022-10-11", "--samples", "10", "--epoch", "0"]
        + [
            "--from_run_id",
            test_run_id,
            "--run_id",
            test_run_id,
            "--config",
            f"{WEATHERGEN_HOME}/integration_tests/small1.yaml",
        ]
    )
    logger.info("run evaluation")
    evaluate_results(test_run_id)
    assert_missing_metrics_file(test_run_id)
    assert_train_loss_below_threshold(test_run_id)
    assert_val_loss_below_threshold(test_run_id)
    logger.info("end test_train")


def evaluate_results(run_id):
    cf = config.load_model_config(run_id, None, None)
    data_root = config.get_path_output(cf, 0)

    with io.ZarrIO(data_root) as reader:
        samples = reader.samples
        fsteps = reader.forecast_steps
        streams = reader.streams

        item = reader.get_data(samples[0], streams[0], fsteps[0])
        ds = item.prediction.as_xarray()
        logger.info(ds)
        item.target.as_xarray()
        logger.info(ds)
        if item.key.with_source:
            ds = item.source.as_xarray()
            logger.info(ds)

    # TODO: test concat multiple samples


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
            metric.get("stream.ERA5.loss_mse.loss_avg", None)
            for metric in reversed(metrics)
            if metric.get("stage") == "train"
        ),
        None,
    )
    assert loss_metric is not None, (
        "'stream.ERA5.loss_mse.loss_avg' metric is missing in metrics file"
    )
    # Check that the loss does not explode in a single epoch
    # This is meant to be a quick test, not a convergence test
    assert loss_metric < 1.25, (
        f"'stream.ERA5.loss_mse.loss_avg' is {loss_metric}, expected to be below 0.25"
    )


def assert_val_loss_below_threshold(run_id):
    """Test that the 'stream.ERA5.loss_mse.loss_avg' metric is below a threshold."""
    metrics = load_metrics(run_id)
    loss_metric = next(
        (
            metric.get("stream.ERA5.loss_mse.loss_avg", None)
            for metric in reversed(metrics)
            if metric.get("stage") == "val"
        ),
        None,
    )
    assert loss_metric is not None, (
        "'stream.ERA5.loss_mse.loss_avg' metric is missing in metrics file"
    )
    # Check that the loss does not explode in a single epoch
    # This is meant to be a quick test, not a convergence test
    assert loss_metric < 1.25, (
        f"'stream.ERA5.loss_mse.loss_avg' is {loss_metric}, expected to be below 0.25"
    )
