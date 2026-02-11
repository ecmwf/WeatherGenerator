"""
Small test for the Weather Generator.
This test must run on a GPU machine.
It performs a training and inference of the Weather Generator model.

Command:
uv run pytest  ./integration_tests/jepa1.py
"""

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from weathergen.run_train import main
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


@pytest.mark.parametrize("test_run_id", ["test_jepa1_" + commit_hash])
def test_train(setup, test_run_id):
    logger.info(f"test_train with run_id {test_run_id} {WEATHERGEN_HOME}")
    
    main(
        [
            "train",
            f"--config={WEATHERGEN_HOME}/integration_tests/jepa1.yaml",
            "--run-id",
            test_run_id,
        ]
    )

    assert_missing_metrics_file(test_run_id)
    assert_nans_in_metrics_file(test_run_id)
    logger.info("end test_train")



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
    
def assert_nans_in_metrics_file(run_id):
    """Test that there are no NaNs in the metrics file."""
    metrics = load_metrics(run_id)
    loss_values_train = np.array(
        [
            entry.get('LossLatentSSLStudentTeacher.loss_avg')
            for entry in metrics if entry.get("stage") == 'train'
        ]
    )
    loss_values_val = np.array(
        [
            entry.get('LossLatentSSLStudentTeacher.loss_avg')
            for entry in metrics if entry.get("stage") == 'val'
        ]
    )
    
    #remove nans if applicable
    loss_values_train = np.array(
        [float(value) if value != 'nan' else np.nan for value in loss_values_train]
    )
    loss_values_val = np.array(
        [float(value) if value != 'nan' else np.nan for value in loss_values_val]
    )
    
    assert not np.isnan(loss_values_train).any(), (
        "NaN values found in training loss metrics!"
    )
    
    assert not np.isnan(loss_values_val).any(), (
        "NaN values found in validation loss metrics!"
    )

