import unittest
import json
import argparse
import pytest
import os

# @pytest.fixture(scope="session")
# def run_id(pytestconfig):
#     return pytestconfig.getoption("run_id")

@pytest.fixture(scope='session')
def run_id(request):
    value = request.config.option.run_id
    return value_value

def test_print_run_id(run_id):
    print ("Displaying run id: %s" % run_id)

def load_metrics(run_id):
    """Helper function to load metrics"""
    file_path = f'./results/{run_id}/metrics.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found for run_id: {run_id}")
    json_str = open(file_path).readlines()
    return json.loads('['+r"".join([s.replace("\n", ",") for s in json_str])[:-1]+']')

def test_missing_metrics_file(run_id):
    """Test that a missing metrics file raises FileNotFoundError."""
    file_path = f'./results/{run_id}/metrics.json'
    assert os.path.exists(file_path), f"Metrics file does not exist for run_id: {run_id}"
    metrics = load_metrics(run_id)
    assert metrics is not None, f"Failed to load metrics for run_id: {run_id}"

def test_train_loss_below_threshold(run_id):
    """Test that the 'stream.era5.loss_mse.loss_avg' metric is below 0.2."""
    metrics = load_metrics(run_id)
    loss_metric = next((metric.get("stream.era5.loss_mse.loss_avg", None) for metric in reversed(metrics) if metric.get('stage') == 'train'), None)
    assert loss_metric is not None, "'stream.era5.loss_mse.loss_avg' metric is missing in metrics file"
    assert loss_metric < 0.15, f"'stream.era5.loss_mse.loss_avg' is {loss_metric}, expected to be below 0.15"

def test_val_loss_below_threshold(run_id):
    """Test that the 'stream.era5.loss_mse.loss_avg' metric is below 0.2."""
    metrics = load_metrics(run_id)
    loss_metric = next((metric.get("stream.era5.loss_mse.loss_avg", None) for metric in reversed(metrics) if metric.get('stage') == 'val'), None)
    assert loss_metric is not None, "'stream.era5.loss_mse.loss_avg' metric is missing in metrics file"
    assert loss_metric < 0.15, f"'stream.era5.loss_mse.loss_avg' is {loss_metric}, expected to be below 0.15"


def test_gpu_performance_above_threshold(run_id):
    """Test that the 'perf.gpu' metric is above 70."""
    metrics = load_metrics(run_id)
    gpu_performance = next((metric.get("perf.gpu", None) for metric in reversed(metrics) if metric.get('stage') == 'train'), None)
    assert gpu_performance is not None, "'perf.gpu' metric is missing in metrics file"
    assert gpu_performance > 70, f"'perf.gpu' is {gpu_performance}, expected to be above 70"
