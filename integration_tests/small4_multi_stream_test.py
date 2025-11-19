"""
Integration test for the Weather Generator with multiple streams and observations.
This test must run on a GPU machine.
It performs training and inference with multiple data sources including gridded and obs data.

Command:
uv run pytest ./integration_tests/small4_multi_stream_test.py
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


@pytest.mark.parametrize("test_run_id", ["test_multi_stream_" + commit_hash])
def test_train_multi_stream(setup, test_run_id):
    """Test training with multiple streams including gridded and observation data."""
    logger.info(f"test_train_multi_stream with run_id {test_run_id} {WEATHERGEN_HOME}")

    train_with_args(
        f"--config={WEATHERGEN_HOME}/integration_tests/small_multi_stream.yaml".split()
        + [
            "--run_id",
            test_run_id,
        ],
        f"{WEATHERGEN_HOME}/integration_tests/streams_multi/",
    )

    infer_multi_stream(test_run_id)
    evaluate_multi_stream_results(test_run_id)
    assert_metrics_file_exists(test_run_id)
    assert_all_stream_losses_below_threshold(test_run_id)
    assert_val_losses_below_threshold(test_run_id)
    logger.info("end test_train_multi_stream")


def infer_multi_stream(run_id):
    """Run inference for multi-stream model."""
    logger.info("run multi-stream inference")
    inference_from_args(
        ["-start", "2022-10-10", "-end", "2022-10-11", "--samples", "10", "--mini_epoch", "0"]
        + [
            "--from_run_id",
            run_id,
            "--run_id",
            run_id,
            "--streams_output",
            "ERA5", "SurfaceCombined", "NPPATMS",
            "--config",
            f"{WEATHERGEN_HOME}/integration_tests/small_multi_stream.yaml",
        ]
    )


def evaluate_multi_stream_results(run_id):
    """Run evaluation for multiple streams."""
    logger.info("run multi-stream evaluation")
    #TODO remove and put in a separate config file
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
                run_id: {
                    "streams": {
                        "ERA5": {
                            "results_base_dir": "./results/",
                            "channels": ["t_850"],
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
                            "results_base_dir": "./results/",
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
                            "results_base_dir": "./results/",
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
    evaluate_from_config(cfg, None)


def load_metrics(run_id):
    """Helper function to load metrics"""
    file_path = get_train_metrics_path(base_path=WEATHERGEN_HOME / "results", run_id=run_id)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found for run_id: {run_id}")
    with open(file_path) as f:
        json_str = f.readlines()
    return json.loads("[" + "".join([s.replace("\n", ",") for s in json_str])[:-1] + "]")


def assert_metrics_file_exists(run_id):
    """Test that the metrics file exists and can be loaded."""
    file_path = get_train_metrics_path(base_path=WEATHERGEN_HOME / "results", run_id=run_id)
    assert os.path.exists(file_path), f"Metrics file does not exist for run_id: {run_id}"
    metrics = load_metrics(run_id)
    logger.info(f"Loaded metrics for run_id: {run_id}: {metrics}")
    assert metrics is not None, f"Failed to load metrics for run_id: {run_id}"


def assert_all_stream_losses_below_threshold(run_id):
    """Test that all stream losses are below threshold."""
    metrics = load_metrics(run_id)
    
    # Define streams and their thresholds
    streams = {
        "ERA5": 2.0,
        "NPPATMS": 2.0,
        "SurfaceCombined": 2.0,
    }
    
    losses = {}
    for stream_name, threshold in streams.items():
        loss = next(
            (
                metric.get(f"stream.{stream_name}.loss_mse.loss_avg", None)
                for metric in reversed(metrics)
                if metric.get("stage") == "train"
            ),
            None,
        )
        assert loss is not None, f"'stream.{stream_name}.loss_mse.loss_avg' metric is missing"
        assert loss < threshold, (
            f"{stream_name} train loss is {loss}, expected to be below {threshold}"
        )
        losses[stream_name] = loss
    
    logger.info(f"Train losses - " + ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()]))


def assert_val_losses_below_threshold(run_id):
    """Test that validation losses for all streams are below threshold."""
    metrics = load_metrics(run_id)
    
    # Define streams and their validation thresholds
    streams = {
        "ERA5": 2.0,
        "NPPATMS": 2.0,
        "SurfaceCombined": 2.0,
    }
    
    val_losses = {}
    for stream_name, threshold in streams.items():
        val_loss = next(
            (
                metric.get(f"stream.{stream_name}.loss_mse.loss_avg", None)
                for metric in reversed(metrics)
                if metric.get("stage") == "val"
            ),
            None,
        )
        assert val_loss is not None, f"'stream.{stream_name}.loss_mse.loss_avg' validation metric is missing"
        assert val_loss < threshold, (
            f"{stream_name} val loss is {val_loss}, expected to be below {threshold}"
        )
        val_losses[stream_name] = val_loss
    
    logger.info(f"Validation losses - " + ", ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()]))


'''
def assert_val_losses_below_threshold(run_id):
    """Test that validation losses for all streams are below threshold."""
    metrics = load_metrics(run_id)
    
    # Check ERA5 validation loss
    era5_val_loss = next(
        (
            metric.get("stream.ERA5.loss_mse.loss_avg", None)
            for metric in reversed(metrics)
            if metric.get("stage") == "val"
        ),
        None,
    )
    assert era5_val_loss is not None, "'stream.ERA5.loss_mse.loss_avg' validation metric is missing"
    era5_val_threshold = 1.5
    assert era5_val_loss < era5_val_threshold, (
        f"ERA5 val loss is {era5_val_loss}, expected to be below {era5_val_threshold}"
    )
    
    # Check SYNOP validation loss
    synop_val_loss = next(
        (
            metric.get("stream.SurfaceCombined.loss_mse.loss_avg", None)
            for metric in reversed(metrics)
            if metric.get("stage") == "val"
        ),
        None,
    )
    assert synop_val_loss is not None, "'stream.SurfaceCombined.loss_mse.loss_avg' validation metric is missing"
    synop_val_threshold = 2.0
    assert synop_val_loss < synop_val_threshold, (
        f"SYNOP val loss is {synop_val_loss}, expected to be below {synop_val_threshold}"
    )
    
    # Check NPPATMS validation loss
    npp_atms_val_loss = next(
        (
            metric.get("stream.NPPATMS.loss_mse.loss_avg", None)
            for metric in reversed(metrics)
            if metric.get("stage") == "val"
        ),
        None,
    )
    assert npp_atms_val_loss is not None, "'stream.NPPATMS.loss_mse.loss_avg' validation metric is missing"
    npp_atms_val_threshold = 2.0
    assert npp_atms_val_loss < npp_atms_val_threshold, (
        f"NPPATMS val loss is {npp_atms_val_loss}, expected to be below {npp_atms_val_threshold}"
    )
    
    logger.info(f"Validation losses - ERA5: {era5_val_loss:.4f}, SYNOP: {synop_val_loss:.4f}, NPPATMS: {npp_atms_val_loss:.4f}")

'''