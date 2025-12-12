"""
Integration test for the Weather Generator with multiple streams and observations.
This test must run on a GPU machine.
It performs training and inference with multiple data sources including gridded and obs data.

Command:
uv run pytest ./integration_tests/small_multi_stream_test.py
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
    assert_stream_losses_below_threshold(test_run_id, stage="train")
    assert_stream_losses_below_threshold(test_run_id, stage="val")
    logger.info("\nend test_train_multi_stream")


def infer_multi_stream(run_id):
    """Run inference for multi-stream model."""
    logger.info("run multi-stream inference")
    inference_from_args(
        ["-start", "2021-10-10", "-end", "2022-10-11", "--samples", "10", "--mini_epoch", "0"]
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
    evaluate_from_config(cfg, None, None)


def load_metrics(run_id):
    """Helper function to load metrics"""
    file_path = get_train_metrics_path(base_path=WEATHERGEN_HOME / "results", run_id=run_id)
    if not file_path.is_file():
        raise FileNotFoundError(f"Metrics file not found for run_id: {run_id}")
    with open(file_path) as f:
        json_str = f.readlines()
    return json.loads("[" + "".join([s.replace("\n", ",") for s in json_str])[:-1] + "]")


def assert_metrics_file_exists(run_id):
    """Test that the metrics file exists and can be loaded."""
    file_path = get_train_metrics_path(base_path=WEATHERGEN_HOME / "results", run_id=run_id)
    assert file_path.is_file(), f"Metrics file does not exist for run_id: {run_id}"
    metrics = load_metrics(run_id)
    logger.info(f"Loaded metrics for run_id: {run_id}: {metrics}")
    assert metrics is not None, f"Failed to load metrics for run_id: {run_id}"


def assert_stream_losses_below_threshold(run_id, stage="train"):
    """
    Test that stream losses are below threshold for a given stage.
    
    Args:
        run_id: The run identifier
        stage: Either "train" or "val"
    """
    metrics = load_metrics(run_id)

    # Thresholds for train and val
    thresholds = {
        "train": {
            "ERA5": 0.2,
            "NPPATMS": 0.5,
            "SurfaceCombined": 0.7,
        },
        "val": {
            "ERA5": 0.2,
            "NPPATMS": 0.4,
            "SurfaceCombined": 0.6,
        },
    }

    stage_thresholds = thresholds[stage]

    losses = {}
    for stream_name, threshold in stage_thresholds.items():
        loss = next(
            (
                metric.get(f"loss.LossPhysical.{stream_name}.mse.loss_avg")
                for metric in reversed(metrics)
                if metric.get("stage") == stage
            ),
            None,
        )

        assert loss is not None, f"'loss.LossPhysical.{stream_name}.mse.loss_avg' {stage} metric is missing"
        assert loss < threshold, (
            f"'loss.LossPhysical.{stream_name}.mse.loss_avg' {stage} loss is {loss}, expected below {threshold}"
        )

        losses[stream_name] = loss

    stage_label = "\nTrain" if stage == "train" else "Validation"
    logger.info(f"{stage_label} losses – " + ", ".join(f"{k}: {v:.4f}" for k, v in losses.items()))
