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
import glob
import shutil
from pathlib import Path
import xarray as xr
import numpy as np

import omegaconf
import pytest

from weathergen.evaluate.run_evaluation import evaluate_from_config
from weathergen.evaluate.export.export_inference import export_from_args
from weathergen.run_train import main
from weathergen.utils.metrics import get_train_metrics_path
from weathergen.common.config import get_model_results
from weathergen.common.io import zarrio_reader



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

    main(
        [
            "train",
            f"--base-config={WEATHERGEN_HOME}/integration_tests/small1.yml",
            "--run-id",
            test_run_id,
        ]
    )

    infer(test_run_id)
    evaluate_results(test_run_id)
    assert_missing_metrics_file(test_run_id)
    assert_train_loss_below_threshold(test_run_id)
    assert_val_loss_below_threshold(test_run_id)
    export_inference(test_run_id)
    check_export(test_run_id)
    logger.info("end test_train")


def infer(run_id):
    logger.info("run inference")
    main(
        [
            "inference",
            "--mini-epoch",
            "0",
            "--from-run-id",
            run_id,
            "--run-id",
            run_id,
            "--config",
            f"{WEATHERGEN_HOME}/integration_tests/small1.yml",
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
                "regions": ["global"],
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
    evaluate_from_config(cfg, None, None)


def load_metrics(run_id):
    """Helper function to load metrics"""
    file_path = get_train_metrics_path(base_path=WEATHERGEN_HOME / "results" / run_id, run_id=run_id)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found for run_id: {run_id}")
    with open(file_path) as f:
        json_str = f.readlines()
    return json.loads("[" + r"".join([s.replace("\n", ",") for s in json_str])[:-1] + "]")


def assert_missing_metrics_file(run_id):
    """Test that a missing metrics file raises FileNotFoundError."""
    file_path = get_train_metrics_path(base_path=WEATHERGEN_HOME / "results"/ run_id, run_id=run_id)
    assert os.path.exists(file_path), f"Metrics file does not exist for run_id: {run_id}"
    metrics = load_metrics(run_id)
    logger.info(f"Loaded metrics for run_id: {run_id}: {metrics}")
    assert metrics is not None, f"Failed to load metrics for run_id: {run_id}"


def assert_train_loss_below_threshold(run_id):
    """Test that the 'LossPhysical.ERA5.mse.avg' metric is below a threshold."""
    metrics = load_metrics(run_id)
    loss_avg_name = "LossPhysical.ERA5.mse.avg"
    loss_metric = next(
        (
            metric.get(loss_avg_name, None)
            for metric in reversed(metrics)
            if metric.get("stage") == "train"
        ),
        None,
    )
    assert loss_metric is not None, f"'{loss_avg_name}' metric is missing in metrics file"
    # Check that the loss does not explode in a single mini_epoch
    # This is meant to be a quick test, not a convergence test
    target = 0.25
    assert loss_metric < target, (
        f"'{loss_avg_name}' is {loss_metric}, expected to be below {target}"
    )


def assert_val_loss_below_threshold(run_id):
    """Test that the 'LossPhysical.ERA5.mse.avg' metric is below a threshold."""
    metrics = load_metrics(run_id)
    loss_avg_name = "LossPhysical.ERA5.mse.avg"
    loss_metric = next(
        (
            metric.get(loss_avg_name, None)
            for metric in reversed(metrics)
            if metric.get("stage") == "val"
        ),
        None,
    )
    assert loss_metric is not None, f"'{loss_avg_name}' metric is missing in metrics file"
    # Check that the loss does not explode in a single mini_epoch
    # This is meant to be a quick test, not a convergence test
    assert loss_metric < 0.2, f"'{loss_avg_name}' is {loss_metric}, expected to be below 0.2"


def export_inference(run_id):
    logger.info("export to netcdf")
    export_from_args(
        ["--run-id", run_id,
         "--stream", "ERA5",
         "--output-dir", f"{WEATHERGEN_HOME}/results/{run_id}",
         "--format", "netcdf",
         "--samples", "0", "1",
         "--fsteps", "1" ,"2"]
    )

def check_export(run_id):
    fname_zarr = get_model_results(run_id, mini_epoch = 0, rank = 0)
    nc_folder = Path(WEATHERGEN_HOME / "results" / run_id)

    with zarrio_reader(fname_zarr) as zio:
        for sample in [0,1]:
            #find timestamp for sample
            out = zio.get_data(sample, "ERA5", 1)
            zarr_ds = out.prediction.as_xarray()
            min_timestamp = np.min(zarr_ds["valid_time"]) - np.timedelta64(6, "h")
            frt = np.datetime_as_string(min_timestamp, unit="h")

            #open correct nc file
            nc_path = glob.glob(f"{str(nc_folder)}/prediction_{frt}*.nc")
            nc_ds = xr.open_dataset(nc_path[0])

            for fstep in [1,2]:
                # extract and compare per sample/fstep
                out = zio.get_data(sample, "ERA5", fstep)
                zarr_ds = out.prediction.as_xarray()
                zarr_values = zarr_ds.sel({"channel" :"t_850"}).values
                nc_values = nc_ds.sel({"forecast_period" : np.timedelta64(6, "h") * fstep,"pressure": 850})["t"].values
                
                assert np.array_equal(np.squeeze(zarr_values),np.squeeze(nc_values)), "exporting data failed"
