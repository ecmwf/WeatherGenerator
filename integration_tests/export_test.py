"""
Small test for the Weather Generator.
This test must run on a GPU machine.
It performs (training, inference - if necessary) and export of the Weather Generator model.

Command:
uv run pytest  ./integration_tests/export_test.py
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


@pytest.mark.parametrize("test_run_id", ["test_small1_" + commit_hash])
def test_export(test_run_id):
    logger.info(f"test_export with run_id {test_run_id} {WEATHERGEN_HOME}")
    if not find_inference(test_run_id):
        logger.info(f"{test_run_id} not found, run train with run_id {test_run_id} {WEATHERGEN_HOME}")
        main(
            [
                "train",
                f"--base-config={WEATHERGEN_HOME}/integration_tests/small1.yml",
                "--run-id",
                test_run_id,
            ]
        )
        infer(test_run_id)
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

def find_inference(run_id):
    try:
        return get_model_results(run_id, mini_epoch=0, rank=0)
    except FileNotFoundError as e:
        return False


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
