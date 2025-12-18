"""
Downloads metrics from MLFlow.
"""

import logging

import mlflow
import polars as pl
import streamlit as st
from mlflow.client import MlflowClient

from weathergen.metrics.mlflow_utils import setup_mlflow as setup_mlflow_utils

_logger = logging.getLogger(__name__)

phase = "train"
exp_lifecycle = "test"
project = "WeatherGenerator"
#experiment_id = "384213844828345"
all_stages = ["train", "val", "eval"]

# Polars utilities
stage_is_eval = pl.col("tags.stage") == "eval"
stage_is_train = pl.col("tags.stage") == "train"
stage_is_val = pl.col("tags.stage") == "val"


# Cache TTL in seconds
ST_TTL_SEC = 3600


class MlFlowUpload:
    tracking_uri = "databricks"
    registry_uri = "databricks-uc"
    experiment_name = "/Shared/weathergen-dev/core-model/defaultExperiment"


@st.cache_resource(ttl=ST_TTL_SEC)
def setup_mflow() -> MlflowClient:
    return setup_mlflow_utils(private_config=None)


@st.cache_data(ttl=ST_TTL_SEC)
def get_experiment_id() -> str:
    client = setup_mflow()
    exp = client.get_experiment_by_name(MlFlowUpload.experiment_name)
    assert exp is not None
    return exp.experiment_id


@st.cache_data(ttl=ST_TTL_SEC, max_entries=2)
def latest_runs():
    """
    Get the latest runs for each WG run_id and stage.
    """
    _logger.info("Downloading latest runs from MLFlow")
    runs_pdf = pl.DataFrame(
        mlflow.search_runs(
            experiment_ids=[get_experiment_id()],
            # filter_string="status='FINISHED' AND tags.completion_status = 'success'",
        )
    )
    runs_pdf = runs_pdf.filter(pl.col("tags.stage").is_in(all_stages))
    latest_run_by_exp = (
        runs_pdf.sort(by="end_time", descending=True)
        .group_by(["tags.run_id", "tags.stage"])
        .agg(pl.col("*").last())
        .sort(by="tags.run_id")
    )
    _logger.info("Number of latest runs: %d", len(runs_pdf))
    return latest_run_by_exp


@st.cache_data(ttl=ST_TTL_SEC, max_entries=2)
def all_runs():
    _logger.info("Downloading all runs from MLFlow")
    runs_pdf = pl.DataFrame(
        mlflow.search_runs(
            experiment_ids=[get_experiment_id()],
            # filter_string="status='FINISHED' AND tags.completion_status = 'success'",
        )
    )
    _logger.info("Number of all runs: %d", len(runs_pdf))
    return runs_pdf
