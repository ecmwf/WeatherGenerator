"""
Downloads metrics from MLFlow.
"""

import logging

import mlflow
import mlflow.client
import polars as pl
import streamlit as st

_logger = logging.getLogger(__name__)

phase = "train"
exp_lifecycle = "test"
project = "WeatherGenerator"
experiment_id = "384213844828345"
all_stages = ["train", "val"]

# Cache TTL in seconds
_ttl_sec = 600


class MlFlowUpload:
    tracking_uri = "databricks"
    registry_uri = "databricks-uc"
    experiment_name = "/Shared/weathergen-dev/core-model/defaultExperiment"


@st.cache_resource(ttl=_ttl_sec)
def setup_mflow():
    # os.environ["DATABRICKS_HOST"] = None
    # os.environ["DATABRICKS_TOKEN"] = None
    mlflow.set_tracking_uri(MlFlowUpload.tracking_uri)
    mlflow.set_registry_uri(MlFlowUpload.registry_uri)
    mlflow_client = mlflow.client.MlflowClient(
        tracking_uri=MlFlowUpload.tracking_uri, registry_uri=MlFlowUpload.registry_uri
    )
    _logger.info("MLFlow tracking URI: %s", mlflow.get_tracking_uri())
    return mlflow_client


@st.cache_data(ttl=_ttl_sec, max_entries=2)
def latest_runs():
    _logger.info("Downloading latest runs from MLFlow")
    runs_pdf = pl.DataFrame(
        mlflow.search_runs(
            experiment_ids=[experiment_id],
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


@st.cache_data(ttl=_ttl_sec, max_entries=2)
def all_runs():
    _logger.info("Downloading all runs from MLFlow")
    runs_pdf = pl.DataFrame(
        mlflow.search_runs(
            experiment_ids=[experiment_id],
            # filter_string="status='FINISHED' AND tags.completion_status = 'success'",
        )
    )
    _logger.info("Number of all runs: %d", len(runs_pdf))
    return runs_pdf
