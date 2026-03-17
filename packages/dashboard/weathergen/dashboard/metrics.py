"""
Downloads metrics from MLFlow.
"""

import datetime
import logging

import mlflow
import polars as pl
import polars.selectors as ps
import streamlit as st
from mlflow.client import MlflowClient

from weathergen.metrics.mlflow_utils import setup_mlflow as setup_mlflow_utils

_logger = logging.getLogger(__name__)

phase = "train"
exp_lifecycle = "test"
project = "WeatherGenerator"
# experiment_id = "384213844828345"
all_stages = ["train", "val", "eval", "inference"]

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


@st.cache_data(ttl=ST_TTL_SEC, max_entries=20)
def latest_runs(
    keep_metrics: bool | tuple[str, ...] = True,
    keep_params: bool | tuple[str, ...] = True,
    latest_runs: bool = False,
):
    """
    Get the latest runs for each WG run_id and stage.
    Returns only specified metrics and params to reduce memory usage.
    """
    _logger.info("Downloading latest runs from MLFlow")
    # A month ago timestamp in milliseconds
    month_ago_ts = int((datetime.datetime.now() - datetime.timedelta(days=30)).timestamp() * 1000)
    runs_pdf = pl.DataFrame(
        mlflow.search_runs(
            experiment_ids=[get_experiment_id()],
            filter_string=f"attributes.start_time >= {month_ago_ts}",
        )
    )
    if keep_metrics is True:
        _logger.info("Keeping metrics columns")
    else:
        _logger.info("Dropping metrics columns")
        # Keep num_samples as it is useful for filtering and grouping.
        keep_metrics_list = ["metrics.num_samples"] if keep_metrics is False else list(keep_metrics)
        runs_pdf = runs_pdf.select(_start_with_reduce("metrics.", keep_metrics_list))
    if keep_params is True:
        _logger.info("Keeping params columns")
    else:
        _logger.info("Dropping params columns")
        # Still keep the wgtags params, as they are useful for filtering and grouping.
        keep_param_prefixes = (
            ["params.wgtags.", "params.world_size"] if keep_params is False else list(keep_params)
        )

        runs_pdf = runs_pdf.select(_start_with_reduce("params.", keep_param_prefixes))
    runs_pdf = runs_pdf.filter(pl.col("tags.stage").is_in(all_stages))

    latest_run_by_exp = (
        runs_pdf.sort(by="end_time", descending=True)
        .group_by(["tags.run_id", "tags.stage"])
        .agg(pl.col("*").last())
        .sort(by="tags.run_id")
    )
    _logger.info("Number of latest runs: %d", len(runs_pdf))
    return latest_run_by_exp


@st.cache_data(ttl=ST_TTL_SEC, max_entries=10)
def all_runs(
    keep_metrics: bool | tuple[str, ...],
    keep_params: bool | tuple[str, ...],
    latest_runs: bool = False,
) -> pl.DataFrame:
    _logger.info("Downloading all runs from MLFlow")
    month_ago_ts = int((datetime.datetime.now() - datetime.timedelta(days=30)).timestamp() * 1000)
    runs_pdf = pl.DataFrame(
        mlflow.search_runs(
            experiment_ids=[get_experiment_id()],
            filter_string=f"attributes.start_time >= {month_ago_ts} " if latest_runs else "",
            max_results=10000,
        )
    )
    _logger.info("Number of all runs: %d %d", len(runs_pdf), len(runs_pdf.columns))
    if keep_metrics is True:
        _logger.info("Keeping metrics columns")
    else:
        _logger.info("Dropping metrics columns")
        # Keep num_samples as it is useful for filtering and grouping.
        keep_metrics_list = ["metrics.num_samples"] if keep_metrics is False else list(keep_metrics)
        runs_pdf = runs_pdf.select(_start_with_reduce("metrics.", keep_metrics_list))
    if keep_params is True:
        _logger.info("Keeping params columns")
    else:
        _logger.info("Dropping params columns")
        # Still keep the wgtags params, as they are useful for filtering and grouping.
        keep_param_prefixes = (
            ["params.wgtags.", "params.world_size"] if keep_params is False else list(keep_params)
        )

        runs_pdf = runs_pdf.select(_start_with_reduce("params.", keep_param_prefixes))
    _logger.info("Number of all runs after filtering: %d %d", len(runs_pdf), len(runs_pdf.columns))
    _logger.info("Columns in all runs: %s", runs_pdf.columns)
    _logger.info("Columns in all runs: %s", runs_pdf)
    return runs_pdf


def _start_with_reduce(remove: str, keep: list[str]) -> pl.Expr:
    e = ~ps.starts_with(remove)
    for p in keep:
        e = e | ps.starts_with(p)
    return e
