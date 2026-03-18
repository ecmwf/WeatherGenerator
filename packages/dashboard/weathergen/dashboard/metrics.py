"""
Downloads metrics from MLFlow.
"""

import datetime
import logging
from collections.abc import Iterator

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


@st.cache_data(ttl=ST_TTL_SEC, persist="disk")
def get_experiment_id() -> str:
    client = setup_mflow()
    exp = client.get_experiment_by_name(MlFlowUpload.experiment_name)
    assert exp is not None
    return exp.experiment_id


def _iter_run_batches(
    filter_string: str,
    keep_metrics: bool | tuple[str, ...],
    keep_params: bool | tuple[str, ...],
    batch_size: int = 1000,
) -> Iterator[pl.DataFrame]:
    """Yield DataFrames of runs in batches, filtering columns early to limit memory usage."""
    # Determine which metric/param prefixes to keep
    if keep_metrics is True:
        keep_metrics_list = None  # keep all
    else:
        keep_metrics_list = ["metrics.num_samples"] if keep_metrics is False else list(keep_metrics)

    if keep_params is True:
        keep_param_prefixes = None  # keep all
    else:
        keep_param_prefixes = (
            ["params.wgtags.", "params.world_size"] if keep_params is False else list(keep_params)
        )

    client = setup_mflow()
    experiment_ids = [get_experiment_id()]
    page_token = None
    total = 0

    while True:
        page = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=batch_size,
            page_token=page_token,
        )
        if not page:
            break

        # Convert batch to dicts, keeping only relevant keys
        rows = []
        for run in page:
            row: dict[str, object] = {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": datetime.datetime.fromtimestamp(run.info.start_time / 1000, tz=datetime.timezone.utc),
                "end_time": datetime.datetime.fromtimestamp(run.info.end_time / 1000, tz=datetime.timezone.utc) if run.info.end_time else None,
            }
            # Tags: always keep
            for k, v in run.data.tags.items():
                row[f"tags.{k}"] = v
            # Metrics: filter early
            for k, v in run.data.metrics.items():
                col = f"metrics.{k}"
                if keep_metrics_list is None or any(col.startswith(p) for p in keep_metrics_list):
                    row[col] = v
            # Params: filter early
            for k, v in run.data.params.items():
                col = f"params.{k}"
                if keep_param_prefixes is None or any(col.startswith(p) for p in keep_param_prefixes):
                    row[col] = v
            rows.append(row)

        batch = pl.DataFrame(rows, infer_schema_length=len(rows))
        total += len(batch)
        _logger.info("Fetched batch of %d runs (%d total)", len(page), total)
        yield batch

        page_token = page.token
        if not page_token:
            break


def _fetch_runs(
    filter_string: str,
    keep_metrics: bool | tuple[str, ...],
    keep_params: bool | tuple[str, ...],
) -> pl.DataFrame:
    """Fetch all runs from MLFlow, filtering columns early to limit memory usage."""
    batches = list(_iter_run_batches(filter_string, keep_metrics, keep_params))
    if not batches:
        _logger.info("No runs found")
        return pl.DataFrame()
    result = pl.concat(batches, how="diagonal")
    _logger.info("Fetched %d runs with %d columns", len(result), len(result.columns))
    return result


@st.cache_data(ttl=ST_TTL_SEC, max_entries=20, persist="disk")
def latest_runs(
    keep_metrics: bool | tuple[str, ...] = True,
    keep_params: bool | tuple[str, ...] = True,
) -> pl.DataFrame:
    """
    Only returns the latest run for each root run id and stage, to avoid duplicates
    retries and chained runs.
    Returns only specified metrics and params to reduce memory usage.
    """
    _logger.info("Downloading latest runs from MLFlow")
    runs_pdf = all_runs(keep_metrics=keep_metrics, keep_params=keep_params, latest_runs=True)
    runs_pdf = runs_pdf.filter(pl.col("tags.stage").is_in(all_stages))

    latest_run_by_exp = (
        runs_pdf.sort(by="start_time", descending=True)
        .group_by(["tags.run_id", "tags.stage"])
        .agg(pl.col("*").last())
        .sort(by="tags.run_id")
    )
    _logger.info("Number of latest runs: %d", len(latest_run_by_exp))
    return latest_run_by_exp


@st.cache_data(ttl=ST_TTL_SEC, max_entries=20, persist="disk")
def all_runs(
    keep_metrics: bool | tuple[str, ...],
    keep_params: bool | tuple[str, ...],
    latest_runs: bool = False,
) -> pl.DataFrame:
    """Download all runs, filtering columns per batch to limit memory usage."""
    _logger.info("Downloading all runs from MLFlow")
    month_ago_ts = int((datetime.datetime.now() - datetime.timedelta(days=30)).timestamp() * 1000)
    filter_string = f"attributes.start_time >= {month_ago_ts} " if latest_runs else ""
    return _fetch_runs(
        filter_string=filter_string,
        keep_metrics=keep_metrics,
        keep_params=keep_params,
    )


@st.cache_data(ttl=ST_TTL_SEC, persist="disk")
def metric_counts() -> pl.DataFrame:
    """Count how many runs have each metric, streaming batch-by-batch to limit memory.

    Returns a DataFrame with columns: metric, example_run_id, count.
    """
    _logger.info("Computing metric counts (streaming)")
    month_ago_ts = int((datetime.datetime.now() - datetime.timedelta(days=30)).timestamp() * 1000)
    # We need all metrics but no params
    batches_iter = _iter_run_batches(
        filter_string=f"attributes.start_time >= {month_ago_ts}",
        keep_metrics=True,
        keep_params=False,
    )

    # Accumulate partial aggregations per batch
    # For each metric: (example_run_id, count)
    acc: dict[str, tuple[str, int]] = {}

    for batch in batches_iter:
        metric_cols = [c for c in batch.columns if c.startswith("metrics.")]
        if not metric_cols or "tags.run_id" not in batch.columns:
            continue
        unpivoted = (
            batch.select(["tags.run_id"] + metric_cols)
            .unpivot(metric_cols, index="tags.run_id", variable_name="metric")
            .drop_nulls()
            .group_by("metric")
            .agg(
                pl.col("tags.run_id").first().alias("example_run_id"),
                pl.col("tags.run_id").count().alias("count"),
            )
        )
        for row in unpivoted.iter_rows(named=True):
            metric = row["metric"]
            if metric in acc:
                _, prev_count = acc[metric]
                acc[metric] = (acc[metric][0], prev_count + row["count"])
            else:
                acc[metric] = (row["example_run_id"], row["count"])

    if not acc:
        return pl.DataFrame(schema={"metric": pl.String, "example_run_id": pl.String, "count": pl.Int64})

    result = pl.DataFrame(
        [{"metric": k, "example_run_id": v[0], "count": v[1]} for k, v in acc.items()]
    ).sort(by="count", descending=True)
    _logger.info("Found %d distinct metrics", len(result))
    return result
