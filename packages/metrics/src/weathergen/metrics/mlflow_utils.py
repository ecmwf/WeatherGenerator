import logging
import os

import mlflow
import mlflow.client
import numpy as np
from mlflow.client import MlflowClient
from mlflow.entities.metric import Metric
from mlflow.entities.run import Run
from xarray import DataArray

from weathergen.common.config import Config
from weathergen.common.platform_env import get_platform_env

_logger = logging.getLogger(__name__)

project_name = "WeatherGenerator"
project_lifecycle = "dev"


class MlFlowUpload:
    tracking_uri = "databricks"
    registry_uri = "databricks-uc"
    experiment_name = "/Shared/weathergen-dev/core-model/defaultExperiment"

    experiment_tags = {
        "project": project_name,
        "lifecycle": project_lifecycle,
    }

    @classmethod
    def run_tags(cls, run_id: str, phase: str, from_run_id: str | None) -> dict[str, str]:
        """
        Returns the tags to be set for a run.
        """
        # Directly calling get_platform_env() here because it may not be available at import time.
        dct = {
            "lifecycle": project_lifecycle,
            "hpc": get_platform_env().get_hpc() or "unknown",
            "run_id": run_id,
            "stage": phase,
            "project": project_name,
            "uploader": get_platform_env().get_hpc_user() or "unknown",
            "completion_status": "success",
        }
        if from_run_id:
            dct["from_run_id"] = from_run_id
        return dct


def log_metrics(
    metrics: list[dict[str, float | int]],
    mlflow_client: MlflowClient,
    mlflow_run_id: str,
):
    """
    Logs the metrics to MLFlow.
    """
    if not metrics:
        return

    # Converts teh metrics to a single batch of metrics object. This limits the IO and DB calls
    def _convert_to_mlflow_metric(dct):
        # Convert the metric to a mlflow metric
        ts = int(dct.get("weathergen.timestamp", 0))
        step = int(dct.get("weathergen.step", 0))
        return [
            Metric(key=k, value=v, timestamp=ts, step=step)
            for k, v in dct.items()
            if not k.startswith("weathergen.")
        ]

    mlflow_metrics = [met for dct in metrics for met in _convert_to_mlflow_metric(dct)]
    mlflow_client.log_batch(
        run_id=mlflow_run_id,
        metrics=mlflow_metrics,
    )


def log_scores(
    metrics_dict: dict[str, dict[str, dict[str, DataArray]]],
    mlflow_client: MlflowClient,
    mlflow_run_id: str,
    channels_set: list[str],
    x_dim="forecast_step",
):
    """
    Logs the evaluation scores to MLFlow.
    metrics_dict: metric -> region -> stream -> DataArray
    """

    ts = 0

    mlflow_metrics = []
    for metric, regions_dict in metrics_dict.items():
        for region, streams_dict in regions_dict.items():
            for stream, data in streams_dict.items():
                for ch in channels_set:
                    # skip if channel is missing or contains NaN
                    if ch not in np.atleast_1d(data.channel.values) or data.isnull().all():
                        _logger.info(
                            f"Skipping channel {ch} for {metric} - {region} - {stream} ",
                            "due to missing data.",
                        )
                        continue
                    _logger.info(f"Collecting data for {metric} - {region} - {stream} - {ch}.")
                    data_ch = data.sel(channel=ch)
                    non_zero_dims = [
                        dim for dim in data_ch.dims if dim != x_dim and data_ch[dim].shape[0] > 1
                    ]
                    if "ens" in non_zero_dims:
                        _logger.info("Uploading ensembles not yet imnplemented")
                    else:
                        if non_zero_dims:
                            _logger.info(
                                f"LinePlot:: Found multiple entries for dimensions: {non_zero_dims}"
                                + ". Averaging..."
                            )
                        averaged = data_ch.mean(
                            dim=[dim for dim in data_ch.dims if dim != x_dim], skipna=True
                        ).sortby(x_dim)
                        label = f"score.{region}.{metric}.{stream}.{ch}"

                        mlflow_metrics.append(
                            [
                                Metric(key=label, value=y, timestamp=ts, step=int(x))
                                for x, y in zip(
                                    averaged[x_dim].values, averaged.values, strict=False
                                )
                            ]
                        )

    all_metrics = [met for dict in mlflow_metrics for met in dict]
    _logger.info(f"Logging total of {len(all_metrics)} metrics to MLFlow.")
    mlflow_client.log_batch(
        run_id=mlflow_run_id,
        metrics=all_metrics,
    )


def setup_mlflow(private_config: Config | None) -> MlflowClient:
    if private_config is None:
        assert os.environ.get("DATABRICKS_HOST") is not None, "DATABRICKS_HOST not set"
        assert os.environ.get("DATABRICKS_TOKEN") is not None, "DATABRICKS_TOKEN not set"
    else:
        os.environ["DATABRICKS_HOST"] = private_config["mlflow"]["tracking_uri"]
        os.environ["DATABRICKS_TOKEN"] = private_config["secrets"]["mlflow_token"]
    mlflow.set_tracking_uri(MlFlowUpload.tracking_uri)
    mlflow.set_registry_uri(MlFlowUpload.registry_uri)
    mlflow_client = mlflow.client.MlflowClient(
        tracking_uri=MlFlowUpload.tracking_uri, registry_uri=MlFlowUpload.registry_uri
    )
    return mlflow_client


def get_experiment_id(private_config: Config | None) -> str:
    client = setup_mlflow(private_config)
    exp = client.get_experiment_by_name(MlFlowUpload.experiment_name)
    assert exp is not None
    return exp.experiment_id


def get_or_create_mlflow_parent_run(mlflow_client: MlflowClient, run_id: str) -> Run:
    exp_name = MlFlowUpload.experiment_name
    _logger.info(f"Setting experiment name to {exp_name}: host: {os.environ['DATABRICKS_HOST']}")
    exp = mlflow.set_experiment(exp_name)
    _logger.info(f"Experiment {exp_name} created with ID {exp.experiment_id}: {exp}")
    runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.run_id='{run_id}' AND tags.stage='unknown'",
    )
    if len(runs) == 0:
        _logger.info(f"No existing parent run found for run_id {run_id}, creating new run")
        return mlflow_client.create_run(
            experiment_id=exp.experiment_id,
            tags=MlFlowUpload.run_tags(run_id, "unknown", from_run_id=None),
            run_name=run_id,
        )
    if len(runs) > 1:
        _logger.warning(
            (
                f"Multiple existing parent runs found for run_id {run_id},",
                f" using the first one: {runs[0].info.run_id}",
            )
        )
    return runs[0]
