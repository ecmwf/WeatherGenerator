import logging
import os

import mlflow
import mlflow.client
from mlflow.entities.metric import Metric
from mlflow.entities.run import Run
from mlflow.client import MlflowClient
from mlflow.entities.param import Param

from weathergen.common.platform_env import get_platform_env
from weathergen.common.config import Config


_logger = logging.getLogger(__name__)

project_name = "WeatherGenerator"
project_lifecycle = "dev"

_platform_env = get_platform_env()


class MlFlowUpload:
    tracking_uri = "databricks"
    registry_uri = "databricks-uc"
    experiment_name = "/Shared/weathergen-dev/core-model/defaultExperiment"

    experiment_tags = {
        "project": project_name,
        "lifecycle": project_lifecycle,
    }

    @classmethod
    def run_tags(cls, run_id: str, phase: str) -> dict[str, str]:
        """
        Returns the tags to be set for a run.
        """
        return {
            "lifecycle": project_lifecycle,
            "hpc": _platform_env.get_hpc() or "unknown",
            "run_id": run_id,
            "stage": phase,
            "project": project_name,
            "uploader": _platform_env.get_hpc_user() or "unknown",
            "completion_status": "success",
        }


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

    mlflow_metrics = [
        met for dct in metrics for met in _convert_to_mlflow_metric(dct)
    ]
    mlflow_client.log_batch(
        run_id=mlflow_run_id,
        metrics=mlflow_metrics,
    )


def setup_mlflow(private_config: Config) -> MlflowClient:
    os.environ["DATABRICKS_HOST"] = private_config["mlflow"]["tracking_uri"]
    os.environ["DATABRICKS_TOKEN"] = private_config["secrets"]["mlflow_token"]
    mlflow.set_tracking_uri(MlFlowUpload.tracking_uri)
    mlflow.set_registry_uri(MlFlowUpload.registry_uri)
    mlflow_client = mlflow.client.MlflowClient(
        tracking_uri=MlFlowUpload.tracking_uri, registry_uri=MlFlowUpload.registry_uri
    )
    return mlflow_client

def get_or_create_mlflow_parent_run(mlflow_client: MlflowClient, run_id: str) -> Run:
    exp_name = MlFlowUpload.experiment_name
    _logger.info(
        f"Setting experiment name to {exp_name}: host: {os.environ['DATABRICKS_HOST']}"
    )
    exp = mlflow.set_experiment(exp_name)
    _logger.info(f"Experiment {exp_name} created with ID {exp.experiment_id}: {exp}")
    l = mlflow_client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"tags.run_id='{run_id}' AND tags.stage='unknown'")
    if len(l) == 0:
        _logger.info(f"No existing parent run found for run_id {run_id}, creating new run")
        return mlflow_client.create_run(
            experiment_id=exp.experiment_id,
            tags=MlFlowUpload.run_tags(run_id, "unknown"),
            run_name=run_id,
            )
    if len(l) > 1:
        _logger.warning(f"Multiple existing parent runs found for run_id {run_id}, using the first one: {l[0].info.run_id}")
    return l[0]
