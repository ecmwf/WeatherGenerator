# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Evaluation-side MLFlow orchestration.

This module contains only the MLFlow plumbing used by the evaluation CLI
(``run_evaluation.py``): setting up the client from the HPC private config and
uploading the merged scores. The generic MLFlow helpers (``MlFlowUpload``,
``setup_mlflow``, ``get_or_create_mlflow_parent_run``, ``log_scores``) stay in
``weathergen.metrics.mlflow_utils`` because they are shared with the dashboard.
"""

import logging
from collections import defaultdict
from pathlib import Path

import mlflow
from mlflow.client import MlflowClient
from omegaconf import DictConfig, OmegaConf
from weathergen.common.platform_env import get_platform_env
from weathergen.metrics.mlflow_utils import (
    MlFlowUpload,
    get_or_create_mlflow_parent_run,
    log_scores,
    setup_mlflow,
)

from weathergen.evaluate.io.wegen_reader import WeatherGenReader
from weathergen.evaluate.plotting.plot_utils import collect_channels
from weathergen.evaluate.utils.dict_utils import triple_nested_dict

_logger = logging.getLogger(__name__)


def setup_mlflow_client() -> MlflowClient:
    """Set up the MLFlow client from the HPC private config."""
    hpc_conf = get_platform_env().get_hpc_config()
    assert hpc_conf is not None, "No HPC config found; cannot set up MLFlow."

    private_home = Path(hpc_conf)
    private_cf = OmegaConf.load(private_home)
    assert isinstance(private_cf, DictConfig)

    mlflow_client = setup_mlflow(private_cf)
    _logger.info(f"MLFlow client set up: {mlflow_client}")
    return mlflow_client


def upload_scores_to_mlflow(
    mlflow_client: MlflowClient,
    runs: dict,
    scores_dict: dict,
    private_paths: dict[str, str],
    phase: str = "eval",
) -> None:
    """Upload the merged scores of all runs to MLFlow.

    ``scores_dict`` is nested ``[metric][region][stream][run_id] -> DataArray``.
    """
    # Reorder to [run_id][metric][region][stream] so each run is uploaded once.
    reordered_dict = defaultdict(triple_nested_dict)
    for metric, regions_dict in scores_dict.items():
        for region, streams_dict in regions_dict.items():
            for stream, runs_dict in streams_dict.items():
                for run_id, data in runs_dict.items():
                    reordered_dict[run_id][metric][region][stream] = data

    channels_set = collect_channels(scores_dict, metric, region, runs)

    for run_id, run in runs.items():
        reader = WeatherGenReader(run, run_id, private_paths)
        from_run_id = reader.inference_cfg["from_run_id"]
        parent_run = get_or_create_mlflow_parent_run(mlflow_client, from_run_id)
        _logger.info(f"MLFlow parent run: {parent_run}")

        with mlflow.start_run(run_id=parent_run.info.run_id):
            with mlflow.start_run(
                run_name=f"{phase}_{from_run_id}_{run_id}",
                parent_run_id=parent_run.info.run_id,
                nested=True,
            ) as mlflow_run:
                mlflow.set_tags(MlFlowUpload.run_tags(run_id, phase, from_run_id))

                log_scores(
                    reordered_dict[run_id],
                    mlflow_client,
                    mlflow_run.info.run_id,
                    channels_set,
                )
