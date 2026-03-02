# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import logging
import math
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

import weathergen.common.config as config

# from weathergen.train.trainer import cfg_keys_to_filter
from weathergen.train.utils import Stage, cfg_keys_to_filter, flatten_dict, get_active_stage_config
from weathergen.utils.distributed import ddp_average
from weathergen.utils.metrics import get_train_metrics_path, read_metrics_file
from weathergen.utils.utils import is_stream_forcing

_weathergen_timestamp = "weathergen.timestamp"
_weathergen_reltime = "weathergen.reltime"
_weathergen_time = "weathergen.time"
_performance_gpu = "perf.gpu"
_performance_memory = "perf.memory"

_logger = logging.getLogger(__name__)

RunId = str


@dataclass
class Metrics:
    run_id: RunId
    stage: Stage
    train: pl.DataFrame
    val: pl.DataFrame
    system: pl.DataFrame

    def by_mode(self, s: str) -> pl.DataFrame:
        match s:
            case "train":
                return self.train
            case "val":
                return self.val
            case "system":
                return self.system
            case _:
                raise ValueError(f"Unknown mode {s}. Use 'train', 'val' or 'system'.")


class TrainLogger:
    #######################################
    def __init__(self, cf, path_run: Path) -> None:
        self.cf = cf
        self.path_run = path_run

    def log_metrics(self, stage: Stage, metrics: dict[str, float]) -> None:
        """
        Log metrics to a file.
        For now, just scalar values are expected. There is no check.
        """
        ## Clean all the metrics to convert to float.
        #  Any other type (numpy etc.) will trigger a serialization error.
        clean_metrics = {
            _weathergen_timestamp: time.time_ns() // 1_000_000,
            _weathergen_time: int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
            "stage": stage,
        }
        for key, value in metrics.items():
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                v = str(v)
            clean_metrics[key] = v

        # TODO: performance: we repeatedly open the file for each call. Better for multiprocessing
        # but we can probably do better and rely for example on the logging module.

        metrics_path = get_train_metrics_path(
            base_path=config.get_path_run(self.cf), run_id=self.cf.general.run_id
        )
        with open(metrics_path, "ab") as f:
            s = json.dumps(clean_metrics) + "\n"
            f.write(s.encode("utf-8"))

    #######################################
    def add_logs(
        self,
        stage: Stage,
        samples: int,
        losses_all: dict,
        stddev_all: dict,
        avg_loss: list[float] = None,
        lr: float = None,
        perf_gpu: float = 0.0,
        perf_mem: float = 0.0,
    ) -> None:
        """
        Log training or validation data
        """
        metrics: dict[str, float] = dict(num_samples=samples)

        if stage == "train":
            metrics["loss_avg_mean"] = np.nanmean(avg_loss)
            metrics["learning_rate"] = lr
            metrics["num_samples"] = int(samples)
            metrics[_performance_gpu] = perf_gpu
            metrics[_performance_memory] = perf_mem

        for key, value in losses_all.items():
            metrics[key] = np.nanmean(value)

        for key, value in stddev_all.items():
            metrics[key] = np.nanmean(value)

        self.log_metrics(stage, metrics)

    #######################################
    @staticmethod
    def read(
        run_id: str,
        model_path: str = None,
        mini_epoch: int | None = None,
        cols_patterns: list[str] | None = None,
    ) -> Metrics:
        """
        Read data for run_id
        """

        # Load config from given model_path if provided, otherwise use path from private config
        if model_path:
            cf = config.load_run_config(run_id=run_id, mini_epoch=mini_epoch, model_path=model_path)
        else:
            cf = config.load_merge_configs(
                private_home=None, from_run_id=run_id, mini_epoch=mini_epoch
            )
        run_id = cf.general.run_id

        result_dir_base = config.get_path_run(cf)
        result_dir = result_dir_base / run_id
        fname_log_train = result_dir / f"{run_id}_train_log.txt"
        fname_log_val = result_dir / f"{run_id}_val_log.txt"

        # training

        # define cols for training
        training_cfg = get_active_stage_config(cf.training_config, {}, cfg_keys_to_filter)
        cols1, cols_train = get_loss_terms_per_stream(cf.streams, training_cfg)
        cols_train += ["dtime", "samples", "mse", "lr"]
        cols1 += [_weathergen_timestamp, "num_samples", "loss_avg_mean", "learning_rate"]
        cols1_patterns = ["loss_avg"] + cols_patterns

        # read training log data
        try:
            with open(fname_log_train, "rb") as f:
                log_train = np.loadtxt(f, delimiter=",")
            log_train = log_train.reshape((log_train.shape[0] // len(cols_train), len(cols_train)))
        except (
            TypeError,
            AttributeError,
            IndexError,
            ZeroDivisionError,
            ValueError,
        ) as e:
            _logger.warning(
                (
                    f"Warning: no training data loaded for run_id={run_id}",
                    "Data loading or reshaping failed — "
                    "possible format, dimension, or logic issue.",
                    f"Due to specific error: {e}",
                )
            )
        except (FileNotFoundError, PermissionError, OSError) as e:
            _logger.error(
                (
                    f"Error: no training data loaded for run_id={run_id}",
                    "File system error occurred while handling the log file.",
                    f"Due to specific error: {e}",
                )
            )
        except Exception:
            _logger.error(
                (
                    f"Error: no training data loaded for run_id={run_id}",
                    f"Due to exception with trace:\n{traceback.format_exc()}",
                )
            )
            log_train = np.array([])

        log_train_df = read_metrics(cf, run_id, "train", cols1, cols1_patterns, result_dir_base)

        # validation
        # define cols for validation
        validation_cfg = get_active_stage_config(
            training_cfg, cf.get("validation_config", {}), cfg_keys_to_filter
        )
        cols2, cols_val = get_loss_terms_per_stream(cf.streams, validation_cfg)
        cols_val = ["dtime", "samples"]
        cols2 += [_weathergen_timestamp, "num_samples"]
        cols2_patterns = ["loss_avg"] + cols_patterns

        # read validation log data
        try:
            with open(fname_log_val, "rb") as f:
                log_val = np.loadtxt(f, delimiter=",")
            log_val = log_val.reshape((log_val.shape[0] // len(cols_val), len(cols_val)))
        except (
            TypeError,
            AttributeError,
            IndexError,
            ZeroDivisionError,
            ValueError,
        ) as e:
            _logger.warning(
                (
                    f"Warning: no validation data loaded for run_id={run_id}",
                    "Data loading or reshaping failed — "
                    "possible format, dimension, or logic issue.",
                    f"Due to specific error: {e}",
                )
            )
        except (FileNotFoundError, PermissionError, OSError) as e:
            _logger.error(
                (
                    f"Error: no validation data loaded for run_id={run_id}",
                    "File system error occurred while handling the log file.",
                    f"Due to specific error: {e}",
                )
            )
        except Exception:
            _logger.error(
                (
                    f"Error: no validation data loaded for run_id={run_id}",
                    f"Due to exception with trace:\n{traceback.format_exc()}",
                )
            )
            log_val = np.array([])
        metrics_val_df = read_metrics(cf, run_id, "val", cols2, cols2_patterns, result_dir_base)

        return Metrics(run_id, "train", log_train_df, metrics_val_df, None)


def read_metrics(
    cf: config.Config,
    run_id: RunId | None,
    stage: Stage | None,
    cols: list[str] | None,
    cols_patterns: list[str] | None,
    results_path: Path,
) -> pl.DataFrame:
    """
    Read metrics for run_id

    stage: stage to load ("train", "val" or empty). If None, all stages are loaded.
    cols: list of columns to load. If None, all columns are loaded.
    run_id: run_id to load. If None, the run_id form the config is used.
    """

    assert cols is None or cols, "cols must be non empty or None"
    if run_id is None:
        run_id = cf.general.run_id
    assert run_id, "run_id must be provided"

    metrics_path = get_train_metrics_path(base_path=results_path, run_id=run_id)
    # TODO: this should be a config option
    df = read_metrics_file(metrics_path)

    if cols_patterns is not None:
        for col_pattern in cols_patterns:
            cols += [col for col in df.columns if col_pattern in col]

    if stage is not None:
        df = df.filter(pl.col("stage") == stage)
    df = df.drop("stage")
    df = clean_df(df, cols)
    return df


def clean_df(df, columns: list[str] | None):
    """
    Selects the required data from the dataframe, and ensures thath all columns are numeric.
    """
    # Convert all string columns to float. type == str they contained nan/inf values
    for k, v in df.schema.items():
        if v == pl.String:
            df = df.with_columns(df[k].cast(pl.Float64).alias(k))

    # Convert timestamp column to date
    df = df.with_columns(
        pl.from_epoch(df[_weathergen_timestamp], time_unit="ms").alias(_weathergen_timestamp)
    )
    df = df.with_columns(
        (df[_weathergen_timestamp] - df[_weathergen_timestamp].min()).alias(_weathergen_reltime)
    )

    if columns:
        columns = list(set(columns))  # remove duplicates
        # Backwards compatibility of "loss_avg_mean" (old) and "loss_avg_0_mean" (new) metric name
        if "loss_avg_mean" not in df.columns:
            idcs = [i for i in range(len(columns)) if columns[i] == "loss_avg_mean"]
            if len(idcs) > 0:
                columns[idcs[0]] = "loss_avg_0_mean"
        df = df.select(columns)
        # Remove all rows where all columns are null
        df = df.filter(~pl.all_horizontal(pl.col(c).is_null() for c in columns))

    return df


def clean_name(s: str) -> str:
    """
    Remove all characters from a string except letters, digits, and underscores.

    Args:
        s (str): The input string.

    Returns:
        str: A new string containing only alphanumeric characters and underscores,
             in the same order and capitalization as they appeared in the input.
    """
    return "".join(c for c in s if c.isalnum() or c == "-" or c == "_")


def get_loss_terms_per_stream(streams, stage_config):
    """
    Extract per stream loss terms
    """
    cols, cols_stage = [], []
    for si in streams:
        if is_stream_forcing(si):
            continue
        for _, loss_config in stage_config.get("losses", {}).items():
            if loss_config.get("type", "LossPhysical") == "LossPhysical":
                for lname, _ in loss_config.loss_fcts.items():
                    cols += [_key_loss(si["name"], lname)]
                    cols_stage += [_clean_stream_name(si["name"]) + lname]
    return cols, cols_stage


def _clean_stream_name(stream_name: str) -> str:
    return stream_name.replace(",", "").replace("/", "_").replace(" ", "_") + ", "


def _key_loss(st_name: str, lf_name: str) -> str:
    st_name = clean_name(st_name)
    return f"LossPhysical.{st_name}.{lf_name}.avg"


def _key_loss_chn(st_name: str, lf_name: str, ch_name: str) -> str:
    st_name = clean_name(st_name)
    return f"stream.{st_name}.loss_{lf_name}.loss_{ch_name}"


def _key_stddev(st_name: str) -> str:
    st_name = clean_name(st_name)
    return f"stream.{st_name}.stddev_avg"


def prepare_losses_for_logging(
    loss_hist: list,
    losses_unweighted_hist: list[dict],
    stddev_unweighted_hist: list[dict],
) -> tuple[list, dict, dict]:
    """
    Aggregates across ranks loss and standard deviation data for logging.

    Returns:
        real_loss (list): List of ddp-averaged scaler losses used for backpropagation.
        losses_all (dict): Dictionary mapping each stream name to its
            per-channel loss tensor.
        stddev_all (dict): Dictionary mapping each stream name to its
            per-channel standard deviation tensor.
    """

    real_loss = [ddp_average(loss).item() for loss in loss_hist]

    losses_all = defaultdict(list)
    stddev_all = defaultdict(list)

    for d in losses_unweighted_hist:
        for key, value in flatten_dict(d).items():
            value = torch.tensor(value, device="cuda") if type(value) is float else value
            losses_all[key].append(ddp_average(value).item())

    for d in stddev_unweighted_hist:
        for key, value in flatten_dict(d).items():
            if value:
                stddev_all[key].append(ddp_average(value).item())

    return real_loss, losses_all, stddev_all
