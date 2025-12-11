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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from torch import Tensor

import weathergen.common.config as config
from weathergen.train.utils import flatten_dict
from weathergen.utils.metrics import get_train_metrics_path, read_metrics_file

_weathergen_timestamp = "weathergen.timestamp"
_weathergen_reltime = "weathergen.reltime"
_weathergen_time = "weathergen.time"
_performance_gpu = "perf.gpu"
_performance_memory = "perf.memory"

_logger = logging.getLogger(__name__)

Stage = Literal["train", "val"]
RunId = str

# All the stages currently implemented:
TRAIN: Stage = "train"
VAL: Stage = "val"


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
            base_path=Path(self.cf.run_path), run_id=self.cf.run_id
        )
        with open(metrics_path, "ab") as f:
            s = json.dumps(clean_metrics) + "\n"
            f.write(s.encode("utf-8"))

    #######################################
    def add_train(
        self,
        samples: int,
        lr: float,
        avg_loss: list[float],
        losses_all: dict,
        stddev_all: dict,
        perf_gpu: float = 0.0,
        perf_mem: float = 0.0,
    ) -> None:
        """
        Log training data
        """
        metrics: dict[str, float] = dict(num_samples=samples)

        log_vals: list[float] = [int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))]
        log_vals += [samples]

        metrics["loss_avg_mean"] = np.nanmean(avg_loss)
        metrics["learning_rate"] = lr
        metrics["num_samples"] = int(samples)
        log_vals += [np.nanmean(avg_loss)]
        log_vals += [lr]

        for key, value in flatten_dict(losses_all).items():
            metrics[key] = np.nanmean(value)

        for key, value in flatten_dict(stddev_all).items():
            metrics[key] = np.nanmean(value)

        with open(self.path_run / f"{self.cf.run_id}_train_log.txt", "ab") as f:
            np.savetxt(f, log_vals)

        log_vals = []
        log_vals += [perf_gpu]
        log_vals += [perf_mem]
        metrics[_performance_gpu] = perf_gpu
        metrics[_performance_memory] = perf_mem
        self.log_metrics("train", metrics)
        with open(self.path_run / (self.cf.run_id + "_perf_log.txt"), "ab") as f:
            np.savetxt(f, log_vals)

    #######################################
    def add_val(
        self, samples: int, losses_all: dict[str, Tensor], stddev_all: dict[str, Tensor]
    ) -> None:
        """
        Log validation data
        """

        metrics: dict[str, float] = dict(num_samples=int(samples))

        log_vals: list[float] = [int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))]
        log_vals += [samples]

        stream_names = [st["name"] for st in self.cf.streams]

        for loss_name, loss_values in losses_all.items():
            metrics[f"loss.{loss_name}.loss_avg"] = loss_values[:, :].nanmean().item()
            st = self.cf.streams[stream_names.index(loss_name.split(".")[1])]
            for k, ch_n in enumerate(st.val_target_channels):
                metrics[f"loss.{loss_name}.{ch_n}"] = loss_values[:, k].nanmean().item()
            log_vals += [loss_values[:, :].nanmean().item()]
        for loss_name, stddev_values in stddev_all.items():
            metrics[f"loss.{loss_name}.stddev_avg"] = stddev_values.nanmean().item()
            log_vals += [stddev_values.nanmean().item()]

        self.log_metrics("val", metrics)
        with open(self.path_run / (self.cf.run_id + "_val_log.txt"), "ab") as f:
            np.savetxt(f, log_vals)

    #######################################
    @staticmethod
    def read(run_id: str, model_path: str = None, mini_epoch: int = -1) -> Metrics:
        """
        Read data for run_id
        """
        # Load config from given model_path if provided, otherwise use path from private config
        if model_path:
            cf = config.load_model_config(
                run_id=run_id, mini_epoch=mini_epoch, model_path=model_path
            )
        else:
            cf = config.load_config(private_home=None, from_run_id=run_id, mini_epoch=mini_epoch)
        run_id = cf.run_id

        result_dir_base = Path(cf.run_path)
        result_dir = result_dir_base / run_id
        fname_log_train = result_dir / f"{run_id}_train_log.txt"
        fname_log_val = result_dir / f"{run_id}_val_log.txt"
        fname_perf_val = result_dir / f"{run_id}_perf_log.txt"

        # training

        # define cols for training
        cols_train = ["dtime", "samples", "mse", "lr"]
        cols1 = [_weathergen_timestamp, "num_samples", "loss_avg_mean", "learning_rate"]
        for si in cf.streams:
            for lf in cf.loss_fcts:
                cols1 += [_key_loss(si["name"], lf[0])]
                cols_train += [
                    si["name"].replace(",", "").replace("/", "_").replace(" ", "_") + ", " + lf[0]
                ]
        with_stddev = [("stats" in lf) for lf in cf.loss_fcts]
        if with_stddev:
            for si in cf.streams:
                cols1 += [_key_stddev(si["name"])]
                cols_train += [
                    si["name"].replace(",", "").replace("/", "_").replace(" ", "_")
                    + ", "
                    + "stddev"
                ]
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

        log_train_df = read_metrics(cf, run_id, "train", cols1, result_dir_base)

        # validation
        # define cols for validation
        cols_val = ["dtime", "samples"]
        cols2 = [_weathergen_timestamp, "num_samples"]
        for si in cf.streams:
            for lf in cf.loss_fcts_val:
                cols_val += [
                    si["name"].replace(",", "").replace("/", "_").replace(" ", "_") + ", " + lf[0]
                ]
                cols2 += [_key_loss(si["name"], lf[0])]
        with_stddev = [("stats" in lf) for lf in cf.loss_fcts_val]
        if with_stddev:
            for si in cf.streams:
                cols2 += [_key_stddev(si["name"])]
                cols_val += [
                    si["name"].replace(",", "").replace("/", "_").replace(" ", "_")
                    + ", "
                    + "stddev"
                ]
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
        metrics_val_df = read_metrics(cf, run_id, "val", cols2, result_dir_base)

        # performance
        # define cols for performance monitoring
        cols_perf = ["GPU", "memory"]
        # read perf log data
        try:
            with open(fname_perf_val, "rb") as f:
                log_perf = np.loadtxt(f, delimiter=",")
            log_perf = log_perf.reshape((log_perf.shape[0] // len(cols_perf), len(cols_perf)))
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
            log_perf = np.array([])
        metrics_system_df = read_metrics(
            cf,
            run_id,
            None,
            [_weathergen_timestamp, _performance_gpu, _performance_memory],
            result_dir_base,
        )

        return Metrics(run_id, "train", log_train_df, metrics_val_df, metrics_system_df)


def read_metrics(
    cf: config.Config,
    run_id: RunId | None,
    stage: Stage | None,
    cols: list[str] | None,
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
        run_id = cf.run_id
    assert run_id, "run_id must be provided"

    metrics_path = get_train_metrics_path(base_path=results_path, run_id=run_id)
    # TODO: this should be a config option
    df = read_metrics_file(metrics_path)
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
    return "".join(c for c in s if c.isalnum() or c == "_")


def _key_loss(st_name: str, lf_name: str) -> str:
    st_name = clean_name(st_name)
    return f"stream.{st_name}.loss_{lf_name}.loss_avg"


def _key_loss_chn(st_name: str, lf_name: str, ch_name: str) -> str:
    st_name = clean_name(st_name)
    return f"stream.{st_name}.loss_{lf_name}.loss_{ch_name}"


def _key_stddev(st_name: str) -> str:
    st_name = clean_name(st_name)
    return f"stream.{st_name}.stddev_avg"
