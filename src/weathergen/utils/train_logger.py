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
import os.path
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from weathergen.utils.config import Config

_weathergen_timestamp = "weathergen.timestamp"
_weathergen_reltime = "weathergen.reltime"
_weathergen_time = "weathergen.time"
_performance_gpu = "perf.gpu"
_performance_memory = "perf.memory"

_logger = logging.getLogger(__name__)

Stage = Literal["train", "val"]
RunId = str


class TrainLogger:
    #######################################
    def __init__(self, cf, path_run) -> None:
        self.cf = cf
        self.path_run = path_run

    def log_metrics(self, stage: Stage, metrics: dict[str, float]) -> None:
        """
        Log metrics to a file.
        For now, just scalar values are expected. There is no check.
        """
        # Clean all the metrics to convert to float. Any other type (numpy etc.) will trigger a serialization error.
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
        with open(os.path.join(self.path_run, "metrics.json"), "ab") as f:
            s = json.dumps(clean_metrics) + "\n"
            f.write(s.encode("utf-8"))

    #######################################
    def add_train(self, samples, lr, loss_avg, stddev_avg, perf_gpu=0.0, perf_mem=0.0) -> None:
        """
        Log training data
        """

        metrics = dict(num_samples=samples)

        log_vals = [int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))]
        log_vals += [samples]

        metrics["loss_avg_0_mean"] = loss_avg[0].mean()
        metrics["learning_rate"] = lr
        metrics["num_samples"] = int(samples)
        log_vals += [loss_avg[0].mean()]
        log_vals += [lr]

        for i_obs, st in enumerate(self.cf.streams):
            st_name = _clean_name(st["name"])
            for j, (lf_name, _) in enumerate(self.cf.loss_fcts):
                lf_name = _clean_name(lf_name)
                metrics[_key_loss(st_name, lf_name)] = loss_avg[j, i_obs]
                if len(stddev_avg) > 0:
                    metrics[_key_stddev(st_name)] = stddev_avg[i_obs]
                log_vals += [loss_avg[j, i_obs]]
        if len(stddev_avg) > 0:
            for i_obs, _rt in enumerate(self.cf.streams):
                log_vals += [stddev_avg[i_obs]]

        with open(self.path_run + self.cf.run_id + "_train_log.txt", "ab") as f:
            np.savetxt(f, log_vals)

        log_vals = []
        log_vals += [perf_gpu]
        log_vals += [perf_mem]
        if perf_gpu > 0.0:
            metrics[_performance_gpu] = perf_gpu
        if perf_mem > 0.0:
            metrics[_performance_memory] = perf_mem
        self.log_metrics("train", metrics)
        with open(self.path_run + self.cf.run_id + "_perf_log.txt", "ab") as f:
            np.savetxt(f, log_vals)

    #######################################
    def add_val(self, samples, loss_avg, stddev_avg) -> None:
        """
        Log validation data
        """

        metrics = dict(num_samples=int(samples))

        log_vals = [int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))]
        log_vals += [samples]

        for i_obs, st in enumerate(self.cf.streams):
            for j, (l_n, _) in enumerate(self.cf.loss_fcts):
                metrics[_key_loss(st["name"], l_n)] = loss_avg[j, i_obs]
                log_vals += [loss_avg[j, i_obs]]
        if len(stddev_avg) > 0:
            for i_obs, st in enumerate(self.cf.streams):
                metrics[_key_stddev(st["name"])] = stddev_avg[i_obs]
                log_vals += [stddev_avg[i_obs]]

        self.log_metrics("val", metrics)
        with open(self.path_run + self.cf.run_id + "_val_log.txt", "ab") as f:
            np.savetxt(f, log_vals)

    #######################################
    @staticmethod
    def read(run_id, epoch=-1):
        """
        Read data for run_id
        """

        cf = Config.load(run_id, epoch)
        run_id = cf.run_id

        fname_log_train = f"./results/{run_id}/{run_id}_train_log.txt"
        fname_log_val = f"./results/{run_id}/{run_id}_val_log.txt"
        fname_perf_val = f"./results/{run_id}/{run_id}_perf_log.txt"
        # fname_config = f"./models/model_{run_id}.json"

        # training

        # define cols for training
        cols_train = ["dtime", "samples", "mse", "lr"]
        cols1 = [_weathergen_timestamp, "num_samples", "loss_avg_0_mean", "learning_rate"]
        for si in cf.streams:
            for _j, lf in enumerate(cf.loss_fcts):
                cols1 += [_key_loss(si["name"], lf[0])]
                cols_train += [
                    si["name"].replace(",", "").replace("/", "_").replace(" ", "_") + ", " + lf[0]
                ]
        with_stddev = [(True if "stats" in lf else False) for lf in cf.loss_fcts]
        if with_stddev:
            for si in cf.streams:
                cols1 += [_key_stddev(si["name"])]
                cols_train += [
                    si["name"].replace(",", "").replace("/", "_").replace(" ", "_")
                    + ", "
                    + "stddev"
                ]
        print(cols_train, cols1)
        # read training log data
        try:
            with open(fname_log_train, "rb") as f:
                log_train = np.loadtxt(f, delimiter=",")
            log_train = log_train.reshape((log_train.shape[0] // len(cols_train), len(cols_train)))
        except:
            print(f"Warning: no training data loaded for run_id={run_id}")
            log_train = np.array([])

        log_train_df = read_metrics(cf, run_id, "train", cols1)

        # validation
        # define cols for validation
        cols_val = ["dtime", "samples"]
        cols2 = [_weathergen_timestamp, "num_samples"]
        for si in cf.streams:
            for _j, lf in enumerate(cf.loss_fcts_val):
                print(si["name"])
                cols_val += [
                    si["name"].replace(",", "").replace("/", "_").replace(" ", "_") + ", " + lf[0]
                ]
                cols2 += [_key_loss(si["name"], lf[0])]
        with_stddev = [(True if "stats" in lf else False) for lf in cf.loss_fcts_val]
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
        except:
            print(f"Warning: no validation data loaded for run_id={run_id}")
            log_val = np.array([])
        metrics_val_df = read_metrics(cf, run_id, "val", cols2)

        # performance
        # define cols for performance monitoring
        cols_perf = ["GPU", "memory"]
        # read perf log data
        try:
            with open(fname_perf_val, "rb") as f:
                log_perf = np.loadtxt(f, delimiter=",")
            log_perf = log_perf.reshape((log_perf.shape[0] // len(cols_perf), len(cols_perf)))
        except:
            print(f"Warning: no performance data loaded for run_id={run_id}")
            log_perf = np.array([])
        metrics_system_df = read_metrics(
            cf, run_id, None, [_weathergen_timestamp, _performance_gpu, _performance_memory]
        )

        return Metrics(run_id, "train", log_train_df, metrics_val_df, metrics_system_df)


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


def read_metrics(
    cf: Config, run_id: RunId | None, stage: Stage | None, cols: list[str] | None
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

    # TODO: this should be a config option
    df = pl.read_ndjson(f"./results/{run_id}/metrics.json")
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
    _logger.info(f"schema {df.schema}")

    if columns:
        df = df.select(columns)
        # Remove all rows where all columns are null
        df = df.filter(~pl.all_horizontal(pl.col(c).is_null() for c in columns))

    return df


def _clean_name(n: str) -> str:
    """Cleans the stream name to only retain alphanumeric characters"""
    return "".join([c for c in n if c.isalnum()]).lower()


def _key_loss(st_name: str, lf_name: str) -> str:
    st_name = _clean_name(st_name)
    lf_name = _clean_name(lf_name)
    return f"stream.{st_name}.loss_{lf_name}.loss_avg"


def _key_stddev(st_name: str) -> str:
    st_name = _clean_name(st_name)
    return f"stream.{st_name}.stddev_avg"


if __name__ == "__main__":
    # uv run python -m weathergen.utils.train_logger
    TrainLogger.read("e6ev2f14")
