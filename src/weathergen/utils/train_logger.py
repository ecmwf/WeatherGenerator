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
import math
import os.path
import time
from pathlib import Path

import numpy as np

from weathergen.utils.config import Config


class TrainLogger:
    #######################################
    def __init__(self, cf, path_run) -> None:
        self.cf = cf
        self.path_run = path_run

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """
        Log metrics to a file.
        For now, just scalar values are expected. There is no check.
        """
        # Clean all the metrics to convert to float. Any other type (numpy etc.) will trigger a serialization error.
        clean_metrics = {
            "weathergen.timestamp": time.time_ns() // 1_000_000,
            "weathergen.time": int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
        }
        for key, value in metrics.items():
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                v = str(v)
            clean_metrics[key] = v

        # TODO: performance: we repeatedly open the file for each call. Better for multiprocessing
        # but we can probably do better and rely for example on the logging module.
        with open(os.path.join(str(self.path_run), "metrics.json"), "ab") as f:
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
        log_vals += [loss_avg[0].mean()]
        log_vals += [lr]

        for i_obs, _rt in enumerate(self.cf.streams):
            for j, _ in enumerate(self.cf.loss_fcts):
                metrics[f"stream_{i_obs}.loss_{j}.loss_avg"] = loss_avg[j, i_obs]
                log_vals += [loss_avg[j, i_obs]]
        if len(stddev_avg) > 0:
            for i_obs, _rt in enumerate(self.cf.streams):
                log_vals += [stddev_avg[i_obs]]
                metrics[f"stream_{i_obs}.stddev_avg"] = stddev_avg[i_obs]

        with open(str(self.path_run) + self.cf.run_id + "_train_log.txt", "ab") as f:
            np.savetxt(f, log_vals)

        log_vals = []
        log_vals += [perf_gpu]
        log_vals += [perf_mem]
        if perf_gpu > 0.0:
            metrics["perf.gpu"] = perf_gpu
        if perf_mem > 0.0:
            metrics["perf.memory"] = perf_mem
        self.log_metrics(metrics)
        with open(str(self.path_run) + self.cf.run_id + "_perf_log.txt", "ab") as f:
            np.savetxt(f, log_vals)

    #######################################
    def add_val(self, samples, loss_avg, stddev_avg) -> None:
        """
        Log validation data
        """

        log_vals = [int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))]
        log_vals += [samples]

        for i_obs, _rt in enumerate(self.cf.streams):
            for j, _ in enumerate(self.cf.loss_fcts):
                log_vals += [loss_avg[j, i_obs]]
        if len(stddev_avg) > 0:
            for i_obs, _rt in enumerate(self.cf.streams):
                log_vals += [stddev_avg[i_obs]]

        with open(str(self.path_run) + self.cf.run_id + "_val_log.txt", "ab") as f:
            np.savetxt(f, log_vals)

    #######################################
    @staticmethod
    def read(run_id, epoch=-1):
        """
        Read data for run_id
        """

        cf = Config.load(run_id, epoch)
        run_id = cf.run_id

        fname_log_train = Path(cf.run_path) / run_id / f"{run_id}_train_log.txt"
        fname_log_val = Path(cf.run_path) / run_id / f"{run_id}_val_log.txt"
        fname_perf_val = Path(cf.run_path) / run_id / f"{run_id}_perf_log.txt"
        # fname_config = Path(cf.run_path) / f"model_{run_id}.json"

        # training

        # define cols for training
        cols_train = ["dtime", "samples", "mse", "lr"]
        for si in cf.streams:
            for _j, lf in enumerate(cf.loss_fcts):
                cols_train += [
                    si["name"].replace(",", "").replace("/", "_").replace(" ", "_") + ", " + lf[0]
                ]
        with_stddev = [(True if "stats" in lf else False) for lf in cf.loss_fcts]
        if with_stddev:
            for si in cf.streams:
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
        except:
            print(f"Warning: no training data loaded for run_id={run_id}")
            log_train = np.array([])

        # validation

        # define cols for validation
        cols_val = ["dtime", "samples"]
        for si in cf.streams:
            for _j, lf in enumerate(cf.loss_fcts_val):
                cols_val += [
                    si["name"].replace(",", "").replace("/", "_").replace(" ", "_") + ", " + lf[0]
                ]
        with_stddev = [(True if "stats" in lf else False) for lf in cf.loss_fcts_val]
        if with_stddev:
            for si in cf.streams:
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

        return ((cols_train, log_train), (cols_val, log_val), (cols_perf, log_perf))
