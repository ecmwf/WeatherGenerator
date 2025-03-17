# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np

from weathergen.utils.config import Config


class TrainLogger:
    #######################################
    def __init__(self, cf, path_run) -> None:
        self.cf = cf
        self.path_run = path_run
        # TODO: add header with col names (loadtxt has an option to skip k header lines)

    #######################################
    def add_train(self, samples, lr, loss_avg, stddev_avg, perf_gpu=0.0, perf_mem=0.0) -> None:
        """
        Log training data
        """

        log_vals = [int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))]
        log_vals += [samples]

        log_vals += [loss_avg[0].mean()]
        log_vals += [lr]

        for i_obs, _rt in enumerate(self.cf.streams):
            for j, _ in enumerate(self.cf.loss_fcts):
                log_vals += [loss_avg[j, i_obs]]
        if len(stddev_avg) > 0:
            for i_obs, _rt in enumerate(self.cf.streams):
                log_vals += [stddev_avg[i_obs]]

        with open(self.path_run + self.cf.run_id + "_train_log.txt", "ab") as f:
            np.savetxt(f, log_vals)

        log_vals = []
        log_vals += [perf_gpu]
        log_vals += [perf_mem]
        with open(self.path_run + self.cf.run_id + "_perf_log.txt", "ab") as f:
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

        fname_log_train = cf.run_path + f'/{run_id}/{run_id}_train_log.txt'
        fname_log_val = cf.run_path + f'/{run_id}/{run_id}_val_log.txt'
        fname_perf_val = cf.run_path + f'/{run_id}/{run_id}_perf_log.txt'
        # fname_config = cf.run_path + f'/model_{run_id}.json'

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
