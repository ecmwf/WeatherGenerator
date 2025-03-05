# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
import numpy as np
import datetime
import pandas as pd

from weathergen.utils.config import Config


class TrainLogger:
    """
    A class for logging training and validation metrics during model training.

    This logger writes metrics to CSV files with multi-level column headers,
    organizing metrics by global training statistics and per-stream channel statistics.

    Attributes:
        cf (Config): Configuration object containing training parameters.
        path_run (str): Base directory path for storing log files.
        train_step (int): Current training step counter.
        val_step (int): Current validation step counter.
        train_file (str): Full path to training log CSV file.
        val_file (str): Full path to validation log CSV file.

    Methods:
      initialize_file(stream_chans: dict, train: bool = False, val: bool = False) -> None:
          Initializes log files and their headers for training and/or validation.

      add_train(samples: int, lr: float, loss_avg: torch.Tensor, stddev_avg: torch.Tensor,
                loss_chn_avg: torch.Tensor, stream_chans: dict,
                perf_gpu: float = 0., perf_mem: float = 0.) -> None:
          Logs training metrics into the CSV file.

      add_val(samples: int, loss_avg: torch.Tensor, stddev_avg: torch.Tensor,
              loss_chn_avg: torch.Tensor, stream_chans: dict) -> None:
          Logs validation metrics into the CSV file.

      read(run_id: str, epoch: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
          Reads the training and validation log files and returns them as Pandas DataFrames.

    Args:
        cf (Config): Configuration object with training parameters.
        path_run (str): Base directory path where log files will be stored.
    """

    def __init__(self, cf: Config, path_run: str) -> None:
        """
        Initializes the TrainLogger with configuration and file paths.
        """
        self.cf = cf
        self.path_run = path_run
        self.train_step = 0
        self.val_step = 0
        self.train_file = self.path_run + self.cf.run_id + "_train_log.csv"
        self.val_file = self.path_run + self.cf.run_id + "_val_log.csv"

    def initialize_file(
        self, stream_chans: dict, train: bool = False, val: bool = False
    ) -> None:
        """
        Initializes CSV log files for training and validation.

        This method creates log files with multi-index column structures based on
        global training statistics and per-stream channel statistics. It should be called
        before logging any training or validation data.

        Args:
            stream_chans (dict): A dictionary mapping stream names to their respective output channels.
            train (bool, optional): If True, initializes the training log file. Defaults to False.
            val (bool, optional): If True, initializes the validation log file. Defaults to False.
        """
        if train:
            columns_config = {
                "global": [
                    "step",
                    "time",
                    "samples",
                    "perf_gpu",
                    "perf_mem",
                    "learning_rate",
                    "loss_mean",
                ],
            }
            for stream in stream_chans:
                columns_config[stream] = (
                    [nm[0] for j, nm in enumerate(self.cf.loss_fcts)]
                    + stream_chans[stream]
                    + ["std"]
                )

            columns = []
            for top_level, sub_columns in columns_config.items():
                columns.extend([(top_level, sub) for sub in sub_columns])

            self.train_multi_cols = pd.MultiIndex.from_tuples(columns)
            df = pd.DataFrame([], columns=self.train_multi_cols)
            df.to_csv(self.train_file, mode="w", header=True, index=False)

            self.train_step = 1

        if val:
            columns_config = {
                "global": ["step", "time", "samples", "loss_mean"],
            }
            for stream in stream_chans:
                columns_config[stream] = (
                    [nm[0] for j, nm in enumerate(self.cf.loss_fcts)]
                    + stream_chans[stream]
                    + ["std"]
                )

            columns = []
            for top_level, sub_columns in columns_config.items():
                columns.extend([(top_level, sub) for sub in sub_columns])

            self.val_multi_cols = pd.MultiIndex.from_tuples(columns)
            df = pd.DataFrame([], columns=self.val_multi_cols)
            df.to_csv(self.val_file, mode="w", header=True, index=False)

            self.val_step = 1

    #######################################
    def add_train(
        self,
        samples: int,
        lr: float,
        loss_avg: torch.Tensor,
        stddev_avg: torch.Tensor,
        loss_chn_avg: torch.Tensor,
        stream_chans: dict,
        perf_gpu: float = 0.0,
        perf_mem: float = 0.0,
    ) -> None:
        """
        Logs training metrics to CSV file with multi-index columns.

        Metrics are organized in a hierarchical structure:
        - Global metrics (step, time, samples, GPU perf, memory perf, learning rate, mean loss)
        - Per-stream channel metrics (loss components, channel-specific losses, standard deviation)

        Args:
            samples (int): Number of samples processed in this step.
            lr (float): Current learning rate.
            loss_avg (torch.Tensor): Array of average loss values across loss functions and streams.
            stddev_avg (torch.Tensor): Array of standard deviation values per stream.
            loss_chn_avg (torch.Tensor): 3D array of channel-specific losses (loss_fns x channels x streams).
            stream_chans (dict): Dictionary mapping stream names to their output channels.
            perf_gpu (float, optional): GPU utilization percentage. Defaults to 0.
            perf_mem (float, optional): GPU memory utilization percentage. Defaults to 0.

        Raises:
          AssertionError: If the logging files were not initialized before calling this method.
        """
        assert (
            self.train_step > 0
        ), "Logging files were not initialized. Call initialize_file before using add_train."

        log_vals = [self.train_step]
        log_vals += [datetime.datetime.now()]
        log_vals += [samples]
        log_vals += [perf_gpu]
        log_vals += [perf_mem]
        log_vals += [lr.item()]
        log_vals += [loss_avg[0].mean().item()]

        for i_obs, rt in enumerate(stream_chans.values()):
            for j, nm in enumerate(self.cf.loss_fcts):
                log_vals += [loss_avg[j, i_obs].item()]
            log_vals += loss_chn_avg[:, 0 : len(rt), i_obs].nanmean(dim=0).tolist()
            if len(stddev_avg) > 0:
                log_vals += [stddev_avg[i_obs].item()]
            else:
                log_vals += [None]

        df = pd.DataFrame([log_vals], columns=self.train_multi_cols)
        df.to_csv(self.train_file, mode="a", header=False, index=False)

        self.train_step += 1

    #######################################
    def add_val(
        self,
        samples: int,
        loss_avg: torch.Tensor,
        stddev_avg: torch.Tensor,
        loss_chn_avg: torch.Tensor,
        stream_chans: dict,
    ) -> None:
        """
        Logs validation metrics to CSV file with multi-index columns.

        Follows same column structure as training logs but without performance metrics.

        Args:
            samples (int): Number of samples processed in this step.
            loss_avg (torch.Tensor): Array of average loss values across loss functions and streams.
            stddev_avg (torch.Tensor): Array of standard deviation values per stream.
            loss_chn_avg (torch.Tensor): 3D array of channel-specific losses (loss_fns x channels x streams).
            stream_chans (dict): Dictionary mapping stream names to their output channels.

        Raises:
          AssertionError: If the logging files were not initialized before calling this method.
        """
        assert (
            self.val_step > 0
        ), "Logging files were not initialized. Call initialize_file before using add_val."

        log_vals = [self.val_step]
        log_vals += [datetime.datetime.now()]
        log_vals += [samples]
        log_vals += [loss_avg[0].mean().item()]

        for i_obs, rt in enumerate(stream_chans.values()):
            for j, nm in enumerate(self.cf.loss_fcts):
                log_vals += [loss_avg[j, i_obs].item()]
            log_vals += loss_chn_avg[:, 0 : len(rt), i_obs].nanmean(dim=0).tolist()
            if len(stddev_avg) > 0:
                log_vals += [stddev_avg[i_obs].item()]
            else:
                log_vals += [None]

        df = pd.DataFrame([log_vals], columns=self.val_multi_cols)
        df.to_csv(self.val_file, mode="a", header=False, index=False)

        self.val_step += 1

    #######################################
    @staticmethod
    def read(run_id, epoch=-1) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads training and validation logs from CSV files.

        Args:
            run_id (str): Unique identifier for the training run.
            epoch (int, optional): Model epoch to load. Defaults to latest (-1).

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                - df_train: Training log DataFrame with multi-index columns
                - df_val: Validation log DataFrame with multi-index columns

        Notes:
            - DataFrames use multi-index columns matching the logging structure
            - Index columns are ('step', 'time') from the original logs
        """

        cf = Config.load(run_id, epoch)
        run_id = cf.run_id

        fname_log_train = f"./results/{run_id}/{run_id}_train_log.csv"
        fname_log_val = f"./results/{run_id}/{run_id}_val_log.csv"

        df_train = pd.read_csv(
            fname_log_train,
            header=[0, 1],
            skipinitialspace=True,
            parse_dates=[("global", "time")],
        )
        df_val = pd.read_csv(
            fname_log_val,
            header=[0, 1],
            skipinitialspace=True,
            parse_dates=[("global", "time")],
        )

        return df_train, df_val
