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

class TrainLogger :
  """
  A class for logging training and validation metrics during model training.

  This logger writes metrics to CSV files with multi-level column headers, 
  organizing metrics by global training statistics and per-stream channel statistics.
  Supports distributed training environments by only initializing files on rank 0.

  Attributes:
      cf (Config): Configuration object containing training parameters.
      path_run (str): Base directory path for storing log files.
      train_step (int): Current training step counter.
      val_step (int): Current validation step counter.
      train_file (str): Full path to training log CSV file.
      val_file (str): Full path to validation log CSV file.

  Args:
      cf (Config): Configuration object with training parameters.
      path_run (str): Base directory path where log files will be stored.
  """
  def __init__( self, cf: Config, path_run: str) -> None:
    """
      Initializes the TrainLogger with configuration and file paths. Creates empty
      log files if current process rank is 0 (for distributed training setups).
    """
    self.cf = cf
    self.path_run = path_run
    self.train_step = 1
    self.val_step = 1
    self.train_file = self.path_run + self.cf.run_id + '_train_log.csv'
    self.val_file = self.path_run + self.cf.run_id + '_val_log.csv'
    if 0 == self.cf.rank:
      # Initialize log files
      with open(self.train_file, "w") as my_empty_csv:
        pass
      with open(self.val_file, "w") as my_empty_csv:
        pass

  #######################################
  def add_train(self, samples: int, lr: float, loss_avg: torch.Tensor,
                 stddev_avg: torch.Tensor, loss_chn_avg: torch.Tensor, 
                 stream_chans: dict, perf_gpu: float = 0., 
                 perf_mem: float = 0.) -> None:
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
    """
    columns_config = {
      'global': ["step", "time", "samples", "perf_gpu", "perf_mem", "learning_rate", "loss_mean"],
    }
    for stream in stream_chans:
      columns_config[stream] = [nm[0] for j, nm in enumerate( self.cf.loss_fcts)] + stream_chans[stream] + ['std']
    
    columns = []
    for top_level, sub_columns in columns_config.items():
        columns.extend([(top_level, sub) for sub in sub_columns])

    multi_cols = pd.MultiIndex.from_tuples(columns)

    log_vals = [self.train_step]
    log_vals += [ int(datetime.datetime.now().strftime(  '%Y%m%d%H%M%S')) ]
    log_vals += [samples]
    log_vals += [perf_gpu]
    log_vals += [perf_mem]
    log_vals += [lr.item()]
    log_vals += [loss_avg[0].mean().item()]

    for i_obs, rt in enumerate( stream_chans.values()) :
      for j, nm in enumerate( self.cf.loss_fcts) :
        log_vals += [ loss_avg[j,i_obs].item() ]
      log_vals += loss_chn_avg[:, 0:len(rt), i_obs].nanmean(dim=0).tolist()
      if len(stddev_avg) > 0 :
        log_vals += [ stddev_avg[i_obs].item() ]
      else:
        log_vals += [ None ]

    df = pd.DataFrame([log_vals], columns=multi_cols)

    with open(self.train_file, 'a') as f:
      df.to_csv(f, header=self.train_step == 1, index=False)

    self.train_step += 1

  #######################################
  def add_val( self, samples: int, loss_avg: torch.Tensor, stddev_avg: torch.Tensor, loss_chn_avg: torch.Tensor, stream_chans: dict) -> None :
    """
    Logs validation metrics to CSV file with multi-index columns.

    Follows same column structure as training logs but without performance metrics.

    Args:
        samples (int): Number of samples processed in this step.
        loss_avg (torch.Tensor): Array of average loss values across loss functions and streams.
        stddev_avg (torch.Tensor): Array of standard deviation values per stream.
        loss_chn_avg (torch.Tensor): 3D array of channel-specific losses (loss_fns x channels x streams).
        stream_chans (dict): Dictionary mapping stream names to their output channels.
    """
    columns_config = {
      'global': ["step", "time", "samples", "loss_mean"],
    }
    for stream in stream_chans:
      columns_config[stream] = [nm[0] for j, nm in enumerate( self.cf.loss_fcts)] + stream_chans[stream] + ['std']
    
    columns = []
    for top_level, sub_columns in columns_config.items():
        columns.extend([(top_level, sub) for sub in sub_columns])

    multi_cols = pd.MultiIndex.from_tuples(columns)
    
    log_vals = [self.val_step]
    log_vals += [ int(datetime.datetime.now().strftime(  '%Y%m%d%H%M%S')) ]
    log_vals += [samples]
    log_vals += [loss_avg[0].mean().item()]

    for i_obs, rt in enumerate( stream_chans.values()) :
      for j, nm in enumerate( self.cf.loss_fcts) :
        log_vals += [ loss_avg[j,i_obs].item() ]
      log_vals += loss_chn_avg[:, 0:len(rt), i_obs].nanmean(dim=0).tolist()
      if len(stddev_avg) > 0 :
        log_vals += [ stddev_avg[i_obs].item() ]
      else:
        log_vals += [ None ]

    df = pd.DataFrame([log_vals], columns=multi_cols)

    with open(self.val_file, 'a') as f:
      df.to_csv(f, header=self.val_step == 1, index=False)

    self.val_step += 1

  #######################################
  @staticmethod
  def read( run_id, epoch=-1) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    cf = Config.load( run_id, epoch)
    run_id = cf.run_id

    fname_log_train = f'./results/{run_id}/{run_id}_train_log.csv'
    fname_log_val = f'./results/{run_id}/{run_id}_val_log.csv'
    
    df_train = pd.read_csv(fname_log_train, index_col=[0,1], skipinitialspace=True)
    df_val = pd.read_csv(fname_log_val, index_col=[0,1], skipinitialspace=True)

    return df_train, df_val
    
