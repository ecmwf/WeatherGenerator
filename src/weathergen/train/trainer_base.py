# (C) Copyright 2024 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import datetime
import string
import random
import pathlib
import itertools
import logging
import json
import yaml
import logging
import code

import numpy as np
import torch
# import mlflow

import pynvml

import torch.distributed as dist
import torch.utils.data.distributed

from weathergen.utils.config import Config
import weathergen.utils.logger
from weathergen.train.utils import get_run_id, str_to_tensor, tensor_to_str, json_to_dict


class Trainer_Base() :

  def __init__( self) :
    pass

  ###########################################
  @staticmethod
  def init_mlflow( cf, rank, run_id_contd = None, run_id_new = False,
                   project='obs_learn_kas_cell_forecast') :

    if 0 == rank :

      run_id = cf.run_id

      slurm_job_id_node = os.environ.get('SLURM_JOB_ID', '-1')
      if slurm_job_id_node != '-1' :
        cf.slurm_job_id = slurm_job_id_node

      # check if offline mode is requested through environment variable or in config
      mlflow_offline_env = os.environ.get('MLFLOW_OFFLINE', '-1')
      if not hasattr( cf, 'mlflow_offline') :
        cf.mlflow_offline = True if mlflow_offline_env != '-1' else False

      rs_uri = './mlflow/' if cf.mlflow_offline else 'https://mlflow.ecmwf.int/'
      mlflow.set_tracking_uri( rs_uri)
      mlflow.set_experiment( project)

      # # we separate the mlflow_id and the run_id/run_name, which is used for all local bookkeeping
      # ml_id = None if run_id_contd is None or run_id_new else cf.mlflow_id
      # mlflow_run = mlflow.start_run( run_id=ml_id, run_name=run_id,
      #                                log_system_metrics=True)
      # cf.mlflow_id = mlflow_run.info.run_id

      # log config (cannot be overwritten so log only at the first start)
      if run_id_contd is None or run_id_new :
        mlflow.log_params( cf.__dict__)
      
      if run_id_contd is not None and run_id_new :
        str = f'Continuing run {run_id_contd} at step={cf.istep} as run {run_id}.'
        logging.getLogger('obslearn').info( str)
      elif run_id_contd is not None :
        logging.getLogger('obslearn').info( f'Continuing run {run_id_contd} at step={cf.istep}.')

  ###########################################
  @staticmethod
  def init_torch( use_cuda = True, num_accs_per_task = 1) :

    torch.set_printoptions( linewidth=120)

    torch.backends.cuda.matmul.allow_tf32 = True

    use_cuda = torch.cuda.is_available()
    if not use_cuda :
      return torch.device( 'cpu')

    local_id_node = os.environ.get('SLURM_LOCALID', '-1')
    if local_id_node == '-1' :
      devices = ['cuda']
    else :
      devices = ['cuda:{}'.format(int(local_id_node) * num_accs_per_task + i) 
                                                                for i in range(num_accs_per_task)]
    torch.cuda.set_device( int(local_id_node) * num_accs_per_task )

    return devices 

  ###########################################
  @staticmethod
  def init_ddp( cf) :

    rank = 0
    num_ranks = 1

    master_node = os.environ.get('MASTER_ADDR', '-1')
    if '-1' == master_node :
      cf.with_ddp=False; cf.rank=rank; cf.num_ranks=num_ranks
      return

    local_rank = int(os.environ.get("SLURM_LOCALID"))
    ranks_per_node = int( os.environ.get('SLURM_TASKS_PER_NODE', '1')[0] )
    rank = int(os.environ.get("SLURM_NODEID")) * ranks_per_node + local_rank
    num_ranks = int(os.environ.get("SLURM_NTASKS"))

    dist.init_process_group( backend='nccl', init_method='tcp://' + master_node + ':1345',
                             timeout=datetime.timedelta(seconds=10*8192),
                             world_size = num_ranks, rank = rank)

    # communicate run id to all nodes
    run_id_int = torch.zeros( 8, dtype=torch.int32).cuda()
    if 0 == rank :
      run_id_int = str_to_tensor( cf.run_id).cuda()
    dist.all_reduce( run_id_int, op=torch.distributed.ReduceOp.SUM )
    cf.run_id = tensor_to_str( run_id_int)

    # communicate data_loader_rng_seed
    if hasattr( cf, 'data_loader_rng_seed') :
      if cf.data_loader_rng_seed is not None :
        l_seed = torch.tensor([cf.data_loader_rng_seed if 0==rank else 0], dtype=torch.int32).cuda()
        dist.all_reduce( l_seed, op=torch.distributed.ReduceOp.SUM )
        cf.data_loader_rng_seed = l_seed.item()

    cf.rank = rank
    cf.num_ranks = num_ranks
    cf.with_ddp = True

    return

  ###########################################
  @staticmethod
  def init_streams( cf : Config, run_id_contd ) :

    if not hasattr( cf, 'streams_directory'):
      return cf

    # use previously specified streams when continuing a run
    if run_id_contd is not None :
      return cf

    if not hasattr( cf, 'streams'):
      cf.streams = [ ]
    elif not isinstance( cf.streams, list) :
      cf.streams = [ ]

    # warn if specified dir does not exist
    if not os.path.isdir( cf.streams_directory) :
      sd = cf.streams_directory
      logging.getLogger('obslearn').warning( f'Streams directory {sd} does not exist.')

    # read all reportypes from directory, append to existing ones
    temp = {}
    for fh in sorted( pathlib.Path( cf.streams_directory).rglob( '*.yml')) :
      stream_parsed = yaml.safe_load( fh.read_text())
      if stream_parsed is not None :
        temp.update( stream_parsed)
    for k,v in temp.items() :
      v['name'] = k
      cf.streams.append( v)

    # sanity checking (at some point, the dict should be parsed into a class)
    rts = [ rt['filenames'] for rt in cf.streams]
    # flatten list
    rts = list( itertools.chain.from_iterable( rts))
    if len(rts) != len( list(set( rts))) :
      logging.getLogger('obslearn').warning( 'Duplicate reportypes specified.')

    cf.num_obs_types = 3

    return cf

  ###########################################
  def init_perf_monitoring( self) :

    self.device_handles, self.device_names = [], []

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    for i in range(device_count):
      handle = pynvml.nvmlDeviceGetHandleByIndex(i)
      self.device_names += [ pynvml.nvmlDeviceGetName(handle) ]
      self.device_handles += [ handle ]

  ###########################################
  def get_perf( self) :

    perf_gpu, perf_mem = 0.0, 0.0
    if len(self.device_handles) > 0 :
      for handle in self.device_handles :
        perf = pynvml.nvmlDeviceGetUtilizationRates( handle)
        perf_gpu += perf.gpu
        perf_mem += perf.memory
      perf_gpu /= len(self.device_handles)
      perf_mem /= len(self.device_handles)

    return perf_gpu, perf_mem

  ###########################################
  def ddp_average( self, val) :
    if self.cf.with_ddp :
      dist.all_reduce( val.cuda(), op=torch.distributed.ReduceOp.AVG )
    return val.cpu()

####################################################################################################
if __name__ == '__main__' :

  from weathergen.utils.config import Config
  from weathergen.train.trainer_base import Trainer_Base

  cf = Config()
  cf.sources_dir = './sources'

  cf = Trainer_Base.init_reportypes( cf)
