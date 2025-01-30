# (C) Copyright 2024 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import code

import os
import subprocess
import glob
import argparse

import pandas as pd

import matplotlib.pyplot as plt

from weathergen.utils.config import Config
from weathergen.utils.train_logger import TrainLogger

out_folder = './plots/'

####################################################################################################
def clean_out_folder() :
  files = glob.glob( out_folder + '*.png')
  for f in files :
    os.remove(f)

####################################################################################################
def get_stream_names( run_id) :
  # return col names from training (should be identical to validation)
  cf = Config.load( run_id, -1)
  return [si['name'].replace(',','').replace('/','_').replace(' ','_') for si in cf.streams]

####################################################################################################
def plot_lr( runs_ids, runs_data, runs_active, x_axis='samples') :

  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color'] + ['r', 'g', 'b', 'k', 'y', 'm']
  fig = plt.figure( figsize=(10,7), dpi=300)

  # train
  idx = 0
  linestyle = '-'

  legend_str = []
  for j,(run_id, run_data) in enumerate(zip( runs_ids, runs_data)) :
    if run_data[idx][1].shape[0] == 0 :
      continue

    x_idx = [i for i,c in enumerate(run_data[idx][0]) if x_axis in c][0]
    data_idxs = [i for i,c in enumerate(run_data[idx][0]) if 'lr'==c][0]

    plt.plot( run_data[idx][1][:,x_idx], run_data[idx][1][:,data_idxs], linestyle,
              color=colors[j % len(colors)])
    legend_str += [ ('R' if runs_active[j] else 'X') + ' : ' + run_id 
                                                      + ' : ' + runs_ids[run_id][1] ]

  if len(legend_str) < 1 :
    return

  plt.legend( legend_str)
  plt.grid( True, which="both", ls="-")
  plt.yscale( 'log')
  plt.title( 'learning rate')
  plt.ylabel( 'lr'); plt.xlabel( x_axis); plt.tight_layout()
  rstr = ''.join([f'{r}_' for r in runs_ids])
  plt.savefig( './plots/{}lr.png'.format( rstr))
  plt.close()

####################################################################################################
def plot_utilization( runs_ids, runs_data, runs_active, x_axis='samples') :

  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color'] + ['r', 'g', 'b', 'k', 'y', 'm']
  fig = plt.figure( figsize=(10,7), dpi=300)
  
  linestyles = ['-', '--', '.-']

  # performance
  idx = 2

  legend_str = []
  for j,(run_id, run_data) in enumerate(zip( runs_ids, runs_data)) :
    if run_data[idx][1].shape[0] == 0 :
      continue

    x_idx = [i for i,c in enumerate(run_data[0][0]) if x_axis in c][0]
    data_idxs = [i for i in range(len(run_data[2][0]))]

    for ii,di in enumerate(data_idxs) :
      plt.plot( run_data[0][1][:,x_idx], run_data[idx][1][:,di], linestyles[ii],
                color=colors[j % len(colors)])
      legend_str += [ ('R' if runs_active[j] else 'X') + ' : ' + run_id + ', ' + run_data[idx][0][ii]
                                                        + ' : ' + runs_ids[run_id][1] ]

  if len(legend_str) < 1 :
    return

  plt.legend( legend_str)
  plt.grid( True, which="both", ls="-")
  # plt.yscale( 'log')
  plt.title( 'utilization')
  plt.ylabel( 'percentage utilization'); plt.xlabel( x_axis); plt.tight_layout()
  rstr = ''.join([f'{r}_' for r in runs_ids])
  plt.savefig( './plots/{}utilization.png'.format( rstr))
  plt.close()

####################################################################################################
def plot_loss_per_stream( modes, runs_ids, runs_data, runs_times, runs_active, stream_names,
                          errs = ['mse'], x_axis='samples', x_type='step', x_scale_log=False) :
  '''
    Plot each stream in stream_names (using matching to data columns) for all run_ids
  '''

  modes = [modes] if type(modes) is not list else modes
  # repeat colors when train and val is plotted simultaneously
  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color'] + ['r', 'g', 'b', 'k', 'm', 'y']
  
  for stream_name in stream_names :

    fig = plt.figure( figsize=(10,7), dpi=300)

    legend_strs = []
    min_val = np.finfo( np.float32).max
    max_val = 0.
    for mode in modes :
      legend_strs += [ [] ]
      for err in errs :

        idx = 0 if mode=='train' else 1
        linestyle = '-' if mode=='train' else ('--x' if len(modes)>1 else '-x')
        linestyle = ':' if 'stddev' in err else linestyle
        alpha = 1.0
        if 'train' in modes and 'val' in modes :
          alpha = 0.35 if 'train' in mode else alpha

        for j,(run_id, run_data) in enumerate(zip( runs_ids, runs_data)) :

          x_idx = [i for i,c in enumerate(run_data[idx][0]) if x_axis in c][0]
          data_idxs = [i for i,c in enumerate(run_data[idx][0]) if err in c]

          for i,col in enumerate( np.array(run_data[idx][0])[data_idxs]) :
            if stream_name in col :
              if run_data[idx][1].shape[0] == 0 :
                continue

              x_vals = run_data[idx][1][:,x_idx] if x_type=='step' else runs_times[j][idx][1]

              plt.plot( x_vals, run_data[idx][1][:,data_idxs[i]], linestyle,
                        color=colors[j % len(colors)], alpha=alpha)
              legend_strs[-1] += [ ('R' if runs_active[j] else 'X') + ' : ' + run_id 
                                                      + ' : ' + runs_ids[run_id][1] + ': ' + col ]

              min_val = np.min( [ min_val, np.nanmin(run_data[idx][1][:,data_idxs[i]]) ])
              max_val = np.max( [ max_val, np.nanmax(run_data[idx][1][:,data_idxs[i]]) ])

    # TODO: ensure that legend is plotted with full opacity
    legend_str = legend_strs[0] 
    if len(legend_str) < 1 :
      plt.close()
      continue

    legend = plt.legend( legend_str, loc='upper right' if not x_scale_log else 'lower left')
    for line in legend.get_lines():
      line.set( alpha=1.0)
    plt.grid( True, which="both", ls="-")
    plt.yscale( 'log')
    # cap at 1.0 in case of divergence of run (through normalziation, max should be around 1.0)
    plt.ylim( [ 0.95*min_val, (None if max_val<2.0 else min( 1.1, 1.025*max_val)) ])
    if x_scale_log :
      plt.xscale( 'log')
    plt.title( stream_name)
    plt.ylabel( 'loss')
    plt.xlabel( x_axis if x_type=='step' else 'rel. time [h]')
    plt.tight_layout()
    rstr = ''.join([f'{r}_' for r in runs_ids])
    plt.savefig( out_folder + '{}{}{}.png'.format( rstr, ''.join( [f'{m}_' for m in modes]), stream_name))
    plt.close()

####################################################################################################
def plot_loss_per_run( modes, run_id, run_desc, run_data, stream_names, 
                       errs = ['mse'], x_axis='samples', x_scale_log=False) :
  '''
    Plot all stream_names (using matching to data columns) for given run_id

    x_axis : {samples,dtime} as used in the column names
  '''

  modes = [modes] if type(modes) is not list else modes
  # repeat colors when train and val is plotted simultaneously
  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color'] + ['r', 'g', 'b', 'k', 'y', 'm']

  fig = plt.figure( figsize=(10,7), dpi=300)

  legend_strs = []
  for mode in modes :
    legend_strs += [ [] ]
    for err in errs :

      idx = 0 if mode=='train' else 1
      linestyle = '-' if mode=='train' else ('--x' if len(modes)>1 else '-x')
      linestyle = ':' if 'stddev' in err else linestyle
      alpha = 1.0
      if 'train' in modes and 'val' in modes :
        alpha = 0.35 if 'train' in mode else alpha

      x_idx = [i for i,c in enumerate(run_data[idx][0]) if x_axis in c][0]
      data_idxs = [i for i,c in enumerate(run_data[idx][0]) if err in c]

      for i,col in enumerate( np.array(run_data[idx][0])[data_idxs]) :
        for j, stream_name in enumerate(stream_names) :
            if stream_name in col :

              # skip when no data is available
              if run_data[idx][1].shape[0] == 0 :
                continue

              plt.plot( run_data[idx][1][:,x_idx], run_data[idx][1][:,data_idxs[i]], linestyle,
                        color=colors[j % len(colors)], alpha=alpha)
              legend_strs[-1] += [col]

  legend_str = legend_strs[0]
  if len(legend_str) < 1 :
    plt.close()
    return

  plt.title( run_id + ' : ' + run_desc[1])
  legend = plt.legend( legend_str, loc='lower left')
  for line in legend.get_lines():
    line.set( alpha=1.0)
  plt.yscale( 'log')
  if x_scale_log :
    plt.xscale( 'log')
  plt.grid( True, which="both", ls="-")
  plt.ylabel( 'loss'); plt.xlabel( 'samples'); plt.tight_layout()
  sstr = ''.join([f'{r}_'.replace(',','').replace('/','_').replace(' ','_') for r in legend_str])
  plt.savefig( out_folder + '{}_{}{}.png'.format( run_id, ''.join( [f'{m}_' for m in modes]), sstr))
  plt.close()

####################################################################################################
if __name__ == '__main__' :

  parser = argparse.ArgumentParser()
  parser.add_argument( '-d', '--delete')
  args = parser.parse_args()
  
  if args.delete == 'True' :
    clean_out_folder()

  runs_ids = { 
              'i18n6wmx' : [1, 'ERA5 test'],
            }


  runs_data = [TrainLogger.read( run_id) for run_id in runs_ids.keys()]

  # extract times and convert back to datetime objects, store absolute time ta and relative one tr 
  runs_times = []
  for rd in runs_data :
    # training
    if len(rd[1][1]) > 0 :
      ta_train = pd.to_datetime( rd[0][1][:,0], format='%Y%m%d%H%M%S')
      diff = (ta_train - ta_train[0])
      tr_train = diff.days * 24 + diff.seconds / 3600.
    else :
      ta_train, tr_train = [], []
    # validation
    if len(rd[1][1]) > 0 :
      ta_val = pd.to_datetime( rd[1][1][:,0], format='%Y%m%d%H%M%S')
      diff = (ta_val - ta_train[0])
      tr_val = diff.days * 24 + diff.seconds / 3600.
    else :
      ta_val, tr_val = [], []
    runs_times.append( [[ta_train, tr_train], [ta_val, tr_val]])


  # determine which runs are still alive (as a process, though they might hang internally)
  ret = subprocess.run(["squeue"], capture_output=True)
  lines = str(ret.stdout).split( '\\n')
  runs_active = [np.array([str(v[0]) in l for l in lines[1:]]).any() for v in runs_ids.values()]

  x_scale_log = False
  x_type = 'rel_time', #'step'
  x_type = 'step'

  # plot learning rate
  plot_lr( runs_ids, runs_data, runs_active)

  # plot performance
  plot_utilization( runs_ids, runs_data, runs_active)

  # TODO: finish
  smoothing_size = 0
  if smoothing_size > 0 :
    # smooth
    x = np.linspace( -np.pi, np.pi, smoothing_size)
    gauss_filter =  np.exp( -np.square(x))
    for j in range(len(runs_data)) :
      for i in range(2) :
        if runs_data[j][i][1].shape[0] <= gauss_filter.shape[0] :
          continue
        for i_ch in range( runs_data[j][i][1].shape[-1]) :
          if not ('mse' in runs_data[j][i][0][i_ch]) :
            continue
          res = np.convolve( runs_data[j][i][1][:,i_ch], gauss_filter, 'same')
          code.interact( local=locals())
          runs_data[j][i][1][:(res.shape[0]-smoothing_size),i_ch] = res[:-(smoothing_size)]


  # compare different runs
  plot_loss_per_stream( ['train', 'val'], runs_ids, runs_data, runs_times, runs_active,
                        ['ERA5'], 
                        x_type=x_type, x_scale_log=x_scale_log)
  plot_loss_per_stream( ['val'], runs_ids, runs_data, runs_times, runs_active,
                        ['ERA5'], 
                        x_type=x_type, x_scale_log=x_scale_log)
  plot_loss_per_stream( ['train'], runs_ids, runs_data, runs_times, runs_active,
                        ['ERA5'], 
                        x_type=x_type, x_scale_log=x_scale_log)
  
  # plot all cols for all run_ids
  for run_id, run_data in zip( runs_ids, runs_data) :
    plot_loss_per_run( ['train', 'val'], run_id, runs_ids[run_id], run_data, get_stream_names( run_id))
    # plot_loss_per_run( ['val'], run_id, runs_ids[run_id], run_data, get_stream_names( run_id))

