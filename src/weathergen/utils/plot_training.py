# (C) Copyright 2025 WeatherGenerator contributors.
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

  linestyle = '-'

  legend_str = []
  for j,(run_id, run_data) in enumerate(zip( runs_ids, runs_data)) :
    if len(run_data[0]) == 0 : # If df is emplty skip
      continue
    
    x_vals = run_data[0]['global'].samples.values if x_axis=='samples' else run_data[0]['global'].step.values

    plt.plot(x_vals, run_data[0]['global'].learning_rate.values, 
             linestyle, color=colors[j % len(colors)])
    legend_str += [ ('R' if runs_active[j] else 'X') + ' : ' + run_id 
                                                      + ' : ' + runs_ids[run_id][1] ]

  if len(legend_str) < 1 :
    return

  plt.legend( legend_str)
  plt.grid( True, which="both", ls="-")
  plt.yscale( 'log')
  plt.title( 'learning rate')
  plt.ylabel( 'lr') 
  plt.xlabel( x_axis)
  plt.tight_layout()
  rstr = ''.join([f'{r}_' for r in runs_ids])
  plt.savefig( './plots/{}lr.png'.format( rstr))
  plt.close()

####################################################################################################
def plot_utilization( runs_ids, runs_data, runs_active, x_axis='samples') :

  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color'] + ['r', 'g', 'b', 'k', 'y', 'm']
  fig = plt.figure( figsize=(10,7), dpi=300)
  
  linestyles = ['-', '--', '.-']

  legend_str = []
  for j,(run_id, run_data) in enumerate(zip( runs_ids, runs_data)) :
    if len(run_data[0]) == 0 : # If df is emplty skip
      continue

    x_vals = run_data[0]['global'].samples.values if x_axis=='samples' else run_data[0]['global'].step.values

    plt.plot(x_vals, run_data[0]['global'].perf_gpu.values, 
             linestyles[0], color=colors[j % len(colors)])
    legend_str += [ ('R' if runs_active[j] else 'X') + ' : ' + run_id + ', GPU : ' + runs_ids[run_id][1] ]
    
    plt.plot(x_vals, run_data[0]['global'].perf_mem.values, 
             linestyles[1], color=colors[j % len(colors)])
    legend_str += [ ('R' if runs_active[j] else 'X') + ' : ' + run_id + ', memory : ' + runs_ids[run_id][1] ]

  if len(legend_str) < 1 :
    return

  plt.legend( legend_str)
  plt.grid( True, which="both", ls="-")
  # plt.yscale( 'log')
  plt.title( 'utilization')
  plt.ylabel( 'percentage utilization')
  plt.xlabel( x_axis)
  plt.tight_layout()
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
      idx = 0 if mode=='train' else 1
      legend_strs += [ [] ]
      for err in errs :

        linestyle = '-' if mode=='train' else ('--x' if len(modes)>1 else '-x')
        linestyle = ':' if 'stddev' in err else linestyle
        alpha = 1.0
        if 'train' in modes and 'val' in modes :
          alpha = 0.35 if 'train' in mode else alpha

        for j,(run_id, run_data) in enumerate(zip( runs_ids, runs_data)) :
          if len(run_data[idx]) == 0 :
            # skip when no data is available
            continue
          if stream_name in run_data[idx] and err in run_data[idx][stream_name]:
            x_vals = run_data[idx]['global'].step.values if x_type=='step' else run_data[idx]['global'].time.values
            y_vals = run_data[idx][stream_name][err].values

            plt.plot( x_vals, y_vals, linestyle, color=colors[j % len(colors)], alpha=alpha)
            legend_strs[-1] += [ ('R' if runs_active[j] else 'X') + ' : ' + run_id 
                                                    + ' : ' + runs_ids[run_id][1] + ': ' + stream_name ]

            min_val = np.min( [ min_val, np.nanmin(y_vals) ])
            max_val = np.max( [ max_val, np.nanmax(y_vals) ])

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
  j = 0
  for mode in modes :
    idx = 0 if mode=='train' else 1
    if len(run_data[idx]) == 0 :
      # skip when no data is available
      continue

    legend_strs += [ [] ]
    for err in errs :
      linestyle = '-' if mode=='train' else ('--x' if len(modes)>1 else '-x')
      linestyle = ':' if 'stddev' in err else linestyle
      alpha = 1.0
      if 'train' in modes and 'val' in modes :
        alpha = 0.35 if 'train' in mode else alpha

      for stream_name in stream_names:
        if stream_name in run_data[idx] and err in run_data[idx][stream_name] :
          x_vals = run_data[idx]['global'].samples.values if x_axis=='samples' else run_data[idx]['global'].step.values
          plt.plot(x_vals, run_data[idx][stream_name][err].values, linestyle, color=colors[j % len(colors)], alpha=alpha)
          legend_strs[-1] += [f"{stream_name}, {err}"]
          j += 1

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
  plt.ylabel( 'loss')
  plt.xlabel(x_axis)
  plt.tight_layout()

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
              'rdbe7qsu' : [15316901, 'Short test']
            }

  runs_data = [TrainLogger.read( run_id) for run_id in runs_ids.keys()]

  # extract times, store absolute time ta and relative one tr 
  runs_times = []
  for rd in runs_data :
    train_df, val_df = rd
    # training times 
    if len(train_df) > 0:
      ta_train = train_df['global'].time
      tr_train = ta_train.diff().dt.total_seconds() / 3600.
    else :
      ta_train, tr_train = [], []
    # Validation times
    if len(val_df) > 0:
      ta_val = val_df['global'].time
      tr_val = ta_train.diff().dt.total_seconds() / 3600.
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

  # compare different runs
  plot_loss_per_stream( ['train', 'val'], runs_ids, runs_data, runs_times, runs_active,
                        ['ERA5', 'FESOM', 'NPP'], 
                        x_type=x_type, x_scale_log=x_scale_log)
  plot_loss_per_stream( ['val'], runs_ids, runs_data, runs_times, runs_active,
                        ['ERA5', 'FESOM', 'NPP'], 
                        x_type=x_type, x_scale_log=x_scale_log)
  plot_loss_per_stream( ['train'], runs_ids, runs_data, runs_times, runs_active,
                        ['ERA5', 'FESOM', 'NPP'], 
                        x_type=x_type, x_scale_log=x_scale_log)
  
  # plot all cols for all run_ids
  for run_id, run_data in zip( runs_ids, runs_data) :
    plot_loss_per_run( ['train', 'val'], run_id, runs_ids[run_id], run_data, get_stream_names( run_id))
    # plot_loss_per_run( ['val'], run_id, runs_ids[run_id], run_data, get_stream_names( run_id))

