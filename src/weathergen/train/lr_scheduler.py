# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import torch
import logging
# import matplotlib.pyplot as plt
import warnings
import code

from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ExponentialLR


class LearningRateScheduler :

  def __init__( self, optimizer, batch_size, num_ranks,
                lr_start, lr_max, lr_final_decay, lr_final,
                n_steps_warmup, n_steps_decay, n_steps_cooldown,
                policy_warmup, policy_decay, policy_cooldown,
                step_contd = -1, scaling_policy='sqrt') :
  # '''
  # Three-phase learning rate schedule

  # optimizer :
  # '''

    # TODO: implement cool down mode that continues a run but performs just cooldown
    # from current learning rate, see https://arxiv.org/abs/2106.04560

    assert lr_final_decay >= lr_final

    self.optimizer = optimizer
    self.batch_size = batch_size
    self.num_ranks = num_ranks

    self.n_steps_warmup = n_steps_warmup
    self.n_steps_decay = n_steps_decay
    self.n_steps_cooldown = n_steps_cooldown

    if scaling_policy=='const' :
      kappa = 1
    elif scaling_policy=='sqrt' :
      kappa = np.sqrt(batch_size * self.num_ranks)
    elif scaling_policy=='linear' :
      kappa = batch_size * self.num_ranks
    else :
      assert False, 'unsupported learning rate policy'

    self.lr_max_scaled = kappa * lr_max
    lr_final_decay_scaled = kappa * lr_final_decay

    self.policy_warmup = policy_warmup
    self.policy_decay = policy_decay
    self.policy_cooldown = policy_cooldown

    self.step_contd = step_contd

    # create learning rate schedulers

    ##########################
    # warmup
    if policy_warmup == 'linear' :
      self.scheduler_warmup = LinearLR( optimizer, start_factor=lr_start/self.lr_max_scaled,
                                        end_factor=1.0, total_iters=n_steps_warmup)

    elif policy_warmup == 'cosine' :
      n_steps = n_steps_warmup + n_steps_decay + 1
      pct_start = n_steps_warmup / n_steps
      self.scheduler_warmup = OneCycleLR( optimizer, max_lr=self.lr_max_scaled,
                                          total_steps=n_steps, pct_start=pct_start,
                                          div_factor=self.lr_max_scaled/lr_start,
                                          final_div_factor=lr_final_decay_scaled/lr_start)
    else :
      if n_steps_warmup > 0 :
        assert False, 'Unsupported warmup policy for learning rate scheduler'

    ##########################
    # decay
    if policy_decay == 'linear' :
      self.scheduler_decay = LinearLR( optimizer, start_factor=1.0,
                                       end_factor=lr_final_decay/self.lr_max_scaled,
                                       total_iters=n_steps_decay)

    elif policy_decay == 'exponential' :
      gamma = np.power(np.float64(lr_final_decay/self.lr_max_scaled), 1.0/np.float64(n_steps_decay))
      self.scheduler_decay = ExponentialLR( optimizer, gamma=gamma)

    elif policy_decay == 'cosine' :
      # OneCycleLR has global state so more work needed to have independent ones 
      assert policy_decay == policy_warmup
      self.scheduler_decay = self.scheduler_warmup

    elif policy_decay == 'sqrt' :
      self.decay_factor = self.lr_max_scaled * np.sqrt( n_steps_warmup)
      self.scheduler_decay = None

    else :
       assert False, 'Unsupported decay policy for learning rate scheduler'

    ##########################
    # cool down
    if policy_cooldown == 'linear' :
      self.scheduler_cooldown = LinearLR( optimizer, start_factor=lr_start/self.lr_max_scaled,
                                    end_factor=lr_final/lr_final_decay if lr_final_decay>0. else 0.,
                                          total_iters=n_steps_cooldown)
    # TODO: this overwrites the cosine scheduler for warmup (seems there are some global vars )
    # elif policy_cooldown == 'cosine' :
    #   self.scheduler_cooldown = torch.optim.lr_scheduler.OneCycleLR( optimizer, 
    #                                                                  max_lr=lr_final_decay, 
    #                                                                  total_steps=n_steps_cooldown, 
    #                                                                  pct_start=0.0)
    else :
      if n_steps_cooldown > 0 :
        assert 'Unsupported cooldown policy for learning rate scheduler'

    # set initial scheduler
    self.cur_scheduler = self.scheduler_warmup if n_steps_warmup > 0 else self.scheduler_decay

    # explicitly track steps to be able to switch between optimizers
    self.i_step = 0
    self.lr = self.cur_scheduler.get_last_lr()

    # advance manually to step_contd (last_epoch parameter for schedulers is not working and
    # this is also more brittle with the different phases)
    # optimizer.step() as required by torch; won't have a material effect since grads are zero at
    # this point
    if self.step_contd > 0 :
      optimizer.step()
      for _ in range( step_contd) :
        self.step()

  #######################################
  def step( self) :
    '''
    Perform one step of learning rate schedule
    '''

    # keep final learning rate
    if self.i_step >= (self.n_steps_warmup + self.n_steps_decay + self.n_steps_cooldown) :
      return self.lr

    if (self.policy_decay == 'sqrt' and self.i_step > self.n_steps_warmup
                and self.i_step < self.n_steps_warmup + self.n_steps_decay) :
      self.lr = (self.decay_factor/np.sqrt(self.i_step)) if self.i_step > 0 else self.lr_max_scaled
      for g in self.optimizer.param_groups:
        g['lr'] = self.lr
    else :
      self.cur_scheduler.step()
      self.lr = self.cur_scheduler.get_last_lr()[0]

    # switch scheduler when learning rate regime completed
    if self.i_step == self.n_steps_warmup :
      self.cur_scheduler = self.scheduler_decay
      str = f'Switching scheduler to {self.cur_scheduler} at scheduler step = {self.i_step}.'
      logging.getLogger('obslearn').info( str)

    # switch scheduler when learning rate completed
    if self.i_step == self.n_steps_warmup + self.n_steps_decay :
      self.cur_scheduler = self.scheduler_cooldown
      str = f'Switching scheduler to {self.cur_scheduler} at scheduler step = {self.i_step}.'
      logging.getLogger('obslearn').info( str)

    self.i_step += 1

    return self.lr

  #######################################
  def get_lr( self) :
    return self.lr 

  #######################################
  @staticmethod
  def plot() :
    '''
      Generate plot of learning rate schedule

      Use as LearningRateScheduler.plot()
    '''

    num_epochs = 42
    num_samples_per_epoch = 4096

    lr_start = 0.000001
    lr_max = 0.000015
    lr_final_decay = 0.000001
    lr_final = 0.0
    lr_steps_warmup = 256
    lr_steps_cooldown = 1024
    lr_steps_warmup = 256
    lr_steps_cooldown = 4096
    lr_policy_warmup = 'cosine'
    lr_policy_decay = 'linear'
    lr_policy_cooldown = 'linear'

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_max)

    scheduler = LearningRateScheduler( optimizer, 1, 1, lr_start, lr_max, lr_final_decay, lr_final,
                            lr_steps_warmup, num_epochs*num_samples_per_epoch, lr_steps_cooldown,
                            lr_policy_warmup, lr_policy_decay, lr_policy_cooldown)
    lrs = []

    for i in range( num_epochs*num_samples_per_epoch + lr_steps_warmup + lr_steps_cooldown + 1023):
      optimizer.step()
      lrs.append( optimizer.param_groups[0]['lr'])
      scheduler.step()

    plt.plot(lrs, 'b')
    # plt.savefig( './plots/lr_schedule.png')

    # second strategy for comparison

    lr_policy_decay = 'cosine'

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_max)

    scheduler = LearningRateScheduler( optimizer, 1, 1, lr_start, lr_max, lr_final_decay, lr_final,
                            lr_steps_warmup, num_epochs*num_samples_per_epoch, lr_steps_cooldown,
                            lr_policy_warmup, lr_policy_decay, lr_policy_cooldown)
    lrs = []

    for i in range( num_epochs*num_samples_per_epoch + lr_steps_warmup + lr_steps_cooldown + 1023):
      optimizer.step()
      lrs.append( optimizer.param_groups[0]['lr'])
      scheduler.step()

    plt.plot(lrs, 'r')
    plt.savefig( './plots/lr_schedule.png')


