# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, OneCycleLR

from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)


class LearningRateScheduler:
    def __init__(
        self,
        optimizer,
        batch_size,
        world_size,
        istep,
        lr_steps,
        lr_cfg,
    ):
        # '''
        # Three-phase learning rate schedule

        # optimizer :
        # '''

        # TODO: implement cool down mode that continues a run but performs just cooldown
        # from current learning rate, see https://arxiv.org/abs/2106.04560

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.world_size = world_size

        # check validity of parameters and adjust if necessary
        self.lr_steps = lr_steps
        self.n_steps_warmup = lr_cfg.num_steps_warmup
        self.n_steps_cooldown = lr_cfg.num_steps_cooldown

        self.n_steps_decay = lr_steps - self.n_steps_warmup - self.n_steps_cooldown

        if is_root():
            logger.debug(f"steps_decay={self.n_steps_decay} lr_steps={lr_steps}")
        # ensure that steps_decay has a reasonable value
        if self.n_steps_decay < int(0.2 * lr_steps):
            self.n_steps_warmup = int(0.1 * lr_steps)
            self.n_steps_cooldown = int(0.05 * lr_steps)
            self.n_steps_decay = lr_steps - self.n_steps_warmup - self.n_steps_cooldown
            s = (
                "cf.lr_steps_warmup and cf.lr_steps_cooldown",
                f" were larger than cf.lr_steps={lr_steps}",
            )
            s += (f". The value have been adjusted to lr_steps_warmup={self.n_steps_warmup} and ",)
            s += (
                f" lr_steps_cooldown={self.n_steps_cooldown} so steps_decay={self.n_steps_decay}.",
            )
            if is_root():
                logger.warning(s)

        assert lr_cfg.lr_final_decay >= lr_cfg.lr_final

        if lr_cfg.parallel_scaling_policy == "const":
            kappa = 1
        elif lr_cfg.parallel_scaling_policy == "sqrt":
            kappa = np.sqrt(batch_size * self.world_size)
        elif lr_cfg.parallel_scaling_policy == "linear":
            kappa = batch_size * self.world_size
        else:
            assert False, "unsupported learning rate policy"

        self.lr_max_scaled = kappa * lr_cfg.lr_max
        lr_final_decay_scaled = kappa * lr_cfg.lr_final_decay

        self.policy_warmup = lr_cfg.policy_warmup
        self.policy_decay = lr_cfg.policy_decay
        self.policy_cooldown = lr_cfg.policy_cooldown

        self.step_contd = istep

        # create learning rate schedulers

        # warmup

        if self.policy_warmup == "linear":
            self.scheduler_warmup = LinearLR(
                optimizer,
                start_factor=lr_cfg.lr_start / self.lr_max_scaled,
                end_factor=1.0,
                total_iters=self.n_steps_warmup,
            )

        elif self.policy_warmup == "cosine":
            n_steps = self.n_steps_warmup + self.n_steps_decay + 1
            pct_start = self.n_steps_warmup / n_steps
            self.scheduler_warmup = OneCycleLR(
                optimizer,
                max_lr=self.lr_max_scaled,
                total_steps=n_steps,
                pct_start=pct_start,
                div_factor=self.lr_max_scaled / lr_cfg.lr_start,
                final_div_factor=lr_final_decay_scaled / lr_cfg.lr_start,
            )
        else:
            if self.n_steps_warmup > 0:
                assert False, "Unsupported warmup policy for learning rate scheduler"

        # decay

        if self.policy_decay == "linear":
            self.scheduler_decay = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=lr_cfg.lr_final_decay / self.lr_max_scaled,
                total_iters=self.n_steps_decay,
            )

        elif self.policy_decay == "exponential":
            gamma = np.power(
                np.float64(lr_cfg.lr_final_decay / self.lr_max_scaled),
                1.0 / np.float64(self.n_steps_decay),
            )
            self.scheduler_decay = ExponentialLR(optimizer, gamma=gamma)

        elif self.policy_decay == "cosine":
            # OneCycleLR has global state so more work needed to have independent ones
            assert self.policy_decay == self.policy_warmup
            self.scheduler_decay = self.scheduler_warmup

        elif self.policy_decay == "sqrt":
            self.decay_factor = self.lr_max_scaled * np.sqrt(self.n_steps_warmup)
            self.scheduler_decay = None

        elif self.policy_decay == "constant":
            self.decay_factor = 0.0
            self.scheduler_decay = None

        else:
            assert False, "Unsupported decay policy for learning rate scheduler"

        # cool down

        if self.policy_cooldown == "linear":
            self.scheduler_cooldown = LinearLR(
                optimizer,
                start_factor=lr_cfg.lr_start / self.lr_max_scaled,
                end_factor=lr_cfg.lr_final / lr_cfg.lr_final_decay
                if lr_cfg.lr_final_decay > 0.0
                else 0.0,
                total_iters=self.n_steps_cooldown,
            )
        # TODO: this overwrites the cosine scheduler for warmup (seems there are some global vars )
        # elif policy_cooldown == 'cosine' :
        # self.scheduler_cooldown = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=lr_final_decay,
        #     total_steps=n_steps_cooldown,
        #     pct_start=0.0,
        # )
        else:
            if self.n_steps_cooldown > 0:
                assert "Unsupported cooldown policy for learning rate scheduler"

        # final setup

        # set initial scheduler
        self.cur_scheduler = (
            self.scheduler_warmup if self.n_steps_warmup > 0 else self.scheduler_decay
        )

        # explicitly track steps to be able to switch between optimizers
        self.i_step = 0
        self.lr = self.cur_scheduler.get_last_lr()

        # advance manually to step_contd (last_mini_epoch parameter for schedulers is not working
        # and this is also more brittle with the different phases)
        # optimizer.step() as required by torch;
        # won't have a material effect since grads are zero at this point
        if self.step_contd > 0:
            optimizer.step()
            for _ in range(self.step_contd):
                self.step()

    def step(self):
        """
        Perform one step of learning rate schedule
        """

        # keep final learning rate
        if self.i_step >= (self.n_steps_warmup + self.n_steps_decay + self.n_steps_cooldown):
            return self.lr

        end_decay = self.n_steps_warmup + self.n_steps_decay
        phase_decay = (self.i_step > self.n_steps_warmup) and (self.i_step <= end_decay)

        if self.policy_decay == "sqrt" and phase_decay:
            self.lr = (
                (self.decay_factor / np.sqrt(self.i_step))
                if self.i_step > 0
                else self.lr_max_scaled
            )
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr
        elif self.policy_decay == "constant" and phase_decay:
            cur_lr = self.lr
            self.lr = self.lr_max_scaled
            # make sure lr_max_scaled rate is used if warm-up end is not lr_max_scaled
            if cur_lr < self.lr:
                for g in self.optimizer.param_groups:
                    g["lr"] = self.lr
        else:
            self.cur_scheduler.step()
            self.lr = self.cur_scheduler.get_last_lr()[0]

        # switch scheduler when learning rate regime completed
        if self.i_step == self.n_steps_warmup:
            self.cur_scheduler = self.scheduler_decay
            str = f"Switching lr scheduler to '{self.policy_decay}' at step = {self.i_step}."
            logging.getLogger("obslearn").info(str)

        # switch scheduler when learning rate completed
        if self.i_step == self.n_steps_warmup + self.n_steps_decay:
            self.cur_scheduler = self.scheduler_cooldown
            str = f"Switching lr scheduler to '{self.policy_cooldown}' at step = {self.i_step}."
            logging.getLogger("obslearn").info(str)

        self.i_step += 1

        return self.lr

    def get_lr(self):
        return self.lr

    @staticmethod
    def plot():
        """
        Generate plot of learning rate schedule

        Use as LearningRateScheduler.plot()
        """

        num_mini_epochs = 42
        num_samples_per_mini_epoch = 4096

        lr_start = 0.000001
        lr_max = 0.000015
        lr_final_decay = 0.000001
        lr_final = 0.0
        lr_steps_warmup = 256
        lr_steps_cooldown = 1024
        lr_steps_warmup = 256
        lr_steps_cooldown = 4096
        lr_policy_warmup = "cosine"
        lr_policy_decay = "linear"
        lr_policy_cooldown = "linear"

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_max)

        scheduler = LearningRateScheduler(
            optimizer,
            1,
            1,
            lr_start,
            lr_max,
            lr_final_decay,
            lr_final,
            lr_steps_warmup,
            num_mini_epochs * num_samples_per_mini_epoch,
            lr_steps_cooldown,
            lr_policy_warmup,
            lr_policy_decay,
            lr_policy_cooldown,
        )
        lrs = []

        for _ in range(
            num_mini_epochs * num_samples_per_mini_epoch
            + lr_steps_warmup
            + lr_steps_cooldown
            + 1023
        ):
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        plt.plot(lrs, "b")
        # plt.savefig( './plots/lr_schedule.png')

        # second strategy for comparison

        lr_policy_decay = "cosine"

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_max)

        scheduler = LearningRateScheduler(
            optimizer,
            1,
            1,
            lr_start,
            lr_max,
            lr_final_decay,
            lr_final,
            lr_steps_warmup,
            num_mini_epochs * num_samples_per_mini_epoch,
            lr_steps_cooldown,
            lr_policy_warmup,
            lr_policy_decay,
            lr_policy_cooldown,
        )
        lrs = []

        for _ in range(
            num_mini_epochs * num_samples_per_mini_epoch
            + lr_steps_warmup
            + lr_steps_cooldown
            + 1023
        ):
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        plt.plot(lrs, "r")
        plt.savefig("./plots/lr_schedule.png")
