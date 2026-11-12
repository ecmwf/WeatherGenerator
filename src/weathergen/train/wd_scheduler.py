# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import torch

from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)


class WeightDecayScheduler:
    """
    Increases the weight decay of all optimizers from wd_start to the final value
    (training_config.optimizer.weight_decay), reaching it after ramp_fraction of all steps and
    keeping it constant afterwards. Policies: "linear" or "cosine" (half-cosine ramp).
    Each param group is scaled relative to its initial weight decay, so groups without weight
    decay (optimizer.no_weight_decay) stay at zero.

    Config (training_config.optimizer.weight_decay_schedule):
        policy: linear | cosine
        wd_start: 0.0
        ramp_fraction: 1.0
    """

    def __init__(
        self,
        optimizers: list[torch.optim.Optimizer],
        wd_final: float,
        schedule_cfg,
        n_steps: int,
        istep: int,
    ):
        self.optimizers = optimizers
        self.wd_final = wd_final
        self.wd_start = schedule_cfg.get("wd_start", 0.0)
        self.policy = schedule_cfg.get("policy", "linear")
        assert self.policy in ("linear", "cosine"), f"unsupported wd policy {self.policy}"
        self.n_steps_ramp = max(1, int(schedule_cfg.get("ramp_fraction", 1.0) * n_steps))
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group.setdefault("initial_weight_decay", group["weight_decay"])

        if is_root():
            logger.info(
                f"Weight decay schedule: {self.policy} from {self.wd_start} to {self.wd_final} "
                f"over {self.n_steps_ramp} steps"
            )
        self.step(istep)

    def get_wd(self, istep: int) -> float:
        t = min(1.0, istep / self.n_steps_ramp)
        if self.policy == "cosine":
            t = 0.5 * (1.0 - np.cos(np.pi * t))
        return self.wd_start + t * (self.wd_final - self.wd_start)

    def step(self, istep: int) -> float:
        wd = self.get_wd(istep)
        factor = wd / self.wd_final if self.wd_final != 0.0 else 1.0
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group["weight_decay"] = group["initial_weight_decay"] * factor
        return wd
