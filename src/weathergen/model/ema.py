# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import copy

import torch


class EMA:
    """
    Taken and modified from https://github.com/NVlabs/edm2/tree/main
    """

    @torch.no_grad()
    def __init__(self, net, halflife_steps=float("inf"), rampup_ratio=0.09):
        self.net = net
        self.halflife_steps = halflife_steps
        self.rampup_ratio = rampup_ratio
        self.ema = copy.deepcopy(net)

    @torch.no_grad()
    def reset(self):
        for p_net, p_ema in zip(self.net.parameters(), self.ema.parameters(), strict=False):
            p_ema.copy_(p_net)

    @torch.no_grad()
    def update(self, cur_steps, batch_size):
        halflife_steps = self.halflife_steps
        if self.rampup_ratio is not None:
            halflife_steps = min(halflife_steps, cur_steps / 1e3 * self.rampup_ratio)
        beta = 0.5 ** (batch_size / max(halflife_steps * 1e3, 1e-6))
        for p_net, p_ema in zip(self.net.parameters(), self.ema.parameters(), strict=False):
            p_ema.lerp_(p_net, 1 - beta)

    @torch.no_grad()
    def forward_eval(self, *args, **kwargs):
        self.ema.eval()
        out = self.ema(*args, **kwargs)
        self.ema.train()
        return out

    @torch.no_grad()
    def get(self):
        for p_net, p_ema in zip(self.net.buffers(), self.ema.buffers(), strict=False):
            p_ema.copy_(p_net)
        return self.ema

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state):
        self.ema.load_state_dict(state)
