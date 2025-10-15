# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch


class EMANeuralNet:
    """
    Taken and modified from https://github.com/NVlabs/edm2/tree/main
    """

    @torch.no_grad()
    def __init__(
        self, net, empty_net, halflife_steps=float("inf"), rampup_ratio=0.09, is_model_sharded=False
    ):
        self.og_net = net
        self.halflife_steps = halflife_steps
        self.rampup_ratio = rampup_ratio
        self.ema_net = empty_net
        self.is_model_sharded = is_model_sharded

        self.reset()

    @torch.no_grad()
    def reset(self):
        self.ema_net.to_empty(device="cuda")
        maybe_sharded_sd = self.og_net.state_dict()
        # this copies correctly tested in pdb
        mkeys, ukeys = self.ema_net.load_state_dict(maybe_sharded_sd, strict=False, assign=False)

    @torch.no_grad()
    def update(self, cur_steps, batch_size):
        halflife_steps = self.halflife_steps
        if self.rampup_ratio is not None:
            halflife_steps = min(halflife_steps, cur_steps / 1e3 * self.rampup_ratio)
        beta = 0.5 ** (batch_size / max(halflife_steps * 1e3, 1e-6))
        for p_net, p_ema in zip(self.og_net.parameters(), self.ema_net.parameters(), strict=False):
            p_ema.lerp_(p_net, 1 - beta)

    @torch.no_grad()
    def forward_eval(self, *args, **kwargs):
        self.ema_net.eval()
        out = self.ema_net(*args, **kwargs)
        self.ema_net.train()
        return out

    def state_dict(self):
        return self.ema_net.state_dict()

    def load_state_dict(self, state, **kwargs):
        self.ema_net.load_state_dict(state, **kwargs)
