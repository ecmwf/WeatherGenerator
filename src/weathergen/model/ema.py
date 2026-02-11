# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch


class EMAModel:
    """
    Taken and modified from https://github.com/NVlabs/edm2/tree/main
    """

    @torch.no_grad()
    def __init__(
        self,
        model,
        empty_model,
        halflife_steps=float("inf"),
        rampup_ratio=0.09,
        is_model_sharded=False,
    ):
        self.original_model = model
        self.halflife_steps = halflife_steps
        self.rampup_ratio = rampup_ratio
        self.ema_model = empty_model
        self.is_model_sharded = is_model_sharded
        # Build a name → param map once
        self.src_params = dict(self.original_model.named_parameters())

        self.reset()

    @torch.no_grad()
    def reset(self):
        """
        This function resets the EMAModel to be the same as the Model.

        It operates via the state_dict to be able to deal with sharded tensors in case
        FSDP2 is used.
        """
        self.ema_model.to_empty(device="cuda")
        for p in self.ema_model.parameters():
            p.requires_grad = False
        maybe_sharded_sd = self.original_model.state_dict()
        # this copies correctly tested in pdb
        mkeys, ukeys = self.ema_model.load_state_dict(maybe_sharded_sd, strict=False, assign=False)
        self.ema_model.eval()

    def requires_grad_(self, flag: bool):
        for p in self.ema_model.parameters():
            p.requires_grad = flag

    @torch.no_grad()
    def update(self, cur_step, batch_size):
        # ensure model remains sharded
        if self.is_model_sharded:
            self.ema_model.reshard()
        # determine correct interpolation params
        halflife_steps = self.halflife_steps
        if self.rampup_ratio is not None:
            halflife_steps = min(halflife_steps, cur_step / 1e3 * self.rampup_ratio)
        beta = 0.5 ** (batch_size / max(halflife_steps * 1e3, 1e-6))

        for name, p_ema in self.ema_model.named_parameters():
            p_src = self.src_params.get(name, None)
            # Due to DDP only being applied only to the student the names may missmatch
            # Thus, we check for the alternate naming scheme
            p_src = self.src_params.get("module." + name, None) if p_src is None else p_src
            if "identity" in name.lower() or "q_cells" in name.lower():
                continue
            if p_src is None:
                # EMA-only param or intentionally excluded
                assert False, f"{name}: All parameters of the EMA model must be in the base model."

            p_ema.lerp_(p_src, 1.0 - beta)

    @torch.no_grad()
    def forward_eval(self, *args, **kwargs):
        self.ema_model.eval()
        out = self.ema_model(*args, **kwargs)
        return out

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state, **kwargs):
        self.ema_model.load_state_dict(state, **kwargs)
