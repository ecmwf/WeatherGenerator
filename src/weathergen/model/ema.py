# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import math

import torch

logger = logging.getLogger(__name__)


class EMAModel:
    """
    Taken and modified from https://github.com/NVlabs/edm2/tree/main

    Optional halflife scheduling
    ----------------------------
    If ``halflife_end`` is provided, the effective halflife is continuously
    annealed from ``halflife_steps`` to ``halflife_end`` over
    ``halflife_ramp_steps`` optimisation steps.

    Three schedule types are available (``halflife_schedule_type``):

    - ``"log_linear"`` (default): linear interpolation in log-halflife space
      (i.e. geometric interpolation of the halflife value).  Natural when the
      halflife spans orders of magnitude.

    - ``"linear"``: linear interpolation in halflife space.  The halflife grows
      by constant absolute increments, so beta increases quickly early on (when
      the halflife is small relative to the increment) and levels off later.
      This front-loads the teacher stabilisation compared to log_linear.

    - ``"cosine_beta"``: cosine interpolation in *beta* space — this exactly
      reproduces the momentum schedule used by I-JEPA, V-JEPA, BYOL, and DINO.
      Those methods ramp beta from a base value toward 1.0 with the formula
      ``beta(t) = beta_start + (beta_end - beta_start) * (1 - cos(pi*t)) / 2``.
      We convert the configured halflife start/end to beta, apply that cosine
      interpolation, then convert back to halflife so the rest of the code
      (including ``rampup_ratio``) still operates on halflife.
    """

    @torch.no_grad()
    def __init__(
        self,
        model,
        empty_model,
        halflife_steps=float("inf"),
        halflife_end=None,
        halflife_ramp_steps=None,
        halflife_schedule_type="log_linear",
        rampup_ratio=0.09,
        is_model_sharded=False,
        random_init=False,
    ):
        self.original_model = model
        self.halflife_steps = halflife_steps
        self.halflife_end = halflife_end
        self.halflife_ramp_steps = halflife_ramp_steps
        self.halflife_schedule_type = halflife_schedule_type
        self.rampup_ratio = rampup_ratio
        self.ema_model = empty_model
        self.is_model_sharded = is_model_sharded
        self.batch_size = 1
        self._random_init = random_init
        # Build a name → param map once
        self.src_params = dict(self.original_model.named_parameters())

        if random_init:
            self._freeze_and_eval()
        else:
            self.reset()

        logger.info(
            "EMAModel initialised: halflife_start=%.1f, halflife_end=%s, "
            "halflife_ramp_steps=%s, schedule_type=%s, rampup_ratio=%s, random_init=%s",
            halflife_steps,
            halflife_end,
            halflife_ramp_steps,
            halflife_schedule_type,
            rampup_ratio,
            random_init,
        )

    @torch.no_grad()
    def _freeze_and_eval(self):
        """Freeze EMA model parameters and set to eval mode (without copying student weights)."""
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.ema_model.eval()

    @torch.no_grad()
    def reset(self):
        """
        This function resets the EMAModel to be the same as the Model.

        If random_init is active, only freezes and sets eval mode without copying
        student weights. Otherwise copies student weights as usual.

        It operates via the state_dict to be able to deal with sharded tensors in case
        FSDP2 is used.
        """
        if self._random_init:
            self._freeze_and_eval()
            return
        self.ema_model.to_empty(device="cuda")
        maybe_sharded_sd = self.original_model.state_dict()
        # Strip "module." prefix from DDP-wrapped student so keys match the unwrapped
        # teacher model. The update() method already handles this mismatch (line 73),
        # but load_state_dict needs matching keys upfront.
        ema_keys = set(self.ema_model.state_dict().keys())
        needs_strip = not any(k in ema_keys for k in maybe_sharded_sd)
        if needs_strip:
            maybe_sharded_sd = {k.removeprefix("module."): v for k, v in maybe_sharded_sd.items()}
        mkeys, ukeys = self.ema_model.load_state_dict(maybe_sharded_sd, strict=False, assign=False)
        self._freeze_and_eval()

    @torch.no_grad()
    def resync_to_student(self):
        """Force resync EMA model to current student weights, regardless of random_init flag."""
        self._random_init = False
        self.reset()

    def requires_grad_(self, flag: bool):
        for p in self.ema_model.parameters():
            p.requires_grad = flag

    def get_current_beta(self, cur_step: int) -> float:
        """
        Get current EMA beta value for monitoring.

        The beta value determines how much the teacher model is updated towards
        the student model at each step. Higher beta means slower teacher updates.

        Args:
            cur_step: Current training step (typically istep * batch_size).

        Returns:
            Current EMA beta value.
        """
        # Scheduled halflife: interpolate from halflife_steps → halflife_end
        if self.halflife_end is not None and self.halflife_ramp_steps is not None and self.halflife_ramp_steps > 0:
            t = min(cur_step / self.halflife_ramp_steps, 1.0)
            if self.halflife_schedule_type == "cosine_beta":
                # Cosine interpolation in beta space — matches I-JEPA / BYOL / DINO.
                # Convert start/end halflife → beta, cosine-interpolate, convert back.
                bs = self.batch_size
                beta_start = 0.5 ** (bs / max(self.halflife_steps, 1e-6))
                beta_end = 0.5 ** (bs / max(self.halflife_end, 1e-6))
                beta_t = beta_start + (beta_end - beta_start) * (1 - math.cos(math.pi * t)) / 2
                halflife_steps = bs * math.log(0.5) / math.log(min(beta_t, 1 - 1e-15))
            elif self.halflife_schedule_type == "linear":
                # Linear interpolation in halflife space.
                # Halflife grows by constant absolute increments, so beta increases
                # quickly early (when halflife is small) and slows down later.
                halflife_steps = self.halflife_steps + t * (self.halflife_end - self.halflife_steps)
            else:
                # "log_linear": geometric interpolation of halflife (linear in log-space)
                log_start = math.log(max(self.halflife_steps, 1e-6))
                log_end = math.log(max(self.halflife_end, 1e-6))
                halflife_steps = math.exp(log_start + t * (log_end - log_start))
        else:
            halflife_steps = self.halflife_steps

        if self.rampup_ratio is not None:
            halflife_steps = min(halflife_steps, cur_step / self.rampup_ratio)
        beta = 0.5 ** (self.batch_size / max(halflife_steps, 1e-6))

        return beta

    @torch.no_grad()
    def update(self, cur_step, batch_size):
        # ensure model remains sharded
        if self.is_model_sharded:
            self.ema_model.reshard()
        # determine correct interpolation params
        self.batch_size = batch_size
        beta = self.get_current_beta(cur_step)

        if self.halflife_ramp_steps and self.halflife_ramp_steps > 0:
            t = min(cur_step / self.halflife_ramp_steps, 1.0)
        else:
            t = None
        if cur_step == batch_size or (cur_step > 0 and cur_step % (batch_size * 1000) == 0):
            logger.info(
                "EMA update: cur_step=%d, batch_size=%d, beta=%.10f, schedule_t=%s",
                cur_step, batch_size, beta,
                f"{t:.4f}" if t is not None else "N/A",
            )

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
