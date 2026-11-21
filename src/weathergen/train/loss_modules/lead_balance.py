# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Lead-time balancing of the physical training loss (`forecast.lead_balance` in the training config).

Forecast errors grow with lead time, so in a loss that sums over output steps the long lead times
dominate the gradient: with pushforward over 10 days, one 10-day step of z_500 carries ~200 times
the MSE of the first 6h step. Lead balancing divides the loss of each channel at each output step
by its expected growth relative to the first step,

    weight(stream, channel, step) = growth[stream][channel][step - first_step] ** (-power),

on top of the channel weights. With power 1 every scored step contributes about as much as the
first step, with the channel mix of the first step; power 0 disables the balancing. The growth
table (e.g. made with scripts/compute_lead_balance.py from the per-step training losses of a
rollout run) is part of the config, so each run records the table it used:

    training_config:
      forecast:
        lead_balance:
          power: 1.0
          first_step: 1     # output step of the first entry of each growth list
          growth:
            ERA5:
              z_500: [1.0, 1.25, ...]

Only training is balanced; validation losses stay unweighted.
"""

import logging
import math

import torch

_logger = logging.getLogger(__name__)


class LeadBalance:
    """Per output step channel weights from a table of expected loss growth with lead time."""

    def __init__(self, growth: dict, power: float = 1.0, first_step: int = 1):
        assert isinstance(first_step, int) and first_step >= 0, (
            f"lead_balance.first_step must be a non-negative integer, got {first_step}."
        )
        assert isinstance(power, int | float) and math.isfinite(power) and power >= 0.0, (
            f"lead_balance.power must be a finite non-negative number, got {power}."
        )
        assert len(growth) > 0, "lead_balance.growth is empty."
        self.power = float(power)
        self.first_step = first_step
        self.growth = {}
        for stream_name, channels in growth.items():
            assert len(channels) > 0, f"lead_balance.growth.{stream_name} is empty."
            self.growth[stream_name] = {}
            for ch, values in channels.items():
                values = [float(v) for v in values]
                assert len(values) > 0, f"lead_balance.growth.{stream_name}.{ch} is empty."
                assert all(math.isfinite(v) and v > 0.0 for v in values), (
                    f"lead_balance.growth.{stream_name}.{ch} must be finite and positive."
                )
                self.growth[stream_name][ch] = values
        # weights per (stream, channels, device): tensor [num_steps, num_channels]
        self._cache = {}

    def weights(self, stream_name: str, channels: list[str], step: int, device) -> torch.Tensor:
        """Weights of the given channels (in this order) at the output step."""
        key = (stream_name, tuple(channels), str(device))
        if key not in self._cache:
            self._cache[key] = self._table(stream_name, channels).to(device)
        table = self._cache[key]
        idx = step - self.first_step
        assert 0 <= idx < table.shape[0], (
            f"lead_balance has no entry for output step {step} of stream {stream_name} "
            f"(steps {self.first_step} to {self.first_step + table.shape[0] - 1})."
        )
        return table[idx]

    def _table(self, stream_name: str, channels: list[str]) -> torch.Tensor:
        assert stream_name in self.growth, (
            f"lead_balance.growth has no entry for stream {stream_name}."
        )
        growth = self.growth[stream_name]
        missing = [ch for ch in channels if ch not in growth]
        assert len(missing) == 0, (
            f"lead_balance.growth.{stream_name} has no entry for channels {missing}."
        )
        num_steps = {len(growth[ch]) for ch in channels}
        assert len(num_steps) == 1, (
            f"lead_balance.growth.{stream_name}: all channels need the same number of steps."
        )
        unused = sorted(set(growth) - set(channels))
        if len(unused) > 0:
            _logger.info(f"lead_balance.growth.{stream_name}: unused channels {unused}.")
        # [num_steps, num_channels]
        values = torch.tensor([growth[ch] for ch in channels], dtype=torch.float64).T
        return values.pow(-self.power).to(torch.float32)


def get_lead_balance(forecast_cfg) -> LeadBalance | None:
    """The lead balancing of a forecast config, or None if it is not configured."""
    cfg = forecast_cfg.get("lead_balance", None) if forecast_cfg else None
    if cfg is None:
        return None
    assert "growth" in cfg, "lead_balance needs a growth table."
    return LeadBalance(
        growth=cfg["growth"], power=cfg.get("power", 1.0), first_step=cfg.get("first_step", 1)
    )
