# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Monitor base class, registry and the fused-reduction `StatBuffer`."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.distributed as dist

from weathergen.train.monitoring.selection import Selection

_REDUCE_OPS = ("sum", "max", "min")
_IDENTITY = {"sum": 0.0, "max": -math.inf, "min": math.inf}

_REGISTRY: dict[str, type[Monitor]] = {}


def register_monitor(name: str):
    """Class decorator: `@register_monitor("weights")` makes the monitor configurable by name."""

    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"Monitor '{name}' is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_registry() -> dict[str, type[Monitor]]:
    return dict(_REGISTRY)


def _merge(op: str, old, new):
    """Combine two partial stats (python numbers or 0-d tensors) with a reduce op."""
    if op == "sum":
        return old + new
    if not isinstance(old, torch.Tensor) and not isinstance(new, torch.Tensor):
        return max(old, new) if op == "max" else min(old, new)
    if not isinstance(old, torch.Tensor):
        old, new = new, old
    if not isinstance(new, torch.Tensor):
        return old.clamp(min=new) if op == "max" else old.clamp(max=new)
    return torch.maximum(old, new) if op == "max" else torch.minimum(old, new)


class StatBuffer:
    """Per-step accumulator of partial stats, reduced across ranks in one collective per op.

    Raw entries are 0-d tensors (kept on device) or python numbers, each with a reduce op
    (sum / max / min). Adding to an existing key combines with that op, so stats accumulate
    across modules of a group, across repeated calls in one step, and across ranks.
    Output metrics are functions of the reduced raw values, evaluated on the host.

    Raw keys are reduced in sorted order, so every rank must add the same *set* of keys
    (callers register identity values for groups they have no data for), but neither the
    insertion order nor python-vs-tensor values need to match across ranks.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._raw: dict[str, tuple[str, object]] = {}
        self._outputs: dict[str, Callable[[dict[str, float]], float | None]] = {}

    def __len__(self):
        return len(self._raw)

    def accumulate(self, key: str, value, op: str) -> None:
        assert op in _REDUCE_OPS, f"unknown reduce op '{op}'"
        if isinstance(value, torch.Tensor):
            value = value.detach().float().reshape(())
        prev = self._raw.get(key)
        if prev is None:
            self._raw[key] = (op, value)
        else:
            assert prev[0] == op, f"stat '{key}' added with ops '{prev[0]}' and '{op}'"
            self._raw[key] = (op, _merge(op, prev[1], value))

    def register(self, key: str, op: str) -> None:
        """Ensure `key` exists (with the op's identity: 0 for sum, ∓inf for max/min), so that
        all ranks reduce the same set of keys even when one has no data for it."""
        if key not in self._raw:
            self.accumulate(key, _IDENTITY[op], op)

    def output(self, key: str, fn: Callable[[dict[str, float]], float | None]) -> None:
        """Register a logged metric computed from reduced raw values (None = not logged)."""
        self._outputs[key] = fn

    def reduce(self) -> dict[str, float]:
        """All-reduce raw stats (one collective per op, one D2H copy) and compute outputs."""
        if not self._raw:
            return {}
        distributed = dist.is_available() and dist.is_initialized()
        reduced, keys = [], []
        for op in _REDUCE_OPS:
            op_keys = sorted(k for k, (o, _) in self._raw.items() if o == op)
            if not op_keys:
                continue
            is_dev = [isinstance(self._raw[k][1], torch.Tensor) for k in op_keys]
            dev_idx = [i for i, d in enumerate(is_dev) if d]
            py_idx = [i for i, d in enumerate(is_dev) if not d]
            parts = []
            if dev_idx:
                parts.append(
                    torch.stack([self._raw[op_keys[i]][1].to(self.device) for i in dev_idx])
                )
            if py_idx:
                py_vals = [float(self._raw[op_keys[i]][1]) for i in py_idx]
                parts.append(torch.tensor(py_vals, dtype=torch.float32, device=self.device))
            vec = torch.cat(parts) if len(parts) > 1 else parts[0]
            if dev_idx and py_idx:
                # restore sorted key order so that the layout is identical on all ranks
                inv = [0] * len(op_keys)
                for pos, i in enumerate(dev_idx + py_idx):
                    inv[i] = pos
                vec = vec[torch.tensor(inv, device=self.device)]
            if distributed:
                dist.all_reduce(vec, op=getattr(dist.ReduceOp, op.upper()))
            reduced.append(vec)
            keys.extend(op_keys)
        values = torch.cat(reduced).cpu().tolist()
        raw = dict(zip(keys, values, strict=True))

        metrics = {}
        for key, fn in self._outputs.items():
            value = fn(raw)
            if value is not None:
                metrics[key] = value
        return metrics


def sqrt_or_none(x: float | None) -> float | None:
    return None if x is None else math.sqrt(max(x, 0.0))


class Monitor:
    """Base class of training-time monitors; all callbacks are optional no-ops and run only
    on steps where the monitor is active.

    Subclasses set stats in a `StatBuffer` via `buf.accumulate` / `buf.output`; logged keys
    are `mon/<name>/<group>/<stat>`.
    """

    name: str = "monitor"

    def __init__(self, cfg, model: torch.nn.Module, selection: Selection):
        self.cfg = cfg
        self.model = model
        self.selection = selection
        self.every = int(cfg.every)
        if self.every <= 0:
            raise ValueError(f"monitoring.monitors.{self.name}.every must be > 0")

    def bind_optimizers(self, optimizers: list[torch.optim.Optimizer]) -> None:
        """Called once after construction; monitors that read optimizer state override this."""

    def is_active(self, istep: int) -> bool:
        return istep % self.every == 0

    def key(self, group: str, stat: str) -> str:
        return f"{self.name}/{group}/{stat}"

    # lifecycle callbacks (no-ops by default)
    def on_forward_start(self, buf: StatBuffer) -> None:
        """Register hooks / enable probes. Called on entering `manager.forward()`."""

    def on_forward_end(self) -> None:
        """Remove hooks / reset probes. Called on leaving `manager.forward()` (also on error)."""

    def on_after_clip(self, buf: StatBuffer, total_norm) -> None:
        """Grads are unscaled and clipped; weights are still pre-step."""

    def on_step_end(self, buf: StatBuffer) -> None:
        """After optimizer step, LR and EMA updates; before the fused reduction."""

    def reset(self) -> None:
        """Release per-step state (hooks, probes, snapshots). Must not run collectives."""
        self.on_forward_end()
