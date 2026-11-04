# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared tensor stat primitives (fp32 accumulation, no full fp32 copies of the input)."""

import contextlib
import math

import torch

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover - torch without distributed support
    DTensor = None


def local_tensor(t: torch.Tensor) -> torch.Tensor:
    """Local shard of a (possibly FSDP2 DTensor) tensor, detached. Never gathers."""
    if DTensor is not None and isinstance(t, DTensor):
        t = t.to_local()
    return t.detach()


@contextlib.contextmanager
def stats_context(device: torch.device):
    """no_grad and autocast disabled, for stats computed inside the (autocast) forward:
    autocast would otherwise run the probe's q·k in bf16 or upcast whole activations."""
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        yield


def sumsq(x: torch.Tensor) -> torch.Tensor:
    """Sum of squares as a 0-d fp32 tensor (fp32 accumulation via vector_norm's dtype)."""
    return torch.linalg.vector_norm(x, dtype=torch.float32).square()


def rms(x: torch.Tensor) -> torch.Tensor:
    """Root mean square over all elements as a 0-d fp32 tensor."""
    x = local_tensor(x)
    if x.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=x.device)
    return torch.linalg.vector_norm(x, dtype=torch.float32) / math.sqrt(x.numel())


def absmax(x: torch.Tensor) -> torch.Tensor:
    """max |x| as a 0-d fp32 tensor, without materialising |x|. x must be non-empty."""
    return torch.maximum(x.amax(), x.amin().neg()).float()


def foreach_sumsq_absmax(
    tensors: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-tensor sum of squares and absmax (fp32) with fused foreach kernels.

    All tensors must be non-empty (the inf-norm of an empty tensor is undefined).
    """
    if not tensors:
        return [], []
    l2 = torch._foreach_norm(tensors, 2, dtype=torch.float32)
    amax = torch._foreach_norm(tensors, float("inf"), dtype=torch.float32)
    return [n.square() for n in l2], list(amax)
