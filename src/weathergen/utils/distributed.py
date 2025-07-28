"""
Utilities for writing distributed pytorch-based code.

This module is adapted from code by Seb Hoffamn at:
https://github.com/sehoffmann/dmlcloud/blob/develop/dmlcloud/core/distributed.py

(same license as the rest of the code)
Copyright (c) 2025, Sebastian Hoffmann
"""

# TODO: copy other utilities from dmlcloud such as root_wrap etc.
# TODO: move the DDP code from trainer.py to this file

import torch
import torch.distributed as dist

SYNC_TIMEOUT_SEC = 60 * 60  # 1 hour


def is_root(pg: dist.ProcessGroup | None = None) -> bool:
    """
    Check if the current rank is the root rank (rank 0).

    Args:
        group (ProcessGroup, optional): The process group to work on.
        If None (default), the default process group will be used.
    """
    if not _is_distributed_initialized():
        # If not initialized, it assumed to be in single process mode.
        # TODO: check what should happen if a process group is passed
        return True
    return dist.get_rank(pg) == 0


def _is_distributed_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """
    Get MPI world size

    Returns:
        int: world size
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Get current rank number

    Returns:
        int: current rank
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def ddp_average(data: torch.Tensor) -> torch.Tensor:
    """
    Average a tensor across DDP ranks

    Params:
        data: tensor to be averaged (arbitrary shape)

    Return :
        tensor with same shape as data, but entries averaged across all DDP ranks
    """
    if _is_distributed_initialized():
        dist.all_reduce(data.cuda(), op=torch.distributed.ReduceOp.AVG)
    return data.cpu()


def all_gather_vlen(tensor: torch.Tensor, group=None) -> list[torch.Tensor]:
    """Gather tensors with the same number of dimensions but different lengths."""

    if not _is_distributed_initialized() :
        return [tensor]

    world_size = dist.get_world_size(group=group)

    # Gather lengths first
    shape = torch.as_tensor(tensor.shape, device=tensor.device)
    shapes = [torch.empty_like(shape) for _ in range(world_size)]
    dist.all_gather(shapes, shape, group=group)

    # Gather data
    inputs = [tensor] * world_size
    outputs = [torch.empty(*_shape, dtype=tensor.dtype, device=tensor.device) for _shape in shapes]
    dist.all_to_all(outputs, inputs, group=group)
    
    return outputs


def all_gather_vdim(tensor: torch.Tensor, group=None) -> list[torch.Tensor]:
    """Gather tensors with different number of dimensions."""
    
    if not _is_distributed_initialized() :
        return [tensor]
    
    world_size = dist.get_world_size(group=group)
    
    # Gather shapes first
    shapes = all_gather_vlen(torch.as_tensor(tensor.shape, device=tensor.device), group=group)
    
    # Gather data
    inputs = [tensor] * world_size
    outputs = [torch.empty(*_shape, dtype=tensor.dtype, device=tensor.device) for _shape in shapes]
    dist.all_to_all(outputs, inputs, group=group)
    
    return outputs
