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


def ddp_average_nan(data: torch.Tensor) -> torch.Tensor:
    """
    Average a tensor across DDP ranks, excluding NaN and 0. when computing average

    Params:
        data: tensor to be averaged (arbitrary shape)

    Return :
        tensor with same shape as data, but entries averaged across all DDP ranks
    """
    if _is_distributed_initialized():
        # set NaNs to zero and communicate the vals
        mask = torch.isnan(data)
        data[mask] = 0.0
        dist.all_reduce(data.cuda(), op=torch.distributed.ReduceOp.SUM)
        # communicate the number of non-Nan values across ranks
        # also treat 0.0 as a nan-type value since loss was not computed there (e.g. empty
        # target) and it should not be taken into account when computing mean
        mask = torch.logical_and(mask, data == 0.0)
        sizes = mask.to(torch.int32)
        dist.all_reduce(sizes.cuda(), op=torch.distributed.ReduceOp.SUM)
        data = data / sizes
    return data.cpu()
