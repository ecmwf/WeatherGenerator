# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


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
    if not _is_distributed_initialized():
        return 1

    return dist.get_world_size()


def get_rank() -> int:
    """
    Get current rank number

    Returns:
        int: current rank
    """
    if not _is_distributed_initialized():
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
        dist.all_reduce(data, op=dist.ReduceOp.AVG)
    return data.cpu()
