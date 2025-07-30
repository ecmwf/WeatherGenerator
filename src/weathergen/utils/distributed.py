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
import os

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

def get_rank(default=0):
    """
    Attempts to determine the rank (i.e., the process ID in a parallel job)
    by checking common environment variables used by various HPC schedulers.

    Parameters:
        default (int): Default rank value to return if no environment variable is found.

    Returns:
        int: The detected rank or the default value.
    """
    var_list = [
        "SLURM_PROCID", # SLURM
        "PMI_RANK", # Intel MPI
        "OMPI_COMM_WORLD_RANK", # Open MPI
        #"MP_CHILD"
        #"RANK",
    ]
    for var in var_list:
        value = os.getenv(var)
        if value is not None:
            return int(value)
    return int(default)

def get_size(default=1):
    """
    Attempts to determine the total number of processes (world size)
    by checking common environment variables used by various HPC schedulers.

    Parameters:
        default (int): Default size value to return if no environment variable is found.

    Returns:
        int: The detected world size or the default value.
    """
    var_list = [
        "SLURM_NTASKS", # SLURM
        "PMI_SIZE", # Intel MPI
        "OMPI_COMM_WORLD_SIZE", # Open MPI
        #"MP_PROCS"
        #"SIZE",
        #"WORLD_SIZE",
    ]
    for var in var_list:
        value = os.getenv(var)
        if value is not None:
            return int(value)
    return int(default)

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
        dist.all_reduce(data.cuda(), op=dist.ReduceOp.AVG)
    return data.cpu()


def all_gather_vlen(tensor: torch.Tensor, group=None) -> list[torch.Tensor]:
    """Gather tensors with the same number of dimensions but different lengths."""

    if not _is_distributed_initialized():
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

    if not _is_distributed_initialized():
        return [tensor]

    world_size = dist.get_world_size(group=group)

    # Gather shapes first
    shapes = all_gather_vlen(torch.as_tensor(tensor.shape, device=tensor.device), group=group)

    # Gather data
    inputs = [tensor] * world_size
    outputs = [torch.empty(*_shape, dtype=tensor.dtype, device=tensor.device) for _shape in shapes]
    dist.all_to_all(outputs, inputs, group=group)

    return outputs

