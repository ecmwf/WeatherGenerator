"""
Utilities for writing distributed pytorch-based code.

This module is adapted from code by Seb Hoffamn at:
https://github.com/sehoffmann/dmlcloud/blob/develop/dmlcloud/core/distributed.py

(same license as the rest of the code)
Copyright (c) 2025, Sebastian Hoffmann
"""

# TODO: copy other utilities from dmlcloud such as root_wrap etc.
# TODO: move the DDP code from trainer.py to this file

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