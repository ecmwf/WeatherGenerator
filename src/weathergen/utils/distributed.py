"""
Utilities for writing distributed pytorch-based code.

This module is adapted from code by Seb Hoffamn at:
https://github.com/sehoffmann/dmlcloud/blob/develop/dmlcloud/core/distributed.py

(same license as the rest of the code)
Copyright (c) 2025, Sebastian Hoffmann
"""

import atexit
import inspect
import os
import random
import sys
from contextlib import contextmanager
from datetime import timedelta
from functools import wraps
from typing import Callable, TYPE_CHECKING

import numpy as np
import torch
import torch.distributed
import torch.distributed as dist

SYNC_TIMEOUT_SEC = 60 * 60  # 1 hour

def is_root(pg: dist.ProcessGroup | None = None) -> bool:
    """
    Check if the current rank is the root rank (rank 0).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None (default), the default process group will be used.
    """
    return dist.get_rank(pg) == 0


def root_only(
    fn: Callable,
    synchronize: bool = True,
    timeout: int = SYNC_TIMEOUT_SEC,
) -> Callable :
    """
    Decorator for methods that should only be called on the root rank.

    If ``synchronize=True``, a monitored_barrier before or after the function call depending on the rank.
    This can be important to prevent timeouts from future all_reduce operations if non-root ranks move on before the root rank has finished.

    Args:
        fn: The function to decorate
        synchronize: If True, a barrier is inserted before or after the function call depending on the rank. Default is True.
        timeout: Timeout in seconds for the monitored_barrier. Default is 24 hours.

    Returns:
        The decorated function or class.

    Examples:

        Annotating an individual function:

        >>> @root_only
        >>> def my_function():
        >>>     print('Only the root rank prints this.')

    """
    pg = dist.group.WORLD # No process group for now
    
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if is_root(pg):
            ret = fn(*args, **kwargs)
            if synchronize:
                dist.monitored_barrier(pg, timeout=timedelta(seconds=timeout), wait_all_ranks=True)
            return ret
        elif synchronize:
            dist.monitored_barrier(pg, timeout=timedelta(seconds=timeout), wait_all_ranks=True)

    return wrapper
