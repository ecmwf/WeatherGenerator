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
import torch

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


def str_to_tensor(s: str, max_len: int, device="cpu") -> torch.Tensor:
    """Encodes a string into a tensor of ASCII values, padded to max_len."""
    ascii_vals = [ord(c) for c in s]
    # Pad with 0 (null character)
    padded_vals = ascii_vals + [0] * (max_len - len(ascii_vals))
    return torch.tensor(padded_vals, dtype=torch.int32, device=device)


def tensor_to_str(t: torch.Tensor) -> str:
    """Decodes a tensor of ASCII values back to a string."""
    # Filter out null padding characters
    ascii_vals = [i for i in t.tolist() if i != 0]
    return "".join([chr(c) for c in ascii_vals])
