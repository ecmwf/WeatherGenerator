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
_SPATIAL_GROUPS: dict[int, tuple[dist.ProcessGroup, int]] = {}


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


def get_spatial_parallel_size(cf) -> int:
    """Return and validate the configured spatial-parallel size."""

    size = int(cf.get("spatial_parallel_size", 1))
    if size < 1:
        raise ValueError("spatial_parallel_size must be at least 1")

    world_size = get_world_size()
    if size > world_size:
        raise ValueError(
            f"spatial_parallel_size ({size}) exceeds world_size ({world_size})"
        )
    if world_size % size:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by spatial_parallel_size ({size})"
        )
    return size


def get_spatial_parallel_group(cf) -> tuple[dist.ProcessGroup | None, int]:
    """Create the consecutive-rank process groups used to shard HEALPix cells.

    All ranks call ``new_group`` in the same order. The returned rank is local to
    the spatial group. A size of one deliberately avoids creating a process group.
    """

    size = get_spatial_parallel_size(cf)
    if size == 1:
        return None, 0
    if not _is_distributed_initialized():
        raise RuntimeError("spatial parallelism requires torch.distributed")

    cached = _SPATIAL_GROUPS.get(size)
    if cached is not None:
        return cached

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    own_group = None
    for first_rank in range(0, world_size, size):
        ranks = list(range(first_rank, first_rank + size))
        group = dist.new_group(ranks=ranks)
        if global_rank in ranks:
            own_group = group

    assert own_group is not None
    result = (own_group, global_rank % size)
    _SPATIAL_GROUPS[size] = result
    return result


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
