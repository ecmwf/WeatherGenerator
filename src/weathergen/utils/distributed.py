# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import dataclasses

import torch
import torch.distributed as dist

SYNC_TIMEOUT_SEC = 60 * 60  # 1 hour
_SPATIAL_GROUPS: dict[int, tuple[dist.ProcessGroup, int]] = {}


def normalize_distributed_config(cf):
    """Move legacy top-level distributed settings into their shared sections."""

    if cf.get("distributed") is None:
        cf["distributed"] = {}
    distributed_cfg = cf["distributed"]

    if distributed_cfg.get("data_parallel") is None:
        distributed_cfg["data_parallel"] = {}
    data_parallel_cfg = distributed_cfg["data_parallel"]
    if "with_ddp" not in data_parallel_cfg:
        data_parallel_cfg["with_ddp"] = cf.get("with_ddp", False)
    if "with_fsdp" not in data_parallel_cfg:
        data_parallel_cfg["with_fsdp"] = cf.get("with_fsdp", False)
    if "find_unused_parameters" not in data_parallel_cfg:
        data_parallel_cfg["find_unused_parameters"] = cf.get(
            "ddp_find_unused_parameters", True
        )
    if "world_size" not in data_parallel_cfg and cf.get("data_parallel_world_size") is not None:
        data_parallel_cfg["world_size"] = cf["data_parallel_world_size"]

    if distributed_cfg.get("spatial_parallel") is None:
        distributed_cfg["spatial_parallel"] = {}
    spatial_cfg = distributed_cfg["spatial_parallel"]
    if "size" not in spatial_cfg:
        spatial_cfg["size"] = cf.get("spatial_parallel_size", 1)
    if "size_original" not in spatial_cfg and cf.get("spatial_parallel_size_original") is not None:
        spatial_cfg["size_original"] = cf["spatial_parallel_size_original"]

    for legacy_key in (
        "with_ddp",
        "with_fsdp",
        "data_parallel_world_size",
        "ddp_find_unused_parameters",
        "spatial_parallel_size",
        "spatial_parallel_size_original",
    ):
        cf.pop(legacy_key, None)

    return cf


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


def get_spatial_parallel_size(spatial_cfg) -> int:
    """Return and validate the configured spatial-parallel size."""

    size = int(spatial_cfg.get("size", 1))
    if size < 1:
        raise ValueError("distributed.spatial_parallel.size must be at least 1")

    world_size = get_world_size()
    if size > world_size:
        raise ValueError(
            f"distributed.spatial_parallel.size ({size}) exceeds world_size ({world_size})"
        )
    if world_size % size:
        raise ValueError(
            "world_size "
            f"({world_size}) must be divisible by distributed.spatial_parallel.size ({size})"
        )
    return size


def get_spatial_parallel_group(spatial_cfg) -> tuple[dist.ProcessGroup | None, int]:
    """Create the consecutive-rank process groups used to shard HEALPix cells.

    All ranks call ``new_group`` in the same order. The returned rank is local to
    the spatial group. A size of one deliberately avoids creating a process group.
    """

    size = get_spatial_parallel_size(spatial_cfg)
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


@dataclasses.dataclass(frozen=True)
class SpatialParallelContext:
    """Distributed topology and HEALPix ownership for one spatial rank."""

    size: int
    rank: int
    group: dist.ProcessGroup | None
    ddp_rank: int
    ddp_world_size: int
    num_cells: int
    local_num_cells: int
    cell_start: int
    cell_end: int

    @classmethod
    def from_config(
        cls,
        cf,
        num_cells: int,
        *,
        create_process_group: bool = False,
    ) -> "SpatialParallelContext":
        """Build and validate the shared spatial-parallel topology."""

        spatial_cfg = cf.distributed.spatial_parallel
        size = get_spatial_parallel_size(spatial_cfg)
        if num_cells % size:
            raise ValueError(
                f"number of HEALPix cells ({num_cells}) must be divisible by "
                f"distributed.spatial_parallel.size ({size})"
            )

        rank = cf.rank % size
        group = None
        if create_process_group:
            group, group_rank = get_spatial_parallel_group(spatial_cfg)
            if group_rank != rank:
                raise RuntimeError(
                    f"spatial process-group rank ({group_rank}) does not match "
                    f"topology rank ({rank})"
                )

        local_num_cells = num_cells // size
        cell_start = rank * local_num_cells
        return cls(
            size=size,
            rank=rank,
            group=group,
            ddp_rank=cf.rank // size,
            ddp_world_size=cf.world_size // size,
            num_cells=num_cells,
            local_num_cells=local_num_cells,
            cell_start=cell_start,
            cell_end=cell_start + local_num_cells,
        )


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
