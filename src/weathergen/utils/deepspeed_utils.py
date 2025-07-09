# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os

import deepspeed
import torch
import torch.distributed as dist

from weathergen.utils.distributed import tensor_to_str, str_to_tensor, is_root

_logger = logging.getLogger(__name__)


def setup_deepspeed_environment(cf):
    """
    Sets up environment variables required for torch.distributed,
    especially in a Slurm environment.
    """
    print("Setting up for deepspeed ZeRO stage 3")
    if "SLURM_PROCID" in os.environ:
        # This is a Slurm job, set up for torch.distributed
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

        # Get master address from the first node in the Slurm nodelist
        try:
            nodelist = os.environ["SLURM_JOB_NODELIST"]
            # This logic might need adjustment based on the nodelist format (e.g., node[1-4])
            # A simple approach is to take the first host listed.
            if "[" in nodelist:
                base = nodelist.split("[")[0]
                num_part = nodelist.split("[")[1].split("]")[0]
                master_node = base + num_part.split(",")[0]
            else:
                master_node = nodelist.split(",")[0]
            os.environ["MASTER_ADDR"] = master_node
        except KeyError:
            _logger.error("SLURM_JOB_NODELIST is not set. Cannot determine MASTER_ADDR.")
            # Fallback or exit can be implemented here

        # Set a default master port
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"  # Default port

    # Initialize the process group if it's not already initialized
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        _logger.info(
            f"Initialized process group with backend 'nccl'. "
            f"Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}"
        )

    # Set the CUDA device for the current process. This is crucial for multi-GPU nodes.
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        _logger.info(f"Rank {dist.get_rank()} set to use CUDA device {local_rank}")

    run_id_tensor = str_to_tensor(cf.run_id, max_len=128).to("cuda")  # Use a helper
    dist.broadcast(run_id_tensor, src=0)
    if not is_root():
        cf.run_id = tensor_to_str(run_id_tensor)  # Use a helper
    _logger.info(f"Rank {dist.get_rank()} synchronized run_id: {cf.run_id}")

    # Broadcast RNG seed
    if hasattr(cf, "data_loader_rng_seed") and cf.data_loader_rng_seed is not None:
        seed_tensor = torch.tensor([cf.data_loader_rng_seed], dtype=torch.int32).to("cuda")
        dist.broadcast(seed_tensor, src=0)
        cf.data_loader_rng_seed = seed_tensor.item()

    cf.rank = dist.get_rank()
    cf.num_ranks = dist.get_world_size()
    cf.with_ddp = True

    return


def get_deepspeed_config(
    batch_size: int,
    gradient_accumulation_steps: int,
    zero_stage: int,
    bf16_enabled: bool,
    gradient_clipping: float,
    zero_optimization_cpu_offload: bool,
    zero_optimization_cpu_offload_params: bool,
):
    """
    Generates a DeepSpeed configuration dictionary.
    """
    config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {},  # Params are controlled by the optimizer passed to initialize
        },
        "bf16": {
            "enabled": bf16_enabled,
        },
        "fp16": {
            "enabled": False,
        },
        "gradient_clipping": gradient_clipping,
        "zero_optimization": {
            "stage": zero_stage,
            "offload_optimizer": {
                "device": "cpu" if zero_optimization_cpu_offload else "none",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu" if zero_optimization_cpu_offload_params else "none",
                "pin_memory": True,
            },
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "checkpoint": {
            "consolidate_checkpoint_files": False,
            "use_node_local_storage": True,
        },
        "steps_per_print": 2000,  # Suppress verbose logging from deepspeed
        "wall_clock_breakdown": False,
    }
    return config


def initialize_deepspeed(model, optimizer, lr_scheduler, config):
    """
    Initializes the DeepSpeed engine.
    """
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config_params=config,
    )
    return model_engine, optimizer, lr_scheduler


def save_deepspeed_checkpoint(deepspeed_engine, base_path, epoch, global_step, client_state):
    """
    Saves a DeepSpeed checkpoint.
    """
    tag = f"epoch{epoch:05d}"
    _logger.info(f"Saving DeepSpeed checkpoint for epoch {epoch} (tag: {tag}) to {base_path}")
    deepspeed_engine.save_checkpoint(
        save_dir=base_path,
        tag=tag,
        client_state=client_state,
    )


def load_deepspeed_checkpoint(
    deepspeed_engine, checkpoint_path, epoch, load_optimizer_states, load_lr_scheduler_states
):
    """
    Loads a DeepSpeed checkpoint.
    """
    tag = f"epoch{epoch:05d}"
    _logger.info(
        f"Loading DeepSpeed checkpoint for epoch {epoch} (tag: {tag}) from {checkpoint_path}"
    )
    load_path, client_state = deepspeed_engine.load_checkpoint(
        load_dir=checkpoint_path,
        tag=tag,
        load_optimizer_states=load_optimizer_states,
        load_lr_scheduler_states=load_lr_scheduler_states,
    )
    if load_path is None:
        raise ValueError(
            f"Failed to load DeepSpeed checkpoint with tag {tag} from {checkpoint_path}"
        )
    return client_state
