# ruff: noqa: T201

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

import pynvml
import torch
import torch.distributed as dist
import torch.multiprocessing

from weathergen.common.config import Config
from weathergen.train.target_and_aux_diffusion import DiffusionLatentTargetEncoder
from weathergen.train.target_and_aux_module_base import PhysicalTargetAndAux
from weathergen.train.target_and_aux_ssl_teacher import EMATeacher
from weathergen.train.utils import str_to_tensor, tensor_to_str
from weathergen.utils.distributed import is_root

PORT = 1345


class TrainerBase:
    def __init__(self):
        self.device_handles = []
        self.device_names = []
        self.cf: Config | None = None

    @staticmethod
    def init_torch(use_cuda=True, num_accs_per_task=1, multiprocessing_method="fork"):
        """
        Initialize torch, set device and multiprocessing method.

        NOTE: If using the Nvidia profiler,
        the multiprocessing method must be set to "spawn".
        The default for linux systems is "fork",
        which prevents traces from being generated with DDP.
        """
        torch.set_printoptions(linewidth=120)

        # This strategy is required by the nvidia profiles
        # to properly trace events in worker processes.
        # This may cause issues with logging. Alternative: "fork"
        torch.multiprocessing.set_start_method(multiprocessing_method, force=True)

        torch.backends.cuda.matmul.allow_tf32 = True

        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            return torch.device("cpu")

        local_id_node = os.environ.get("SLURM_LOCALID", "-1")
        if local_id_node == "-1":
            devices = ["cuda"]
        else:
            devices = [
                f"cuda:{int(local_id_node) * num_accs_per_task + i}"
                for i in range(num_accs_per_task)
            ]
        torch.cuda.set_device(int(local_id_node) * num_accs_per_task)

        return devices

    @staticmethod
    def init_ddp(cf):
        """Initializes the distributed environment."""
        rank = 0
        local_rank = 0

        if not dist.is_available():
            print("Distributed training is not available.")
            return

        # dist.set_debug_level(dist.DebugLevel.DETAIL)
        world_size = int(os.environ.get("WORLD_SIZE", "-1"))
        if world_size == -1:
            # Called using SLURM instead of torchrun
            world_size = int(os.environ.get("SLURM_NTASKS", "1"))

        if not dist.is_initialized() and world_size > 1:
            # These environment variables are typically set by the launch utility
            # (e.g., torchrun, Slurm)
            local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
            if local_rank == -1:
                # Called using SLURM instead of torchrun
                local_rank = int(os.environ.get("SLURM_LOCALID"))
            rank = int(os.environ.get("RANK", "-1"))
            if rank == -1:
                ranks_per_node = int(os.environ.get("SLURM_TASKS_PER_NODE", "1")[0])
                rank = int(os.environ.get("SLURM_NODEID")) * ranks_per_node + local_rank
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", f"{PORT}")  # Default port

            if torch.accelerator.is_available():
                device_type = torch.accelerator.current_accelerator()
                device = torch.device(f"{device_type}:{local_rank}")
                torch.accelerator.set_device_index(local_rank)
                print(f"DDP initialization: device={device}, rank={rank}, world_size={world_size}")
            else:
                device = torch.device("cpu")
                print(f"Running on device {device}")

            backend = torch.distributed.get_default_backend_for_device(device)
            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                device_id=device,
                rank=rank,
                init_method=f"tcp://{master_addr}:{master_port}",
            )
            print(f"Process group initialized ({backend}).")

            if is_root():
                print("DDP initialized: root.")
            # Wait for all ranks to reach this point

            dist.barrier()
            # communicate run id to all nodes
            len_run_id = len(cf.run_id)
            run_id_int = torch.zeros(len_run_id, dtype=torch.int32).to(device)
            if is_root():
                print(f"Communicating run_id to all nodes: {cf.run_id}")
                run_id_int = str_to_tensor(cf.run_id).to(device)
            dist.all_reduce(run_id_int, op=torch.distributed.ReduceOp.SUM)
            if not is_root():
                cf.run_id = tensor_to_str(run_id_int)
            print(f"rank: {rank} has run_id: {cf.run_id}")

            # communicate data_loader_rng_seed
            if hasattr(cf, "data_loader_rng_seed"):
                if cf.data_loader_rng_seed is not None:
                    l_seed = torch.tensor(
                        [cf.data_loader_rng_seed if rank == 0 else 0], dtype=torch.int32
                    ).cuda()
                    dist.all_reduce(l_seed, op=torch.distributed.ReduceOp.SUM)
                    cf.data_loader_rng_seed = l_seed.item()

        cf.world_size = world_size
        cf.rank = rank
        cf.local_rank = local_rank
        cf.with_ddp = world_size > 1

        return cf

    def init_perf_monitoring(self):
        self.device_handles, self.device_names = [], []

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            self.device_names += [pynvml.nvmlDeviceGetName(handle)]
            self.device_handles += [handle]

    def get_perf(self):
        perf_gpu, perf_mem = 0.0, 0.0
        if len(self.device_handles) > 0:
            for handle in self.device_handles:
                perf = pynvml.nvmlDeviceGetUtilizationRates(handle)
                perf_gpu += perf.gpu
                perf_mem += perf.memory
            perf_gpu /= len(self.device_handles)
            perf_mem /= len(self.device_handles)

        return perf_gpu, perf_mem


# should be moved to its own file so as to prevent cyclical imports
def get_target_and_aux_calculator(config, model, rng, batch_size, **kwargs):
    target_and_aux_calc = config.get("training_mode_config", None).get("target_and_aux_calc", None)
    if target_and_aux_calc is None or target_and_aux_calc == "identity":
        return PhysicalTargetAndAux(model, rng, config)
    elif target_and_aux_calc == "EMATeacher":
        return EMATeacher(model, rng, kwargs["ema_model"], batch_size)
    elif target_and_aux_calc == "DiffusionLatentTargetEncoder":
        return DiffusionLatentTargetEncoder(model)
    else:
        raise NotImplementedError(f"{target_and_aux_calc} is not implemented")
