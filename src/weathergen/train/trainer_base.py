# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import errno
import logging
import os
import socket

import pynvml
import torch
import torch.distributed as dist
import torch.multiprocessing

from weathergen.utils.config import Config
from weathergen.utils.distributed import is_root

_logger = logging.getLogger(__name__)
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
        world_size = 1

        if not dist.is_available():
            _logger.info("Distributed training is not available.")
            return

        if not dist.is_initialized() and (cf.with_ddp or cf.with_fsdp):
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
            world_size = int(os.environ.get("WORLD_SIZE", "-1"))
            if world_size == -1:
                # Called using SLURM instead of torchrun
                world_size = int(os.environ.get("SLURM_NTASKS"))
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", f"{PORT}")  # Default port

            if torch.accelerator.is_available():
                device_type = torch.accelerator.current_accelerator()
                device = torch.device(f"{device_type}:{rank}")
                torch.accelerator.set_device_index(rank)
                _logger.info(f"DDP initialization: rank={rank}, world_size={world_size}")
            else:
                device = torch.device("cpu")
                _logger.info(f"Running on device {device}")

            backend = torch.distributed.get_default_backend_for_device(device)
            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                device_id=device,
                rank=rank,
                init_method=f"tcp://{master_addr}:{master_port}",
            )
            _logger.info(f"Process group initialized ({backend}).")

            if rank == 0:
                # Check that port 1345 is available, raise an error if not
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind((master_addr, PORT))
                    except OSError as e:
                        if e.errno == errno.EADDRINUSE:
                            _logger.error(
                                (
                                    f"Port 1345 is already in use on {master_addr}.",
                                    " Please check your network configuration.",
                                )
                            )
                            raise
                        else:
                            _logger.error(f"Error while binding to port 1345 on {master_addr}: {e}")
                            raise

            if is_root():
                _logger.info("DDP initialized: root.")
            # Wait for all ranks to reach this point
            dist.barrier()

            cf.rank = rank
            cf.world_size = world_size
            cf.with_ddp = True

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
