# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os
import os
import socket
import time
import errno
import contextlib
import socket
import errno
import contextlib
import socket


import pynvml
import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.utils.data.distributed

from weathergen.train.utils import str_to_tensor, tensor_to_str

_logger = logging.getLogger(__name__)


class Trainer_Base:
    def __init__(self):
        pass

    ###########################################
    @staticmethod
    def init_torch(use_cuda=True, num_accs_per_task=1, multiprocessing_method="fork"):
        """
        Initialize torch, set device and multiprocessing method.

        NOTE: If using the Nvidia profiler, the multiprocessing method must be set to "spawn".
        The default for linux systems is "fork", which prevents traces from being generated with DDP.
        """
        torch.set_printoptions(linewidth=120)

        # This strategy is required by the nvidia profiles to properly trace events in worker processes.
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

    ###########################################
    @staticmethod
    def init_ddp(cf):
        rank = 0
        num_ranks = 1

        master_node = os.environ.get("MASTER_ADDR", "-1")
        if master_node == "-1":
            cf.with_ddp = False
            cf.rank = rank
            cf.num_ranks = num_ranks
            return

        local_rank = int(os.environ.get("SLURM_LOCALID"))
        ranks_per_node = int(os.environ.get("SLURM_TASKS_PER_NODE", "1")[0])
        rank = int(os.environ.get("SLURM_NODEID")) * ranks_per_node + local_rank
        num_ranks = int(os.environ.get("SLURM_NTASKS"))

        if rank == 0:
            # Check that port 1345 is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((master_node, 1345))
                except OSError as e:
                    if e.errno == errno.EADDRINUSE:
                        _logger.error(
                            f"Port 1345 is already in use on {master_node}. Please check your network configuration."
                        )
                        raise
                    else:
                        _logger.error(
                            f"Error while binding to port 1345 on {master_node}: {e}"
                        )
                        raise


        dist.init_process_group(
            backend="nccl",
            init_method="tcp://" + master_node + ":1345",
            timeout=datetime.timedelta(seconds=10 * 8192),
            world_size=num_ranks,
            rank=rank,
        )

        # communicate run id to all nodes
        run_id_int = torch.zeros(8, dtype=torch.int32).cuda()
        if rank == 0:
            run_id_int = str_to_tensor(cf.run_id).cuda()
        dist.all_reduce(run_id_int, op=torch.distributed.ReduceOp.SUM)
        cf.run_id = tensor_to_str(run_id_int)

        # communicate data_loader_rng_seed
        if hasattr(cf, "data_loader_rng_seed"):
            if cf.data_loader_rng_seed is not None:
                l_seed = torch.tensor(
                    [cf.data_loader_rng_seed if rank == 0 else 0], dtype=torch.int32
                ).cuda()
                dist.all_reduce(l_seed, op=torch.distributed.ReduceOp.SUM)
                cf.data_loader_rng_seed = l_seed.item()

        cf.rank = rank
        cf.num_ranks = num_ranks
        cf.with_ddp = True

        return

    def init_perf_monitoring(self):
        self.device_handles, self.device_names = [], []

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            self.device_names += [pynvml.nvmlDeviceGetName(handle)]
            self.device_handles += [handle]

    ###########################################
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

    ###########################################
    def ddp_average(self, val):
        if self.cf.with_ddp:
            dist.all_reduce(val.cuda(), op=torch.distributed.ReduceOp.AVG)
        return val.cpu()
