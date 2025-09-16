# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import errno
import logging
import os
import socket

import pynvml
import torch
import torch.distributed as dist
import torch.multiprocessing

from weathergen.common.config import Config
from weathergen.train.utils import str_to_tensor, tensor_to_str
from weathergen.utils.config import Config
from weathergen.utils.distributed import is_root, get_rank_from_env, get_size_from_env

_logger = logging.getLogger(__name__)


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

        #local_id_node = os.environ.get("SLURM_LOCALID", "-1")
        rank = get_rank_from_env(default="-1")
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        num_gpus = len(cuda_visible_devices.split(',')) if cuda_visible_devices else 0
        rank_local = rank % num_gpus

        if rank == "-1":
            devices = ["cuda"]
        else:
            devices = [
                f"cuda:{int(rank_local) * num_accs_per_task + i}"
                for i in range(num_accs_per_task)
            ]
        torch.cuda.set_device(int(rank_local) * num_accs_per_task)

        return devices

    @staticmethod
    def init_ddp(cf):
        rank = 0
        num_ranks = 1

        master_node = os.environ.get("MASTER_ADDR", "-1")
        if master_node == "-1":
            cf.with_ddp = False
            cf.rank = rank
            cf.num_ranks = num_ranks
            _logger.info(
                "DDP not initialized. MASTER_ADDR not set. Running in single process mode."
            )
            _logger.info(f"rank: {rank} has run_id: {cf.run_id}")
            return

        # local_rank = int(os.environ.get("SLURM_LOCALID"))
        # ranks_per_node = int(os.environ.get("SLURM_TASKS_PER_NODE", "1")[0])
        # rank = int(os.environ.get("SLURM_NODEID")) * ranks_per_node + local_rank
        # num_ranks = int(os.environ.get("SLURM_NTASKS"))
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        num_gpus             = len(cuda_visible_devices.split(',')) if cuda_visible_devices else 0
        
        rank                 = get_rank_from_env(default=0)
        num_ranks            = get_size_from_env(default=1)
        rank_local           = rank % num_gpus

        _logger.info(
            f"DDP initialization: local_rank={rank_local}, ranks_per_node={num_gpus}, "
            f"rank={rank}, num_ranks={num_ranks}"
        )

        master_port=os.getenv("MASTER_PORT")
        if rank == 0:
            # Check that port master_port is available, raise an error if not
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((master_node, master_port))
                except OSError as e:
                    if e.errno == errno.EADDRINUSE:
                        _logger.error(
                            (
                                f"Port {master_port} is already in use on {master_node}.",
                                " Please check your network configuration.",
                            )
                        )
                        raise
                    else:
                        _logger.error(f"Error while binding to port {master_port} on {master_node}: {e}")
                        raise

        _logger.info(
            f"Initializing DDP with rank {rank} out of {num_ranks} on master_node:{master_node}."
        )

        dist.init_process_group(
            backend="nccl",
            init_method="tcp://" + master_node + f":{master_port}",
            timeout=datetime.timedelta(seconds=240),
            world_size=num_ranks,
            rank=rank,
            #device_id=torch.device("cuda", rank_local),
        )
        if is_root():
            _logger.info("DDP initialized: root.")
        # Wait for all ranks to reach this point
        dist.barrier()

        _logger.info(
            f"TORCH DISTRIBUTED INFO: rank:{dist.get_rank()} world_size:{dist.get_world_size()}"
        )

        # communicate run id to all nodes
        len_run_id = len(cf.run_id)
        run_id_int = torch.zeros(len_run_id, dtype=torch.int32).cuda()
        if is_root():
            _logger.info(f"Communicating run_id to all nodes: {cf.run_id}")
            run_id_int = str_to_tensor(cf.run_id).cuda()
        dist.all_reduce(run_id_int, op=torch.distributed.ReduceOp.SUM)
        if not is_root():
            cf.run_id = tensor_to_str(run_id_int)
        _logger.info(f"rank: {rank} has run_id: {cf.run_id}")

        # communicate data_loader_rng_seed
        if hasattr(cf, "data_loader_rng_seed"):
            if cf.data_loader_rng_seed is not None:
                l_seed = torch.tensor(
                    [cf.data_loader_rng_seed if rank == 0 else 0], dtype=torch.int32
                ).cuda()
                dist.all_reduce(l_seed, op=torch.distributed.ReduceOp.SUM)
                cf.data_loader_rng_seed = l_seed.item()

        # TODO: move outside of the config
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
