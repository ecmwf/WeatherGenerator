import logging
import platform
from datetime import datetime

import torch
from omegaconf import OmegaConf
from torch.profiler import record_function

import weathergen.common.config as config
from weathergen.utils.distributed import get_rank

logger: logging.Logger = logging.getLogger(__name__)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)


def stop_record_memory_history() -> None:
    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot(cfg: dict | OmegaConf) -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    base_path = config.get_path_profiler(cfg)

    timestamp = datetime.now().strftime(TIME_FORMAT_STR)

    file_prefix = base_path / f"{timestamp}_rank_{get_rank()}"

    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return


def trace_handler(cfg: dict | OmegaConf, prof: torch.profiler.profile) -> None:
    # Prefix for file names.
    base_path = config.get_path_profiler(cfg)

    timestamp = datetime.now().strftime(TIME_FORMAT_STR)

    file_prefix = base_path / f"{timestamp}_rank_{get_rank()}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    on_aarch64 = platform.machine() == "aarch64"
    if not on_aarch64:
        prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
    else:
        logger.info("[profiler] Memory distribution timeline skipped on aarch64")


def wrap_module_forward_with_profiling(model, prefix=""):
    """
    Recursively wrap all nn.Module forward methods with profiling context
    """
    for name, module in model.named_children():
        module_name = f"{prefix}.{name}" if prefix else name

        # Skip standard PyTorch modules (they're already traced)
        if type(module).__module__.startswith("torch.nn.modules"):
            # Still recurse into children
            wrap_module_forward_with_profiling(module, module_name)
            continue

        # Wrap custom modules
        original_forward = module.forward

        def make_profiled_forward(mod_name, orig_forward):
            def profiled_forward(*args, **kwargs):
                with record_function(f"nn.Module: {mod_name}"):
                    return orig_forward(*args, **kwargs)

            return profiled_forward

        module.forward = make_profiled_forward(module_name, original_forward)

        # Recurse into children
        wrap_module_forward_with_profiling(module, module_name)
