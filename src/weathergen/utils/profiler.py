from typing import Optional, Any
import time
import torch



class ProfilerSection(object):
    def __init__(self, name: str, profile: bool = False):
        self.profile = profile
        self.name = name

    def __enter__(self):
        if self.profile:
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, *args, **kwargs):
        if self.profile:
            # torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()


class CudaExecutionTimer(object):
    def __init__(self, stream: Optional[torch.cuda.Stream] = None):
        self._stream = stream
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self._start_event.record(stream=self._stream)
        return self

    def __exit__(self, *args, **kwargs):
        self._end_event.record(stream=self._stream)

    def time_elapsed(self) -> float:
        self._end_event.synchronize()
        return self._start_event.elapsed_time(self._end_event)


class ExecutionTimer(object):
    def __init__(self, name: str, profile: bool = False):
        self._name = name
        self._profile = profile

    def __enter__(self):
        torch.cuda.nvtx.range_push(self._name)
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        torch.cuda.nvtx.range_pop()
        self._stop_time = time.time()

    def start(self):
        self._start_time = time.time()

    def time_elapsed(self) -> float:
        if not hasattr(self, "_stop_time"):
            return time.time() - self._start_time

        return self._stop_time - self._start_time


