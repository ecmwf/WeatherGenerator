import time

class ExecutionTimer(object):
    def __init__(self, name: str, profile: bool = False):
        self._name = name
        self._profile = profile

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self._stop_time = time.time()

    def start(self):
        self._start_time = time.time()

    def time_elapsed(self) -> float:
        if not hasattr(self, "_stop_time"):
            return time.time() - self._start_time

        return self._stop_time - self._start_time