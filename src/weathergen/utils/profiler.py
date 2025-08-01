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