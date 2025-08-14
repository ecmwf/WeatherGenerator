import torch

class ProfilerSection:
    """
    Context manager for profiling sections of PyTorch code using NVTX ranges.

    Attributes:
        name (str): The name of the section for profiling.
        profile (bool): Whether profiling is enabled.
    """

    def __init__(self, name: str, profile: bool = False):
        """
        Initialize a ProfilerSection.

        Args:
            name (str): Name of the section.
            profile (bool): Whether to enable profiling (default: False).
        """
        self.profile = profile
        self.name = name

    def __enter__(self):
        """
        Enter the context.

        If profiling is enabled, push a new NVTX range with the given name.
        NVTX ranges can be visualized in NVIDIA profiling tools to see timing.
        """
        if self.profile:
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context.

        If profiling is enabled, pop the NVTX range started in __enter__.
        """
        if self.profile:
            torch.cuda.nvtx.range_pop()
