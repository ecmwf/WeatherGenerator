import dataclasses

import torch


@dataclasses.dataclass
class TargetAuxOutput:
    """
    A dataclass to encapsulate the TargetAndAuxCalculator output and give a clear API.
    """

    physical: dict[str, torch.Tensor]
    latent: dict[str, torch.Tensor]
    aux_outputs: dict[str, torch.Tensor]


class TargetAndAuxModuleBase:
    def __init__(self, cf, model, **kwargs):
        pass

    def reset(self):
        pass

    def update_state_pre_backward(self, istep, batch, model, **kwargs) -> None:
        pass

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        pass

    def compute(self, sample, *args, **kwargs) -> TargetAuxOutput:
        pass

    def to_device(self, device):
        pass


class PhysicalTargetAndAux(TargetAndAuxModuleBase):
    def __init__(self, cf, model, **kwargs):
        return

    def reset(self):
        return

    def update_state_pre_backward(self, istep, batch, model, **kwargs):
        return

    def update_state_post_opt_step(self, istep, batch, model, **kwargs):
        return

    def compute(self, sample, *args, **kwargs) -> TargetAuxOutput:
        return TargetAuxOutput(physical=sample.streams_data, latent=None, aux_outputs=None)

    def to_device(self, device):
        return
