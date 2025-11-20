from typing import Any


class TargetAndAuxModuleBase:
    def __init__(self, model, rng, **kwargs):
        pass

    def reset(self):
        pass

    def update_state_pre_backward(self, istep, batch, model, **kwargs) -> None:
        pass

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        pass

    def compute(self, *args, **kwargs) -> tuple[Any, Any]:
        pass

    def to_device(self, device):
        pass


class IdentityTargetAndAux(TargetAndAuxModuleBase):
    def __init__(self, model, rng, **kwargs):
        return

    def reset(self):
        return

    def update_state_pre_backward(self, istep, batch, model, **kwargs):
        return

    def update_state_post_opt_step(self, istep, batch, model, **kwargs):
        return

    def compute(self, istep, batch, *args, **kwargs):
        return {"physical": batch[0]}, None

    def to_device(self, device):
        return
