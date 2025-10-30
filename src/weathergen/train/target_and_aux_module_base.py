from typing import Any

class TargetAndAuxModuleBase:
    def __init__(self, model, rng, **kwargs):
        pass

    def update_state_pre_backward(self, istep, batch, model, **kwargs) -> None:
        pass

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        pass

    def compute(self, *args, **kwargs) -> tuple[Any, Any]:
        pass


class IdentityTargetAndAux(TargetAndAuxModuleBase):
    def __init__(self, model, rng, config):
        return

    def update_state_pre_backward(self, istep, batch, model, **kwargs):
        return

    def update_state_post_opt_step(self, istep, batch, model, **kwargs):
        return

    def compute(self, istep, batch, model):
        return batch[0], None


