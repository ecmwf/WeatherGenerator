from typing import Any

from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase


class EMATeacher(TargetAndAuxModuleBase):
    def __init__(self, model, rng, ema_model, batch_size, **kwargs):
        # One of the issues is that the teacher model may have a different architecture
        # to the student, e.g. JEPA. So we need quite a flexible way to instantiate the
        # the teacher. Because of the device sharding etc that requires quite a bit of
        # massaging we assume that the teacher creates the EMA model correctly. However,
        # note that you cannot assume that model.state_dict equals ema_model.state_dict
        self.ema_model = ema_model
        self.batch_size = batch_size

        self.reset()

    def reset(self, batch_size=None):
        self.ema_model.reset()
        if batch_size is not None:
            self.batch_size = batch_size

    def update_state_pre_backward(self, istep, batch, model, **kwargs) -> None:
        return

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        self.ema_model.update(istep, self.batch_size)

    def compute(
        self, bidx, batch, model_params, model, forecast_offset, forecast_steps
    ) -> tuple[Any, Any]:
        return self.ema_model.forward_eval(
            model_params, batch, forecast_offset, forecast_steps
        ), None
