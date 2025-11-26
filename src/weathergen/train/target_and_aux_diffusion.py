from typing import Any

import torch

from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase


class DiffusionLatentTargetEncoder(TargetAndAuxModuleBase):
    def __init__(self, model):
        # Todo: make sure this is a frozen clone or forward without gradients in compute()
        self.model = model

    def compute(
        self, bidx, batch, model_params, model, forecast_offset, forecast_steps
    ) -> tuple[Any, Any]:
        with torch.no_grad():
            tokens, posteriors = self.model.encode(model_params=model_params, batch=batch)
        return {"latent": [tokens]}, posteriors
