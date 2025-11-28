from typing import Any

import torch

from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase, TargetAuxOutput


class DiffusionLatentTargetEncoder(TargetAndAuxModuleBase):
    def __init__(self, model):
        # Todo: make sure this is a frozen clone or forward without gradients in compute()
        self.model = model

    def compute(
        self, bidx, batch, model_params, model, forecast_offset, forecast_steps
    ) -> tuple[Any, Any]:
        (_, _, _, metadata) = batch
        with torch.no_grad():
            tokens, posteriors = self.model(
                model_params=model_params,
                batch=batch,
                forecast_offset=None,
                forecast_steps=None,
                encode_only=True,
            )
        return TargetAuxOutput(
            physical=None, latent=[tokens], aux_outputs={"noise_level_rn": metadata.noise_level_rn}
        )
