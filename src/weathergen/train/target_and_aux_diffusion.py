from typing import Any

import torch

from weathergen.datasets.batch import Sample
from weathergen.model.model import ModelParams
from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase, TargetAuxOutput


class DiffusionLatentTargetEncoder(TargetAndAuxModuleBase):
    def __init__(self, model):
        # Todo: make sure this is a frozen clone or forward without gradients in compute()
        self.encoder = model.encoder

    def compute(self, sample: Sample, model_params: ModelParams, **kwargs) -> tuple[Any, Any]:
        noise_level_rn = sample.meta_info["ERA5"].params[
            "noise_level_rn"
        ]  # TODO: adjust for multiple streams
        with torch.no_grad():
            tokens, posteriors = self.encoder(model_params=model_params, sample=sample)
        return TargetAuxOutput(
            physical=None, latent=tokens, aux_outputs={"noise_level_rn": noise_level_rn}
        )
