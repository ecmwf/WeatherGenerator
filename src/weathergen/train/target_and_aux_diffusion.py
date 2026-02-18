from typing import Any

import torch

from weathergen.datasets.batch import ModelBatch
from weathergen.model.model import ModelParams
from weathergen.train.target_and_aux_module_base import (
    PhysicalTargetAndAux,
)


class DiffusionLatentTargetEncoder(PhysicalTargetAndAux):
    def __init__(self, model):
        # Todo: make sure this is a frozen clone or forward without gradients in compute()
        self.encoder = model.encoder

    def compute(
        self,
        istep: int,
        batch: ModelBatch,
        model_params: ModelParams,
        model: torch.nn.Module,
        *args,
        **kwargs,
    ) -> tuple[Any, Any]:
        noise_level_rn = (
            batch.samples[0].meta_info["ERA5"].params["noise_level_rn"]
        )  # TODO: adjust for multiple streams

        with torch.no_grad():
            tokens, posteriors = self.encoder(model_params=model_params, batch=batch)

        target_aux_output = super().compute(istep, batch, model_params, model)

        # TODO: currently hard-coding 0
        target_aux_output.add_latent_target(0, "diffusion_latent", tokens)

        # TODO: write function in TargetAuxOutput class
        target_aux_output.aux_outputs = {"noise_level_rn": noise_level_rn}

        return target_aux_output
