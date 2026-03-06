from typing import Any

import torch

from weathergen.datasets.batch import ModelBatch
from weathergen.model.model import ModelParams
from weathergen.train.target_and_aux_module_base import (
    TargetAndAuxModuleBase,
    TargetAuxOutput,
)


class DiffusionLatentTargetEncoder(TargetAndAuxModuleBase):
    """Computes latent diffusion targets by running the training model's frozen encoder.

    No separate encoder copy is maintained. The training model's encoder is assumed to be
    frozen (via freeze_modules config) and is used directly in compute(). This guarantees
    that the encoder producing targets is identical to the one in the forward pass and
    avoids illegal memory access errors from running two independent flash-attention models.
    """

    def __init__(self):
        pass

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

        # The encoder is frozen (requires_grad=False via freeze_modules), so no_grad is
        # used only to avoid unnecessary autograd bookkeeping and reduce memory usage.
        # Unwrap DDP wrapper if present to access the encoder directly.
        unwrapped = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            tokens, posteriors = unwrapped.encoder(model_params=model_params, batch=batch)

        output_idxs = batch.get_output_idxs()
        assert len(output_idxs) > 0

        target_aux_output = TargetAuxOutput(batch.get_output_len(), output_idxs)

        # TODO: currently hard-coding 0
        target_aux_output.add_latent_target(0, "diffusion_latent", tokens.detach().clone())

        # TODO: write function in TargetAuxOutput class
        target_aux_output.aux_outputs = {"noise_level_rn": noise_level_rn}

        return target_aux_output
