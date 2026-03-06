from typing import Any

import torch

from weathergen.datasets.batch import ModelBatch
from weathergen.model.model import ModelParams
from weathergen.train.target_and_aux_module_base import (
    TargetAndAuxModuleBase,
    TargetAuxOutput,
)


class DiffusionLatentTargetEncoder(TargetAndAuxModuleBase):
    def __init__(self, encoder, is_model_sharded=True):
        # Todo: make sure this is a frozen clone or forward without gradients in compute()
        self.encoder = encoder

        self.is_model_sharded = is_model_sharded
        # Build a name → param map once
        self.src_params = dict(self.encoder.named_parameters())

        # self.reset()

    @torch.no_grad()
    def reset(self):
        """
        This function resets the EMAModel to be the same as the Model.

        It operates via the state_dict to be able to deal with sharded tensors in case
        FSDP2 is used.
        """
        breakpoint()
        self.encoder.to_empty(device="cuda")
        breakpoint()
        for p in self.encoder.parameters():
            p.requires_grad = False
        maybe_sharded_sd = self.encoder.state_dict()
        mkeys, ukeys = self.encoder.load_state_dict(maybe_sharded_sd, strict=False, assign=False)
        self.encoder.eval()

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        if self.is_model_sharded:
            self.encoder.reshard()

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

        # TODO: check if there are scenarios where the encoder needs to be set to eval
        with torch.no_grad():
            tokens, posteriors = self.encoder.encoder(model_params=model_params, batch=batch)
        # NOTE: must not set to train afterwards unless it was already in train

        output_idxs = batch.get_output_idxs()
        assert len(output_idxs) > 0

        target_aux_output = TargetAuxOutput(batch.get_output_len(), output_idxs)

        # TODO: currently hard-coding 0
        target_aux_output.add_latent_target(0, "diffusion_latent", tokens)

        # TODO: write function in TargetAuxOutput class
        target_aux_output.aux_outputs = {"noise_level_rn": noise_level_rn}

        return target_aux_output
