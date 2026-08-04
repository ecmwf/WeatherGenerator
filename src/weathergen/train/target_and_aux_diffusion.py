from typing import Any

import torch

from weathergen.datasets.batch import ModelBatch, get_noise_level_rn
from weathergen.model.model import ModelParams
from weathergen.model.utils import apply_fct_to_blocks, freeze_weights, set_to_eval
from weathergen.train.target_and_aux_module_base import (
    TargetAndAuxModuleBase,
    TargetAuxOutput,
)


class DiffusionLatentTargetEncoder(TargetAndAuxModuleBase):
    def __init__(self, encoder, is_model_sharded=True):
        # Todo: make sure this is a frozen clone or forward without gradients in compute()
        self.encoder = encoder

        apply_fct_to_blocks(self.encoder, ".*", freeze_weights)
        apply_fct_to_blocks(self.encoder, ".*", set_to_eval)

        self.is_model_sharded = is_model_sharded
        self._fixed_noise_level: float | None = None
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
        # TODO: This needs fixing, might need to use apply_fct_to_blocks as in init()

        self.encoder.to_empty(device="cuda")
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
        # During validation (model in eval mode), use fixed noise level
        # so that sigma = exp(eta * p_std + p_mean) is deterministic
        if model.training:
            noise_level_rn = get_noise_level_rn(
                batch.samples[0].meta_info
            )  # TODO: adjust for multiple streams
        else:
            noise_level_rn = self._fixed_noise_level if self._fixed_noise_level is not None else 0.0

        # TODO: check if there are scenarios where the encoder needs to be set to eval
        with torch.no_grad():
            self.encoder.encoder.eval()  # NOTE: might be redundant
            tokens, posteriors, intermediates = self.encoder.encoder(
                model_params=model_params, batch=batch
            )
            shape = (len(batch), batch.get_num_steps(), *tokens.shape[1:])
            tokens_multi = tokens.reshape(shape)
        # NOTE: must not set to train afterwards unless it was already in train

        output_idxs = batch.get_output_idxs()
        assert len(output_idxs) > 0

        # The encoder produces a single target latent (tokens_multi[:, -1]) regardless of
        # how many forecast steps are requested.  Initialise with a single slot so that
        # _expand_targets_to_match_preds (in trainer.py) replicates the target across all
        # forecast steps automatically — both for T-step autoregressive rollouts and for the
        # single-step ODE-trajectory case.
        target_aux_output = TargetAuxOutput(1, [0])
        target_aux_output.add_latent_target(0, "diffusion_latent", tokens_multi[:, -1])

        # TODO: write function in TargetAuxOutput class
        target_aux_output.aux_outputs = {"noise_level_rn": noise_level_rn}

        return target_aux_output
