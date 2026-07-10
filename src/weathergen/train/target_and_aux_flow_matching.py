# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import torch

from weathergen.datasets.batch import ModelBatch
from weathergen.model.model import ModelParams
from weathergen.model.utils import apply_fct_to_blocks, freeze_weights, set_to_eval
from weathergen.train.target_and_aux_module_base import (
    TargetAndAuxModuleBase,
    TargetAuxOutput,
)


class FlowMatchingTargetEncoder(TargetAndAuxModuleBase):
    """Provides the clean latent target ``z`` for flow-matching / score-matching training.

    Structurally identical to
    :class:`weathergen.train.target_and_aux_diffusion.DiffusionLatentTargetEncoder`
    (a frozen, eval-mode encoder clone re-encodes the target sample to produce a stable,
    gradient-free ``z``), but stores it under ``"flow_target"``. The flow-specific
    regression target (velocity / noise / score) is assembled in
    :class:`weathergen.train.loss_modules.LossFlowMatching`, which reads the noise
    ``eps`` and time ``t`` the engine stashed into the (source) batch metadata. This
    keeps all path math behind the single ``GaussianPath`` used by the engine.
    """

    def __init__(self, encoder, is_model_sharded=True):
        self.encoder = encoder
        apply_fct_to_blocks(self.encoder, ".*", freeze_weights)
        apply_fct_to_blocks(self.encoder, ".*", set_to_eval)
        self.is_model_sharded = is_model_sharded
        # Set by the validation harness (mirrors the diffusion target encoder); the flow
        # engine draws eps and reads t from the batch, so this is only kept for parity.
        self._fixed_noise_level: float | None = None
        self.src_params = dict(self.encoder.named_parameters())

    @torch.no_grad()
    def reset(self):
        self.encoder.to_empty(device="cuda")
        for p in self.encoder.parameters():
            p.requires_grad = False
        maybe_sharded_sd = self.encoder.state_dict()
        self.encoder.load_state_dict(maybe_sharded_sd, strict=False, assign=False)
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
        with torch.no_grad():
            self.encoder.encoder.eval()
            tokens, _ = self.encoder.encoder(model_params=model_params, batch=batch)
            shape = (len(batch), batch.get_num_steps(), *tokens.shape[1:])
            tokens_multi = tokens.reshape(shape)

        output_idxs = batch.get_output_idxs()
        assert len(output_idxs) > 0

        # Single clean target latent (most recent step), replicated across forecast steps by
        # _expand_targets_to_match_preds in trainer.py.
        target_aux_output = TargetAuxOutput(1, [0])
        target_aux_output.add_latent_target(0, "flow_target", tokens_multi[:, -1])
        return target_aux_output
