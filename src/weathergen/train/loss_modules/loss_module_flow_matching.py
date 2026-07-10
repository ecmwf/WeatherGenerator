# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import torch
from omegaconf import DictConfig
from torch import Tensor

import weathergen.train.loss_modules.loss_functions as loss_fns
from weathergen.model.flow_matching import GaussianPath
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

_logger = logging.getLogger(__name__)


class LossFlowMatching(LossModuleBase):
    """Conditional flow-matching / denoising-score-matching loss in latent space.

    Regresses the engine's *raw* prediction (velocity / noise / score, per
    ``fe_flow_prediction_type``) against the analytic conditional target
    (course Eq. 37 / Alg. 4 / Eq. 40), assembled here from:
      - ``z``  : the clean latent from :class:`FlowMatchingTargetEncoder`
                 (``targets.latent[...]["flow_target"]``),
      - ``eps``, ``t`` : the noise and time the engine drew, stashed into the (source)
                 batch metadata during the forward pass and read here.

    Using the same :class:`GaussianPath` as the engine guarantees the target matches the
    exact ``x_t`` the network saw.
    """

    def __init__(
        self,
        cf: DictConfig,
        mode_cfg: DictConfig,
        stage: Stage,
        device: str,
        **loss_fcts: dict,
    ):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.name = "LossFlowMatching"

        self.path = GaussianPath(cf.get("fe_flow_path", "condot"))
        self.prediction_type = cf.get("fe_flow_prediction_type", "velocity")

        self.loss_fcts = [
            [getattr(loss_fns, name), params.get("weight", 1.0), name]
            for name, params in loss_fcts.items()
        ]

    def _get_fstep_weights(self, forecast_steps):
        timestep_weight_config = self.cf.get("timestep_weight")
        if timestep_weight_config is None:
            return [1.0 for _ in range(forecast_steps)]
        weights_timestep_fct = getattr(loss_fns, timestep_weight_config[0])
        return weights_timestep_fct(forecast_steps, timestep_weight_config[1])

    @staticmethod
    def _read_eps_t(metadata, device) -> tuple[Tensor, Tensor]:
        """Read the engine-stashed noise ``eps`` and time ``t`` from the source metadata.

        ``metadata`` is the tuple from ``extract_batch_metadata``: element [1] is the list
        of source-sample metadata; the flow engine stashed ``fm_eps``/``fm_t`` into
        sample 0 (single-sample forecast setup).
        """
        source_meta = metadata[1][0]
        params = source_meta.params
        assert "fm_eps" in params and "fm_t" in params, (
            "fm_eps/fm_t not found in batch metadata — FlowMatchingForecastEngine."
            "training_forward must run before the loss (it stashes them)."
        )
        eps = params["fm_eps"].to(device)
        t = torch.as_tensor(params["fm_t"], device=device, dtype=torch.float32)
        return eps, t

    def compute_loss(self, preds: dict, targets: dict, **kwargs) -> LossValues:
        losses_all: dict[str, Tensor] = {
            f"{self.name}.{name}": torch.zeros(1, device=self.device)
            for _, _, name in self.loss_fcts
        }

        pred_tokens_all = [
            pl["latent_state"].z_pre_norm
            for pl in preds.latent
            if pl and "latent_state" in pl
        ]
        target_tokens_all = [latent["flow_target"] for latent in targets.latent if latent]

        # Ensemble mode does not call predict_latent → no latent predictions.
        if not pred_tokens_all:
            nan = torch.tensor(torch.nan).to(self.device)
            return LossValues(
                loss=torch.zeros(1, device=self.device),
                losses_all={f"{self.name}.{n}": nan for _, _, n in self.loss_fcts},
                stddev_all={"latent": nan},
            )

        eps, t = self._read_eps_t(kwargs["metadata"], self.device)
        fstep_loss_weights = self._get_fstep_weights(len(target_tokens_all))

        loss_fsteps = torch.tensor(0.0, device=self.device, requires_grad=True)
        ctr_fsteps = 0
        for z_target, pred_tokens, fstep_weight in zip(
            target_tokens_all, pred_tokens_all, fstep_loss_weights, strict=True
        ):
            # Analytic conditional target for the configured parameterization.
            target = self.path.conditional_target(z_target, eps, t, self.prediction_type)

            loss_fstep = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_loss_fcts = 0
            for loss_fct, loss_fct_weight, loss_fct_name in self.loss_fcts:
                loss_lfct, _ = loss_fct(target=target, pred=pred_tokens)
                losses_all[f"{self.name}.{loss_fct_name}"] += loss_lfct
                loss_fstep = loss_fstep + loss_fct_weight * loss_lfct
                ctr_loss_fcts += 1 if loss_lfct > 0.0 else 0

            loss_fsteps = loss_fsteps + (
                loss_fstep * fstep_weight / ctr_loss_fcts if ctr_loss_fcts > 0 else 0
            )
            ctr_fsteps += 1 if ctr_loss_fcts > 0 else 0

        loss = loss_fsteps / (ctr_fsteps if ctr_fsteps > 0 else 1.0)
        for _, loss_values in losses_all.items():
            loss_values /= ctr_fsteps if ctr_fsteps > 0 else 1.0
            loss_values[loss_values == 0.0] = torch.nan

        return LossValues(
            loss=loss,
            losses_all=losses_all,
            stddev_all={"latent": torch.tensor(torch.nan).to(self.device)},
        )
