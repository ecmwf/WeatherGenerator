# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ----------------------------------------------------------------------------
# Third-Party Attribution: NVLABS/EDM (Elucidating the Design of Diffusion Models)
# This file incorporates code originally from the 'NVlabs/edm' repository.
#
# Original Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Third-Party Attribution: facebookresearch/DiT (Scalable Diffusion Models with Transformers (DiT))
# This file incorporates code originally from the 'facebookresearch/DiT' repository,
# with adaptations.
#
# The original code is licensed under CC-BY-NC.
# ----------------------------------------------------------------------------

"""
Latent diffusion forecast engines: the shared base DiffusionForecastEngineBase and the
components common to all diffusion-type formulations (conditioning, registers, noise embedding,
inference sigma schedule). The formulations are sibling subclasses, selected by the exclusive
``diffusion.type`` switch (see build_diffusion_forecast_engine):

    diffusion:
      enabled: True
      type: edm | flow_matching
      # shared: conditioning, conditioning_type, conditioning_embed_dim, latent_dim,
      #   num_register_tokens, frequency_embedding_dim, embedding_dim, noise_embedding, rollout,
      #   predict_residual, impute_latent_noise_std, classifier_free_guidance_prob
      sampler: {num_steps, num_ensemble_members}
      edm: {...}            # only for type: edm, see edm.py
      flow_matching: {...}  # only for type: flow_matching, see flow_matching.py

The losses mirror this structure (LossLatentDiffusionBase with LossLatentDiffusion for edm and
LossFlowMatching for flow_matching); both use DiffusionLatentTargetEncoder for the target.

Possible future direction (not implemented): EDM is exactly flow matching on the
variance-exploding Gaussian path with a denoiser parameterization (flow_matching.path: ve,
prediction_type: denoiser reproduces an EDM training step bit-for-bit). The two subclasses could
therefore be collapsed into a single engine and a single loss that delegate to a pluggable
"formulation" object (path coefficients, input/output preconditioning, noise embedding input,
training target and loss weight, sampler), built once from the config and shared by engine and
loss, with EDM being one configuration of it. This requires moving the EDM-only features
(stochastic churn, the loss_weighting variants, the dit_legacy noise embedding) into that
formulation and should only be done once flow matching has been validated at scale.
"""

import abc
import dataclasses
import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

from weathergen.common.config import Config
from weathergen.datasets.batch import BatchSamples
from weathergen.datasets.masking import get_noise_level_cfg
from weathergen.model.engines import DecodeSpec, ForecastEngineBase, ForecastingEngine
from weathergen.utils.inference_logger import ScalarSeriesJSONLWriter

if TYPE_CHECKING:
    from weathergen.model.model import Model, ModelOutput, ModelParams


logger = logging.getLogger(__name__)


class DiffusionRegisters(torch.nn.Module):
    """Learnable register tokens of the denoiser, shape (1, R, D).

    Kept in their own module (rather than as a bare Parameter on DiffusionForecastEngineBase) so
    that checkpoint loading can (re)initialize them via reset_parameters() without treating the
    whole diffusion engine as a new module. Initialized like the encoder's register tokens
    (EncoderModule.q_aux): uniform in [0, 1/D).
    """

    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.dim = dim
        self.tokens = torch.nn.Parameter(torch.empty(1, num_registers, dim))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.uniform_(self.tokens, 0.0, 1.0 / self.dim)


class SpatialAdaLN(torch.nn.Module):
    """Per-token (spatial) AdaLN-Zero for forecast conditioning.

    Given conditioning tokens c (B, H, D) and noised tokens x (B, H, D):
      scale, shift, gate = MLP(c)           # (B, H, D) each
      x_mod = LayerNorm(x) * (1+scale) + shift

    Returns (x_mod, gate). The caller should apply  raw_out = raw_out * gate
    before the preconditioner so that each HEALPix cell can independently
    up/down-weight the denoiser's output based on the quality or content of
    the conditioning token at that cell.

    Zero-initialised: at the start of training scale=shift=gate=0, so the
    modulation is a no-op and the model degrades gracefully to the unguided case.
    """

    def __init__(self, dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
        # SiLU activation followed by a linear projection; zero-init ensures
        # scale=shift=gate=0 at initialisation (identity / no-op).
        self.proj = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(dim, 3 * dim, bias=True),
        )
        torch.nn.init.zeros_(self.proj[-1].weight)
        torch.nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """x, c: (B, H, D).  Returns (x_modulated, gate) both (B, H, D)."""
        scale, shift, gate = self.proj(c).chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate


class DiffusionForecastConditioning:
    """Diffusion conditioning type for ForecastingEngine's blocks.

    Wraps the raw ``diffusion.conditioning_type`` config value so call sites compare
    against ``conditioning_type`` directly instead of re-reading the config string everywhere.
    """

    def __init__(self, conditioning_type: str | None):
        self.conditioning_type = conditioning_type

    def __eq__(self, other: object) -> bool:
        """Value equality, so, e.g., ``self.conditioning == "ada_ln"`` works directly instead
        of having to compare against ``self.conditioning.conditioning_type``.
        """
        if isinstance(other, DiffusionForecastConditioning):
            other = other.conditioning_type
        return self.conditioning_type == other

    def __hash__(self) -> int:
        return hash(self.conditioning_type)

    def __repr__(self) -> str:
        return f"DiffusionForecastConditioning({self.conditioning_type!r})"

    def validate(self, conditioning: torch.Tensor | None) -> None:
        """Check conditioning is present/absent exactly when this mode expects it.

        concatenate/concatenate_hiddendim bake conditioning into tokens before calling the
        engine (see DiffusionForecastEngineBase._net_forward() below), so they expect
        conditioning=None when reaching ForecastingEngine.forward().
        """
        if self.conditioning_type in {"ada_ln", "cross_attn", "cross_attn_rev", "additive"}:
            assert conditioning is not None, (
                f"conditioning must be provided for '{self.conditioning_type}' "
                "diffusion model conditioning"
            )
        elif self.conditioning_type is None:
            assert conditioning is None, (
                "conditioning should not be provided when diffusion model conditioning is disabled"
            )


DIFFUSION_TYPES = ("edm", "flow_matching")
# keys of diffusion.sampler shared by all diffusion types; the type-specific sampler options
# are in diffusion.<type>.sampler
_SHARED_SAMPLER_KEYS = ("num_steps", "num_ensemble_members")


def validate_diffusion_config(diff_cf: Config) -> str:
    """Check that the diffusion config selects exactly one diffusion type and only contains the
    section of that type. Returns the diffusion type."""
    diffusion_type = diff_cf.get("type", None)
    assert diffusion_type in DIFFUSION_TYPES, (
        f"diffusion.type must be one of {DIFFUSION_TYPES}, got {diffusion_type!r}"
    )
    inactive = [t for t in DIFFUSION_TYPES if t != diffusion_type and t in diff_cf]
    assert not inactive, (
        f"diffusion.type is {diffusion_type!r}, remove the section(s) diffusion.{inactive} of "
        "the inactive diffusion type(s)"
    )
    misplaced = [k for k in diff_cf.get("sampler", {}) if k not in _SHARED_SAMPLER_KEYS]
    assert not misplaced, (
        f"diffusion.sampler only holds the shared options {_SHARED_SAMPLER_KEYS}; move "
        f"{misplaced} to diffusion.{diffusion_type}.sampler"
    )
    return diffusion_type


def build_diffusion_forecast_engine(
    cf: Config, num_healpix_cells: int, forecast_engine: ForecastingEngine
) -> "DiffusionForecastEngineBase":
    """Wrap forecast_engine into the diffusion forecast engine selected by diffusion.type."""
    diffusion_type = validate_diffusion_config(cf.diffusion)
    # local imports: the engines subclass DiffusionForecastEngineBase defined in this module
    if diffusion_type == "flow_matching":
        from weathergen.model.flow_matching import FlowMatchingForecastEngine

        return FlowMatchingForecastEngine(cf, num_healpix_cells, forecast_engine)
    from weathergen.model.edm import EDMForecastEngine

    return EDMForecastEngine(cf, num_healpix_cells, forecast_engine)


@dataclasses.dataclass
class InferenceSigmaRange:
    """Noise level range [sigma_min, sigma_max] of an inference schedule, and the training noise
    level distribution it was derived from (None if not derived from it)."""

    sigma_min: float
    sigma_max: float
    train_distribution: str | None = None
    train_sigma_min: float | None = None
    train_sigma_max: float | None = None


def get_inference_sigma_range(
    cf: Config, sampler_cf: Config, sigma_data: float, cfg_name: str
) -> InferenceSigmaRange:
    """
    Get the sigma range [sigma_min, sigma_max] of the inference schedule.

    Values not set in sampler_cf (the config section cfg_name) are derived from the *training*
    noise level distribution (masking_strategy_config.noise_level of the target strategies in
    training_config): the network only learns to denoise reliably within the sigma range
    seen during training. Using the wrong distribution here truncates the schedule and
    leaves the sample under-denoised (e.g. applying the log-normal p_mean/p_std formula to a
    model trained with log_uniform noise stops the ODE far above sigma_min).
      - sigma_max: upper bound of the training distribution. Beyond this the denoiser is in
        untrained territory and poisons the trajectory.
      - sigma_min: quantile ``sigma_min_quantile`` of the training distribution, floored by
        sigma_data * 0.01 for numerical stability (avoids dividing by near-zero sigma in the
        ODE drift).
    """
    sigma_min, sigma_max = sampler_cf.get("sigma_min"), sampler_cf.get("sigma_max")
    noise_level_cfg = get_noise_level_cfg(cf.get("training_config", {}))
    r = InferenceSigmaRange(sigma_min, sigma_max)
    if noise_level_cfg is None:
        assert sigma_min is not None and sigma_max is not None, (
            f"{cfg_name}.sigma_min and sigma_max have to be set if no target strategy "
            "of training_config defines masking_strategy_config.noise_level"
        )
    else:
        q = sampler_cf.get("sigma_min_quantile", 0.05)
        r.train_distribution = noise_level_cfg.get("distribution", "log_normal")
        if r.train_distribution == "log_uniform":
            # log(sigma) ~ Uniform[log(sigma_min), log(sigma_max)]; quantiles are linear
            # in log-space, so sigma at quantile q is exp(log_min + q * (log_max - log_min)).
            log_min = math.log(noise_level_cfg["sigma_min"])
            log_max = math.log(noise_level_cfg["sigma_max"])
            r.train_sigma_min = math.exp(log_min)
            r.train_sigma_max = math.exp(log_max)
            sigma_min_from_dist = math.exp(log_min + q * (log_max - log_min))
        elif r.train_distribution == "log_normal":
            # log(sigma) ~ N(p_mean, p_std). Range is taken as ~99.7% interval
            # (p_mean -/+ 3 p_std); sigma at quantile q is exp(p_mean + Phi^-1(q) * p_std),
            # with Phi^-1 approximated by standard z-scores (default q=0.05).
            p_mean, p_std = noise_level_cfg["p_mean"], noise_level_cfg["p_std"]
            r.train_sigma_min = math.exp(p_mean - 3.0 * p_std)
            r.train_sigma_max = math.exp(p_mean + 3.0 * p_std)
            _z_scores = {0.01: -2.326, 0.025: -1.960, 0.05: -1.645, 0.10: -1.282}
            _z = _z_scores.get(q, -1.645)
            sigma_min_from_dist = math.exp(p_mean + _z * p_std)
        else:
            raise ValueError(f"Unsupported noise_level distribution: {r.train_distribution}")

        if sigma_max is None:
            r.sigma_max = r.train_sigma_max
        if sigma_min is None:
            r.sigma_min = max(sigma_min_from_dist, sigma_data * 0.01)
    r.sigma_min = float(r.sigma_min)
    r.sigma_max = float(r.sigma_max)

    assert 0.0 < r.sigma_min < r.sigma_max, (
        f"{cfg_name} requires 0 < sigma_min < sigma_max "
        f"(got sigma_min={r.sigma_min}, sigma_max={r.sigma_max})"
    )
    if noise_level_cfg is not None and (
        r.sigma_min < r.train_sigma_min or r.sigma_max > r.train_sigma_max
    ):
        logger.warning(
            f"Inference sigma range [{r.sigma_min}, {r.sigma_max}] exceeds the "
            f"training noise range [{r.train_sigma_min:.4f}, {r.train_sigma_max:.4f}] "
            f"({r.train_distribution})."
        )
    return r


def rho_sigma_steps(
    sigma_min: float, sigma_max: float, rho: float, num_steps: int, device
) -> torch.Tensor:
    """The ``num_steps + 1`` rho-spaced noise levels (EDM Eq. 5, float64) of the sampler,
    descending from sigma_max to sigma_min, with a terminal sigma_N = 0."""
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0


class DiffusionForecastEngineBase(ForecastEngineBase, abc.ABC):
    """
    Base of the latent diffusion forecast engines (EDMForecastEngine, FlowMatchingForecastEngine).

    Owns everything that does not depend on the diffusion formulation: the conditioning
    (date/time embedding, forecast conditioning, latent projections, spatial AdaLN, diffusion
    registers) around the wrapped ForecastingEngine (_net_forward), the noise embedder, the
    split of the input window (prepare_fe_input), the routing between training and inference
    (forward), trajectory/ensemble/rollout inference and its decoding, and the sampling
    diagnostics writer. Subclasses implement training_forward and _run_sampler.
    """

    # Fallback for diffusion.noise_embedding.type
    default_noise_embedding_type = "log_fourier"

    def __init__(self, cf: Config, num_healpix_cells: int, forecast_engine: ForecastingEngine):
        super().__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells
        self.net = forecast_engine
        self.diff_cf = self.cf.diffusion
        self.frequency_embedding_dim = self.diff_cf.frequency_embedding_dim
        self.embedding_dim = self.diff_cf.embedding_dim
        noise_emb_cfg = self.diff_cf.get("noise_embedding", None) or {}
        self.noise_embedder = NoiseEmbedder(
            embedding_dim=self.embedding_dim,
            frequency_embedding_dim=self.frequency_embedding_dim,
            embedding_type=noise_emb_cfg.get("type", self.default_noise_embedding_type),
            f_min=noise_emb_cfg.get("f_min", 0.5),
            f_max=noise_emb_cfg.get("f_max", 2 * math.pi * 4),
            include_raw=noise_emb_cfg.get("include_raw", True),
        )
        self.conditioning = self.diff_cf.get("conditioning", None)
        self.conditioning_type = self.diff_cf.get("conditioning_type", None)

        _date_time_modes = {"date_time", "date", "time"}
        assert self.conditioning not in _date_time_modes or self.conditioning_type == "ada_ln", (
            f"diffusion.conditioning_type must be 'ada_ln' when "
            f"diffusion.conditioning is '{self.conditioning}' "
            f"(got '{self.conditioning_type}')"
        )
        _ada_ln = self.conditioning_type == "ada_ln"
        assert self.diff_cf.get("conditioning_embed_dim", None) is not None or not _ada_ln, (
            "diffusion.conditioning_embed_dim must be set when "
            "diffusion.conditioning_type is 'ada_ln'"
        )
        _offset = self.cf.get("training_config", {}).get("forecast", {}).get("offset", 0)
        assert self.conditioning not in _date_time_modes or _offset == 0, (
            f"forecast.offset must be 0 when diffusion.conditioning is "
            f"'{self.conditioning}' (got offset={_offset})"
        )
        _input_num_steps = (
            self.cf.get("training_config", {})
            .get("model_input", {})
            .get("forecasting", {})
            .get("num_steps_input", 0)
        )
        # assert self.conditioning != "forecast" or _input_num_steps == 2, (
        #     f"forecast.input_num_steps must be 2 when diffusion.conditioning is "
        #     f"'{self.conditioning}' (got input_num_steps={_input_num_steps})"
        # )
        assert self.conditioning not in ["date_time", "date", "time"] or _input_num_steps == 1, (
            f"forecast.input_num_steps must be 1 when diffusion.conditioning is "
            f"'{self.conditioning}' (got input_num_steps={_input_num_steps})"
        )
        assert self.conditioning != "forecast" or self.conditioning_type in {
            "cross_attn",
            "additive",
            "cross_attn_rev",
            "concatenate",
            "concatenate_hiddendim",
            "concatenate_hdMLP",
            "spatial_ada_ln",
        }, (
            "diffusion.conditioning_type must be 'cross_attn', "
            "'additive', 'cross_attn_rev', 'concatenate', 'concatenate_hiddendim', ",
            f"'concatenate_hdMLP', or 'spatial_ada_ln' when "
            f"diffusion.conditioning is 'forecast' "
            f"(got '{self.conditioning_type}')",
        )

        if self.conditioning and (self.conditioning in ["date_time", "date", "time"]):
            self.datetime_embedder = DateTimeEncoder(self.conditioning)

        # Optional MLP projections for an expanded diffusion latent space:
        # projects encoder tokens (ae_global_dim_embed -> diffusion.latent_dim) before denoising
        # and back (diffusion.latent_dim -> ae_global_dim_embed) after. The conditioning tokens
        # get their own up-projection so the denoiser can tell them apart from the noisy tokens.
        # When diffusion.latent_dim == ae_global_dim_embed (default), these are None (no-op).
        _enc_dim = self.cf.ae_global_dim_embed
        _lat_dim = self.diff_cf.get("latent_dim") or _enc_dim
        if _lat_dim != _enc_dim:
            self.latent_proj_up = torch.nn.Linear(_enc_dim, _lat_dim, bias=False)
            self.latent_proj_up_cond = torch.nn.Linear(_enc_dim, _lat_dim, bias=False)
            self.latent_proj_down = torch.nn.Linear(_lat_dim, _enc_dim, bias=False)
        else:
            self.latent_proj_up = None
            self.latent_proj_up_cond = None
            self.latent_proj_down = None

        # Spatial AdaLN: per-cell modulation using the conditioning token at each HEALPix cell.
        # Only instantiated for the 'spatial_ada_ln' conditioning type.
        if self.conditioning_type == "spatial_ada_ln":
            self.spatial_ada_ln = SpatialAdaLN(dim=_lat_dim, norm_eps=self.cf.get("norm_eps", 1e-4))
        else:
            self.spatial_ada_ln = None

        if self.conditioning_type == "concatenate_hdMLP":
            self.concat_hd_proj = torch.nn.Linear(2 * _lat_dim, _lat_dim, bias=False)

        # Inference sampler options shared by all diffusion types
        sampler_cf = self.diff_cf.get("sampler", {})
        self.num_inference_steps = sampler_cf.get("num_steps", None)
        self.num_ensemble_members = sampler_cf.get("num_ensemble_members", 1)
        self.rollout = self.diff_cf.get("rollout", False)

        # Encoder register/class tokens are ignored by the diffusion engine (not diffused).
        self.num_aux_tokens = self.cf.num_register_tokens + self.cf.num_class_tokens
        # Learnable registers of the denoiser, independent of the encoder's registers.
        self.num_diffusion_registers = self.diff_cf.get("num_register_tokens", 0) or 0
        self.diffusion_registers = (
            DiffusionRegisters(self.num_diffusion_registers, _lat_dim)
            if self.num_diffusion_registers > 0
            else None
        )

        self.log_analytics = cf.get("inference_logging", {}).get(
            "fe_diffusion_log_analytics", False
        )
        # Target tokens for sampling diagnostics; only kept when analytics logging is enabled.
        self.cur_token: torch.Tensor | None = None

        # Inference-mode only: persists per-ODE-step sampling diagnostics to a JSONL file instead
        # of plotting them inline.
        # See scripts/analytics/diffusion/plot_diffusion_sampling_diagnostics.py.
        self._diagnostics_writer = None
        if cf.stage == "inference":
            self._diagnostics_writer = ScalarSeriesJSONLWriter(
                cf,
                name="diffusion_sampling_diagnostics",
                enabled=self.log_analytics,
            )

    def prepare_fe_input(
        self, tokens: torch.Tensor, meta_info: dict, forecast_offset: int
    ) -> tuple[torch.Tensor, dict]:
        """Split the loaded input-step window into conditioning and the rollout starting state.

        tokens[:,0] = t (most recent), tokens[:,1] = t-1, ..., tokens[:,-1] = t-(T-1) (oldest).
        For diffusion.conditioning != "forecast" this is the same plain sum used by non-diffusion
        engines. For "forecast" conditioning, the older steps become the conditioning (returned
        via meta_info for training_forward/inference_forward to pick up) and the newest step (t)
        is the denoising target during training.
        """
        if self.conditioning != "forecast":
            return tokens.sum(axis=1), meta_info

        if self.cf.stage == "inference" and tokens.shape[1] == 1 and forecast_offset == 1:
            # Single-input-step inference path:
            # test_config.model_input.forecasting.num_steps_input=1 paired with
            # test_config.forecast.offset=1, so the diffusion model's single forecast step
            # lines up with a deterministic offset=1 model's +1-step lead time. There is no
            # older step to sum over — the sole loaded step IS the conditioning.
            conditioning_tokens = tokens[:, 0]
        else:
            cfg_name = "test_config" if self.cf.stage == "inference" else "training_config"
            assert tokens.shape[1] > 1, (
                f"diffusion.conditioning='forecast' needs >=2 input steps (conditioning + "
                f"target); got tokens.shape[1]={tokens.shape[1]}. Set "
                f"{cfg_name}.model_input.forecasting.num_steps_input>=2"
                + (
                    " or forecast.offset=1 for single-step inference"
                    if self.cf.stage == "inference"
                    else ""
                )
                + "."
            )
            if self.cf.stage == "inference":
                logger.debug("Using most recent steps as conditioning tokens for inference.")
                conditioning_tokens = tokens[:, 1:].sum(axis=1)
            else:
                # Conditioning: all older context steps [t-1, ..., t-(T-1)].
                # Denoising target: t (newest)
                conditioning_tokens = tokens[:, 1:].sum(axis=1)
                latent_cond_noise_std = self.diff_cf.get("impute_latent_noise_std", 0.0)
                conditioning_tokens = (
                    conditioning_tokens
                    + torch.randn_like(conditioning_tokens) * latent_cond_noise_std
                )
                # Occasionally dropout conditioning for classifier free guidance
                cfg_prob = self.diff_cf.get("classifier_free_guidance_prob", 0.0)
                if np.random.rand() < cfg_prob:
                    conditioning_tokens = torch.zeros_like(conditioning_tokens)
        # X_t (tokens[:, 0], most recent) is the diffusion denoising target;
        # older steps are conditioning.
        meta_info["conditioning_tokens"] = conditioning_tokens
        return tokens[:, 0], meta_info

    def _run_inference_step(
        self,
        fstep: int,
        num_steps: int,
        meta_info: dict,
        coords: torch.Tensor | None,
        model: "Model | None" = None,
        model_params: "ModelParams | None" = None,
        batch: "BatchSamples | None" = None,
        output: "ModelOutput | None" = None,
    ) -> torch.Tensor:
        """Run one diffusion inference step and resolve it to the plain tensor Model.forward
        expects, attaching a DecodeSpec to meta_info when this step needs diffusion-specific
        decoding (bypassing the generic training_mode-gated decode in Model._decode_step).
        Trajectory-mode analytics (only when inference_logging.fe_diffusion_log_analytics is
        enabled) are decoded and stashed into output.diffusion_steps_output directly here,
        rather than deferred to model.py.
        """
        raw = self.inference_forward(
            fstep=fstep, num_steps=num_steps, meta_info=meta_info, coords=coords
        )
        is_trajectory = self.num_ensemble_members <= 1

        if not is_trajectory and not self.rollout:
            # Ensemble without rollout: no diffusion-specific decode handling defined for this
            # combination; falls through to the generic training_mode-gated decode like any
            # other engine (conditioning_tokens is intentionally left untouched, matching prior
            # behaviour).
            return raw

        cond = meta_info.get("conditioning_tokens")
        predict_residual = self.diff_cf.get("predict_residual", False)
        final = raw[-1] if is_trajectory else raw
        result = cond + final if predict_residual else final
        meta_info["conditioning_tokens"] = result

        # Trajectory mode: by default only the final ODE state is a real forecast-step
        # prediction, decoded once (like any other step) by Model._decode_step.
        if is_trajectory and not self.rollout and self.log_analytics:
            output.diffusion_steps_output = self.decode_diffusion_steps(
                model,
                model_params,
                fstep,
                raw,
                batch,
                output.forecast_offset,
                cond,
                predict_residual,
            )

        return result

    def decode_spec(self, training: bool) -> DecodeSpec | None:
        """Static decode instructions for a forecast step (see ForecastEngineBase.decode_spec).

        A pure function of this engine's config (num_ensemble_members, diffusion.rollout) — not
        of any sample/runtime data — even though it varies by trajectory/rollout/ensemble mode.
        None during training, and for ensembling without rollout (both fall back to the generic
        training_mode-gated decode, matching _run_inference_step's handling of that case).
        """
        if training or self.cf.stage != "inference":
            return None
        is_trajectory = self.num_ensemble_members <= 1
        if is_trajectory and not self.rollout:
            return DecodeSpec(decode_latent=True, merge_ensemble=False)
        if not is_trajectory and not self.rollout:
            return None
        # Unified rollout/ensemble path — handles both:
        # - rollout (diffusion.rollout=True): raw is a list; take the final ODE state
        # - ensemble (N > 1): raw is already a (N, healpix_cells, embed_dim) tensor
        return DecodeSpec(decode_latent=False, merge_ensemble=True)

    def decode_diffusion_steps(
        self,
        model: "Model",
        model_params: "ModelParams",
        step: int,
        trajectory: list[torch.Tensor],
        source_samples: "BatchSamples",
        forecast_offset: int,
        cond: torch.Tensor,
        predict_residual: bool,
    ) -> "ModelOutput":
        """
        Decode every ODE denoising step into its own forecast-shaped ``ModelOutput`` (only if
        ``inference_logging.fe_diffusion_log_analytics``).

        The result is shaped like a forecast rollout of ``len(trajectory)`` pseudo-steps, matching
        the indexing ``predict_decoders``/``predict_latent`` expect, but it is written via the
        dedicated ``DiffusionStepsZarrWriter`` independent of ``validation_io.write_output`` or
        ``OutputBatchData`` and not readable by the ``evaluate`` package. It is never merged
        into the real output returned by ``Model.forward``.
        """
        # local import: avoids a model<->diffusion module-level import cycle
        from weathergen.model.model import ModelOutput

        analytics_output = ModelOutput(
            list(range(forecast_offset, forecast_offset + len(trajectory))),
            forecast_offset,
            source_samples,
        )
        for i, toks in enumerate(trajectory):
            toks_abs = cond + toks if predict_residual else toks
            # out_step indexes analytics_output.physical/.latent directly (global forecast step,
            # not chunk-local), so it must start at forecast_offset like the rest of the list,
            # else the leading empty padding slot stays at the wrong end and write_output (which
            # only reads steps >= forecast_offset) never sees the first ODE step.
            out_step = forecast_offset + i
            analytics_output = model.predict_decoders(
                model_params, step, toks_abs, source_samples, analytics_output, out_step=out_step
            )
            analytics_output = model.predict_latent(
                model_params, step, toks_abs, source_samples, analytics_output, out_step=out_step
            )
        return analytics_output

    def forward(
        self,
        tokens: torch.Tensor = None,
        fstep: int = None,
        meta_info: dict | None = None,
        coords: torch.Tensor = None,
        num_steps: int = 10,
        model: "Model | None" = None,
        model_params: "ModelParams | None" = None,
        batch: "BatchSamples | None" = None,
        output: "ModelOutput | None" = None,
    ) -> torch.Tensor:
        """
        Forward pass that routes to training_forward or inference_forward based on model status.

        During training:
            - calls training_forward with tokens, fstep, meta_info, coords
            - extracts datetime conditioning from meta_info and passes through datetime embedder
            - adds noise to target and returns denoised prediction

        During inference:
            - calls inference_forward with fstep, num_steps, and meta_info
            - generates samples via iterative diffusion steps with conditional temporal modulation

        Args:
            tokens: Training tensor of shape (B, H, D) - required during training
            fstep: Forecast step index - required for both modes
            meta_info: Stream-independent meta information: noise_level (training),
                timestamp (date/time conditioning), conditioning_tokens (forecast conditioning)
            coords: Optional coordinate tensor
            num_steps: Number of diffusion steps for inference (default: 30)
            model, model_params, batch, output: only used during inference, to decode and stash
                trajectory analytics (inference_logging.fe_diffusion_log_analytics) directly into
                output.diffusion_steps_output; unused during training.

        Returns:
            torch.Tensor: Model output (denoised prediction during training,
                         or generated sample during inference)

        Raises:
            ValueError: If required arguments are missing for current mode
        """
        # called during training in training mode
        # called during training in training mode
        if self.training:
            if tokens is None or fstep is None or meta_info is None:
                raise ValueError(
                    f"During training, tokens, fstep, and meta_info are required. Got "
                    f"tokens={tokens is not None}, fstep={fstep}, meta_info={meta_info is not None}"
                )
            return self.training_forward(
                tokens=tokens,
                fstep=fstep,
                meta_info=meta_info,
                coords=coords,
            )
        else:
            # called in evaluation mode :
            # decide btw pure noise generation (inference) vs denoising a sample for
            # evaluation (train) using the stage variable
            if self.cf.stage == "train" or self.cf.stage == "train_continue":
                # NOTE: temporary for analysing denoising
                return self.training_forward(
                    tokens=tokens,
                    fstep=fstep,
                    meta_info=meta_info,
                    coords=coords,
                )
            elif self.cf.stage == "inference":
                if fstep is None:
                    raise ValueError(f"During inference, fstep is required. Got fstep={fstep}")
                self.cur_token = (
                    tokens.detach() if self.log_analytics and tokens is not None else None
                )
                # Allow the number of ODE denoising steps to be set from the config.
                # Falls back to the `num_steps` argument default when not configured.
                num_steps = self.num_inference_steps or num_steps
                return self._run_inference_step(
                    fstep, num_steps, meta_info, coords, model, model_params, batch, output
                )

    @abc.abstractmethod
    def training_forward(
        self,
        tokens: torch.Tensor,
        fstep: int,
        meta_info: dict,
        coords: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Model forward call during training: noise the target tokens (B, A+H, D) at the
        per-sample noise level meta_info["noise_level"] and return the model's estimate of the
        clean target tokens (B, A+H, D), with zero aux tokens (see _prepend_aux_tokens).
        """

    @abc.abstractmethod
    def _run_sampler(
        self,
        c: torch.Tensor | None,
        fstep: int,
        num_steps: int,
        coords: torch.Tensor | None,
        batch_size: int = 1,
        return_diffusion_steps: bool = False,
    ) -> "tuple[torch.Tensor, list[torch.Tensor] | None]":
        """Generate a sample by integrating from pure noise.

        Args:
            c: Conditioning tensor (or ``None``).  For ensemble mode this has
                shape ``(batch_size, num_healpix_cells, embed_dim)``.
            fstep: Forecast step index.
            num_steps: Number of integration steps.
            coords: Optional spatial coordinates.
            batch_size: Number of independent noise realisations to integrate in
                parallel.  Defaults to 1 (trajectory / single-sample mode).
            return_diffusion_steps: When ``True``, also return the list of intermediate
                states (one per step).  Set to ``False`` in ensemble mode to
                avoid storing the full trajectory N times.

        Returns:
            ``(final_x, intermediate_x)`` where *final_x* has shape
            ``(batch_size, num_aux_tokens + num_healpix_cells, embed_dim)`` and *intermediate_x*
            is either a list of per-step tensors (when ``return_diffusion_steps=True``)
            or ``None``.
        """

    def _get_conditioning(self, meta_info: dict) -> torch.Tensor | npt.NDArray | None:
        """Conditioning signal for the denoiser from the forecast engine's meta_info."""
        if self.conditioning in ["date_time", "date", "time"]:
            return meta_info["timestamp"]
        if self.conditioning == "forecast":
            # X_{t-1} as conditioning (prepare_fe_input extracts the newest step as target,
            # and passes the older steps here)
            return meta_info["conditioning_tokens"]
        return None

    def _prepend_aux_tokens(self, x_cells: torch.Tensor) -> torch.Tensor:
        """Prepend zero aux tokens to restore the encoder token layout expected downstream."""
        if self.num_aux_tokens == 0:
            return x_cells
        aux = x_cells.new_zeros(x_cells.shape[0], self.num_aux_tokens, x_cells.shape[2])
        return torch.cat([aux, x_cells], dim=1)

    def _net_forward(
        self,
        net_input: torch.Tensor,
        c: torch.Tensor | None,
        noise_emb: torch.Tensor,
        fstep: int,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the noise-conditioned backbone on the (input-preconditioned) healpix-cell tokens
        net_input (B, H, D): embeds/strips/projects the conditioning c according to the
        conditioning type, handles the latent projections and the diffusion registers, and
        returns the raw network output (B, H, D) before output preconditioning.
        """
        # Precondition input and feed through network
        if self.conditioning in ["date_time", "date", "time"]:
            c = self.datetime_embedder(c).to(net_input.device)
            if c.ndim == 2:
                # one timestamp per sample: (B, D_c) -> (B, 1, D_c) to broadcast over cells
                c = c.unsqueeze(1)

        # Strip the encoder's aux tokens from conditioning and RoPE coords (x holds cells only).
        n_aux = self.num_aux_tokens
        n_registers = self.num_diffusion_registers
        if self.conditioning == "forecast" and c is not None:
            c = c[:, n_aux:]
        coords_c = coords[:, n_aux:] if coords is not None else None

        # Project from `ae_global_dim_embed` to `diffusion.latent_dim` with a separate projection
        # for conditioning tokens (if applicable).
        # For ada_ln, `c` is an embedded scalar signal, not encoder tokens — skip its projection.
        if self.latent_proj_up is not None:
            net_input = self.latent_proj_up(net_input)
            if c is not None and self.conditioning_type not in {"ada_ln"}:
                c = self.latent_proj_up_cond(c)

        # Prepend the learnable diffusion registers (zero RoPE coords): (B, H) -> (B, R+H).
        coords_x = coords_c
        if n_registers > 0:
            registers = self.diffusion_registers.tokens.to(dtype=net_input.dtype)
            net_input = torch.cat([registers.expand(net_input.shape[0], -1, -1), net_input], dim=1)
            if coords_c is not None:
                coords_x = torch.cat(
                    [coords_c.new_zeros(coords_c.shape[0], n_registers, 2), coords_c], dim=1
                )
            # Token-wise conditioning types need c at R+H tokens: zero-pad the register slots.
            if (
                self.conditioning == "forecast"
                and c is not None
                and self.conditioning_type not in {"concatenate", "cross_attn"}
            ):
                c = torch.cat([c.new_zeros(c.shape[0], n_registers, c.shape[2]), c], dim=1)
        coords = coords_x

        if self.conditioning_type == "concatenate":
            # Concatenate conditioning tokens along sequence dim:
            # (B, R+H, D') cat (B, H, D') -> (B, R+2H, D'), with matching RoPE coords
            combined = torch.cat([net_input, c], dim=1)
            coords_combined = torch.cat([coords, coords_c], dim=1) if coords is not None else None
            raw_out = self.net(
                combined,
                fstep=fstep,
                coords=coords_combined,
                noise_emb=noise_emb,
                conditioning=None,
            )
            raw_out = raw_out[:, : net_input.shape[1], :]  # Slice back to (B, R+H, D')
        elif self.conditioning_type == "concatenate_hiddendim":
            # Concatenate along hidden dim: (B, R+H, D') cat (B, R+H, D') -> (B, R+H, 2D')
            # ForecastingEngine runs at 2D' throughout and projects back to D' via out_proj
            combined = torch.cat([net_input, c], dim=2)
            raw_out = self.net(
                combined, fstep=fstep, coords=coords, noise_emb=noise_emb, conditioning=None
            )
        elif self.conditioning_type == "concatenate_hdMLP":
            # Concatenate along hidden dim then project back:
            # (B, R+H, D') cat (B, R+H, D') -> (B, R+H, 2D') -> Linear -> (B, R+H, D')
            combined = torch.cat([net_input, c], dim=2)
            projected = self.concat_hd_proj(combined)
            raw_out = self.net(
                projected, fstep=fstep, coords=coords, noise_emb=noise_emb, conditioning=None
            )
        elif self.conditioning_type == "spatial_ada_ln":
            # Pre-modulate each token by the corresponding conditioning token.
            # scale/shift/gate are (B, R+H, D') — per-token and per-channel.
            # gate is applied to raw_out so the network can suppress or amplify each cell's
            # denoised contribution based on the conditioning quality at that cell.
            net_input_mod, spatial_gate = self.spatial_ada_ln(net_input, c)
            raw_out = self.net(
                net_input_mod, fstep=fstep, coords=coords, noise_emb=noise_emb, conditioning=None
            )
            raw_out = raw_out * spatial_gate
        else:
            raw_out = self.net(
                net_input, fstep=fstep, coords=coords, noise_emb=noise_emb, conditioning=c
            )

        # Drop the diffusion registers: (B, R+H, D') -> (B, H, D')
        raw_out = raw_out[:, n_registers:]
        if self.latent_proj_down is not None:
            raw_out = self.latent_proj_down(raw_out)
        return raw_out

    def inference_forward(
        self,
        fstep: int,
        num_steps: int = 50,
        meta_info: dict | None = None,
        coords: torch.Tensor = None,
    ) -> "list[torch.Tensor] | torch.Tensor":
        """
        Forward pass of the diffusion model during inference.

        Iteratively denoises a random sample using the learned score function,
        with optional temporal conditioning extracted from meta_info.
        https://github.com/NVlabs/edm/blob/main/generate.py

        When ``diffusion.sampler.num_ensemble_members > 1`` in the config all N members
        are denoised in a single batched ODE pass and the final tensor of shape
        ``(N, num_healpix_cells, embed_dim)`` is returned directly.  The model
        forward pass in ``model.py`` detects ensemble mode by checking
        ``tokens.shape[0] > 1`` and routes to the ensemble decoding branch.

        Args:
            fstep: Forecast step index for the network
            num_steps: Number of diffusion denoising steps (default: 50)
            meta_info: Stream-independent meta information containing the timestamp
                (date/time conditioning) or conditioning_tokens (forecast conditioning)
            coords: Optional coordinate tensor for spatial conditioning
        Returns:
            list[Tensor]: ODE trajectory (one tensor per denoising step) when
                ``diffusion.sampler.num_ensemble_members == 1`` (default / trajectory mode).
            Tensor: shape ``(N, num_healpix_cells, embed_dim)`` when
                ``diffusion.sampler.num_ensemble_members > 1`` (ensemble mode).
        """

        # Extract conditioning (mirrors training_forward).
        c = self._get_conditioning(meta_info)

        num_ensemble_members: int = self.num_ensemble_members

        # Ensemble mode: draw N independent samples in one batched ODE pass.
        if num_ensemble_members > 1:
            logger.info(f"Diffusion ensemble mode: generating {num_ensemble_members} members.")
            # Build batched conditioning of shape (N, healpix_cells, embed_dim).
            # conditioning_tokens is (1, H, D) on the first rollout step (encoder output) and
            # (N, H, D) on subsequent steps (stored by model.py after the previous ensemble step).
            # expand() is a no-op when the leading dim already matches N, so this handles both.
            c_batched = c.expand(num_ensemble_members, *c.shape[1:]) if c is not None else None
            final_x, _ = self._run_sampler(
                c=c_batched,
                fstep=fstep,
                num_steps=num_steps,
                coords=coords,
                batch_size=num_ensemble_members,
                return_diffusion_steps=False,
            )
            return final_x

        # Default trajectory mode: return all intermediate ODE states (existing behaviour).
        _, intermediate_x = self._run_sampler(
            c=c,
            fstep=fstep,
            num_steps=num_steps,
            coords=coords,
            return_diffusion_steps=True,
        )
        return intermediate_x


# NOTE: Adapted from DiT codebase:
class NoiseEmbedder(torch.nn.Module):
    """
    Embeds the scalar noise level c_noise = ln(sigma) / 4 into a vector representation.

    embedding_type:
      - "dit_legacy": DiT timestep embedding, frequencies 10000^(-k/half) in [1e-4, 1]. Should not
        be used for new runs, only for backward compatibility with legacy checkpoints.
      - "log_fourier": log-spaced frequencies in [f_min, f_max] (rad per unit c_noise), fitted to
        the c_noise range. Optionally the raw c_noise is appended as an extra, directly monotone
        channel. Recommended: f_min=0.5, f_max=8*pi, include_raw (the f_* defaults here).

    The default embedding_type is "dit_legacy" only temporarily, for backward compatibility with
    checkpoints trained before this option existed; new runs should use "log_fourier".
    """

    def __init__(
        self,
        embedding_dim: int,
        frequency_embedding_dim: int,
        dtype=torch.bfloat16,
        embedding_type: str = "dit_legacy",
        f_min: float = 0.5,
        f_max: float = 2 * math.pi * 4,
        include_raw: bool = True,
    ):
        super().__init__()
        assert embedding_type in ("dit_legacy", "log_fourier"), (
            f"Unsupported noise embedding type: {embedding_type}"
        )
        assert 0 < f_min < f_max, f"Need 0 < f_min < f_max (got {f_min}, {f_max})"
        self.dtype = dtype
        self.embedding_type = embedding_type
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.include_raw = embedding_type == "log_fourier" and include_raw
        in_dim = frequency_embedding_dim + (1 if self.include_raw else 0)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, embedding_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(embedding_dim, embedding_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    def log_fourier_embedding(self, t: torch.Tensor):
        """
        Sinusoidal features of c_noise at log-spaced frequencies in [f_min, f_max].
        Frequencies are computed on the fly in fp32 (not a buffer: the model is built on the meta
        device and to_empty()'d, which would leave a buffer uninitialised).
        :return: (N, frequency_embedding_dim [+1]) Tensor.
        """
        t = t.reshape(-1, 1).float()
        half = self.frequency_embedding_dim // 2
        freqs = torch.logspace(
            math.log10(self.f_min),
            math.log10(self.f_max),
            half,
            dtype=torch.float32,
            device=t.device,
        )
        args = t * freqs[None]
        parts = [torch.cos(args), torch.sin(args)]
        if self.frequency_embedding_dim % 2:
            parts.append(torch.zeros_like(t))
        if self.include_raw:
            parts.append(t)
        return torch.cat(parts, dim=-1)

    def timestep_embedding(self, t: float, max_period: int = 10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a scalar or 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # Ensure t is 1D
        if t.ndim == 0:
            t = t.view(1)

        half = self.frequency_embedding_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=self.dtype) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: float):
        if self.embedding_type == "log_fourier":
            t_freq = self.log_fourier_embedding(t)
        else:
            t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class DateTimeEncoder(torch.nn.Module):
    """
    Encodes timestamp(s) into multi-frequency sinusoidal calendar embeddings.

    Inspired by cBottle (Climate in a Bottle) with k=1..8 frequency scales.
    Captures seasonal (day-of-year) and diurnal (time-of-day) cycles at multiple timescales.

    Input shape:  scalar or any tensor shape (...)
    Output shape:  (..., 32) — 8 frequencies × 4 components (cos/sin per signal)

    Output structure for k=1..num_frequencies:
        [cos(2πk·doy_frac), sin(2πk·doy_frac), cos(2πk·tod_frac), sin(2πk·tod_frac)]
    where:
    - doy_frac = day_of_year / days_in_year
    - tod_frac = seconds_of_day / 86400.0
    """

    def __init__(self, conditioning: str):
        super().__init__()
        self.num_frequencies = 8
        assert conditioning in ["date_time", "date", "time"], (
            f"Unsupported conditioning: {conditioning}"
        )
        self.date_only = conditioning == "date"
        self.time_only = conditioning == "time"

    def forward(self, timestamp: npt.NDArray | np.datetime64) -> torch.Tensor:
        """
        Encode numpy datetime64 timestamps into 32D multi-frequency calendar embeddings.

        Args:
            timestamp: np.datetime64 scalar or array of timestamps

        Returns:
            torch.Tensor of shape (..., 32) containing multi-frequency embeddings
        """

        # TODO: Consider adding local time encoding (e.g., using longitude)

        timestamp = np.asarray(timestamp)
        orig_shape = timestamp.shape
        timestamp_flat = timestamp.reshape(-1)

        two_pi = 2.0 * np.pi

        # --- Extract time components ---
        ts_int64 = timestamp_flat.astype("int64")  # seconds since Unix epoch
        seconds_in_day = 86400.0
        tod_frac = (ts_int64 % int(seconds_in_day)) / seconds_in_day  # [0, 1)

        # --- Extract day of year ---
        day_np = timestamp_flat.astype("datetime64[D]")
        year_start = day_np.astype("datetime64[Y]").astype("datetime64[D]")
        next_year_start = (day_np.astype("datetime64[Y]") + np.timedelta64(1, "Y")).astype(
            "datetime64[D]"
        )

        day_of_year_0 = (day_np - year_start).astype(np.int64)  # [0, 365] or [0, 366]
        days_in_year = (next_year_start - year_start).astype(np.int64)  # 365 or 366
        doy_frac = day_of_year_0.astype(np.float32) / days_in_year.astype(np.float32)  # [0, 1)

        # --- Multi-frequency sinusoidal embeddings (vectorized over k) ---
        k = np.arange(1, self.num_frequencies + 1, dtype=np.float32)[None, :]
        doy_phase = two_pi * doy_frac[:, None] * k
        tod_phase = two_pi * tod_frac[:, None] * k

        doy_cos = (
            np.cos(doy_phase).astype(np.float32)
            if not self.time_only
            else np.zeros_like(doy_phase).astype(np.float32)
        )
        doy_sin = (
            np.sin(doy_phase).astype(np.float32)
            if not self.time_only
            else np.zeros_like(doy_phase).astype(np.float32)
        )
        tod_cos = (
            np.cos(tod_phase).astype(np.float32)
            if not self.date_only
            else np.zeros_like(tod_phase).astype(np.float32)
        )
        tod_sin = (
            np.sin(tod_phase).astype(np.float32)
            if not self.date_only
            else np.zeros_like(tod_phase).astype(np.float32)
        )

        # Stack all components: (N, K, 4) -> (N, K*4)
        out = np.stack([doy_cos, doy_sin, tod_cos, tod_sin], axis=-1)
        out = out.reshape(out.shape[0], self.num_frequencies * 4)
        out = torch.from_numpy(out).float()

        return out.reshape(*orig_shape, self.num_frequencies * 4)
