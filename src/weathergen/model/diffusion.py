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


import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
import tqdm

from weathergen.common.config import Config
from weathergen.datasets.batch import BatchSamples
from weathergen.datasets.masking import get_noise_level_cfg
from weathergen.model.engines import ForecastingEngine
from weathergen.utils.inference_logger import ScalarSeriesJSONLWriter

if TYPE_CHECKING:
    from weathergen.model.model import Model, ModelOutput, ModelParams

logger = logging.getLogger(__name__)


class DiffusionRegisters(torch.nn.Module):
    """Learnable register tokens of the denoiser, shape (1, R, D).

    Kept in their own module (rather than as a bare Parameter on DiffusionForecastEngine) so that
    checkpoint loading can (re)initialize them via reset_parameters() without treating the whole
    diffusion engine as a new module. Initialized like the encoder's register tokens
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
        engine (see DiffusionForecastEngine.denoise() below), so they expect conditioning=None
        when reaching ForecastingEngine.forward().
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


class DiffusionForecastEngine(torch.nn.Module):
    # Adopted from https://github.com/NVlabs/edm/blob/main/training/loss.py#L72

    def __init__(self, cf: Config, num_healpix_cells: int, forecast_engine: ForecastingEngine):
        super().__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells
        self.net = forecast_engine
        self.preconditioner = Preconditioner()
        self.diff_cf = self.cf.diffusion
        self.frequency_embedding_dim = self.diff_cf.frequency_embedding_dim
        self.embedding_dim = self.diff_cf.embedding_dim
        noise_emb_cfg = self.diff_cf.get("noise_embedding", None) or {}
        if "type" not in noise_emb_cfg:
            logger.warning(
                "diffusion.noise_embedding.type not set in config; falling back to 'dit_legacy' "
                "for backward compatibility with existing checkpoints. For new runs we recommend "
                "diffusion.noise_embedding: {type: log_fourier, f_min: 0.5, f_max: 25.13 (8*pi), "
                "include_raw: true}."
            )
        self.noise_embedder = NoiseEmbedder(
            embedding_dim=self.embedding_dim,
            frequency_embedding_dim=self.frequency_embedding_dim,
            embedding_type=noise_emb_cfg.get("type", "dit_legacy"),
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

        # Parameters
        self.sigma_data = self.diff_cf.sigma_data
        # When True, use EDM preconditioning (c_skip/c_out, EDM Eq. 7) in denoise().
        # When False (default), the network predicts x0 directly (c_skip=0, c_out=1).
        self.edm_preconditioning = self.diff_cf.get("edm_preconditioning", False)

        # Inference sampler (EDM Karras et al. 2022, Algorithm 2)
        sampler_cf = self.diff_cf.get("sampler", {})
        self.rho = sampler_cf.get("rho", 7.0)
        self.sigma_min_quantile = sampler_cf.get("sigma_min_quantile", 0.05)
        self._set_inference_sigma_range(sampler_cf.get("sigma_min"), sampler_cf.get("sigma_max"))
        self.num_inference_steps = sampler_cf.get("num_steps", None)
        self.num_ensemble_members = sampler_cf.get("num_ensemble_members", 1)
        # s_churn == 0 (the default) keeps the deterministic Heun sampler.
        self.s_churn = float(sampler_cf.get("s_churn", 0.0))
        self.s_min = float(sampler_cf.get("s_min", 0.0))
        _s_max = sampler_cf.get("s_max", None)
        self.s_max = math.inf if _s_max is None else float(_s_max)
        self.s_noise = float(sampler_cf.get("s_noise", 1.0))

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

    def _set_inference_sigma_range(self, sigma_min: float | None, sigma_max: float | None):
        """
        Set the sigma range [sigma_min, sigma_max] of the inference schedule.

        Values not set in diffusion.sampler are derived from the *training* noise level
        distribution (masking_strategy_config.noise_level of the target strategies in
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
        noise_level_cfg = get_noise_level_cfg(self.cf.get("training_config", {}))
        self.train_noise_distribution = None
        self.train_sigma_min, self.train_sigma_max = None, None
        if noise_level_cfg is None:
            assert sigma_min is not None and sigma_max is not None, (
                "diffusion.sampler.sigma_min and sigma_max have to be set if no target strategy "
                "of training_config defines masking_strategy_config.noise_level"
            )
        else:
            q = self.sigma_min_quantile
            self.train_noise_distribution = noise_level_cfg.get("distribution", "log_normal")
            if self.train_noise_distribution == "log_uniform":
                # log(sigma) ~ Uniform[log(sigma_min), log(sigma_max)]; quantiles are linear
                # in log-space, so sigma at quantile q is exp(log_min + q * (log_max - log_min)).
                log_min = math.log(noise_level_cfg["sigma_min"])
                log_max = math.log(noise_level_cfg["sigma_max"])
                self.train_sigma_min = math.exp(log_min)
                self.train_sigma_max = math.exp(log_max)
                sigma_min_from_dist = math.exp(log_min + q * (log_max - log_min))
            elif self.train_noise_distribution == "log_normal":
                # log(sigma) ~ N(p_mean, p_std). Range is taken as ~99.7% interval
                # (p_mean -/+ 3 p_std); sigma at quantile q is exp(p_mean + Phi^-1(q) * p_std),
                # with Phi^-1 approximated by standard z-scores (default q=0.05).
                p_mean, p_std = noise_level_cfg["p_mean"], noise_level_cfg["p_std"]
                self.train_sigma_min = math.exp(p_mean - 3.0 * p_std)
                self.train_sigma_max = math.exp(p_mean + 3.0 * p_std)
                _z_scores = {0.01: -2.326, 0.025: -1.960, 0.05: -1.645, 0.10: -1.282}
                _z = _z_scores.get(q, -1.645)
                sigma_min_from_dist = math.exp(p_mean + _z * p_std)
            else:
                raise ValueError(
                    f"Unsupported noise_level distribution: {self.train_noise_distribution}"
                )

            if sigma_max is None:
                sigma_max = self.train_sigma_max
            if sigma_min is None:
                sigma_min = max(sigma_min_from_dist, self.sigma_data * 0.01)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        assert 0.0 < self.sigma_min < self.sigma_max, (
            "diffusion.sampler requires 0 < sigma_min < sigma_max "
            f"(got sigma_min={self.sigma_min}, sigma_max={self.sigma_max})"
        )
        if noise_level_cfg is not None and (
            self.sigma_min < self.train_sigma_min or self.sigma_max > self.train_sigma_max
        ):
            logger.warning(
                f"Inference sigma range [{self.sigma_min}, {self.sigma_max}] exceeds the "
                f"training noise range [{self.train_sigma_min:.4f}, {self.train_sigma_max:.4f}] "
                f"({self.train_noise_distribution})."
            )

    def forward(
        self,
        tokens: torch.Tensor = None,
        fstep: int = None,
        meta_info: dict | None = None,
        coords: torch.Tensor = None,
        num_steps: int = 10,
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
                return self.inference_forward(
                    fstep=fstep,
                    num_steps=num_steps,
                    meta_info=meta_info,
                    coords=coords,
                )

    def training_forward(
        self,
        tokens: torch.Tensor,
        fstep: int,
        meta_info: dict,
        coords: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Model forward call during training. Unpacks the conditioning c = [x_{t-k}, ..., x_{t}], the
        target y = x_{t+1}, and the random noise eta from the data, computes the diffusion noise
        level sigma, and feeds the noisy target along with the conditioning and sigma through the
        model to return a denoised prediction.
        """
        # Retrieve conditionings [0:-1], target [-1], and noise from data object.
        # TOOD: The data retrieval ignores batch and stream dimension for now (has to be adapted).
        # c = [data.get_input_data(t) for t in range(data.get_sample_len() - 1)]
        # y = data.get_input_data(-1)
        # eta = data.get_input_metadata(-1)

        # y is always the target to denoise (set by DiffusionLatentTargetEncoder.pre_compute)
        y = tokens
        assert y is not None, (
            "diffusion_target_tokens not found in meta_info — "
            "DiffusionLatentTargetEncoder.pre_compute must be called before training_forward"
        )

        c = None
        if self.conditioning in ["date_time", "date", "time"]:
            c = meta_info["timestamp"]
        elif self.conditioning == "forecast":
            # c = meta_info["ERA5"].params["conditioning_tokens"]
            # X_{t-1} as conditioning (model.py extracts last step as target,
            # passes second-to-last here)
            c = meta_info["conditioning_tokens"]

        assert "noise_level" in meta_info, (
            "noise_level (sample meta_info, drawn by the Masker from "
            "masking_strategy_config.noise_level) is required for training"
        )
        # noise level sigma, one per sample: (B,) -> (B, 1, 1) to broadcast over cells and channels
        sigma = torch.as_tensor(
            meta_info["noise_level"], dtype=torch.float32, device=tokens.device
        ).reshape(-1, 1, 1)
        assert sigma.shape[0] == y.shape[0], (
            f"got {sigma.shape[0]} noise levels for {y.shape[0]} samples"
        )
        # Only the healpix-cell tokens are diffused.
        y_cells = y[:, self.num_aux_tokens :]
        x_noisy = y_cells + torch.randn_like(y_cells) * sigma

        denoised = self.denoise(x=x_noisy, c=c, sigma=sigma, fstep=fstep, coords=coords)
        return self._prepend_aux_tokens(denoised)

    def _prepend_aux_tokens(self, x_cells: torch.Tensor) -> torch.Tensor:
        """Prepend zero aux tokens to restore the encoder token layout expected downstream."""
        if self.num_aux_tokens == 0:
            return x_cells
        aux = x_cells.new_zeros(x_cells.shape[0], self.num_aux_tokens, x_cells.shape[2])
        return torch.cat([aux, x_cells], dim=1)

    def denoise(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        sigma: torch.Tensor,
        fstep: int,
        coords: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        The actual diffusion step, where the model removes noise from the input x under
        consideration of a conditioning c (e.g., previous time steps) and the current diffusion
        noise level sigma: either a single noise level of shape (1,) or one noise level per sample
        of shape (B, 1, 1).
        """
        # Scaling coefficients (EDM Eq. 7). With EDM preconditioning enabled, c_skip/c_out
        # keep the network output O(1) across all sigma and make the denoiser output D -> x
        # as sigma -> 0 (skip connection dominates), which stabilises the low-sigma tail of
        # the ODE. With it disabled (default) the network predicts x0 directly (c_skip=0,
        # c_out=1). c_in and c_noise are the EDM values in both cases.
        if self.edm_preconditioning:
            c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
            c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        else:
            c_skip = 0
            c_out = 1
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = sigma.log() / 4

        # Embed noise level: (1, D_emb) or (B, D_emb)
        noise_emb = self.noise_embedder(c_noise.reshape(-1))

        # Precondition input and feed through network
        if self.conditioning in ["date_time", "date", "time"]:
            c = self.datetime_embedder(c).to(x.device)
            if c.ndim == 2:
                # one timestamp per sample: (B, D_c) -> (B, 1, D_c) to broadcast over cells
                c = c.unsqueeze(1)

        net_input = c_in * x

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
        return c_skip * x + c_out * raw_out  # Eq. (7) in EDM paper

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
        c = None
        if self.conditioning in ["date_time", "date", "time"]:
            c = meta_info["timestamp"]
        elif self.conditioning == "forecast":
            c = meta_info["conditioning_tokens"]

        num_ensemble_members: int = self.num_ensemble_members

        # Ensemble mode: draw N independent samples in one batched ODE pass.
        if num_ensemble_members > 1:
            logger.info(f"Diffusion ensemble mode: generating {num_ensemble_members} members.")
            # Build batched conditioning of shape (N, healpix_cells, embed_dim).
            # conditioning_tokens is (1, H, D) on the first rollout step (encoder output) and
            # (N, H, D) on subsequent steps (stored by model.py after the previous ensemble step).
            # expand() is a no-op when the leading dim already matches N, so this handles both.
            c_batched = c.expand(num_ensemble_members, *c.shape[1:]) if c is not None else None
            final_x, _ = self._run_ode(
                c=c_batched,
                fstep=fstep,
                num_steps=num_steps,
                coords=coords,
                batch_size=num_ensemble_members,
                return_diffusion_steps=False,
            )
            return final_x

        # Default trajectory mode: return all intermediate ODE states (existing behaviour).
        _, intermediate_x = self._run_ode(
            c=c,
            fstep=fstep,
            num_steps=num_steps,
            coords=coords,
            return_diffusion_steps=True,
        )
        return intermediate_x

    def _stochastic_churn(
        self, x_cur: torch.Tensor, t_cur: torch.Tensor, num_steps: int, sigma_max_eff: float
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """EDM Algorithm 2 churn step: temporarily raise the noise level from ``t_cur`` to
        ``t_post`` by injecting fresh Gaussian noise, so the subsequent denoise+Heun step acts
        as a Langevin corrector. Returns ``(x_post, t_post)``.

        ``diffusion.sampler.s_max`` is capped at ``sigma_max_eff`` (the top of the training-aligned
        inference schedule) so churn can neither operate at nor raise the noise level into the
        untrained high-sigma tail.

        No-op — returns ``(x_cur, t_cur)`` with the global RNG stream untouched — when
        ``s_churn <= 0`` (the default), ``t_cur`` is outside ``[s_min, s_max]``, ``gamma``
        rounds to 0, or the ``s_max`` cap leaves nothing to add.

        Works for both trajectory mode (``x_cur`` is ``(1, H, D)``) and ensemble mode
        (``x_cur`` is ``(N, H, D)`` — each member gets independent churn noise).
        """
        if self.s_churn <= 0.0:
            return x_cur, t_cur
        s_max = min(self.s_max, sigma_max_eff)
        sigma = t_cur.item()
        if not (self.s_min <= sigma <= s_max):
            return x_cur, t_cur
        gamma = min(self.s_churn / num_steps, math.sqrt(2.0) - 1.0)
        sigma_post = min((1.0 + gamma) * sigma, s_max)
        if sigma_post <= sigma:
            return x_cur, t_cur
        t_post = torch.full_like(t_cur, sigma_post)
        x_post = x_cur + math.sqrt(sigma_post**2 - sigma**2) * self.s_noise * torch.randn_like(
            x_cur
        )
        return x_post, t_post

    def _run_ode(
        self,
        c: torch.Tensor | None,
        fstep: int,
        num_steps: int,
        coords: torch.Tensor | None,
        batch_size: int = 1,
        return_diffusion_steps: bool = False,
    ) -> "tuple[torch.Tensor, list[torch.Tensor] | None]":
        """Run one complete ODE denoising trajectory from pure noise.

        Args:
            c: Conditioning tensor (or ``None``).  For ensemble mode this has
                shape ``(batch_size, num_healpix_cells, embed_dim)``.
            fstep: Forecast step index passed through to :meth:`denoise`.
            num_steps: Number of ODE integration steps.
            coords: Optional spatial coordinates for :meth:`denoise`.
            batch_size: Number of independent noise realisations to denoise in
                parallel.  Defaults to 1 (trajectory / single-sample mode).
            return_diffusion_steps: When ``True``, also return the list of intermediate
                states (one per ODE step).  Set to ``False`` in ensemble mode to
                avoid storing the full trajectory N times.

        Returns:
            ``(final_x, intermediate_x)`` where *final_x* has shape
            ``(batch_size, num_aux_tokens + num_healpix_cells, embed_dim)`` and *intermediate_x*
            is either a list of per-step tensors (when ``return_diffusion_steps=True``)
            or ``None``.
        """
        # Only the healpix-cell tokens are diffused.
        x = torch.randn(batch_size, self.num_healpix_cells, self.cf.ae_global_dim_embed).to(
            device="cuda"
        )
        target_cells = (
            self.cur_token[:, self.num_aux_tokens :] if self.cur_token is not None else None
        )

        # Inference sigma range, see _set_inference_sigma_range()
        sigma_max_eff = self.sigma_max
        sigma_min_eff = self.sigma_min
        _churn_msg = (
            f", stochastic churn: s_churn={self.s_churn}, s_min={self.s_min}, "
            f"s_max={self.s_max}, s_noise={self.s_noise}"
            if self.s_churn > 0
            else " (deterministic Heun sampler)"
        )
        logger.debug(
            f"Inference sigma schedule: sigma_max={sigma_max_eff:.4f}, "
            f"sigma_min={sigma_min_eff:.4f} (training noise: {self.train_noise_distribution}, "
            f"range {self.train_sigma_min}-{self.train_sigma_max}), "
            f"sigma_data={self.sigma_data}, rho={self.rho}, num_steps={num_steps}{_churn_msg}"
        )

        # --- Time step discretization (EDM Eq. 5) with training-aligned bounds ---
        step_indices = torch.arange(num_steps, dtype=torch.float64, device="cuda")
        t_steps = (
            sigma_max_eff ** (1 / self.rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min_eff ** (1 / self.rho) - sigma_max_eff ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        # --- Per-step tracking for diagnostics ---
        track = {
            "sigma": [],
            "sigma_post": [],  # post-churn sigma; == "sigma" for the deterministic sampler
            "x_std": [],
            "denoised_std": [],
            "l2_to_target": [],
            "cosine_to_target": [],
            "c_skip": [],
            "d_cur_norm": [],
            "d_cur_step_norm": [],
            "residual_std": [],
        }

        # Per-step intermediate denoised states (one per ODE step).
        intermediate_x: list[torch.Tensor] = [] if return_diffusion_steps else None

        # Main sampling loop. position=1 nests this bar below the per-sample/batch bar
        # (position=0) in trainer.py's validation loop, so progress is visible at both
        # levels during inference.
        x_next = x * t_steps[0]
        step_pairs = list(zip(t_steps[:-1], t_steps[1:], strict=False))
        for i, (t_cur, t_next) in enumerate(
            tqdm.tqdm(
                step_pairs,
                desc="denoising step",
                position=1,
                leave=False,
                disable=self.cf.rank > 0,
            )
        ):  # 0, ..., N-1
            t_cur = torch.tensor([t_cur], device="cuda").float()
            t_next = torch.tensor([t_next], device="cuda").float()

            x_cur = x_next

            # Increase noise temporarily (EDM Algorithm 2 churn).
            x_post, t_post = self._stochastic_churn(x_cur, t_cur, num_steps, sigma_max_eff)

            # Euler step.
            denoised = self.denoise(x=x_post, c=c, sigma=t_post, fstep=fstep, coords=coords)
            d_cur = (x_post - denoised) / t_post
            x_next = x_post + (t_next - t_post) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(x=x_next, c=c, sigma=t_next, fstep=fstep, coords=coords)
                d_prime = (x_next - denoised) / t_next
                x_next = x_post + (t_next - t_post) * (0.5 * d_cur + 0.5 * d_prime)

            # --- Record diagnostics (only written when analytics logging is enabled) ---
            if self.log_analytics:
                with torch.no_grad():
                    s = t_cur.item()
                    track["sigma"].append(s)
                    track["sigma_post"].append(t_post.item())
                    track["c_skip"].append(self.sigma_data**2 / (s**2 + self.sigma_data**2))
                    track["x_std"].append(x_next.std().item())
                    track["denoised_std"].append(denoised.std().item())
                    track["d_cur_norm"].append(d_cur.norm().item())
                    track["d_cur_step_norm"].append(((t_next - t_post) * d_cur).norm().item())
                    track["residual_std"].append((x_post - denoised).std().item())
                    if target_cells is not None:
                        track["l2_to_target"].append((x_next - target_cells).norm().item())

            if return_diffusion_steps:
                intermediate_x.append(self._prepend_aux_tokens(x_next))

        # self._diagnostics_writer is None outside inference.
        if self._diagnostics_writer is not None:
            self._diagnostics_writer.write(track)

        return self._prepend_aux_tokens(x_next), intermediate_x


def _decode_diffusion_steps(
    model: "Model",
    model_params: "ModelParams",
    step: int,
    tokens: list[torch.Tensor],
    source_samples: "BatchSamples",
    forecast_offset: int,
    cond: torch.Tensor,
    predict_residual: bool,
) -> "ModelOutput":
    """
    Decode every ODE denoising step into its own forecast-shaped ``ModelOutput`` (only if
    ``inference_logging.fe_diffusion_log_analytics``).

    The result is shaped like a forecast rollout of ``len(tokens)`` pseudo-steps, matching
    the indexing ``predict_decoders``/``predict_latent`` expect, but it is written via the
    dedicated ``DiffusionStepsZarrWriter`` independent of ``validation_io.write_output`` or
    ``OutputBatchData`` and not readable by the ``evaluate`` package. It is never merged
    into the real output returned by ``Model.forward``.
    """
    from weathergen.model.model import ModelOutput  # local import: avoids a model<->diffusion cycle

    analytics_output = ModelOutput(
        list(range(forecast_offset, forecast_offset + len(tokens))),
        forecast_offset,
        source_samples,
    )
    for i, toks in enumerate(tokens):
        toks_abs = cond + toks if predict_residual else toks
        # out_step indexes analytics_output.physical/.latent directly (global forecast step, not
        # chunk-local), so it must start at forecast_offset like the rest of the list, else
        # the leading empty padding slot stays at the wrong end and write_output (which only
        # reads steps >= forecast_offset) never sees the first ODE step.
        out_step = forecast_offset + i
        analytics_output = model.predict_decoders(
            model_params, step, toks_abs, source_samples, analytics_output, out_step=out_step
        )
        analytics_output = model.predict_latent(
            model_params, step, toks_abs, source_samples, analytics_output, out_step=out_step
        )
    return analytics_output


class Preconditioner:
    # Preconditioner, e.g., to concatenate previous frames to the input
    def __init__(self):
        pass

    def precondition(self, x, c):
        return x


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
