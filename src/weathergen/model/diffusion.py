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

import numpy as np
import torch

from weathergen.common.config import Config, get_path_run
from weathergen.datasets.batch import SampleMetaData, get_noise_level_rn
from weathergen.model.engines import ForecastingEngine

logger = logging.getLogger(__name__)


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


class DiffusionForecastEngine(torch.nn.Module):
    # Adopted from https://github.com/NVlabs/edm/blob/main/training/loss.py#L72

    def __init__(self, cf: Config, num_healpix_cells: int, forecast_engine: ForecastingEngine):
        super().__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells
        self.net = forecast_engine
        self.preconditioner = Preconditioner()
        self.frequency_embedding_dim = self.cf.frequency_embedding_dim
        self.embedding_dim = self.cf.embedding_dim
        self.noise_embedder = NoiseEmbedder(
            embedding_dim=self.embedding_dim, frequency_embedding_dim=self.frequency_embedding_dim
        )
        self.conditioning = self.cf.get("fe_diffusion_model_conditioning", None)
        self.conditioning_type = self.cf.get("fe_diffusion_model_conditioning_type", None)

        _date_time_modes = {"date_time", "date", "time"}
        assert self.conditioning not in _date_time_modes or self.conditioning_type == "ada_ln", (
            f"fe_diffusion_model_conditioning_type must be 'ada_ln' when "
            f"fe_diffusion_model_conditioning is '{self.conditioning}' "
            f"(got '{self.conditioning_type}')"
        )
        _ada_ln = self.conditioning_type == "ada_ln"
        assert self.cf.get("diffusion_conditioning_embed_dim", None) is not None or not _ada_ln, (
            "diffusion_conditioning_embed_dim must be set when "
            "fe_diffusion_model_conditioning_type is 'ada_ln'"
        )
        _offset = self.cf.get("training_config", {}).get("forecast", {}).get("offset", 0)
        assert self.conditioning not in _date_time_modes or _offset == 0, (
            f"forecast.offset must be 0 when fe_diffusion_model_conditioning is "
            f"'{self.conditioning}' (got offset={_offset})"
        )
        _input_num_steps = (
            self.cf.get("training_config", {})
            .get("model_input", {})
            .get("forecasting", {})
            .get("num_steps_input", 0)
        )
        # assert self.conditioning != "forecast" or _input_num_steps == 2, (
        #     f"forecast.input_num_steps must be 2 when fe_diffusion_model_conditioning is "
        #     f"'{self.conditioning}' (got input_num_steps={_input_num_steps})"
        # )
        assert self.conditioning not in ["date_time", "date", "time"] or _input_num_steps == 1, (
            f"forecast.input_num_steps must be 1 when fe_diffusion_model_conditioning is "
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
            f"fe_diffusion_model_conditioning_type must be 'cross_attn', 'additive', 'cross_attn_rev', 'concatenate', 'concatenate_hiddendim', 'concatenate_hdMLP', or 'spatial_ada_ln' when "
            f"fe_diffusion_model_conditioning is 'forecast' "
            f"(got '{self.conditioning_type}')"
        )

        if self.conditioning and (self.conditioning in ["date_time", "date", "time"]):
            self.datetime_embedder = DateTimeEncoder(self.conditioning)

        # Optional MLP projections for an expanded diffusion latent space:
        # projects encoder tokens (ae_global_dim_embed -> fe_diffusion_latent_dim) before denoising
        # and back (fe_diffusion_latent_dim -> ae_global_dim_embed) after.
        # When fe_diffusion_latent_dim == ae_global_dim_embed (default), these are None (no-op).
        _enc_dim = self.cf.ae_global_dim_embed
        _lat_dim = self.cf.get("fe_diffusion_latent_dim", _enc_dim)
        if _lat_dim != _enc_dim:
            self.latent_proj_up = torch.nn.Linear(_enc_dim, _lat_dim, bias=False)
            self.latent_proj_down = torch.nn.Linear(_lat_dim, _enc_dim, bias=False)
        else:
            self.latent_proj_up = None
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
        self.sigma_min = self.cf.sigma_min
        self.sigma_max = self.cf.sigma_max
        self.sigma_data = self.cf.sigma_data
        self.rho = self.cf.rho
        self.p_mean = self.cf.p_mean
        self.p_std = self.cf.p_std
        self.noise_distribution = self.cf.get("noise_distribution", "log_normal")
        # When True, use EDM preconditioning (c_skip/c_out, EDM Eq. 7) in denoise().
        # When False (default), the network predicts x0 directly (c_skip=0, c_out=1).
        self.edm_preconditioning = self.cf.get("fe_diffusion_edm_preconditioning", False)
        self.cur_token = None  # TODO: re move after single sample experiments
        self._noised_tokens: torch.Tensor | None = None
        self._fixed_noise_level: float | None = None

        self._noise = None

        # Log-space bounds of the training noise distribution (log_uniform).
        # noise_level_rn ~ Uniform[log(sigma_min), log(sigma_max)], so sigma = exp(noise_level_rn).
        self.train_log_min = math.log(self.sigma_min)
        self.train_log_max = math.log(self.sigma_max)

    def forward(
        self,
        tokens: torch.Tensor = None,
        fstep: int = None,
        meta_info: dict[str, SampleMetaData] = None,
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
            meta_info: Sample metadata dict containing timestamps - required for both modes
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
                    f"During training, tokens, fstep, and meta_info are required. "
                    f"Got tokens={tokens is not None}, fstep={fstep}, meta_info={meta_info is not None}"
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
                self.cur_token = tokens.detach() if tokens is not None else None
                # Allow the number of ODE denoising steps to be set from the config.
                # Falls back to the `num_steps` argument default when not configured.
                num_steps = self.cf.get("fe_diffusion_num_inference_steps", None) or num_steps
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
        meta_info: dict[str, SampleMetaData],
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

        self.cur_token = tokens.detach()

        # y is always the target to denoise (set by DiffusionLatentTargetEncoder.pre_compute)
        y = tokens
        assert y is not None, (
            "diffusion_target_tokens not found in meta_info — "
            "DiffusionLatentTargetEncoder.pre_compute must be called before training_forward"
        )

        c = None
        if self.conditioning in ["date_time", "date", "time"]:
            c = meta_info["ERA5"].params["timestamp"]
        elif self.conditioning == "forecast":
            # c = meta_info["ERA5"].params["conditioning_tokens"]          # X_{t-1} as conditioning (model.py extracts last step as target, passes second-to-last here)
            c = meta_info["LATENT_CONDITIONING_TOKENS"]

        if self.training:
            noise_level_rn = torch.tensor(
                [get_noise_level_rn(meta_info)], device=tokens.device
            )
        else:
            # During validation, use fixed noise level (default: 0.0)
            noise_level_rn = torch.tensor(
                [self._fixed_noise_level if self._fixed_noise_level is not None else 0.0],
                device=tokens.device,
            )

        # Compute sigma from noise_level_rn.
        # log_normal: noise_level_rn is eta ~ N(0,1); sigma = exp(eta * p_std + p_mean)
        # log_uniform: noise_level_rn is log_sigma directly; sigma = exp(noise_level_rn)
        # during validation, noise_level_rn is set to a fixed value (default: 0.0), so sigma = exp(0) = 1.0 (no noise) by default
        if self.noise_distribution == "log_uniform" or not self.training:
            sigma = noise_level_rn.exp()
        elif self.noise_distribution == "log_normal":
            sigma = (noise_level_rn * self.p_std + self.p_mean).exp()
        else:
            raise ValueError(f"Unsupported noise_distribution: {self.noise_distribution}")
        n = torch.randn_like(y) * sigma

        self._noised_tokens = (y + n).detach()

        return self.denoise(x=y + n, c=c, sigma=sigma, fstep=fstep, coords=coords)

    def denoise(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        sigma: float,
        fstep: int,
        coords: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        The actual diffusion step, where the model removes noise from the input x under
        consideration of a conditioning c (e.g., previous time steps) and the current diffusion
        noise level sigma.
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

        # Embed noise level
        noise_emb = self.noise_embedder(c_noise)

        # Precondition input and feed through network
        if self.conditioning in ["date_time", "date", "time"]:
            c = self.datetime_embedder(c).to(x.device)

        net_input = c_in * x

        # Project input tokens and (where applicable) conditioning tokens from the encoder
        # latent space (ae_global_dim_embed) up to the diffusion latent space (fe_diffusion_latent_dim).
        # For ada_ln, `c` is an embedded scalar signal, not encoder tokens — skip its projection.
        if self.latent_proj_up is not None:
            net_input = self.latent_proj_up(net_input)
            if c is not None and self.conditioning_type not in {"ada_ln"}:
                c = self.latent_proj_up(c)

        if self.conditioning_type == "concatenate":
            # Concatenate conditioning tokens along sequence dim: (B, H, D') cat (B, H, D') -> (B, 2H, D')
            # Also double coords so 2D RoPE matches the doubled sequence length
            combined = torch.cat([net_input, c], dim=1)
            coords_combined = torch.cat([coords, coords], dim=1) if coords is not None else None
            raw_out = self.net(
                combined,
                fstep=fstep,
                coords=coords_combined,
                noise_emb=noise_emb,
                conditioning=None,
            )
            raw_out = raw_out[:, : x.shape[1], :]  # Slice back to (B, H, D')
            if self.latent_proj_down is not None:
                raw_out = self.latent_proj_down(raw_out)
            return c_skip * x + c_out * raw_out  # Eq. (7) in EDM paper

        if self.conditioning_type == "concatenate_hiddendim":
            # Concatenate along hidden dim: (B, H, D') cat (B, H, D') -> (B, H, 2D')
            # ForecastingEngine runs at 2D' throughout and projects back to D' via out_proj
            combined = torch.cat([net_input, c], dim=2)
            raw_out = self.net(
                combined, fstep=fstep, coords=coords, noise_emb=noise_emb, conditioning=None
            )
            if self.latent_proj_down is not None:
                raw_out = self.latent_proj_down(raw_out)
            return c_skip * x + c_out * raw_out  # Eq. (7) in EDM paper

        if self.conditioning_type == "concatenate_hdMLP":
            # Concatenate along hidden dim then project back: (B, H, D') cat (B, H, D') -> (B, H, 2D') -> Linear -> (B, H, D')
            combined = torch.cat([net_input, c], dim=2)
            projected = self.concat_hd_proj(combined)
            raw_out = self.net(
                projected, fstep=fstep, coords=coords, noise_emb=noise_emb, conditioning=None
            )
            if self.latent_proj_down is not None:
                raw_out = self.latent_proj_down(raw_out)
            return c_skip * x + c_out * raw_out  # Eq. (7) in EDM paper

        if self.conditioning_type == "spatial_ada_ln":
            # Pre-modulate each HEALPix cell's token by the corresponding conditioning token.
            # scale/shift/gate are (B, H, D') — per-cell and per-channel.
            # gate is applied to raw_out so the network can suppress or amplify each cell's
            # denoised contribution based on the conditioning quality at that cell.
            net_input_mod, spatial_gate = self.spatial_ada_ln(net_input, c)
            raw_out = self.net(
                net_input_mod, fstep=fstep, coords=coords, noise_emb=noise_emb, conditioning=None
            )
            raw_out = raw_out * spatial_gate
            if self.latent_proj_down is not None:
                raw_out = self.latent_proj_down(raw_out)
            return c_skip * x + c_out * raw_out  # Eq. (7) in EDM paper

        raw_out = self.net(
            net_input, fstep=fstep, coords=coords, noise_emb=noise_emb, conditioning=c
        )
        if self.latent_proj_down is not None:
            raw_out = self.latent_proj_down(raw_out)
        return c_skip * x + c_out * raw_out  # Eq. (7) in EDM paper

    def inference_forward(
        self,
        fstep: int,
        num_steps: int = 50,
        meta_info: dict[str, SampleMetaData] = None,
        coords: torch.Tensor = None,
    ) -> "list[torch.Tensor] | torch.Tensor":
        """
        Forward pass of the diffusion model during inference.

        Iteratively denoises a random sample using the learned score function,
        with optional temporal conditioning extracted from meta_info.
        https://github.com/NVlabs/edm/blob/main/generate.py

        When ``fe_diffusion_num_ensemble_members > 1`` in the config all N members
        are denoised in a single batched ODE pass and the final tensor of shape
        ``(N, num_healpix_cells, embed_dim)`` is returned directly.  The model
        forward pass in ``model.py`` detects ensemble mode by checking
        ``tokens.shape[0] > 1`` and routes to the ensemble decoding branch.

        Args:
            fstep: Forecast step index for the network
            num_steps: Number of diffusion denoising steps (default: 50)
            meta_info: Optional sample metadata dict containing timestamps for temporal conditioning
            coords: Optional coordinate tensor for spatial conditioning
        Returns:
            list[Tensor]: ODE trajectory (one tensor per denoising step) when
                ``fe_diffusion_num_ensemble_members == 1`` (default / trajectory mode).
            Tensor: shape ``(N, num_healpix_cells, embed_dim)`` when
                ``fe_diffusion_num_ensemble_members > 1`` (ensemble mode).
        """

        # Extract conditioning (mirrors training_forward).
        c = None
        if self.conditioning in ["date_time", "date", "time"]:
            c = meta_info["ERA5"].params["timestamp"]
        elif self.conditioning == "forecast":
            c = meta_info["LATENT_CONDITIONING_TOKENS"]

        num_ensemble_members: int = self.cf.get("fe_diffusion_num_ensemble_members", 1)

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
                log_diagnostics=True,
                return_trajectory=False,
            )
            return final_x

        # Default trajectory mode: return all intermediate ODE states (existing behaviour).
        _, intermediate_x = self._run_ode(
            c=c,
            fstep=fstep,
            num_steps=num_steps,
            coords=coords,
            log_diagnostics=True,
            return_trajectory=True,
        )
        return intermediate_x

    def _run_ode(
        self,
        c: torch.Tensor | None,
        fstep: int,
        num_steps: int,
        coords: torch.Tensor | None,
        batch_size: int = 1,
        log_diagnostics: bool = True,
        return_trajectory: bool = False,
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
            log_diagnostics: Whether to emit the sigma-schedule log message and
                save the diagnostic plot.
            return_trajectory: When ``True``, also return the list of intermediate
                states (one per ODE step).  Set to ``False`` in ensemble mode to
                avoid storing the full trajectory N times.

        Returns:
            ``(final_x, intermediate_x)`` where *final_x* has shape
            ``(batch_size, num_healpix_cells, embed_dim)`` and *intermediate_x*
            is either a list of per-step tensors (when ``return_trajectory=True``)
            or ``None``.
        """
        # The encoder prepends register/class tokens to the healpix-cell latents,
        # so the sampled latent must include them to match the target latent shape.
        num_tokens = self.cf.num_register_tokens + self.cf.num_class_tokens + self.num_healpix_cells
        x = torch.randn(batch_size, num_tokens, self.cf.ae_global_dim_embed).to(device="cuda")

        # --- Training-aligned sigma bounds ---
        # The network only learns to denoise reliably within the sigma range seen during
        # training, so the inference schedule bounds are derived from the *training* noise
        # distribution. Using the wrong distribution here truncates the schedule and leaves
        # the sample under-denoised (e.g. applying the log-normal p_mean/p_std formula to a
        # model trained with log_uniform noise stops the ODE far above sigma_min).
        #   - sigma_max_eff: upper bound of the training distribution (capped by config).
        #     Beyond this the denoiser is in untrained territory and poisons the trajectory.
        #   - sigma_min_eff: quantile ``sigma_min_quantile`` of the training distribution,
        #     floored by the config sigma_min and by sigma_data * 0.01 for numerical
        #     stability (avoids dividing by near-zero sigma in the ODE drift).
        sigma_min_quantile = self.cf.get("sigma_min_quantile", 0.05)
        if self.noise_distribution == "log_uniform":
            # log(sigma) ~ Uniform[log(sigma_min), log(sigma_max)]; quantiles are linear
            # in log-space, so sigma at quantile q is exp(log_min + q * (log_max - log_min)).
            sigma_max_train = math.exp(self.train_log_max)
            log_q = self.train_log_min + sigma_min_quantile * (
                self.train_log_max - self.train_log_min
            )
            sigma_min_from_dist = math.exp(log_q)
        else:
            # log_normal: log(sigma) ~ N(p_mean, p_std). Cap sigma_max at ~99.7th percentile
            # (p_mean + 3 p_std); sigma at quantile q is exp(p_mean + Phi^-1(q) * p_std),
            # with Phi^-1 approximated by standard z-scores (default q=0.05).
            sigma_max_train = math.exp(self.p_mean + 3.0 * self.p_std)
            _z_scores = {0.01: -2.326, 0.025: -1.960, 0.05: -1.645, 0.10: -1.282}
            _z = _z_scores.get(sigma_min_quantile, -1.645)
            sigma_min_from_dist = math.exp(self.p_mean + _z * self.p_std)

        sigma_max_eff = min(self.sigma_max, sigma_max_train)
        sigma_min_eff = max(self.sigma_min, sigma_min_from_dist, self.sigma_data * 0.01)
        if log_diagnostics:
            logger.info(
                f"Inference sigma schedule ({self.noise_distribution}): "
                f"sigma_max_eff={sigma_max_eff:.4f} (config={self.sigma_max}, train_max={sigma_max_train:.4f}), "
                f"sigma_min_eff={sigma_min_eff:.4f} "
                f"(config={self.sigma_min}, dist q={sigma_min_quantile:.3f}/{sigma_min_from_dist:.4f}), "
                f"sigma_data={self.sigma_data}, rho={self.rho}, num_steps={num_steps}"
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
            "x_std": [],
            "denoised_std": [],
            "l2_to_target": [],
            "cosine_to_target": [],
            "c_skip": [],
            "d_cur_norm": [],
            "d_cur_step_norm": [],
            "residual_std": [],
            "x": [x.cpu()],
        }

        # Per-step intermediate denoised states (one per ODE step).
        # Only populated when return_trajectory=True.
        intermediate_x: list[torch.Tensor] = [] if return_trajectory else None

        # Main sampling loop.
        x_next = x * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:], strict=False)
        ):  # 0, ..., N-1
            t_cur = torch.tensor([t_cur], device="cuda").float()
            t_next = torch.tensor([t_next], device="cuda").float()

            x_cur = x_next

            # Increase noise temporarily. (Stochastic sampling; not used for now)
            # gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            # t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            # x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * s_noise * torch.randn_like(x_cur)
            x_hat = x_cur
            t_hat = t_cur

            # Euler step.
            denoised = self.denoise(x=x_hat, c=c, sigma=t_hat, fstep=fstep, coords=coords)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(x=x_next, c=c, sigma=t_next, fstep=fstep, coords=coords)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            # --- Record diagnostics ---
            with torch.no_grad():
                s = t_cur.item()
                track["sigma"].append(s)
                track["c_skip"].append(self.sigma_data**2 / (s**2 + self.sigma_data**2))
                track["x_std"].append(x_next.std().item())
                track["denoised_std"].append(denoised.std().item())
                track["d_cur_norm"].append(d_cur.norm().item())
                track["d_cur_step_norm"].append(((t_next - t_hat) * d_cur).norm().item())
                track["residual_std"].append((x_hat - denoised).std().item())
                track["x"].append(x_next.cpu())
                if self.cur_token is not None:
                    track["l2_to_target"].append((x_next - self.cur_token).norm().item())
                    track["x"].append(self.cur_token.cpu())

            if return_trajectory:
                intermediate_x.append(x_next)

        if log_diagnostics:
            self._plot_sampling_diagnostics(track, num_steps)

        return x_next, intermediate_x

    def _plot_sampling_diagnostics(self, track: dict, num_steps: int) -> None:
        """Save a diagnostic plot of the sampling trajectory."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = list(range(len(track["sigma"])))
        has_target = len(track["l2_to_target"]) > 0
        n_plots = 7

        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)

        # 1) Sigma schedule
        axes[0].semilogy(steps, track["sigma"], "o-", markersize=3)
        axes[0].set_ylabel("sigma (noise level)")
        axes[0].set_title(
            f"Sampling diagnostics  |  sigma_max_eff={track['sigma'][0]:.2f}, "
            f"sigma_data={self.sigma_data}, steps={num_steps}"
        )
        axes[0].axhline(
            self.sigma_data, color="grey", ls="--", lw=0.8, label=f"sigma_data={self.sigma_data}"
        )
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # 2) Std of x_next and denoised estimate
        axes[1].plot(steps, track["x_std"], "o-", markersize=3, label="x (noisy state)")
        axes[1].plot(steps, track["denoised_std"], "s-", markersize=3, label="denoised estimate")
        if self.cur_token is not None:
            target_std = self.cur_token.std().item()
            axes[1].axhline(
                target_std, color="grey", ls="--", lw=0.8, label=f"target std={target_std:.3f}"
            )
        axes[1].set_ylabel("std")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        if has_target:
            # 3) L2 error to target
            axes[2].plot(steps, track["l2_to_target"], "o-", markersize=3, color="tab:red")
            axes[2].set_ylabel("L2 error to target")
            axes[2].grid(True, alpha=0.3)

        # 4) d_cur norm and step norm
        axes[3].semilogy(steps, track["d_cur_norm"], "o-", markersize=3, label="||d_cur||")
        axes[3].semilogy(
            steps,
            track["d_cur_step_norm"],
            "^-",
            markersize=3,
            label="||(t_next - t_hat) * d_cur||",
        )
        axes[3].set_ylabel("norm (log scale)")
        axes[3].set_title("ODE drift norms")
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)

        # 5) Residual std: Std(x_hat - denoised)
        axes[4].semilogy(steps, track["residual_std"], "s-", markersize=3, color="tab:orange")
        axes[4].set_ylabel("std (log scale)")
        axes[4].set_title("Std(x_hat - denoised)")
        axes[4].grid(True, alpha=0.3)

        # 6) Residual std zoomed to [0, 1]
        axes[5].plot(steps, track["residual_std"], "s-", markersize=3, color="tab:orange")
        axes[5].set_ylim(0, 1)
        axes[5].set_ylabel("std (clipped to 1)")
        axes[5].set_title("Std(x_hat - denoised)  [y ≤ 1]")
        axes[5].grid(True, alpha=0.3)

        # 7) Std of x_next over sampling steps
        axes[6].semilogy(steps, track["x_std"], "o-", markersize=3, color="tab:blue")
        axes[6].set_ylabel("std (log scale)")
        axes[6].set_title("Std of x_next over denoising steps")
        axes[6].grid(True, alpha=0.3)

        axes[-1].set_xlabel("sampling step")
        fig.tight_layout()

        out_dir = get_path_run(self.cf)
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path_base = out_dir / "plots" / "validation" / "plots"
        out_path_base.mkdir(exist_ok=True, parents=True)
        fig.savefig(out_path_base / "sampling_diagnostics.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved sampling diagnostics to {out_path_base / 'sampling_diagnostics.png'}")


class Preconditioner:
    # Preconditioner, e.g., to concatenate previous frames to the input
    def __init__(self):
        pass

    def precondition(self, x, c):
        return x


# NOTE: Adapted from DiT codebase:
class NoiseEmbedder(torch.nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, embedding_dim: int, frequency_embedding_dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_dim, embedding_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(embedding_dim, embedding_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

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

    def forward(self, timestamp: np.ndarray | np.datetime64) -> torch.Tensor:
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
