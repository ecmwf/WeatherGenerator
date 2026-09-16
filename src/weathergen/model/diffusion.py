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
from weathergen.datasets.batch import SampleMetaData
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

        # --- Particle guidance (repulsion-only diverse ensemble sampling) ---
        # When enabled *and* more than one ensemble member is sampled, the members are
        # denoised jointly with a repulsive force between them; see
        # _particle_guidance_repulsion. Disabled by default, in which case the sampler is
        # exactly the plain EDM ODE and no extra work is done.
        self.particle_guidance = self.cf.get("diffusion_particle_guidance", False)
        # Repulsion magnitude as a fraction of the ODE drift at sigma_max (see _run_ode).
        self.pg_strength = self.cf.get("diffusion_particle_guidance_strength", 0.1)
        # Annealing exponent p in alpha(sigma) = strength * (sigma / sigma_max_eff) ** p.
        # Repulsion is strongest at high sigma, where the trajectory still decides which
        # mode it falls into, and is annealed to zero so members land on-manifold.
        self.pg_sigma_power = self.cf.get("diffusion_particle_guidance_sigma_power", 1.0)
        # Space in which pairwise member distances are measured: "x0" (denoiser output,
        # recommended) or "xt" (the noisy state itself).
        self.pg_kernel_space = self.cf.get("diffusion_particle_guidance_kernel_space", "x0")
        # Time constant tau, in forecast steps, of the exponential fade applied across the
        # rollout on top of the sigma schedule; 0 disables the fade.
        self.pg_fstep_decay = self.cf.get("diffusion_particle_guidance_fstep_decay", 0.0)
        # First forecast step of the rollout. Fades are measured from here so that they
        # start at 1.0 on the first step actually generated, rather than part-way down
        # the curve when forecast.offset > 0. Shared with the conditioning noise fade.
        self.fstep_offset = self._stage_forecast_offset()
        # Number of HEALPix cells whose particle systems are solved at once; 0 = all.
        # Purely a memory/speed trade-off, the result is identical either way.
        self.pg_cell_chunk = self.cf.get("diffusion_particle_guidance_cell_chunk", 4096)
        assert self.pg_kernel_space in {"x0", "xt"}, (
            f"diffusion_particle_guidance_kernel_space must be 'x0' or 'xt' "
            f"(got '{self.pg_kernel_space}')"
        )

        # EDM stochastic sampler (Karras et al. 2022, Algorithm 2) knobs — inference only.
        # s_churn == 0 (the default) keeps the deterministic Heun sampler, bit-identical.
        # See _stochastic_churn() and _run_ode().
        self.s_churn = float(self.cf.get("fe_diffusion_s_churn", 0.0))
        self.s_min = float(self.cf.get("fe_diffusion_s_min", 0.0))
        _s_max = self.cf.get("fe_diffusion_s_max", None)
        self.s_max = math.inf if _s_max is None else float(_s_max)
        self.s_noise = float(self.cf.get("fe_diffusion_s_noise", 1.0))

        # --- Conditioning noise (per-member perturbation of the latent conditioning) ---
        # Inference only; see _perturb_conditioning(). 0.0 (the default) leaves the
        # conditioning untouched and does not draw from the RNG at all.
        self.cond_noise_std = float(self.cf.get("diffusion_conditioning_noise_std", 0.0))
        # Time constant tau, in forecast steps, of an exponential fade exp(-fstep / tau) on
        # the amplitude across the rollout; 0 keeps it constant.
        self.cond_noise_fstep_decay = float(
            self.cf.get("diffusion_conditioning_noise_fstep_decay", 0.0)
        )
        # Norm the amplitude is relative to: "token" (each HEALPix token's own RMS, so the
        # perturbation follows the local token magnitude) or "member" (one RMS per member,
        # i.e. a spatially uniform amplitude).
        self.cond_noise_norm_scope = self.cf.get("diffusion_conditioning_noise_norm_scope", "token")
        assert self.cond_noise_norm_scope in {"token", "member"}, (
            f"diffusion_conditioning_noise_norm_scope must be 'token' or 'member' "
            f"(got '{self.cond_noise_norm_scope}')"
        )
        # How the perturbation is applied along the denoising trajectory:
        #   "fixed" (default) one draw per forecast step, held constant through the whole
        #           ODE -- the member keeps a persistent perturbed state to forecast from.
        #   "cads"  CADS (Sadat et al., ICLR 2024, arXiv:2310.17347): re-corrupted at every
        #           denoising step under the gamma(t) schedule below, so the conditioning is
        #           destroyed at high sigma and fully restored by the end of the ODE.
        self.cond_noise_schedule = self.cf.get("diffusion_conditioning_noise_schedule", "fixed")
        assert self.cond_noise_schedule in {"cads", "fixed"}, (
            f"diffusion_conditioning_noise_schedule must be 'cads' or 'fixed' "
            f"(got '{self.cond_noise_schedule}')"
        )
        # CADS gamma(t) thresholds on trajectory progress t (1 at the first denoising step,
        # 0 at the last): no corruption for t <= tau1, full corruption for t >= tau2, linear
        # in between. Paper defaults tau1=0.6, tau2=1.0.
        self.cond_noise_tau1 = float(self.cf.get("diffusion_conditioning_noise_tau1", 0.6))
        self.cond_noise_tau2 = float(self.cf.get("diffusion_conditioning_noise_tau2", 1.0))
        assert 0.0 <= self.cond_noise_tau1 <= self.cond_noise_tau2 <= 1.0, (
            f"diffusion_conditioning_noise_tau1/tau2 must satisfy 0 <= tau1 <= tau2 <= 1 "
            f"(got tau1={self.cond_noise_tau1}, tau2={self.cond_noise_tau2})"
        )
        # CADS rescaling mix psi: fraction of the corrupted conditioning taken from the
        # variant renormalised back to the clean mean/std. 1.0 (the paper's recommendation)
        # rescales fully, 0.0 disables it.
        self.cond_noise_rescale_psi = float(
            self.cf.get("diffusion_conditioning_noise_rescale_psi", 1.0)
        )
        assert 0.0 <= self.cond_noise_rescale_psi <= 1.0, (
            f"diffusion_conditioning_noise_rescale_psi must be in [0, 1] "
            f"(got {self.cond_noise_rescale_psi})"
        )

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
            noise_stream = self.cf.get("diffusion", {}).get("noise_stream", "ERA5")
            noise_level_rn = torch.tensor(
                [meta_info[noise_stream].params["noise_level_rn"]], device=tokens.device
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
            # Perturb after expanding, so each member gets its own draw.
            c_batched = self._store_perturbed_conditioning(c_batched, fstep, meta_info)
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
        c = self._store_perturbed_conditioning(c, fstep, meta_info)
        _, intermediate_x = self._run_ode(
            c=c,
            fstep=fstep,
            num_steps=num_steps,
            coords=coords,
            log_diagnostics=True,
            return_trajectory=True,
        )
        return intermediate_x

    def _stage_forecast_offset(self) -> int:
        """``forecast.offset`` of the stage this run is executing.

        ``validation_config`` and ``test_config`` are overlays on ``training_config``
        (see ``Trainer.__init__``), so at inference the offset is whichever of them
        defines it, most specific first -- reading ``training_config`` alone would report
        the training offset for a run whose ``test_config.forecast.offset`` differs.
        """
        stage_keys = (
            ["test_config", "validation_config", "training_config"]
            if self.cf.get("stage", None) == "inference"
            else ["training_config"]
        )
        for key in stage_keys:
            offset = self.cf.get(key, {}).get("forecast", {}).get("offset", None)
            if offset is not None:
                return offset
        return 0

    def _fstep_fade(self, fstep: int, tau: float) -> float:
        """Across-rollout fade factor exp(-fstep / tau); 1.0 when ``tau`` is 0 (disabled).

        Measured from ``fstep_offset`` (the rollout's first forecast step) and clamped at
        zero below it, so the first generated step always receives the full strength even
        when ``forecast.offset > 0``.
        """
        if not tau:
            return 1.0
        return math.exp(-max(0, fstep - self.fstep_offset) / tau)

    def _pg_fstep_fade(self, fstep: int) -> float:
        """Particle-guidance strength fade across the rollout; see :meth:`_fstep_fade`."""
        return self._fstep_fade(fstep, self.pg_fstep_decay)

    @staticmethod
    def _pg_scale(drift: torch.Tensor, force: torch.Tensor, alpha: float) -> torch.Tensor:
        """Scale factor putting the repulsion at ``alpha`` times the ODE drift magnitude.

        The raw potential gradient has no natural scale: it depends on the kernel
        bandwidth, the latent width and the member spread, so a bare strength constant
        would need re-tuning for every configuration (and, with the median bandwidth,
        would sit several orders of magnitude away from 1). Rescaling the whole force by
        one scalar per step leaves the *direction* of the potential gradient untouched --
        it only fixes the overall magnitude, which is exactly what the alpha(sigma)
        schedule is for -- while making the strength knob dimensionless: 0.1 means the
        repulsion perturbs each ODE step by ~10%.
        """
        return alpha * drift.norm() / force.norm().clamp_min(1e-12)

    def _particle_guidance_repulsion(
        self,
        x: torch.Tensor,
        denoised: torch.Tensor,
        sigma: torch.Tensor,
        sigma_max_eff: float,
        fstep: int,
    ) -> "tuple[torch.Tensor, float, float]":
        """Repulsive force between ensemble members, computed per HEALPix cell.

        Implements the repulsion-only ("fixed potential") variant of particle guidance
        (Corso et al., 2024). The N members are drawn from the joint distribution

            p(x^1, ..., x^N)  ~  prod_i p_sigma(x^i) * exp(-alpha(sigma) * Phi(x^1..x^N))

        instead of independently, with the similarity potential

            Phi = sum_cells sum_{i<j} k(z^i, z^j),    k = RBF kernel

        so each member picks up the extra drift -alpha * grad_{x^i} Phi, pushing it away
        from the other members. Note what is *not* done here: each member keeps its own
        score untouched. That is what separates this from SVGD, which additionally
        replaces every particle's score by a kernel-weighted average over all particles
        and locks the driving and repulsive terms at a fixed relative weight. Keeping the
        score intact means the members stay (approximately) marginal samples of p_sigma
        and the strength is a free schedule; with alpha -> 0 the plain EDM ODE is
        recovered exactly.

        The kernel acts *per HEALPix cell*: for each cell the N member latents form their
        own independent N-particle system in R^D. One kernel over the whole flattened
        (num_tokens * D) state would be useless at this dimensionality -- with N ~ 10
        particles in ~10^6 dimensions an RBF either saturates (k -> 0 for every pair, no
        repulsion) or goes flat under the median bandwidth (a uniform push away from the
        centroid). The per-cell form also produces *local* spread, which is what an
        ensemble forecast actually wants.

        Register and class tokens are not spatial, so they receive zero force.

        Note on kernel_space="x0": the kernel is evaluated on the denoiser output
        x0_hat = D(x; sigma) while the force is applied to x, i.e. the Jacobian
        dx0_hat/dx is approximated by the identity. Taking it exactly would mean
        backpropagating through the denoiser at every ODE step. x0-space is preferred
        over xt-space because at high sigma the pairwise distances between the x^i are
        dominated by their independent noise draws rather than by any real difference in
        the forecast they encode -- repulsion there would push apart along noise
        directions, which the next denoiser call simply undoes.

        On top of the sigma schedule the strength is faded across the rollout as
        exp(-fstep / tau). This is not cosmetic: the members share identical conditioning
        only at the first forecast step, which is the one place the repulsion is doing
        real work -- choosing different modes from the same information. Afterwards the
        conditioning is per-member and already diverged, so the model's own dynamics grow
        the spread and further repulsion double-counts it. Worse, the effect compounds:
        k steps of a (1 + eps) inflation give (1 + eps)^k, making the total injected
        spread a function of how far you rolled out. Under the fade the product
        telescopes to ~exp(eps * tau), bounded independently of rollout length.

        Args:
            x: Current noisy state, shape (N, T, D).
            denoised: Denoiser output D(x; sigma) at this step, shape (N, T, D).
            sigma: Current noise level (scalar tensor).
            sigma_max_eff: Upper bound of the sampling schedule, used to normalise the
                annealing factor.
            fstep: Global forecast step, used for the across-rollout fade. Counted from
                the first generated step, so forecast.offset > 0 still starts at 1.0.

        Returns:
            ``(force, alpha, spread)``. *force* has shape (N, T, D) and points away from
            the other members (zero on the register/class tokens); it is unnormalised, the
            caller sets its magnitude. *alpha* is the annealing factor at this sigma and
            forecast step.
            *spread* is the RMS pairwise member distance in kernel space -- the quantity
            particle guidance exists to increase, and the main diagnostic for tuning.
        """
        n_members = x.shape[0]
        n_special = self.cf.num_register_tokens + self.cf.num_class_tokens

        alpha = self.pg_strength * (float(sigma.item()) / sigma_max_eff) ** self.pg_sigma_power
        alpha *= self._pg_fstep_fade(fstep)

        # Kernel coordinates (N, H, D) with the non-spatial tokens dropped, transposed to
        # (H, N, D) so that each HEALPix cell is one independent particle system.
        z_src = denoised if self.pg_kernel_space == "x0" else x
        z_all = z_src[:, n_special:, :].detach().transpose(0, 1)
        n_cells = z_all.shape[0]

        out = torch.zeros_like(x, dtype=torch.float32)
        out_spatial = out[:, n_special:, :]  # view: writes below land in `out`
        off_diag = ~torch.eye(n_members, dtype=torch.bool, device=x.device)

        d2_sum = torch.zeros((), device=x.device, dtype=torch.float64)
        d2_count = 0

        # The cells are independent, so chunking over them is exact -- unlike chunking over
        # members, which would silently drop the couplings this whole method is about. It
        # is worth doing: an fp32 copy of every cell at once is N * H * D * 4 bytes (~0.8 GB
        # for 8 members at healpix level 5 and d2048), and the force doubles that.
        chunk = self.pg_cell_chunk or n_cells
        for start in range(0, n_cells, chunk):
            z = z_all[start : start + chunk].float()

            # Pairwise squared distances per cell, (chunk, N, N). Expanded form rather than
            # cdist, whose accurate mode would materialise a (chunk, N, N, D) tensor.
            z_sq = z.pow(2).sum(dim=-1)
            d2 = (
                z_sq.unsqueeze(2) + z_sq.unsqueeze(1) - 2.0 * torch.bmm(z, z.transpose(1, 2))
            ).clamp_min(0.0)

            # Median heuristic on the off-diagonal entries: h = median(||zi - zj||^2) / log N,
            # which makes sum_j k_ij ~ 1 and keeps the repulsion from either saturating or
            # vanishing. Recomputed at every step and for every cell, because the member
            # spread changes by orders of magnitude along the trajectory and across cells --
            # any fixed bandwidth would be wrong almost everywhere.
            med = d2[:, off_diag].median(dim=1).values  # (chunk,)
            h = (med / math.log(n_members)).clamp_min(1e-12).view(-1, 1, 1)

            k = torch.exp(-d2 / h)  # (chunk, N, N)

            # -grad_{z_i} Phi = (2/h) * sum_j k_ij (z_i - z_j), pointing away from the other
            # members. The i == j term vanishes, so leaving the diagonal of k in is harmless.
            force = (2.0 / h) * (k.sum(dim=-1, keepdim=True) * z - torch.bmm(k, z))
            out_spatial[:, start : start + chunk, :] = force.transpose(0, 1)

            d2_sum += d2[:, off_diag].sum().double()
            d2_count += d2.shape[0] * n_members * (n_members - 1)

        spread = math.sqrt(d2_sum.item() / d2_count)
        return out.to(x.dtype), alpha, spread

    def _stochastic_churn(
        self, x_cur: torch.Tensor, t_cur: torch.Tensor, num_steps: int, sigma_max_eff: float
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """EDM Algorithm 2 churn step: temporarily raise the noise level from ``t_cur`` to
        ``t_hat`` by injecting fresh Gaussian noise, so the subsequent denoise+Heun step acts
        as a Langevin corrector. Returns ``(x_hat, t_hat)``.

        With ``s_max = min(fe_diffusion_s_max, sigma_max_eff)`` and
        ``gamma = min(s_churn / num_steps, sqrt(2) - 1)`` (only for ``t_cur`` inside
        ``[s_min, s_max]``)::

            t_hat = min((1 + gamma) * t_cur, s_max)
            x_hat = x_cur + sqrt(t_hat**2 - t_cur**2) * s_noise * N(0, I)

        ``fe_diffusion_s_max`` is capped at ``sigma_max_eff`` (the top of the training-aligned
        inference schedule) so churn can neither operate at nor raise the noise level into the
        untrained high-sigma tail.

        No-op — returns ``(x_cur, t_cur)`` with the global RNG stream **untouched** — when
        ``s_churn <= 0`` (the default), ``t_cur`` is outside ``[s_min, s_max]``, ``gamma``
        rounds to 0, or the ``s_max`` cap leaves nothing to add. The RNG guard matters: an
        unconditional ``torch.randn_like(...) * 0`` would still advance the RNG and shift the
        initial noise of every later sample / forecast step.

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
        sigma_hat = min((1.0 + gamma) * sigma, s_max)
        if sigma_hat <= sigma:
            # gamma == 0, or the s_max cap clamps t_hat back to t_cur — nothing to inject.
            return x_cur, t_cur
        t_hat = torch.full_like(t_cur, sigma_hat)
        x_hat = x_cur + math.sqrt(sigma_hat**2 - sigma**2) * self.s_noise * torch.randn_like(x_cur)
        return x_hat, t_hat

    def _cads_gamma(self, step_idx: int, num_steps: int) -> float:
        """CADS conditioning-annealing factor at denoising step ``step_idx``.

        ``t`` is the progress along the sampling trajectory, 1.0 at the first step (pure
        noise) down to 0.0 at the last, and (Sadat et al., arXiv:2310.17347, Eq. 3)::

            gamma(t) = 1                        t <= tau1   (clean conditioning)
                       (tau2 - t)/(tau2 - tau1) tau1 < t < tau2
                       0                        t >= tau2   (conditioning destroyed)

        Progress is measured in step index rather than in sigma: the Karras schedule drops
        sigma by orders of magnitude over the first few steps, so tau thresholds expressed
        in sigma would concentrate the whole ramp into them, while the paper's thresholds
        were tuned against a time variable that is roughly linear in the step index.
        """
        t = 1.0 - step_idx / max(num_steps - 1, 1)
        if t <= self.cond_noise_tau1:
            return 1.0
        if t >= self.cond_noise_tau2:
            return 0.0
        return (self.cond_noise_tau2 - t) / (self.cond_noise_tau2 - self.cond_noise_tau1)

    def _cond_noise_amplitude(self, fstep: int) -> float:
        """Noise scale ``s`` at forecast step ``fstep``, after the across-rollout fade."""
        return self.cond_noise_std * self._fstep_fade(fstep, self.cond_noise_fstep_decay)

    def _perturb_conditioning(
        self, c: torch.Tensor | None, fstep: int, gamma: float | None = None
    ) -> torch.Tensor | None:
        """Perturb the latent conditioning tokens per ensemble member (inference only).

        The ensemble's members are denoised from the *same* conditioning, so at the first
        forecast step everything that separates them is the initial noise draw (plus
        particle guidance, if enabled). Perturbing the conditioning instead treats the
        analysis/previous state as uncertain: each member forecasts from its own slightly
        different information, and the model's own dynamics grow that difference over the
        rollout. It is complementary to particle guidance -- that one pushes members apart
        *within* one denoising pass, this one hands them different information to begin
        with.

        Two ways of applying it along the denoising trajectory, selected by
        ``diffusion_conditioning_noise_schedule``:

        ``gamma is None`` -- "fixed": plain additive noise, drawn once per forecast step and
        held constant through the whole ODE::

            c_hat = c + s * scale * n

        so the member carries a persistent perturbed state, the latent analogue of an
        ensemble of initial conditions.

        ``gamma is not None`` -- "cads" (Sadat et al., ICLR 2024, arXiv:2310.17347): the
        variance-preserving mix of their Eq. 2, redrawn at every denoising step with
        ``gamma`` from :meth:`_cads_gamma` rising from 0 to 1 along the trajectory::

            c_hat = sqrt(gamma) * c + s * sqrt(1 - gamma) * scale * n

        The conditioning is therefore destroyed at high sigma -- where the trajectory
        decides which mode it falls into, and where diversity is won -- and fully restored
        by the end of the ODE, where condition alignment is what matters. Because it is
        clean again at the end, the member is left with an unbiased conditioning rather
        than a persistent offset, which is the substantive difference from "fixed".

        The amplitude is relative to the token norm rather than absolute, because latent
        token magnitudes vary by orders of magnitude across cells and checkpoints, so a
        fixed std would be a different perturbation for every run. With
        ``diffusion_conditioning_noise_norm_scope``::

            "token"  scale_h = rms(c[:, h, :])   per HEALPix token (default)
            "member" scale   = rms(c[n])         one value per member

        so ``diffusion_conditioning_noise_std = 0.01`` means "perturb each conditioning
        token by ~1% of its own magnitude". (CADS assumes a standardised conditioning
        vector and uses an absolute s; scaling by the token RMS is what makes their s
        transferable to a latent state that is not unit-scale.)

        With ``diffusion_conditioning_noise_rescale_psi`` > 0 the corrupted conditioning is
        renormalised back to the clean per-member mean/std and mixed back in (their Eq. 4-5,
        psi=1 recommended); the paper reports this prevents divergence at high noise scales
        at a small cost in diversity. Note that it also means ``s`` sets the noise-to-signal
        ratio inside the ramp rather than an absolute magnitude: at gamma=0 there is no
        signal left to be relative to, and rescaling restores the clean scale whatever s
        was.

        The noise is drawn independently per member: on the first rollout step ``c`` is a
        ``(1, H, D)`` tensor expanded to ``(N, H, D)``, and expanding before perturbing is
        what makes the members differ; on later steps ``c`` is already per-member.

        Both schedules additionally fade the amplitude across the rollout as
        exp(-fstep / tau) (``diffusion_conditioning_noise_fstep_decay``), for the same
        reason the particle guidance fade exists: the conditioning is identical across
        members only at the first forecast step, which is where the perturbation buys
        spread that is not already there. Later steps start from conditioning that has
        diverged on its own, and re-injecting a fixed relative perturbation at every step
        compounds -- k steps of (1 + eps) give (1 + eps)^k, so the injected spread would
        otherwise depend on how far you rolled out. This axis is absent from CADS, which
        generates one sample rather than a rollout.

        Register and class tokens are left untouched, matching particle guidance: they
        carry global state rather than a location's forecast, and perturbing them moves
        every cell at once instead of adding local uncertainty.

        Single-sample (trajectory) inference is perturbed as well. There is no within-batch
        spread to create there, but independent runs then differ from one another, which is
        how an ensemble is assembled from separate jobs rather than from one batched pass.

        No-op -- returns ``c`` unchanged, RNG untouched -- when the feature is off, when the
        conditioning is not the latent forecast state (the date/time modes condition on a
        timestamp, which is not a thing to add Gaussian noise to), when the fade has decayed
        the amplitude to zero, or when ``gamma == 1`` (the CADS schedule leaves most of the
        trajectory uncorrupted, so this is the common case). Returning the *same object* is
        what the callers use to detect that nothing happened.

        Note that a perturbed conditioning is materialised: without noise the ensemble
        conditioning is a stride-0 ``expand`` view costing nothing, so enabling this adds
        one real ``(N, H, D)`` tensor while the perturbed copy is in use.
        """
        if c is None or self.conditioning != "forecast" or self.cond_noise_std <= 0.0:
            return c
        if gamma is not None and gamma >= 1.0:
            return c

        s_eff = self._cond_noise_amplitude(fstep)
        if s_eff <= 0.0:
            return c

        n_special = self.cf.num_register_tokens + self.cf.num_class_tokens
        spatial = c[:, n_special:, :].float()
        if self.cond_noise_norm_scope == "token":
            # (N, H, 1): every token gets noise proportional to its own magnitude.
            scale = spatial.pow(2).mean(dim=-1, keepdim=True).sqrt()
        else:
            # (N, 1, 1): one amplitude per member, uniform over the map.
            scale = spatial.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()

        noise = torch.randn_like(spatial) * scale * s_eff
        if gamma is None:
            noised = spatial + noise
        else:
            noised = math.sqrt(gamma) * spatial + math.sqrt(1.0 - gamma) * noise
            if self.cond_noise_rescale_psi > 0.0:
                # Renormalise to the clean per-member statistics (the analogue of CADS'
                # per-sample rescaling) and mix back in with psi.
                dims = (1, 2)
                mean_c = spatial.mean(dim=dims, keepdim=True)
                std_c = spatial.std(dim=dims, keepdim=True)
                mean_n = noised.mean(dim=dims, keepdim=True)
                std_n = noised.std(dim=dims, keepdim=True).clamp_min(1e-12)
                rescaled = (noised - mean_n) / std_n * std_c + mean_c
                psi = self.cond_noise_rescale_psi
                noised = psi * rescaled + (1.0 - psi) * noised

        perturbed = c.clone()  # materialises the expand() view; see docstring
        perturbed[:, n_special:, :] = noised.to(c.dtype)
        return perturbed

    def _store_perturbed_conditioning(
        self,
        c: torch.Tensor | None,
        fstep: int,
        meta_info: dict[str, SampleMetaData] | None,
    ) -> torch.Tensor | None:
        """Apply the "fixed" conditioning perturbation once, before the ODE starts.

        Only for ``diffusion_conditioning_noise_schedule="fixed"``; the CADS schedule
        re-corrupts the conditioning inside :meth:`_run_ode` at every denoising step and
        ends on the clean one, so there is nothing to apply (or to store) here.

        The stored conditioning is kept in sync because with
        ``fe_diffusion_predict_residual`` the caller (``model.py``) adds the tensor held in
        ``meta_info`` back onto the network output: it has to be the state the denoiser was
        actually conditioned on, or the residual would be taken relative to the clean state
        and the perturbation would silently cancel. Nothing is written when the
        perturbation is a no-op.
        """
        if self.cond_noise_schedule != "fixed":
            return c
        perturbed = self._perturb_conditioning(c, fstep)
        if perturbed is not c and meta_info is not None:
            meta_info["LATENT_CONDITIONING_TOKENS"] = perturbed
            logger.info(
                f"Conditioning noise (fixed): std={self.cond_noise_std}, "
                f"scope={self.cond_noise_norm_scope}, tau={self.cond_noise_fstep_decay}, "
                f"fstep={fstep}, fstep_fade="
                f"{self._fstep_fade(fstep, self.cond_noise_fstep_decay):.4f}, "
                f"s_eff={self._cond_noise_amplitude(fstep):.4g}, members={c.shape[0]}"
            )
        return perturbed

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
            _churn = (
                f", stochastic churn: s_churn={self.s_churn}, s_min={self.s_min}, "
                f"s_max={self.s_max}, s_noise={self.s_noise}"
                if self.s_churn > 0
                else " (deterministic Heun sampler)"
            )
            logger.info(
                f"Inference sigma schedule ({self.noise_distribution}): "
                f"sigma_max_eff={sigma_max_eff:.4f} (config={self.sigma_max}, train_max={sigma_max_train:.4f}), "
                f"sigma_min_eff={sigma_min_eff:.4f} "
                f"(config={self.sigma_min}, dist q={sigma_min_quantile:.3f}/{sigma_min_from_dist:.4f}), "
                f"sigma_data={self.sigma_data}, rho={self.rho}, num_steps={num_steps}{_churn}"
            )

        # Particle guidance couples the members, so it needs more than one of them.
        pg_active = self.particle_guidance and batch_size > 1
        if log_diagnostics and self.particle_guidance and batch_size == 1:
            logger.warning(
                "diffusion_particle_guidance is enabled but only one sample is being "
                "denoised, so it is inactive; it requires "
                "fe_diffusion_num_ensemble_members > 1."
            )
        # CADS re-corrupts the conditioning at every denoising step; the "fixed" schedule
        # has already perturbed it once before the ODE (see _store_perturbed_conditioning).
        cads_active = (
            c is not None
            and self.conditioning == "forecast"
            and self.cond_noise_schedule == "cads"
            and self._cond_noise_amplitude(fstep) > 0.0
        )
        if log_diagnostics and cads_active:
            n_corrupted = sum(self._cads_gamma(i, num_steps) < 1.0 for i in range(num_steps))
            logger.info(
                f"Conditioning noise (CADS): s={self.cond_noise_std}, "
                f"tau1={self.cond_noise_tau1}, tau2={self.cond_noise_tau2}, "
                f"psi={self.cond_noise_rescale_psi}, scope={self.cond_noise_norm_scope}, "
                f"rollout tau={self.cond_noise_fstep_decay}, fstep={fstep}, fstep_fade="
                f"{self._fstep_fade(fstep, self.cond_noise_fstep_decay):.4f}, "
                f"s_eff={self._cond_noise_amplitude(fstep):.4g}, "
                f"corrupted steps={n_corrupted}/{num_steps}, members={batch_size}"
            )
        if log_diagnostics and pg_active:
            # The sampling diagnostics plot is written to a fixed path and so only ever
            # shows the last forecast step; this line is how the across-rollout fade is
            # actually observable.
            logger.info(
                f"Particle guidance active: strength={self.pg_strength}, "
                f"sigma_power={self.pg_sigma_power}, kernel_space={self.pg_kernel_space}, "
                f"members={batch_size}, fstep={fstep}, tau={self.pg_fstep_decay}, "
                f"fstep_fade={self._pg_fstep_fade(fstep):.4f}"
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
            "sigma_hat": [],  # post-churn sigma; == "sigma" for the deterministic sampler
            "x_std": [],
            "denoised_std": [],
            "l2_to_target": [],
            "cosine_to_target": [],
            "c_skip": [],
            "d_cur_norm": [],
            "d_cur_step_norm": [],
            "residual_std": [],
            "pg_alpha": [],
            "pg_spread": [],
            "cads_gamma": [],
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

            # Increase noise temporarily (EDM Algorithm 2 churn). No-op — x_hat is x_cur,
            # t_hat is t_cur, RNG untouched — when fe_diffusion_s_churn == 0 (the default),
            # so the deterministic Heun sampler below is unchanged. sigma_max_eff caps the
            # churn so it never reaches into the untrained high-sigma tail.
            x_hat, t_hat = self._stochastic_churn(x_cur, t_cur, num_steps, sigma_max_eff)

            # CADS: one corrupted conditioning per denoising step, used by *both* stages of
            # the Heun step -- re-drawing it between the Euler and the correction evaluation
            # would have them measure two different vector fields, and the second-order
            # correction assumes they are the same one.
            c_step = c
            if cads_active:
                gamma = self._cads_gamma(i, num_steps)
                c_step = self._perturb_conditioning(c, fstep, gamma=gamma)

            # Euler step.
            denoised = self.denoise(x=x_hat, c=c_step, sigma=t_hat, fstep=fstep, coords=coords)
            d_cur = (x_hat - denoised) / t_hat
            if pg_active:
                pg_force, pg_alpha, pg_spread = self._particle_guidance_repulsion(
                    x_hat, denoised, t_hat, sigma_max_eff, fstep
                )
                d_cur = d_cur - self._pg_scale(d_cur, pg_force, pg_alpha) * pg_force
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(
                    x=x_next, c=c_step, sigma=t_next, fstep=fstep, coords=coords
                )
                d_prime = (x_next - denoised) / t_next
                if pg_active:
                    pg_force_p, pg_alpha_p, _ = self._particle_guidance_repulsion(
                        x_next, denoised, t_next, sigma_max_eff, fstep
                    )
                    d_prime = d_prime - self._pg_scale(d_prime, pg_force_p, pg_alpha_p) * pg_force_p
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            # --- Record diagnostics ---
            with torch.no_grad():
                s = t_cur.item()
                track["sigma"].append(s)
                track["sigma_hat"].append(t_hat.item())
                track["c_skip"].append(self.sigma_data**2 / (s**2 + self.sigma_data**2))
                track["x_std"].append(x_next.std().item())
                track["denoised_std"].append(denoised.std().item())
                track["d_cur_norm"].append(d_cur.norm().item())
                track["d_cur_step_norm"].append(((t_next - t_hat) * d_cur).norm().item())
                track["residual_std"].append((x_hat - denoised).std().item())
                if pg_active:
                    track["pg_alpha"].append(pg_alpha)
                    track["pg_spread"].append(pg_spread)
                if cads_active:
                    track["cads_gamma"].append(gamma)
                track["x"].append(x_next.cpu())
                if self.cur_token is not None:
                    track["l2_to_target"].append((x_next - self.cur_token).norm().item())
                    track["x"].append(self.cur_token.cpu())

            if return_trajectory:
                # Move to CPU immediately so the GPU segment containing x_next
                # is fully freed after the next step's allocations.  Keeping all
                # num_steps tensors live on GPU causes non-releasable fragmentation
                # (each step's denoised/d_cur holes land in segments that also hold
                # earlier x_next entries, so those segments can never be returned to
                # CUDA even after empty_cache()).  The decoder/writer accesses the
                # trajectory sequentially, so a .to(device) there is sufficient.
                intermediate_x.append(x_next.cpu())

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
        has_pg = len(track.get("pg_spread", [])) > 0
        has_cads = len(track.get("cads_gamma", [])) > 0
        n_plots = 7 + int(has_pg) + int(has_cads)

        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)

        # 1) Sigma schedule
        axes[0].semilogy(steps, track["sigma"], "o-", markersize=3, label="sigma (schedule)")
        if track.get("sigma_hat") and track["sigma_hat"] != track["sigma"]:
            # Stochastic sampler: show the per-step noise bump from churn.
            axes[0].semilogy(
                steps,
                track["sigma_hat"],
                "x--",
                markersize=4,
                color="tab:green",
                label="sigma_hat (post-churn)",
            )
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

        if has_pg:
            # 8) Particle guidance: member spread (what it is trying to increase) against
            # the annealing factor (how hard it is pushing).
            axes[7].plot(
                steps,
                track["pg_spread"],
                "o-",
                markersize=3,
                color="tab:green",
                label="RMS pairwise member distance",
            )
            axes[7].set_ylabel("member spread")
            axes[7].grid(True, alpha=0.3)
            axes[7].set_title("Particle guidance")
            axes[7].legend(fontsize=8, loc="upper left")
            ax_alpha = axes[7].twinx()
            ax_alpha.plot(
                steps, track["pg_alpha"], "--", lw=1, color="tab:purple", label="alpha(sigma)"
            )
            ax_alpha.set_ylabel("guidance strength")
            ax_alpha.legend(fontsize=8, loc="upper right")

        if has_cads:
            # 9) CADS conditioning annealing: how much of the conditioning the denoiser
            # actually sees at each step, and the noise weight that replaces the rest.
            ax_cads = axes[7 + int(has_pg)]
            ax_cads.plot(
                steps,
                track["cads_gamma"],
                "o-",
                markersize=3,
                color="tab:brown",
                label="gamma(t) (conditioning kept)",
            )
            ax_cads.set_ylabel("gamma")
            ax_cads.set_ylim(-0.05, 1.05)
            ax_cads.set_title(
                f"CADS conditioning annealing  |  s={self.cond_noise_std}, "
                f"tau1={self.cond_noise_tau1}, tau2={self.cond_noise_tau2}, "
                f"psi={self.cond_noise_rescale_psi}"
            )
            ax_cads.grid(True, alpha=0.3)
            ax_cads.legend(fontsize=8, loc="upper left")
            ax_noise = ax_cads.twinx()
            ax_noise.plot(
                steps,
                [math.sqrt(1.0 - g) for g in track["cads_gamma"]],
                "--",
                lw=1,
                color="tab:red",
                label="sqrt(1 - gamma) (noise weight)",
            )
            ax_noise.set_ylabel("noise weight")
            ax_noise.legend(fontsize=8, loc="upper right")

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
