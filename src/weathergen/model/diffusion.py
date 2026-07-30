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
            f"diffusion_conditioning_embed_dim must be set when "
            f"fe_diffusion_model_conditioning_type is 'ada_ln'"
        )
        _offset = self.cf.get("training_config", {}).get("forecast", {}).get("offset", 0)        
        assert self.conditioning not in _date_time_modes or _offset == 0, (
            f"forecast.offset must be 0 when fe_diffusion_model_conditioning is "
            f"'{self.conditioning}' (got offset={_offset})"
        )
        _input_num_steps = self.cf.get("training_config", {}).get("model_input", {}).get("forecasting", {}).get("num_steps_input", 0)
        assert self.conditioning != "forecast" or _input_num_steps == 2, (
            f"forecast.input_num_steps must be 2 when fe_diffusion_model_conditioning is "
            f"'{self.conditioning}' (got input_num_steps={_input_num_steps})"
        )
        assert self.conditioning not in ["date_time", "date", "time"] or _input_num_steps == 1, (
            f"forecast.input_num_steps must be 1 when fe_diffusion_model_conditioning is "
            f"'{self.conditioning}' (got input_num_steps={_input_num_steps})"
        )
        assert self.conditioning != "forecast" or self.conditioning_type in {"cross_attn", "additive", "cross_attn_rev", "concatenate", "concatenate_hiddendim", "concatenate_hdMLP"}, (
            f"fe_diffusion_model_conditioning_type must be 'cross_attn', 'additive', 'cross_attn_rev', 'concatenate', 'concatenate_hiddendim', or 'concatenate_hdMLP' when "
            f"fe_diffusion_model_conditioning is 'forecast' "
            f"(got '{self.conditioning_type}')"
        )

        if self.conditioning and (self.conditioning in ["date_time", "date", "time"]):
            self.datetime_embedder = DateTimeEncoder(self.conditioning)

        # Optional MLP projections for an expanded diffusion latent space:
        # projects encoder tokens (ae_global_dim_embed -> fe_diffusion_latent_dim) before denoising
        # and back (fe_diffusion_latent_dim -> ae_global_dim_embed) after.
        # Set fe_diffusion_latent_dim to null (or omit) to disable (default behaviour).
        _enc_dim = self.cf.ae_global_dim_embed
        _lat_dim = self.cf.get("fe_diffusion_latent_dim", None) or _enc_dim
        if _lat_dim != _enc_dim:
            self.latent_proj_up = torch.nn.Linear(_enc_dim, _lat_dim, bias=False)
            self.latent_proj_down = torch.nn.Linear(_lat_dim, _enc_dim, bias=False)
        else:
            self.latent_proj_up = None
            self.latent_proj_down = None

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
        self.cur_token = None  # TODO: re move after single sample experiments
        self._noised_tokens: torch.Tensor | None = None
        self._fixed_noise_level: float | None = None
        # Optional ODEDiagnostics (per-ODE-step maps/spectra), attached by the trainer; the decoder
        # closure it needs is bound by model.py.
        self.diagnostics = None

        self._noise = None

    def forward(
        self,
        tokens: torch.Tensor = None,
        fstep: int = None,
        meta_info: dict[str, SampleMetaData] = None,
        coords: torch.Tensor = None,
        num_steps: int | None = None,
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
            num_steps: Number of diffusion ODE steps for inference. If None (the default, and what
                model.py passes), it is read from config key ``fe_diffusion_num_steps``, which
                itself defaults to 10 — preserving the historical hardcoded value bit-identically.
                Set ``fe_diffusion_num_steps`` in the config (or via ``--options``) to override.

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
                # num_steps is not threaded through by model.py, so an explicit arg is rare; fall
                # back to the config key (default 10 = the historical hardcoded value).
                if num_steps is None:
                    num_steps = self.cf.get("fe_diffusion_num_steps", 10)
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
            c = meta_info["ERA5"].params["conditioning_tokens"]          # X_{t-1} as conditioning (model.py extracts last step as target, passes second-to-last here)

        if self.training:
            noise_level_rn = torch.tensor(
                [meta_info["ERA5"].params["noise_level_rn"]], device=tokens.device
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
        # Compute scaling conditionings (EDM Eq. 7 — disabled for direct prediction)
        # c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        # c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
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

        # Project input tokens and (where applicable) conditioning tokens from encoder latent
        # space (ae_global_dim_embed) up to the diffusion latent space (fe_diffusion_latent_dim).
        # ada_ln conditioning is an embedded scalar signal, not encoder tokens — skip projection.
        if self.latent_proj_up is not None:
            net_input = self.latent_proj_up(net_input)
            if c is not None and self.conditioning_type not in {"ada_ln"}:
                c = self.latent_proj_up(c)

        if self.conditioning_type == "concatenate":
            # Concatenate conditioning tokens along sequence dim: (B, H, D') cat (B, H, D') -> (B, 2H, D')
            # Also double coords so 2D RoPE matches the doubled sequence length
            combined = torch.cat([net_input, c], dim=1)
            coords_combined = torch.cat([coords, coords], dim=1) if coords is not None else None
            raw_out = self.net(combined, fstep=fstep, coords=coords_combined, noise_emb=noise_emb, conditioning=None)
            raw_out = raw_out[:, : x.shape[1], :]  # Slice back to (B, H, D')
            if self.latent_proj_down is not None:
                raw_out = self.latent_proj_down(raw_out)
            return c_skip * x + c_out * raw_out  # Eq. (7) in EDM paper

        if self.conditioning_type == "concatenate_hiddendim":
            # Concatenate along hidden dim: (B, H, D') cat (B, H, D') -> (B, H, 2D')
            # ForecastingEngine runs at 2D' throughout and projects back to D' via out_proj
            combined = torch.cat([net_input, c], dim=2)
            raw_out = self.net(combined, fstep=fstep, coords=coords, noise_emb=noise_emb, conditioning=None)
            if self.latent_proj_down is not None:
                raw_out = self.latent_proj_down(raw_out)
            return c_skip * x + c_out * raw_out  # Eq. (7) in EDM paper

        if self.conditioning_type == "concatenate_hdMLP":
            # Concatenate along hidden dim then project: (B, H, D') cat (B, H, D') -> (B, H, 2D') -> Linear -> (B, H, D')
            combined = torch.cat([net_input, c], dim=2)
            projected = self.concat_hd_proj(combined)
            raw_out = self.net(projected, fstep=fstep, coords=coords, noise_emb=noise_emb, conditioning=None)
            if self.latent_proj_down is not None:
                raw_out = self.latent_proj_down(raw_out)
            return c_skip * x + c_out * raw_out  # Eq. (7) in EDM paper

        raw_out = self.net(net_input, fstep=fstep, coords=coords, noise_emb=noise_emb, conditioning=c)
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
            c = meta_info["ERA5"].params["conditioning_tokens"]

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
        x = torch.randn(batch_size, self.num_healpix_cells, self.cf.ae_global_dim_embed).to(device="cuda")

        # --- Training-aligned sigma bounds ---
        # Training noise: sigma = exp(eta * p_std + p_mean), eta ~ N(0,1).
        # The network only learns to denoise reliably within the training distribution.
        #   - sigma_max_eff: cap at 99.7th percentile = exp(p_mean + 3*p_std)
        #     Beyond this, the denoiser is in untrained territory → garbage predictions
        #     that poison the entire ODE trajectory.
        #   - sigma_min_eff: floor at a level where the network still contributes.
        #     With EDM preconditioning, c_skip = sigma_data^2/(sigma^2+sigma_data^2).
        #     At sigma << sigma_data, c_skip → 1, meaning the output ≈ input (skip
        #     connection dominates) and the network can no longer correct errors.
        #     We stop at sigma_min = max(config value, sigma_data * 0.01), which gives
        #     c_skip ≈ 0.9999 — still some network contribution, and avoids the
        #     numerical instability of dividing by near-zero sigma in the ODE.
        sigma_max_train = math.exp(self.p_mean + 3.0 * self.p_std)
        sigma_max_eff = min(self.sigma_max, sigma_max_train)
        # sigma_max_eff = sigma_max_eff * 3.0

        # --- Training-distribution-aligned sigma_min ---
        # sigma_min_quantile controls what fraction of training samples fall below sigma_min_eff.
        # sigma at quantile q of log-normal(p_mean, p_std): exp(p_mean + Φ⁻¹(q) * p_std).
        # Φ⁻¹ approximated via its standard z-scores; default q=0.05 (5th percentile).
        #   q=0.10 → z=-1.282 → exp(1.5-1.538)≈0.96   (stops right at sigma≈1)
        #   q=0.05 → z=-1.645 → exp(1.5-1.974)≈0.62
        #   q=0.01 → z=-2.326 → exp(1.5-2.791)≈0.27
        sigma_min_quantile = self.cf.get("sigma_min_quantile", 0.05)
        _z_scores = {0.01: -2.326, 0.025: -1.960, 0.05: -1.645, 0.10: -1.282}
        _z = _z_scores.get(sigma_min_quantile, -1.645)
        sigma_min_from_dist = math.exp(self.p_mean + _z * self.p_std)
        sigma_min_eff = max(self.sigma_min, sigma_min_from_dist, self.sigma_data * 0.01)
        if log_diagnostics:
            logger.info(
                f"Inference sigma schedule: "
                f"sigma_max_eff={sigma_max_eff:.4f} (config={self.sigma_max}, train 3σ={sigma_max_train:.4f}), "
                f"sigma_min_eff={sigma_min_eff:.4f} "
                f"(config={self.sigma_min}, dist q={sigma_min_quantile:.3f}/{sigma_min_from_dist:.4f}), "
                f"sigma_data={self.sigma_data}, rho={self.rho}, num_steps={num_steps}"
            )
        # sigma_min_eff = self.cf.get("sigma_min", 0.002)

        # --- Time step discretization (EDM Eq. 5) with training-aligned bounds ---
        step_indices = torch.arange(num_steps, dtype=torch.float64, device="cuda")
        t_steps = (
            sigma_max_eff ** (1 / self.rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min_eff ** (1 / self.rho) - sigma_max_eff ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
        # t_steps = torch.cat(
        #     [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        # )  # t_N = 0

        # --- Per-step tracking for diagnostics ---
        track = {
            "sigma": [],
            "x_std": [],
            "denoised_std": [],
            "rmse_x_t": [],
            "rmse_x0_hat": [],
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

        # Per-ODE-step maps/spectra. Needs a reference target (absent in rollout mode) and a
        # single sample (ensemble batches members on dim 0, with no per-member target).
        diag = self.diagnostics if (self.cur_token is not None and batch_size == 1) else None
        if diag is not None:
            diag.begin(self.cur_token)
        elif self.diagnostics is not None:
            logger.info(
                "ODE diagnostics disabled: no reference target (rollout mode) or "
                f"batch_size={batch_size} > 1."
            )

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
            # Denoised (clean-latent) estimate x0_hat at t_cur, captured before the Heun
            # correction below reassigns `denoised` to D(x_next, t_next).
            x0_hat = denoised
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
                    # Per-element RMSE (‖·‖/√numel), size-independent and comparable to
                    # sigma_data=1. Both the noisy state x_next (last step = terminal sample at
                    # sigma=0) and the denoised estimate x0_hat=D(x_hat) at this sigma.
                    _rn = self.cur_token.numel() ** 0.5
                    track["rmse_x_t"].append((x_next - self.cur_token).norm().item() / _rn)
                    track["rmse_x0_hat"].append((x0_hat - self.cur_token).norm().item() / _rn)
                    track["x"].append(self.cur_token.cpu())
                if diag is not None:
                    # x_hat is the noisy state at t_cur; x0_hat is D(x_hat) at t_cur.
                    diag.on_step(i, t_hat.item(), x_hat, x0_hat)

            if return_trajectory:
                intermediate_x.append(x_next)

        if diag is not None:
            # The actual decoded sample (sigma=0). force=True so it is always recorded regardless
            # of every_n_steps; the denoiser is undefined at the terminal node.
            diag.on_step(num_steps, t_steps[num_steps].item(), x_next, None, force=True)

        if log_diagnostics:
            self._plot_sampling_diagnostics(track, num_steps)

        return x_next, intermediate_x

    def _plot_sampling_diagnostics(self, track: dict, num_steps: int) -> None:
        """Save a diagnostic plot of the sampling trajectory."""
        import matplotlib

        matplotlib.use("Agg")
        import os
        import matplotlib.pyplot as plt

        steps = list(range(len(track["sigma"])))
        has_target = len(track["rmse_x_t"]) > 0
        n_plots = 4 if has_target else 3

        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)
        i = 0

        # 1) Sigma schedule
        axes[i].semilogy(steps, track["sigma"], "o-", markersize=3)
        axes[i].set_ylabel("sigma (noise level)")
        axes[i].set_title(
            f"Sampling diagnostics  |  sigma_max_eff={track['sigma'][0]:.2f}, "
            f"sigma_data={self.sigma_data}, steps={num_steps}"
        )
        axes[i].axhline(
            self.sigma_data, color="grey", ls="--", lw=0.8, label=f"sigma_data={self.sigma_data}"
        )
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
        i += 1

        # 2) Per-element RMSE to target: noisy state x_t vs denoised estimate x̂₀ (comparable to
        # sigma_data=1). x̂₀ sits near the target from the first step; x_t only reaches it at the
        # terminal node (last point = the returned sample at sigma=0).
        if has_target:
            axes[i].plot(steps, track["rmse_x_t"], "o-", markersize=3, color="tab:blue",
                         label="rmse(x_t, z)  (noisy state)")
            axes[i].plot(steps, track["rmse_x0_hat"], "s-", markersize=3, color="tab:red",
                         label=r"rmse($\hat{x}_0$, z)  (denoised estimate)")
            axes[i].set_ylabel("RMSE to target (per-element)")
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
            i += 1

        # 3) Std of x_next and denoised estimate
        axes[i].plot(steps, track["x_std"], "o-", markersize=3, label="x (noisy state)")
        axes[i].plot(steps, track["denoised_std"], "s-", markersize=3, label="denoised estimate")
        if self.cur_token is not None:
            target_std = self.cur_token.std().item()
            axes[i].axhline(
                target_std, color="grey", ls="--", lw=0.8, label=f"target std={target_std:.3f}"
            )
        axes[i].set_ylabel("std")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
        i += 1

        # 4) ODE drift norms
        axes[i].semilogy(steps, track["d_cur_norm"], "o-", markersize=3, label="||d_cur||")
        axes[i].semilogy(steps, track["d_cur_step_norm"], "^-", markersize=3,
                         label="||(t_next - t_hat) * d_cur||")
        axes[i].set_ylabel("norm (log scale)")
        axes[i].set_title("ODE drift norms")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

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
        assert conditioning in ["date_time", "date", "time"], f"Unsupported conditioning: {conditioning}"
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

        doy_cos = np.cos(doy_phase).astype(np.float32) if not self.time_only else np.zeros_like(doy_phase).astype(np.float32)
        doy_sin = np.sin(doy_phase).astype(np.float32) if not self.time_only else np.zeros_like(doy_phase).astype(np.float32)
        tod_cos = np.cos(tod_phase).astype(np.float32) if not self.date_only else np.zeros_like(tod_phase).astype(np.float32)
        tod_sin = np.sin(tod_phase).astype(np.float32) if not self.date_only else np.zeros_like(tod_phase).astype(np.float32)

        # Stack all components: (N, K, 4) -> (N, K*4)
        out = np.stack([doy_cos, doy_sin, tod_cos, tod_sin], axis=-1)
        out = out.reshape(out.shape[0], self.num_frequencies * 4)
        out = torch.from_numpy(out).float()

        return out.reshape(*orig_shape, self.num_frequencies * 4)
