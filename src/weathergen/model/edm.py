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

"""
EDM (Karras et al. 2022) latent diffusion forecast engine, selected by ``diffusion.type: edm``.

EDM-specific options (in addition to the shared ones, see weathergen.model.diffusion):

    diffusion:
      type: edm
      edm:
        sigma_data: 1.0
        preconditioning: False  # EDM c_skip/c_out (Eq. 7); False: direct x0 prediction
        loss_weighting: edm     # edm | min_snr_gamma | uniform_x0, see LossLatentDiffusion
        min_snr_gamma: 5.0
        sampler: {rho, sigma_min, sigma_max, sigma_min_quantile, s_churn, s_min, s_max, s_noise}
"""

import logging
import math

import torch
import tqdm

from weathergen.common.config import Config
from weathergen.model.diffusion import (
    DiffusionForecastEngineBase,
    get_inference_sigma_range,
    rho_sigma_steps,
)
from weathergen.model.engines import ForecastingEngine

logger = logging.getLogger(__name__)


class EDMForecastEngine(DiffusionForecastEngineBase):
    # Adopted from https://github.com/NVlabs/edm/blob/main/training/loss.py#L72

    # Fallback for diffusion.noise_embedding.type, kept for backward compatibility with
    # existing checkpoints.
    default_noise_embedding_type = "dit_legacy"

    def __init__(self, cf: Config, num_healpix_cells: int, forecast_engine: ForecastingEngine):
        super().__init__(cf, num_healpix_cells, forecast_engine)
        if "type" not in (self.diff_cf.get("noise_embedding", None) or {}):
            logger.warning(
                "diffusion.noise_embedding.type not set in config; falling back to 'dit_legacy' "
                "for backward compatibility with existing checkpoints. For new runs we recommend "
                "diffusion.noise_embedding: {type: log_fourier, f_min: 0.5, f_max: 25.13 (8*pi), "
                "include_raw: true}."
            )
        self.preconditioner = Preconditioner()
        edm_cf = self.diff_cf.edm

        # Parameters
        self.sigma_data = edm_cf.sigma_data
        # When True, use EDM preconditioning (c_skip/c_out, EDM Eq. 7) in denoise().
        # When False (default), the network predicts x0 directly (c_skip=0, c_out=1).
        self.edm_preconditioning = edm_cf.get("preconditioning", False)

        # Inference sampler (EDM Karras et al. 2022, Algorithm 2)
        sampler_cf = edm_cf.get("sampler", {})
        self.rho = sampler_cf.get("rho", 7.0)
        sigma_range = get_inference_sigma_range(
            cf, sampler_cf, self.sigma_data, "diffusion.edm.sampler"
        )
        self.sigma_min, self.sigma_max = sigma_range.sigma_min, sigma_range.sigma_max
        self.train_noise_distribution = sigma_range.train_distribution
        self.train_sigma_min = sigma_range.train_sigma_min
        self.train_sigma_max = sigma_range.train_sigma_max
        # s_churn == 0 (the default) keeps the deterministic Heun sampler.
        self.s_churn = float(sampler_cf.get("s_churn", 0.0))
        self.s_min = float(sampler_cf.get("s_min", 0.0))
        _s_max = sampler_cf.get("s_max", None)
        self.s_max = math.inf if _s_max is None else float(_s_max)
        self.s_noise = float(sampler_cf.get("s_noise", 1.0))

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

        c = self._get_conditioning(meta_info)

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

        raw_out = self._net_forward(c_in * x, c=c, noise_emb=noise_emb, fstep=fstep, coords=coords)
        return c_skip * x + c_out * raw_out  # Eq. (7) in EDM paper

    def _stochastic_churn(
        self, x_cur: torch.Tensor, t_cur: torch.Tensor, num_steps: int, sigma_max_eff: float
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """EDM Algorithm 2 churn step: temporarily raise the noise level from ``t_cur`` to
        ``t_post`` by injecting fresh Gaussian noise, so the subsequent denoise+Heun step acts
        as a Langevin corrector. Returns ``(x_post, t_post)``.

        ``diffusion.edm.sampler.s_max`` is capped at ``sigma_max_eff`` (the top of the
        training-aligned inference schedule) so churn can neither operate at nor raise the noise
        level into the untrained high-sigma tail.

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

    def _run_sampler(
        self,
        c: torch.Tensor | None,
        fstep: int,
        num_steps: int,
        coords: torch.Tensor | None,
        batch_size: int = 1,
        return_diffusion_steps: bool = False,
    ) -> "tuple[torch.Tensor, list[torch.Tensor] | None]":
        """Run one complete EDM ODE (Heun) denoising trajectory from pure noise, see
        DiffusionForecastEngineBase._run_sampler."""
        # Only the healpix-cell tokens are diffused.
        x = torch.randn(batch_size, self.num_healpix_cells, self.cf.ae_global_dim_embed).to(
            device="cuda"
        )
        target_cells = (
            self.cur_token[:, self.num_aux_tokens :] if self.cur_token is not None else None
        )

        # Inference sigma range, see get_inference_sigma_range()
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
        t_steps = rho_sigma_steps(self.sigma_min, self.sigma_max, self.rho, num_steps, "cuda")

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


class Preconditioner:
    # Preconditioner, e.g., to concatenate previous frames to the input
    def __init__(self):
        pass

    def precondition(self, x, c):
        return x
