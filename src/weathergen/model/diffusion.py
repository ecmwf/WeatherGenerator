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
        # Read calendar mode from config (default to 'both')
        self.calendar_mode = self.cf.get("fe_diffusion_calendar_mode", "both")
        self.datetime_embedder = DateTimeEncoder(mode=self.calendar_mode)

        # Optional date/time conditioning: project the calendar embedding into the
        # noise-embedding space so it can simply be summed with the noise embedding and
        # consumed by the existing DiT LinearNormConditioning channel. None disables it.
        self.dt_conditioning_mode = self.cf.get("fe_diffusion_model_conditioning", None)
        if self.dt_conditioning_mode == "date_time":
            self.dt_proj = torch.nn.Linear(
                self.datetime_embedder.embedding_dim, self.embedding_dim
            )
            # Start near zero so the model recovers its unconditional behaviour at init.
            torch.nn.init.normal_(self.dt_proj.weight, std=1e-3)
            torch.nn.init.zeros_(self.dt_proj.bias)
        elif self.dt_conditioning_mode in (None, "none"):
            self.dt_proj = None
        else:
            raise ValueError(
                f"Unknown fe_diffusion_model_conditioning={self.dt_conditioning_mode!r}; "
                "supported: 'date_time', 'none', null."
            )

        # Parameters
        self.sigma_min = self.cf.sigma_min
        self.sigma_max = self.cf.sigma_max
        self.sigma_data = self.cf.sigma_data
        self.rho = self.cf.rho
        self.p_mean = self.cf.p_mean
        self.p_std = self.cf.p_std
        self.cur_token = None  # TODO: re move after single sample experiments
        self._noised_tokens: torch.Tensor | None = None
        self._fixed_noise_level: float | None = None

        self._noise = None

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

        c = None

        y = tokens

        if self.training:
            eta = torch.tensor([meta_info["ERA5"].params["noise_level_rn"]], device=tokens.device)
        else:
            # During validation, use fixed noise level (default: 0.0 = mean of noise distribution)
            noise_level = self._fixed_noise_level if self._fixed_noise_level is not None else 0.0
            eta = torch.tensor([noise_level], device=tokens.device)

        # Compute sigma (noise level) from eta and create noise tensor
        sigma = (eta * self.p_std + self.p_mean).exp()
        n = torch.randn_like(y) * sigma

        self._noised_tokens = (y + n).detach()

        dt_emb = self._compute_dt_emb(meta_info, device=tokens.device, dtype=tokens.dtype)

        return self.denoise(
            x=y + n, c=c, sigma=sigma, fstep=fstep, coords=coords, dt_emb=dt_emb
        )

    def _compute_dt_emb(
        self,
        meta_info: dict[str, SampleMetaData] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Encode the sample's forecast valid time and project it into noise-emb space.

        Returns ``None`` when date/time conditioning is disabled. Raises if conditioning is
        enabled but no valid_time was attached upstream by the data sampler.
        """
        if self.dt_proj is None:
            return None

        if meta_info is None or "ERA5" not in meta_info:
            raise ValueError(
                "fe_diffusion_model_conditioning='date_time' but no ERA5 meta_info was "
                "passed to the diffusion forecast engine."
            )
        valid_time = meta_info["ERA5"].valid_time
        if valid_time is None:
            raise ValueError(
                "fe_diffusion_model_conditioning='date_time' but meta_info['ERA5'].valid_time "
                "is None. Sampler must stamp valid_time on SampleMetaData."
            )

        # DateTimeEncoder takes np.datetime64 (scalar or array) and returns a CPU float tensor.
        valid_time_np = np.asarray(valid_time)
        dt_raw = self.datetime_embedder(valid_time_np).to(device=device)
        # Ensure a leading batch dim so the (..., embedding_dim) result broadcasts with
        # noise_emb of shape (1, embedding_dim).
        if dt_raw.dim() == 1:
            dt_raw = dt_raw.unsqueeze(0)
        dt_emb = self.dt_proj(dt_raw.to(dtype=self.dt_proj.weight.dtype))
        return dt_emb.to(dtype=dtype)

    def denoise(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        sigma: float,
        fstep: int,
        coords: torch.Tensor = None,
        dt_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The actual diffusion step, where the model removes noise from the input x under
        consideration of a conditioning c (e.g., previous time steps) and the current diffusion
        noise level sigma.
        """
        # Compute scaling conditionings (EDM Eq. 7 — disabled for direct prediction)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = sigma.log() / 4

        # Embed noise level
        noise_emb = self.noise_embedder(c_noise)

        # Fold optional date/time conditioning into the noise embedding so the existing
        # LinearNormConditioning channel in the DiT blocks consumes it without changes.
        if dt_emb is not None:
            noise_emb = noise_emb + dt_emb.to(noise_emb.dtype)

        # Precondition input and feed through network
        x = self.preconditioner.precondition(x, c)  # currently does nothing

        return c_skip * x + c_out * self.net(
            c_in * x, fstep=fstep, coords=coords, noise_emb=noise_emb, ada_ln_aux=c
        )  # Eq. (7) in EDM paper

    def inference_forward(
        self,
        fstep: int,
        num_steps: int = 50,
        meta_info: dict[str, SampleMetaData] = None,
        coords: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion model during inference.

        Iteratively denoises a random sample using the learned score function,
        with optional temporal conditioning extracted from meta_info.
        https://github.com/NVlabs/edm/blob/main/generate.py

        Args:
            fstep: Forecast step index for the network
            num_steps: Number of diffusion denoising steps (default: 30)
            meta_info: Optional sample metadata dict containing timestamps for temporal conditioning
            coords: Optional coordinate tensor for spatial conditioning
        Returns:
            torch.Tensor: Generated sample of shape (1, num_healpix_cells, ae_global_dim_embed)
        """

        # Extract conditioning from meta_info (same as training_forward)
        c = None

        # Sample pure noise (assuming single batch element for now)
        # torch.manual_seed(42)
        x = torch.randn(1, self.num_healpix_cells, self.cf.ae_global_dim_embed).to(device="cuda")

        # Pre-compute date/time conditioning embedding once for all ODE steps.
        dt_emb = self._compute_dt_emb(meta_info, device=x.device, dtype=x.dtype)

        ### OLD WAY OF COMPUTING SIGMA SCHEDULE
        # # Time step discretization.
        # step_indices = torch.arange(num_steps, dtype=torch.float64, device="cuda")
        # t_steps = (
        #     self.sigma_max ** (1 / self.rho)
        #     + step_indices
        #     / (num_steps - 1)
        #     * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        # ) ** self.rho
        # t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        ### NEW WAY OF COMPUTING SIGMA SCHEDULE WITH TRAINING-ALIGNED BOUNDS AND DIAGNOSTICS
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
        sigma_min_eff = max(self.sigma_min, self.sigma_data * 0.01)
        logger.info(
            f"Inference sigma schedule: "
            f"sigma_max_eff={sigma_max_eff:.4f} (config={self.sigma_max}, train 3σ={sigma_max_train:.4f}), "
            f"sigma_min_eff={sigma_min_eff:.4f} (config={self.sigma_min}), "
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
        # t_steps = torch.cat(
        #     [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        # )  # t_N = 0

        # --- Per-step tracking for diagnostics ---
        track = {
            "sigma": [],
            "x_std": [],
            "denoised_std": [],
            "l2_to_target": [],
            "cosine_to_target": [],
            "c_skip": [],
            "x": [x.cpu()],
        }

        # Per-step intermediate denoised states (one per ODE step).
        # Returned to the caller so they can be treated as a forecast-step dimension.
        intermediate_x: list[torch.Tensor] = []

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
            denoised = self.denoise(
                x=x_hat, c=c, sigma=t_hat, fstep=fstep, coords=coords, dt_emb=dt_emb
            )
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(
                    x=x_next, c=c, sigma=t_next, fstep=fstep, coords=coords, dt_emb=dt_emb
                )
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            # --- Record diagnostics ---
            with torch.no_grad():
                s = t_cur.item()
                track["sigma"].append(s)
                track["c_skip"].append(self.sigma_data**2 / (s**2 + self.sigma_data**2))
                track["x_std"].append(x_next.std().item())
                track["denoised_std"].append(denoised.std().item())
                track["x"].append(x_next.cpu())
                if self.cur_token is not None:
                    track["l2_to_target"].append((x_next - self.cur_token).norm().item())
                    track["x"].append(self.cur_token.cpu())

            # Record intermediate denoised state for this ODE step.
            intermediate_x.append(x_next)

        self._plot_sampling_diagnostics(track, num_steps)

        return intermediate_x

    def _plot_sampling_diagnostics(self, track: dict, num_steps: int) -> None:
        """Save a diagnostic plot of the sampling trajectory."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = list(range(len(track["sigma"])))
        has_target = len(track["l2_to_target"]) > 0
        n_plots = 3

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

    Modes:
      - 'both': day-of-year and time-of-day (default, 32D)
      - 'day_of_year': only day-of-year (16D)
      - 'time_of_day': only time-of-day (16D)
    """

    def __init__(self, mode: str = "both"):
        super().__init__()
        self.num_frequencies = 8
        self.mode = mode
        if mode == "both":
            self.embedding_dim = self.num_frequencies * 4
        elif mode in ("day_of_year", "time_of_day"):
            self.embedding_dim = self.num_frequencies * 2
        else:
            raise ValueError(f"Unknown DateTimeEncoder mode: {mode}")

    def forward(self, timestamp: np.ndarray) -> torch.Tensor:
        orig_shape = timestamp.shape
        timestamp_flat = timestamp.reshape(-1)
        two_pi = 2.0 * np.pi

        # --- Extract time components ---
        ts_int64 = timestamp_flat.astype("int64")  # seconds since Unix epoch
        seconds_in_day = 86400.0
        seconds_of_day = (ts_int64 % int(seconds_in_day)) / seconds_in_day  # [0, 1)

        # --- Extract day of year ---
        day_np = timestamp_flat.astype("datetime64[D]")
        year_start = day_np.astype("datetime64[Y]").astype("datetime64[D]")
        next_year_start = (day_np.astype("datetime64[Y]") + np.timedelta64(1, "Y")).astype("datetime64[D]")
        day_of_year_0 = (day_np - year_start).astype(np.int64)  # [0, 365] or [0, 366]
        days_in_year = (next_year_start - year_start).astype(np.int64)  # 365 or 366
        doy_frac = day_of_year_0.astype(np.float32) / days_in_year.astype(np.float32)  # [0, 1)

        embeddings = []
        for k in range(1, self.num_frequencies + 1):
            k_float = float(k)
            if self.mode in ("both", "day_of_year"):
                doy_phase = two_pi * k_float * doy_frac
                doy_cos = np.cos(doy_phase).astype(np.float32)
                doy_sin = np.sin(doy_phase).astype(np.float32)
                embeddings.append(doy_cos)
                embeddings.append(doy_sin)
            if self.mode in ("both", "time_of_day"):
                tot_phase = k_float * two_pi * seconds_of_day
                tot_cos = np.cos(tot_phase).astype(np.float32)
                tot_sin = np.sin(tot_phase).astype(np.float32)
                embeddings.append(tot_cos)
                embeddings.append(tot_sin)

        out = np.stack(embeddings, axis=-1)
        out = torch.from_numpy(out).float()
        return out.reshape(*orig_shape, self.embedding_dim)
