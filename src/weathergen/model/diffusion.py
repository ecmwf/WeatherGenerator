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

        # TODO: remove after single sample experiments
        if self.cur_token is not None:
            # logger.info("checking single sampling")
            assert self.cur_token[0].shape == tokens[0].shape, (
                "first token shape was different between iterations "
                "– violates single sample overfitting with difference"
            )
            assert torch.equal(self.cur_token[0], tokens[0]), (
                f"first token was different between iterations "
                f"– violates single sample overfitting {self.cur_token[0] - tokens[0]}"
            )
            assert torch.equal(self.cur_token, tokens), (
                f"tokens were different between iterations "
                f"– violates single sample overfitting {self.cur_token - tokens}"
            )
        self.cur_token = tokens.detach()

        # return self.inference(fstep=fstep, num_steps=10, coords=coords)

        c = 1  # TODO: add correct preconditioning (e.g., sample/s in previous time step)
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

        return self.denoise(x=y + n, c=c, sigma=sigma, fstep=fstep, coords=coords)

    def denoise(self, x: torch.Tensor, c: torch.Tensor, sigma: float, fstep: int, coords: torch.Tensor = None) -> torch.Tensor:
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

        # Precondition input and feed through network
        x = self.preconditioner.precondition(x, c)

        # Direct prediction: network outputs denoised estimate directly
        # return self.net(x, fstep=fstep, coords=coords, noise_emb=noise_emb)
        return c_skip * x + c_out * self.net(
            c_in * x, fstep=fstep, coords=coords, noise_emb=noise_emb
        )  # Eq. (7) in EDM paper

    def inference(
        self,
        fstep: int,
        num_steps: int = 50,
        coords: torch.Tensor = None,
    ) -> torch.Tensor:
        # Forward pass of the diffusion model during inference (Heun sampler)
        # https://github.com/NVlabs/edm/blob/main/generate.py

        # Sample pure noise (assuming single batch element for now)
        torch.manual_seed(42)
        x = torch.randn(1, self.num_healpix_cells, self.cf.ae_global_dim_embed).to(device="cuda")
        # x = self.cur_token * 1.0 + x * 0.1

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

        # --- Per-step tracking for diagnostics ---
        track = {
            "sigma": [], "x_std": [], "denoised_std": [],
            "l2_to_target": [], "cosine_to_target": [],
            "c_skip": [], "x": [x.cpu()]
        }

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
            denoised = self.denoise(x=x_hat, c=None, sigma=t_hat, fstep=fstep, coords=coords)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(x=x_next, c=None, sigma=t_next, fstep=fstep, coords=coords)
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
                    # flat_d = denoised.reshape(-1).float()
                    # flat_t = self.cur_token.reshape(-1).float()
                    # track["l2_to_target"].append((flat_d - flat_t).norm().item())
                    # track["cosine_to_target"].append(
                    #     torch.nn.functional.cosine_similarity(flat_d.unsqueeze(0), flat_t.unsqueeze(0)).item()
                    # )
                    track["l2_to_target"].append((x_next - self.cur_token).norm().item())
        track["x"].append(self.cur_token.cpu())

        self._plot_sampling_diagnostics(track, num_steps)
        return x_next

    def _plot_sampling_diagnostics(self, track: dict, num_steps: int) -> None:
        """Save a diagnostic plot of the sampling trajectory."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

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
        axes[0].axhline(self.sigma_data, color="grey", ls="--", lw=0.8, label=f"sigma_data={self.sigma_data}")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # 3) Std of x_next and denoised estimate
        axes[1].plot(steps, track["x_std"], "o-", markersize=3, label="x (noisy state)")
        axes[1].plot(steps, track["denoised_std"], "s-", markersize=3, label="denoised estimate")
        if self.cur_token is not None:
            target_std = self.cur_token.std().item()
            axes[1].axhline(target_std, color="grey", ls="--", lw=0.8, label=f"target std={target_std:.3f}")
        axes[1].set_ylabel("std")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        if has_target:
            # 4) L2 error to target
            axes[2].plot(steps, track["l2_to_target"], "o-", markersize=3, color="tab:red")
            axes[2].set_ylabel("L2 error to target")
            axes[2].grid(True, alpha=0.3)

            # # 5) Cosine similarity to target
            # axes[4].plot(steps, track["cosine_to_target"], "o-", markersize=3, color="tab:green")
            # axes[4].set_ylabel("cosine sim to target")
            # axes[4].set_ylim(-1.05, 1.05)
            # axes[4].axhline(1.0, color="grey", ls="--", lw=0.8)
            # axes[4].grid(True, alpha=0.3)

        axes[-1].set_xlabel("sampling step")
        fig.tight_layout()

        out_dir = get_path_run(self.cf)
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path_base = out_dir / "plots" / "validation" / "plots"
        out_path_base.mkdir(exist_ok=True, parents=True)
        fig.savefig(out_path_base / "sampling_diagnostics.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved sampling diagnostics to {out_path_base / 'sampling_diagnostics.png'}")

        vmin, vmax = track["x"][-1].min().item(), track["x"][-1].max().item()
        for s_idx, x in enumerate(track["x"]):
            fig, axes2 = plt.subplots(1, 2, figsize=(12, 5))

            abs_max = max(abs(vmin), abs(vmax)) * 0.1
            # im0 = axes2[0].imshow(x[0].t().cpu(), aspect="auto", cmap="seismic",
            #                        norm=mcolors.SymLogNorm(linthresh=1e-2, vmin=-abs_max, vmax=abs_max))
            im0 = axes2[0].imshow(x[0].t().cpu(), aspect="auto", cmap="seismic", vmin=vmin, vmax=vmax)
            plt.colorbar(im0, ax=axes2[0])
            if s_idx == len(track["x"]) - 1:
                axes2[0].set_title(f"Target")
            else:
                axes2[0].set_title(f"Sample at step {s_idx}")
            axes2[0].set_xlabel("healpix cell")
            axes2[0].set_ylabel("embedding dim")

            diff = (x[0].cpu() - track["x"][-1][0].cpu()).t()
            # im1 = axes2[1].imshow(diff, aspect="auto", cmap="bwr",
            #                        norm=mcolors.SymLogNorm(linthresh=1e-2, vmin=-0.2, vmax=0.2))
            im1 = axes2[1].imshow(diff, aspect="auto", cmap="bwr", vmin=-1, vmax=1)
            plt.colorbar(im1, ax=axes2[1])
            axes2[1].set_title("Difference to target")
            axes2[1].set_xlabel("healpix cell")

            fig.tight_layout()
            plt.savefig(out_path_base / f"sample_{s_idx:05d}.png", dpi=100)
            plt.close(fig)
            logger.info(f"Saved sample visualization to {out_path_base / f'sample_{s_idx:05d}.png'}")


    # # --- OLD inference (before training-aligned sigma & diagnostics) ---
    # def inference(
    #     self,
    #     fstep: int,
    #     num_steps: int = 30,
    #     coords: torch.Tensor = None,
    # ) -> torch.Tensor:
    #     # Forward pass of the diffusion model during inference
    #     # https://github.com/NVlabs/edm/blob/main/generate.py
    #
    #     # Sample noise (assuming single batch element for now)
    #     torch.manual_seed(42)
    #     x = torch.randn(1, self.num_healpix_cells, self.cf.ae_global_dim_embed).to(device="cuda") * 1.0
    #
    #     x = self.cur_token * 0.0 + x
    #
    #     # Time step discretization.
    #     step_indices = torch.arange(num_steps, dtype=torch.float64, device="cuda")
    #     t_steps = (
    #         self.sigma_max ** (1 / self.rho)
    #         + step_indices
    #         / (num_steps - 1)
    #         * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
    #     ) ** self.rho
    #     t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    #
    #     # Main sampling loop.
    #     x_next = x * t_steps[0]
    #     for i, (t_cur, t_next) in enumerate(
    #         zip(t_steps[:-1], t_steps[1:], strict=False)
    #     ):  # 0, ..., N-1
    #         t_cur = torch.tensor([t_cur], device="cuda").float()
    #         t_next = torch.tensor([t_next], device="cuda").float()
    #
    #         print(i, t_cur.item())
    #
    #         x_cur = x_next
    #
    #         x_hat = x_cur
    #         t_hat = t_cur
    #
    #         # Euler step.
    #         denoised = self.denoise(x=x_hat, c=None, sigma=t_hat, fstep=fstep, coords=coords)
    #         d_cur = (x_hat - denoised) / t_hat
    #         x_next = x_hat + (t_next - t_hat) * d_cur
    #
    #         # Apply 2nd order correction.
    #         if i < num_steps - 1:
    #             denoised = self.denoise(x=x_next, c=None, sigma=t_next, fstep=fstep, coords=coords)
    #             d_prime = (x_next - denoised) / t_next
    #             x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    #
    #     return x_next
    # # --- END OLD inference ---


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
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
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
