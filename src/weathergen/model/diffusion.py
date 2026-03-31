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

from weathergen.common.config import Config
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
        self.datetime_embedder = DateTimeEncoder()

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

        self.cur_token = tokens

        c = meta_info["ERA5"].params["timestamp"]  # TODO: add correct preconditioning (e.g., sample/s in previous time step, datetime encoding, etc.)
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

        self._noised_tokens = y + n

        print(f"date was: {c}")

        return self.denoise(x=y + n, c=c, sigma=sigma, fstep=fstep)

    def denoise(self, x: torch.Tensor, c: torch.Tensor, sigma: float, fstep: int) -> torch.Tensor:
        """
        The actual diffusion step, where the model removes noise from the input x under
        consideration of a conditioning c (e.g., previous time steps) and the current diffusion
        noise level sigma.
        """
        # Compute scaling conditionings
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = sigma.log() / 4

        # Embed noise level
        noise_emb = self.noise_embedder(c_noise)

        # Precondition input and feed through network
        x = self.preconditioner.precondition(x, c) #currently does nothing
        c = self.datetime_embedder(c).to(x.device)
    
        return c_skip * x + c_out * self.net(
            c_in * x, fstep=fstep, noise_emb=noise_emb, ada_ln_aux=c
        )  # Eq. (7) in EDM paper

    def inference(
        self,
        fstep: int,
        num_steps: int = 30,
    ) -> torch.Tensor:
        # Forward pass of the diffusion model during inference
        # https://github.com/NVlabs/edm/blob/main/generate.py

        # Sample noise (assuming single batch element for now)
        x = torch.randn(1, self.num_healpix_cells, self.cf.ae_global_dim_embed).to(device="cuda")

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device="cuda")
        t_steps = (
            self.sigma_max ** (1 / self.rho)
            + step_indices
            / (num_steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        x_next = x * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:], strict=False)
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily. (Stochastic sampling; not used for now)
            # gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            # t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            # x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * s_noise * torch.randn_like(x_cur)
            x_hat = x_cur
            t_hat = t_cur

            # Euler step.
            denoised = self.denoise(x=x_hat, c=None, sigma=t_hat, fstep=fstep)  # c to be discussed
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(x=x_next, c=None, sigma=t_next, fstep=fstep)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


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

class DateTimeEncoder(torch.nn.Module):
    """
    Encodes timestamp(s) into multi-frequency sinusoidal calendar embeddings.
    
    Inspired by cBottle (Climate in a Bottle) with k=1..8 frequency scales.
    Captures seasonal (day-of-year) and diurnal (time-of-day) cycles at multiple timescales.
    
    Input shape:  scalar or any tensor shape (...)
    Output shape:  (..., 32) — 8 frequencies × 4 components (cos/sin per signal)
    
    Output structure for k=1..8:
      [cos(2πk·doy/365.25), sin(2πk·doy/365.25), cos(k·t), sin(k·t)]
    where:
      - doy = day of year (0-365.25)
      - t = 2π·seconds_of_day/86400 (time of day in radians, UTC)
    """

    def __init__(self):
        super().__init__()
        self.num_frequencies = 8

    def forward(self, timestamp: np.ndarray) -> torch.Tensor:
        """
        Encode numpy datetime64 timestamps into 32D multi-frequency calendar embeddings.

        Args:
            timestamp: np.datetime64 scalar or array of timestamps
            
        Returns:
            torch.Tensor of shape (..., 32) containing multi-frequency embeddings
        """

        # TODO: Consider adding local time encoding (e.g., using longitude)

        orig_shape = timestamp.shape
        timestamp_flat = timestamp.reshape(-1)
        
        two_pi = 2.0 * np.pi
        
        # --- Extract time components ---
        ts_int64 = timestamp_flat.astype('int64')  # seconds since Unix epoch
        seconds_in_day = 86400.0
        seconds_of_day = (ts_int64 % int(seconds_in_day)) / seconds_in_day  # [0, 1)
        
        # --- Extract day of year ---
        day_np = timestamp_flat.astype('datetime64[D]')
        year_start = day_np.astype('datetime64[Y]').astype('datetime64[D]')
        next_year_start = (day_np.astype('datetime64[Y]') + np.timedelta64(1, 'Y')).astype('datetime64[D]')
        
        day_of_year_0 = (day_np - year_start).astype(np.int64)  # [0, 365] or [0, 366]
        days_in_year = (next_year_start - year_start).astype(np.int64)  # 365 or 366
        doy_frac = day_of_year_0.astype(np.float32) / days_in_year.astype(np.float32)  # [0, 1)
        
        # --- Multi-frequency sinusoidal embeddings ---
        # Build output for all 8 frequency scales
        embeddings = []
        for k in range(1, self.num_frequencies + 1):
            k_float = float(k)
            
            # Day-of-year components: cos(2π·k·doy/365.25), sin(2π·k·doy/365.25)
            doy_phase = two_pi * k_float * doy_frac
            doy_cos = np.cos(doy_phase).astype(np.float32)
            doy_sin = np.sin(doy_phase).astype(np.float32)
            
            # Time-of-day components: cos(k·t), sin(k·t) where t = 2π·seconds_of_day
            tot_phase = k_float * two_pi * seconds_of_day
            tot_cos = np.cos(tot_phase).astype(np.float32)
            tot_sin = np.sin(tot_phase).astype(np.float32)
            
            embeddings.append(doy_cos)
            embeddings.append(doy_sin)
            embeddings.append(tot_cos)
            embeddings.append(tot_sin)
        
        # Stack all components: (N, 32)
        out = np.stack(embeddings, axis=-1)
        out = torch.from_numpy(out).float()
        
        return out.reshape(*orig_shape, self.num_frequencies * 4)