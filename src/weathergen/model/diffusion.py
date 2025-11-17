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

import torch
from torch.nn.functional import silu
from dataclass import dataclass
import weathergen.common.config as config
import numpy as np


from weathergen.model.engines import ForecastingEngine




@dataclass
class BatchData:
    """
    Mock function for the data that will be provided to the diffusion model. Will change.
    """

    model_samples: dict
    target_samples: dict

    def get_sample_len(self):
        return len(list(self.model_samples.keys()))

    def get_input_data(self, t: int):
        return self.model_samples[t]["data"]

    def get_input_metadata(self, t: int):
        return self.model_samples[t]["metadata"]

    def get_target_data(self, t: int):
        return self.target_samples[t]["data"]

    def get_target_metadata(self, t: int):
        return self.target_samples[t]["metadata"]


class DiffusionForecastEngine(torch.nn.Module):
    # Adopted from https://github.com/NVlabs/edm/blob/main/training/loss.py#L72

    def __init__(
        self,
        forecast_engine: ForecastingEngine,
        noise_channels: int, #TODO: how are the determined – dimension of latent space? batch size?
        emb_channels: int, #NOTE: might be wise to choose as a function of noise_channels
        sigma_min: float = 0.002,  # Adapt to GenCast?
        sigma_max: float = 80,
        sigma_data: float = 0.5,
        rho: float = 7,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        noise_encoding: str = 'positional',
    ):
        super().__init__()
        self.net = forecast_engine
        self.preconditioner = Preconditioner()
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if noise_encoding == 'positional' else FourierEmbedding(num_channels=noise_channels) #TODO: Compare with method used in GenCast
        self.map_layer0 = torch.nn.Linear(noise_channels, emb_channels)
        self.map_layer1 = torch.nn.Linear(emb_channels, noise_channels)

        # Parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.p_mean = p_mean
        self.p_std = p_std

    def forward(self, data: BatchData) -> torch.Tensor:
        """
        Model forward call during training. Unpacks the conditioning c = [x_{t-k}, ..., x_{t}], the
        target y = x_{t+1}, and the random noise eta from the data, computes the diffusion noise
        level sigma, and feeds the noisy target along with the conditioning and sigma through the
        model to return a denoised prediction.
        """
        # Retrieve conditionings [0:-1], target [-1], and noise from data object.
        # TOOD: The data retrieval ignores batch and stream dimension for now (has to be adapted).
        c = [data.get_input_data(t) for t in range(data.get_sample_len() - 1)]
        y = data.get_input_data(-1)
        eta = data.get_input_metadata(-1)

        # Compute sigma (noise level) from eta
        # noise = torch.randn(y.shape, device=y.device)  # now eta from MultiStreamDataSampler
        sigma = (eta * self.p_std + self.p_mean).exp()
        n = torch.randn_like(y) * sigma
        return self.denoise(x=y + n, c=c, sigma=sigma)

        # Compute loss -- move this to a separate loss calculator
        # weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2  # Table 1
        # loss = weight * ((y_hat - y) ** 2)

    def denoise(self, x: torch.Tensor, c: torch.Tensor, sigma: float) -> torch.Tensor:
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
        emb = self.embed_noise(c_noise.unsqueeze(-1))  #TODO: check dimensionality

        # Precondition input and feed through network
        x = self.preconditioner.precondition(x, c)
        return c_skip * x + c_out * self.net(c_in * x, emb)  # Eq. (7) in EDM paper

    #TODO: May need to be updated from DiT paper code (currently taken from EDM paper code)
    def embed_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Embed noise level using embedding network and MLPs.
        """
        emb = self.map_noise(noise)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) #TODO: check why this is done in the EBM code 'swap cos/sin'
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))
        return emb

    def inference(
        self,
        x: torch.Tensor,
        num_steps: int = 30,
    ) -> torch.Tensor:
        # Forward pass of the diffusion model during inference
        # https://github.com/NVlabs/edm/blob/main/generate.py

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=x.device)
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
            denoised = self.denoise(x=x_hat, c=None, sigma=t_hat)  # c to be discussed
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(x=x_next, c=None, sigma=t_next)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


class Preconditioner:
    # Preconditioner, e.g., to concatenate previous frames to the input
    def __init__(self):
        pass

    def precondition(self, x, c):
        return x


#NOTE: this is currently based on EDM, not on DiT
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

# #NOTE: this is currently based on EDM, not on DiT
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#TODO: Verify if need to add copyright notice.
#NOTE: This will be imported into attention.py.
#NOTE: This is currently based on GenCast, not on DiT
class LinearNormConditioning(torch.nn.Module):
    """Module for norm conditioning, adapted from GenCast.

    Conditions the normalization of `inputs` by applying a linear layer to the
    `norm_conditioning` which produces the scale and offset for each channel.
    """

    def __init__(self, feature_size: int, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype

        self.conditional_linear_layer = torch.nn.Linear(
            in_features=feature_size,
            out_features=2 * feature_size,
        )
        # Optional: initialize weights similar to TruncatedNormal(stddev=1e-8)
        torch.nn.init.normal_(self.conditional_linear_layer.weight, std=1e-8)
        torch.nn.init.zeros_(self.conditional_linear_layer.bias)

    def forward(self, inputs, norm_conditioning, dtype = None):
        # norm_conditioning: [batch, feature_size]
        # inputs: [batch, ..., feature_size]
        conditional_scale_offset = self.conditional_linear_layer(norm_conditioning.to(self.dtype))
        scale_minus_one, offset = torch.chunk(conditional_scale_offset, 2, dim=-1)
        scale = scale_minus_one + 1.0
        # Reshape scale and offset for broadcasting if needed
        while scale.dim() < inputs.dim():
            scale = scale.unsqueeze(1)
            offset = offset.unsqueeze(1)
        return (inputs * scale + offset).to(self.dtype) #TODO: check if to(self.dtype) needed here
