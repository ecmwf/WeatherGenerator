# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import partial

import torch
import torch.nn as nn

import numpy as np

class DiagonalGaussianDistribution:
    """
    Used to represent a learned Gaussian Distribution as typical in a VAE
    Code taken and adapted from: https://github.com/Jiawei-Yang/DeTok/tree/main
    """
    def __init__(self, parameters, deterministic=False, channel_dim=1):
        self.parameters = parameters.float()
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=channel_dim)
        self.sum_dims = tuple(range(1, self.mean.dim()))
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=self.sum_dims,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=self.sum_dims,
                )

    def nll(self, sample, dims=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims or self.sum_dims,
        )

    def mode(self):
        return self.mean


class InterpolatedLatents(nn.Module):
    """
    Code taken and adapted from: https://github.com/Jiawei-Yang/DeTok/tree/main
    """

    def __init__(self):
        super().__init__()
        self.to_posteriors = partial(DiagonalGaussianDistribution, channel_dim=-1)

    def interpolate_with_noise(self, z, sampling=False, noise_level=-1):
        posteriors = self.to_posteriors(z)
        z_latents = posteriors.sample() if sampling else posteriors.mean

        if self.training and self.gamma > 0.0:
            device = z_latents.device
            bsz, n_tokens, chans = z_latents.shape
            if noise_level > 0.0:
                noise_level_tensor = torch.full((bsz, 1, 1), noise_level, device=device)
            else:
                noise_level_tensor = torch.rand(bsz, 1, 1, device=device)
            noise_level_tensor = noise_level_tensor.expand(-1, n_tokens, chans)
            noise = torch.randn(bsz, n_tokens, chans, device=device) * self.gamma
            if self.use_additive_noise:
                z_latents = z_latents + noise_level_tensor * noise
            else:
                z_latents = (1 - noise_level_tensor) * z_latents + noise_level_tensor * noise

        return z_latents, posteriors
