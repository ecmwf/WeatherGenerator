# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import torch
import torch.nn as nn

from weathergen.model.norms import SaturateEncodings


class DiagonalGaussianDistribution:
    """
    Used to represent a learned Gaussian Distribution as typical in a VAE
    Code taken and adapted from: https://github.com/Jiawei-Yang/DeTok/tree/main
    """

    def __init__(self, deterministic=False, channel_dim=1):
        self.deterministic = deterministic
        self.channel_dim = channel_dim
        self.parameters = None
        self.mean = None
        self.logvar = None
        self.sum_dims = None
        self.std = None
        self.var = None

    def reset_parameters(self, parameters):
        self.parameters = parameters.float()
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=self.channel_dim)
        self.sum_dims = tuple(range(1, self.mean.dim()))
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
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


class LatentInterpolator(nn.Module):
    """
    Code taken and adapted from: https://github.com/Jiawei-Yang/DeTok/tree/main
    """

    def __init__(
        self,
        gamma,
        dim,
        use_additive_noise=False,
        deterministic=False,
        saturate_encodings=None,
    ):
        super().__init__()

        assert deterministic or saturate_encodings is None, (
            "Cannot use saturate_encodings without deterministic"
        )
        self.gamma = gamma
        self.saturate_encodings = saturate_encodings
        self.use_additive_noise = use_additive_noise
        self.diag_gaussian = DiagonalGaussianDistribution(
            deterministic=deterministic, channel_dim=-1
        )
        self.mean_and_var = nn.Sequential(
            nn.Linear(dim, 2 * dim, bias=False),
            SaturateEncodings(saturate_encodings)
            if saturate_encodings is not None
            else nn.Identity(),
        )

    def interpolate_with_noise(self, z, batch_size=1, sampling=False, noise_level=-1):
        assert batch_size == 1, (
            "Given how we chunk in assimilate_local, dealing with batch_size greater than 1 is not "
            + "supported at the moment"
        )
        self.diag_gaussian.reset_parameters(self.mean_and_var(z))
        z_latents = self.diag_gaussian.sample() if sampling else self.diag_gaussian.mean

        if self.training and self.gamma > 0.0:
            device = z_latents.device
            s = z_latents.shape
            if noise_level > 0.0:
                noise_level_tensor = torch.full(batch_size, noise_level, device=device)
            else:
                noise_level_tensor = torch.rand(batch_size, device=device)
            noise = torch.randn(s, device=device) * self.gamma
            if self.use_additive_noise:
                z_latents = z_latents + noise_level_tensor * noise
            else:
                z_latents = (1 - noise_level_tensor) * z_latents + noise_level_tensor * noise

        return z_latents, self.diag_gaussian
