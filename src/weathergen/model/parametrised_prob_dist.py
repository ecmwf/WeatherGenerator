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

    def reset_parameters(self, parameters):
        self.parameters = parameters.float()
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=self.channel_dim)
        self.sum_dims = tuple(range(1, self.mean.dim()))
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    # NOTE: old sampling code
    #def sample(self):
    #    x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
    #    return x

    def sample(self, num_samples=1):
        """
        Draw samples from the distribution.
        
        Args:
            num_samples: Number of independent samples to draw
            
        Returns:
            Tensor of shape [num_samples, *mean.shape] if num_samples > 1,
            otherwise shape [*mean.shape]
        """
        if num_samples == 1:
            x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
            return x
        else:
            # Generate multiple samples efficiently
            samples = []
            for _ in range(num_samples):
                eps = torch.randn_like(self.mean)
                samples.append(self.mean + self.std * eps)
            return torch.stack(samples, dim=0)  # [num_samples, *mean.shape]

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

    # NOTE: old interpolate_with_noise code
    #def interpolate_with_noise(self, z, batch_size=1, sampling=False, noise_level=-1):
    #    assert batch_size == 1, (
    #        "Given how we chunk in assimilate_local, dealing with batch_size greater than 1 is not "
    #        + "supported at the moment"
    #    )
    #    self.diag_gaussian.reset_parameters(self.mean_and_var(z))
    #    z_latents = self.diag_gaussian.sample() if sampling else self.diag_gaussian.mean
    #
    #    if self.training and self.gamma > 0.0:
    #        device = z_latents.device
    #        s = z_latents.shape
    #        if noise_level > 0.0:
    #            noise_level_tensor = torch.full(batch_size, noise_level, device=device)
    #        else:
    #            noise_level_tensor = torch.rand(batch_size, device=device)
    #        noise = torch.randn(s, device=device) * self.gamma
    #        if self.use_additive_noise:
    #            z_latents = z_latents + noise_level_tensor * noise
    #        else:
    #            z_latents = (1 - noise_level_tensor) * z_latents + noise_level_tensor * noise
    #
    #    return z_latents, self.diag_gaussian

    def interpolate_with_noise(
        self, 
        z, 
        batch_size=1, 
        sampling=False, 
        noise_level=-1,
        num_samples=1,
    ):
        """
        Process latents through the stochastic bottleneck.
        
        Args:
            z: Input latent tensor
            batch_size: Batch size (must be 1 for current implementation)
            sampling: Whether to sample (True) or use mean (False)
            noise_level: Training noise level (-1 for random)
            num_samples: Number of samples to draw from posterior (for inference ensembles)
            
        Returns:
            z_latents: Sampled or mean latents. Shape depends on num_samples:
                - If num_samples == 1: original shape
                - If num_samples > 1: [num_samples, *original_shape]
            posterior: The DiagonalGaussianDistribution object
        """
        assert batch_size == 1, (
            "Given how we chunk in assimilate_local, dealing with batch_size greater than 1 is not "
            + "supported at the moment"
        )
        
        # Compute posterior parameters
        self.diag_gaussian.reset_parameters(self.mean_and_var(z))
        
        # Draw sample(s)
        if sampling:
            z_latents = self.diag_gaussian.sample(num_samples=num_samples)
        else:
            z_latents = self.diag_gaussian.mean
            if num_samples > 1:
                # Return mean repeated for consistency
                z_latents = z_latents.unsqueeze(0).expand(num_samples, *z_latents.shape)

        # Training-time noise injection (only applied during training with num_samples==1)
        if self.training and self.gamma > 0.0 and num_samples == 1:
            device = z_latents.device
            s = z_latents.shape
            if noise_level > 0.0:
                noise_level_tensor = torch.full((batch_size,), noise_level, device=device)
            else:
                noise_level_tensor = torch.rand(batch_size, device=device)
            noise = torch.randn(s, device=device) * self.gamma
            if self.use_additive_noise:
                z_latents = z_latents + noise_level_tensor * noise
            else:
                z_latents = (1 - noise_level_tensor) * z_latents + noise_level_tensor * noise

        return z_latents, self.diag_gaussian

# Add this new utility class (used for parameter conversion and sampling)
class GaussianMixtureDiag(nn.Module):
    """
    Simple diagonal-covariance GMM helper: parameter transform + sampling.
    Shapes:
      raw_logits:   [N, K]
      raw_means:    [N, K, C]
      raw_log_scales: [N, K, C] (unconstrained)
    """
    def __init__(self, num_components: int, event_dim: int, min_scale: float = 1e-6):
        super().__init__()
        self.K = int(num_components)
        self.C = int(event_dim)
        self.min_scale = float(min_scale)

    @staticmethod
    def params_from_raw(raw_logits: torch.Tensor,
                        raw_means: torch.Tensor,
                        raw_log_scales: torch.Tensor,
                        min_scale: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert raw head outputs to valid GMM params.
        Returns:
          pi: [N, K], mu: [N, K, C], sigma: [N, K, C]
        """
        pi = nn.functional.softmax(raw_logits, dim=-1)                  # [N, K]
        sigma = nn.functional.softplus(raw_log_scales) + min_scale      # [N, K, C]
        return pi, raw_means, sigma

    @torch.no_grad()
    def sample(self, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Draw num_samples per item. Returns [S, N, C].
          pi: [N, K], mu: [N, K, C], sigma: [N, K, C]
        """
        device = mu.device
        dtype = mu.dtype
        N, K, C = mu.shape
        S = int(num_samples)

        # Categorical over components per item
        cat = torch.distributions.Categorical(probs=pi)
        z = cat.sample((S,))  # [S, N]

        # Gather per-sample component params
        mu_S = mu.unsqueeze(0).expand(S, -1, -1, -1)        # [S, N, K, C]
        sg_S = sigma.unsqueeze(0).expand(S, -1, -1, -1)     # [S, N, K, C]
        z_idx = z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, C)  # [S, N, 1, C]
        mu_sel = torch.gather(mu_S, dim=2, index=z_idx).squeeze(2)  # [S, N, C]
        sg_sel = torch.gather(sg_S, dim=2, index=z_idx).squeeze(2)  # [S, N, C]

        eps = torch.randn((S, N, C), device=device, dtype=dtype)
        return mu_sel + eps * sg_sel