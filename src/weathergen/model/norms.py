# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
import torch.nn as nn
import torch.nn.functional as F


# from https://github.com/meta-llama/llama/blob/main/llama/model.py
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability.
            Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class AdaLayerNorm(torch.nn.Module):
    """
    AdaLayerNorm for embedding auxiliary information
    """

    def __init__(
        self, dim_embed_x, dim_aux, norm_elementwise_affine: bool = False, norm_eps: float = 1e-5
    ):
        super().__init__()

        # simple 2-layer MLP for embedding auxiliary information
        self.embed_aux = torch.nn.ModuleList()
        self.embed_aux.append(torch.nn.Linear(dim_aux, 4 * dim_aux))
        self.embed_aux.append(torch.nn.SiLU())
        self.embed_aux.append(torch.nn.Linear(4 * dim_aux, 2 * dim_embed_x))

        self.norm = torch.nn.LayerNorm(dim_embed_x, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.embed_aux:
            aux = block(aux)
        scale, shift = aux.split(aux.shape[-1] // 2, dim=-1)

        x = self.norm(x) * (1 + scale) + shift

        return x
    
class AdaLayerNormFinal(torch.nn.Module):
    """
    AdaLayerNorm from DiT for the final output gate only, i.e. only scale
    """
    
    def __init__(
        self, dim_embed_x, dim_aux, norm_elementwise_affine: bool = False, norm_eps: float = 1e-5
    ):
        super().__init__()

        # simple 2-layer MLP for embedding auxiliary information
        self.embed_aux = torch.nn.ModuleList()
        self.embed_aux.append(torch.nn.Linear(dim_aux, 4 * dim_aux))
        self.embed_aux.append(torch.nn.SiLU())
        self.embed_aux.append(torch.nn.Linear(4 * dim_aux, dim_embed_x))

        self.norm = torch.nn.LayerNorm(dim_embed_x, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.embed_aux:
            aux = block(aux)
        scale = aux

        x = self.norm(x) * (1 + scale)

        return x



class SaturateEncodings(nn.Module):
    """A common alternative to a KL regularisation prevent outliers in the latent space when
    learning an auto-encoder for latent generative model, an example value for the scale factor is 5
    """

    def __init__(self, scale_factor):
        super().__init__()

        self.scale_factor_squared = scale_factor**2

    def forward(self, x):
        return x / torch.sqrt(1 + (x**2 / self.scale_factor_squared))
