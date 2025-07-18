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


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x2 * F.silu(x1)

class AdaLayerNormLayer(torch.nn.Module):
    """
    AdaLayerNorm for embedding auxiliary information as done in DiT (Peebles & Xie) with zero initialisation
    https://arxiv.org/pdf/2212.09748

    This module thus wraps a layer (e.g. self-attention or feedforward nn) and applies LayerNorm followed by scale and
    shift before the layer and a final scaling after the layer as well as the final residual layer.

    layer is a function that takes 2 arguments the first the latent and the second is the conditioning signal
    """

    def __init__(
        self,
        dim,
        dim_aux,
        layer,
        norm_eps: float = 1e-6,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(dim_aux, 4 * dim_aux), SwiGLU(), nn.Linear(2 * dim_aux, 3 * dim, bias=True)
        )

        self.ln = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.layer = layer

        # Initialize weights to zero for modulation and gating layers
        self.initialise_weights()

    def initialise_weights(self):
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, x_lens, **kwargs) -> torch.Tensor:
        # the -1 in torch.repeat_interleave(..) is because x_lens is designed for use with flash attention and
        # thus has a spurious 0 at the beginning to satisfy the flash attention api
        shift, scale, gate = self.adaLN_modulation(c)[torch.repeat_interleave(x_lens) - 1].chunk(
            3, dim=1
        )
        kwargs["x_lens"] = x_lens
        return (
            gate
            * self.layer(
                modulate(
                    self.ln(x),
                    shift,
                    scale,
                ),
                **kwargs,
            )
            + x
        )
