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

from weathergen.model.norms import AdaLayerNorm, RMSNorm


class LayerScale(nn.Module):
    """Per-channel learnable scaling, as in CaiT (Touvron et al., 2021).

    Applies a learned per-channel scaling factor to the input. When used before
    residual connections, it allows the network to gradually incorporate new
    layer contributions during training.

    Args:
        dim: Number of channels/features to scale.
        init_value: Initial value for the scaling factors. Use 1e-5 for LayerScale
            or 0.0 for ReZero initialization.
    """

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class StochasticDepth(nn.Module):
    """Stochastic Depth / DropPath regularization (Huang et al., 2016).

    Randomly drops entire residual paths during training. This acts as a form
    of regularization and enables training deeper networks.

    Args:
        drop_prob: Probability of dropping the path. 0.0 means no dropping.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample dropout (batch dimension)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob  # Scale to maintain expected value


class NamedLinear(torch.nn.Module):
    def __init__(self, name: str | None = None, **kwargs):
        super(NamedLinear, self).__init__()
        self.linear = nn.Linear(**kwargs)
        if name is not None:
            self.name = name

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


class MLP(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_layers=2,
        hidden_factor=2,
        pre_layer_norm=True,
        dropout_rate=0.0,
        nonlin=torch.nn.GELU,
        with_residual=False,
        norm_type="LayerNorm",
        dim_aux=None,
        norm_eps=1e-5,
        name: str | None = None,
        layer_scale_init: float | None = None,
        stochastic_depth_rate: float = 0.0,
    ):
        """Constructor

        Args:
            layer_scale_init: If not None, applies LayerScale with this init value.
                Use 1e-5 for LayerScale, 0.0 for ReZero.
            stochastic_depth_rate: Probability of dropping this block during training.
        """

        super(MLP, self).__init__()

        if name is not None:
            self.name = name

        assert num_layers >= 2

        self.with_residual = with_residual
        self.with_aux = dim_aux is not None
        dim_hidden = int(dim_in * hidden_factor)

        self.layers = torch.nn.ModuleList()

        norm = torch.nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm

        if pre_layer_norm:
            self.layers.append(
                norm(dim_in, eps=norm_eps)
                if dim_aux is None
                else AdaLayerNorm(dim_in, dim_aux, norm_eps=norm_eps)
            )

        self.layers.append(torch.nn.Linear(dim_in, dim_hidden))
        self.layers.append(nonlin())
        self.layers.append(torch.nn.Dropout(p=dropout_rate))

        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(nonlin())
            self.layers.append(torch.nn.Dropout(p=dropout_rate))

        self.layers.append(torch.nn.Linear(dim_hidden, dim_out))

        # LayerScale: per-channel learned scaling before residual
        self.layer_scale = (
            LayerScale(dim_out, layer_scale_init) if layer_scale_init is not None else None
        )

        # Stochastic Depth: randomly drop residual path during training
        self.drop_path = (
            StochasticDepth(stochastic_depth_rate) if stochastic_depth_rate > 0.0 else None
        )

    def forward(self, *args):
        x, x_in, aux = args[0], args[0], args[-1]

        for i, layer in enumerate(self.layers):
            x = layer(x, aux) if (i == 0 and self.with_aux) else layer(x)

        # Apply LayerScale before residual
        if self.layer_scale is not None:
            x = self.layer_scale(x)

        # Apply Stochastic Depth before residual
        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.with_residual:
            if x.shape[-1] == x_in.shape[-1]:
                x = x_in + x
            else:
                assert x.shape[-1] % x_in.shape[-1] == 0
                x = x + x_in.repeat([*[1 for _ in x.shape[:-1]], x.shape[-1] // x_in.shape[-1]])

        return x
