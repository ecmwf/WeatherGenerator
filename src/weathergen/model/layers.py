# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ----------------------------------------------------------------------------
# Third-Party Attribution: facebookresearch/DiT (Scalable Diffusion Models with Transformers (DiT))
# This file incorporates code originally from the 'facebookresearch/DiT' repository,
# with adaptations.
#
# The original code is licensed under CC-BY-NC.
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Third-Party Attribution: google-deepmind/graphcast (several associated papers)
# This file incorporates code originally from the 'google-deepmind/graphcast' repository,
# with adaptations.
#
# The original code is licensed under Apache 2.0.
# Original Copyright 2024 DeepMind Technologies Limited.
# ----------------------------------------------------------------------------


import torch
import torch.nn as nn

from weathergen.model.norms import AdaLayerNorm, RMSNorm


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
        with_noise_conditioning=False,
    ):
        """Constructor"""

        super(MLP, self).__init__()

        if name is not None:
            self.name = name

        assert num_layers >= 2

        self.with_residual = with_residual
        self.with_aux = dim_aux is not None
        self.with_noise_conditioning = with_noise_conditioning
        dim_hidden = int(dim_in * hidden_factor)

        self.layers = torch.nn.ModuleList()

        norm = torch.nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm

        if pre_layer_norm:
            self.layers.append(
                norm(dim_in, eps=norm_eps)
                if dim_aux is None
                else AdaLayerNorm(dim_in, dim_aux, norm_eps=norm_eps)
            )

        if with_noise_conditioning:
            self.noise_conditioning = LinearNormConditioning(
                dim_in
            )  # TODO: chech if should pass some dtype?

        self.layers.append(torch.nn.Linear(dim_in, dim_hidden))
        self.layers.append(nonlin())
        self.layers.append(torch.nn.Dropout(p=dropout_rate))

        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(nonlin())
            self.layers.append(torch.nn.Dropout(p=dropout_rate))

        self.layers.append(torch.nn.Linear(dim_hidden, dim_out))

    # TODO: expanded args, must check dependencies (previously aux = args[-1])
    def forward(self, *args):
        x, x_in = args[0], args[0]
        if len(args) == 2:
            aux = args[1]
        elif len(args) > 2:
            aux = args[-1]
            noise_emb = args[1] if self.with_noise_conditioning else None

        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearNormConditioning):
                x = layer(x, noise_emb)  # noise embedding
            else:
                x = layer(x, aux) if (i == 0 and self.with_aux) else layer(x)

        if self.with_residual:
            if x.shape[-1] == x_in.shape[-1]:
                x = x_in + x
            else:
                assert x.shape[-1] % x_in.shape[-1] == 0
                x = x + x_in.repeat([*[1 for _ in x.shape[:-1]], x.shape[-1] // x_in.shape[-1]])

        return x


# NOTE: Inspired by GenCast/DiT.
class LinearNormConditioning(torch.nn.Module):
    """Module for norm conditioning, adapted from GenCast with additional gate parameter from DiT.

    Conditions the normalization of `inputs` by applying a linear layer to the
    `norm_conditioning` which produces the scale and offset for each channel.
    """

    def __init__(self, latent_space_dim: int, noise_emb_dim: int = 512, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype

        self.conditional_linear_layer = torch.nn.Linear(
            in_features=noise_emb_dim,
            out_features=3 * latent_space_dim,
        )
        # Optional: initialize weights similar to TruncatedNormal(stddev=1e-8)
        torch.nn.init.normal_(self.conditional_linear_layer.weight, std=1e-8)
        torch.nn.init.zeros_(self.conditional_linear_layer.bias)

    def forward(self, inputs, noise_emb):
        conditional_scale_offset = self.conditional_linear_layer(noise_emb.to(self.dtype))
        scale_minus_one, offset, gate = torch.chunk(conditional_scale_offset, 3, dim=-1)
        scale = scale_minus_one + 1.0

        # Reshape scale and offset for broadcasting if needed
        while scale.dim() < inputs.dim():
            scale = scale.unsqueeze(1)
            offset = offset.unsqueeze(1)
        return (inputs * scale + offset).to(
            self.dtype
        ), gate  # TODO: check if to(self.dtype) needed here
