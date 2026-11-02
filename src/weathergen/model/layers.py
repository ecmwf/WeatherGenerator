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

from weathergen.model.norms import AdaLayerNorm, AdaLNZero, RMSNorm, SwiGLU


class RegisterTokens(torch.nn.Module):
    """Learnable register tokens, shape (1, R, D).

    Kept in their own module (rather than as a bare Parameter on the owning engine) so that
    checkpoint loading can (re)initialize them via reset_parameters() without treating the whole
    owning engine as a new module. Initialized like the encoder's register tokens
    (EncoderModule.q_aux): uniform in [0, 1/D).
    """

    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.dim = dim
        self.tokens = torch.nn.Parameter(torch.empty(1, num_registers, dim))
        self.reset_parameters()

    @property
    def num_registers(self) -> int:
        return self.tokens.shape[1]

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.uniform_(self.tokens, 0.0, 1.0 / self.dim)

    def prepend(
        self,
        tokens: torch.Tensor,
        coords: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Prepend the registers to tokens (B, N, D) -> (B, R+N, D).

        coords (B or 1, N, 2) get zero RoPE coordinates at the register slots. conditioning, if
        given, has to be token-wise aligned with tokens and is zero-padded at the register slots.
        """
        num_registers = self.num_registers
        registers = self.tokens.to(dtype=tokens.dtype).expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([registers, tokens], dim=1)
        if coords is not None:
            pad = coords.new_zeros(coords.shape[0], num_registers, coords.shape[2])
            coords = torch.cat([pad, coords], dim=1)
        if conditioning is not None:
            pad = conditioning.new_zeros(
                conditioning.shape[0], num_registers, conditioning.shape[2]
            )
            conditioning = torch.cat([pad, conditioning], dim=1)
        return tokens, coords, conditioning

    def strip(self, tokens: torch.Tensor) -> torch.Tensor:
        """Drop the registers again: (B, R+N, D) -> (B, N, D)."""
        return tokens[:, self.num_registers :]


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


class KwargLayerNorm(torch.nn.Module):
    """LayerNorm that discards block-call kwargs (coords, emb, ada_ln_aux, ...).

    Lets a plain normalization layer share the uniform call signature used for MLP/attention
    blocks in a checkpointed ModuleList, instead of being special-cased by isinstance checks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = torch.nn.LayerNorm(*args, **kwargs)

    def forward(self, x, **kwargs):
        return self.norm(x)


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
        mlp_type="mlp",
        name: str | None = None,
        is_dit=False,
        dit_is_cond=False,
    ):
        """Constructor"""

        super(MLP, self).__init__()

        if name is not None:
            self.name = name

        assert num_layers >= 2

        self.with_residual = with_residual
        self.with_aux = dim_aux is not None
        self.is_dit = is_dit
        self.dit_is_cond = dit_is_cond
        self.mlp_type = mlp_type.lower()
        dim_hidden = int(dim_in * hidden_factor)

        if self.mlp_type not in {"mlp", "swiglu"}:
            raise ValueError(f"Unsupported mlp_type: {mlp_type}")

        if self.mlp_type == "swiglu":
            # Align with the standard LLaMA-style SwiGLU hidden-width rule.
            dim_hidden = max(1, int(2 * dim_hidden / 3))

        self.layers = torch.nn.ModuleList()

        norm = torch.nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm

        if is_dit:
            if dit_is_cond:
                assert dim_aux is not None, "For DIT, need to provide dim_aux for ada layer norm"
            assert with_residual, "DIT attention should always have residual connection"
            self.lnorm = (
                AdaLNZero(dim_in, dim_aux, norm_eps=norm_eps)
                if dim_aux is not None
                else norm(dim_in, eps=norm_eps)
            )
            self.noise_conditioning = LinearNormConditioning(dim_in)
        elif dim_aux is not None:
            self.layers.append(AdaLayerNorm(dim_in, dim_aux, norm_eps=norm_eps))
        else:
            self.layers.append(norm(dim_in, eps=norm_eps))

        if self.mlp_type == "swiglu":
            self.layers.append(torch.nn.Linear(dim_in, 2 * dim_hidden))
            self.layers.append(SwiGLU())
            self.layers.append(torch.nn.Dropout(p=dropout_rate))
            for _ in range(num_layers - 2):
                self.layers.append(torch.nn.Linear(dim_hidden, 2 * dim_hidden))
                self.layers.append(SwiGLU())
                self.layers.append(torch.nn.Dropout(p=dropout_rate))
        else:
            self.layers.append(torch.nn.Linear(dim_in, dim_hidden))
            self.layers.append(nonlin())
            self.layers.append(torch.nn.Dropout(p=dropout_rate))

            for _ in range(num_layers - 2):
                self.layers.append(torch.nn.Linear(dim_hidden, dim_hidden))
                self.layers.append(nonlin())
                self.layers.append(torch.nn.Dropout(p=dropout_rate))

        self.layers.append(torch.nn.Linear(dim_hidden, dim_out))

    def _apply_conditioning(self, x, *, emb, ada_ln_aux):
        """Normalize x and, for DIT blocks, apply the AdaLN/noise gate.

        Returns (x, gate); gate is the int 1 (no-op) unless this is a DIT block.
        """
        if not self.is_dit:
            if self.with_aux and ada_ln_aux is None:
                raise ValueError("Auxiliary input required but not provided")
            return x, 1

        if self.dit_is_cond:
            assert ada_ln_aux is not None, "Need auxiliary input for conditional DIT"
            x, cond_gate = self.lnorm(x, ada_ln_aux)
        else:
            x = self.lnorm(x)
            cond_gate = 1
        assert emb is not None, "Need noise embedding for noise conditioning in DIT"
        x, noise_gate = self.noise_conditioning(x, emb)
        return x, cond_gate * noise_gate

    def forward(self, x, *, emb=None, ada_ln_aux=None, **kwargs):
        # keyword-only, with **kwargs discarded: lets every block in a checkpointed
        # ModuleList (MLP, attention, ...) be called with the same keyword arguments,
        # regardless of which ones a given block actually uses (e.g. coords, x_lens).
        x_in = x
        x, gate = self._apply_conditioning(x, emb=emb, ada_ln_aux=ada_ln_aux)

        for layer in self.layers:
            x = layer(x, ada_ln_aux) if isinstance(layer, AdaLayerNorm) else layer(x)

        if self.with_residual:
            # gate is 1 (no-op) unless this is a DIT block, so this is unconditional.
            x = x * gate
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

        # Reshape scale, offset and gate for broadcasting if needed
        while scale.dim() < inputs.dim():
            scale = scale.unsqueeze(1)
            offset = offset.unsqueeze(1)
            gate = gate.unsqueeze(1)
        return (inputs * scale + offset).to(
            self.dtype
        ), gate  # TODO: check if to(self.dtype) needed here
