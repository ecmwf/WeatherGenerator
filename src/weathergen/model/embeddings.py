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
from torch.utils.checkpoint import checkpoint

from weathergen.model.attention import MultiSelfAttentionHead
from weathergen.model.layers import MLP

# from weathergen.model.mlp import MLP
from weathergen.model.norms import RMSNorm
from weathergen.model.positional_encoding import positional_encoding_harmonic


class StreamEmbedTransformer(torch.nn.Module):
    def __init__(
        self,
        mode,
        num_tokens,
        token_size,
        num_channels,
        dim_embed,
        dim_out,
        num_blocks,
        num_heads,
        dropout_rate=0.0,
        norm_type="LayerNorm",
        embed_size_centroids=64,
        unembed_mode="full",
        embed_gradient_checkpoint_mode=True,
        stream_name="stream_embed",
    ):
        """Constructor

        unembed_mode : { 'full' , 'block'}
          full : monolithic (and correspondingly large) unembedding network that maps from
                 (num_tokens x dim_embed) to dim_out, allowing for mixing between channels/columns
          block : per-channel/column unembedding network
                (which is hence a block-sparse form of full)
        """

        super(StreamEmbedTransformer, self).__init__()

        self.name = f"StreamEmbedder_{stream_name}"

        self.num_tokens = num_tokens
        self.token_size = token_size
        self.num_channels = num_channels
        self.dim_in = token_size if mode == "channels" else num_channels
        self.dim_embed = dim_embed
        self.dim_out = dim_out
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.embed_size_centroids = embed_size_centroids
        self.unembed_mode = unembed_mode
        self.embed_gradient_checkpoint_mode = embed_gradient_checkpoint_mode

        norm = torch.nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_blocks):
            self.layers.append(
                MultiSelfAttentionHead(
                    self.dim_embed,
                    self.num_heads,
                    dropout_rate=dropout_rate,
                    with_qk_lnorm=True,
                    with_flash=True,
                )
            )
            self.layers.append(
                MLP(
                    self.dim_embed,
                    self.dim_embed,
                    hidden_factor=2,
                    dropout_rate=dropout_rate,
                    with_residual=True,
                )
            )

        if mode == "channels":
            self.embed = torch.nn.Linear(self.dim_in, self.dim_embed)

            if self.unembed_mode == "full":
                self.ln_final = norm(num_channels * self.dim_embed, eps=1e-03)
                self.unembed = torch.nn.Linear(
                    num_channels * self.dim_embed,
                    self.num_tokens * self.dim_out - embed_size_centroids,
                )

            elif self.unembed_mode == "block":
                # modify embed_size_centroids to ensure no additional padding is needed
                rem = (self.num_tokens * self.dim_out - embed_size_centroids) % num_channels
                embed_size_centroids += rem
                dim_out = (self.num_tokens * self.dim_out - embed_size_centroids) // num_channels
                self.unembed = torch.nn.ModuleList(
                    [torch.nn.Linear(dim_embed, dim_out) for _ in range(num_channels)]
                    # [
                    #     torch.nn.Sequential(
                    #         torch.nn.Linear(dim_embed, max(dim_embed//2,4*dim_out)),
                    #         torch.nn.GELU(),
                    #         torch.nn.Linear(max(dim_embed//2,4*dim_out), dim_out)
                    #     ) for _ in range(num_channels)
                    # ]
                )
                self.ln_final = torch.nn.ModuleList(
                    [norm(dim_embed, eps=1e-6) for _ in range(num_channels)]
                )

            else:
                assert False

            self.forward = self.forward_channels

        elif mode == "columns":
            assert embed_size_centroids == 0
            self.embed = torch.nn.Linear(self.dim_in, self.dim_embed)

            assert self.unembed_mode == "block"  # only supported mode at the moment
            # padding needed if the unembedded columns cannot be concatenated to dim_out (e.g GPSRO)
            self.pad = self.dim_out % token_size
            self.out_pad = torch.nn.Parameter(torch.zeros(self.pad), requires_grad=False)
            self.unembed = torch.nn.Linear(
                self.dim_embed,
                self.num_tokens * ((self.dim_out - embed_size_centroids) // token_size),
            )
            self.ln_final = norm(dim_out, eps=1e-6)
            self.forward = self.forward_columns

            # TODO: factorization when sqrt is not int
            dim1 = int(np.sqrt(dim_out))
            assert dim1 * dim1 == dim_out
            self.unembed1 = torch.nn.Linear(self.dim_embed, dim1)
            self.unembed_nonlin = torch.nn.GELU()
            self.unembed2 = torch.nn.Linear(self.token_size, dim1)

        else:
            assert False

        self.dropout_final = torch.nn.Dropout(0.1)
        self.embed_centroids = torch.nn.Linear(5, embed_size_centroids)

    def forward_channels(self, x_in, centroids):
        peh = positional_encoding_harmonic

        if self.embed_gradient_checkpoint_mode:
            # embed provided input data
            x = peh(checkpoint(self.embed, x_in.transpose(-2, -1), use_reentrant=False))

            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)

            # read out
            if self.unembed_mode == "full":
                out = checkpoint(self.unembed, self.ln_final(x.flatten(-2, -1)), use_reentrant=False)
            elif self.unembed_mode == "block":
                out = [
                    checkpoint(ue, ln(x[:, i]), use_reentrant=False)
                    for i, (ue, ln) in enumerate(zip(self.unembed, self.ln_final, strict=True))
                ]
                out = torch.stack(out, dim=1).flatten(-2, -1)
            else:
                assert False

        else:
            # embed provided input data
            x = peh(self.embed(x_in.transpose(-2, -1)))

            for layer in self.layers:
                x = layer(x)

            # read out
            if self.unembed_mode == "full":
                out = self.unembed(self.ln_final(x.flatten(-2, -1)))
            elif self.unembed_mode == "block":
                out = [
                    ue(ln(x[:, i]))
                    for i, (ue, ln) in enumerate(zip(self.unembed, self.ln_final, strict=True))
                ]
                out = torch.stack(out, dim=1).flatten(-2, -1)
            else:
                assert False

        # append centroids
        if self.embed_size_centroids > 0:
            out = torch.cat([out, self.embed_centroids(centroids)], -1)
        # if self.embed_size_centroids==0 and self.dim_out is not divisible by #channels with
        # unembed_mode block then we need to pad to have the expected output shape
        if out.shape[-1] < self.dim_out:
            out = torch.nn.functional.pad(out, [0, self.dim_out - out.shape[-1]], value=0.0)
        # final reshape
        out = self.dropout_final(out.reshape(-1, self.num_tokens, self.dim_out))

        return out

    def forward_columns(self, x_in, centroids):
        # embed provided input data
        x = positional_encoding_harmonic(checkpoint(self.embed, x_in, use_reentrant=False))

        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)

        out = checkpoint(self.unembed1, x, use_reentrant=False)
        out = self.unembed_nonlin(out)
        out = checkpoint(self.unembed2, out.transpose(-2, -1), use_reentrant=False)
        out = out.flatten(-2, -1).unsqueeze(1)

        # final normalize and dropout
        out = self.dropout_final(self.ln_final(out))

        return out.to(torch.float16)


class StreamEmbedLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out, stream_name="stream_embed"):
        """Constructor"""

        super(StreamEmbedLinear, self).__init__()

        self.name = f"StreamEmbedder_{stream_name}"
        self.layer = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x):
        # x = checkpoint( self.layer, x.flatten( -2, -1), use_reentrant=True)
        x = self.layer(x.flatten(-2, -1))

        return x
