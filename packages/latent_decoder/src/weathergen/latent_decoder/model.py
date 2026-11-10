# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A simple decoder, deliberately not a copy of the WeatherGenerator decoder, though it follows
its outline (see Model.predict_decoders): each target's coordinates are embedded into a query,
which attends to the latent tokens of the target's cell and its 8 neighbours, and a prediction
head maps the result to the target channels. Register tokens are not used."""

import torch
from torch import nn


class LatentDecoder(nn.Module):
    def __init__(
        self,
        dim_coords: int,
        num_channels: int,
        dim_latent: int = 2048,
        dim_embed: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        # the model's own decoders also LayerNorm their latent input
        self.latent_norm = nn.LayerNorm(dim_latent)
        self.latent_proj = nn.Linear(dim_latent, dim_embed)
        # which of the 9 cells a token comes from
        self.nbour_embed = nn.Parameter(torch.zeros(9, dim_embed))
        self.embed_target_coords = nn.Sequential(
            nn.Linear(dim_coords, dim_embed), nn.GELU(), nn.Linear(dim_embed, dim_embed)
        )
        self.target_readout = nn.MultiheadAttention(dim_embed, num_heads, batch_first=True)
        self.pred_head = nn.Sequential(
            nn.LayerNorm(dim_embed),
            nn.Linear(dim_embed, 4 * dim_embed),
            nn.GELU(),
            nn.Linear(4 * dim_embed, num_channels),
        )

    def forward(self, tokens, nbours, target_coords):
        """tokens [cell, dim_latent] of one state, nbours [n, 9] cell indices, target_coords
        [n, dim_coords] -> [n, num_channels]."""
        kv = self.latent_proj(self.latent_norm(tokens))[nbours] + self.nbour_embed
        q = self.embed_target_coords(target_coords)[:, None]
        h, _ = self.target_readout(q, kv, kv, need_weights=False)
        return self.pred_head(h[:, 0] + q[:, 0])
