"""A simple decoder head of our own, deliberately not a copy of the WeatherGenerator decoder.

Each target attends to the 9 latent tokens of its cell and the cell's neighbours, with a query
made from the target's own features (position, time, geoinfos), and an MLP maps the result to
the target channels. Register tokens are not used.
"""

import torch
from torch import nn


class Head(nn.Module):
    def __init__(
        self, n_features: int, n_out: int, dim_latent: int = 2048, dim: int = 256, heads: int = 4
    ):
        super().__init__()
        # the model's own decoders also LayerNorm their latent input
        self.latent_norm = nn.LayerNorm(dim_latent)
        self.latent_proj = nn.Linear(dim_latent, dim)
        self.slot = nn.Parameter(torch.zeros(9, dim))  # which of the 9 cells a token comes from
        self.query = nn.Sequential(nn.Linear(n_features, dim), nn.GELU(), nn.Linear(dim, dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, n_out)
        )

    def forward(self, tokens, nbours, features):
        """tokens [cell, dim_latent] of one state, nbours [n, 9] cell indices, features [n, f]
        -> [n, n_out]."""
        kv = self.latent_proj(self.latent_norm(tokens))[nbours] + self.slot
        q = self.query(features)[:, None]
        h, _ = self.attn(q, kv, kv, need_weights=False)
        return self.out(h[:, 0] + q[:, 0])
