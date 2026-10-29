"""Pair latent states with the observations in their time windows."""

import numpy as np
import torch

from weathergen.latent_decoder import geometry
from weathergen.latent_decoder.latents import STEP

# geoinfos are used as they are, except station height, which is in metres
GEO_SCALE = {"stalt": 1e-3}
FORECAST_STEPS = range(1, 41)


def pairs(bases, first, last, steps=FORECAST_STEPS):
    """(base, step) for every init with first <= base <= last. Step 0 is left out by default:
    its window is the source window, whose observations were model inputs."""
    b = bases[(bases >= np.datetime64(first, "s")) & (bases <= np.datetime64(last, "s"))]
    return [(base, k) for base in b for k in steps]


def features(obs, geoinfos, k, window, cells):
    """[n, f]: position (6), time within the window (1), lead (1), geoinfos."""
    dt = (obs["times"] - np.datetime64(window[0], "ns")) / STEP.astype("timedelta64[ns]")
    geo = obs["geo"] * np.array([GEO_SCALE.get(g, 1.0) for g in geoinfos], dtype=np.float32)
    return np.concatenate(
        [
            geometry.position_features(obs["lat"], obs["lon"], cells),
            dt[:, None].astype(np.float32),
            np.full((len(cells), 1), k / 40, np.float32),
            np.nan_to_num(geo),
        ],
        axis=1,
    ).astype(np.float32)


class StateDataset(torch.utils.data.Dataset):
    """One item per (base, step): the state's tokens and every report in its window."""

    def __init__(self, latents, obs, pairs, max_obs=None, seed=0):
        self.latents, self.obs, self.pairs = latents, obs, pairs
        self.max_obs, self.seed = max_obs, seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        base, k = self.pairs[i]
        state = self.latents.state(base, k)
        obs = self.obs.window(*state.window)
        if self.max_obs and len(obs["lat"]) > self.max_obs:
            pick = np.random.default_rng((self.seed, i)).choice(
                len(obs["lat"]), self.max_obs, replace=False
            )
            obs = {name: v[pick] for name, v in obs.items()}
        cells = geometry.cell_of(obs["lat"], obs["lon"])
        return {
            "base": str(base),
            "step": k,
            # float16 halves what workers pass around; it is well below the run-to-run noise
            "tokens": torch.from_numpy(state.tokens.astype(np.float16)),
            "nbours": torch.from_numpy(geometry.neighbours()[cells]),
            "features": torch.from_numpy(features(obs, self.obs.geoinfos, k, state.window, cells)),
            "values": torch.from_numpy(obs["values"].astype(np.float32)),
            "times": obs["times"].astype("datetime64[s]").astype(np.int64),
            "lat": obs["lat"],
            "lon": obs["lon"],
        }
