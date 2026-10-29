"""Read stored latent states. This is the only module that knows how they are stored.

Everything else asks for the state of init `base` at step `k`:

    lat = open_latents("/ec/weathergen/latents/sdnbfpve-v1-raw/2023")
    s = lat.state(lat.bases[0], 1)    # s.tokens [12288, 2048], s.registers [64, 2048], s.window

State k of an init with base time b covers [b + 6h*k, b + 6h*(k+1)). Tokens are HEALPix level-5
cells in nested order. A new storage format needs a new class with `bases`, `steps` and
`state`, and a line in `open_latents`.
"""

import json
import os
import pathlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import zarr
from zarr.storage import ZipStore

STEP = np.timedelta64(6, "h")
N_CELLS = 12288


@dataclass
class State:
    tokens: npt.NDArray  # [cell, dim] float32
    registers: npt.NDArray  # [register, dim] float32
    window: tuple[np.datetime64, np.datetime64]  # [start, end)


def open_latents(path):
    """Open a directory of stored latents, whichever of the known formats it holds."""
    p = pathlib.Path(path)
    if (
        (p / "MANIFEST.json").exists()
        or next(p.glob("*/MANIFEST.json"), None)
        or next(p.glob("*/*/MANIFEST.json"), None)
    ):
        return RawRuns(p)
    if next(p.glob("[0-9][0-9][0-9][0-9].zarr"), None):
        return YearStores(p)
    raise ValueError(f"{p}: no known latent format (raw runs with MANIFEST.json, or YYYY.zarr)")


def _window(base, k, valid_interval=None):
    if valid_interval is None:
        return base + STEP * k, base + STEP * (k + 1)
    return tuple(np.asarray(valid_interval).astype("datetime64[s]"))


class RawRuns:
    """Inference output as written by WeatherGenerator: one directory per run of 8 inits,
    4 rank zips of 2 inits each, and the MANIFEST.json written when the run was archived."""

    def __init__(self, root: pathlib.Path):
        manifests = sorted(
            [
                *root.glob("MANIFEST.json"),
                *root.glob("*/MANIFEST.json"),
                *root.glob("*/*/MANIFEST.json"),
            ]
        )
        self._where = {}
        for m in manifests:
            for zname, bases in json.loads(m.read_text())["inits"].items():
                for pos, b in enumerate(bases):
                    # a run can hold an init twice (the loader wraps around); keep the first
                    self._where.setdefault(np.datetime64(b, "s"), (m.parent / zname, pos))
        self.bases = np.array(sorted(self._where), dtype="datetime64[s]")
        self.steps = 41
        self._open = {}

    def _group(self, path):
        # zip handles must not be shared across forked data-loader workers
        key = (os.getpid(), path)
        if key not in self._open:
            self._open[key] = zarr.open_group(ZipStore(path, mode="r"), mode="r")
        return self._open[key]

    def state(self, base, k) -> State:
        base = np.datetime64(base, "s")
        path, pos = self._where[base]
        g = self._group(path)
        sample = sorted(g.group_keys(), key=int)[pos]
        s = g[sample]["latent"][str(k)]
        tokens = s["prediction"]["data"][:][:, :, 0]
        vi = s["valid_interval"][:] if "valid_interval" in s else None
        window = _window(base, k, vi)
        if tokens.shape[0] != N_CELLS or window[0] != base + STEP * k:
            raise ValueError(f"{path.name}:{sample} step {k} is not state {k} of {base}")
        return State(tokens, s["register_tokens"][:], window)


class YearStores:
    """One zarr store per year, [init, step, cell, dim], as made by the fp16 converter."""

    def __init__(self, root: pathlib.Path):
        self._stores = {}
        bases = []
        for path in sorted(root.glob("[0-9][0-9][0-9][0-9].zarr")):
            g = zarr.open_group(path, mode="r")
            b = g["base_time"][:].astype("datetime64[s]")
            filled = g["filled"][:]
            self._stores[int(path.stem)] = (g, b)
            bases.append(b[filled])
        self.bases = np.sort(np.concatenate(bases))
        self.steps = 41

    def state(self, base, k) -> State:
        base = np.datetime64(base, "s")
        g, b = self._stores[base.astype(object).year]
        i = int(np.searchsorted(b, base))
        if b[i] != base or not g["filled"][i]:
            raise KeyError(f"{base} is not in the store")
        window = (
            g["valid_start"][i, k].astype("datetime64[s]"),
            g["valid_end"][i, k].astype("datetime64[s]"),
        )
        return State(
            g["tokens"][i, k].astype(np.float32), g["registers"][i, k].astype(np.float32), window
        )
