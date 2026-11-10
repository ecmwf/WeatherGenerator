# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Read stored latent states. This is the only module that knows how they are stored.

Everything else asks for the state of the init at `init_time` after `fstep` forecast steps:

    reader = LatentReader("/ec/weathergen/latents/sdnbfpve-v1-raw/2023")
    state = reader.get_state(reader.init_times[0], 1)
    state.tokens  # [12288, 2048]; state.register_tokens [64, 2048]; state.dtr the window

State fstep of an init at time t covers [t + 6h * fstep, t + 6h * (fstep + 1)). Tokens are
HEALPix level-5 cells in nested order.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import zarr
from zarr.storage import ZipStore

from weathergen.latent_decoder.geometry import NUM_HEALPIX_CELLS

TIME_STEP = np.timedelta64(6, "h")
NUM_FSTEPS = 41


@dataclass
class DTRange:
    """Time window [start, end), as in weathergen.datasets.data_reader_base."""

    start: np.datetime64
    end: np.datetime64


@dataclass
class LatentState:
    tokens: npt.NDArray  # [cell, dim] float32
    register_tokens: npt.NDArray  # [register, dim] float32
    dtr: DTRange


def state_window(init_time, fstep) -> DTRange:
    start = np.datetime64(init_time, "s") + TIME_STEP * fstep
    return DTRange(start, start + TIME_STEP)


class LatentReader:
    """Inference output as written by WeatherGenerator: one directory per run of 8 inits,
    4 rank zips of 2 inits each, and the MANIFEST.json written when the run was archived.
    `path` can be a run, a year of runs, or a directory of years."""

    def __init__(self, path):
        path = Path(path)
        manifests = sorted(
            [
                *path.glob("MANIFEST.json"),
                *path.glob("*/MANIFEST.json"),
                *path.glob("*/*/MANIFEST.json"),
            ]
        )
        if not manifests:
            raise ValueError(f"{path}: no MANIFEST.json of archived latent runs")
        self._where = {}
        for m in manifests:
            for zip_name, init_times in json.loads(m.read_text())["inits"].items():
                for i, t in enumerate(init_times):
                    # a run can hold an init twice (the loader wraps around); keep the first
                    self._where.setdefault(np.datetime64(t, "s"), (m.parent / zip_name, i))
        self.init_times = np.array(sorted(self._where), dtype="datetime64[s]")
        self._open = {}

    def _group(self, path):
        # zip handles must not be shared across forked data-loader workers
        key = (os.getpid(), path)
        if key not in self._open:
            self._open[key] = zarr.open_group(ZipStore(path, mode="r"), mode="r")
        return self._open[key]

    def get_state(self, init_time, fstep) -> LatentState:
        init_time = np.datetime64(init_time, "s")
        path, i = self._where[init_time]
        group = self._group(path)
        sample = sorted(group.group_keys(), key=int)[i]
        s = group[sample]["latent"][str(fstep)]
        tokens = s["prediction"]["data"][:][:, :, 0]
        dtr = state_window(init_time, fstep)
        if "valid_interval" in s:
            start = s["valid_interval"][:].astype("datetime64[s]")[0]
        else:
            start = dtr.start
        if tokens.shape[0] != NUM_HEALPIX_CELLS or start != dtr.start:
            raise ValueError(
                f"{path.name}:{sample} step {fstep} is not step {fstep} of {init_time}"
            )
        return LatentState(tokens, s["register_tokens"][:], dtr)
