# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""pe_global with and without pe_global_latlon. Runs on CPU."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

# weathergen.model imports flash_attn thus skip where it is not installed
pytest.importorskip("flash_attn", reason="flash_attn required by weathergen.model")

from weathergen.model.model import ModelParams  # noqa: E402

HEALPIX_LEVEL = 5
DIM_EMBED = 2048
NUM_CELLS = 12 * 4**HEALPIX_LEVEL
NUM_QUERIES = 2


def _model_params(**kwargs):
    cfg = {
        "healpix_level": HEALPIX_LEVEL,
        "ae_local_dim_embed": 64,
        "ae_global_dim_embed": DIM_EMBED,
        "ae_local_num_queries": NUM_QUERIES,
        "attention_dtype": "fp32",
        "rope_2D": False,
    }
    cfg.update(kwargs)
    cf = OmegaConf.create(cfg)
    return ModelParams(cf).create(cf)


def _old_pe_global(dtype):
    # pe_global as built before pe_global_latlon was added
    pe = torch.zeros(NUM_CELLS, NUM_QUERIES, DIM_EMBED, dtype=dtype)
    xs = 2.0 * np.pi * torch.arange(0, DIM_EMBED, 2) / DIM_EMBED
    queries = torch.outer(8 * torch.arange(NUM_QUERIES), xs)
    cells = torch.outer(torch.arange(NUM_CELLS), xs)
    pe[..., 0::2] = 0.5 * torch.sin(queries)
    pe[..., 0::2] += torch.sin(cells).unsqueeze(1).repeat((1, NUM_QUERIES, 1))
    pe[..., 1::2] = 0.5 * torch.cos(queries)
    pe[..., 1::2] += torch.cos(cells).unsqueeze(1).repeat((1, NUM_QUERIES, 1))
    return pe


def _min_pair_dist(codes, chunk=2048):
    sq = (codes * codes).sum(1)
    dmin = float("inf")
    for i in range(0, codes.shape[0], chunk):
        d2 = sq[i : i + chunk, None] + sq[None, :] - 2 * codes[i : i + chunk] @ codes.T
        idx = torch.arange(i, min(i + chunk, codes.shape[0]))
        d2[idx - i, idx] = float("inf")
        dmin = min(dmin, d2.min().clamp(min=0).sqrt().item())
    return dmin


@pytest.mark.parametrize("dtype_name,dtype", [("fp32", torch.float32), ("bf16", torch.bfloat16)])
@pytest.mark.parametrize("latlon", [None, False])
def test_pe_global_default_unchanged(dtype_name, dtype, latlon):
    """Option off or missing gives old pe_global bit for bit thus old checkpoints still work."""
    kwargs = {"attention_dtype": dtype_name}
    if latlon is not None:
        kwargs["pe_global_latlon"] = latlon
    pe = _model_params(**kwargs).pe_global.data
    assert torch.equal(pe, _old_pe_global(dtype))


def test_pe_global_latlon_distinct_codes():
    """With pe_global_latlon every healpix cell gets its own code."""
    codes = _model_params(pe_global_latlon=True).pe_global.data[:, 0, :]
    # old code repeats every DIM_EMBED / 2 cells thus its closest pair is about 0
    assert _min_pair_dist(codes) > 1.0
