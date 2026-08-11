# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for RNG seeding / reproducibility in TrainerBase."""

import random

import numpy as np
import torch
from omegaconf import OmegaConf

from weathergen.train.trainer_base import TrainerBase


def _draw_all() -> tuple[float, float, float]:
    """Draw one sample from each of the torch, NumPy, and Python global RNGs."""
    return (
        torch.rand(1).item(),
        float(np.random.random()),
        random.random(),
    )


def _make_cf(rng_seed=None) -> OmegaConf:
    cf = OmegaConf.create({"data_loading": {} if rng_seed is None else {"rng_seed": rng_seed}})
    OmegaConf.set_struct(cf, False)
    return cf


def test_init_seeds_is_reproducible():
    """Seeding with the same value yields identical draws from every global RNG."""
    TrainerBase.init_seeds(_make_cf(42))
    first = _draw_all()
    weight_first = torch.nn.Linear(8, 8).weight.detach().clone()

    TrainerBase.init_seeds(_make_cf(42))
    second = _draw_all()
    weight_second = torch.nn.Linear(8, 8).weight.detach().clone()

    assert first == second
    # weight initialization (the reason torch is seeded) must be reproducible
    assert torch.equal(weight_first, weight_second)


def test_init_seeds_differs_for_different_seeds():
    """Different seeds diversify the RNG streams (guards against a no-op seed)."""
    TrainerBase.init_seeds(_make_cf(1))
    first = _draw_all()
    TrainerBase.init_seeds(_make_cf(2))
    second = _draw_all()

    assert first != second


def test_init_seeds_respects_config_seed():
    """A seed supplied in the config is kept (not overwritten by the time fallback)."""
    cf = TrainerBase.init_seeds(_make_cf(777))
    assert cf.data_loading.rng_seed == 777


def test_init_seeds_fills_missing_seed():
    """A missing seed is filled with a positive fallback so runs are seeded by default."""
    cf = TrainerBase.init_seeds(_make_cf())
    assert isinstance(cf.data_loading.rng_seed, int)
    assert cf.data_loading.rng_seed >= 1


def test_init_seeds_clamps_nonpositive_seed():
    """Seed 0 / negative seeds are clamped to >= 1 (0 breaks per-rank seed derivation)."""
    for seed in (0, -5):
        cf = TrainerBase.init_seeds(_make_cf(seed))
        assert cf.data_loading.rng_seed == 1


def test_init_ddp_keeps_resolved_seed():
    """init_ddp leaves the seed resolved by init_seeds alone (single-process case)."""
    if not torch.distributed.is_available():
        return

    cf = TrainerBase.init_seeds(_make_cf(2024))
    cf = TrainerBase.init_ddp(cf)

    assert cf.data_loading.rng_seed == 2024
    assert cf.world_size == 1
    assert cf.rank == 0
    assert not cf.with_ddp
