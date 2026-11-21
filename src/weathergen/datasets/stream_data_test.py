# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for reproducibility of spoofed stream data."""

import numpy as np
import pytest
import torch

from weathergen.datasets.stream_data import StreamData, spoof
from weathergen.train.utils import TRAIN, VAL


def _spoof(rng: np.random.Generator):
    return spoof(
        healpix_level=2,
        datetime=np.datetime64("2020-01-01T00:00:00"),
        geoinfo_size=3,
        num_channels=4,
        rng=rng,
    )


def test_spoof_is_reproducible_for_equal_rng():
    """The same seeded rng yields identical spoofed tokens (the selected cells)."""
    first = _spoof(np.random.default_rng(123))
    second = _spoof(np.random.default_rng(123))

    assert np.array_equal(first.coords, second.coords)


def test_spoof_uses_passed_rng_not_global():
    """spoof draws from the supplied rng, so the unseeded global RNG cannot affect it."""
    np.random.seed(0)
    seeded = _spoof(np.random.default_rng(123))

    np.random.seed(9999)  # perturb the global RNG that spoof no longer uses
    same = _spoof(np.random.default_rng(123))

    assert np.array_equal(seeded.coords, same.coords)


@pytest.mark.parametrize("stage", [TRAIN, VAL])
def test_target_coords_carry_raw_points_and_order_outside_training(stage):
    """Writing predictions needs the raw points and their order, without target values."""
    stream_data = StreamData(idx=0, input_steps=1, output_steps=2, healpix_cells=12)
    times = np.repeat(np.datetime64("2020-01-01T06:00"), 3)
    coords_raw = torch.zeros((3, 2))
    idxs_inv = torch.tensor([2, 0, 1])

    stream_data.add_target_coords(
        stage, 1, times, coords_raw, torch.zeros((3, 4)), torch.zeros(12), idxs_inv, False
    )

    assert np.array_equal(stream_data.target_times_raw[1], times)
    assert stream_data.target_coords_raw[1] is coords_raw
    # the order is only needed to write output, so training does not keep it
    expected = torch.tensor([], dtype=torch.int64) if stage == TRAIN else idxs_inv
    assert torch.equal(stream_data.idxs_inv[1], expected)
