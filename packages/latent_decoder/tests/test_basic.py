# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch
from astropy_healpix.healpy import ang2pix

from weathergen.latent_decoder import geometry
from weathergen.latent_decoder.latents import LatentReader, state_window
from weathergen.latent_decoder.model import LatentDecoder
from weathergen.latent_decoder.run_train import get_samples, load_config


def test_cell_and_neighbours():
    nbours = geometry.hp_nbours()
    assert nbours.shape == (geometry.NUM_HEALPIX_CELLS, 9)
    assert (nbours[:, 0] == np.arange(geometry.NUM_HEALPIX_CELLS)).all()
    assert nbours.min() >= 0 and nbours.max() < geometry.NUM_HEALPIX_CELLS
    # the cell of a point near a cell's centre is that cell, and the offset is small
    lat, lon = np.array([47.3]), np.array([8.5])
    cell = geometry.cell_of(lat, lon)
    f = geometry.position_features(lat, lon, cell)
    assert np.abs(f[0, 3:]).max() < 1
    # cell centres follow the tokenizer's convention (phi = lon + 180): each maps back to its cell
    clat, clon = geometry.cell_centres()
    assert (geometry.cell_of(clat, clon) == np.arange(geometry.NUM_HEALPIX_CELLS)).all()
    # and the neighbours of a cell are close to it
    d = (
        geometry.unit_vectors(clat[nbours[:, 1:]], clon[nbours[:, 1:]])
        - geometry.unit_vectors(clat, clon)[:, None]
    )
    assert np.linalg.norm(d, axis=-1).max() < np.deg2rad(4.0)


def test_window_and_samples():
    t = np.datetime64("2023-06-01T00", "s")
    dtr = state_window(t, 3)
    assert (dtr.start, dtr.end) == (
        np.datetime64("2023-06-01T18", "s"),
        np.datetime64("2023-06-02T00", "s"),
    )
    init_times = t + np.timedelta64(6, "h") * np.arange(8)
    samples = get_samples(init_times, [["2023-06-01T06:00", "2023-06-01T12:00"]])
    assert [s[0] for s in samples[::40]] == list(init_times[1:3])
    assert {fstep for _, fstep in samples} == set(range(1, 41))


def test_config():
    cf, val_cf = load_config([], ["results_path=/tmp/x", "validation_config.init_time_stride=1"])
    assert val_cf.init_time_stride == 1
    # the validation config falls back to the training config
    assert (
        val_cf.learning_rate_scheduling.lr_max == cf.training_config.learning_rate_scheduling.lr_max
    )


def test_decoder_shapes():
    model = LatentDecoder(dim_coords=15, num_channels=6, dim_latent=32, dim_embed=16, num_heads=2)
    out = model(torch.randn(100, 32), torch.randint(0, 100, (7, 9)), torch.randn(7, 15))
    assert out.shape == (7, 6)


def test_no_latents(tmp_path):
    with pytest.raises(ValueError):
        LatentReader(tmp_path)


def test_cells_follow_the_tokenizer():
    """WeatherGenerator's tokenizer puts the centre of each token's cell into that token."""
    tokenizer_utils = pytest.importorskip("weathergen.datasets.tokenizer_utils")
    lat, lon = geometry.cell_centres()
    thetas, phis = tokenizer_utils.theta_phi_to_standard_coords(np.stack([lat, lon], axis=1))
    cells = ang2pix(geometry.NSIDE, thetas, phis, nest=True)
    np.testing.assert_array_equal(cells, np.arange(geometry.NUM_HEALPIX_CELLS))
