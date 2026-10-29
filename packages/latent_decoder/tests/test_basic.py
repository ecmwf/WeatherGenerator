import numpy as np
import pytest
import torch
from astropy_healpix.healpy import ang2pix

from weathergen.latent_decoder import geometry
from weathergen.latent_decoder.dataset import pairs
from weathergen.latent_decoder.head import Head
from weathergen.latent_decoder.latents import _window, open_latents


def test_cell_and_neighbours():
    nb = geometry.neighbours()
    assert nb.shape == (geometry.N_CELLS, 9)
    assert (nb[:, 0] == np.arange(geometry.N_CELLS)).all()
    assert nb.min() >= 0 and nb.max() < geometry.N_CELLS
    # the cell of a point near a cell's centre is that cell, and the offset is small
    lat, lon = np.array([47.3]), np.array([8.5])
    cell = geometry.cell_of(lat, lon)
    f = geometry.position_features(lat, lon, cell)
    assert np.abs(f[0, 3:]).max() < 1
    # cell centres follow the tokenizer's convention (phi = lon + 180): each maps back to its cell
    clat, clon = geometry.cell_centres()
    assert (geometry.cell_of(clat, clon) == np.arange(geometry.N_CELLS)).all()
    # and the neighbours of a cell are close to it
    d = (
        geometry.unit_vectors(clat[nb[:, 1:]], clon[nb[:, 1:]])
        - geometry.unit_vectors(clat, clon)[:, None]
    )
    assert np.linalg.norm(d, axis=-1).max() < np.deg2rad(4.0)


def test_window_and_pairs():
    b = np.datetime64("2023-06-01T00", "s")
    assert _window(b, 3) == (
        np.datetime64("2023-06-01T18", "s"),
        np.datetime64("2023-06-02T00", "s"),
    )
    bases = b + np.timedelta64(6, "h") * np.arange(8)
    p = pairs(bases, "2023-06-01T06", "2023-06-01T12")
    assert [x[0] for x in p[::40]] == list(bases[1:3]) and {k for _, k in p} == set(range(1, 41))


def test_head_shapes():
    head = Head(n_features=15, n_out=6, dim_latent=32, dim=16, heads=2)
    out = head(torch.randn(100, 32), torch.randint(0, 100, (7, 9)), torch.randn(7, 15))
    assert out.shape == (7, 6)


def test_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        open_latents(tmp_path)


def test_cells_follow_the_tokenizer():
    """WeatherGenerator's tokenizer puts the centre of each token's cell into that token."""
    tokenizer_utils = pytest.importorskip("weathergen.datasets.tokenizer_utils")
    lat, lon = geometry.cell_centres()
    thetas, phis = tokenizer_utils.theta_phi_to_standard_coords(np.stack([lat, lon], axis=1))
    cells = ang2pix(geometry.NSIDE, thetas, phis, nest=True)
    np.testing.assert_array_equal(cells, np.arange(geometry.N_CELLS))
