# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the inference-time per-ODE-step diagnostics.

Drives ``ODEDiagnostics`` through its real lifecycle with a stub decoder, so the collection,
the ``idxs_inv`` re-ordering and the figure production are exercised without a model.
"""

import numpy as np
import pytest
import torch

from weathergen.evaluate.scores.psd import _legendre_gauss_weights, _octahedral_lons_per_lat
from weathergen.model.inference_diagnostics import ODEDiagnostics

NSIDE = 8
NPIX = 12 * NSIDE**2
DIM = 6
CHANNELS = ["2t", "q_850"]
ALL_CHANNELS = ["z_500", "2t", "10u", "q_850"]


def _o96_like_grid(nlat: int = 16):
    """A small octahedral grid, so detect_grid_type recognises it and the SHT path runs."""
    lons_per_lat = _octahedral_lons_per_lat(nlat)
    nodes, _ = _legendre_gauss_weights(nlat)
    theta = np.flip(np.arccos(nodes))
    lats = np.concatenate([np.full(n, 90.0 - np.degrees(t)) for t, n in
                           zip(theta, lons_per_lat, strict=True)])
    lons = np.concatenate([360.0 * np.arange(n) / n for n in lons_per_lat])
    return lats, lons


class _StubDecoder:
    """Decodes latents to a physical field by broadcasting per-cell means onto the grid."""

    def __init__(self, n_points: int, n_channels: int, stream: str = "ERA5"):
        self.n_points = n_points
        self.n_channels = n_channels
        self.stream = stream
        self.calls = 0

    def __call__(self, tokens: torch.Tensor) -> dict:
        self.calls += 1
        scale = float(tokens.mean())
        rng = np.random.default_rng(0)
        field = rng.standard_normal((self.n_points, self.n_channels)) + scale
        return {self.stream: (torch.from_numpy(field).float(),)}


def _make_diagnostics(tmp_path, n_points, **kwargs):
    return ODEDiagnostics(
        out_dir=tmp_path,
        stream="ERA5",
        channels=CHANNELS,
        channel_names=ALL_CHANNELS,
        nside=NSIDE,
        denormalize=lambda _stream, data: data * 2.0 + 1.0,
        **kwargs,
    )


def _target_aux(n_points, n_channels, idxs_inv=None):
    rng = np.random.default_rng(1)
    return {
        "target": (torch.from_numpy(rng.standard_normal((n_points, n_channels))).float(),),
        "target_coords": (torch.zeros(n_points, 2),),
        "idxs_inv": (idxs_inv,),
    }


def _run(diag, decoder, n_steps=3):
    diag.set_batch(0)
    diag.bind_decoder(decoder)
    diag.begin(torch.randn(1, NPIX, DIM))
    for i in range(n_steps):
        diag.on_step(i, 1.0 - i / n_steps, torch.randn(1, NPIX, DIM), torch.randn(1, NPIX, DIM))


def test_unknown_channel_is_rejected_early(tmp_path):
    """A typo in diag_channels must fail at construction, not after a 50-step sample."""
    with pytest.raises(ValueError, match="not in stream"):
        ODEDiagnostics(
            out_dir=tmp_path,
            stream="ERA5",
            channels=["2t", "nope"],
            channel_names=ALL_CHANNELS,
            nside=NSIDE,
            denormalize=lambda _s, d: d,
        )


def test_disabled_for_later_batches(tmp_path):
    diag = _make_diagnostics(tmp_path, 100)
    decoder = _StubDecoder(100, len(ALL_CHANNELS))

    diag.set_batch(1)
    diag.bind_decoder(decoder)
    diag.begin(torch.randn(1, NPIX, DIM))
    diag.on_step(0, 1.0, torch.randn(1, NPIX, DIM), torch.randn(1, NPIX, DIM))

    assert not diag.enabled
    assert decoder.calls == 0
    assert diag.steps == []


def test_every_n_steps_subsamples(tmp_path):
    diag = _make_diagnostics(tmp_path, 100, every_n_steps=3)
    decoder = _StubDecoder(100, len(ALL_CHANNELS))
    _run(diag, decoder, n_steps=10)

    assert diag.steps == [0, 3, 6, 9]
    # 1 decode for the target + 2 per recorded step.
    assert decoder.calls == 1 + 2 * len(diag.steps)


def test_render_writes_maps_and_spectra(tmp_path):
    lats, lons = _o96_like_grid()
    n_points = lats.size
    diag = _make_diagnostics(tmp_path, n_points, latent_channels=0)
    decoder = _StubDecoder(n_points, len(ALL_CHANNELS))
    _run(diag, decoder, n_steps=3)

    aux = _target_aux(n_points, len(ALL_CHANNELS))
    aux["target_coords"] = (torch.from_numpy(np.stack([lats, lons], axis=1)).float(),)
    diag.render(aux)

    spectra = sorted(p.name for p in (tmp_path / "spectra").glob("*.png"))
    assert [s for s in spectra if s.startswith("step")] == [
        "step000.png", "step001.png", "step002.png"
    ]
    # One evolution overlay for the latent plus one per physical channel.
    assert {s for s in spectra if s.startswith("evolution")} == {
        "evolution_latent.png", "evolution_2t.png", "evolution_q_850.png"
    }

    maps = sorted(p.name for p in (tmp_path / "maps").glob("*.png"))
    assert maps == [f"step{i:03d}_{c}.png" for i in range(3) for c in CHANNELS]


def test_terminal_frame_is_always_recorded_without_x0_hat(tmp_path):
    """The final decoded sample (x0_hat=None, force=True) must record even if step % n != 0."""
    diag = _make_diagnostics(tmp_path, 100, every_n_steps=3)
    decoder = _StubDecoder(100, len(ALL_CHANNELS))
    diag.set_batch(0)
    diag.bind_decoder(decoder)
    diag.begin(torch.randn(1, NPIX, DIM))
    for i in range(8):
        diag.on_step(i, 1.0 - i / 8, torch.randn(1, NPIX, DIM), torch.randn(1, NPIX, DIM))
    # step 8 is not a multiple of 3, but force=True records it anyway.
    diag.on_step(8, 0.0, torch.randn(1, NPIX, DIM), None, force=True)

    assert diag.steps == [0, 3, 6, 8]
    assert diag.phys["x0_hat"][-1] is None
    assert diag.latent_psd["x0_hat"][-1] is None
    assert diag.phys["x_t"][-1] is not None  # the output field is still recorded


def test_terminal_frame_renders_with_three_map_panels(tmp_path):
    """At the terminal frame the x0_hat panel is dropped, leaving x_t | decode(z) | truth."""
    lats, lons = _o96_like_grid()
    n_points = lats.size
    diag = _make_diagnostics(tmp_path, n_points, latent_channels=0)
    decoder = _StubDecoder(n_points, len(ALL_CHANNELS))
    diag.set_batch(0)
    diag.bind_decoder(decoder)
    diag.begin(torch.randn(1, NPIX, DIM))
    diag.on_step(0, 1.0, torch.randn(1, NPIX, DIM), torch.randn(1, NPIX, DIM))
    diag.on_step(1, 0.0, torch.randn(1, NPIX, DIM), None, force=True)

    aux = _target_aux(n_points, len(ALL_CHANNELS))
    aux["target_coords"] = (torch.from_numpy(np.stack([lats, lons], axis=1)).float(),)
    diag.render(aux)  # must not raise on the None (terminal) x0_hat frame

    # Both frames produced a figure for each channel, including the terminal one.
    maps = sorted(p.name for p in (tmp_path / "maps").glob("*.png"))
    assert {"step000_2t.png", "step000_q_850.png",
            "step001_2t.png", "step001_q_850.png"} <= set(maps)


def test_render_applies_idxs_inv_to_predictions(tmp_path):
    """Predictions and targets must be permuted identically, as write_output does."""
    lats, lons = _o96_like_grid()
    n_points = lats.size
    diag = _make_diagnostics(tmp_path, n_points, latent_channels=0)
    _run(diag, _StubDecoder(n_points, len(ALL_CHANNELS)), n_steps=1)

    before = diag.phys["x_t"][0].copy()
    perm = torch.from_numpy(np.random.default_rng(3).permutation(n_points))
    aux = _target_aux(n_points, len(ALL_CHANNELS), idxs_inv=perm)
    aux["target_coords"] = (torch.from_numpy(np.stack([lats, lons], axis=1)).float(),)
    diag.render(aux)

    np.testing.assert_array_equal(diag.phys["x_t"][0], before[perm.numpy()])


def test_render_is_a_noop_without_collected_steps(tmp_path):
    diag = _make_diagnostics(tmp_path, 50)
    diag.set_batch(0)
    diag.render(_target_aux(50, len(ALL_CHANNELS)))

    assert not (tmp_path / "spectra").exists()
    assert not (tmp_path / "maps").exists()


def test_latent_channel_subset_is_stable_across_steps(tmp_path):
    """The same channels must be used at every step, or the curves are not comparable."""
    diag = _make_diagnostics(tmp_path, 50, latent_channels=2)
    tokens = torch.randn(1, NPIX, DIM)

    first = diag._latent_psd(tokens)[1]
    second = diag._latent_psd(tokens)[1]

    np.testing.assert_allclose(first, second)


def test_token_count_mismatch_is_reported(tmp_path):
    diag = _make_diagnostics(tmp_path, 50)
    with pytest.raises(ValueError, match="not a multiple of npix"):
        diag._latent_maps(torch.randn(1, NPIX + 3, DIM))
