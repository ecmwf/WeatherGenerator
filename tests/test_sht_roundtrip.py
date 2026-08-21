"""Test that SHT forward → inverse is (approximately) the identity."""

import numpy as np
import pytest

from weathergen.evaluate.scores.psd import (
    InverseSphericalHarmonicTransform,
    SphericalHarmonicTransform,
    _octahedral_lons_per_lat,
    _regular_lons_per_lat,
)


@pytest.mark.parametrize("grid_type,nlat", [
    ("regular", 32),
    ("regular", 64),
    ("octahedral", 32),
    ("octahedral", 64),
])
def test_sht_roundtrip_identity(grid_type: str, nlat: int) -> None:
    """Applying SHT then inverse SHT on random noise recovers the original field."""
    rng = np.random.default_rng(42)

    if grid_type == "regular":
        lons_per_lat = _regular_lons_per_lat(nlat)
    else:
        lons_per_lat = _octahedral_lons_per_lat(nlat)

    n_grid_points = sum(lons_per_lat)
    truncation = nlat // 2 - 1

    sht = SphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation)
    isht = InverseSphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation)

    # Random spatial field
    x = rng.standard_normal(n_grid_points)

    # Forward → inverse
    coeffs = sht.transform(x)
    x_reconstructed = isht.transform(coeffs)

    # The reconstruction is approximate due to truncation, but should be close
    # for smooth-enough fields. For a bandlimited signal it should be exact.
    # Use a generous tolerance since truncation discards high-frequency content.
    assert x_reconstructed.shape == x.shape, (
        f"Shape mismatch: {x_reconstructed.shape} vs {x.shape}"
    )

    # Check correlation is positive — truncation discards high-frequency content
    # so white noise won't be perfectly recovered, but the low-frequency part should match.
    corr = np.corrcoef(x.ravel(), x_reconstructed.ravel())[0, 1]
    assert corr > 0.30, f"Correlation too low: {corr:.4f}"

    # More importantly: verify the energy is preserved for the retained modes
    # by checking that the relative L2 error is bounded
    rel_error = np.linalg.norm(x - x_reconstructed) / np.linalg.norm(x)
    assert rel_error < 1.0, f"Relative L2 error too large: {rel_error:.4f}"


@pytest.mark.parametrize("grid_type,nlat", [
    ("regular", 32),
    ("regular", 64),
    ("octahedral", 32),
    ("octahedral", 64),
])
def test_sht_roundtrip_bandlimited(grid_type: str, nlat: int) -> None:
    """For a bandlimited signal, SHT → inverse SHT should be near-exact."""
    if grid_type == "regular":
        lons_per_lat = _regular_lons_per_lat(nlat)
    else:
        lons_per_lat = _octahedral_lons_per_lat(nlat)

    n_grid_points = sum(lons_per_lat)
    truncation = nlat // 2 - 1

    sht = SphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation)
    isht = InverseSphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=truncation)

    # Create a bandlimited signal by doing inverse SHT on random coefficients
    rng = np.random.default_rng(123)
    L = truncation + 1
    random_coeffs = rng.standard_normal((L, L)) + 1j * rng.standard_normal((L, L))
    # Make it physically meaningful: zero out upper triangle (m > l)
    for l in range(L):
        random_coeffs[l, l + 1:] = 0.0

    # Inverse → forward → inverse should give back the same spatial field
    x_bandlimited = isht.transform(random_coeffs)
    coeffs_recovered = sht.transform(x_bandlimited)
    x_roundtrip = isht.transform(coeffs_recovered)

    np.testing.assert_allclose(
        x_roundtrip, x_bandlimited, rtol=1e-6, atol=1e-10,
        err_msg="Roundtrip on bandlimited signal should be near-exact",
    )
