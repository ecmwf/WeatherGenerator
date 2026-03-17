"""Tests for spectral masking via spherical harmonics on HEALPix."""

import logging
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pytest

from weathergen.datasets.spectral_masking import (
    _best_nside,
    apply_spectral_masking,
    apply_spectral_masking_binned,
    forward_sht,
    generate_spectral_mask_bands,
    inverse_sht,
    mask_alm_bands,
)

logger = logging.getLogger(__name__)

NSIDE = 64
LMAX = 2 * NSIDE  # 128
NPIX = hp.nside2npix(NSIDE)  # 49152

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots" / "spectral_masking"


# ── Round-trip tests (no masking) ────────────────────────────────────────


class TestRoundTrip:
    """Verify forward_sht → inverse_sht recovers the original map."""

    def test_constant_field(self):
        """A constant field has only l=0; round-trip should be exact."""
        const_val = 42.0
        map_nested = np.full(NPIX, const_val)

        alm = forward_sht(map_nested, NSIDE, LMAX)
        recovered = inverse_sht(alm, NSIDE, LMAX)

        np.testing.assert_allclose(recovered, map_nested, atol=1e-6)

    def test_smooth_random_field(self):
        """Generate a band-limited field via random alm, verify round-trip."""
        rng = np.random.default_rng(123)
        nalm = hp.Alm.getsize(LMAX)

        # Create random alm (band-limited by construction)
        alm_orig = np.zeros(nalm, dtype=np.complex128)
        for ell in range(LMAX + 1):
            for m in range(ell + 1):
                idx = hp.Alm.getidx(LMAX, ell, m)
                real = rng.standard_normal()
                imag = rng.standard_normal() if m > 0 else 0.0
                alm_orig[idx] = complex(real, imag)

        # alm → map → alm round-trip
        map_ring = hp.alm2map(alm_orig, NSIDE, lmax=LMAX)
        map_nested = hp.reorder(map_ring, r2n=True)

        alm_recovered = forward_sht(map_nested, NSIDE, LMAX)
        map_recovered = inverse_sht(alm_recovered, NSIDE, LMAX)

        np.testing.assert_allclose(map_recovered, map_nested, atol=1e-6)


# ── Band generation tests ───────────────────────────────────────────────


class TestGenerateSpectralMaskBands:
    """Verify properties of generated spectral mask bands."""

    @pytest.fixture()
    def bands_fixed_seed(self):
        rng = np.random.default_rng(42)
        return generate_spectral_mask_bands(LMAX, max_num_bands=4, max_log_fraction=0.10, rng=rng)

    def test_band_count(self, bands_fixed_seed):
        assert 1 <= len(bands_fixed_seed) <= 4

    def test_l_values_in_range(self, bands_fixed_seed):
        for l_start, l_end in bands_fixed_seed:
            assert 1 <= l_start <= l_end <= LMAX

    def test_total_log_coverage_within_budget(self, bands_fixed_seed):
        total_log = np.log(LMAX)
        budget = 0.10 * total_log
        coverage = sum(np.log(end + 1) - np.log(start) for start, end in bands_fixed_seed)
        # Allow some slack for integer rounding
        assert coverage <= budget * 1.5

    def test_non_overlapping(self, bands_fixed_seed):
        for i in range(len(bands_fixed_seed) - 1):
            assert bands_fixed_seed[i][1] < bands_fixed_seed[i + 1][0]

    def test_deterministic_with_seed(self):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        bands1 = generate_spectral_mask_bands(LMAX, rng=rng1)
        bands2 = generate_spectral_mask_bands(LMAX, rng=rng2)
        assert bands1 == bands2

    def test_many_seeds_valid(self):
        """Run many seeds to check robustness of band generation."""
        for seed in range(100):
            rng = np.random.default_rng(seed)
            bands = generate_spectral_mask_bands(LMAX, rng=rng)
            assert len(bands) >= 1
            for l_start, l_end in bands:
                assert 1 <= l_start <= l_end <= LMAX


# ── Masking tests ────────────────────────────────────────────────────────


class TestMasking:
    """Verify spectral masking behavior."""

    def test_empty_bands_identity(self):
        """No bands masked → output matches input (within SHT precision)."""
        rng = np.random.default_rng(7)
        map_nested = rng.standard_normal(NPIX)

        result = apply_spectral_masking(map_nested, NSIDE, LMAX, bands=[])
        np.testing.assert_allclose(result, map_nested, atol=1e-10)

    def test_mask_all_but_monopole(self):
        """Masking l=1..lmax leaves only the global mean (l=0)."""
        rng = np.random.default_rng(8)
        map_nested = rng.standard_normal(NPIX)
        global_mean = map_nested.mean()

        result = apply_spectral_masking(map_nested, NSIDE, LMAX, bands=[(1, LMAX)])

        # Result should be approximately constant = global mean
        np.testing.assert_allclose(result, global_mean, atol=1e-4)

    def test_multichannel(self):
        """Multi-channel [C, npix] input works correctly."""
        rng = np.random.default_rng(9)
        n_channels = 3
        map_nested = rng.standard_normal((n_channels, NPIX))

        bands = [(5, 15)]
        result = apply_spectral_masking(map_nested, NSIDE, LMAX, bands)

        assert result.shape == map_nested.shape

        # Verify each channel independently
        for c in range(n_channels):
            expected = apply_spectral_masking(map_nested[c], NSIDE, LMAX, bands)
            np.testing.assert_allclose(result[c], expected, atol=1e-10)

    def test_specific_band_zeroed(self):
        """Verify that masked l values have near-zero power after masking."""
        rng = np.random.default_rng(10)
        map_nested = rng.standard_normal(NPIX)

        band = (10, 30)
        result = apply_spectral_masking(map_nested, NSIDE, LMAX, bands=[band])

        # Check power spectrum of result
        result_ring = hp.reorder(result, n2r=True)
        cl = hp.anafast(result_ring, lmax=LMAX)

        # Power in masked band should be near zero
        for ell in range(band[0], band[1] + 1):
            assert cl[ell] < 1e-10, f"Power at l={ell} should be ~0, got {cl[ell]}"


# ── Plotting tests ───────────────────────────────────────────────────────


class TestPlotting:
    """Generate diagnostic plots for visual verification."""

    def test_mollweide_projection(self):
        """Plot original, masked, and difference maps in Mollweide projection."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)

        # Create a smooth random field
        nalm = hp.Alm.getsize(LMAX)
        alm_orig = np.zeros(nalm, dtype=np.complex128)
        for ell in range(LMAX + 1):
            for m in range(ell + 1):
                idx = hp.Alm.getidx(LMAX, ell, m)
                # Weight by 1/l to make it smooth
                weight = 1.0 / (1 + ell)
                real = rng.standard_normal() * weight
                imag = rng.standard_normal() * weight if m > 0 else 0.0
                alm_orig[idx] = complex(real, imag)

        map_ring = hp.alm2map(alm_orig, NSIDE, lmax=LMAX)
        map_nested = hp.reorder(map_ring, r2n=True)

        bands = [(5, 20), (50, 70)]
        masked = apply_spectral_masking(map_nested, NSIDE, LMAX, bands)
        diff = map_nested - masked

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, data, title in zip(
            axes,
            [map_nested, masked, diff],
            ["Original", "Masked (bands removed)", "Difference (removed signal)"],
        ):
            plt.sca(ax)
            hp.mollview(data, nest=True, title=title, hold=True)

        out_path = PLOT_DIR / "mollweide_spectral_masking.png"
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_power_spectrum_before_after(self):
        """Plot power spectrum before and after masking to confirm band removal."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        map_nested = rng.standard_normal(NPIX)

        bands = [(10, 30), (60, 80)]
        masked = apply_spectral_masking(map_nested, NSIDE, LMAX, bands)

        # Compute power spectra (need ring ordering for anafast)
        orig_ring = hp.reorder(map_nested, n2r=True)
        masked_ring = hp.reorder(masked, n2r=True)

        cl_orig = hp.anafast(orig_ring, lmax=LMAX)
        cl_masked = hp.anafast(masked_ring, lmax=LMAX)

        ells = np.arange(len(cl_orig))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(ells[1:], cl_orig[1:], label="Original", alpha=0.8)
        ax.semilogy(ells[1:], cl_masked[1:], label="After masking", alpha=0.8)

        # Shade masked bands
        for l_start, l_end in bands:
            ax.axvspan(l_start, l_end, alpha=0.2, color="red", label="Masked band")

        ax.set_xlabel("Multipole l")
        ax.set_ylabel("Power C_l")
        ax.set_title("Power spectrum before/after spectral masking")
        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

        out_path = PLOT_DIR / "power_spectrum_masking.png"
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Verify masked bands have near-zero power
        for l_start, l_end in bands:
            for ell in range(l_start, l_end + 1):
                assert cl_masked[ell] < 1e-10, f"l={ell} should have ~0 power"


# ── ERA5 integration test ───────────────────────────────────────────────

# ERA5 data path on santis
ERA5_DATA_PATH = Path("/capstor/store/cscs/userlab/ch17/data")
ERA5_FILENAME = "aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"
ERA5_NSIDE = 32  # ~1° resolution (12288 pixels), close to O96 grid density
ERA5_LMAX = 2 * ERA5_NSIDE  # 64

# Variables to test: name → index in anemoi dataset
ERA5_VARIABLES = {"t_850": 44, "z_500": 96, "msl": 11}


def _bin_to_healpix(
    latitudes: np.typing.NDArray,
    longitudes: np.typing.NDArray,
    field: np.typing.NDArray,
    nside: int,
) -> np.typing.NDArray:
    """Bin irregular grid data onto HEALPix (NESTED) via nearest-neighbor averaging.

    Args:
        latitudes: Latitude array in degrees [-90, 90], shape (n_points,).
        longitudes: Longitude array in degrees [-180, 180] or [0, 360], shape (n_points,).
        field: Data values, shape (n_points,).
        nside: HEALPix nside parameter.

    Returns:
        HEALPix map in NESTED ordering, shape (npix,). Unfilled pixels are set to UNSEEN.
    """
    npix = hp.nside2npix(nside)
    theta = np.deg2rad(90.0 - latitudes)  # colatitude
    phi = np.deg2rad(longitudes % 360.0)  # ensure [0, 2pi)

    pix_indices = hp.ang2pix(nside, theta, phi, nest=True)

    # Average values falling into the same pixel
    healpix_map = np.full(npix, hp.UNSEEN)
    counts = np.zeros(npix, dtype=np.int64)
    sums = np.zeros(npix, dtype=np.float64)

    np.add.at(sums, pix_indices, field.astype(np.float64))
    np.add.at(counts, pix_indices, 1)

    filled = counts > 0
    healpix_map[filled] = sums[filled] / counts[filled]

    return healpix_map


@pytest.mark.skipif(
    not (ERA5_DATA_PATH / ERA5_FILENAME).exists(),
    reason="ERA5 data not available",
)
class TestERA5SpectralMasking:
    """Apply spectral masking to real ERA5 data loaded via the anemoi data reader."""

    @pytest.fixture(scope="class")
    def era5_dataset(self):
        """Load ERA5 dataset via anemoi."""
        import anemoi.datasets as ad

        ds = ad.open_dataset(ERA5_DATA_PATH / ERA5_FILENAME)
        return ds

    @pytest.fixture(scope="class")
    def era5_healpix_maps(self, era5_dataset):
        """Load a single timestep and bin each variable onto HEALPix."""
        ds = era5_dataset
        # Pick a timestep near the middle of the dataset
        t_idx = len(ds.dates) // 2
        logger.info(f"Loading ERA5 timestep {t_idx}: {ds.dates[t_idx]}")

        # ds[t_idx] shape: (1, n_variables, 1, n_gridpoints) → squeeze
        snapshot = ds[t_idx].squeeze()  # (n_variables, n_gridpoints)

        lats = ds.latitudes
        lons = ds.longitudes

        maps = {}
        for var_name, var_idx in ERA5_VARIABLES.items():
            field = snapshot[var_idx]
            healpix_map = _bin_to_healpix(lats, lons, field, ERA5_NSIDE)
            # Fill UNSEEN pixels with global mean for SHT compatibility
            valid = healpix_map != hp.UNSEEN
            healpix_map[~valid] = np.mean(healpix_map[valid])
            maps[var_name] = healpix_map

        return maps

    def test_era5_mollweide_spectral_masking(self, era5_healpix_maps):
        """Plot ERA5 fields: original | masked | difference for each variable."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42)
        bands = generate_spectral_mask_bands(ERA5_LMAX, max_num_bands=3, rng=rng)
        logger.info(f"Spectral mask bands: {bands}")

        for var_name, healpix_map in era5_healpix_maps.items():
            masked = apply_spectral_masking(healpix_map, ERA5_NSIDE, ERA5_LMAX, bands)
            diff = healpix_map - masked

            fig = plt.figure(figsize=(18, 5))

            for i, (data, title) in enumerate(
                [
                    (healpix_map, f"{var_name} — Original"),
                    (masked, f"{var_name} — After spectral masking"),
                    (diff, f"{var_name} — Removed signal"),
                ]
            ):
                hp.mollview(
                    data,
                    nest=True,
                    title=title,
                    sub=(1, 3, i + 1),
                    fig=fig,
                )

            out_path = PLOT_DIR / f"era5_{var_name}_spectral_masking.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            assert out_path.exists()
            assert out_path.stat().st_size > 0
            logger.info(f"Saved {out_path}")

    def test_era5_power_spectrum(self, era5_healpix_maps):
        """Plot power spectra of ERA5 fields before/after masking."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42)
        bands = generate_spectral_mask_bands(ERA5_LMAX, max_num_bands=3, rng=rng)

        fig, axes = plt.subplots(1, len(ERA5_VARIABLES), figsize=(7 * len(ERA5_VARIABLES), 6))
        if len(ERA5_VARIABLES) == 1:
            axes = [axes]

        for ax, (var_name, healpix_map) in zip(axes, era5_healpix_maps.items()):
            masked = apply_spectral_masking(healpix_map, ERA5_NSIDE, ERA5_LMAX, bands)

            cl_orig = hp.anafast(hp.reorder(healpix_map, n2r=True), lmax=ERA5_LMAX)
            cl_masked = hp.anafast(hp.reorder(masked, n2r=True), lmax=ERA5_LMAX)

            ells = np.arange(len(cl_orig))
            ax.semilogy(ells[1:], cl_orig[1:], label="Original", alpha=0.8)
            ax.semilogy(ells[1:], cl_masked[1:], label="After masking", alpha=0.8)

            for l_start, l_end in bands:
                ax.axvspan(l_start, l_end, alpha=0.2, color="red", label="Masked band")

            ax.set_xlabel("Multipole l")
            ax.set_ylabel("Power C_l")
            ax.set_title(f"{var_name}")
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=8)

        fig.suptitle("ERA5 power spectra — before/after spectral masking", fontsize=14)
        fig.tight_layout()

        out_path = PLOT_DIR / "era5_power_spectra.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Verify masked bands have zero power by checking alm coefficients directly.
        # (Using anafast would introduce error because it calls map2alm with iter=0.)
        for var_name, healpix_map in era5_healpix_maps.items():
            alm = forward_sht(healpix_map, ERA5_NSIDE, ERA5_LMAX)
            alm_masked = mask_alm_bands(alm, ERA5_LMAX, bands)
            for l_start, l_end in bands:
                for ell in range(l_start, l_end + 1):
                    for m in range(ell + 1):
                        idx = hp.Alm.getidx(ERA5_LMAX, ell, m)
                        assert alm_masked[idx] == 0.0, (
                            f"{var_name}: alm[l={ell},m={m}] should be 0, "
                            f"got {alm_masked[idx]}"
                        )


# ── ERA5 masking fraction sweep ────────────────────────────────────────

MASKING_FRACTIONS = [0.05, 0.10, 0.20, 0.40]


@pytest.mark.skipif(
    not (ERA5_DATA_PATH / ERA5_FILENAME).exists(),
    reason="ERA5 data not available",
)
class TestERA5FractionSweep:
    """Sweep max_log_fraction from 5% to 40% on ERA5 data (1 band, min 1%)."""

    @pytest.fixture(scope="class")
    def era5_healpix_maps(self):
        """Load a single ERA5 timestep and bin onto HEALPix."""
        import anemoi.datasets as ad

        ds = ad.open_dataset(ERA5_DATA_PATH / ERA5_FILENAME)
        t_idx = len(ds.dates) // 2
        logger.info(f"Loading ERA5 timestep {t_idx}: {ds.dates[t_idx]}")

        snapshot = ds[t_idx].squeeze()
        lats = ds.latitudes
        lons = ds.longitudes

        maps = {}
        for var_name, var_idx in ERA5_VARIABLES.items():
            field = snapshot[var_idx]
            healpix_map = _bin_to_healpix(lats, lons, field, ERA5_NSIDE)
            valid = healpix_map != hp.UNSEEN
            healpix_map[~valid] = np.mean(healpix_map[valid])
            maps[var_name] = healpix_map

        return maps

    def test_era5_fraction_sweep_power_spectra(self, era5_healpix_maps):
        """Power spectra + band diagram for each masking fraction on ERA5 t_850."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        var_name = "t_850"
        healpix_map = era5_healpix_maps[var_name]

        cl_orig = hp.anafast(hp.reorder(healpix_map, n2r=True), lmax=ERA5_LMAX)
        ells = np.arange(len(cl_orig))

        all_bands: dict[float, list[tuple[int, int]]] = {}
        all_cl_masked: dict[float, np.ndarray] = {}
        for frac in MASKING_FRACTIONS:
            rng = np.random.default_rng(42)
            bands = generate_spectral_mask_bands(
                ERA5_LMAX, max_num_bands=1, max_log_fraction=frac, min_log_fraction=0.01, rng=rng
            )
            masked = apply_spectral_masking(healpix_map, ERA5_NSIDE, ERA5_LMAX, bands)
            all_bands[frac] = bands
            all_cl_masked[frac] = hp.anafast(hp.reorder(masked, n2r=True), lmax=ERA5_LMAX)
            logger.info(f"  {int(frac * 100)}% → bands: {bands}")

        n_fracs = len(MASKING_FRACTIONS)

        # ── Power spectra panels (log x-axis) ──
        fig, axes = plt.subplots(1, n_fracs, figsize=(6 * n_fracs, 5), sharey=True)
        for ax, frac in zip(axes, MASKING_FRACTIONS):
            ax.loglog(ells[1:], cl_orig[1:], label="Original", color="black", alpha=0.5)
            ax.loglog(ells[1:], all_cl_masked[frac][1:], label="Masked", alpha=0.9)
            for l_start, l_end in all_bands[frac]:
                ax.axvspan(l_start, l_end, alpha=0.15, color="red")
            ax.set_xlabel("Multipole l")
            if ax is axes[0]:
                ax.set_ylabel("Power C_l")
            ax.set_title(f"{int(frac * 100)}% log-fraction")
            ax.legend(fontsize=7)

        fig.suptitle(f"ERA5 {var_name} — power spectra, 1 band masking sweep", fontsize=13)
        fig.tight_layout()
        out = PLOT_DIR / "era5_fraction_sweep_power_spectra.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        assert out.exists()

        # ── Frequency bands diagram (log x-axis) ──
        fig, ax = plt.subplots(figsize=(10, 3 + 0.6 * n_fracs))
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_fracs))

        for i, frac in enumerate(MASKING_FRACTIONS):
            y = i
            for l_start, l_end in all_bands[frac]:
                ax.barh(
                    y,
                    l_end - l_start + 1,
                    left=l_start,
                    height=0.6,
                    color=colors[i],
                    edgecolor="black",
                    linewidth=0.5,
                )

        ax.set_xscale("log")
        ax.set_yticks(range(n_fracs))
        ax.set_yticklabels([f"{int(f * 100)}% log-frac" for f in MASKING_FRACTIONS])
        ax.set_xlabel("Multipole l")
        ax.set_xlim(1, ERA5_LMAX * 1.3)
        ax.set_title(f"ERA5 {var_name} — masked frequency band (1 band, min 1%)")
        fig.tight_layout()
        out = PLOT_DIR / "era5_fraction_sweep_bands.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        assert out.exists()

    def test_era5_fraction_sweep_mollweide(self, era5_healpix_maps):
        """Mollweide maps at each masking fraction for all ERA5 variables."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        n_fracs = len(MASKING_FRACTIONS)
        n_vars = len(ERA5_VARIABLES)

        for var_name, healpix_map in era5_healpix_maps.items():
            fig = plt.figure(figsize=(6 * n_fracs, 10))

            for col, frac in enumerate(MASKING_FRACTIONS):
                rng = np.random.default_rng(42)
                bands = generate_spectral_mask_bands(
                    ERA5_LMAX, max_num_bands=1, max_log_fraction=frac, min_log_fraction=0.01, rng=rng
                )
                masked = apply_spectral_masking(healpix_map, ERA5_NSIDE, ERA5_LMAX, bands)
                diff = healpix_map - masked

                for row, (data, label) in enumerate(
                    [
                        (masked, "Masked"),
                        (diff, "Removed signal"),
                    ]
                ):
                    subplot_idx = row * n_fracs + col + 1
                    hp.mollview(
                        data,
                        nest=True,
                        title=f"{label} — {int(frac * 100)}%",
                        sub=(2, n_fracs, subplot_idx),
                        fig=fig,
                    )

            fig.suptitle(
                f"ERA5 {var_name} — spectral masking fraction sweep", fontsize=14, y=1.02
            )
            out = PLOT_DIR / f"era5_{var_name}_fraction_sweep_mollweide.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            assert out.exists()
            logger.info(f"Saved {out}")

    def test_era5_fraction_sweep_alm_verified(self, era5_healpix_maps):
        """Verify alm coefficients are exactly zero in masked bands for all fractions."""
        var_name = "t_850"
        healpix_map = era5_healpix_maps[var_name]

        for frac in MASKING_FRACTIONS:
            rng = np.random.default_rng(42)
            bands = generate_spectral_mask_bands(
                ERA5_LMAX, max_num_bands=1, max_log_fraction=frac, min_log_fraction=0.01, rng=rng
            )
            alm = forward_sht(healpix_map, ERA5_NSIDE, ERA5_LMAX)
            alm_masked = mask_alm_bands(alm, ERA5_LMAX, bands)

            for l_start, l_end in bands:
                for ell in range(l_start, l_end + 1):
                    for m in range(ell + 1):
                        idx = hp.Alm.getidx(ERA5_LMAX, ell, m)
                        assert alm_masked[idx] == 0.0, (
                            f"frac={frac}: alm[l={ell},m={m}] should be 0"
                        )


# ── Pipeline integration tests ──────────────────────────────────────────


class TestMaskerSpectralStrategy:
    """Test the 'spectral' masking strategy integrated in Masker."""

    def test_generate_cell_mask_spectral_returns_all_true(self):
        """Spectral strategy should return all-True mask (all cells kept)."""
        from weathergen.datasets.masking import Masker

        healpix_level = 2
        masker = Masker(healpix_level, stage="train")
        masker.rng = np.random.default_rng(42)

        num_cells = 12 * (4**healpix_level)
        mask, params = masker._generate_cell_mask(
            num_cells,
            strategy="spectral",
            masking_strategy_config={"lmax": 16, "max_num_bands": 2},
        )

        # All cells should be kept
        assert mask.all(), "Spectral mask should keep all cells"

    def test_generate_cell_mask_spectral_stores_band_info(self):
        """Spectral strategy should store band info in masking_params."""
        from weathergen.datasets.masking import Masker

        healpix_level = 2
        masker = Masker(healpix_level, stage="train")
        masker.rng = np.random.default_rng(42)

        num_cells = 12 * (4**healpix_level)
        mask, params = masker._generate_cell_mask(
            num_cells,
            strategy="spectral",
            masking_strategy_config={"lmax": 16, "max_num_bands": 2},
        )

        assert "spectral_bands" in params
        assert "spectral_lmax" in params
        assert params["spectral_lmax"] == 16
        assert len(params["spectral_bands"]) >= 1

    def test_generate_cell_mask_spectral_default_lmax(self):
        """When lmax not specified, default to 2*nside."""
        from weathergen.datasets.masking import Masker

        healpix_level = 3
        masker = Masker(healpix_level, stage="train")
        masker.rng = np.random.default_rng(42)

        num_cells = 12 * (4**healpix_level)
        mask, params = masker._generate_cell_mask(
            num_cells,
            strategy="spectral",
            masking_strategy_config={},
        )

        expected_nside = 2**healpix_level
        assert params["spectral_lmax"] == 2 * expected_nside


def _make_healpix_coords(nside: int) -> np.ndarray:
    """Generate (lat, lon) coords in degrees for all HEALPix pixels (NESTED)."""
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)
    lat = 90.0 - np.rad2deg(theta)
    lon = np.rad2deg(phi)
    return np.column_stack([lat, lon]).astype(np.float32)


class TestBestNside:
    """Test _best_nside helper."""

    def test_exact_healpix_sizes(self):
        for level in range(8):
            nside = 2**level
            npix = 12 * nside**2
            assert _best_nside(npix) == nside

    def test_o96_grid(self):
        """O96 reduced Gaussian grid (~40320 points) → nside=64 (npix=49152)."""
        nside = _best_nside(40320)
        assert nside == 64  # |49152-40320| < |12288-40320|


class TestApplySpectralMaskingBinned:
    """Test apply_spectral_masking_binned on non-HEALPix grids."""

    def test_on_healpix_grid_matches_direct(self):
        """On a native HEALPix grid, binned approach should match direct SHT."""
        nside = 8
        npix = hp.nside2npix(nside)
        lmax = 2 * nside
        rng = np.random.default_rng(42)

        coords = _make_healpix_coords(nside)
        data = rng.standard_normal((npix, 2)).astype(np.float64)
        bands = [(3, 8)]

        result_binned = apply_spectral_masking_binned(data, coords, lmax, bands)
        # Direct method for comparison
        result_direct = np.empty_like(data)
        for c in range(2):
            result_direct[:, c] = apply_spectral_masking(data[:, c], nside, lmax, bands)

        np.testing.assert_allclose(result_binned, result_direct, atol=1e-6)

    def test_non_healpix_grid_runs(self):
        """Spectral masking on a regular lat-lon grid should not error."""
        lats = np.linspace(-90, 90, 50)
        lons = np.linspace(0, 360, 100, endpoint=False)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()]).astype(np.float32)
        n_points = coords.shape[0]

        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_points, 3)).astype(np.float64)
        bands = [(2, 10)]
        lmax = 32

        result = apply_spectral_masking_binned(data, coords, lmax, bands)
        assert result.shape == data.shape
        assert not np.allclose(result, data, atol=1e-6)

    def test_empty_bands_identity(self):
        """No bands → data unchanged."""
        coords = _make_healpix_coords(4)
        data = np.random.default_rng(1).standard_normal((hp.nside2npix(4), 2))
        result = apply_spectral_masking_binned(data, coords, lmax=8, bands=[])
        np.testing.assert_array_equal(result, data)


class TestApplySpectralToRdataList:
    """Test the _apply_spectral_to_rdata_list helper."""

    def test_modifies_data_values(self):
        """Spectral masking should change data values."""
        from weathergen.common.io import IOReaderData
        from weathergen.datasets.multi_stream_data_sampler import _apply_spectral_to_rdata_list

        nside = 4
        npix = 12 * nside**2
        lmax = 2 * nside
        n_channels = 3

        rng = np.random.default_rng(42)
        data = rng.standard_normal((npix, n_channels)).astype(np.float64)
        coords = _make_healpix_coords(nside)
        geoinfos = np.zeros((npix, 1), dtype=np.float32)
        datetimes = np.array([np.datetime64("2023-01-01")] * npix)

        rdata = IOReaderData(
            coords=coords, geoinfos=geoinfos, data=data, datetimes=datetimes
        )

        bands = [(2, 5)]
        result = _apply_spectral_to_rdata_list([rdata], bands, lmax)

        assert len(result) == 1
        assert result[0].data.shape == data.shape
        # Data should be modified
        assert not np.allclose(result[0].data, data, atol=1e-6)

    def test_preserves_spoof_data(self):
        """Spoof data should pass through unchanged."""
        from weathergen.common.io import IOReaderData
        from weathergen.datasets.multi_stream_data_sampler import _apply_spectral_to_rdata_list

        nside = 4
        npix = 12 * nside**2
        lmax = 2 * nside

        data = np.ones((npix, 2), dtype=np.float32)
        coords = _make_healpix_coords(nside)
        geoinfos = np.zeros((npix, 1), dtype=np.float32)
        datetimes = np.array([np.datetime64("2023-01-01")] * npix)

        rdata = IOReaderData(
            coords=coords, geoinfos=geoinfos, data=data, datetimes=datetimes
        )
        rdata.is_spoof = True

        bands = [(2, 5)]
        result = _apply_spectral_to_rdata_list([rdata], bands, lmax)

        assert len(result) == 1
        # Spoof data should be returned as-is
        assert result[0] is rdata

    def test_does_not_modify_original(self):
        """Original rdata should not be modified."""
        from weathergen.common.io import IOReaderData
        from weathergen.datasets.multi_stream_data_sampler import _apply_spectral_to_rdata_list

        nside = 4
        npix = 12 * nside**2
        lmax = 2 * nside

        rng = np.random.default_rng(99)
        data = rng.standard_normal((npix, 2)).astype(np.float64)
        original_data = data.copy()
        coords = _make_healpix_coords(nside)
        geoinfos = np.zeros((npix, 1), dtype=np.float32)
        datetimes = np.array([np.datetime64("2023-01-01")] * npix)

        rdata = IOReaderData(
            coords=coords, geoinfos=geoinfos, data=data, datetimes=datetimes
        )

        bands = [(2, 5)]
        _apply_spectral_to_rdata_list([rdata], bands, lmax)

        # Original data should be unchanged
        np.testing.assert_array_equal(rdata.data, original_data)
