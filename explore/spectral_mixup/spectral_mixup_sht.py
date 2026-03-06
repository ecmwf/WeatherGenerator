"""
Spectral Mix-Up with Spherical Harmonic Transform — global ERA5 analysis.

Companion to spectral_mixup.py (which uses 2D FFT on a regional crop).
This script operates on the *full globe* using a proper Spherical Harmonic
Transform (SHT), with ERA5 data regridded from HEALPix onto a Gaussian grid.

Frequency splitting is done by total wavenumber ℓ (the natural scale
parameter on the sphere): low-ℓ captures planetary-scale structure, high-ℓ
captures synoptic / mesoscale detail.

Usage (from WeatherGenerator root):
    uv run python explore/spectral_mixup/spectral_mixup_sht.py [--out-dir DIR] [--nlat 128]

Figures are saved to OUT_DIR (default: explore/spectral_mixup/output_sht/).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.interpolate import griddata

from sht_helpers import (
    InverseSphericalHarmonicTransform,
    SphericalHarmonicTransform,
    legendre_gauss_weights,
)

# ---------------------------------------------------------------------------
# 1 — Resolve data paths (same as spectral_mixup.py)
# ---------------------------------------------------------------------------

SCRIPT_DIR    = Path(__file__).parent
WORKTREE_ROOT = SCRIPT_DIR / "../.."
WG_PRIVATE    = WORKTREE_ROOT / "../WeatherGenerator-private"

paths_cfg  = yaml.safe_load(open(WG_PRIVATE / "hpc/santis/config/paths.yml"))
DATA_PATHS = paths_cfg["data_paths"]

era5_cfg   = yaml.safe_load(open(WORKTREE_ROOT / "config/streams/era5_1deg/era5.yml"))
ERA5_FNAME = era5_cfg["ERA5"]["filenames"][0]


def resolve_path(fname, data_paths):
    """Mirrors multi_stream_data_sampler.py path resolution."""
    p = Path(fname)
    if p.exists():
        return p
    for base in data_paths:
        c = Path(base) / fname
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find: {fname}")


# ---------------------------------------------------------------------------
# 2 — Constants
# ---------------------------------------------------------------------------

VARS = ["2t", "t_850", "u_850", "v_850"]
T1   = np.datetime64("2020-01-15T12:00:00")
T2   = np.datetime64("2020-04-15T12:00:00")

VAR_LABELS = {
    "2t":    "2 m temperature [K]",
    "t_850": "850 hPa temperature [K]",
    "u_850": "850 hPa U-wind [m/s]",
    "v_850": "850 hPa V-wind [m/s]",
}
CMAPS = {"2t": "RdBu_r", "t_850": "RdBu_r", "u_850": "bwr", "v_850": "bwr"}


# ---------------------------------------------------------------------------
# 3 — Gaussian grid construction
# ---------------------------------------------------------------------------


def build_gaussian_grid(nlat: int) -> dict:
    """Build a regular Gaussian grid with *nlat* latitude rings.

    Returns a dict with:
        lats_1d   – 1D latitudes  (degrees, S→N for meshgrid/plotting)
        lons_1d   – 1D longitudes (degrees, 0→360 exclusive)
        LAT, LON  – 2D meshgrids  (nlat × nlon)
        lons_per_lat – [nlon]*nlat  (for SHT constructor)
        nlon      – number of longitude points
    """
    nlon = 2 * nlat

    # Gaussian co-latitudes (returned as cos(theta) in [-1, 1])
    cos_theta, _ = legendre_gauss_weights(nlat)
    # cos_theta ascending [-1, 1]; arccos gives descending [π, 0]; flip → ascending [0, π]
    theta = np.flip(np.arccos(cos_theta))           # ascending co-lat [0,π] = N→S
    lats_deg = 90.0 - np.degrees(theta)             # geographic lat [90,-90] = N→S
    lats_deg = lats_deg[::-1].copy()                 # flip to S→N for meshgrid

    lons_deg = np.linspace(0, 360, nlon, endpoint=False)

    LAT, LON = np.meshgrid(lats_deg, lons_deg, indexing="ij")

    return {
        "lats_1d": lats_deg,
        "lons_1d": lons_deg,
        "LAT": LAT,
        "LON": LON,
        "lons_per_lat": [nlon] * nlat,
        "nlon": nlon,
        "nlat": nlat,
    }


# ---------------------------------------------------------------------------
# 4 — Load ERA5 + regrid to Gaussian grid
# ---------------------------------------------------------------------------


def load_era5():
    import anemoi.datasets as ad

    era5_path = resolve_path(ERA5_FNAME, DATA_PATHS)
    print(f"ERA5 path: {era5_path}")
    era5 = ad.open_dataset(era5_path)
    print(f"ERA5 shape: {era5.shape}  | {len(era5.variables)} variables")
    return era5


def extract(ds, time_idx, var_names):
    """Return {var: 1D array (n_grid,)} for a given time index."""
    snapshot = ds[time_idx][:, 0, :]
    var_list = list(ds.variables)
    return {v: snapshot[var_list.index(v)] for v in var_names}


def lon360_to_180(lons):
    return np.where(lons > 180, lons - 360, lons)


def regrid_to_gaussian(era5, gg: dict):
    """Interpolate ERA5 fields at T1 & T2 onto the Gaussian grid.

    To handle the dateline, we triplicate points in longitude:
    shifted copies at lon-360 and lon+360, so griddata sees continuous
    coverage around 0°/360°.

    Returns
    -------
    fields : dict  mapping (time_label, var) → 2D array (nlat, nlon)
    """
    # Source coordinates (HEALPix), in [0, 360) convention
    src_lats = era5.latitudes
    src_lons = era5.longitudes % 360.0

    # Triplicate for dateline wrapping
    src_lats_3 = np.concatenate([src_lats, src_lats, src_lats])
    src_lons_3 = np.concatenate([src_lons - 360.0, src_lons, src_lons + 360.0])
    src_pts = np.column_stack([src_lats_3, src_lons_3])

    target_pts = np.column_stack([gg["LAT"].ravel(), gg["LON"].ravel()])

    # Identify time indices
    t1_idx = int(np.where(era5.dates == T1)[0][0])
    t2_idx = int(np.where(era5.dates == T2)[0][0])
    print(f"ERA5 time indices: T1={t1_idx}, T2={t2_idx}")

    raw_t1 = extract(era5, t1_idx, VARS)
    raw_t2 = extract(era5, t2_idx, VARS)

    fields = {}
    for t_label, raw in [("t1", raw_t1), ("t2", raw_t2)]:
        for var in VARS:
            vals = raw[var]
            vals_3 = np.concatenate([vals, vals, vals])
            print(f"  Regridding {t_label} {var} ...")
            interp = griddata(src_pts, vals_3, target_pts, method="linear")
            field_2d = interp.reshape(gg["nlat"], gg["nlon"])

            # Fill any NaN near poles with nearest-neighbor
            if np.isnan(field_2d).any():
                nn = griddata(src_pts, vals_3, target_pts, method="nearest")
                nn_2d = nn.reshape(gg["nlat"], gg["nlon"])
                mask = np.isnan(field_2d)
                field_2d[mask] = nn_2d[mask]

            fields[(t_label, var)] = field_2d

    return fields


# ---------------------------------------------------------------------------
# 5 — SHT-based frequency splitting
# ---------------------------------------------------------------------------


def build_sht_pair(gg: dict, dtype=torch.float64):
    """Create forward and inverse SHT modules for the given Gaussian grid.

    Uses full spectral resolution: lmax = mmax = nlat.  A Gaussian grid with
    nlat latitudes and nlon = 2*nlat longitudes can exactly represent
    harmonics up to degree nlat-1 (the SHT arrays are sized lmax = nlat so
    that index lmax-1 = nlat-1 is included).

    Note: the anemoi-core *loss* functions use nlat//2 ("quadratic"
    truncation) to avoid aliasing in products of fields.  For a pure
    forward-inverse round-trip that is NOT needed.
    """
    lmax = gg["nlat"]
    mmax = gg["nlat"]
    forward_sht = SphericalHarmonicTransform(
        lons_per_lat=gg["lons_per_lat"], lmax=lmax, mmax=mmax
    ).to(dtype=dtype)
    inverse_sht = InverseSphericalHarmonicTransform(
        lons_per_lat=gg["lons_per_lat"], lmax=lmax, mmax=mmax
    ).to(dtype=dtype)
    return forward_sht, inverse_sht, lmax, mmax


def field_to_ring_order(field_2d: np.ndarray) -> np.ndarray:
    """Flatten a (nlat, nlon) array into ring order expected by SHT.

    Our 2D field has row 0 = southernmost (S→N convention from meshgrid).
    The SHT expects ring 0 = northernmost (ascending co-latitude = N→S).
    So we flip rows before ravelling.
    """
    return field_2d[::-1, :].ravel()


def ring_order_to_field(flat: np.ndarray, nlat: int, nlon: int) -> np.ndarray:
    """Reverse of field_to_ring_order: ring-ordered (N→S) flat → (nlat, nlon) S→N."""
    return flat.reshape(nlat, nlon)[::-1, :].copy()


def split_freq_sht(
    field_2d: np.ndarray,
    forward_sht: SphericalHarmonicTransform,
    inverse_sht: InverseSphericalHarmonicTransform,
    lmax: int,
    low_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a (nlat, nlon) field into low-ℓ and high-ℓ components via SHT.

    Parameters
    ----------
    field_2d : (nlat, nlon) array
    forward_sht, inverse_sht : SHT modules
    lmax : int
        Spectral truncation (total wavenumber + 1).
    low_frac : float
        Fraction of total wavenumber range to keep in the low-frequency part.

    Returns
    -------
    low, high : (nlat, nlon) arrays
    """
    nlat, nlon = field_2d.shape
    flat = field_to_ring_order(field_2d)
    x = torch.from_numpy(flat).unsqueeze(0).to(torch.float64)  # (1, grid)

    coeffs = forward_sht(x)  # (1, lmax, mmax), complex

    l_cutoff = max(1, int(low_frac * lmax))

    # Low-frequency: keep ℓ <= l_cutoff
    coeffs_low = coeffs.clone()
    coeffs_low[:, l_cutoff:, :] = 0.0

    # High-frequency: keep ℓ > l_cutoff
    coeffs_high = coeffs.clone()
    coeffs_high[:, :l_cutoff, :] = 0.0

    low_flat = inverse_sht(coeffs_low).squeeze(0).numpy()
    high_flat = inverse_sht(coeffs_high).squeeze(0).numpy()

    low = ring_order_to_field(low_flat, nlat, nlon)
    high = ring_order_to_field(high_flat, nlat, nlon)

    return low, high


# ---------------------------------------------------------------------------
# 6 — Panel mixing (same logic as spectral_mixup.py)
# ---------------------------------------------------------------------------


PANEL_LABELS = {
    "A": "A: ERA5 T1 (round-trip)",
    "B": "B: ERA5 T2 (round-trip)",
    "C": "C: low-ℓ T1 + high-ℓ T2 (same season)",
    "D": "D: low-ℓ T1 + high-ℓ T2 (+3 months)",
}


def make_panels(
    fields: dict,
    var: str,
    forward_sht,
    inverse_sht,
    lmax: int,
    low_frac: float = 0.10,
) -> dict:
    lo_t1, hi_t1 = split_freq_sht(fields[("t1", var)], forward_sht, inverse_sht, lmax, low_frac)
    lo_t2, hi_t2 = split_freq_sht(fields[("t2", var)], forward_sht, inverse_sht, lmax, low_frac)
    return {
        "A": lo_t1 + hi_t1,   # T1 round-trip
        "B": lo_t2 + hi_t2,   # T2 round-trip
        "C": lo_t1 + hi_t2,   # low from winter T1, high from spring T2
        "D": lo_t1 + hi_t2,   # same as C (both are cross-season mixes)
    }


# ---------------------------------------------------------------------------
# 7 — Plotting helpers
# ---------------------------------------------------------------------------


def plot_global(panels, var, gg, low_frac):
    """4-panel Mollweide map of the mixed fields."""
    proj = ccrs.Mollweide()
    fig, axes = plt.subplots(
        2, 2, figsize=(18, 10),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    axes = axes.ravel()

    vmin, vmax = np.nanpercentile(panels["A"], [2, 98])

    for ax, key in zip(axes, ["A", "B", "C", "D"]):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        im = ax.pcolormesh(
            gg["lons_1d"], gg["lats_1d"], panels[key],
            cmap=CMAPS[var], vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), shading="auto", rasterized=True,
        )
        ax.set_title(PANEL_LABELS[key], fontsize=10)

    fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.6, pad=0.03,
                 label=VAR_LABELS[var])
    fig.suptitle(
        f"Global SHT — {VAR_LABELS[var]}  |  low_frac={low_frac}  |  ℓ_cut={max(1, int(low_frac * (gg['nlat'] // 2)))}",
        fontsize=13, fontweight="bold",
    )
    return fig


def plot_europe(panels, var, gg, low_frac):
    """4-panel PlateCarree zoom over Europe for visual comparison with FFT script."""
    proj = ccrs.PlateCarree()
    extent = [-15, 40, 30, 72]

    fig, axes = plt.subplots(
        1, 4, figsize=(22, 5),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    vmin, vmax = np.nanpercentile(panels["A"], [2, 98])
    ims = []
    for ax, key in zip(axes, ["A", "B", "C", "D"]):
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        im = ax.pcolormesh(
            gg["lons_1d"], gg["lats_1d"], panels[key],
            cmap=CMAPS[var], vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), shading="auto", rasterized=True,
        )
        ax.set_title(PANEL_LABELS[key], fontsize=9)
        ims.append(im)

    fig.colorbar(ims[-1], ax=axes, orientation="vertical", shrink=0.8, pad=0.01,
                 label=VAR_LABELS[var])
    fig.suptitle(
        f"Europe zoom (SHT) — {VAR_LABELS[var]}  |  low_frac={low_frac}",
        fontsize=13, fontweight="bold",
    )
    return fig


def plot_diff(panels, var, gg, low_frac):
    """Difference maps: mixed − T2 round-trip."""
    proj = ccrs.Mollweide()

    diffs = {
        "C − B: cross-season mix-up": panels["C"] - panels["B"],
    }
    amax = max(np.nanpercentile(np.abs(v), 98) for v in diffs.values())

    fig, axes = plt.subplots(
        1, 1, figsize=(10, 6),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    if not hasattr(axes, "__iter__"):
        axes = [axes]

    fig.suptitle(
        f"{VAR_LABELS[var]} — difference map (SHT)  |  low_frac={low_frac}",
        fontsize=12, fontweight="bold",
    )
    for ax, (title, data) in zip(axes, diffs.items()):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        im = ax.pcolormesh(
            gg["lons_1d"], gg["lats_1d"], data,
            cmap="bwr", vmin=-amax, vmax=amax,
            transform=ccrs.PlateCarree(), shading="auto", rasterized=True,
        )
        ax.set_title(title, fontsize=10)

    fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.6, pad=0.05,
                 label=f"Δ {VAR_LABELS[var]}")
    return fig


def angular_power_spectrum(coeffs: torch.Tensor, lmax: int):
    """Compute angular power spectrum P(ℓ) = Σ_m |c_ℓ^m|² from SHT coefficients.

    Parameters
    ----------
    coeffs : (1, lmax, mmax) complex tensor
    lmax : int

    Returns
    -------
    ell : 1D array of degrees ℓ
    power : 1D array of power P(ℓ)
    """
    c = coeffs.squeeze(0)  # (lmax, mmax)
    power_lm = torch.abs(c) ** 2
    # Sum over m for each ℓ
    power = power_lm.sum(dim=1).numpy()  # (lmax,)
    ell = np.arange(lmax)
    return ell, power


def plot_spectra(fields, var, forward_sht, inverse_sht, lmax, gg, low_frac=0.10):
    """Angular power spectrum for original and mixed fields."""
    panels = make_panels(fields, var, forward_sht, inverse_sht, lmax, low_frac)

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label, ls, color in [
        ("A", "A: ERA5 T1 (round-trip)", "-", "C0"),
        ("B", "B: ERA5 T2 (round-trip)", "--", "C1"),
        ("C", "C: low-ℓ T1 + high-ℓ T2", "-.", "C2"),
    ]:
        flat = field_to_ring_order(panels[key])
        x = torch.from_numpy(flat).unsqueeze(0).to(torch.float64)
        coeffs = forward_sht(x)
        ell, power = angular_power_spectrum(coeffs, lmax)
        ax.semilogy(ell[1:], power[1:], ls=ls, color=color, label=label)

    # Mark the cutoff
    l_cutoff = max(1, int(low_frac * lmax))
    ax.axvline(l_cutoff, color="gray", ls=":", alpha=0.7, label=f"ℓ_cut = {l_cutoff}")

    ax.set_xlabel("Total wavenumber ℓ")
    ax.set_ylabel("Power P(ℓ) = Σ_m |c_ℓ^m|²")
    ax.set_title(f"Angular power spectrum — {VAR_LABELS[var]}  |  low_frac={low_frac}")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(out_dir: Path, nlat: int = 128):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Gaussian grid: N{nlat // 2}  (nlat={nlat}, nlon={2 * nlat})")

    # Build Gaussian grid
    gg = build_gaussian_grid(nlat)

    # Build SHT pair
    print("Building SHT modules ...")
    forward_sht, inverse_sht, lmax, mmax = build_sht_pair(gg)
    print(f"  lmax={lmax}, mmax={mmax}, n_grid_points={forward_sht.n_grid_points}")

    # Load and regrid ERA5
    era5 = load_era5()
    print("\nRegridding ERA5 → Gaussian grid ...")
    fields = regrid_to_gaussian(era5, gg)

    # -- Round-trip sanity checks -------------------------------------------
    LOW_FRACS = [0.05, 0.10, 0.20]

    # 1) Synthetic test: create a band-limited field via iSHT, roundtrip it.
    #    This should give machine-precision error (~1e-10 for float64).
    print("\n=== Synthetic round-trip (band-limited field) ===")
    torch.manual_seed(42)
    synth_coeffs = torch.randn(1, lmax, mmax, dtype=torch.float64)
    synth_coeffs = torch.complex(synth_coeffs, torch.randn_like(synth_coeffs))
    synth_coeffs[:, :, 0] = synth_coeffs[:, :, 0].real  # m=0 must be real
    # Zero below diagonal (l < m has no meaning)
    for m in range(mmax):
        synth_coeffs[:, :m, m] = 0.0
    synth_field = inverse_sht(synth_coeffs)
    synth_rt = inverse_sht(forward_sht(synth_field))
    err_synth = torch.max(torch.abs(synth_field - synth_rt)).item()
    print(f"  max |iSHT(SHT(synth)) - synth| = {err_synth:.2e}  (expect ~1e-10)")

    # 2) ERA5 field: the truncation error measures how much spectral content
    #    sits beyond the SHT truncation (artifacts from griddata regridding).
    print("\n=== ERA5 truncation diagnostic ===")
    f = fields[("t1", "2t")]
    flat = field_to_ring_order(f)
    x = torch.from_numpy(flat).unsqueeze(0).to(torch.float64)
    x_trunc = inverse_sht(forward_sht(x))
    err_trunc = torch.max(torch.abs(x - x_trunc)).item()
    print(f"  max |iSHT(SHT(field)) - field| = {err_trunc:.2e}  (truncation error)")
    # Double round-trip should be near-exact:
    x_trunc2 = inverse_sht(forward_sht(x_trunc))
    err_double = torch.max(torch.abs(x_trunc - x_trunc2)).item()
    print(f"  max |double roundtrip error|    = {err_double:.2e}  (expect ~1e-10)")

    # 3) Split round-trip: low + high must exactly equal the *truncated* field.
    print("\n=== Split round-trip: low + high = iSHT(SHT(field)) ===")
    f_trunc = ring_order_to_field(x_trunc.squeeze(0).numpy(), gg["nlat"], gg["nlon"])
    for lf in LOW_FRACS:
        lo, hi = split_freq_sht(f, forward_sht, inverse_sht, lmax, lf)
        err_vs_trunc = np.max(np.abs((lo + hi) - f_trunc))
        err_vs_orig = np.max(np.abs((lo + hi) - f))
        print(f"  low_frac={lf}: |low+high - truncated| = {err_vs_trunc:.2e}  "
              f"|low+high - original| = {err_vs_orig:.2e}")

    # -- Generate plots -----------------------------------------------------
    print("\n=== Generating plots ===")

    for var in VARS:
        for lf in LOW_FRACS:
            panels = make_panels(fields, var, forward_sht, inverse_sht, lmax, lf)

            # Global maps
            fig = plot_global(panels, var, gg, lf)
            fname = out_dir / f"global_{var}_lf{int(lf * 100):02d}.png"
            fig.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fname.name}")

            # Europe zoom
            fig = plot_europe(panels, var, gg, lf)
            fname = out_dir / f"europe_{var}_lf{int(lf * 100):02d}.png"
            fig.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fname.name}")

    # Difference maps (default low_frac)
    for var in VARS:
        panels = make_panels(fields, var, forward_sht, inverse_sht, lmax, 0.10)
        fig = plot_diff(panels, var, gg, 0.10)
        fname = out_dir / f"diff_{var}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname.name}")

    # Angular power spectra
    for var in VARS:
        fig = plot_spectra(fields, var, forward_sht, inverse_sht, lmax, gg, 0.10)
        fname = out_dir / f"spectrum_{var}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname.name}")

    print(f"\nDone. All figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=SCRIPT_DIR / "output_sht",
        help="Directory to write PNG figures (default: explore/spectral_mixup/output_sht/)",
    )
    parser.add_argument(
        "--nlat", type=int, default=128,
        help="Number of Gaussian latitudes (default: 128 = N64 grid, ~1.4° resolution)",
    )
    args = parser.parse_args()
    main(args.out_dir, nlat=args.nlat)
