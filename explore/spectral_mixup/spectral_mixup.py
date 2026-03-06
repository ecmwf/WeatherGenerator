"""
Spectral Mix-Up Visual Exploration — script version of spectral_mixup.ipynb.

Usage (from worktree root):
    uv run python explore/spectral_mixup/spectral_mixup.py [--out-dir OUT_DIR]

Figures are saved to OUT_DIR (default: explore/spectral_mixup/output/).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# 1 — Resolve data paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).parent
WORKTREE_ROOT = SCRIPT_DIR / "../.."
WG_PRIVATE   = WORKTREE_ROOT / "../WeatherGenerator-private"

paths_cfg   = yaml.safe_load(open(WG_PRIVATE / "hpc/santis/config/paths.yml"))
DATA_PATHS  = paths_cfg["data_paths"]

era5_cfg    = yaml.safe_load(open(WORKTREE_ROOT / "config/streams/era5_1deg/era5.yml"))
cerra_cfg   = yaml.safe_load(open(WORKTREE_ROOT / "config/streams/cerra_seviri/cerra.yml"))
ERA5_FNAME  = era5_cfg["ERA5"]["filenames"][0]
CERRA_FNAME = cerra_cfg["CERRA"]["filenames"][0]


def resolve_path(fname, data_paths):
    """Mirrors multi_stream_data_sampler.py lines 168-183."""
    p = Path(fname)
    if p.exists():
        return p
    for base in data_paths:
        c = Path(base) / fname
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find: {fname}")


# ---------------------------------------------------------------------------
# 2 — Dataset loading helpers
# ---------------------------------------------------------------------------

def load_datasets():
    import anemoi.datasets as ad

    era5_path  = resolve_path(ERA5_FNAME,  DATA_PATHS)
    cerra_path = resolve_path(CERRA_FNAME, DATA_PATHS)
    print(f"ERA5  path: {era5_path}")
    print(f"CERRA path: {cerra_path}")

    era5  = ad.open_dataset(era5_path)
    cerra = ad.open_dataset(cerra_path)
    print(f"ERA5  shape: {era5.shape}  | {len(era5.variables)} variables")
    print(f"CERRA shape: {cerra.shape} | {len(cerra.variables)} variables")
    return era5, cerra


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
# 3 — Field extraction
# ---------------------------------------------------------------------------

def extract(ds, time_idx, var_names):
    """Return {var: 1D array (n_grid,)} for a given time index."""
    snapshot = ds[time_idx][:, 0, :]        # (n_vars, n_grid)
    var_list = list(ds.variables)
    return {v: snapshot[var_list.index(v)] for v in var_names}


# ---------------------------------------------------------------------------
# 4 — Interpolation onto regular lat-lon grid
# ---------------------------------------------------------------------------

def lon360_to_180(lons):
    return np.where(lons > 180, lons - 360, lons)


def build_target_grid(cerra_lats, cerra_lons_180, nlat=500, nlon=700):
    lat_min, lat_max = cerra_lats.min(), cerra_lats.max()
    lon_min, lon_max = cerra_lons_180.min(), cerra_lons_180.max()
    lat_grid = np.linspace(lat_min, lat_max, nlat)
    lon_grid = np.linspace(lon_min, lon_max, nlon)
    LAT, LON = np.meshgrid(lat_grid, lon_grid, indexing="ij")
    return lat_grid, lon_grid, LAT, LON, lat_min, lat_max, lon_min, lon_max


def domain_mask(lats, lons_180, lat_min, lat_max, lon_min, lon_max, margin=2.0):
    return (
        (lats      >= lat_min - margin) & (lats      <= lat_max + margin) &
        (lons_180  >= lon_min - margin) & (lons_180  <= lon_max + margin)
    )


def to_grid(values_1d, src_pts, LAT, LON):
    return griddata(src_pts, values_1d, (LAT, LON), method="linear")


def interpolate_all(era5, cerra):
    era5_lons_180  = lon360_to_180(era5.longitudes)
    cerra_lons_180 = lon360_to_180(cerra.longitudes)

    lat_grid, lon_grid, LAT, LON, lat_min, lat_max, lon_min, lon_max = \
        build_target_grid(cerra.latitudes, cerra_lons_180)

    print(f"Target domain: lat [{lat_min:.2f}, {lat_max:.2f}] "
          f"lon [{lon_min:.2f}, {lon_max:.2f}]")
    print(f"Target grid: {LAT.shape[0]} × {LAT.shape[1]} = {LAT.size:,} cells")

    e_mask = domain_mask(era5.latitudes,  era5_lons_180,
                         lat_min, lat_max, lon_min, lon_max)
    c_mask = domain_mask(cerra.latitudes, cerra_lons_180,
                         lat_min, lat_max, lon_min, lon_max)
    print(f"ERA5  pts in domain: {e_mask.sum():,}")
    print(f"CERRA pts in domain: {c_mask.sum():,}")

    era5_pts  = (era5.latitudes[e_mask],  era5_lons_180[e_mask])
    cerra_pts = (cerra.latitudes[c_mask], cerra_lons_180[c_mask])

    # Load time slices
    t1_e = np.where(era5.dates  == T1)[0][0]
    t2_e = np.where(era5.dates  == T2)[0][0]
    t1_c = np.where(cerra.dates == T1)[0][0]
    t2_c = np.where(cerra.dates == T2)[0][0]
    print(f"Time indices — ERA5 T1={t1_e} T2={t2_e} | CERRA T1={t1_c} T2={t2_c}")

    era5_t1_raw  = extract(era5,  t1_e, VARS)
    era5_t2_raw  = extract(era5,  t2_e, VARS)
    cerra_t1_raw = extract(cerra, t1_c, VARS)
    cerra_t2_raw = extract(cerra, t2_c, VARS)

    grid = {}
    datasets = [
        ("era5",  "t1", era5_t1_raw,  e_mask, era5_pts),
        ("era5",  "t2", era5_t2_raw,  e_mask, era5_pts),
        ("cerra", "t1", cerra_t1_raw, c_mask, cerra_pts),
        ("cerra", "t2", cerra_t2_raw, c_mask, cerra_pts),
    ]
    for ds_name, t_label, raw, mask, pts in datasets:
        for var in VARS:
            print(f"  Interpolating {ds_name} {t_label} {var} ...")
            grid[(ds_name, t_label, var)] = to_grid(raw[var][mask], pts, LAT, LON)

    return grid, lat_grid, lon_grid


# ---------------------------------------------------------------------------
# 5 — FFT frequency splitting
# ---------------------------------------------------------------------------

def split_freq(field, low_frac=0.10):
    """Round-trip safe split into (low_freq, high_freq). NaN-aware."""
    fill_val = np.nanmean(field)
    f = np.where(np.isnan(field), fill_val, field)

    F  = np.fft.rfft2(f)
    ky = np.fft.fftfreq(field.shape[0])[:, None]
    kx = np.fft.rfftfreq(field.shape[1])[None, :]
    kr = np.sqrt(ky**2 + kx**2)

    thr  = np.quantile(kr, low_frac)
    low  = np.fft.irfft2(F * (kr <= thr), s=field.shape)
    high = np.fft.irfft2(F * (kr >  thr), s=field.shape)

    nan_mask = np.isnan(field)
    low[nan_mask]  = np.nan
    high[nan_mask] = np.nan
    return low, high


# ---------------------------------------------------------------------------
# 6 — Panel mixing
# ---------------------------------------------------------------------------

def make_panels(grid, var, low_frac=0.10):
    lo_era5_t1,  hi_era5_t1  = split_freq(grid[("era5",  "t1", var)], low_frac)
    lo_cerra_t1, hi_cerra_t1 = split_freq(grid[("cerra", "t1", var)], low_frac)
    _,           hi_cerra_t2 = split_freq(grid[("cerra", "t2", var)], low_frac)
    return {
        "A": lo_era5_t1  + hi_era5_t1,
        "B": lo_cerra_t1 + hi_cerra_t1,
        "C": lo_era5_t1  + hi_cerra_t1,
        "D": lo_era5_t1  + hi_cerra_t2,
    }


PANEL_LABELS = {
    "A": "A: ERA5 T1 (round-trip)",
    "B": "B: CERRA T1 (round-trip)",
    "C": "C: low ERA5-T1 + high CERRA-T1",
    "D": "D: low ERA5-T1 + high CERRA-T2 (+3 months)",
}

# ---------------------------------------------------------------------------
# 7 — Plotting helpers
# ---------------------------------------------------------------------------

def _base_axes(panels, var, lon_grid, lat_grid, lon_min, lon_max, lat_min, lat_max):
    proj   = ccrs.PlateCarree()
    extent = [lon_min, lon_max, lat_min, lat_max]
    fig, axes = plt.subplots(
        1, 4, figsize=(22, 5),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    vmin, vmax = np.nanpercentile(panels["B"], [2, 98])
    ims = []
    for ax, key in zip(axes, ["A", "B", "C", "D"]):
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")
        im = ax.pcolormesh(
            lon_grid, lat_grid, panels[key],
            cmap=CMAPS[var], vmin=vmin, vmax=vmax,
            transform=proj, shading="auto", rasterized=True,
        )
        ax.set_title(PANEL_LABELS[key], fontsize=9)
        ims.append(im)
    fig.colorbar(ims[-1], ax=axes, orientation="vertical",
                 shrink=0.8, pad=0.01, label=VAR_LABELS[var])
    return fig


def plot_variable(grid, var, lon_grid, lat_grid,
                  lon_min, lon_max, lat_min, lat_max, low_frac=0.10):
    panels = make_panels(grid, var, low_frac)
    fig = _base_axes(panels, var, lon_grid, lat_grid,
                     lon_min, lon_max, lat_min, lat_max)
    fig.suptitle(f"{VAR_LABELS[var]}  |  low_frac={low_frac}",
                 fontsize=13, fontweight="bold")
    return fig


def plot_diff(grid, var, lon_grid, lat_grid,
              lon_min, lon_max, lat_min, lat_max, low_frac=0.10):
    panels = make_panels(grid, var, low_frac)
    proj   = ccrs.PlateCarree()
    extent = [lon_min, lon_max, lat_min, lat_max]

    diffs = {
        "C − B: same-time mix-up": panels["C"] - panels["B"],
        "D − B: cross-season mix-up (+3 months)": panels["D"] - panels["B"],
    }
    amax = max(np.nanpercentile(np.abs(v), 98) for v in diffs.values())

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    fig.suptitle(
        f"{VAR_LABELS[var]}  —  difference maps  |  low_frac={low_frac}",
        fontsize=12, fontweight="bold",
    )
    ims = []
    for ax, (title, data) in zip(axes, diffs.items()):
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")
        im = ax.pcolormesh(
            lon_grid, lat_grid, data,
            cmap="bwr", vmin=-amax, vmax=amax,
            transform=proj, shading="auto", rasterized=True,
        )
        ax.set_title(title, fontsize=10)
        ims.append(im)
    fig.colorbar(ims[-1], ax=axes, orientation="vertical",
                 shrink=0.8, pad=0.01, label=f"Δ {VAR_LABELS[var]}")
    return fig


def radial_power_spectrum(field):
    f = np.where(np.isnan(field), np.nanmean(field), field)
    F = np.fft.rfft2(f)
    P = np.abs(F) ** 2

    ky = np.fft.fftfreq(field.shape[0])[:, None]
    kx = np.fft.rfftfreq(field.shape[1])[None, :]
    kr = np.sqrt(ky**2 + kx**2)

    n_bins = min(field.shape[0] // 2, field.shape[1] // 2)
    bins   = np.linspace(0, kr.max(), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    power  = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (kr >= bins[i]) & (kr < bins[i + 1])
        if mask.any():
            power[i] = P[mask].mean()
    return centers, power


def plot_spectra(grid, var, low_frac=0.10):
    panels = make_panels(grid, var, low_frac)
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label, ls in [
        ("A", "A: ERA5 T1 (round-trip)",      "-"),
        ("B", "B: CERRA T1 (round-trip)",     "--"),
        ("C", "C: low-ERA5 + high-CERRA T1",  "-."),
        ("D", "D: low-ERA5 + high-CERRA T2",  ":"),
    ]:
        k, p = radial_power_spectrum(panels[key])
        ax.loglog(k[1:], p[1:], ls=ls, label=label)
    ax.set_xlabel("Radial wavenumber [grid units⁻¹]")
    ax.set_ylabel("Mean power")
    ax.set_title(f"Power spectrum — {VAR_LABELS[var]}  |  low_frac={low_frac}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")

    # Load
    era5, cerra = load_datasets()

    # Interpolate (slow)
    grid, lat_grid, lon_grid = interpolate_all(era5, cerra)

    # Bounding box (used by all plot functions)
    lat_min, lat_max = lat_grid[0],  lat_grid[-1]
    lon_min, lon_max = lon_grid[0],  lon_grid[-1]
    kwargs = dict(lon_grid=lon_grid, lat_grid=lat_grid,
                  lon_min=lon_min, lon_max=lon_max,
                  lat_min=lat_min, lat_max=lat_max)

    LOW_FRACS = [0.05, 0.10, 0.20]

    # Round-trip sanity check
    print("\nRound-trip sanity check:")
    for lf in LOW_FRACS:
        f   = grid[("era5", "t1", "2t")]
        lo, hi = split_freq(f, lf)
        err = np.nanmax(np.abs((lo + hi) - f))
        print(f"  low_frac={lf}: max |low+high - original| = {err:.2e}")

    # Panel maps
    for var in VARS:
        for lf in LOW_FRACS:
            fig = plot_variable(grid, var, low_frac=lf, **kwargs)
            fname = out_dir / f"panels_{var}_lf{int(lf*100):02d}.png"
            fig.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {fname.name}")

    # Difference maps
    for var in VARS:
        fig = plot_diff(grid, var, low_frac=0.10, **kwargs)
        fname = out_dir / f"diff_{var}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname.name}")

    # Power spectra
    for var in VARS:
        fig = plot_spectra(grid, var, low_frac=0.10)
        fname = out_dir / f"spectrum_{var}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname.name}")

    print("\nDone. All figures written to", out_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path,
                        default=SCRIPT_DIR / "output",
                        help="Directory to write PNG figures (default: explore/spectral_mixup/output/)")
    args = parser.parse_args()
    main(args.out_dir)
