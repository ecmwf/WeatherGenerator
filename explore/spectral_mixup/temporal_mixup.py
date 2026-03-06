"""
Temporal Spectral Mix-Up — within-dataset exploration (ERA5, CERRA, SEVIRI).

For each dataset, two snapshots ~3 months apart are split into low- and
high-frequency components via 2-D FFT.  Fine-scale texture is then swapped
between the two time points to expose coherence (or incoherence) between
synoptic-scale flow and mesoscale detail.

Panel convention
----------------
A : T1 round-trip   (2020-01-15 12 UTC — original)
B : T2 round-trip   (2020-04-15 12 UTC — original)
C : low(T1) + high(T2)  — winter background + spring fine-scale
D : low(T2) + high(T1)  — spring background + winter fine-scale

Usage (from worktree root)
--------------------------
    uv run python explore/spectral_mixup/temporal_mixup.py [--out-dir DIR]

Figures are written as PNG to OUT_DIR (default: explore/spectral_mixup/output/).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import yaml
import zarr
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR    = Path(__file__).parent
WORKTREE_ROOT = SCRIPT_DIR / "../.."
WG_PRIVATE    = WORKTREE_ROOT / "../WeatherGenerator-private"

_paths_cfg  = yaml.safe_load(open(WG_PRIVATE / "hpc/santis/config/paths.yml"))
DATA_PATHS  = _paths_cfg["data_paths"]

_era5_cfg   = yaml.safe_load(open(WORKTREE_ROOT / "config/streams/era5_1deg/era5.yml"))
_cerra_cfg  = yaml.safe_load(open(WORKTREE_ROOT / "config/streams/cerra_seviri/cerra.yml"))
_seviri_cfg = yaml.safe_load(open(WORKTREE_ROOT / "config/streams/cerra_seviri/seviri.yml"))


def resolve_path(fname, data_paths):
    p = Path(fname)
    if p.exists():
        return p
    for base in data_paths:
        c = Path(base) / fname
        if c.exists():
            return c
    raise FileNotFoundError(fname)


ERA5_PATH  = resolve_path(_era5_cfg["ERA5"]["filenames"][0],   DATA_PATHS)
CERRA_PATH = resolve_path(_cerra_cfg["CERRA"]["filenames"][0], DATA_PATHS)
try:
    SEVIRI_PATH = resolve_path(_seviri_cfg["SEVIRI"]["filenames"][0], DATA_PATHS)
except FileNotFoundError:
    SEVIRI_PATH = resolve_path("observations-file-2014-2024-seviri-o256-v1.zarr", DATA_PATHS)

# ---------------------------------------------------------------------------
# Time points
# ---------------------------------------------------------------------------

T1 = np.datetime64("2020-01-15T12:00:00")
T2 = np.datetime64("2020-04-15T12:00:00")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

ERA5_VARS  = ["2t", "t_850"]
CERRA_VARS = ["2t", "t_850"]
SEVIRI_BANDS = ["obsvalue_rawbt_ir_108", "obsvalue_rawbt_wv_062"]

VAR_LABELS = {
    "2t":                       "2 m temperature [K]",
    "t_850":                    "850 hPa temperature [K]",
    "obsvalue_rawbt_ir_108":    "IR 10.8 µm BT [K]",
    "obsvalue_rawbt_wv_062":    "WV 6.2 µm BT [K]",
}
CMAPS = {
    "2t":                       "RdBu_r",
    "t_850":                    "RdBu_r",
    "obsvalue_rawbt_ir_108":    "RdBu_r",
    "obsvalue_rawbt_wv_062":    "PuOr",
}

# Spectral band sweep.
# Fractions are linear fractions of the *log-wavenumber* range [log(kr_min), log(kr_max)],
# so each band covers an equal multiplicative (octave-like) interval of wavenumbers.
# At 500×700 grid over the CERRA domain the six bands span roughly:
#   0–17 %  → ~7300–2900 km  (planetary / synoptic)
#   17–33 % → ~2900–1200 km  (synoptic)
#   33–50 % → ~1200–480 km   (meso-α)
#   50–67 % → ~480–190 km    (meso-β)
#   67–83 % → ~190–75 km     (meso-γ / convective)
#   83–100% → ~75–40 km      (near-Nyquist / fine-scale)
SPEC_BANDS = [
    (0.000, 0.167, "log 0–17 %"),
    (0.167, 0.333, "log 17–33 %"),
    (0.333, 0.500, "log 33–50 %"),
    (0.500, 0.667, "log 50–67 %"),
    (0.667, 0.833, "log 67–83 %"),
    (0.833, 1.000, "log 83–100 %"),
]

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def lon360_to_180(lons):
    return np.where(lons > 180, lons - 360, lons)


def build_grid(cerra):
    lons_180 = lon360_to_180(cerra.longitudes)
    lat_min, lat_max = float(cerra.latitudes.min()), float(cerra.latitudes.max())
    lon_min, lon_max = float(lons_180.min()),        float(lons_180.max())
    nlat, nlon = 500, 700
    lat_grid = np.linspace(lat_min, lat_max, nlat)
    lon_grid = np.linspace(lon_min, lon_max, nlon)
    LAT, LON = np.meshgrid(lat_grid, lon_grid, indexing="ij")
    return lat_grid, lon_grid, LAT, LON, lat_min, lat_max, lon_min, lon_max


def domain_mask(lats, lons_180, lat_min, lat_max, lon_min, lon_max, margin=2.0):
    return (
        (lats      >= lat_min - margin) & (lats      <= lat_max + margin) &
        (lons_180  >= lon_min - margin) & (lons_180  <= lon_max + margin)
    )


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def extract_anemoi(ds, time_idx, var_names):
    snap = ds[time_idx][:, 0, :]
    vlist = list(ds.variables)
    return {v: snap[vlist.index(v)] for v in var_names}


def seviri_rows(hrly_idx, t, base=np.datetime64("1970-01-01T00:00:00")):
    h = int((t - base) / np.timedelta64(1, "h"))
    return int(hrly_idx[h]), int(hrly_idx[h + 1])


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def to_grid(vals, src_lats, src_lons, LAT, LON, method="linear"):
    return griddata((src_lats, src_lons), vals, (LAT, LON), method=method)


def interpolate_all(LAT, LON, lat_min, lat_max, lon_min, lon_max):
    import anemoi.datasets as ad

    era5  = ad.open_dataset(ERA5_PATH)
    cerra = ad.open_dataset(CERRA_PATH)
    seviri_store = zarr.open(SEVIRI_PATH)
    sev_data     = seviri_store["data"]
    sev_cols     = sev_data.attrs["colnames"]
    sev_hrly     = seviri_store["idx_197001010000_1"]

    # Time indices
    era5_tidx  = {t: int(np.where(era5.dates  == T)[0][0]) for t, T in [("t1", T1), ("t2", T2)]}
    cerra_tidx = {t: int(np.where(cerra.dates == T)[0][0]) for t, T in [("t1", T1), ("t2", T2)]}
    sev_row    = {t: seviri_rows(sev_hrly, T)               for t, T in [("t1", T1), ("t2", T2)]}

    era5_lons_180  = lon360_to_180(era5.longitudes)
    cerra_lons_180 = lon360_to_180(cerra.longitudes)

    e_mask = domain_mask(era5.latitudes,  era5_lons_180,  lat_min, lat_max, lon_min, lon_max)
    c_mask = domain_mask(cerra.latitudes, cerra_lons_180, lat_min, lat_max, lon_min, lon_max)

    e_pts = (era5.latitudes[e_mask],  era5_lons_180[e_mask])
    c_pts = (cerra.latitudes[c_mask], cerra_lons_180[c_mask])

    def _interp(vals, pts, method="linear"):
        return to_grid(vals, pts[0], pts[1], LAT, LON, method=method)

    grid = {}

    print("ERA5 ...")
    for t_label, t_idx in era5_tidx.items():
        raw = extract_anemoi(era5, t_idx, ERA5_VARS)
        for var in ERA5_VARS:
            print(f"  era5 {t_label} {var}")
            grid[("era5", t_label, var)] = _interp(raw[var][e_mask], e_pts)

    print("CERRA ...")
    for t_label, t_idx in cerra_tidx.items():
        raw = extract_anemoi(cerra, t_idx, CERRA_VARS)
        for var in CERRA_VARS:
            print(f"  cerra {t_label} {var}")
            grid[("cerra", t_label, var)] = _interp(raw[var][c_mask], c_pts)

    print("SEVIRI (nearest-neighbour) ...")
    for t_label, (r0, r1) in sev_row.items():
        obs  = sev_data[r0:r1]
        lats = obs[:, 0]
        lons = obs[:, 1]   # already −180/+180
        smask = domain_mask(lats, lons, lat_min, lat_max, lon_min, lon_max)
        s_lats = lats[smask]
        s_lons = lons[smask]
        print(f"  seviri {t_label}: {smask.sum():,} obs in domain")
        for band in SEVIRI_BANDS:
            col = sev_cols.index(band)
            vals = obs[smask, col]
            print(f"    {band}")
            grid[("seviri", t_label, band)] = to_grid(vals, s_lats, s_lons,
                                                       LAT, LON, method="nearest")

    return grid


# ---------------------------------------------------------------------------
# FFT splitting
# ---------------------------------------------------------------------------

def _log_thr(kr, frac):
    """
    Wavenumber threshold at linear fraction `frac` of the log-wavenumber range.

    Maps frac ∈ [0, 1]  →  threshold ∈ [kr_min, kr_max]  on a log scale:
        thr = exp(log(kr_min) + frac * (log(kr_max) − log(kr_min)))
            = kr_min * (kr_max / kr_min) ** frac

    frac = 0  →  smallest non-zero wavenumber (largest resolved scale)
    frac = 0.5 →  geometric mean of kr_min and kr_max
    frac = 1  →  largest wavenumber (Nyquist scale)
    """
    kr_pos  = kr[kr > 0]
    log_min = np.log(kr_pos.min())
    log_max = np.log(kr_pos.max())
    return np.exp(log_min + frac * (log_max - log_min))


def split_freq(field, low_frac):
    """Round-trip safe, NaN-aware 2-D spectral split.

    `low_frac` is a linear fraction of the log-wavenumber range (see `_log_thr`).
    """
    fill = np.nanmean(field)
    f = np.where(np.isnan(field), fill, field)
    F  = np.fft.rfft2(f)
    ky = np.fft.fftfreq(field.shape[0])[:, None]
    kx = np.fft.rfftfreq(field.shape[1])[None, :]
    kr = np.sqrt(ky**2 + kx**2)
    thr  = _log_thr(kr, low_frac)
    low  = np.fft.irfft2(F * (kr <= thr), s=field.shape)
    high = np.fft.irfft2(F * (kr >  thr), s=field.shape)
    nm   = np.isnan(field)
    low[nm]  = np.nan
    high[nm] = np.nan
    return low, high


def make_panels(f_t1, f_t2, low_frac):
    lo1, hi1 = split_freq(f_t1, low_frac)
    lo2, hi2 = split_freq(f_t2, low_frac)
    return {
        "A": lo1 + hi1,
        "B": lo2 + hi2,
        "C": lo1 + hi2,
        "D": lo2 + hi1,
    }


def _band_mask(kr, q_lo, q_hi):
    """Boolean mask for rfft2 coefficients in log-wavenumber fraction band [q_lo, q_hi].

    q_hi = 1.0 uses np.inf as the upper bound so that floating-point rounding in
    exp(log(kr_max)) cannot accidentally leave the highest-frequency coefficients
    uncovered — guaranteeing exact round-trip (sum of all bands = original field).
    """
    thr_hi = _log_thr(kr, q_hi) if q_hi < 1.0 else np.inf
    if q_lo == 0.0:
        return kr <= thr_hi          # include DC component (kr = 0)
    thr_lo = _log_thr(kr, q_lo)
    return (kr > thr_lo) & (kr <= thr_hi)


def extract_band(field, q_lo, q_hi):
    """
    Return the spatial content of `field` in the radial wavenumber band
    [q_lo, q_hi] (log-wavenumber fraction — see `_log_thr`).

    Round-trip property: sum over all non-overlapping bands = original field.
    NaN boundary values are filled with the field mean before the FFT and
    restored afterwards.
    """
    fill = np.nanmean(field)
    f = np.where(np.isnan(field), fill, field)
    F  = np.fft.rfft2(f)
    ky = np.fft.fftfreq(field.shape[0])[:, None]
    kx = np.fft.rfftfreq(field.shape[1])[None, :]
    kr = np.sqrt(ky**2 + kx**2)
    band = np.fft.irfft2(F * _band_mask(kr, q_lo, q_hi), s=field.shape)
    nm = np.isnan(field)
    band[nm] = np.nan
    return band


# ---------------------------------------------------------------------------
# Panel title strings (same as notebook)
# ---------------------------------------------------------------------------

PANEL_TITLES = {
    "A": "A  T1 (round-trip)\n2020-01-15 12 UTC — original",
    "B": "B  T2 (round-trip)\n2020-04-15 12 UTC — original",
    "C": "C  low(T1) + high(T2)\nwinter background + spring fine-scale",
    "D": "D  low(T2) + high(T1)\nspring background + winter fine-scale",
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_geo_axes(nrows, ncols, figsize):
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             subplot_kw={"projection": proj},
                             constrained_layout=True)
    return fig, np.atleast_1d(axes).ravel(), proj


def _add_geo(ax, extent, proj):
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")


def plot_panels(panels, title, var_label, cmap,
                lat_grid, lon_grid, extent, low_frac=0.10,
                figsize=(22, 5)):
    fig, axes, proj = _make_geo_axes(1, 4, figsize)
    fig.suptitle(f"{title}   |   low_frac = {low_frac}",
                 fontsize=12, fontweight="bold")

    ref = np.concatenate([panels["A"].ravel(), panels["B"].ravel()])
    ref = ref[np.isfinite(ref)]
    vmin, vmax = np.percentile(ref, [2, 98])

    ims = []
    for ax, key in zip(axes, ["A", "B", "C", "D"]):
        _add_geo(ax, extent, proj)
        im = ax.pcolormesh(lon_grid, lat_grid, panels[key],
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=proj, shading="auto", rasterized=True)
        ax.set_title(PANEL_TITLES[key], fontsize=8, loc="left")
        ims.append(im)
    fig.colorbar(ims[-1], ax=list(axes), orientation="vertical",
                 shrink=0.8, pad=0.01, label=var_label)
    return fig


def plot_diff(panels, title, var_label,
              lat_grid, lon_grid, extent, low_frac,
              figsize=(14, 5)):
    diffs = {
        "C − A\n(spring fine-scale on winter bg)": panels["C"] - panels["A"],
        "D − B\n(winter fine-scale on spring bg)": panels["D"] - panels["B"],
    }
    amax = max(np.nanpercentile(np.abs(v), 98) for v in diffs.values())

    fig, axes, proj = _make_geo_axes(1, 2, figsize)
    fig.suptitle(f"{title}  —  difference maps   |   low_frac = {low_frac}",
                 fontsize=11, fontweight="bold")
    ims = []
    for ax, (ttl, data) in zip(axes, diffs.items()):
        _add_geo(ax, extent, proj)
        im = ax.pcolormesh(lon_grid, lat_grid, data,
                           cmap="bwr", vmin=-amax, vmax=amax,
                           transform=proj, shading="auto", rasterized=True)
        ax.set_title(ttl, fontsize=8, loc="left")
        ims.append(im)
    fig.colorbar(ims[-1], ax=list(axes), orientation="vertical",
                 shrink=0.8, pad=0.01, label=f"Δ {var_label}")
    return fig


def plot_spectra(panel_sets, titles, lat_grid, figsize=(18, 4)):
    """panel_sets: list of panels dicts, titles: matching list of strings."""
    def _rps(field):
        f = np.where(np.isnan(field), np.nanmean(field), field)
        P = np.abs(np.fft.rfft2(f)) ** 2
        ky = np.fft.fftfreq(field.shape[0])[:, None]
        kx = np.fft.rfftfreq(field.shape[1])[None, :]
        kr = np.sqrt(ky**2 + kx**2)
        nb = min(field.shape[0] // 2, field.shape[1] // 2)
        bins = np.linspace(0, kr.max(), nb + 1)
        ctrs = 0.5 * (bins[:-1] + bins[1:])
        pw = np.array([P[(kr >= bins[i]) & (kr < bins[i+1])].mean()
                       if ((kr >= bins[i]) & (kr < bins[i+1])).any() else 0
                       for i in range(nb)])
        return ctrs, pw

    fig, axes = plt.subplots(1, len(panel_sets), figsize=figsize,
                             constrained_layout=True)
    fig.suptitle("Radially-averaged 2D power spectra  |  low_frac = 0.10",
                 fontsize=12, fontweight="bold")
    for ax, panels, title in zip(np.atleast_1d(axes), panel_sets, titles):
        for key, ls, lbl in [("A", "-",  "A: T1 original"),
                              ("B", "--", "B: T2 original"),
                              ("C", "-.", "C: low(T1)+high(T2)"),
                              ("D", ":",  "D: low(T2)+high(T1)")]:
            k, p = _rps(panels[key])
            ax.loglog(k[1:], p[1:], ls=ls, label=lbl)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Radial wavenumber [grid units⁻¹]")
        ax.set_ylabel("Mean power")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
    return fig


def plot_band_content(f_t1, f_t2, title, bands,
                      lat_grid, lon_grid, extent, cmap, var_label):
    """
    2 rows × N-band columns.  Each cell shows the isolated content of that
    spectral band: what spatial patterns live at those frequencies?

    - Row 0 (T1, Jan): winter spectral structure
    - Row 1 (T2, Apr): spring spectral structure

    Colorbar is symmetric and shared per column so T1 and T2 amplitudes are
    directly comparable.  The first band (includes DC) uses the field range
    instead of a symmetric scale because it carries the mean offset.
    """
    nb = len(bands)
    fields = [f_t1, f_t2]
    t_labels = ["T1  2020-01-15 12 UTC", "T2  2020-04-15 12 UTC"]
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2, nb, figsize=(4.5 * nb, 8),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    fig.suptitle(
        f"{title}\nIsolated spectral band content  "
        f"(quantile fractions of radial wavenumber distribution)",
        fontsize=11, fontweight="bold",
    )

    for col, (q_lo, q_hi, band_label) in enumerate(bands):
        # Extract bands for both time points
        b = [extract_band(f, q_lo, q_hi) for f in fields]

        # Colorbar range: symmetric for all bands (DC offset lives in band 0
        # but symmetric still works — it centres on zero which is fine for
        # anomalies; for the first band the offset is visible as a warm/cold bias)
        amax = max(np.nanpercentile(np.abs(arr[np.isfinite(arr)]), 98)
                   for arr in b)
        vmin, vmax = -amax, amax

        for row, (arr, t_lbl) in enumerate(zip(b, t_labels)):
            ax = axes[row, col]
            _add_geo(ax, extent, proj)
            im = ax.pcolormesh(
                lon_grid, lat_grid, arr,
                cmap=cmap, vmin=vmin, vmax=vmax,
                transform=proj, shading="auto", rasterized=True,
            )
            if col == 0:
                ax.set_ylabel(t_lbl, fontsize=8)
            if row == 0:
                ax.set_title(f"Band  {band_label}", fontsize=8)
        fig.colorbar(im, ax=axes[:, col], orientation="vertical",
                     shrink=0.7, pad=0.02, label=var_label if col == nb - 1 else "")

    return fig


def plot_band_masked(f_t1, f_t2, title, bands,
                     lat_grid, lon_grid, extent, cmap, var_label):
    """
    2 rows × (1 + N-band) columns.

    - Column 0: original field (reference)
    - Columns 1…N: original field with that band *removed* (zeroed out in
      Fourier space), i.e. field − band_content

    All panels share the same colour scale (2nd–98th percentile of the
    originals) so it is easy to see how much each band contributes to the
    total field appearance.  Regions that change strongly when a band is
    removed are the ones most dominated by that frequency range.
    """
    nb = len(bands)
    fields = [f_t1, f_t2]
    t_labels = ["T1  2020-01-15 12 UTC", "T2  2020-04-15 12 UTC"]
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2, 1 + nb, figsize=(4.5 * (1 + nb), 8),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    fig.suptitle(
        f"{title}\nField with each spectral band removed  "
        f"(field − band_content; same colour scale as original)",
        fontsize=11, fontweight="bold",
    )

    # Shared colour scale from the originals
    ref = np.concatenate([f_t1[np.isfinite(f_t1)], f_t2[np.isfinite(f_t2)]])
    vmin, vmax = np.percentile(ref, [2, 98])

    for row, (f, t_lbl) in enumerate(zip(fields, t_labels)):
        # Col 0: original
        ax = axes[row, 0]
        _add_geo(ax, extent, proj)
        im0 = ax.pcolormesh(lon_grid, lat_grid, f,
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            transform=proj, shading="auto", rasterized=True)
        ax.set_title("Original" if row == 0 else "", fontsize=8)
        if row == 0:
            ax.set_title("Original", fontsize=8)
        ax.set_ylabel(t_lbl, fontsize=8)

        # Cols 1…N: field minus each band
        for col, (q_lo, q_hi, band_label) in enumerate(bands, start=1):
            band = extract_band(f, q_lo, q_hi)
            masked = f - band
            ax = axes[row, col]
            _add_geo(ax, extent, proj)
            ax.pcolormesh(lon_grid, lat_grid, masked,
                          cmap=cmap, vmin=vmin, vmax=vmax,
                          transform=proj, shading="auto", rasterized=True)
            if row == 0:
                ax.set_title(f"Remove  {band_label}", fontsize=8)

    fig.colorbar(im0, ax=axes, orientation="vertical",
                 shrink=0.6, pad=0.01, label=var_label)
    return fig


def plot_band_swap(f_t1, f_t2, title, bands,
                   lat_grid, lon_grid, extent, cmap, var_label):
    """
    3 rows × N-band columns.

    For each band [q_lo, q_hi], show what happens when *only that band* is
    swapped from T2 into T1 (everything outside the band stays from T1):

        swapped = T1 − band_content(T1) + band_content(T2)

    Row layout
    ----------
    Row 0 : T1 original  (repeated each column as reference)
    Row 1 : T1 with only this band from T2  (the temporal chimera)
    Row 2 : Injection = band_content(T2) − band_content(T1)
             (what was added; symmetric diverging scale)

    This isolates the contribution of each frequency range to the overall
    temporal mismatch seen in panels C and D of the main mix-up figures.
    """
    nb = len(bands)
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        3, nb, figsize=(4.5 * nb, 12),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    fig.suptitle(
        f"{title}\nPer-band temporal swap: only this band comes from T2, "
        f"all other bands from T1\n"
        f"Row 0: T1 original  |  Row 1: swapped field  |  "
        f"Row 2: injection (band_T2 − band_T1)",
        fontsize=10, fontweight="bold",
    )

    ref = f_t1[np.isfinite(f_t1)]
    vmin_f, vmax_f = np.percentile(ref, [2, 98])

    row_labels = [
        "T1 original\n(reference)",
        "T1 with only\nthis band from T2",
        "Injection\n(band_T2 − band_T1)",
    ]

    for col, (q_lo, q_hi, band_label) in enumerate(bands):
        b1 = extract_band(f_t1, q_lo, q_hi)
        b2 = extract_band(f_t2, q_lo, q_hi)
        swapped   = f_t1 - b1 + b2
        injection = b2 - b1
        inj_amax  = np.nanpercentile(np.abs(injection[np.isfinite(injection)]), 98)

        panels_data = [
            (f_t1,     cmap,   vmin_f,      vmax_f),
            (swapped,  cmap,   vmin_f,      vmax_f),
            (injection, "bwr", -inj_amax,   inj_amax),
        ]

        for row, (data, cm, vmin, vmax) in enumerate(panels_data):
            ax = axes[row, col]
            _add_geo(ax, extent, proj)
            im = ax.pcolormesh(
                lon_grid, lat_grid, data,
                cmap=cm, vmin=vmin, vmax=vmax,
                transform=proj, shading="auto", rasterized=True,
            )
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8)
            if row == 0:
                ax.set_title(f"Band  {band_label}", fontsize=8)
            # Per-cell colorbar only for row 2 (injection has its own scale)
            if row == 2:
                fig.colorbar(im, ax=ax, orientation="vertical",
                             shrink=0.8, pad=0.02,
                             label=f"Δ {var_label}" if col == nb - 1 else "")

    # Shared colorbar for rows 0 and 1
    im_ref = axes[0, -1].collections[0]
    fig.colorbar(im_ref, ax=axes[:2, :], orientation="vertical",
                 shrink=0.5, pad=0.01, label=var_label)
    return fig


def plot_cross_dataset(grid, lat_grid, lon_grid, extent, low_frac=0.10,
                        figsize=(18, 5)):
    """Panel C side-by-side across ERA5, CERRA, SEVIRI."""
    configs = [
        ("era5",   "2t",                  "ERA5 — 2 m temperature [K]",     "RdBu_r"),
        ("cerra",  "2t",                  "CERRA — 2 m temperature [K]",    "RdBu_r"),
        ("seviri", "obsvalue_rawbt_ir_108","SEVIRI — IR 10.8 µm BT [K]",    "RdBu_r"),
    ]
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             subplot_kw={"projection": proj},
                             constrained_layout=True)
    fig.suptitle(
        "Panel C  [low(T1) + high(T2)]  across datasets\n"
        "winter background (2020-01-15) + spring fine-scale (2020-04-15)",
        fontsize=12, fontweight="bold"
    )
    for ax, (ds, var, title, cmap) in zip(axes, configs):
        panels = make_panels(grid[(ds, "t1", var)], grid[(ds, "t2", var)],
                             low_frac=low_frac)
        vmin, vmax = np.nanpercentile(panels["A"][np.isfinite(panels["A"])], [2, 98])
        _add_geo(ax, extent, proj)
        im = ax.pcolormesh(lon_grid, lat_grid, panels["C"],
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=proj, shading="auto", rasterized=True)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.9, pad=0.02)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir.resolve()}\n")

    import anemoi.datasets as ad
    cerra_ds = ad.open_dataset(CERRA_PATH)
    lat_grid, lon_grid, LAT, LON, lat_min, lat_max, lon_min, lon_max = \
        build_grid(cerra_ds)
    del cerra_ds

    extent = [lon_min, lon_max, lat_min, lat_max]

    # Interpolate everything
    grid = interpolate_all(LAT, LON, lat_min, lat_max, lon_min, lon_max)

    # Round-trip sanity
    print("\nRound-trip sanity (ERA5 2t):")
    for lf in [0.05, 0.10, 0.20, 0.40, 0.60]:
        f = grid[("era5", "t1", "2t")]
        lo, hi = split_freq(f, lf)
        print(f"  low_frac={lf}: max |low+high-original| = {np.nanmax(np.abs(lo+hi-f)):.2e}")

    LOW_FRAC = 0.10
    LOW_FRACS = [0.05, 0.10, 0.20, 0.40, 0.60]

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {p.name}")

    # ── ERA5 ──────────────────────────────────────────────────────────────
    print("\nPlotting ERA5 ...")
    for var in ERA5_VARS:
        label = VAR_LABELS[var]
        cmap  = CMAPS[var]
        for lf in LOW_FRACS:
            panels = make_panels(grid[("era5", "t1", var)],
                                 grid[("era5", "t2", var)], low_frac=lf)
            save(plot_panels(panels, f"ERA5 — {label}", label, cmap,
                             lat_grid, lon_grid, extent, low_frac=lf),
                 f"era5_panels_{var}_lf{int(lf*100):02d}.png")
        # diff at default lf
        panels = make_panels(grid[("era5", "t1", var)],
                             grid[("era5", "t2", var)], low_frac=LOW_FRAC)
        save(plot_diff(panels, f"ERA5 — {label}", label,
                       lat_grid, lon_grid, extent, low_frac=LOW_FRAC),
             f"era5_diff_{var}.png")

    # ── CERRA ─────────────────────────────────────────────────────────────
    print("\nPlotting CERRA ...")
    for var in CERRA_VARS:
        label = VAR_LABELS[var]
        cmap  = CMAPS[var]
        for lf in LOW_FRACS:
            panels = make_panels(grid[("cerra", "t1", var)],
                                 grid[("cerra", "t2", var)], low_frac=lf)
            save(plot_panels(panels, f"CERRA — {label}", label, cmap,
                             lat_grid, lon_grid, extent, low_frac=lf),
                 f"cerra_panels_{var}_lf{int(lf*100):02d}.png")
        panels = make_panels(grid[("cerra", "t1", var)],
                             grid[("cerra", "t2", var)], low_frac=LOW_FRAC)
        save(plot_diff(panels, f"CERRA — {label}", label,
                       lat_grid, lon_grid, extent, low_frac=LOW_FRAC),
             f"cerra_diff_{var}.png")

    # ── SEVIRI ────────────────────────────────────────────────────────────
    print("\nPlotting SEVIRI ...")
    for band in SEVIRI_BANDS:
        label = VAR_LABELS[band]
        cmap  = CMAPS[band]
        for lf in LOW_FRACS:
            panels = make_panels(grid[("seviri", "t1", band)],
                                 grid[("seviri", "t2", band)], low_frac=lf)
            save(plot_panels(panels, f"SEVIRI — {label}", label, cmap,
                             lat_grid, lon_grid, extent, low_frac=lf),
                 f"seviri_panels_{band.split('_')[-1]}_lf{int(lf*100):02d}.png")
        panels = make_panels(grid[("seviri", "t1", band)],
                             grid[("seviri", "t2", band)], low_frac=LOW_FRAC)
        save(plot_diff(panels, f"SEVIRI — {label}", label,
                       lat_grid, lon_grid, extent, low_frac=LOW_FRAC),
             f"seviri_diff_{band.split('_')[-1]}.png")

    # ── Spectral band sweep ───────────────────────────────────────────────
    # Three figures per (dataset, variable):
    #   *_band_content : isolated content of each spectral band
    #   *_band_masked  : field with each band removed
    #   *_band_swap    : swap only one band at a time from T2 into T1
    print("\nPlotting spectral band sweep ...")
    band_sweep_configs = [
        ("era5",   "2t",                   "ERA5 — 2 m temperature"),
        ("cerra",  "2t",                   "CERRA — 2 m temperature"),
        ("seviri", "obsvalue_rawbt_ir_108", "SEVIRI — IR 10.8 µm BT"),
    ]
    for ds, var, ds_title in band_sweep_configs:
        f_t1  = grid[(ds, "t1", var)]
        f_t2  = grid[(ds, "t2", var)]
        label = VAR_LABELS[var]
        cm    = CMAPS[var]
        slug  = f"{ds}_{var.split('_')[-1]}"   # e.g. era5_2t, cerra_2t, seviri_108

        save(plot_band_content(f_t1, f_t2, ds_title, SPEC_BANDS,
                               lat_grid, lon_grid, extent, cm, label),
             f"{slug}_band_content.png")

        save(plot_band_masked(f_t1, f_t2, ds_title, SPEC_BANDS,
                              lat_grid, lon_grid, extent, cm, label),
             f"{slug}_band_masked.png")

        save(plot_band_swap(f_t1, f_t2, ds_title, SPEC_BANDS,
                            lat_grid, lon_grid, extent, cm, label),
             f"{slug}_band_swap.png")

    # ── Cross-dataset panel C ──────────────────────────────────────────────
    print("\nPlotting cross-dataset comparison ...")
    save(plot_cross_dataset(grid, lat_grid, lon_grid, extent, low_frac=LOW_FRAC),
         "cross_dataset_panel_C.png")

    # ── Power spectra ─────────────────────────────────────────────────────
    print("\nPlotting power spectra ...")
    spectrum_configs = [
        ("era5",   "2t",                  "ERA5 — 2 m temperature"),
        ("cerra",  "2t",                  "CERRA — 2 m temperature"),
        ("seviri", "obsvalue_rawbt_ir_108", "SEVIRI — IR 10.8 µm BT"),
    ]
    panel_sets = [make_panels(grid[(ds, "t1", var)], grid[(ds, "t2", var)],
                              low_frac=LOW_FRAC)
                  for ds, var, _ in spectrum_configs]
    titles = [t for _, _, t in spectrum_configs]
    save(plot_spectra(panel_sets, titles, lat_grid), "power_spectra.png")

    print(f"\nAll figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=SCRIPT_DIR / "output",
        help="Output directory for PNG figures (default: explore/spectral_mixup/output/)",
    )
    args = parser.parse_args()
    main(args.out_dir)
