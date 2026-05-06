"""Visualize self-flow noise augmentation on real ERA5 data.

Loads a single timestep from the ERA5 zarr dataset, applies student (self-flow)
and teacher (uniform) noise, and plots them side-by-side for correctness inspection:
  - Student has two distinct noise levels (masked cells = high noise, rest = low noise)
  - Teacher has uniform low noise everywhere
  - Same eps realization is shared (same seed)
  - On unmasked cells, student == teacher (verified numerically)

Usage:
    uv run python scripts/visualize_self_flow_noise.py
    uv run python scripts/visualize_self_flow_noise.py --t_val 0.9 --s_val 0.1 --noise_rate 0.3
    uv run python scripts/visualize_self_flow_noise.py --channel 5 --save viz.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy_healpix.healpy import ang2pix
from pathlib import Path

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.noise_schedule import (
    CosineNoiseSchedule,
    apply_self_flow_noise,
    apply_uniform_noise,
)
from weathergen.datasets.tokenizer_utils import theta_phi_to_standard_coords


ERA5_PATH = Path("/capstor/store/cscs/userlab/ch17/data/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr")
HEALPIX_LEVEL = 5  # matches default training config


def load_era5_timestep(time_idx: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a single ERA5 timestep and return (coords, data, channel_names).

    Returns unnormalized-looking data (still normalized by dataset stats, as in training).
    """
    tw_handler = TimeWindowHandler(
        t_start=np.datetime64("2020-06-15"),
        t_end=np.datetime64("2020-06-16"),
        t_window_len_hours=np.timedelta64(6, "h"),
        t_window_step_hours=np.timedelta64(6, "h"),
    )

    stream_info = {
        "name": "ERA5",
        "source": None,
        "target": None,
        "source_exclude": ["w_", "skt", "tcw", "cp", "tp"],
        "target_exclude": ["w_", "slor", "sdor", "tcw", "cp", "tp"],
        "geoinfo_channels": [],
    }

    reader = DataReaderAnemoi(
        tw_handler=tw_handler,
        filename=ERA5_PATH,
        stream_info=stream_info,
    )

    rdata = reader.get_source(np.int64(time_idx))
    channel_names = reader.source_channels
    return rdata.coords, rdata.data, channel_names


def make_noise_mask(nside: int, noise_rate: float, seed: int = 42) -> torch.Tensor:
    """Random binary mask over HEALPix cells: True = high noise."""
    npix = 12 * nside**2
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.random(npix) < noise_rate)


def coords_to_cell_ids(coords: np.ndarray, nside: int) -> np.ndarray:
    """Map lat/lon coords to HEALPix cell IDs."""
    thetas, phis = theta_phi_to_standard_coords(coords)
    return ang2pix(nside, thetas, phis, nest=True)


def plot_mollweide(
    ax: plt.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
) -> None:
    """Scatter plot on Mollweide projection."""
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])
    sc = ax.scatter(lon_rad, lat_rad, c=values, s=0.3, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize self-flow noise on real ERA5 data")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to visualize")
    parser.add_argument("--noise_rate", type=float, default=0.8, help="Fraction of cells with high noise")
    parser.add_argument("--s_val", type=float, default=0.3, help="Noise level s (low)")
    parser.add_argument("--t_val", type=float, default=0.7, help="Noise level t (high, masked cells)")
    parser.add_argument("--seed", type=int, default=777, help="Noise seed (shared by student/teacher)")
    parser.add_argument("--save", type=str, default=None, help="Save figure to path instead of showing")
    args = parser.parse_args()

    # Load real data
    print("Loading ERA5 data...")
    coords, data_np, channel_names = load_era5_timestep()
    data = torch.tensor(data_np)
    num_points = data.shape[0]
    print(f"Loaded: {num_points} points, {data.shape[1]} channels")
    print(f"Visualizing channel {args.channel}: '{channel_names[args.channel]}'")

    # Noise schedule
    schedule = CosineNoiseSchedule()
    alpha_s, sigma_s = schedule.alpha_sigma(args.s_val)
    alpha_t, sigma_t = schedule.alpha_sigma(args.t_val)
    print(f"Noise levels: s={args.s_val:.2f} (alpha={alpha_s:.3f}, sigma={sigma_s:.3f}), "
          f"t={args.t_val:.2f} (alpha={alpha_t:.3f}, sigma={sigma_t:.3f})")

    # Build per-point noise mask via HEALPix cells (same as training pipeline)
    nside = 2**HEALPIX_LEVEL
    cell_ids = coords_to_cell_ids(coords, nside)
    noise_mask_cells = make_noise_mask(nside, args.noise_rate)
    point_noise_mask = noise_mask_cells[cell_ids]  # per-point mask

    n_masked = point_noise_mask.sum().item()
    print(f"HEALPix level={HEALPIX_LEVEL}, nside={nside}")
    print(f"Points masked (high noise): {n_masked}/{num_points} ({n_masked/num_points:.1%})")

    # Apply noise (exactly as in the training pipeline)
    student_data = apply_self_flow_noise(
        data, point_noise_mask, alpha_s, sigma_s, alpha_t, sigma_t, args.seed,
    )
    teacher_data = apply_uniform_noise(data, alpha_s, sigma_s, args.seed)

    ch = args.channel
    clean = data[:, ch].numpy()
    student = student_data[:, ch].numpy()
    teacher = teacher_data[:, ch].numpy()

    # Shared color range
    vmin = min(clean.min(), student.min(), teacher.min())
    vmax = max(clean.max(), student.max(), teacher.max())

    # --- Figure 1: Data comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), subplot_kw={"projection": "mollweide"})
    fig.suptitle(
        f"Self-Flow Noise on ERA5 — '{channel_names[ch]}' (s={args.s_val}, t={args.t_val})",
        fontsize=11,
    )

    plot_mollweide(axes[0, 0], coords, clean, "Clean data (model target)", vmin=vmin, vmax=vmax)
    plot_mollweide(axes[0, 1], coords, point_noise_mask.numpy().astype(float),
                   f"Noise mask (rate={args.noise_rate})", vmin=0, vmax=1, cmap="coolwarm")
    plot_mollweide(axes[1, 0], coords, student, "Student input (self-flow noise)", vmin=vmin, vmax=vmax)
    plot_mollweide(axes[1, 1], coords, teacher, "Teacher input (uniform noise at s)", vmin=vmin, vmax=vmax)

    plt.tight_layout()

    # --- Figure 2: Residuals ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8), subplot_kw={"projection": "mollweide"})
    fig2.suptitle("Noise Residuals & Consistency Check", fontsize=11)

    student_residual = student - clean
    teacher_residual = teacher - clean
    res_lim = max(abs(student_residual).max(), abs(teacher_residual).max())

    plot_mollweide(axes2[0, 0], coords, student_residual,
                   "Student residual (noised - clean)", vmin=-res_lim, vmax=res_lim)
    plot_mollweide(axes2[0, 1], coords, teacher_residual,
                   "Teacher residual (noised - clean)", vmin=-res_lim, vmax=res_lim)

    # Consistency: on unmasked points, student == teacher
    mask_np = point_noise_mask.numpy()
    diff_unmasked = np.where(~mask_np, student - teacher, 0.0)
    diff_masked = np.where(mask_np, student - teacher, 0.0)

    plot_mollweide(axes2[1, 0], coords, diff_unmasked,
                   "Student - Teacher (unmasked, should be 0)", cmap="PuOr")
    plot_mollweide(axes2[1, 1], coords, diff_masked,
                   "Student - Teacher (masked, shows noise difference)", cmap="PuOr")

    plt.tight_layout()

    # --- Numerical checks ---
    max_diff_unmasked = np.abs(diff_unmasked).max()
    print(f"\nConsistency checks:")
    print(f"  Max |student - teacher| on UNMASKED points: {max_diff_unmasked:.2e} (should be 0)")
    if max_diff_unmasked < 1e-6:
        print("  PASS: Student and teacher agree on unmasked points (same eps, same alpha_s/sigma_s)")
    else:
        print("  FAIL: Student and teacher DISAGREE on unmasked points!")

    mean_abs_diff_masked = np.abs(diff_masked[mask_np]).mean() if mask_np.any() else 0.0
    print(f"  Mean |student - teacher| on MASKED points: {mean_abs_diff_masked:.4f} (should be > 0)")
    print(f"  VP check s: alpha_s^2 + sigma_s^2 = {alpha_s**2 + sigma_s**2:.6f} (should be 1)")
    print(f"  VP check t: alpha_t^2 + sigma_t^2 = {alpha_t**2 + sigma_t**2:.6f} (should be 1)")

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        residual_path = args.save.replace(".", "_residuals.", 1)
        fig2.savefig(residual_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {args.save} and {residual_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
