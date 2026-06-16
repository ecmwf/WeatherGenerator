# ruff: noqa: T201
# ruff: noqa: TID251

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
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy_healpix.healpy import ang2pix
from numpy.typing import NDArray

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.noise_schedule import (
    DEFAULT_VARIABLE_COVARIANCE_EIGENVALUES,
    CosineNoiseSchedule,
    LinearInformationNoiseSchedule,
    NoiseSchedule,
    VariableCovarianceLinearInformationNoiseSchedule,
    apply_self_flow_noise,
    apply_uniform_noise,
    covariance_information_from_sigma,
    sample_noise_coordinates,
)
from weathergen.datasets.tokenizer_utils import theta_phi_to_standard_coords

logger = logging.getLogger(__name__)

ERA5_PATH = Path(
    "/capstor/store/cscs/userlab/ch17/data/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"
)
HEALPIX_LEVEL = 5  # matches default training config
VARIABLE_GROUPS = ("t", "q", "z", "u", "v")


def load_era5_timestep(time_idx: int = 0) -> tuple[NDArray, NDArray, list[str]]:
    """Load a single ERA5 timestep and return (coords, data, channel_names).

    Returns normalized model-space data, matching the training pipeline.
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
    rdata.data = reader.normalize_source_channels(rdata.data.copy())
    channel_names = reader.source_channels
    return rdata.coords, rdata.data, channel_names


def make_noise_mask(nside: int, noise_rate: float, seed: int = 42) -> torch.Tensor:
    """Random binary mask over HEALPix cells: True = high noise."""
    npix = 12 * nside**2
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.random(npix) < noise_rate)


def coords_to_cell_ids(coords: NDArray, nside: int) -> NDArray:
    """Map lat/lon coords to HEALPix cell IDs."""
    thetas, phis = theta_phi_to_standard_coords(coords)
    return ang2pix(nside, thetas, phis, nest=True)


def plot_mollweide(
    ax: plt.Axes,
    coords: NDArray,
    values: NDArray,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
) -> None:
    """Scatter plot on Mollweide projection."""
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])
    sc = ax.scatter(
        lon_rad,
        lat_rad,
        c=values,
        s=0.3,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.figure.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)


def make_noise_schedule(
    name: str,
    t_min: float,
    t_max: float,
) -> NoiseSchedule | VariableCovarianceLinearInformationNoiseSchedule:
    if name == "linear_information":
        return LinearInformationNoiseSchedule(t_min=t_min, t_max=t_max)
    if name == "variable_covariance_linear_information":
        return VariableCovarianceLinearInformationNoiseSchedule(t_min=t_min, t_max=t_max)
    if name == "cosine":
        return CosineNoiseSchedule()
    raise ValueError(f"Unsupported noise schedule '{name}'")


def schedule_information(
    schedule: NoiseSchedule | VariableCovarianceLinearInformationNoiseSchedule,
    t: float,
    group: str | None = None,
) -> float:
    if isinstance(schedule, VariableCovarianceLinearInformationNoiseSchedule):
        if group is None:
            return float(np.mean([schedule.information(t, g) for g in VARIABLE_GROUPS]))
        return schedule.information(t, group)

    _, sigma = schedule.alpha_sigma(t)
    if group is not None:
        return covariance_information_from_sigma(
            DEFAULT_VARIABLE_COVARIANCE_EIGENVALUES[group],
            sigma,
        )
    return -np.log(sigma)


def schedule_bin_gains(
    schedule: NoiseSchedule | VariableCovarianceLinearInformationNoiseSchedule,
    t_min: float,
    t_max: float,
    num_bins: int = 8,
    group: str | None = None,
) -> tuple[NDArray, NDArray]:
    edges = np.linspace(t_min, t_max, num_bins + 1)
    information = np.array([schedule_information(schedule, float(t), group) for t in edges])
    return edges, information[:-1] - information[1:]


def sampled_pair_gain_slope_cv(
    schedule: NoiseSchedule | VariableCovarianceLinearInformationNoiseSchedule,
    t_min: float,
    t_max: float,
    group: str,
    num_pairs: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(num_pairs):
        s_raw, t_raw = sample_noise_coordinates(rng, t_min=t_min, t_max=t_max)
        gain = schedule_information(schedule, s_raw, group) - schedule_information(
            schedule,
            t_raw,
            group,
        )
        slopes.append(gain / (t_raw - s_raw))
    slopes_np = np.asarray(slopes)
    return float(slopes_np.std() / slopes_np.mean())


def schedule_alpha_sigma_for_channels(
    schedule: NoiseSchedule | VariableCovarianceLinearInformationNoiseSchedule,
    t: float,
    channel_names: list[str],
) -> tuple[float | NDArray, float | NDArray]:
    if isinstance(schedule, VariableCovarianceLinearInformationNoiseSchedule):
        return schedule.alpha_sigma_for_channels(t, channel_names)
    return schedule.alpha_sigma(t)


def channel_coefficient(value: float | NDArray, channel: int) -> float:
    value_np = np.asarray(value)
    if value_np.ndim == 0:
        return float(value_np)
    return float(value_np[channel])


def plot_schedule_diagnostics(
    schedule: NoiseSchedule | VariableCovarianceLinearInformationNoiseSchedule,
    t_min: float,
    t_max: float,
    s_val: float,
    t_val: float,
) -> plt.Figure:
    grid = np.linspace(t_min, t_max, 256)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Noise schedule diagnostics", fontsize=11)

    if isinstance(schedule, VariableCovarianceLinearInformationNoiseSchedule):
        for group in VARIABLE_GROUPS:
            alpha_sigma = np.array([schedule.alpha_sigma_for_group(group, float(t)) for t in grid])
            axes[0].plot(grid, alpha_sigma[:, 1], label=f"sigma {group}")
            axes[1].plot(
                grid,
                [schedule_information(schedule, float(t), group) for t in grid],
                label=group,
            )
        rel_ranges = []
        for group in VARIABLE_GROUPS:
            _, gains = schedule_bin_gains(schedule, t_min, t_max, group=group)
            rel_ranges.append((gains.max() - gains.min()) / gains.mean())
        axes[2].bar(VARIABLE_GROUPS, rel_ranges)
        axes[2].set_title("Covariance gain relative range")
        axes[2].set_ylabel("fraction")
    else:
        alpha_sigma = np.array([schedule.alpha_sigma(float(t)) for t in grid])
        information = -np.log(alpha_sigma[:, 1])
        edges, gains = schedule_bin_gains(schedule, t_min, t_max)
        mids = 0.5 * (edges[:-1] + edges[1:])

        axes[0].plot(grid, alpha_sigma[:, 0], label="alpha")
        axes[0].plot(grid, alpha_sigma[:, 1], label="sigma")
        axes[1].plot(grid, information, label="-log(sigma)")
        axes[1].scatter(
            [s_val, t_val],
            [schedule_information(schedule, s_val), schedule_information(schedule, t_val)],
        )
        axes[2].bar(mids, gains, width=np.diff(edges) * 0.85)
        axes[2].set_title("Equal-bin information gain")
        axes[2].set_xlabel("noise interval")
        axes[2].set_ylabel("nats")

    axes[0].axvline(s_val, color="tab:green", linestyle="--", linewidth=1, label="s")
    axes[0].axvline(t_val, color="tab:red", linestyle="--", linewidth=1, label="t")
    axes[0].set_title("VP coefficients")
    axes[0].set_xlabel("noise coordinate")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Covariance information")
    axes[1].set_xlabel("noise coordinate")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig


def root_mean_square(values: NDArray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def correlation(left: NDArray, right: NDArray) -> float:
    if left.size < 2 or right.size < 2:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def plot_empirical_diagnostics(
    clean: NDArray,
    student: NDArray,
    teacher: NDArray,
    mask: NDArray,
    alpha_s: float,
    sigma_s: float,
    alpha_t: float,
    sigma_t: float,
) -> tuple[plt.Figure, dict[str, float]]:
    low_mask = ~mask
    alpha = np.where(mask, alpha_t, alpha_s)
    sigma = np.where(mask, sigma_t, sigma_s)
    student_innovation = student - alpha * clean
    student_eps = student_innovation / sigma
    teacher_eps = (teacher - alpha_s * clean) / sigma_s

    low_rms = root_mean_square(student_innovation[low_mask])
    high_rms = root_mean_square(student_innovation[mask])
    expected_ratio = sigma_t / sigma_s
    measured_ratio = high_rms / low_rms if low_rms > 0 else float("nan")

    metrics = {
        "student_eps_mean": float(np.mean(student_eps)),
        "student_eps_std": float(np.std(student_eps)),
        "teacher_eps_mean": float(np.mean(teacher_eps)),
        "teacher_eps_std": float(np.std(teacher_eps)),
        "innovation_rms_low": low_rms,
        "innovation_rms_high": high_rms,
        "innovation_rms_ratio": measured_ratio,
        "expected_rms_ratio": expected_ratio,
        "student_clean_corr_low": correlation(student[low_mask], clean[low_mask]),
        "student_clean_corr_high": correlation(student[mask], clean[mask]),
        "teacher_clean_corr": correlation(teacher, clean),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Empirical noise diagnostics for selected channel", fontsize=11)

    axes[0, 0].hist(student_eps[low_mask], bins=80, alpha=0.6, density=True, label="student low")
    axes[0, 0].hist(student_eps[mask], bins=80, alpha=0.6, density=True, label="student high")
    axes[0, 0].set_title("Recovered eps distribution")
    axes[0, 0].legend(fontsize=8)

    x = np.arange(2)
    axes[0, 1].bar(x - 0.18, [low_rms, high_rms], width=0.36, label="measured")
    axes[0, 1].bar(x + 0.18, [sigma_s, sigma_t], width=0.36, label="expected sigma")
    axes[0, 1].set_xticks(x, ["low/no-mask", "high/mask"])
    axes[0, 1].set_title(
        f"Innovation RMS ratio={measured_ratio:.2f}; expected={expected_ratio:.2f}"
    )
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(
        ["student eps", "teacher eps"],
        [metrics["student_eps_std"], metrics["teacher_eps_std"]],
    )
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1, label="expected")
    axes[1, 0].set_title(
        f"Recovered eps stds; means="
        f"{metrics['student_eps_mean']:.3f}/{metrics['teacher_eps_mean']:.3f}"
    )
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(
        ["student low", "student high", "teacher"],
        [
            metrics["student_clean_corr_low"],
            metrics["student_clean_corr_high"],
            metrics["teacher_clean_corr"],
        ],
    )
    axes[1, 1].axhline(alpha_s, color="tab:green", linestyle="--", linewidth=1, label="alpha_s")
    axes[1, 1].axhline(alpha_t, color="tab:red", linestyle="--", linewidth=1, label="alpha_t")
    axes[1, 1].set_title("Clean/noised correlation")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    return fig, metrics


def companion_output_path(path: str, suffix: str) -> Path:
    output_path = Path(path)
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Visualize self-flow noise on real ERA5 data")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to visualize")
    parser.add_argument(
        "--noise_rate",
        type=float,
        default=0.8,
        help="Fraction of cells with high noise",
    )
    parser.add_argument("--s_val", type=float, default=0.1, help="Noise level s (low)")
    parser.add_argument(
        "--t_val",
        type=float,
        default=0.9,
        help="Noise level t (high, masked cells)",
    )
    parser.add_argument(
        "--schedule",
        choices=["linear_information", "cosine", "variable_covariance_linear_information"],
        default="variable_covariance_linear_information",
        help="Noise schedule to visualize",
    )
    parser.add_argument(
        "--schedule_t_min",
        type=float,
        default=0.1,
        help="Lower bound used by the linear-information schedule",
    )
    parser.add_argument(
        "--schedule_t_max",
        type=float,
        default=0.9,
        help="Upper bound used by the linear-information schedule",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=777,
        help="Noise seed (shared by student/teacher)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save figure to path instead of showing",
    )
    parser.add_argument(
        "--num_pair_samples",
        type=int,
        default=4096,
        help="Self-flow sampled pairs for covariance gain slope diagnostics",
    )
    args = parser.parse_args()

    # Load real data
    logger.info("Loading ERA5 data...")
    coords, data_np, channel_names = load_era5_timestep()
    data = torch.tensor(data_np)
    num_points = data.shape[0]
    logger.info(f"Loaded: {num_points} points, {data.shape[1]} channels")
    logger.info(f"Visualizing channel {args.channel}: '{channel_names[args.channel]}'")

    # Noise schedule
    schedule = make_noise_schedule(args.schedule, args.schedule_t_min, args.schedule_t_max)
    alpha_s_all, sigma_s_all = schedule_alpha_sigma_for_channels(
        schedule, args.s_val, channel_names
    )
    alpha_t_all, sigma_t_all = schedule_alpha_sigma_for_channels(
        schedule, args.t_val, channel_names
    )
    alpha_s = channel_coefficient(alpha_s_all, args.channel)
    sigma_s = channel_coefficient(sigma_s_all, args.channel)
    alpha_t = channel_coefficient(alpha_t_all, args.channel)
    sigma_t = channel_coefficient(sigma_t_all, args.channel)
    info_gain = schedule_information(schedule, args.s_val) - schedule_information(
        schedule, args.t_val
    )
    edges, gains = schedule_bin_gains(schedule, args.schedule_t_min, args.schedule_t_max)
    rel_gain_range = (gains.max() - gains.min()) / gains.mean()
    logger.info(
        f"Noise schedule: {args.schedule} over [{args.schedule_t_min:.2f}, "
        f"{args.schedule_t_max:.2f}]"
    )
    logger.info(
        f"Noise levels: s={args.s_val:.2f} (alpha={alpha_s:.3f}, sigma={sigma_s:.3f}), "
        f"t={args.t_val:.2f} (alpha={alpha_t:.3f}, sigma={sigma_t:.3f}), "
        f"information gain={info_gain:.3f} nats"
    )
    logger.info(
        f"Equal-bin information gain range: {gains.min():.4f}..{gains.max():.4f} "
        f"nats (relative range={rel_gain_range:.2%}; edges={np.round(edges, 3)})"
    )
    logger.info("Covariance-aware equal-bin gain by variable group:")
    for group in VARIABLE_GROUPS:
        _, group_gains = schedule_bin_gains(
            schedule,
            args.schedule_t_min,
            args.schedule_t_max,
            group=group,
        )
        group_rel = (group_gains.max() - group_gains.min()) / group_gains.mean()
        logger.info(
            f"  {group}: {group_gains.min():.5f}..{group_gains.max():.5f} "
            f"nats/mode (relative range={group_rel:.2%})"
        )
    logger.info("Self-flow sampled pair gain slope CV by variable group:")
    for group in VARIABLE_GROUPS:
        slope_cv = sampled_pair_gain_slope_cv(
            schedule,
            args.schedule_t_min,
            args.schedule_t_max,
            group,
            args.num_pair_samples,
            args.seed,
        )
        logger.info(f"  {group}: {slope_cv:.2%}")
    fig3 = plot_schedule_diagnostics(
        schedule,
        args.schedule_t_min,
        args.schedule_t_max,
        args.s_val,
        args.t_val,
    )

    # Build per-point noise mask via HEALPix cells (same as training pipeline)
    nside = 2**HEALPIX_LEVEL
    cell_ids = coords_to_cell_ids(coords, nside)
    noise_mask_cells = make_noise_mask(nside, args.noise_rate)
    point_noise_mask = noise_mask_cells[cell_ids]  # per-point mask

    n_masked = point_noise_mask.sum().item()
    logger.info(f"HEALPix level={HEALPIX_LEVEL}, nside={nside}")
    logger.info(
        f"Points masked (high noise): {n_masked}/{num_points} ({n_masked / num_points:.1%})"
    )

    # Apply noise (exactly as in the training pipeline)
    student_data = apply_self_flow_noise(
        data,
        point_noise_mask,
        alpha_s_all,
        sigma_s_all,
        alpha_t_all,
        sigma_t_all,
        args.seed,
    )
    teacher_data = apply_uniform_noise(data, alpha_s_all, sigma_s_all, args.seed)

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
    plot_mollweide(
        axes[0, 1],
        coords,
        point_noise_mask.numpy().astype(float),
        f"Noise mask (rate={args.noise_rate})",
        vmin=0,
        vmax=1,
        cmap="coolwarm",
    )
    plot_mollweide(
        axes[1, 0],
        coords,
        student,
        "Student input (self-flow noise)",
        vmin=vmin,
        vmax=vmax,
    )
    plot_mollweide(
        axes[1, 1],
        coords,
        teacher,
        "Teacher input (uniform noise at s)",
        vmin=vmin,
        vmax=vmax,
    )

    plt.tight_layout()

    # --- Figure 2: Residuals ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8), subplot_kw={"projection": "mollweide"})
    fig2.suptitle("Noise Residuals & Consistency Check", fontsize=11)

    student_residual = student - clean
    teacher_residual = teacher - clean
    res_lim = max(abs(student_residual).max(), abs(teacher_residual).max())

    plot_mollweide(
        axes2[0, 0],
        coords,
        student_residual,
        "Student residual (noised - clean)",
        vmin=-res_lim,
        vmax=res_lim,
    )
    plot_mollweide(
        axes2[0, 1],
        coords,
        teacher_residual,
        "Teacher residual (noised - clean)",
        vmin=-res_lim,
        vmax=res_lim,
    )

    # Consistency: on unmasked points, student == teacher
    mask_np = point_noise_mask.numpy()
    diff_unmasked = np.where(~mask_np, student - teacher, 0.0)
    diff_masked = np.where(mask_np, student - teacher, 0.0)
    fig4, empirical_metrics = plot_empirical_diagnostics(
        clean,
        student,
        teacher,
        mask_np,
        alpha_s,
        sigma_s,
        alpha_t,
        sigma_t,
    )

    plot_mollweide(
        axes2[1, 0], coords, diff_unmasked, "Student - Teacher (unmasked, should be 0)", cmap="PuOr"
    )
    plot_mollweide(
        axes2[1, 1],
        coords,
        diff_masked,
        "Student - Teacher (masked, shows noise difference)",
        cmap="PuOr",
    )

    plt.tight_layout()

    # --- Numerical checks ---
    max_diff_unmasked = np.abs(diff_unmasked).max()
    logger.info("\nConsistency checks:")
    logger.info(
        f"  Max |student - teacher| on UNMASKED points: {max_diff_unmasked:.2e} (should be 0)"
    )
    if max_diff_unmasked < 1e-6:
        logger.info(
            "  PASS: Student and teacher agree on unmasked points (same eps, same alpha_s/sigma_s)"
        )
    else:
        logger.info("  FAIL: Student and teacher DISAGREE on unmasked points!")

    mean_abs_diff_masked = np.abs(diff_masked[mask_np]).mean() if mask_np.any() else 0.0
    logger.info(
        f"  Mean |student - teacher| on MASKED points: {mean_abs_diff_masked:.4f} (should be > 0)"
    )
    logger.info(
        f"  VP check s: alpha_s^2 + sigma_s^2 = {alpha_s**2 + sigma_s**2:.6f} (should be 1)"
    )
    logger.info(
        f"  VP check t: alpha_t^2 + sigma_t^2 = {alpha_t**2 + sigma_t**2:.6f} (should be 1)"
    )
    logger.info(
        "  Recovered eps mean/std (student): "
        f"{empirical_metrics['student_eps_mean']:.3f}/"
        f"{empirical_metrics['student_eps_std']:.3f}"
    )
    logger.info(
        "  Recovered eps mean/std (teacher): "
        f"{empirical_metrics['teacher_eps_mean']:.3f}/"
        f"{empirical_metrics['teacher_eps_std']:.3f}"
    )
    logger.info(
        "  Innovation RMS ratio high/low: "
        f"{empirical_metrics['innovation_rms_ratio']:.3f} "
        f"(expected {empirical_metrics['expected_rms_ratio']:.3f})"
    )

    if args.save:
        save_path = Path(args.save)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        residual_path = companion_output_path(args.save, "residuals")
        fig2.savefig(residual_path, dpi=150, bbox_inches="tight")
        schedule_path = companion_output_path(args.save, "schedule")
        fig3.savefig(schedule_path, dpi=150, bbox_inches="tight")
        empirical_path = companion_output_path(args.save, "empirical")
        fig4.savefig(empirical_path, dpi=150, bbox_inches="tight")
        logger.info(f"\nSaved: {save_path}, {residual_path}, {schedule_path}, and {empirical_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
