"""
Latent space diagnostics for encoder analysis.

Loads a trained model checkpoint and runs a set of validation batches through the
encoder to compute the following statistics (see run docstring for details):

  1. Spatial variance map         – variance across the batch at each HEALPix cell
  2. Tail behaviour               – abs-max/min and excess kurtosis (per channel)
  3. Channel correlation matrix   – Pearson correlation between latent channels
  4. Spatial total variation      – average |z[i] – z[j]| over neighbouring cells
  5. Per-channel Q-Q vs N(0,1)   – per-channel marginal distribution check
  6. Obs-mask correlation         – correlation of each channel with obs-presence mask
  7. Temporal smoothness          – L2 distance between consecutive time-step latents
  8. Dead / saturated channels    – per-channel variance across all (B×H) samples

Usage (single-GPU, no SLURM):
    uv run python scripts/analyse_latent_space.py \\
        --run_id yk85n9s7 \\
        [--mini_epoch -1] \\
        [--n_batches 8] \\
        [--obs_streams AVHRR georing] \\
        [--out_dir plots/latent_diag/yk85n9s7]

uv run python scripts/analyse_latent_space.py \
    --run_id yk85n9s7 \
    --n_batches 8 \
    --out_dir plots/latent_diag/yk85n9s7
"""

import argparse
import logging
from pathlib import Path

import astropy_healpix as ah
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import weathergen.common.config as config
from weathergen.common.logger import init_loggers
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.model import ModelParams
from weathergen.model.model_interface import get_model, load_model
from weathergen.train.utils import (
    VAL,
    cfg_keys_to_filter,
    filter_config_by_enabled,
    get_active_stage_config,
)
from weathergen.utils.utils import get_dtype

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latent space diagnostics for a trained encoder.")
    p.add_argument("--run_id", required=True, help="Run ID of the checkpoint to analyse.")
    p.add_argument(
        "--mini_epoch", type=int, default=-1, help="Checkpoint mini-epoch (-1 = latest)."
    )
    p.add_argument(
        "--n_batches", type=int, default=8, help="Number of validation batches to process."
    )
    p.add_argument(
        "--n_cell_sub",
        type=int,
        default=1024,
        help="Number of HEALPix cells to subsample per batch for channel-level stats.",
    )
    p.add_argument(
        "--obs_streams",
        nargs="*",
        default=[],
        help="Stream names that contain observations (for stat 6). "
        "If empty, all non-ERA5 streams are treated as obs streams.",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for plots (default: plots/latent_diag/<run_id>).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config & model setup
# ---------------------------------------------------------------------------


def setup_config(run_id: str, mini_epoch: int):
    """Load run config and override DDP/FSDP settings for single-GPU usage."""
    cf = config.load_merge_configs(
        private_home=None,
        from_run_id=run_id,
        mini_epoch=mini_epoch if mini_epoch != -1 else None,
    )
    # Force single-GPU, no distributed training
    cf.with_ddp = False
    cf.with_fsdp = False
    cf.rank = 0
    cf.local_rank = 0
    cf.world_size = 1
    cf.world_size_original = 1
    cf.stage = "inference"
    # Reduce data loading workers for an interactive script
    cf.data_loading.num_workers = min(cf.data_loading.get("num_workers", 4), 4)
    return cf


def build_dataset_and_model(cf, mini_epoch: int):
    """Create the validation dataset, model, and model_params."""
    training_cfg = filter_config_by_enabled(cf.get("training_config"), cfg_keys_to_filter)
    validation_cfg = get_active_stage_config(
        training_cfg, cf.get("validation_config", {}), cfg_keys_to_filter
    )
    test_cfg = get_active_stage_config(
        validation_cfg, cf.get("test_config", {}), cfg_keys_to_filter
    )

    dataset = MultiStreamDataSampler(cf, test_cfg, stage=VAL)
    logger.info(f"Dataset has {len(dataset)} samples.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(cf, test_cfg.training_mode, dataset, overrides={})
    model = load_model(cf, model, device, run_id=cf.from_run_id, mini_epoch=mini_epoch)
    model = model.to(device).eval()
    logger.info("Model loaded and set to eval mode.")

    model_params = ModelParams(cf).create(cf)
    model_params = model_params.to(device)

    mixed_precision_dtype = get_dtype(cf.mixed_precision_dtype)

    return dataset, model, model_params, device, test_cfg, mixed_precision_dtype


def make_dataloader(cf, dataset, test_cfg, n_batches: int):
    loader_params = {
        "batch_size": None,
        "batch_sampler": None,
        "shuffle": False,
        "num_workers": cf.data_loading.get("num_workers", 2),
        "pin_memory": False,
        "persistent_workers": False,
    }
    return torch.utils.data.DataLoader(dataset, **loader_params, sampler=None)


# ---------------------------------------------------------------------------
# Observation mask extraction
# ---------------------------------------------------------------------------


def extract_obs_mask(
    batch, num_cells: int, obs_stream_names: list[str], all_stream_names: list[str]
) -> torch.Tensor | None:
    """Return a (num_cells,) binary tensor: 1 where any obs stream has data.

    Returns None if no obs streams are available.
    """
    if not obs_stream_names:
        return None

    mask = torch.zeros(num_cells, dtype=torch.float32)
    for sample in batch.source_samples.samples:
        for stream_name in obs_stream_names:
            sd = sample.streams_data.get(stream_name)
            if sd is None:
                continue
            for step_lens in sd.source_tokens_lens:
                if step_lens is None:
                    continue
                lens_cpu = step_lens.cpu().float()
                if lens_cpu.shape[0] == num_cells:
                    mask = (mask + (lens_cpu > 0).float()).clamp(max=1.0)
    return mask


# ---------------------------------------------------------------------------
# Statistics accumulators
# ---------------------------------------------------------------------------


class LatentAccumulator:
    """
    Accumulates latent tensors across batches for offline diagnostic computation.

    Two data tracks:
      - Full spatial track: running sum and sum-of-squares per HEALPix cell
        (shape H) for the spatial variance map (stat 1) and total variation (stat 4).
      - Subsampled track: random subset of cells kept for per-channel stats
        (stats 2, 3, 5, 6, 7, 8).
    """

    def __init__(self, num_cells: int, n_cell_sub: int):
        self.H = num_cells
        self.n_sub = n_cell_sub
        self.D = None  # set on first batch

        # spatial variance map: track sum of z[b, h, :] to compute
        #   spatial_var[h] = 1 - ||mean_z[h, :]||² / D
        # (exact because LayerNorm forces ||z[b, h, :]||² = D)
        self._sum_z: torch.Tensor | None = None  # (H, D) float32
        self._n_total: int = 0  # total number of (b, h) samples

        # subsampled latents for channel-level stats: list of (N_sub, D) cpu tensors
        self.subsample: list[torch.Tensor] = []

        # temporal smoothness: keep the last batch latent for computing inter-step diffs
        self._prev_latent: torch.Tensor | None = None  # (B, H, D)
        self.temporal_diffs: list[float] = []  # per-batch mean L2

        # obs mask correlation: list of (H,) binary masks and matching (H, D) means
        self.obs_mask_latent_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    # ------------------------------------------------------------------ #

    def update(self, z: torch.Tensor, obs_mask: torch.Tensor | None = None):
        """
        Process one batch of latents.

        Parameters
        ----------
        z : (N, H, D)  float tensor where N = B*T
        obs_mask : (H,) binary float tensor or None
        """
        z = z.detach().float().cpu()
        N, H, D = z.shape

        assert H == self.H, f"Expected {self.H} cells, got {H}"

        # initialise on first call
        if self.D is None:
            self.D = D
            self._sum_z = torch.zeros(H, D, dtype=torch.float64)

        # ---- stat 1: spatial variance map ----------------------------
        # accumulate sum(z) per (h, d) across all batch samples
        self._sum_z += z.sum(dim=0).double()
        self._n_total += N

        # ---- stat 7: temporal smoothness -----------------------------
        # Treat successive batches as if they were successive time steps.
        # (For true temporal smoothness, set T>1 in config and split on T.)
        if self._prev_latent is not None:
            prev = self._prev_latent  # (N', H, D)
            # compare first sample of each
            diff = (z[0] - prev[-1]).norm(dim=-1).mean().item()  # mean L2 over H
            self.temporal_diffs.append(diff)
        self._prev_latent = z

        # ---- subsampled track (stats 2, 3, 5, 6, 8) -----------------
        cell_idx = torch.randperm(H)[: self.n_sub]
        z_sub = z[:, cell_idx, :].reshape(-1, D)  # (N * n_sub, D)
        self.subsample.append(z_sub)

        # obs mask correlation pair
        if obs_mask is not None:
            mask_sub = obs_mask[cell_idx]  # (n_sub,)
            mean_per_cell = z[:, cell_idx, :].mean(dim=0)  # (n_sub, D)
            self.obs_mask_latent_pairs.append((mask_sub.cpu(), mean_per_cell.cpu()))

    # ------------------------------------------------------------------ #

    def finalise(self):
        """Compute all summary statistics from the accumulated data."""
        assert self.D is not None, "No batches were processed."
        D = self.D
        H = self.H

        # ---- stat 1: spatial variance map ----------------------------
        # spatial_var[h] = 1 - ||mean_z[h, :]||² / D
        # (||z[b,h,:]||² = D by LayerNorm, so Var_b[z_d] summed over d = D - ||mu||²)
        mean_z = (self._sum_z / self._n_total).float()  # (H, D)
        spatial_var = 1.0 - (mean_z**2).sum(dim=-1) / D  # (H,)

        # ---- combined subsample for all channel-level stats ----------
        z_all = torch.cat(self.subsample, dim=0)  # (M, D)
        M = z_all.shape[0]

        # ---- stat 2: tail behaviour (per channel) -------------------
        abs_max = z_all.abs().max(dim=0).values  # (D,)
        ch_mean = z_all.mean(dim=0)  # (D,)
        ch_var = z_all.var(dim=0)  # (D,)
        # excess kurtosis: E[(z - mu)^4] / var^2 - 3
        z_c = z_all - ch_mean.unsqueeze(0)  # centred
        ch_kurt = (z_c**4).mean(dim=0) / (ch_var**2 + 1e-8) - 3.0  # (D,)

        # ---- stat 3: channel correlation matrix ----------------------
        # Use a random subsample of rows to keep memory feasible
        idx = torch.randperm(M)[: min(M, 16384)]
        z_s = z_all[idx]  # (S, D)
        z_n = z_s - z_s.mean(dim=0, keepdim=True)
        z_n = z_n / (z_n.std(dim=0, keepdim=True) + 1e-8)
        corr_matrix = (z_n.T @ z_n) / z_n.shape[0]  # (D, D)

        # ---- stat 8: dead / saturated channels ----------------------
        ch_var_full = z_all.var(dim=0)  # (D,)

        # ---- stat 6: obs-mask channel correlation -------------------
        obs_corr = None
        if self.obs_mask_latent_pairs:
            masks, means = zip(*self.obs_mask_latent_pairs, strict=False)
            masks = torch.stack(list(masks), dim=0).flatten()  # (n_pairs * n_sub,)
            means = torch.cat(list(means), dim=0)  # (n_pairs * n_sub, D)
            # Pearson r per channel
            mask_c = masks - masks.mean()
            mean_c = means - means.mean(dim=0, keepdim=True)
            num = (mask_c.unsqueeze(-1) * mean_c).sum(dim=0)  # (D,)
            den = mask_c.norm() * mean_c.norm(dim=0) + 1e-8  # (D,)
            obs_corr = num / den  # (D,)

        return {
            "spatial_var": spatial_var,  # (H,)
            "abs_max": abs_max,  # (D,)
            "ch_mean": ch_mean,  # (D,)
            "ch_var": ch_var,  # (D,)
            "ch_kurt": ch_kurt,  # (D,) excess kurtosis
            "corr_matrix": corr_matrix,  # (D, D)
            "z_all": z_all,  # (M, D) full subsample
            "obs_corr": obs_corr,  # (D,) or None
            "temporal_diffs": self.temporal_diffs,
            "ch_var_full": ch_var_full,  # (D,) for dead/saturated channel check
        }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _save(fig, path: Path, title: str):
    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_spatial_var_map(spatial_var: torch.Tensor, healpix_level: int, run_id: str, out_dir: Path):
    """Stat 1 – HEALPix variance map projected to equirectangular."""
    nside = 2**healpix_level
    npix = spatial_var.shape[0]

    try:
        import astropy_healpix.healpy as hp

        fig = plt.figure(figsize=(10, 5))
        hp.mollview(
            spatial_var.numpy(),
            nest=True,
            title=f"Spatial variance map — {run_id}",
            unit="Var across batch (channel-averaged)",
            fig=fig.number,
            cmap="hot",
        )
        _save(
            fig,
            out_dir / "01_spatial_var_map.png",
            title=f"Stat 1 – Spatial variance map ({run_id})",
        )
    except Exception as e:
        logger.warning(f"HEALPix mollview failed ({e}); falling back to sorted-cell plot.")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(spatial_var.numpy(), lw=0.3, alpha=0.8)
        ax.set_xlabel("HEALPix cell index")
        ax.set_ylabel("Variance (channel-averaged)")
        _save(
            fig,
            out_dir / "01_spatial_var_map.png",
            title=f"Stat 1 – Spatial variance map ({run_id})",
        )


def plot_tail_behaviour(abs_max: torch.Tensor, ch_kurt: torch.Tensor, run_id: str, out_dir: Path):
    """Stat 2 – Per-channel abs-max and excess kurtosis histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(abs_max.numpy(), bins=60, color="steelblue", edgecolor="none")
    axes[0].axvline(3.0, color="crimson", ls="--", label="±3σ reference")
    axes[0].set_xlabel("Abs-max value per channel")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].set_title("Per-channel absolute maximum")

    axes[1].hist(ch_kurt.numpy(), bins=60, color="darkorange", edgecolor="none")
    axes[1].axvline(0.0, color="crimson", ls="--", label="Gaussian = 0")
    axes[1].set_xlabel("Excess kurtosis per channel")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].set_title("Per-channel excess kurtosis")

    _save(
        fig,
        out_dir / "02_tail_behaviour.png",
        title=f"Stat 2 – Tail behaviour ({run_id})\n"
        f"abs-max median={abs_max.median():.2f}, "
        f"excess-kurt median={ch_kurt.median():.2f}",
    )


def plot_channel_correlation(corr_matrix: torch.Tensor, run_id: str, out_dir: Path):
    """Stat 3 – Channel correlation matrix heat-map (D×D may be large; plot subset)."""
    C = corr_matrix.shape[0]
    # Show at most 256 channels for readability
    step = max(1, C // 256)
    sub = corr_matrix[::step, ::step].numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sub, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Channel index (subsampled)")
    ax.set_ylabel("Channel index (subsampled)")
    _save(
        fig,
        out_dir / "03_channel_correlation.png",
        title=f"Stat 3 – Channel correlation matrix ({run_id})\n"
        f"(showing every {step}th channel, {sub.shape[0]}×{sub.shape[1]} grid)\n"
        f"Off-diag abs-mean={np.abs(sub[~np.eye(sub.shape[0], dtype=bool)]).mean():.3f}",
    )


def plot_total_variation(
    spatial_var: torch.Tensor,
    healpix_level: int,
    z_subsampled_mean: torch.Tensor,
    run_id: str,
    out_dir: Path,
):
    """Stat 4 – Spatial total variation via HEALPix neighbours."""
    nside = 2**healpix_level
    H = spatial_var.shape[0]

    # Get all 8 neighbours for every cell (nested ordering)
    all_pix = np.arange(H)
    try:
        nbrs = ah.neighbours(all_pix, nside, order="nested")  # (8, H)
        nbrs = np.clip(nbrs, 0, H - 1)  # replace -1 (missing nbr) with self
        val = spatial_var.numpy()
        tv_per_cell = np.abs(val[nbrs] - val[None, :]).mean(axis=0)  # (H,)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(tv_per_cell, bins=80, color="teal", edgecolor="none")
        ax.set_xlabel("Total variation per cell")
        ax.set_ylabel("Count")
        _save(
            fig,
            out_dir / "04_spatial_total_variation.png",
            title=f"Stat 4 – Spatial total variation ({run_id})\n"
            f"mean TV = {tv_per_cell.mean():.4f}, "
            f"max TV = {tv_per_cell.max():.4f}",
        )
    except Exception as e:
        logger.warning(f"Could not compute TV via neighbours ({e}); skipping stat 4.")


def plot_qq(z_all: torch.Tensor, run_id: str, out_dir: Path, n_channels_shown: int = 16):
    """Stat 5 – Q-Q plot for a selection of channels vs N(0,1)."""
    D = z_all.shape[1]
    channel_idxs = torch.linspace(0, D - 1, n_channels_shown).long()

    fig, axes = plt.subplots(4, 4, figsize=(12, 10))
    axes_flat = axes.flatten()
    ref_quantiles = np.linspace(0.01, 0.99, 200)
    from scipy.stats import norm

    ref_vals = norm.ppf(ref_quantiles)

    for i, ch in enumerate(channel_idxs):
        ax = axes_flat[i]
        vals = z_all[:, ch].numpy()
        data_quantiles = np.quantile(vals, ref_quantiles)
        ax.scatter(ref_vals, data_quantiles, s=3, alpha=0.6)
        ax.plot(ref_vals, ref_vals, color="crimson", lw=1, ls="--", label="N(0,1)")
        ax.set_title(f"ch {ch.item()}", fontsize=8)
        ax.set_xlabel("Theoretical", fontsize=7)
        ax.set_ylabel("Observed", fontsize=7)
        ax.tick_params(labelsize=6)

    _save(
        fig,
        out_dir / "05_qq_plots.png",
        title=f"Stat 5 – Q-Q plots vs N(0,1) ({run_id}, {n_channels_shown} channels sampled)",
    )


def plot_obs_correlation(obs_corr: torch.Tensor | None, run_id: str, out_dir: Path):
    """Stat 6 – Per-channel Pearson correlation with obs-presence mask."""
    if obs_corr is None:
        logger.info("Stat 6 skipped (no obs streams provided).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(obs_corr.numpy(), bins=60, color="mediumseagreen", edgecolor="none")
    axes[0].set_xlabel("Pearson r (channel vs obs mask)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of obs-mask correlation across channels")

    axes[1].plot(obs_corr.numpy(), lw=0.5, alpha=0.8)
    axes[1].axhline(0, color="crimson", ls="--")
    axes[1].set_xlabel("Channel index")
    axes[1].set_ylabel("Pearson r")
    axes[1].set_title("Obs-mask correlation per channel")

    _save(
        fig,
        out_dir / "06_obs_mask_correlation.png",
        title=f"Stat 6 – Obs-mask correlation ({run_id})\n"
        f"abs-mean={obs_corr.abs().mean():.4f}, "
        f"max={obs_corr.abs().max():.4f}",
    )


def plot_temporal_smoothness(temporal_diffs: list[float], run_id: str, out_dir: Path):
    """Stat 7 – Mean-L2 distance between consecutive batch latents."""
    if len(temporal_diffs) < 2:
        logger.info("Stat 7 skipped (fewer than 2 batch transitions).")
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(temporal_diffs, marker="o", ms=5)
    ax.set_xlabel("Batch transition index")
    ax.set_ylabel("Mean L2 (over HEALPix cells)")
    _save(
        fig,
        out_dir / "07_temporal_smoothness.png",
        title=f"Stat 7 – Temporal latent smoothness ({run_id})\n"
        f"mean={np.mean(temporal_diffs):.4f}, "
        f"std={np.std(temporal_diffs):.4f}",
    )


def plot_dead_channels(ch_var_full: torch.Tensor, run_id: str, out_dir: Path):
    """Stat 8 – Per-channel variance to detect dead or saturated channels."""
    D = ch_var_full.shape[0]
    dead_thresh = 0.01
    sat_thresh = 5.0
    n_dead = (ch_var_full < dead_thresh).sum().item()
    n_sat = (ch_var_full > sat_thresh).sum().item()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(ch_var_full.numpy(), bins=80, color="slateblue", edgecolor="none")
    axes[0].axvline(
        dead_thresh, color="crimson", ls="--", label=f"dead < {dead_thresh} ({n_dead} ch)"
    )
    axes[0].axvline(
        sat_thresh, color="darkorange", ls="--", label=f"saturated > {sat_thresh} ({n_sat} ch)"
    )
    axes[0].set_xlabel("Per-channel variance")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Distribution of per-channel variance")

    axes[1].semilogy(np.sort(ch_var_full.numpy()))
    axes[1].axhline(dead_thresh, color="crimson", ls="--", label=f"dead threshold {dead_thresh}")
    axes[1].axhline(sat_thresh, color="darkorange", ls="--", label=f"sat threshold {sat_thresh}")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Variance (log scale)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Sorted per-channel variances")

    _save(
        fig,
        out_dir / "08_dead_saturated_channels.png",
        title=f"Stat 8 – Dead/saturated channels ({run_id})\n"
        f"dead (<{dead_thresh}): {n_dead}/{D},  saturated (>{sat_thresh}): {n_sat}/{D}",
    )


# ---------------------------------------------------------------------------
# Scalar summary to stdout / log
# ---------------------------------------------------------------------------


def print_summary(stats: dict, run_id: str, out_dir: Path):
    z_all = stats["z_all"]
    kurt = stats["ch_kurt"]
    abs_max = stats["abs_max"]
    cv = stats["ch_var_full"]
    sv = stats["spatial_var"]

    lines = [
        f"\n{'=' * 60}",
        f"  Latent space summary — run_id: {run_id}",
        f"{'=' * 60}",
        f"  Samples collected (subsampled): {z_all.shape[0]:>8,d}  ×  D={z_all.shape[1]}",
        "\n  [Stat 1] Spatial variance map",
        f"    mean={sv.mean():.4f}  std={sv.std():.4f}  max={sv.max():.4f}  min={sv.min():.4f}",
        "\n  [Stat 2] Tail behaviour (across channels)",
        f"    abs-max:  mean={abs_max.mean():.2f}  median={abs_max.median():.2f}  "
        f"max={abs_max.max():.2f}",
        f"    ex-kurt:  mean={kurt.mean():.3f}  median={kurt.median():.3f}  max={kurt.max():.3f}",
        "\n  [Stat 4] Spatial total variation",
        "    (see plot for histogram)",
        "\n  [Stat 7] Temporal smoothness",
    ]

    if stats["temporal_diffs"]:
        td = stats["temporal_diffs"]
        lines.append(f"    mean-L2 transitions: {np.mean(td):.4f}  std={np.std(td):.4f}")
    else:
        lines.append("    (fewer than 2 batches — not computed)")

    n_dead = (cv < 0.01).sum().item()
    n_sat = (cv > 5.0).sum().item()
    lines += [
        "\n  [Stat 8] Dead/saturated channels",
        f"    dead (<0.01 var): {n_dead}/{cv.shape[0]}",
        f"    saturated (>5.0 var): {n_sat}/{cv.shape[0]}",
    ]

    if stats["obs_corr"] is not None:
        oc = stats["obs_corr"]
        lines += [
            "\n  [Stat 6] Obs-mask correlation",
            f"    abs-mean={oc.abs().mean():.4f}  "
            f"top-10 channels: {oc.abs().topk(10).indices.tolist()}",
        ]

    lines.append(f"{'=' * 60}\n")

    output = "\n".join(lines)
    print(output)

    stats_path = out_dir / "stats.txt"
    stats_path.write_text(output)
    logger.info(f"Saved: {stats_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    run_id = args.run_id

    out_dir = Path(args.out_dir) if args.out_dir else Path("plots") / "latent_diag" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading config for run_id={run_id}, mini_epoch={args.mini_epoch}.")
    cf = setup_config(run_id, args.mini_epoch)
    init_loggers(cf.general.run_id)

    logger.info("Building dataset and model …")
    dataset, model, model_params, device, test_cfg, mp_dtype = build_dataset_and_model(
        cf, args.mini_epoch
    )

    num_cells = 12 * 4**cf.healpix_level
    healpix_level = cf.healpix_level
    logger.info(f"healpix_level={healpix_level}, num_cells={num_cells}, D=?")

    dataloader = make_dataloader(cf, dataset, test_cfg, args.n_batches)

    # Determine obs stream names
    all_stream_names = list(cf.streams.keys())
    obs_streams = args.obs_streams
    if not obs_streams:
        # Heuristic: treat any stream that is not 'ERA5' (case-insensitive) as obs
        obs_streams = [s for s in all_stream_names if "era5" not in s.lower()]
        logger.info(f"Auto-detected obs streams: {obs_streams}")

    accumulator = LatentAccumulator(num_cells=num_cells, n_cell_sub=args.n_cell_sub)

    logger.info(f"Running encoder on {args.n_batches} validation batches …")
    n_processed = 0
    with torch.no_grad():
        for batch in dataloader:
            if n_processed >= args.n_batches:
                break

            batch.to_device(device)

            # Encode — wrap in autocast to match the mixed-precision dtype the model
            # was trained with (same as trainer.train / trainer.validate do)
            with torch.autocast(
                device_type=device.type, dtype=mp_dtype, enabled=cf.with_mixed_precision
            ):
                tokens, _ = model.encoder(model_params, batch.get_source_samples())
            # tokens: (B*T, H, D) after LayerNorm — cast to float32 for accumulation

            obs_mask = extract_obs_mask(batch, num_cells, obs_streams, all_stream_names)

            accumulator.update(tokens, obs_mask)
            n_processed += 1
            logger.info(
                f"  Batch {n_processed}/{args.n_batches} done  "
                f"(latent shape: {tuple(tokens.shape)})"
            )

    logger.info("Computing summary statistics …")
    stats = accumulator.finalise()

    print_summary(stats, run_id, out_dir)

    logger.info(f"Saving plots to {out_dir} …")
    plot_spatial_var_map(stats["spatial_var"], healpix_level, run_id, out_dir)
    plot_tail_behaviour(stats["abs_max"], stats["ch_kurt"], run_id, out_dir)
    plot_channel_correlation(stats["corr_matrix"], run_id, out_dir)
    plot_total_variation(stats["spatial_var"], healpix_level, stats["z_all"], run_id, out_dir)
    plot_qq(stats["z_all"], run_id, out_dir)
    plot_obs_correlation(stats["obs_corr"], run_id, out_dir)
    plot_temporal_smoothness(stats["temporal_diffs"], run_id, out_dir)
    plot_dead_channels(stats["ch_var_full"], run_id, out_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
