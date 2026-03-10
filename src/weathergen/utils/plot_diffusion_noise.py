# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Plotting utilities for comparing the diffusion noise distribution (sigma)
with the distribution of encoded tokens from the model encoder.

The noise level sigma is derived from the EDM parameterization (Karras et al.):
    eta ~ N(0, 1)
    sigma = exp(eta * p_std + p_mean)

So log(sigma) ~ N(p_mean, p_std^2), i.e. sigma follows a log-normal distribution.

These functions are called from the Trainer during validation to produce diagnostic plots.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for HPC / headless environments
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def sample_sigma(p_mean: float, p_std: float, n_samples: int = 100_000) -> np.ndarray:
    """Sample sigma values from the diffusion noise distribution.

    Args:
        p_mean: Mean of log(sigma) distribution.
        p_std: Std of log(sigma) distribution.
        n_samples: Number of samples to draw.

    Returns:
        Array of sigma values.
    """
    eta = np.random.standard_normal(n_samples)
    return np.exp(eta * p_std + p_mean)


def compute_loss_weight(sigma: np.ndarray, sigma_data: float = 0.5) -> np.ndarray:
    """Compute the EDM loss weighting lambda(sigma).

    lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
    """
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2


def plot_noise_vs_tokens(
    p_mean: float,
    p_std: float,
    token_values: np.ndarray,
    sigma_data: float = 0.5,
    n_samples: int = 200_000,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot noise distribution compared with the encoded token value distribution.

    Produces a 2x2 figure:
      - Panel 1: Encoded token value distribution (histogram + mean/std lines)
      - Panel 2: |token| distribution overlaid with the sigma distribution
      - Panel 3: log(sigma) distribution vs log(|token|) distribution
      - Panel 4: Noise-to-signal ratio: sigma / token_std

    Args:
        p_mean: p_mean hyperparameter from config.
        p_std: p_std hyperparameter from config.
        token_values: Flattened numpy array of encoded token values (from encoder output).
        sigma_data: sigma_data hyperparameter.
        n_samples: Number of sigma samples for the noise distribution.
        output_path: If set, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    token_std = float(np.std(token_values))
    token_mean = float(np.mean(token_values))
    token_abs_mean = float(np.mean(np.abs(token_values)))
    sigma = sample_sigma(p_mean, p_std, n_samples)
    label_noise = f"sigma (p_mean={p_mean}, p_std={p_std})"

    # --- Panel 1: Token value distribution ---
    ax = axes[0, 0]
    ax.hist(
        token_values, bins=300, density=True, alpha=0.7, color="steelblue", label="Token values"
    )
    ax.axvline(token_mean, color="red", ls="--", lw=1.5, label=f"mean={token_mean:.3f}")
    ax.axvline(token_mean + token_std, color="orange", ls="--", lw=1, label=f"std={token_std:.3f}")
    ax.axvline(token_mean - token_std, color="orange", ls="--", lw=1)
    ax.set_xlabel("Token value")
    ax.set_ylabel("Density")
    ax.set_title(f"Encoded Token Distribution (n={len(token_values):,})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: |token| distribution vs sigma distribution ---
    ax = axes[0, 1]
    abs_tokens = np.abs(token_values)
    ax.hist(
        abs_tokens,
        bins=300,
        density=True,
        alpha=0.5,
        color="steelblue",
        label=f"|tokens| (mean={token_abs_mean:.3f})",
    )
    sigma_clipped = sigma[sigma < np.percentile(abs_tokens, 99.5)]
    ax.hist(sigma_clipped, bins=200, density=True, alpha=0.4, color="coral", label=label_noise)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Density")
    ax.set_title("|Token values| vs sigma magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: log-scale comparison ---
    ax = axes[1, 0]
    ax.hist(
        np.log10(np.abs(token_values) + 1e-12),
        bins=200,
        density=True,
        alpha=0.5,
        color="steelblue",
        label="log10(|tokens|)",
    )
    ax.hist(np.log10(sigma), bins=200, density=True, alpha=0.4, color="coral", label=label_noise)
    ax.axvline(
        np.log10(sigma_data), color="k", ls="--", lw=1.5, label=f"sigma_data={sigma_data}"
    )
    ax.set_xlabel("log10 scale")
    ax.set_ylabel("Density")
    ax.set_title("log10(|tokens|) vs log10(sigma)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Noise magnitude relative to token std ---
    ax = axes[1, 1]
    ratio = sigma / (token_std + 1e-12)
    ax.hist(np.log10(ratio), bins=200, density=True, alpha=0.6, color="coral", label=label_noise)
    ax.axvline(0, color="k", ls="--", lw=1.5, label="sigma = token_std")
    ax.set_xlabel("log10(sigma / token_std)")
    ax.set_ylabel("Density")
    ax.set_title(f"Noise / Token scale ratio (token_std={token_std:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Diffusion Noise vs Encoded Tokens  |  p_mean={p_mean}, p_std={p_std},"
        f" sigma_data={sigma_data}",
        fontsize=13,
    )
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info(f"Saved diffusion noise vs tokens plot to {output_path}")

    plt.close(fig)
    return fig
