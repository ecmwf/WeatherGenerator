# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collapse monitoring metrics for SSL training (JEPA/DINO).

This module implements metrics to detect representation collapse during self-supervised learning:
- Effective Rank (RankMe): Entropy of normalized singular values
- Singular Value Spectrum: Top-k singular values and concentration ratio
- Per-Dimension Variance: Min/mean/max variance across embedding dimensions
- Prototype Entropy: Normalized entropy of DINO prototype assignments
- EMA Beta: Current teacher momentum value

References:
- RankMe (ICML 2023): https://arxiv.org/abs/2210.02885
- C-JEPA (NeurIPS 2024): https://arxiv.org/abs/2410.19560
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CollapseMonitor:
    """
    Monitor for detecting representation collapse during SSL training.

    Computes and caches various collapse indicators that can be logged
    at configurable intervals to minimize computational overhead.
    """

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        """
        Initialize the collapse monitor.

        Args:
            config: Configuration dictionary with collapse_monitoring settings.
            device: Device to use for computations.
        """
        self.device = device
        self.enabled = config.get("enabled", False)
        self.compute_frequency = config.get("compute_frequency", 100)
        self.log_frequency = config.get("log_frequency", 100)

        # Metric configurations
        metrics_config = config.get("metrics", {})

        self.effective_rank_config = metrics_config.get("effective_rank", {})
        self.singular_values_config = metrics_config.get("singular_values", {})
        self.dimension_variance_config = metrics_config.get("dimension_variance", {})
        self.prototype_entropy_config = metrics_config.get("prototype_entropy", {})
        self.ema_beta_config = metrics_config.get("ema_beta", {})

        # Cache for accumulating metrics between log intervals
        self._metrics_cache: dict[str, list[float]] = defaultdict(list)

    def should_compute(self, step: int) -> bool:
        """Check if metrics should be computed at this step."""
        return self.enabled and step % self.compute_frequency == 0

    def should_log(self, step: int) -> bool:
        """Check if metrics should be logged at this step."""
        return self.enabled and step % self.log_frequency == 0

    def compute_metrics(
        self,
        student_latent: torch.Tensor | None = None,
        teacher_latent: torch.Tensor | None = None,
        prototype_probs: torch.Tensor | None = None,
        ema_beta: float | None = None,
        loss_type: str | None = None,
    ) -> dict[str, float]:
        """
        Compute all enabled collapse monitoring metrics.

        Args:
            student_latent: Student model latent representations [B, N, D] or [B, D].
            teacher_latent: Teacher model latent representations [B, N, D] or [B, D].
            prototype_probs: Post-softmax prototype assignment probabilities [B, K] (DINO only).
            ema_beta: Current EMA momentum value.
            loss_type: Type of SSL loss ("JEPA" or "DINO").

        Returns:
            Dictionary of computed metrics.
        """
        if not self.enabled:
            return {}

        metrics: dict[str, float] = {}

        # Determine which tensors to monitor based on config
        tensors_to_monitor: dict[str, torch.Tensor | None] = {}

        effective_rank_source = self.effective_rank_config.get("tensor_source", "both")
        sv_source = self.singular_values_config.get("tensor_source", "both")
        var_source = self.dimension_variance_config.get("tensor_source", "both")

        # Build tensor dict based on what's requested
        if effective_rank_source in ("student", "both") or sv_source in (
            "student",
            "both",
        ) or var_source in ("student", "both"):
            tensors_to_monitor["student"] = student_latent

        if effective_rank_source in ("teacher", "both") or sv_source in (
            "teacher",
            "both",
        ) or var_source in ("teacher", "both"):
            tensors_to_monitor["teacher"] = teacher_latent

        # Compute effective rank
        if self.effective_rank_config.get("enabled", True):
            sample_size = self.effective_rank_config.get("sample_size", 2048)
            for name, tensor in tensors_to_monitor.items():
                if tensor is not None:
                    source = self.effective_rank_config.get("tensor_source", "both")
                    if source == "both" or source == name:
                        eff_rank = self._compute_effective_rank(tensor, sample_size)
                        metrics[f"collapse.{name}.effective_rank"] = eff_rank

        # Compute singular value spectrum
        if self.singular_values_config.get("enabled", True):
            top_k = self.singular_values_config.get("top_k", 10)
            sample_size = self.singular_values_config.get("sample_size", 2048)
            for name, tensor in tensors_to_monitor.items():
                if tensor is not None:
                    source = self.singular_values_config.get("tensor_source", "both")
                    if source == "both" or source == name:
                        sv_metrics = self._compute_singular_values(tensor, top_k, sample_size)
                        for key, value in sv_metrics.items():
                            metrics[f"collapse.{name}.{key}"] = value

        # Compute per-dimension variance
        if self.dimension_variance_config.get("enabled", True):
            for name, tensor in tensors_to_monitor.items():
                if tensor is not None:
                    source = self.dimension_variance_config.get("tensor_source", "both")
                    if source == "both" or source == name:
                        var_metrics = self._compute_dimension_variance(tensor)
                        for key, value in var_metrics.items():
                            metrics[f"collapse.{name}.{key}"] = value

        # Compute prototype entropy (DINO only)
        if (
            self.prototype_entropy_config.get("enabled", True)
            and prototype_probs is not None
            and loss_type == "DINO"
        ):
            entropy = self._compute_prototype_entropy(prototype_probs)
            metrics["collapse.dino.prototype_entropy"] = entropy

        # Log EMA beta
        if self.ema_beta_config.get("enabled", True) and ema_beta is not None:
            metrics["collapse.ema_beta"] = ema_beta

        # Cache metrics for averaging
        for key, value in metrics.items():
            self._metrics_cache[key].append(value)

        return metrics

    def get_cached_metrics(self) -> dict[str, float]:
        """
        Get averaged cached metrics and clear the cache.

        Returns:
            Dictionary of averaged metrics since last call.
        """
        averaged_metrics: dict[str, float] = {}
        for key, values in self._metrics_cache.items():
            if values:
                averaged_metrics[key] = sum(values) / len(values)

        self._metrics_cache.clear()
        return averaged_metrics

    def _flatten_to_samples(self, z: torch.Tensor) -> torch.Tensor:
        """
        Flatten patch dimension into sample dimension.

        Treats [B, N, D] as [B*N, D] where each patch is an independent sample.
        This is consistent with C-JEPA/VICReg approach.

        Args:
            z: Tensor of shape [B, N, D] or [B, D].

        Returns:
            Tensor of shape [B*N, D] or [B, D].
        """
        if z.ndim == 3:
            return z.reshape(-1, z.shape[-1])
        return z

    def _sample_rows(self, z: torch.Tensor, sample_size: int) -> torch.Tensor:
        """
        Randomly sample rows to reduce SVD computation cost.

        Args:
            z: Tensor of shape [N, D].
            sample_size: Maximum number of samples (0 = no sampling).

        Returns:
            Sampled tensor of shape [min(N, sample_size), D].
        """
        if sample_size <= 0 or z.shape[0] <= sample_size:
            return z

        indices = torch.randperm(z.shape[0], device=z.device)[:sample_size]
        return z[indices]

    def _compute_effective_rank(self, z: torch.Tensor, sample_size: int = 2048) -> float:
        """
        Compute effective rank via entropy of normalized singular values (RankMe).

        The effective rank measures how many dimensions are actually being used
        in the representation. A low effective rank indicates collapse.

        Args:
            z: Latent representations [B, N, D] or [B, D].
            sample_size: Maximum samples for SVD computation.

        Returns:
            Effective rank (exp of entropy of normalized singular values).
        """
        z = self._flatten_to_samples(z.detach())
        z = self._sample_rows(z, sample_size)

        # Center the data
        z_centered = z - z.mean(dim=0, keepdim=True)

        # Compute SVD
        try:
            _, s, _ = torch.linalg.svd(z_centered, full_matrices=False)
        except RuntimeError:
            # SVD can fail on degenerate matrices
            logger.warning("SVD failed in effective rank computation")
            return 0.0

        # Normalize singular values to get a probability distribution
        s_normalized = s / (s.sum() + 1e-8)

        # Compute entropy
        entropy = -torch.sum(s_normalized * torch.log(s_normalized + 1e-8))

        # Effective rank is exp(entropy)
        effective_rank = torch.exp(entropy)

        return effective_rank.item()

    def _compute_singular_values(
        self, z: torch.Tensor, top_k: int = 10, sample_size: int = 2048
    ) -> dict[str, float]:
        """
        Compute top-k singular values and concentration ratio.

        The concentration ratio (top SV / sum of all SVs) indicates how much
        variance is captured by the largest singular value. High concentration
        suggests dimensional collapse.

        Args:
            z: Latent representations [B, N, D] or [B, D].
            top_k: Number of top singular values to return.
            sample_size: Maximum samples for SVD computation.

        Returns:
            Dictionary with top-k singular values and concentration ratio.
        """
        z = self._flatten_to_samples(z.detach())
        z = self._sample_rows(z, sample_size)

        # Center the data
        z_centered = z - z.mean(dim=0, keepdim=True)

        # Compute SVD
        try:
            _, s, _ = torch.linalg.svd(z_centered, full_matrices=False)
        except RuntimeError:
            logger.warning("SVD failed in singular value computation")
            return {}

        metrics: dict[str, float] = {}

        # Top-k singular values
        for i in range(min(top_k, len(s))):
            metrics[f"singular_value_{i}"] = s[i].item()

        # Concentration ratio (top SV / sum)
        s_sum = s.sum() + 1e-8
        metrics["sv_concentration"] = (s[0] / s_sum).item()

        return metrics

    def _compute_dimension_variance(self, z: torch.Tensor) -> dict[str, float]:
        """
        Compute per-dimension variance statistics.

        Low minimum variance indicates "dead" dimensions that are not being used.
        Large variance ratio (max/min) suggests imbalanced dimension usage.

        Args:
            z: Latent representations [B, N, D] or [B, D].

        Returns:
            Dictionary with var_min, var_mean, var_max.
        """
        z = self._flatten_to_samples(z.detach())

        # Compute variance along sample dimension
        var_per_dim = z.var(dim=0)

        return {
            "var_min": var_per_dim.min().item(),
            "var_mean": var_per_dim.mean().item(),
            "var_max": var_per_dim.max().item(),
        }

    def _compute_prototype_entropy(self, probs: torch.Tensor) -> float:
        """
        Compute normalized entropy of DINO prototype assignments.

        Low entropy indicates collapse to few prototypes. Entropy is normalized
        to [0, 1] range where 1 means uniform distribution.

        Args:
            probs: Post-softmax prototype assignment probabilities [B, K].

        Returns:
            Normalized entropy in [0, 1].
        """
        probs = probs.detach()

        # Average across batch to get prototype usage distribution
        avg_probs = probs.mean(dim=0)

        # Compute entropy
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

        # Normalize by maximum possible entropy (uniform distribution)
        num_prototypes = probs.shape[1]
        max_entropy = torch.log(torch.tensor(float(num_prototypes), device=probs.device))

        normalized_entropy = entropy / (max_entropy + 1e-8)

        return normalized_entropy.item()
