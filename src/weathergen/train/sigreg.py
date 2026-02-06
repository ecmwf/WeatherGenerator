# ruff: noqa: N806

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Vendored from https://github.com/rbalestr-lab/lejepa
# Original code: Copyright (c) Randall Balestriero, licensed under Apache 2.0.
# Adapted: removed distributed all_reduce (WG handles this separately).

from __future__ import annotations

import torch
import torch.nn as nn


class EppsPulley(nn.Module):
    """Univariate normality test via empirical characteristic function.

    Computes T = N * integral |phi_empirical(t) - phi_normal(t)|^2 w(t) dt
    using trapezoidal integration on [0, t_max].

    Args:
        t_max: Upper integration bound.
        n_points: Number of quadrature points (must be odd).
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17) -> None:
        super().__init__()
        assert n_points % 2 == 1, "n_points must be odd"
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Epps-Pulley statistic.

        Args:
            x: Tensor of shape (*, N, K) — N samples, K projected slices.

        Returns:
            Tensor of shape (*, K) — test statistic per slice.
        """
        N = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_mean = torch.cos(x_t).mean(-3)  # (*, K, n_points)
        sin_mean = torch.sin(x_t).mean(-3)  # (*, K, n_points)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N


class SlicingUnivariateTest(nn.Module):
    """Multivariate normality test via random 1D projections.

    Projects D-dimensional data onto num_slices random unit directions
    and aggregates univariate EppsPulley statistics.

    Args:
        univariate_test: An EppsPulley instance (or compatible).
        num_slices: Number of random projection directions.
    """

    def __init__(self, univariate_test: nn.Module, num_slices: int = 1024) -> None:
        super().__init__()
        self.univariate_test = univariate_test
        self.num_slices = num_slices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sliced multivariate normality statistic.

        Args:
            x: Tensor of shape (*, N, D) — N samples in D dimensions.

        Returns:
            Scalar tensor — mean test statistic across slices.
        """
        # Generate random projection directions (no grad needed)
        with torch.no_grad():
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, dtype=x.dtype)
            A = A / A.norm(p=2, dim=0)
        # Project and compute per-slice statistics
        projected = x @ A  # (*, N, num_slices)
        stats = self.univariate_test(projected)  # (*, num_slices)
        return stats.mean()


class SIGReg(nn.Module):
    """SIGReg loss: SlicingUnivariateTest(EppsPulley).

    Measures deviation of embeddings from isotropic Gaussian N(0, I).

    Args:
        num_slices: Number of random projections.
        n_points: Number of quadrature points for EppsPulley.
        t_max: Upper integration bound for EppsPulley.
    """

    def __init__(
        self,
        num_slices: int = 1024,
        n_points: int = 17,
        t_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.test = SlicingUnivariateTest(
            univariate_test=EppsPulley(t_max=t_max, n_points=n_points),
            num_slices=num_slices,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            x: Embeddings of shape (*, N, D).

        Returns:
            Scalar loss — 0 when x ~ N(0, I), large when collapsed/non-Gaussian.
        """
        return self.test(x)
