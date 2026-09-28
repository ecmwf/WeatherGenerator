# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""NAM (Northern Annular Mode) index computation.

Fits a leading-EOF-based annular-mode index from a single sample's own
forecast-lead-time trajectory.
"""
# TODO: Implement different modes to fit leading EOF
# (own trajectory, target-based and climatology-based)

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

_logger = logging.getLogger(__name__)

# SVD needs at least a handful of forecast steps to give a meaningful mode.
_MIN_TIME_STEPS = 4


def compute_nam_index(
    anomaly: np.typing.NDArray,
    lat: np.typing.NDArray,
    significance_level: float = 0.05,
) -> tuple[np.typing.NDArray, np.typing.NDArray, float] | None:
    """Compute an annular-mode index from a (time, space) anomaly field.

    Parameters
    ----------
    anomaly : np.typing.NDArray
        Anomaly field, shape ``(n_time, n_points)``. ``n_time`` is typically
        one sample's forecast steps; ``n_points`` the polar-cap domain points.
        If computed against the sample's own trajectory mean rather than a
        climatology, the resulting index is not comparable across samples.
    lat : np.typing.NDArray
        Latitude of each point, shape ``(n_points,)``.
    significance_level : float
        Two-tailed p-value threshold for masking the spatial pattern.

    Returns
    -------
    tuple[index, pattern, explained_variance_ratio] | None
        ``index``: shape ``(n_time,)``, the standardized (z-scored) leading
        PC. ``pattern``: shape ``(n_points,)``, the per-point regression of
        the anomaly onto the index (NaN at points dropped as all-NaN, 0
        where not statistically significant). ``None`` if there are too few
        time steps or valid points to fit an EOF.
    """
    n_time, n_points = anomaly.shape
    if n_time < _MIN_TIME_STEPS:
        _logger.debug(f"Too few time steps ({n_time}) to fit a NAM EOF, skipping.")
        return None

    valid_points = ~np.any(np.isnan(anomaly), axis=0)
    if valid_points.sum() < 2:
        _logger.debug("Too few valid (non-NaN) domain points to fit a NAM EOF, skipping.")
        return None

    anomaly_valid = anomaly[:, valid_points]
    lat_valid = lat[valid_points]

    # cos(lat) area-weighting before the SVD (matches EOF.cosine_latitude_weights).
    # TODO: Normalize according to actual grid density
    weights = np.cos(np.radians(lat_valid))
    weighted = anomaly_valid * weights[None, :]

    try:
        u, s, _vt = np.linalg.svd(weighted, full_matrices=False)
    except np.linalg.LinAlgError:
        _logger.warning("SVD failed while fitting NAM EOF, skipping.")
        return None

    pc1 = u[:, 0] * s[0]
    explained_variance_ratio = float(s[0] ** 2 / np.sum(s**2))

    pc_std = pc1.std()
    if pc_std == 0:
        _logger.debug("Degenerate (zero-variance) leading PC, skipping.")
        return None
    # Index = standardized PC1 (matches AnnularModes.eigenvals_timeseries).
    index = (pc1 - pc1.mean()) / pc_std

    pattern_valid = _regress_pattern(anomaly_valid, index, significance_level)

    # Sign convention: positive index must correspond to a negative
    # (low pressure/height) pattern over the polar-cap domain.
    if np.nanmean(pattern_valid) > 0:
        index = -index
        pattern_valid = -pattern_valid

    pattern = np.full(n_points, np.nan)
    pattern[valid_points] = pattern_valid

    return index, pattern, explained_variance_ratio


def _regress_pattern(
    anomaly: np.typing.NDArray,
    index: np.typing.NDArray,
    significance_level: float,
) -> np.typing.NDArray:
    """Per-gridpoint OLS regression of unweighted ``anomaly`` onto ``index``.

    Masks the slope to 0 where the two-tailed p-value of the Pearson
    correlation is not below ``significance_level`` (matches
    ``EOF.project_eofs``).
    """
    n_time = anomaly.shape[0]
    x = anomaly - anomaly.mean(axis=0, keepdims=True)
    y = index - index.mean()

    numerator = x.T @ y
    x_ss = np.sum(x**2, axis=0)
    y_ss = np.sum(y**2)

    with np.errstate(invalid="ignore", divide="ignore"):
        r = numerator / np.sqrt(x_ss * y_ss)
        slope = numerator / y_ss

    df = n_time - 2
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = r * np.sqrt(df / (1 - r**2))
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=df)

    return np.where(np.isfinite(p_value) & (p_value < significance_level), slope, 0.0)
