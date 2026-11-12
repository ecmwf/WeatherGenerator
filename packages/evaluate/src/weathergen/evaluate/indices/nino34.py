# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Nino 3.4 index computation (ENSO SST-anomaly index).

Nino 3.4 is the cos(lat)-weighted mean SST anomaly over a fixed
equatorial-Pacific box (5N-5S, 170W-120W). It is conventionally computed
from *monthly-mean* SST anomalies against a monthly climatology (see
``climatology_match: monthly`` in the index config), but the computation
itself is agnostic to the forecast-step cadence.
"""

from __future__ import annotations

import logging

import numpy as np

_logger = logging.getLogger(__name__)


def compute_nino34_index(
    anomaly: np.typing.NDArray,
    lat: np.typing.NDArray,
    **_kwargs,
) -> tuple[np.typing.NDArray, dict] | None:
    """Compute the Nino 3.4 index from a (time, space) SST-anomaly field.

    Parameters
    ----------
    anomaly : np.typing.NDArray
        SST anomaly field, shape ``(n_time, n_points)``, already restricted
        to the Nino 3.4 box (5N-5S, 170W-120W).
    lat : np.typing.NDArray
        Latitude of each point, shape ``(n_points,)``, used for cos(lat)
        weighting of the box mean. NOTE: cos(lat) alone is only exact
        area-weighting on a regular lat-lon grid; on a reduced Gaussian grid
        (e.g. O96) point density along a latitude band also varies, so this
        under/overweights some points.
        # TODO: Normalize according to actual grid density (same caveat as
        # ``annular_mode.compute_annular_mode_index``).
    **_kwargs
        Accepted and ignored (e.g. ``significance_level``), kept for
        interface parity with other ``INDEX_REGISTRY`` callables (see
        ``_index_single_sample``).

    Returns
    -------
    tuple[index, diagnostics] | None
        ``index``: shape ``(n_time,)``, the area-weighted mean anomaly per
        forecast step (the Nino 3.4 index itself, not standardized).
        ``diagnostics``: empty dict -- unlike the EOF-based indices, a plain
        spatial mean has no meaningful "pattern" or "explained variance
        ratio" to report. ``None`` if there are too few valid (non-NaN)
        points in the box.
    """
    valid_points = ~np.any(np.isnan(anomaly), axis=0)
    if valid_points.sum() == 0:
        _logger.debug("No valid (non-NaN) points in the Nino 3.4 box, skipping.")
        return None

    anomaly_valid = anomaly[:, valid_points]
    lat_valid = lat[valid_points]

    # cos(lat) weighting (same convention as the annular-mode indices).
    weights = np.cos(np.radians(lat_valid))
    weights_sum = weights.sum()
    if weights_sum == 0:
        _logger.debug("Degenerate (zero-sum) weights for the Nino 3.4 box, skipping.")
        return None
    weights = weights / weights_sum

    index = np.nansum(anomaly_valid * weights[None, :], axis=1)

    return index, {}
