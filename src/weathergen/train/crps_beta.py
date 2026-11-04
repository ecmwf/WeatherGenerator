# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Spread scale beta of the kernel CRPS set from the measured memorisation ratio
R = MSE_train / MSE_test (weathergen-research crps-underdispersion note v7).

A CRPS-trained ensemble settles its spread at the training error, so on unseen data
SSR_test = SSR_train * sqrt(R). Scaling the repulsion term by beta inflates the spread by
kappa(beta) = beta / sqrt(2 - beta^2), and beta = sqrt(2 / (1 + R)) gives SSR_test = 1.

training_config.crps_beta_from_r (all optional except enabled):
  enabled: true
  start_date, end_date: period of training data on which MSE_train is measured, with the
      validation protocol (default: the validation period shifted back by one year)
  loss_fct: loss whose per-channel values are compared (default mse, the ensemble-mean MSE)
  beta_min, beta_max: clip range of beta (default 1.0, 1.2)
"""

import json
import logging
import math
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def beta_from_r(r: float, beta_min: float = 1.0, beta_max: float = 1.2) -> float:
    """Calibrating spread scale beta* = sqrt(2 / (1 + R)), clipped to [beta_min, beta_max]."""
    beta = math.sqrt(2.0 / (1.0 + r))
    return float(min(max(beta, beta_min), beta_max))


def mse_ratio(losses_train: dict, losses_test: dict, loss_fct: str = "mse") -> tuple[float, int]:
    """
    R = mean over (stream, step, channel) of mean(loss_train) / mean(loss_test).

    losses_*: flattened logging dicts {"<module>.<stream>.<step>.<loss_fct>.<channel>": [values]}
    (prepare_losses_for_logging). Returns R and the number of ratios averaged.
    """
    ratios = []
    for key, vals_test in losses_test.items():
        parts = key.split(".")
        if len(parts) < 2 or parts[-2] != loss_fct or key not in losses_train:
            continue
        with warnings.catch_warnings():  # all-NaN keys (e.g. spoofed targets) are skipped
            warnings.simplefilter("ignore", RuntimeWarning)
            m_train = np.nanmean(np.asarray(losses_train[key], dtype=np.float64))
            m_test = np.nanmean(np.asarray(vals_test, dtype=np.float64))
        if np.isfinite(m_train) and np.isfinite(m_test) and m_train > 0 and m_test > 0:
            ratios.append(m_train / m_test)
    if not ratios:
        return math.nan, 0
    return float(np.mean(ratios)), len(ratios)


def append_record(path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def last_beta(path) -> float | None:
    """beta of the last record written to path, or None."""
    try:
        with open(path) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    except FileNotFoundError:
        return None
    return float(json.loads(lines[-1])["beta"]) if lines else None
