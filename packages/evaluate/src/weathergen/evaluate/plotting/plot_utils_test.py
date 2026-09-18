# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the PSD target-averaging helpers in ``plot_utils``."""

import numpy as np

from weathergen.evaluate.plotting.plot_utils import (
    _average_target_psd,
    _target_legend_label,
)


def _ds(freq, target, pred=None):
    return {
        "frequencies": np.asarray(freq, dtype=float),
        "psd_target": np.asarray(target, dtype=float),
        "psd_prediction": np.asarray(pred if pred is not None else target, dtype=float),
        "psd_method": "sht",
    }


def test_average_target_psd_two_runs():
    """The mean is taken over power, per frequency bin."""
    datasets = [_ds([1, 2, 3], [1.0, 2.0, 4.0]), _ds([1, 2, 3], [3.0, 4.0, 6.0])]

    freq, tar_mean, n_used = _average_target_psd(datasets)

    np.testing.assert_allclose(freq, [1, 2, 3])
    np.testing.assert_allclose(tar_mean, [2.0, 3.0, 5.0])
    assert n_used == 2


def test_average_target_psd_single_run_is_identity():
    """A one-run overlay must reproduce that run's target exactly."""
    datasets = [_ds([1, 2, 3], [1.0, 2.0, 4.0])]

    freq, tar_mean, n_used = _average_target_psd(datasets)

    np.testing.assert_allclose(freq, [1, 2, 3])
    np.testing.assert_allclose(tar_mean, [1.0, 2.0, 4.0])
    assert n_used == 1


def test_average_target_psd_drops_mismatched_axis(caplog):
    """Runs on a different frequency axis are excluded rather than crashing the plot."""
    datasets = [
        _ds([1, 2, 3], [1.0, 2.0, 4.0]),
        _ds([1, 2], [10.0, 10.0]),
        _ds([1, 2, 3], [3.0, 4.0, 6.0]),
    ]

    with caplog.at_level("WARNING"):
        freq, tar_mean, n_used = _average_target_psd(datasets, context="t_2m step 0")

    np.testing.assert_allclose(freq, [1, 2, 3])
    np.testing.assert_allclose(tar_mean, [2.0, 3.0, 5.0])
    assert n_used == 2
    assert "t_2m step 0" in caplog.text


def test_average_target_psd_ignores_nan():
    """NaN bins in one run do not poison the mean."""
    datasets = [_ds([1, 2], [np.nan, 2.0]), _ds([1, 2], [4.0, 6.0])]

    _, tar_mean, n_used = _average_target_psd(datasets)

    np.testing.assert_allclose(tar_mean, [4.0, 4.0])
    assert n_used == 2


def test_target_legend_label_states_averaging():
    """The legend must say when the target line is an average, and how many runs it spans."""
    assert _target_legend_label(1) == "Target"
    assert _target_legend_label(3) == "Target (mean of 3 runs)"
