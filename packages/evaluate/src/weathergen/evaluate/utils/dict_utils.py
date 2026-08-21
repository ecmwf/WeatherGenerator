# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Lightweight dict/config utility functions."""

from collections import defaultdict

import omegaconf as oc


def nested_dict():
    """Two-level nested dict factory: dict[key1][key2] = value"""
    return defaultdict(dict)


def triple_nested_dict():
    """Three-level nested dict factory: dict[key1][key2][key3] = value"""
    return defaultdict(nested_dict)


def merge(dst: dict, src: dict) -> dict:
    """Recursively merge *src* into *dst*. Values in *src* overwrite *dst*.

    Parameters
    ----------
    dst : dict
        Destination dictionary.
    src : dict
        Source dictionary.

    Returns
    -------
    dict
        Merged dictionary (same object as *dst*).
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def parse_metric_params(metrics) -> oc.DictConfig:
    """Convert a mixed list of str/dict metrics into a ``{name: params}`` DictConfig.

    The config may look like::

        metrics:
          - fbi:
              thresh: 280
          - rmse

    In Python that becomes ``[{'fbi': {'thresh': 280}}, 'rmse']``.
    This function converts it to ``{'fbi': {'thresh': 280}, 'rmse': {}}``.
    """
    out = oc.DictConfig({})
    for metric in metrics:
        if isinstance(metric, str):
            out = oc.OmegaConf.merge(out, {metric: {}})
        else:
            out = oc.OmegaConf.merge(out, metric)
    return out
