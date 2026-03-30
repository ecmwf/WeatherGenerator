# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Backward-compatible re-exports from the utils sub-modules.

All public names that were previously defined directly in this file are
re-exported here so that existing ``from weathergen.evaluate.utils.utils
import ...`` statements continue to work without modification.
"""

# ruff: noqa: F401
from weathergen.evaluate.utils.array_utils import (
    bias_ranges,
    calc_bounds,
    calc_val,
    common_ranges,
    scalar_coord_to_dim,
)
from weathergen.evaluate.utils.dict_utils import (
    merge,
    nested_dict,
    parse_metric_params,
    triple_nested_dict,
)
from weathergen.evaluate.utils.plotting import (
    _plot_score_maps_per_stream,
    _plot_single_sample,
    _resolve_num_plot_workers,
    _scatter_plot_single,
    _score_map_fstep_worker,
    plot_data,
    plot_score_maps_per_stream,
    plot_summary,
)
from weathergen.evaluate.utils.scoring import (
    _score_single_fstep,
    calc_scores_per_stream,
    get_next_data,
    metric_list_to_json,
)

__all__ = [
    # array_utils
    "bias_ranges",
    "calc_bounds",
    "calc_val",
    "common_ranges",
    "scalar_coord_to_dim",
    # dict_utils
    "merge",
    "nested_dict",
    "parse_metric_params",
    "triple_nested_dict",
    # plotting
    "_plot_score_maps_per_stream",
    "_plot_single_sample",
    "_resolve_num_plot_workers",
    "_scatter_plot_single",
    "_score_map_fstep_worker",
    "plot_data",
    "plot_score_maps_per_stream",
    "plot_summary",
    # scoring
    "_score_single_fstep",
    "calc_scores_per_stream",
    "get_next_data",
    "metric_list_to_json",
]
