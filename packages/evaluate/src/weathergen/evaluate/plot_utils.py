# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np

_logger = logging.getLogger(__name__)


def collect_streams(runs):
    """Get all unique streams across runs, sorted."""
    return sorted({s for run in runs.values() for s in run["streams"].keys()})


def collect_channels(scores_dict, metric, region, runs):
    """Get all unique channels available for given metric and region across runs."""
    channels = set()
    if metric not in scores_dict or region not in scores_dict[metric]:
        return []
    for _stream, run_data in scores_dict[metric][region].items():
        for run_id in runs:
            if run_id not in run_data:
                continue
            values = run_data[run_id]["channel"].values
            channels.update(np.atleast_1d(values))
    return list(channels)


def plot_metric_region(metric, region, runs, scores_dict, plotter, print_summary):
    """Plot data for all streams and channels for a given metric and region."""
    streams_set = collect_streams(runs)
    channels_set = collect_channels(scores_dict, metric, region, runs)

    for stream in streams_set:
        for ch in channels_set:
            selected_data, labels, run_ids = [], [], []

            for run_id, data in scores_dict[metric][region].get(stream, {}).items():
                # skip if channel is missing or contains NaN
                if ch not in np.atleast_1d(data.channel.values) or data.isnull().any():
                    continue

                selected_data.append(data.sel(channel=ch))
                labels.append(runs[run_id].get("label", run_id))
                run_ids.append(run_id)

            if selected_data:
                _logger.info(
                    f"Creating plot for {metric} - {region} - {stream} - {ch}."
                )
                name = "_".join([metric, region] + sorted(set(run_ids)) + [stream, ch])
                plotter.plot(
                    selected_data,
                    labels,
                    tag=name,
                    x_dim="forecast_step",
                    y_dim=metric,
                    print_summary=print_summary,
                )
