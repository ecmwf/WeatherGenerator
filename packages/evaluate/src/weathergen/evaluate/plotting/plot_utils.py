# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import xarray as xr

_logger = logging.getLogger(__name__)


def collect_streams(runs: dict):
    """Get all unique streams across runs, sorted.

    Parameters
    ----------
    runs : dict
        The dictionary containing all run configs.

    Returns
    -------
    set
        all available streams
    """
    return sorted({s for run in runs.values() for s in run["streams"].keys()})


def collect_channels(scores_dict: dict, metric: str, region: str, runs) -> list[str]:
    """Get all unique channels available for given metric and region across runs.

    Parameters
    ----------
    scores_dict : dict
        The dictionary containing all computed metrics.
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    Returns
    -------
    list
        returns a list with all available channels
    """
    channels = set()
    if metric not in scores_dict or region not in scores_dict[metric]:
        return []
    for _stream, run_data in scores_dict[metric][region].items():
        for run_id in runs:
            if run_id not in run_data:
                continue
            values = run_data[run_id]["channel"].values
            channels.update([str(x) for x in np.atleast_1d(values)])
    return list(channels)


def plot_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    plotter: object,
    print_summary: bool,
) -> None:
    """Plot data for all streams and channels for a given metric and region.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    scores_dict : dict
        The dictionary containing all computed metrics.
    plotter:
        Plotter object to handle the plotting part
    print_summary: bool
        Option to print plot values to screen

    """
    streams_set = collect_streams(runs)
    channels_set = collect_channels(scores_dict, metric, region, runs)

    for stream in streams_set:
        for ch in channels_set:
            selected_data, labels, run_ids = [], [], []

            for run_id, data in scores_dict[metric][region].get(stream, {}).items():
                # skip if channel is missing or contains NaN
                if ch not in np.atleast_1d(data.channel.values) or data.isnull().all():
                    continue

                selected_data.append(data.sel(channel=ch))
                labels.append(runs[run_id].get("label", run_id))
                run_ids.append(run_id)

            if selected_data:
                _logger.info(f"Creating plot for {metric} - {region} - {stream} - {ch}.")

                name = create_filename(
                    prefix=[metric, region], middle=sorted(set(run_ids)), suffix=[stream, ch]
                )

                selected_data, time_dim = _assign_time_coord(selected_data)

                plotter.plot(
                    selected_data,
                    labels,
                    tag=name,
                    x_dim=time_dim,
                    y_dim=metric,
                    print_summary=print_summary,
                )


def _assign_time_coord(selected_data: list[xr.DataArray]) -> tuple[xr.DataArray, str]:
    """Ensure that lead_time coordinate exists in the data array.

    Parameters
    ----------
    selected_data : list[xarray.DataArray]
        The data array to check.

    Returns
    -------
    xarray.DataArray
        The data array with lead_time coordinate ensured.

    time_dim : str
        The name of the time dimension used for x-axis.
    """

    time_dim = "forecast_step"

    for data in selected_data:
        if "forecast_step" not in data.dims and "forecast_step" not in data.coords:
            raise ValueError(
                "forecast_step coordinate not found in data dimensions or coordinates."
            )

        if "lead_time" not in data.coords and "lead_time" not in data.dims:
            _logger.warning(
                "lead_time coordinate not found for all plotted data; "
                "using forecast_step as x-axis."
            )
            return selected_data, time_dim

    # Swap forecast_step with lead_time if all available run_ids have lead_time coord
    time_dim = "lead_time"

    for i, data in enumerate(selected_data):
        if data.coords["lead_time"].shape == data.coords["forecast_step"].shape:
            selected_data[i] = data.swap_dims({"forecast_step": "lead_time"})

    return selected_data, time_dim


def ratio_plot_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    plotter: object,
    print_summary: bool,
) -> None:
    """Plot ratio data for all streams and channels for a given metric and region.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    scores_dict : dict
        The dictionary containing all computed metrics.
    plotter:
        Plotter object to handle the plotting part
    print_summary: bool
        Option to print plot values to screen

    """
    streams_set = collect_streams(runs)

    for stream in streams_set:
        selected_data = []
        labels = []
        run_ids = []
        for run_id, run_data in runs.items():
            data = scores_dict.get(metric, {}).get(region, {}).get(stream, {}).get(run_id)
            if data.isnull().all():
                continue
            selected_data.append(data)
            label = run_data.get("label", run_id)
            if label != run_id:
                label = f"{run_id} - {label}"
            labels.append(label)
            run_ids.append(run_id)

        if len(selected_data) > 0:
            _logger.info(f"Creating Ratio plot for {metric} - {stream}")

            name = create_filename(
                prefix=[metric, region], middle=sorted(set(run_ids)), suffix=[stream]
            )
            plotter.ratio_plot(
                selected_data,
                run_ids,
                labels,
                tag=name,
                x_dim="channel",
                y_dim=metric,
                print_summary=print_summary,
            )


def heat_maps_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    plotter: object,
) -> None:
    """Plot ratio data for all streams and channels for a given metric and region.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    scores_dict : dict
        The dictionary containing all computed metrics.
    plotter:
        Plotter object to handle the plotting part
    print_summary: bool
        Option to print plot values to screen

    """
    streams_set = collect_streams(runs)

    for stream in streams_set:
        selected_data = []
        labels = []
        run_ids = []
        for run_id in runs:
            data = scores_dict.get(metric, {}).get(region, {}).get(stream, {}).get(run_id)
            if data.isnull().all():
                continue

            selected_data.append(data)
            label = runs[run_id].get("label", run_id)
            if label != run_id:
                label = f"{run_id} - {label}"
            labels.append(label)
            run_ids.append(run_id)

        if len(selected_data) > 0:
            _logger.info(f"Creating Heat maps for {metric} - {stream}")
            name = create_filename(
                prefix=[metric, region], middle=sorted(set(run_ids)), suffix=[stream]
            )
            selected_data, time_dim = _assign_time_coord(selected_data)

            plotter.heat_map(
                selected_data,
                labels,
                metric=metric,
                tag=name,
                x_dim=time_dim,
            )


def score_card_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    sc_plotter: object,
) -> None:
    """
    Create score cards for all streams and channels for a given metric and region.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    scores_dict : dict
        The dictionary containing all computed metrics.
    sc_plotter:
        Plotter object to handle the plotting part
    """
    streams_set = collect_streams(runs)
    channels_set = collect_channels(scores_dict, metric, region, runs)

    for stream in streams_set:
        selected_data, run_ids = [], []
        for run_id, data in scores_dict[metric][region].get(stream, {}).items():
            if data.isnull().all():
                continue
            selected_data.append(data)
            run_ids.append(run_id)

        if selected_data:
            _logger.info(f"Creating score cards for {metric} - {region} - {stream}.")
            name = "_".join([metric, region, stream])
            sc_plotter.plot(selected_data, run_ids, metric, channels_set, name)


def bar_plot_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    br_plotter: object,
) -> None:
    """
    Create bar plots for all streams and run_ids for a given metric and region.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    scores_dict : dict
        The dictionary containing all computed metrics.
    plotter:
        Plotter object to handle the plotting part
    """
    streams_set = collect_streams(runs)
    channels_set = collect_channels(scores_dict, metric, region, runs)

    for stream in streams_set:
        selected_data, run_ids = [], []

        for run_id, data in scores_dict[metric][region].get(stream, {}).items():
            if data.isnull().all():
                continue
            selected_data.append(data)
            run_ids.append(run_id)

        if selected_data:
            _logger.info(f"Creating bar plots for {metric} - {region} - {stream}.")
            name = "_".join([metric, region, stream])
            br_plotter.plot(selected_data, run_ids, metric, channels_set, name)


class DefaultMarkerSize:
    """
    Utility class for managing default configuration values, such as marker sizes
    for various data streams.
    """

    _marker_size_stream = {
        "era5": 2.5,
        "imerg": 0.25,
        "cerra": 0.1,
    }

    _default_marker_size = 0.5

    @classmethod
    def get_marker_size(cls, stream_name: str) -> float:
        """
        Get the default marker size for a given stream name.

        Parameters
        ----------
        stream_name : str
            The name of the stream.

        Returns
        -------
        float
            The default marker size for the stream.
        """
        return cls._marker_size_stream.get(stream_name.lower(), cls._default_marker_size)

    @classmethod
    def list_streams(cls):
        """
        List all streams with defined marker sizes.

        Returns
        -------
        list[str]
            List of stream names.
        """
        return list(cls._marker_size_stream.keys())


def create_filename(
    *,
    prefix: Sequence[str] = (),
    middle: Iterable[str] = (),
    suffix: Sequence[str] = (),
    sep: str = "_",
    max_len: int = 255,
):
    """
    Join strings as: prefix + middle + suffix, truncating only `middle`
    to ensure the final string does not exceed max_len.

    Parameters
    ----------
    prefix : Sequence[str]
        Parts that must appear before the truncated section.
    middle : Iterable[str]
        Parts that may be truncated (order preserved).
    suffix : Sequence[str]
        Parts that must appear after the truncated section.
    sep : str
        Separator used for joining.
    max_len : int
        Maximum total length of the joined string.

    Returns
    -------
    str
        The joined string, with only `middle` truncated if necessary.
    """

    pref, mid, suf = map(lambda x: list(map(str, x)), (prefix, middle, suffix))
    fixed = sep.join(pref + suf)
    avail = max_len - len(fixed)

    if mid and pref:
        avail -= len(sep)
    if mid and suf:
        avail -= len(sep)

    truncated_middle, used = [], 0

    for x in mid:
        d = len(x) + (len(sep) if truncated_middle else 0)
        if used + d > avail:
            break
        truncated_middle.append(x)
        used += d

    if len(truncated_middle) < len(mid):
        _logger.warning(
            f"Filename truncated: only {len(truncated_middle)} of {len(mid)} middle parts used "
            f"to keep length <= {max_len}."
        )

    return sep.join(prefix + truncated_middle + suffix)
