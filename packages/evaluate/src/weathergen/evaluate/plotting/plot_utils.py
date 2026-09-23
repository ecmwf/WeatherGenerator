# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import re
from collections.abc import Iterable, Sequence

import numpy as np
import xarray as xr
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

# Animations (gif/mp4) are stitched from per-frame static plots with Pillow or imageio, both of
# which require raster frames. Vector/document formats like "pdf"/"svg"/"eps" are valid choices
# for the static plots themselves (via image_format) but can't be decoded as animation frames.
RASTER_ANIMATION_FORMATS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"}


# Shared helpers
def calculate_average_over_dim(
    x_dim: str, baseline_var: xr.DataArray, data_var: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate average over xarray dimensions that are larger than 1. Those might be the
    forecast-steps or the samples.

    Parameters
    ----------
    x_dim: str
        The dimension for which an average will not be calculated.
    baseline_var: xr.DataArray
        xarray DataArray with the scores of the baseline model for a specific channel/variable
    data_var: xr.DataArray
        xarray DataArray with the scores of the comparison model for a specific channel/variable

    Returns
    -------
    baseline_score: xarray DataArray
        The baseline average scores over the dimensions not specified by x_dim
    model_score: xarray DataArray
        The model average scores over the dimensions not specified by x_dim
    """
    non_zero_dims = [
        dim for dim in baseline_var.dims if dim != x_dim and baseline_var[dim].shape[0] > 1
    ]

    if non_zero_dims:
        _logger.info(f"Found multiple entries for dimensions: {non_zero_dims}. Averaging...")

    baseline_score = baseline_var.mean(
        dim=[dim for dim in baseline_var.dims if dim != x_dim], skipna=True
    )
    model_score = data_var.mean(dim=[dim for dim in data_var.dims if dim != x_dim], skipna=True)

    return baseline_score, model_score


def lower_is_better(metric: str) -> bool:
    """Determine whether lower or higher is better."""
    return metric in {"l1", "l2", "mae", "mse", "rmse", "vrmse", "bias", "crps", "spread"}


def compute_offsets(n, spacing=0.11):
    """Compute symmetric offsets for *n* items centred around zero.

    Parameters
    ----------
    n : int
        Number of items to offset.
    spacing : float
        Distance between consecutive offsets (default ``0.11``).

    Returns
    -------
    NDArray
        Array of length *n* with offsets centred at zero.
    """
    idx = np.arange(n)
    return (idx - (n - 1) / 2.0) * spacing


def align_labels(da: xr.DataArray, labels: list[str], x_dim: str) -> xr.DataArray:
    """
    Reindex a DataArray to include all labels in the canonical order.
    Missing variables are filled with NaN.
    """
    labels = np.array(labels, dtype=object)
    return da.reindex({x_dim: labels})


def format_datetime(dt):
    """Format a numpy datetime64 value as a human-readable string.

    Parameters
    ----------
    dt : numpy.datetime64
        Datetime value to format.

    Returns
    -------
    str
        Formatted string in ``'%Y-%m-%d T%H:%M:%S'`` format.
    """
    return dt.astype("datetime64[m]").astype(datetime.datetime).strftime("%Y-%m-%d T%H:%M:%S")


def channel_sort_key(name: str) -> tuple[int, str, int]:
    """
    Sorting key for channel names like 't_850', 'z_500', etc.
    Splits the name into a prefix and a number suffix for sorting.
    """
    m = re.match(r"(.+?)_(\d+)$", name)
    if m:
        prefix, number = m.groups()
        return (0, prefix, int(number))
    else:
        return (1, name, float("inf"))


def clean_label(s: str) -> str:
    """Replace underscores and hyphens with spaces, then strip whitespace.

    Parameters
    ----------
    s : str
        Raw label string (e.g. ``'lead_time'``).

    Returns
    -------
    str
        Cleaned label (e.g. ``'lead time'``).
    """
    return re.sub(r"[_\-]+", " ", s).strip()


def filter_set(items: list, allowed: set | None) -> list:
    """Return *items* filtered to *allowed*, or all items if *allowed* is ``None``."""
    if allowed is None:
        return items
    return [x for x in items if x in allowed]


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

    @staticmethod
    def compute_marker_size(marker_size_base: float, scale: bool, lat: NDArray) -> float | NDArray:
        """Return marker sizes, optionally scaled by latitude.

        When *scale* is truthy, markers at higher latitudes are enlarged
        to compensate for the convergence of meridians (HEALPix point
        clustering), keeping visual coverage roughly uniform.

        Parameters
        ----------
        marker_size_base : float
            Base marker size in matplotlib scatter units (pt²).
        scale : bool
            If ``True``, scale marker size by ``1 / cos²(lat)``.
        lat : array-like
            Latitude values (degrees) for every point.

        Returns
        -------
        float or NDArray
            Scalar when *scale* is ``False``; array matching *lat* otherwise.
        """
        if not scale:
            return marker_size_base
        return np.clip(
            marker_size_base / np.cos(np.radians(lat)) ** 2,
            a_min=marker_size_base,
            a_max=marker_size_base * 10.0,
        )

    @classmethod
    def auto_marker_size(
        cls,
        n_points: int,
        fig_width_in: float,
        fig_height_in: float,
        stream_default: float,
        scale: bool,
        lat: NDArray,
        *,
        density_threshold: int = 200_000,
        fill_factor: float = 1.8,
        max_size: float = 4.0,
    ) -> float | NDArray:
        """Compute marker size adapting to point density and figure area.

        For dense grids (≥ *density_threshold* points, e.g. n320, CERRA)
        the marker size is derived from the figure area so that points
        fill the globe without white gaps.  Sparser grids (e.g. o96)
        use the stream default as-is.

        Parameters
        ----------
        n_points : int
            Total number of data points to render.
        fig_width_in : float
            Figure width in inches.
        fig_height_in : float
            Figure height in inches.
        stream_default : float
            Default marker size for the stream (from ``get_marker_size``).
        scale : bool
            If ``True``, scale marker size by latitude.
        lat : array-like
            Latitude values (degrees) for every point.
        density_threshold : int
            Minimum point count for auto-sizing to kick in.
        fill_factor : float
            Multiplier for the area-per-point ratio.
        max_size : float
            Upper clamp for the auto-computed base size.

        Returns
        -------
        float or NDArray
            Marker size(s) ready to pass to ``ax.scatter(s=...)``.
        """
        if n_points >= density_threshold:
            fig_area_pt2 = (fig_width_in * 72) * (fig_height_in * 72)
            base = float(np.clip(fig_area_pt2 / n_points * fill_factor, 0.05, max_size))
        else:
            base = stream_default

        return cls.compute_marker_size(base, scale, lat)


def _flatten_or_average(arr: NDArray) -> NDArray:
    """Flatten array or average across non-quantile dimensions.

    Parameters
    ----------
    arr : NDArray
        Input array, possibly multi-dimensional.

    Returns
    -------
    NDArray
        Flattened 1D array, averaged across extra dimensions if needed.
    """
    if arr.ndim > 1:
        return np.mean(arr, axis=tuple(range(1, arr.ndim))).flatten()
    return arr


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
            selected_data, labels, run_ids, colors = [], [], [], []

            for run_id, data in scores_dict[metric][region].get(stream, {}).items():
                # skip if channel is missing or contains NaN
                if ch not in np.atleast_1d(data.channel.values) or data.isnull().all():
                    continue

                selected_data.append(data.sel(channel=ch))
                labels.append(runs[run_id].get("label", run_id))
                run_ids.append(run_id)
                colors.append(runs[run_id].get("color", None))

            if selected_data:
                _logger.info(f"Creating line plot for {metric} - {region} - {stream} - {ch}.")

                name = create_filename(
                    prefix=[metric, region], middle=sorted(set(run_ids)), suffix=[stream, ch]
                )

                selected_data, time_dim = _assign_time_coord(selected_data)

                title = f"{metric.upper()} | {stream} | {ch}"

                plotter.plot(
                    selected_data,
                    labels,
                    tag=name,
                    x_dim=time_dim,
                    y_dim=metric,
                    print_summary=print_summary,
                    title=title,
                    colors=colors,
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
        lead_time = data.coords["lead_time"]
        forecast_step = data.coords["forecast_step"]

        if (
            lead_time.dims == forecast_step.dims
            and lead_time.shape == forecast_step.shape
            and lead_time.ndim == 1
        ):
            selected_data[i] = data.swap_dims({"forecast_step": "lead_time"})
        else:
            _logger.warning(
                "lead_time coordinate is not compatible with forecast_step for all plotted data; "
                "using forecast_step as x-axis."
            )
            time_dim = "forecast_step"
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
        colors = []
        for run_id, run_data in runs.items():
            data = scores_dict.get(metric, {}).get(region, {}).get(stream, {}).get(run_id)
            if data is None or data.isnull().all():
                continue
            selected_data.append(data)
            label = run_data.get("label", run_id)
            if label != run_id:
                label = f"{run_id} - {label}"
            labels.append(label)
            run_ids.append(run_id)
            colors.append(run_data.get("color", None))

        if len(selected_data) > 0:
            _logger.info(f"Creating ratio plot for {metric} - {stream}")

            name = create_filename(
                prefix=[metric, region], middle=sorted(set(run_ids)), suffix=[stream]
            )
            plotter.ratio_plot(
                data=selected_data,
                run_ids=run_ids,
                labels=labels,
                y_dim=metric,
                tag=name,
                print_summary=print_summary,
                colors=colors,
            )


def heat_maps_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    plotter: object,
) -> None:
    """Plot heat map data for all streams and channels for a given metric and region.

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
            if data is None or data.isnull().all():
                continue

            selected_data.append(data)
            label = runs[run_id].get("label", run_id)
            if label != run_id:
                label = f"{run_id} - {label}"
            labels.append(label)
            run_ids.append(run_id)

        if len(selected_data) > 0:
            _logger.info(f"Creating heat maps for {metric} - {stream}")
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

        if len(selected_data) >= 2:
            _logger.info(f"Creating score cards for {metric} - {region} - {stream}.")
            name = "_".join([metric, region, stream])
            sc_plotter.plot(selected_data, run_ids, metric, channels_set, name)
        elif len(selected_data) == 1:
            _logger.info(
                f"Skipping score card for {metric} - {region} - {stream}: "
                f"only one run available (need at least 2 to compare)."
            )


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


def quantile_plot_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    plotter: object,
) -> None:
    """
    Create quantile-quantile (Q-Q) plots for extreme value analysis for all streams
    and channels for a given metric and region.

    Parameters
    ----------
    metric: str
        String specifying the metric to plot (should be 'qq_analysis')
    region: str
        String specifying the region to plot
    runs: dict
        Dictionary containing the config for all runs
    scores_dict : dict
        The dictionary containing all computed metrics.
    plotter:
        Plotter object to handle the plotting part. Must have a qq_plot method.
    """
    streams_set = collect_streams(runs)
    channels_set = collect_channels(scores_dict, metric, region, runs)

    for stream in streams_set:
        for ch in channels_set:
            selected_data, labels, run_ids = [], [], []
            qq_full_data = []  # Store full Q-Q datasets for detailed plotting

            for run_id, data in scores_dict[metric][region].get(stream, {}).items():
                # skip if channel is missing
                if ch not in np.atleast_1d(data.channel.values):
                    continue

                # Select channel
                data_for_channel = data.sel(channel=ch) if "channel" in data.dims else data

                # Check for NaN
                if data_for_channel.isnull().all():
                    continue

                # For qq_analysis, extract Q-Q data from attributes
                if metric == "qq_analysis" and "p_quantiles" in data_for_channel.attrs:
                    attrs = data_for_channel.attrs
                    # Convert to numpy arrays once
                    p_quantiles_arr = np.array(attrs["p_quantiles"])
                    gt_quantiles_arr = np.array(attrs["gt_quantiles"])
                    qq_deviation_arr = np.array(attrs["qq_deviation"])
                    qq_deviation_norm_arr = np.array(attrs["qq_deviation_normalized"])
                    quantile_levels = np.array(attrs["quantile_levels"])
                    extreme_low_mse = float(np.mean(np.array(attrs["extreme_low_mse"])))
                    extreme_high_mse = float(np.mean(np.array(attrs["extreme_high_mse"])))

                    qq_dataset = xr.Dataset(
                        {
                            "quantile_levels": (["quantile"], quantile_levels),
                            "p_quantiles": (["quantile"], _flatten_or_average(p_quantiles_arr)),
                            "gt_quantiles": (["quantile"], _flatten_or_average(gt_quantiles_arr)),
                            "qq_deviation": (["quantile"], _flatten_or_average(qq_deviation_arr)),
                            "qq_deviation_normalized": (
                                ["quantile"],
                                _flatten_or_average(qq_deviation_norm_arr),
                            ),
                            "extreme_low_mse": ([], extreme_low_mse),
                            "extreme_high_mse": ([], extreme_high_mse),
                        }
                    )
                    # Store extreme percentiles for plotting
                    qq_dataset.attrs["extreme_percentiles"] = tuple(attrs["extreme_percentiles"])
                    qq_full_data.append(qq_dataset)

                selected_data.append(data_for_channel)
                labels.append(runs[run_id].get("label", run_id))
                run_ids.append(run_id)

            if selected_data:
                _logger.info(f"Creating Q-Q plot for {metric} - {region} - {stream} - {ch}.")

                name = create_filename(
                    prefix=[metric, region], middle=sorted(set(run_ids)), suffix=[stream, ch]
                )

                # Check if plotter has qq_plot method and Q-Q data is available
                if hasattr(plotter, "qq_plot") and qq_full_data:
                    _logger.info(f"Creating Q-Q plot with {len(qq_full_data)} dataset(s).")
                    # Extract extreme_percentiles from dataset
                    extreme_pct = qq_full_data[0].attrs["extreme_percentiles"]
                    plotter.qq_plot(
                        qq_full_data,
                        labels,
                        tag=name,
                        metric=metric,
                        extreme_percentiles=extreme_pct,
                    )
                else:
                    # Skip plotting if no Q-Q data available
                    _logger.warning(
                        f"Q-Q data not available for {metric} - {region} - {stream} - {ch}. "
                        f"Skipping plot generation."
                    )


def _extract_psd_attrs(data_ch: xr.DataArray, fstep: int, ch: str) -> list[dict] | None:
    """Extract PSD curve data from DataArray attrs for a given fstep/channel.

    Returns a single-element list of dicts ready for the plotter, or None if keys are missing.
    """
    attrs = data_ch.attrs
    fp = f"fstep_{fstep}/"

    for prefix in (f"{fp}{ch}/", fp):
        if f"{prefix}frequencies" in attrs and f"{prefix}psd_target" in attrs:
            return [
                {
                    "frequencies": np.array(attrs[f"{prefix}frequencies"]),
                    "psd_target": np.array(attrs[f"{prefix}psd_target"]),
                    "psd_prediction": np.array(attrs[f"{prefix}psd_prediction"]),
                    "psd_method": attrs.get(f"{fp}psd_method", attrs.get("psd_method", "sht")),
                }
            ]
    return None


def _compute_psd_gap_grid(
    per_fstep_datasets: dict[int, dict], label: str = "", variable: str = ""
) -> tuple[NDArray, list[int], NDArray] | None:
    """Build a (n_fsteps, n_freq) grid of log(prediction) - log(target) for one run.

    Positive = over-prediction, negative = under-prediction — matching this codebase's existing
    ``calc_bias`` convention (``p - gt``, score.py) and the PSD ratio panels' ``pred / target``
    (ratio > 1 = over-prediction), rather than the mathematically-arbitrary opposite sign.

    Forecast steps whose frequency array doesn't match the first usable step's shape are
    dropped (with a warning) rather than raising, since PSD's frequency grid can in principle
    differ across steps depending on NaN-masking.

    Returns
    -------
    tuple[NDArray, list[int], NDArray] | None
        ``(freq, used_fsteps, grid)``, or None if fewer than two forecast steps were usable.
    """
    fsteps = sorted(per_fstep_datasets)
    ref_freq = np.asarray(per_fstep_datasets[fsteps[0]]["frequencies"])
    rows, used_fsteps = [], []
    for fstep in fsteps:
        ds = per_fstep_datasets[fstep]
        freq = np.asarray(ds["frequencies"])
        if freq.shape != ref_freq.shape:
            _logger.warning(
                f"PSD gap heatmap ({label} / {variable}): fstep {fstep} has a differently "
                "shaped frequency grid; excluding it from the heatmap."
            )
            continue
        tar = np.asarray(ds["psd_target"])
        pred = np.asarray(ds["psd_prediction"])
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.log(np.where(pred > 0, pred, np.nan)) - np.log(
                np.where(tar > 0, tar, np.nan)
            )
        rows.append(gap)
        used_fsteps.append(fstep)

    if len(rows) < 2:
        return None
    return ref_freq, used_fsteps, np.vstack(rows)


def psd_plot_metric_region(
    metric: str,
    region: str,
    runs: dict,
    scores_dict: dict,
    plotter: object,
) -> None:
    """Create PSD plots for all streams and channels for a given metric and region.

    PSD curves (frequencies, target PSD, prediction PSD) are stored in
    ``score.attrs`` by ``Scores.calc_psd`` and read back here.

    For a given forecast step, all runs are overlaid on one plot. Evolution plots (across
    forecast steps) remain one-per-run, accompanied by a per-run frequency x forecast-step
    gap heatmap and a per-run, per-forecast-step animated gif (unless disabled via
    ``plotter.psd_animate``).
    """
    streams_set = collect_streams(runs)
    channels_set = collect_channels(scores_dict, metric, region, runs)

    for stream in streams_set:
        for ch in channels_set:
            # First pass: gather each run's per-fstep PSD data for this stream/channel.
            run_fstep_datasets: dict[str, dict] = {}
            run_labels: dict[str, str] = {}
            for run_id, data in scores_dict[metric][region].get(stream, {}).items():
                if ch not in np.atleast_1d(data.channel.values):
                    continue

                data_ch = data.sel(channel=ch) if "channel" in data.dims else data
                if data_ch.isnull().all():
                    continue

                attr_fsteps = data_ch.attrs.get("attr_fsteps", [])
                if not attr_fsteps:
                    _logger.warning(f"PSD attrs missing for {run_id}/{stream}/{ch}. Skipping.")
                    continue

                per_fstep_datasets = {}
                for fstep in attr_fsteps:
                    psd_datasets = _extract_psd_attrs(data_ch, fstep, ch)
                    if psd_datasets is None:
                        continue
                    per_fstep_datasets[fstep] = psd_datasets[0]

                if not per_fstep_datasets:
                    continue

                run_fstep_datasets[run_id] = per_fstep_datasets
                run_labels[run_id] = runs[run_id].get("label", run_id)

            if not run_fstep_datasets:
                continue

            # Second pass: one combined plot per forecast step, overlaying every run
            # that has data for it.
            all_fsteps = sorted({fstep for d in run_fstep_datasets.values() for fstep in d})
            for fstep in all_fsteps:
                run_ids = [rid for rid, d in run_fstep_datasets.items() if fstep in d]
                psd_datasets = [run_fstep_datasets[rid][fstep] for rid in run_ids]
                labels = [run_labels[rid] for rid in run_ids]

                method_tag = psd_datasets[0].get("psd_method", "sht")
                name = create_filename(
                    prefix=[metric, method_tag, region],
                    middle=run_ids,
                    suffix=[stream, ch, f"fstep{fstep}"],
                )
                plotter.psd_plot(
                    psd_datasets,
                    labels,
                    tag=name,
                    variable=ch,
                    forecast_step=str(fstep),
                )

            # Third pass: per-run evolution plot, per-run gap heatmap (colour scale shared
            # across every run being compared for this stream/channel, computed up front),
            # plus (if enabled) per-run per-fstep frames assembled into an animated gif.
            run_gap_grids: dict[str, tuple[NDArray, list[int], NDArray]] = {}
            for run_id, per_fstep_datasets in run_fstep_datasets.items():
                result = _compute_psd_gap_grid(
                    per_fstep_datasets, label=run_labels[run_id], variable=ch
                )
                if result is not None:
                    run_gap_grids[run_id] = result

            if run_gap_grids:
                all_gap_vals = np.concatenate(
                    [grid.ravel() for _, _, grid in run_gap_grids.values()]
                )
                gap_min = float(np.nanmin(all_gap_vals))
                gap_max = float(np.nanmax(all_gap_vals))
                # Extend to include 0 (perfect prediction) so TwoSlopeNorm's centre is always
                # valid, even if every run happens to be biased the same direction. TwoSlopeNorm
                # requires vmin < vcenter(=0) < vmax strictly, so nudge either bound off zero
                # with an epsilon scaled to the data when it would otherwise land exactly (or
                # only) on one side.
                eps = max(abs(gap_min), abs(gap_max), 1e-9) * 1e-3
                gap_vmin = min(gap_min, -eps)
                gap_vmax = max(gap_max, eps)

            for run_id, per_fstep_datasets in run_fstep_datasets.items():
                if len(per_fstep_datasets) < 2:
                    continue

                method_tag = next(iter(per_fstep_datasets.values())).get("psd_method", "sht")
                evo_name = create_filename(
                    prefix=[metric, method_tag, region],
                    middle=[run_id],
                    suffix=[stream, ch, "evolution"],
                )
                plotter.psd_evolution_plot(
                    per_fstep_datasets,
                    tag=evo_name,
                    variable=ch,
                    label=run_labels[run_id],
                )

                if run_id in run_gap_grids:
                    freq, used_fsteps, grid = run_gap_grids[run_id]
                    heatmap_name = create_filename(
                        prefix=[metric, method_tag, region],
                        middle=[run_id],
                        suffix=[stream, ch, "gap_heatmap"],
                    )
                    plotter.psd_gap_heatmap(
                        freq,
                        used_fsteps,
                        grid,
                        vmin=gap_vmin,
                        vmax=gap_vmax,
                        tag=heatmap_name,
                        variable=ch,
                        label=run_labels[run_id],
                        psd_method=method_tag,
                    )

                if not getattr(plotter, "psd_animate", True):
                    continue

                # Single-run frames (one per forecast step) -> animated gif. The literal
                # "frame" token keeps these filenames distinct from pass 2's combined-run
                # per-fstep plots, which degenerate to the same middle=[run_id] when only
                # one run is being compared.
                frame_paths = []
                for fstep in sorted(per_fstep_datasets):
                    frame_name = create_filename(
                        prefix=[metric, method_tag, region],
                        middle=[run_id],
                        suffix=[stream, ch, "frame", f"fstep{int(fstep):03d}"],
                    )
                    frame_paths.append(
                        plotter.psd_plot(
                            [per_fstep_datasets[fstep]],
                            [run_labels[run_id]],
                            tag=frame_name,
                            variable=ch,
                            forecast_step=str(fstep),
                        )
                    )
                anim_name = create_filename(
                    prefix=[metric, method_tag, region],
                    middle=[run_id],
                    suffix=[stream, ch, "animation"],
                )
                plotter.psd_gif(frame_paths, tag=anim_name)
    _logger.info(f"PSD plots saved successfully into: {plotter.out_plot_dir_psd}")


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
