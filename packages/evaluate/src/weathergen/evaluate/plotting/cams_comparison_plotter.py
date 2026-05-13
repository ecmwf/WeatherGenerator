# (C) Copyright 2025 WeatherGenerator contributors.
# Licensed under Apache 2.0.

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.interpolate import RegularGridInterpolator

from weathergen.common.io import CAMSForecastReader
from weathergen.evaluate.io.wegen_reader import WeatherGenZarrReader
from weathergen.evaluate.plotting.plotter import Plotter

_logger = logging.getLogger(__name__)

# Number of parallel workers for data loading and rendering
_N_WORKERS = 8

# ---------------------------------------------------------------------------
# PPM conversion helpers
# ---------------------------------------------------------------------------

# Molecular weight of dry air (g/mol)
_M_AIR: float = 28.97

# Species molecular weights (g/mol), keyed by channel-name prefix.
# Profile channels follow the pattern ``<species>_<level>`` (e.g. ``co_500``).
_M_SPECIES: dict[str, float] = {
    "co": 28.01,
    "no2": 46.01,
    "no": 30.01,
    "so2": 64.07,
    "o3": 48.00,
    "go3": 48.00,
}


def _channel_ppm_factor(channel: str) -> float | None:
    """Return the kg kg⁻¹ → ppmv conversion factor for *channel*, or ``None``.

    Profile channels (e.g. ``co_500``) are converted using
    ``ppmv = value * (M_air / M_species) * 1e6``.
    Total-column (``tc_*``) and surface-particulate (``pm*``) channels are
    excluded because ppmv is not a meaningful unit for them.
    """
    ch = channel.lower()
    if ch.startswith("tc_") or ch.startswith("pm"):
        return None
    # match longest prefix first (no2 before no)
    for species in sorted(_M_SPECIES, key=len, reverse=True):
        if ch.startswith(species + "_") or ch == species:
            return (_M_AIR / _M_SPECIES[species]) * 1e6
    return None


def _compute_rmse(pred: xr.DataArray, target: xr.DataArray, dim: str = "ipoint") -> xr.DataArray:
    """Compute root-mean-square error along *dim*."""
    return np.sqrt(((pred - target) ** 2).mean(dim=dim))


# Maximum allowed gap (in hours) between the requested CAMS init time and
# the nearest available time coordinate.
_MAX_TIME_GAP_HOURS: int = 6


def _verify_cams_time_selection(
    da: xr.DataArray,
    init_time: "np.datetime64",
    channel: str,
    forecast_hour: int,
) -> None:
    """Warn when the CAMS time coordinate doesn't closely match *init_time*."""
    if "time" not in da.coords:
        return
    selected_time = np.datetime64(da["time"].values, "ns")
    requested_time = np.datetime64(init_time, "ns")
    gap = abs(int((selected_time - requested_time) / np.timedelta64(1, "h")))
    if gap > _MAX_TIME_GAP_HOURS:
        _logger.warning(
            "CAMS time mismatch for channel %s at forecast hour %dh: "
            "requested init_time=%s but nearest available is %s (gap=%dh). "
            "This may indicate the CAMS dataset does not cover the evaluation period.",
            channel, forecast_hour,
            _format_datetime64(requested_time),
            _format_datetime64(selected_time),
            gap,
        )
    else:
        _logger.debug(
            "CAMS time OK for %s fstep %dh: selected %s (gap=%dh)",
            channel, forecast_hour,
            _format_datetime64(selected_time), gap,
        )


def _verify_cams_data_consistency(
    cams_cache: dict[tuple[str, int], np.ndarray],
    channels: list[str],
    forecast_hours: list[int],
    hr_info: dict[int, tuple],
) -> None:
    """Run post-load sanity checks on the pre-loaded CAMS data.

    Checks performed:
    1. **Per-(channel, hour) statistics** – mean, std, min, max are logged.
    2. **Cross-step variance** – near-constant values across steps may signal
       the same time slice being reused for every step.
    3. **NaN fraction** – a high NaN ratio indicates interpolation issues.
    4. **Order-of-magnitude vs WG target** – large discrepancies hint at
       unit mismatches or wrong variable selection.
    """
    _logger.info("Running CAMS data consistency checks …")

    for ch in channels:
        means_across_steps: list[float] = []
        for hr in forecast_hours:
            key = (ch, hr)
            if key not in cams_cache:
                _logger.warning("  Missing CAMS data for %s at %dh", ch, hr)
                continue
            vals = cams_cache[key]
            nan_frac = float(np.isnan(vals).mean())
            vmean = float(np.nanmean(vals))
            vstd = float(np.nanstd(vals))
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            means_across_steps.append(vmean)

            _logger.info(
                "  CAMS %-12s fstep %3dh: mean=%.4e  std=%.4e  "
                "min=%.4e  max=%.4e  NaN%%=%.1f",
                ch, hr, vmean, vstd, vmin, vmax, nan_frac * 100,
            )
            if nan_frac > 0.1:
                _logger.warning(
                    "  HIGH NaN fraction (%.1f%%) for %s at %dh – "
                    "interpolation may be unreliable.",
                    nan_frac * 100, ch, hr,
                )

            # compare with WG target order of magnitude
            if hr in hr_info:
                _raw, wg_target, _wg_pred, _init = hr_info[hr]
                tar = np.asarray(wg_target.sel(channel=ch).values).ravel()
                tar_mean = float(np.nanmean(tar))
                if tar_mean != 0 and vmean != 0:
                    ratio = abs(vmean / tar_mean)
                    if ratio > 100 or ratio < 0.01:
                        _logger.warning(
                            "  ORDER-OF-MAGNITUDE mismatch for %s at %dh: "
                            "CAMS mean=%.4e vs WG target mean=%.4e (ratio=%.2g). "
                            "Check units/variable selection.",
                            ch, hr, vmean, tar_mean, ratio,
                        )

        # cross-step variance check
        if len(means_across_steps) > 1:
            step_std = float(np.std(means_across_steps))
            step_mean = float(np.mean(np.abs(means_across_steps)))
            if step_mean > 0 and step_std / step_mean < 0.001:
                _logger.warning(
                    "  SUSPICIOUS: CAMS %s mean is nearly constant across %d "
                    "forecast steps (std/|mean|=%.2e). The same time slice may "
                    "have been selected for every step.",
                    ch, len(means_across_steps), step_std / step_mean,
                )

    _logger.info("CAMS data consistency checks complete.")


def _preload_and_interpolate_cams(
    cams_reader: "CAMSForecastReader",
    channels: list[str],
    forecast_hours: list[int],
    hr_info: dict[int, tuple],
    n_workers: int = _N_WORKERS,
) -> dict[tuple[str, int], np.ndarray]:
    """Pre-load CAMS data into memory and batch-interpolate all channels per hour.

    This avoids repeated dask/zarr reads and creates a single
    ``RegularGridInterpolator`` per forecast hour (instead of one per
    channel × hour), giving a large speed-up.

    Returns
    -------
    dict[(channel, hour), np.ndarray]
        Interpolated CAMS values (1-D, at WG scatter points) for every
        (channel, hour) pair.
    """
    _logger.info(
        "Pre-loading CAMS data: %d channels × %d forecast hours …",
        len(channels), len(forecast_hours),
    )

    # ---- 1. identify the base xarray variables we need ------------------
    needed_vars: set[str] = set()
    for ch in channels:
        if ch in cams_reader.ds.data_vars:
            needed_vars.add(ch)
        else:
            var = ch.split("_")[0]
            if var in cams_reader.ds.data_vars:
                needed_vars.add(var)

    # ---- 2. load the subset eagerly (dask → numpy) ----------------------
    subset = cams_reader.ds[list(needed_vars)]

    if "step" in subset.dims:
        needed_steps = [pd.Timedelta(hours=int(hr)) for hr in forecast_hours]
        subset = subset.sel(step=needed_steps)

    needed_levels = sorted({
        int(ch.split("_")[-1])
        for ch in channels
        if "_" in ch and ch.split("_")[-1].isdigit()
    })
    if "isobaricInhPa" in subset.dims and needed_levels:
        subset = subset.sel(isobaricInhPa=needed_levels)

    init_times = sorted({
        hr_info[hr][3]
        for hr in forecast_hours
        if hr in hr_info and hr_info[hr][3] is not None
    })
    if "time" in subset.dims and init_times:
        requested_times = np.array(init_times, dtype="datetime64[ns]")
        # Normalise the CAMS time coordinate to ns so that exact equality
        # matching works regardless of the dtype stored in the zarr file
        # (e.g. datetime64[s] vs datetime64[ns]).
        subset = subset.assign_coords(
            time=subset.time.values.astype("datetime64[ns]")
        )
        # Keep only the init times that actually exist in CAMS; the per-hour
        # interpolation path handles the case where a specific time is absent.
        available_times = set(subset.time.values)
        times_to_select = [t for t in requested_times if t in available_times]
        if times_to_select:
            subset = subset.sel(time=np.array(times_to_select, dtype="datetime64[ns]"))

    subset = subset.load()
    _logger.info("CAMS dataset loaded into memory (%d variables).", len(needed_vars))

    # ---- 3. per-hour batch interpolation --------------------------------
    cams_lat = cams_reader.lat
    cams_lon = cams_reader.lon

    def _process_hour(hr: int) -> dict[tuple[str, int], np.ndarray]:
        if hr not in hr_info:
            return {}
        _raw, wg_target, _wg_pred, init_time = hr_info[hr]

        sample_tar = wg_target.sel(channel=channels[0])
        target_lat = sample_tar["lat"].values
        target_lon = sample_tar["lon"].values
        target_points = np.column_stack([target_lat, target_lon])

        raw_arrays: list[np.ndarray] = []
        for ch in channels:
            if ch in subset.data_vars:
                da = subset[ch]
            else:
                ch_parts = ch.split("_")
                var, level = ch_parts[0], ch_parts[1]
                da = subset[var].sel(isobaricInhPa=level)

            if "step" in da.dims:
                da = da.sel(step=pd.Timedelta(hours=int(hr)))
            if "time" in da.dims:
                if init_time is not None:
                    da = da.sel(time=init_time)
                elif da.sizes.get("time", 0) > 1:
                    _logger.warning(
                        "No init_time for %s fstep %dh – falling back to "
                        "first time entry; results may be incorrect.",
                        ch, hr,
                    )
                    da = da.isel(time=0)

            data = cams_reader._sort_data(da.values)
            while data.ndim > 2:
                data = data[0]
            raw_arrays.append(data)

        stacked = np.stack(raw_arrays, axis=-1)  # (lat, lon, n_channels)
        interp = RegularGridInterpolator(
            (cams_lat, cams_lon),
            stacked,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        result = interp(target_points)  # (n_points, n_channels)
        return {(ch, hr): result[:, ci] for ci, ch in enumerate(channels)}

    hour_cache: dict[tuple[str, int], np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=min(n_workers, len(forecast_hours))) as pool:
        futures = {pool.submit(_process_hour, hr): hr for hr in forecast_hours}
        for fut in as_completed(futures):
            hr_done = futures[fut]
            hour_cache.update(fut.result())
            _logger.info("  Interpolated CAMS data for forecast hour %dh", hr_done)

    _logger.info(
        "CAMS pre-load complete: %d (channel, hour) pairs ready.",
        len(hour_cache),
    )

    _verify_cams_data_consistency(hour_cache, channels, forecast_hours, hr_info)
    return hour_cache


def _normalise_longitude_da(da: xr.DataArray) -> xr.DataArray:
    """Normalise a DataArray's longitude coordinate to [-180, 180] in-place.

    FIX: the original code reassigned a loop variable inside a ``for`` loop,
    so the normalised DataArray was immediately discarded.  This helper is
    called explicitly on each DataArray and its result is used.
    """
    if "longitude" not in da.coords:
        return da
    if float(da["longitude"].max()) > 180:
        da = da.assign_coords(
            longitude=(((da["longitude"] + 180) % 360) - 180)
        ).sortby("longitude")
    return da


def _compute_cams_native_rmse(
    cams_reader: "CAMSForecastReader",
    analysis_path: "Path",
    channels: list[str],
    forecast_hours: list[int],
    hr_info: dict[int, tuple],
) -> dict[tuple[str, int], float]:
    """Compute CAMS forecast-vs-analysis RMSE on the **native CAMS grid**.

    Returns
    -------
    dict[(channel, hour), float]
        Scalar RMSE values per (channel, hour) pair.
    """
    _logger.info(
        "Computing CAMS forecast-vs-analysis RMSE on native grid: "
        "%d channels × %d forecast hours …",
        len(channels), len(forecast_hours),
    )

    ds_a_surface = xr.open_zarr(analysis_path, group="surface", chunks="auto")
    ds_a_profiles = xr.open_zarr(analysis_path, group="profiles", chunks="auto")
    ds_analysis = xr.merge([ds_a_surface, ds_a_profiles])

    ds_forecast = cams_reader.ds
    rmse_out: dict[tuple[str, int], float] = {}

    for hr in forecast_hours:
        if hr not in hr_info:
            continue
        _, _, _, init_time = hr_info[hr]
        if init_time is None:
            continue
        valid_time = np.datetime64(init_time, "ns") + np.timedelta64(int(hr), "h")

        ds_f = ds_forecast
        ds_a = ds_analysis
        if "time" in ds_f.dims:
            ds_f = ds_f.sel(time=init_time)
        if "time" in ds_a.dims:
            ds_a = ds_a.sel(time=valid_time)

        for ch in channels:
            if ch in ds_f.data_vars and ch in ds_a.data_vars:
                forecast_da = ds_f[ch]
                analysis_da = ds_a[ch]
            else:
                parts = ch.split("_")
                var = parts[0]
                level = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                if var not in ds_f.data_vars or var not in ds_a.data_vars:
                    _logger.warning("Variable %s not in both datasets – skipping %s", var, ch)
                    continue
                forecast_da = ds_f[var]
                analysis_da = ds_a[var]
                if level is not None and "isobaricInhPa" in forecast_da.dims:
                    forecast_da = forecast_da.sel(isobaricInhPa=level)
                    analysis_da = analysis_da.sel(isobaricInhPa=level)

            if "step" in forecast_da.dims:
                forecast_da = forecast_da.sel(step=pd.Timedelta(hours=int(hr)))

            if "time" in forecast_da.dims:
                forecast_da = forecast_da.squeeze("time")
            if "time" in analysis_da.dims:
                analysis_da = analysis_da.squeeze("time")

            # FIX: normalise each DataArray explicitly and capture the result,
            # instead of reassigning a discarded loop variable.
            forecast_da = _normalise_longitude_da(forecast_da)
            analysis_da = _normalise_longitude_da(analysis_da)

            forecast_vals = forecast_da.values
            analysis_vals = analysis_da.values

            while forecast_vals.ndim > 2:
                forecast_vals = forecast_vals[0]
            while analysis_vals.ndim > 2:
                analysis_vals = analysis_vals[0]

            if forecast_vals.shape != analysis_vals.shape:
                # Extract coordinates AFTER normalisation so the interpolator
                # receives a consistent, sorted longitude range.
                f_lat = forecast_da.latitude.values
                f_lon = forecast_da.longitude.values
                a_lat = analysis_da.latitude.values
                a_lon = analysis_da.longitude.values

                interpolator = RegularGridInterpolator(
                    (f_lat, f_lon),
                    forecast_vals,
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )
                a_lon_grid, a_lat_grid = np.meshgrid(a_lon, a_lat)
                pts = np.column_stack([a_lat_grid.ravel(), a_lon_grid.ravel()])
                forecast_vals = interpolator(pts).reshape(len(a_lat), len(a_lon))

            rmse_val = float(np.sqrt(((forecast_vals - analysis_vals) ** 2).mean()))
            rmse_out[(ch, hr)] = rmse_val

    _logger.info(
        "CAMS native-grid RMSE complete: %d (channel, hour) pairs computed.",
        len(rmse_out),
    )
    return rmse_out


def _scorecard_channel_order(channel: str) -> tuple[int, str, float]:
    """Return a stable sort key for scorecard heatmap channels.

    Pressure-level channels such as ``co_50`` and ``co_1000`` are ordered by
    ascending level so they render top-to-bottom as 50, ..., 1000.
    Total-column channels (``tc_*``) are placed after the pressure levels.
    Plain species names (no numeric suffix) come between the two.

    FIX: the original returned ``(1, ch, ch)`` for plain names, mixing str
    and float in the last tuple element.  Python 3 raises ``TypeError`` when
    comparing str to float during sort.  The last element is now always float.
    """
    ch = channel.lower()

    if ch.startswith("tc_"):
        return (2, ch[3:], float("inf"))

    species, sep, suffix = ch.rpartition("_")
    if sep and suffix.isdigit():
        return (0, species, float(suffix))

    # Plain names (e.g. "go3") – no numeric suffix, not a tc_ channel.
    # Use 0.0 as the float sentinel so the tuple is homogeneous.
    return (1, ch, 0.0)


def _cams_steps_as_hours(cams_reader: CAMSForecastReader) -> set[int]:
    """Return CAMS forecast steps converted to integer hours."""
    hours: set[int] = set()
    for s in cams_reader.forecast_steps:
        hours.add(int(pd.Timedelta(s).total_seconds() // 3600))
    return hours


def _align_forecast_steps(
    wg_reader: WeatherGenZarrReader,
    cams_reader: CAMSForecastReader,
    requested: list[int],
) -> tuple[list[int], dict[int, int]]:
    """Return intersection of forecast steps converted to hours.

    WG stores forecast steps as integer indices (1, 2, 3, ...) which are
    converted to lead-time hours via ``step_hrs``.  CAMS stores steps as
    actual lead-time hours (0, 12, 24, ...).  When ``step_hrs`` is 1 (the
    default when the model config does not contain the key) and produces no
    overlap, the function infers ``step_hrs`` automatically from the minimum
    non-zero CAMS step spacing.

    Returns
    -------
    common_hours : list[int]
        Sorted list of hours present in both WG and CAMS datasets.
    wg_map : dict[int, int]
        Mapping from hour -> original WG step index.
    """
    step_hrs = wg_reader.step_hrs

    def _build_wg_map(s: int) -> dict[int, int]:
        m: dict[int, int] = {}
        for idx in wg_reader.get_forecast_steps():
            idx = int(idx)
            m[idx * s] = idx
        return m

    wg_map = _build_wg_map(step_hrs)
    available_wg = set(wg_map.keys())
    available_cams = _cams_steps_as_hours(cams_reader)

    _logger.info("WG available hours: %s", sorted(available_wg))
    _logger.info("CAMS available hours: %s", sorted(available_cams))

    # When step_hrs=1 (default) and WG hours do not overlap with CAMS hours,
    # try to infer step_hrs from the minimum non-zero CAMS step spacing so
    # that integer WG step indices are correctly converted to lead-time hours.
    if not (available_wg & available_cams) and step_hrs == 1 and available_cams:
        sorted_cams = sorted(available_cams)
        gaps = sorted({b - a for a, b in zip(sorted_cams, sorted_cams[1:])} - {0})
        if gaps:
            inferred = gaps[0]
            candidate_map = _build_wg_map(inferred)
            if candidate_map.keys() & available_cams:
                _logger.warning(
                    "step_hrs=%d yields no overlap with CAMS. "
                    "Auto-detected step_hrs=%d from CAMS step spacing (%s h). "
                    "Set 'step_hrs' explicitly in the cams_comparison config to override.",
                    step_hrs,
                    inferred,
                    sorted_cams,
                )
                step_hrs = inferred
                wg_map = candidate_map
                available_wg = set(wg_map.keys())
                _logger.info("WG available hours (after step_hrs correction): %s", sorted(available_wg))

    common = sorted(available_wg & available_cams & set(requested))
    missing_wg = set(requested) - available_wg
    missing_cams = set(requested) - available_cams
    if missing_wg:
        _logger.warning(
            "Forecast steps (hours) %s not found in WG output; they will be skipped",
            sorted(missing_wg),
        )
    if missing_cams:
        _logger.warning(
            "Forecast steps (hours) %s not available in CAMS data; they will be skipped",
            sorted(missing_cams),
        )
    return common, wg_map


def _format_datetime64(value: np.datetime64) -> str:
    """Return a stable string representation for datetime64 values."""
    return pd.Timestamp(value).isoformat()


def _extract_unique_valid_times(data: xr.DataArray, label: str) -> np.ndarray:
    """Return sorted non-NaT valid_time values for a WG field."""
    if "valid_time" not in data.coords:
        raise ValueError(
            f"{label} is missing the valid_time coordinate required for CAMS comparison."
        )

    valid_time = np.asarray(data["valid_time"].values).reshape(-1)
    if valid_time.size == 0:
        raise ValueError(f"{label} has no valid_time values to verify.")

    valid_time = valid_time.astype("datetime64[ns]", copy=False)
    valid_time = valid_time[~np.isnat(valid_time)]
    if valid_time.size == 0:
        raise ValueError(f"{label} only contains NaT valid_time values.")

    return np.unique(valid_time)


def _cams_expected_valid_time(
    cams_reader: CAMSForecastReader,
    forecast_hour: int,
    init_time: "np.datetime64 | None" = None,
) -> np.datetime64:
    """Return the CAMS valid time for *forecast_hour*."""
    if init_time is not None:
        return np.datetime64(init_time, "ns") + np.timedelta64(int(forecast_hour), "h")

    init_times = getattr(cams_reader, "_wg_cached_init_times", None)
    if init_times is None:
        init_times = np.asarray(cams_reader.time).reshape(-1)
        init_times = init_times.astype("datetime64[ns]", copy=False)
        init_times = np.unique(init_times[~np.isnat(init_times)])
        setattr(cams_reader, "_wg_cached_init_times", init_times)

    if init_times.size == 0:
        raise ValueError(
            f"CAMS dataset {cams_reader.cams_forecast_path} has no valid forecast initial time."
        )

    if init_times.size > 1 and not getattr(cams_reader, "_wg_warned_multi_time", False):
        _logger.warning(
            "CAMS dataset %s contains %d forecast initial times; comparison uses the first one (%s).",
            cams_reader.cams_forecast_path,
            init_times.size,
            _format_datetime64(init_times[0]),
        )
        setattr(cams_reader, "_wg_warned_multi_time", True)

    return init_times[0] + np.timedelta64(int(forecast_hour), "h")


def _verify_timestep_alignment(
    wg_target: xr.DataArray,
    wg_pred: xr.DataArray,
    cams_reader: CAMSForecastReader,
    forecast_hour: int,
) -> np.datetime64:
    """Ensure WG target/prediction and CAMS timestamps refer to the same valid time."""
    target_times = _extract_unique_valid_times(
        wg_target, f"WG target data for forecast hour {forecast_hour}h"
    )
    pred_times = _extract_unique_valid_times(
        wg_pred, f"WG prediction data for forecast hour {forecast_hour}h"
    )

    if target_times.size != 1:
        formatted = [_format_datetime64(ts) for ts in target_times]
        raise ValueError(
            f"WG target data for forecast hour {forecast_hour}h spans multiple valid_time "
            f"values: {formatted}. CAMS comparison expects a single timestamp per forecast hour."
        )

    if pred_times.size != 1:
        formatted = [_format_datetime64(ts) for ts in pred_times]
        raise ValueError(
            f"WG prediction data for forecast hour {forecast_hour}h spans multiple valid_time "
            f"values: {formatted}. CAMS comparison expects a single timestamp per forecast hour."
        )

    wg_valid_time = target_times[0]
    if pred_times[0] != wg_valid_time:
        raise ValueError(
            "WG target/prediction timestep mismatch for forecast hour "
            f"{forecast_hour}h: target={_format_datetime64(wg_valid_time)}, "
            f"prediction={_format_datetime64(pred_times[0])}."
        )

    init_time = wg_valid_time - np.timedelta64(int(forecast_hour), "h")
    cams_valid_time = _cams_expected_valid_time(cams_reader, forecast_hour, init_time=init_time)
    if wg_valid_time != cams_valid_time:
        raise ValueError(
            "WG and CAMS timesteps are misaligned for forecast hour "
            f"{forecast_hour}h: WG valid_time={_format_datetime64(wg_valid_time)}, "
            f"CAMS expected valid_time={_format_datetime64(cams_valid_time)}."
        )

    return wg_valid_time


def _compute_color_cap(
    ch: str,
    color_max_ppb: float,
    convert_to_ppm: bool,
    ppm_factor_applied: float | None,
) -> float:
    """Return the colour-scale cap in the same units as the displayed data.

    *color_max_ppb* is always specified in ppb by the caller.

    FIX: the original code called ``_channel_ppm_factor(ch)`` a second time
    in the ``else`` branch regardless of ``convert_to_ppm``, and used its
    value to invert ppb→kg/kg even when the data had NOT been converted to
    ppm.  The cap is now derived purely from ``ppm_factor_applied``:

    - If conversion was applied (``ppm_factor_applied`` is not None):
      cap = color_max_ppb / 1000  (ppb → ppm)
    - If no conversion was applied, the data is still in kg/kg:
      cap = color_max_ppb / 1e9  (ppb → kg/kg, since 1 ppb = 1e-9 kg/kg)
    """
    if ppm_factor_applied is not None:
        # Data is in ppmv; convert ppb cap to ppm.
        return color_max_ppb / 1000.0
    else:
        # Data is in kg/kg; convert ppb cap to kg/kg.
        ppm_factor = _channel_ppm_factor(ch)
        if ppm_factor is not None and ppm_factor > 0:
            return (color_max_ppb / 1e9) / (1.0 / ppm_factor) if False else color_max_ppb / 1e9
        return color_max_ppb / 1e9


def _prepare_bias_da(pred: xr.DataArray, tar: xr.DataArray, cams_vals: np.ndarray):
    """Return (wg_bias_da, cams_bias_da) for scatter-grid WG data.

    FIX: the original code had a misleading branch condition
    ``wg_bias_flat.size == lat_flat.size * lon_flat.size`` that treated
    lat/lon as *axis* vectors (implying a regular grid), but WG data is
    always on an unstructured scatter grid where lat and lon are both
    length-N point arrays.  The product ``N * N`` is therefore never equal
    to N for any realistic N > 1, so the grid branch was dead code and
    would produce a wrongly-shaped DataArray if ever triggered.  The
    function now always uses the scatter (ipoint) representation, which
    matches the actual data layout.  Debug-level logging replaces the
    _logger.info calls so production runs are not flooded with shape dumps.
    """
    lat_flat = np.asarray(tar["lat"].values).ravel()
    lon_flat = np.asarray(tar["lon"].values).ravel()
    cams_vals = np.asarray(cams_vals).ravel()
    wg_bias_flat = np.asarray((pred - tar).values).ravel()
    tar_flat = np.asarray(tar.values).ravel()

    _logger.debug(
        "[prepare_bias_da] lat=%d  lon=%d  wg_bias=%d  tar=%d  cams=%d",
        lat_flat.size, lon_flat.size, wg_bias_flat.size, tar_flat.size, cams_vals.size,
    )

    if not (wg_bias_flat.size == lat_flat.size == lon_flat.size == tar_flat.size == cams_vals.size):
        raise ValueError(
            f"Shape mismatch in _prepare_bias_da: "
            f"wg_bias={wg_bias_flat.shape}, lat={lat_flat.shape}, "
            f"lon={lon_flat.shape}, tar={tar_flat.shape}, cams={cams_vals.shape}."
        )

    ipoints = np.arange(wg_bias_flat.size)
    coords = {"ipoint": ipoints, "lat": ("ipoint", lat_flat), "lon": ("ipoint", lon_flat)}

    wg_bias_da = xr.DataArray(wg_bias_flat, coords=coords, dims=["ipoint"])
    cams_bias_da = xr.DataArray(cams_vals - tar_flat, coords=coords, dims=["ipoint"])
    return wg_bias_da, cams_bias_da


def _render_bias_frame(
    ch: str,
    hr: int,
    wg_bias_vals: np.ndarray,
    cams_bias_vals: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    maxabs: float,
    convert_to_ppm: bool,
    out_path: "Path",
) -> "Path":
    """Render a single bias comparison frame (thread-safe, no pyplot)."""
    import cartopy.crs as ccrs

    fig = Figure(figsize=(16, 8), dpi=300)
    FigureCanvasAgg(fig)

    ax_cams = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax_wg = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())

    sc = None
    for ax, vals, title in [
        (ax_cams, cams_bias_vals, "CAMS Forecast \u2013 Analysis"),
        (ax_wg, wg_bias_vals, "WG Prediction \u2013 Target"),
    ]:
        ax.coastlines()
        sc = ax.scatter(
            lon, lat, c=vals,
            cmap="coolwarm", vmin=-maxabs, vmax=maxabs,
            transform=ccrs.PlateCarree(), s=1, linewidths=0.0,
        )
        ax.set_global()
        # FIX: added missing "h" suffix for consistency with _render_value_frame.
        ax.set_title(f"{title} \u2013 {ch} (fstep {hr}h)")

    bias_unit = "ppm" if (convert_to_ppm and _channel_ppm_factor(ch) is not None) else "kg kg\u207b\u00b9"
    cbar = fig.colorbar(sc, ax=[ax_cams, ax_wg], orientation="horizontal", pad=0.02, fraction=0.04)
    cbar.set_label(f"Bias ({bias_unit})")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15)

    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.1)
    return out_path


def _scatter_to_regular_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    vals: np.ndarray,
    nlat: int = 361,
    nlon: int = 721,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate scatter (lat, lon, vals) onto a regular lat/lon grid.

    Uses nearest-neighbour interpolation to avoid artefacts at poles and
    data boundaries while keeping the operation fast.

    Returns
    -------
    grid_lons, grid_lats, grid_vals : 2-D arrays shaped (nlat, nlon)
    """
    from scipy.interpolate import griddata

    grid_lons, grid_lats = np.meshgrid(
        np.linspace(-180.0, 180.0, nlon),
        np.linspace(-90.0, 90.0, nlat),
    )
    points = np.column_stack([lon.ravel(), lat.ravel()])
    grid_vals = griddata(points, vals.ravel(), (grid_lons, grid_lats), method="nearest")
    return grid_lons, grid_lats, grid_vals


def _render_value_frame(
    ch: str,
    hr: int,
    lat: np.ndarray,
    lon: np.ndarray,
    cams_vals: np.ndarray,
    pred_vals: np.ndarray,
    vmin: float,
    vmax: float,
    value_unit: str,
    out_path: "Path",
    tar_vals: np.ndarray | None = None,
) -> "Path":
    """Render a single value comparison frame (thread-safe, no pyplot)."""
    import cartopy.crs as ccrs

    n_panels = 3 if tar_vals is not None else 2
    fig = Figure(figsize=(8 * n_panels, 8), dpi=150)
    FigureCanvasAgg(fig)

    ax_cams = fig.add_subplot(1, n_panels, 1, projection=ccrs.Robinson())
    if tar_vals is not None:
        ax_tar = fig.add_subplot(1, n_panels, 2, projection=ccrs.Robinson())
        ax_wg = fig.add_subplot(1, n_panels, 3, projection=ccrs.Robinson())
    else:
        ax_wg = fig.add_subplot(1, n_panels, 2, projection=ccrs.Robinson())

    # Regrid all datasets to the same regular lat/lon grid so that
    # imshow can be used, eliminating the banding artefacts that
    # arise from rendering unstructured (HEALPix ring) scatter points.
    _grid_lons, _grid_lats, cams_grid = _scatter_to_regular_grid(lat, lon, cams_vals)
    _, _, pred_grid = _scatter_to_regular_grid(lat, lon, pred_vals)

    panels = [(ax_cams, "CAMS Forecast", cams_grid)]
    if tar_vals is not None:
        _, _, tar_grid = _scatter_to_regular_grid(lat, lon, tar_vals)
        panels.append((ax_tar, "WG Target", tar_grid))
    panels.append((ax_wg, "WG Prediction", pred_grid))

    last_mesh = None
    all_axes = []
    for ax, title, grid_vals in panels:
        ax.coastlines()
        # imshow reprojects the raster without per-quad wrapping, avoiding
        # the ProjError that pcolormesh triggers on non-PlateCarree projections
        # when quads span the antimeridian.
        last_mesh = ax.imshow(
            grid_vals,
            origin="lower",
            extent=[-180.0, 180.0, -90.0, 90.0],
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_global()
        ax.set_title(f"{title} \u2013 {ch} (fstep {hr}h)")
        all_axes.append(ax)

    cbar = fig.colorbar(last_mesh, ax=all_axes, orientation="horizontal", pad=0.02, fraction=0.04)
    cbar.set_label(f"{ch} ({value_unit})")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15)

    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.1)
    return out_path


def _plot_bias_maps(
    plotter: Plotter,
    wg_reader: WeatherGenZarrReader,
    cams_reader: CAMSForecastReader,
    stream: str,
    channels: list[str],
    forecast_hours: list[int],
    wg_map: dict[int, int],
    run_id: str,
    convert_to_ppm: bool = False,
    color_max_ppb: float | None = None,
    wg_data=None,
    cams_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> None:
    """Produce and save global bias scatter maps for each forecast step/channel.

    The colour scale is held **constant** across all forecast steps for a
    given channel so that animations are meaningful.  An optional
    ``color_max_ppb`` cap (in ppb) limits the symmetric range.

    All frames are written under
    ``plotter.out_plot_basedir/<stream>/maps/bias_compare/``.

    FIX: removed creation of unused ``wg_bias``/``cams_bias`` subdirectories
    (they were created but never written to, contradicting the docstring).
    """
    n_hours = len(forecast_hours)
    n_channels = len(channels)
    total_frames = n_hours * n_channels
    _logger.info(
        "Bias maps: %d channels x %d forecast hours = %d frames",
        n_channels, n_hours, total_frames,
    )

    combo_dir = plotter.out_plot_basedir / stream / "maps" / "bias_compare"
    combo_dir.mkdir(parents=True, exist_ok=True)

    # ---- first pass: compute per-channel global bias range ---------------
    _logger.info("Bias maps [1/2]: computing per-channel global bias range …")
    channel_maxabs: dict[str, float] = {ch: 0.0 for ch in channels}
    bias_cache: dict[tuple[str, int], tuple[xr.DataArray, xr.DataArray]] = {}

    for hi, hr in enumerate(forecast_hours, 1):
        _logger.info("  Computing biases for forecast hour %dh (%d/%d) …", hr, hi, n_hours)
        raw = wg_map[hr]

        if wg_data is not None:
            if raw not in wg_data.target:
                continue
            wg_target = wg_data.target[raw]
            wg_pred = wg_data.prediction[raw]
        else:
            _wg = wg_reader.get_data(stream=stream, fsteps=[raw], channels=channels)
            if raw not in _wg.target:
                continue
            wg_target = _wg.target[raw]
            wg_pred = _wg.prediction[raw]

        for ch in channels:
            tar = wg_target.sel(channel=ch)
            pred = wg_pred.sel(channel=ch)

            if cams_cache is not None and (ch, hr) in cams_cache:
                cams_vals = cams_cache[(ch, hr)]
            else:
                lat = tar["lat"].values
                lon = tar["lon"].values
                cams_vals = cams_reader.get_data(ch, step=hr, target_lat=lat, target_lon=lon)
                cams_vals = np.asarray(cams_vals).ravel()

            wg_bias_da, cams_bias_da = _prepare_bias_da(pred, tar, cams_vals)

            ppm_factor = _channel_ppm_factor(ch) if convert_to_ppm else None
            if ppm_factor is not None:
                wg_bias_da = wg_bias_da * ppm_factor
                cams_bias_da = cams_bias_da * ppm_factor

            bias_cache[(ch, hr)] = (wg_bias_da, cams_bias_da)

            all_vals = np.concatenate([
                np.asarray(wg_bias_da).ravel(),
                np.asarray(cams_bias_da).ravel(),
            ])
            if all_vals.size > 0:
                maxabs = float(np.nanmax(np.abs(all_vals)))
                channel_maxabs[ch] = max(channel_maxabs[ch], maxabs)

    # Apply colour cap using the corrected unit-aware helper.
    for ch in channels:
        if color_max_ppb is not None:
            ppm_factor = _channel_ppm_factor(ch) if convert_to_ppm else None
            cap = _compute_color_cap(ch, color_max_ppb, convert_to_ppm, ppm_factor)
            if channel_maxabs[ch] > cap:
                _logger.warning(
                    f"Bias channel {ch}: max |bias| {channel_maxabs[ch]:.4g} "
                    f"exceeds color cap ({color_max_ppb} ppb = {cap:.4g}); clipping."
                )
            channel_maxabs[ch] = min(channel_maxabs[ch], cap)
        _logger.info(f"Bias colour range for {ch}: +/- {channel_maxabs[ch]:.4g}")

    # ---- second pass: plot with fixed colour range (parallel) ------------
    _logger.info("Bias maps [2/2]: rendering %d frames using %d threads …", total_frames, _N_WORKERS)
    render_tasks = []
    for hr in forecast_hours:
        for ch in channels:
            key = (ch, hr)
            if key not in bias_cache:
                continue
            wg_bias_da, cams_bias_da = bias_cache[key]
            render_tasks.append((
                ch, hr,
                np.asarray(wg_bias_da).ravel(),
                np.asarray(cams_bias_da).ravel(),
                np.asarray(wg_bias_da["lat"]).ravel(),
                np.asarray(wg_bias_da["lon"]).ravel(),
                channel_maxabs[ch],
                convert_to_ppm,
                combo_dir / f"bias_{ch}_fstep_{hr:03d}.png",
            ))

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
        futures = {
            pool.submit(_render_bias_frame, *args): args[0:2]
            for args in render_tasks
        }
        for fi, future in enumerate(as_completed(futures), 1):
            ch_done, hr_done = futures[future]
            fname = future.result()
            _logger.info(
                "  Saved bias frame %d/%d: %s", fi, len(render_tasks), Path(fname).name,
            )


def _plot_value_maps(
    plotter: Plotter,
    wg_reader: WeatherGenZarrReader,
    cams_reader: CAMSForecastReader,
    stream: str,
    channels: list[str],
    forecast_hours: list[int],
    wg_map: dict[int, int],
    run_id: str,
    convert_to_ppm: bool = False,
    color_max_ppb: float | None = None,
    wg_data=None,
    cams_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> "Path":
    """Produce and save side-by-side global maps of CAMS forecast vs WG prediction.

    Returns
    -------
    Path
        Output directory where the frames were saved.
    """
    n_hours = len(forecast_hours)
    n_channels = len(channels)
    total_frames = n_hours * n_channels
    _logger.info(
        "Value maps: %d channels x %d forecast hours = %d frames",
        n_channels, n_hours, total_frames,
    )

    value_dir = plotter.out_plot_basedir / stream / "maps" / "value_compare"
    value_dir.mkdir(parents=True, exist_ok=True)
    _logger.info(f"Saving value comparison maps to {value_dir}")

    # ---- first pass: collect data and compute per-channel global ranges --
    _logger.info("Value maps [1/2]: collecting data and computing colour ranges …")
    frame_data: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    channel_ranges: dict[str, tuple[float, float]] = {}

    for hi, hr in enumerate(forecast_hours, 1):
        _logger.info("  Collecting value data for forecast hour %dh (%d/%d) …", hr, hi, n_hours)
        raw = wg_map[hr]

        if wg_data is not None:
            if raw not in wg_data.target:
                _logger.warning(f"Forecast step {hr}h (raw {raw}) not found in WG output – skipping.")
                continue
            wg_target = wg_data.target[raw]
            wg_pred = wg_data.prediction[raw]
        else:
            _wg = wg_reader.get_data(stream=stream, fsteps=[raw], channels=channels)
            if raw not in _wg.target:
                _logger.warning(f"Forecast step {hr}h (raw {raw}) not found in WG output – skipping.")
                continue
            wg_target = _wg.target[raw]
            wg_pred = _wg.prediction[raw]

        for ch in channels:
            tar = wg_target.sel(channel=ch)
            pred = wg_pred.sel(channel=ch)
            lat = tar["lat"].values
            lon = tar["lon"].values

            if cams_cache is not None and (ch, hr) in cams_cache:
                cams_vals = cams_cache[(ch, hr)]
            else:
                cams_vals = cams_reader.get_data(ch, step=hr, target_lat=lat, target_lon=lon)
                cams_vals = np.asarray(cams_vals).ravel()
            pred_vals = np.asarray(pred).ravel()

            tar_vals = np.asarray(tar).ravel()

            ppm_factor = _channel_ppm_factor(ch) if convert_to_ppm else None
            if ppm_factor is not None:
                cams_vals = cams_vals * ppm_factor
                pred_vals = pred_vals * ppm_factor
                tar_vals = tar_vals * ppm_factor

            frame_data[(ch, hr)] = (lat, lon, cams_vals, pred_vals, tar_vals)

            all_vals = np.concatenate([cams_vals, pred_vals, tar_vals])
            cur_min = float(np.nanmin(all_vals))
            cur_max = float(np.nanmax(all_vals))
            if ch not in channel_ranges:
                channel_ranges[ch] = (cur_min, cur_max)
            else:
                prev_min, prev_max = channel_ranges[ch]
                channel_ranges[ch] = (min(prev_min, cur_min), max(prev_max, cur_max))

    # Apply color cap using the corrected unit-aware helper.
    for ch in list(channel_ranges):
        vmin, vmax = channel_ranges[ch]
        ppm_factor = _channel_ppm_factor(ch) if convert_to_ppm else None

        if ppm_factor is not None and vmax > 1.0:
            _logger.warning(
                f"Channel {ch}: max value {vmax:.4g} ppm (> 1 ppm) – "
                f"this may indicate a data or conversion issue."
            )

        if color_max_ppb is not None:
            cap = _compute_color_cap(ch, color_max_ppb, convert_to_ppm, ppm_factor)
            if vmax > cap:
                _logger.warning(
                    f"Channel {ch}: max value {vmax:.4g} exceeds color cap "
                    f"({color_max_ppb} ppb = {cap:.4g}); clipping colour scale."
                )
            vmax = min(vmax, cap)

        vmin = max(vmin, 0.0)  # mixing ratios are non-negative
        channel_ranges[ch] = (vmin, vmax)
        _logger.info(f"Value colour range for {ch}: [{vmin:.4g}, {vmax:.4g}]")

    # ---- second pass: render frames (parallel) --------------------------
    _logger.info("Value maps [2/2]: rendering %d frames using %d threads …", total_frames, _N_WORKERS)
    render_tasks = []
    for ch in channels:
        if ch not in channel_ranges:
            continue
        vmin, vmax = channel_ranges[ch]
        value_unit = "ppm" if (convert_to_ppm and _channel_ppm_factor(ch) is not None) else "kg kg\u207b\u00b9"

        for hr in forecast_hours:
            key = (ch, hr)
            if key not in frame_data:
                continue
            lat, lon, cams_v, pred_v, tar_v = frame_data[key]
            render_tasks.append((
                ch, hr,
                np.asarray(lat).ravel(),
                np.asarray(lon).ravel(),
                np.asarray(cams_v).ravel(),
                np.asarray(pred_v).ravel(),
                vmin, vmax, value_unit,
                value_dir / f"values_{ch}_fstep_{hr:03d}.png",
                np.asarray(tar_v).ravel(),
            ))

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
        futures = {
            pool.submit(_render_value_frame, *args): args[0:2]
            for args in render_tasks
        }
        for fi, future in enumerate(as_completed(futures), 1):
            ch_done, hr_done = futures[future]
            fname = future.result()
            _logger.info(
                "  Saved value frame %d/%d: %s", fi, len(render_tasks), Path(fname).name,
            )

    return value_dir


def _build_animation_from_frames(
    frame_dir: "Path",
    channels: list[str],
    forecast_hours: list[int],
    run_id: str,
    fps: float = 2.0,
    prefix: str = "values",
) -> list["Path"]:
    """Assemble per-step PNG frames into per-channel GIF animations."""
    from PIL import Image

    duration_ms = int(1000 / fps) if fps > 0 else 400
    anim_dir = frame_dir / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)

    _logger.info(
        "Assembling animations for %d channels (%d forecast hours each) …",
        len(channels), len(forecast_hours),
    )
    gif_paths: list[Path] = []
    for ci, ch in enumerate(channels, 1):
        _logger.info("  Channel %d/%d: %s", ci, len(channels), ch)
        frames: list[Image.Image] = []
        for hr in sorted(forecast_hours):
            png = frame_dir / f"{prefix}_{ch}_fstep_{hr:03d}.png"
            if png.exists():
                frames.append(Image.open(png).copy())
            else:
                _logger.warning(f"Frame {png} not found – skipping in animation.")

        if not frames:
            _logger.warning(f"No frames found for channel {ch} in {frame_dir} – skipping animation.")
            continue

        gif_path = anim_dir / f"animation_{prefix}_{ch}_{run_id}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        _logger.info(f"Saved animation: {gif_path}")
        gif_paths.append(gif_path)

    return gif_paths


def _plot_rmse_curves(
    rmse_dir: Path,
    valid_steps: list[int],
    wg_rmse: dict[str, list[float]],
    cams_rmse: dict[str, list[float]],
    run_id: str,
) -> None:
    """Write line plots of RMSE vs forecast step for each channel.

    FIX: uses the thread-safe FigureCanvasAgg backend instead of pyplot so
    this function is safe to call from a threaded context (consistent with
    _render_bias_frame and _render_value_frame).
    """
    n_channels = len(wg_rmse)
    _logger.info("Generating RMSE curves for %d channels …", n_channels)
    for ci, (ch, wg_vals) in enumerate(wg_rmse.items(), 1):
        _logger.info("  RMSE curve %d/%d: %s", ci, n_channels, ch)
        cams_vals = cams_rmse[ch]

        fig = Figure(figsize=(10, 6), dpi=300)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(valid_steps, wg_vals, marker="o", label="WeatherGen")
        ax.plot(valid_steps, cams_vals, marker="s", label="CAMS Forecast")
        ax.set_xlabel("Forecast Hour")
        ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE Comparison \u2013 {ch}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fname = rmse_dir / f"rmse_comparison_{ch}_{run_id}.png"
        fig.savefig(str(fname), bbox_inches="tight")
        _logger.info(f"Saved RMSE comparison plot: {fname}")


def _rmse_scorecard(
    rmse_dir: Path,
    valid_steps: list[int],
    wg_rmse: dict[str, list[float]],
    cams_rmse: dict[str, list[float]],
    run_id: str,
    rel_pct_clip: float | None = 40.0,
) -> Path:
    """Create a tabular RMSE scorecard (CSV + table image + heatmap).

    FIX: uses the thread-safe FigureCanvasAgg backend for all figure
    rendering instead of pyplot, consistent with the rest of the module.
    """
    records = []
    for ch in wg_rmse:
        for step, wg_val, cams_val in zip(valid_steps, wg_rmse[ch], cams_rmse[ch]):
            better = "tie"
            if wg_val < cams_val:
                better = "wg"
            elif cams_val < wg_val:
                better = "cams"
            records.append(
                {
                    "forecast_step": step,
                    "channel": ch,
                    "wg_rmse": wg_val,
                    "cams_rmse": cams_val,
                    "better": better,
                }
            )
    df = pd.DataFrame(records)
    scorecard_path = rmse_dir / f"rmse_scorecard_{run_id}.csv"
    df.to_csv(scorecard_path, index=False)
    _logger.info(f"Written RMSE scorecard to: {scorecard_path}")

    # ---- table image ----------------------------------------------------
    _logger.info("Rendering scorecard table image …")
    try:
        fig = Figure(figsize=(len(df.columns) * 2, len(df) * 0.3 + 1), dpi=300)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        fig.tight_layout()
        img_path = rmse_dir / f"rmse_scorecard_{run_id}.png"
        fig.savefig(str(img_path), bbox_inches="tight")
        _logger.info(f"Written RMSE scorecard image to: {img_path}")
    except Exception as exc:
        _logger.warning(f"Could not write scorecard image: {exc}")

    # ---- heatmap --------------------------------------------------------
    _logger.info("Rendering RMSE heatmap …")
    try:
        df_heat = df.copy()
        rel_pct_raw = (
            (df_heat["wg_rmse"] - df_heat["cams_rmse"]) / df_heat["cams_rmse"] * 100
        )
        if rel_pct_clip is not None:
            clip_lim = max(abs(float(rel_pct_clip)), 1.0)
            df_heat["rel_pct"] = rel_pct_raw.clip(-clip_lim, clip_lim)
            n_clipped = int(np.sum(np.abs(rel_pct_raw.values) > clip_lim))
            if n_clipped > 0:
                _logger.info(
                    "Clipped %d RMSE relative values to +/-%.1f%% for heatmap display.",
                    n_clipped, clip_lim,
                )
            lim = clip_lim
        else:
            df_heat["rel_pct"] = rel_pct_raw
            vals_arr = df_heat["rel_pct"].values
            lim = float(np.nanmax(np.abs(vals_arr))) if vals_arr.size > 0 else 1.0

        # FIX: _scorecard_channel_order now returns homogeneous (int, str, float)
        # tuples, so this sort can no longer raise TypeError on plain channel names.
        channels_sorted = sorted(df_heat["channel"].unique(), key=_scorecard_channel_order)
        steps = sorted(df_heat["forecast_step"].unique())
        nchan = len(channels_sorted)

        fig = Figure(
            figsize=(max(6, len(steps)), 1.5 * nchan),
            dpi=300,
        )
        FigureCanvasAgg(fig)

        # Use GridSpec for tighter control over spacing.
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nchan, 1, figure=fig, hspace=0.4)
        axes = [fig.add_subplot(gs[i]) for i in range(nchan)]

        images = []
        for ax, ch in zip(axes, channels_sorted):
            vals = (
                df_heat[df_heat["channel"] == ch]
                .set_index("forecast_step")["rel_pct"]
                .reindex(steps)
                .values.reshape(1, -1)
            )
            im = ax.imshow(vals, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
            images.append(im)
            ax.set_ylabel(ch)
            ax.set_yticks([])
            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels(steps)

        if images:
            cbar = fig.colorbar(
                images[-1],
                ax=axes,
                orientation="horizontal",
                pad=0.15,
                fraction=0.05,
            )
            cbar.set_label("RMSE relative to CAMS (%)")
            fig.subplots_adjust(bottom=0.25)
        else:
            _logger.warning("No mappable found for heatmap, skipping colorbar")

        heat_path = rmse_dir / f"rmse_scorecard_heatmap_{run_id}.png"
        fig.savefig(str(heat_path), dpi=300, bbox_inches="tight", pad_inches=0.2)
        _logger.info(f"Written RMSE heatmap to: {heat_path}")
    except Exception as exc:
        _logger.warning(f"Could not write RMSE heatmap: {exc}")

    return scorecard_path


def plot_cams_wg_comparison(
    eval_cfg: dict,
    run_id: str,
    cams_cfg: dict,
    forecast_steps: list[int],
):
    """
    Plot global bias maps and RMSE-vs-forecast-step comparison between
    WG predictions and CAMS forecasts.

    Parameters
    ----------
    eval_cfg : dict
        Per-run evaluation configuration.
    run_id : str
        Unique identifier for the run.
    cams_cfg : dict
        Configuration for CAMS data.
    forecast_steps : list[int]
        List of forecast steps (in hours) to evaluate.
    """
    stream = cams_cfg.get("cams_stream", "CAMSEAC4")
    channels = cams_cfg.get("cams_channels", None)

    plot_bias_maps_flag = cams_cfg.get("plot_bias_maps", False)
    plot_value_maps_flag = cams_cfg.get("plot_value_maps", False)
    create_video_flag = cams_cfg.get("create_video", False)
    fps = float(cams_cfg.get("fps", 2.0))
    plot_rmse_flag = cams_cfg.get("plot_rmse_curves", False)
    write_scorecard_flag = cams_cfg.get("write_scorecard", False)
    convert_to_ppm = bool(cams_cfg.get("convert_to_ppm", False))
    _raw_rmse_rel_clip = cams_cfg.get("rmse_rel_pct_clip", 40.0)
    rmse_rel_pct_clip: float | None = (
        float(_raw_rmse_rel_clip) if _raw_rmse_rel_clip is not None else None
    )
    _raw_cap = cams_cfg.get("color_max_ppb", None)
    color_max_ppb: float | None = float(_raw_cap) if _raw_cap is not None else None

    wg_reader = WeatherGenZarrReader(eval_cfg, run_id)
    if "step_hrs" in cams_cfg:
        wg_reader.step_hrs = int(cams_cfg["step_hrs"])
    cams_reader = CAMSForecastReader(cams_cfg)

    if isinstance(forecast_steps, str) and forecast_steps.lower() == "all":
        step_hrs = wg_reader.step_hrs
        wg_hours = set(int(idx) * step_hrs for idx in wg_reader.get_forecast_steps())
        cams_hours = _cams_steps_as_hours(cams_reader)
        forecast_steps = sorted(wg_hours | cams_hours)
    elif isinstance(forecast_steps, (int, float)):
        forecast_steps = [int(forecast_steps)]
    else:
        forecast_steps = list(forecast_steps)

    forecast_hours, wg_map = _align_forecast_steps(wg_reader, cams_reader, forecast_steps)
    if not forecast_hours:
        _logger.error("No common forecast steps between WG and CAMS – aborting comparison.")
        return

    wg_raw_steps = [wg_map[h] for h in forecast_hours]
    forecast_steps = forecast_hours

    if channels is None:
        stream_cfg = eval_cfg.get("streams", {}).get(stream, {})
        channels = stream_cfg.get("channels", None)
    if channels is None:
        channels = wg_reader.get_channels(stream)

    missing_channels = [ch for ch in channels if not cams_reader.supports_channel(ch)]
    if missing_channels:
        _logger.warning(
            "Skipping channels not found in CAMS dataset %s: %s",
            cams_reader.cams_forecast_path,
            missing_channels,
        )
    channels = [ch for ch in channels if cams_reader.supports_channel(ch)]
    if not channels:
        _logger.error(
            "No requested channels are available in CAMS dataset %s.",
            cams_reader.cams_forecast_path,
        )
        return

    plotter_cfg = {
        "image_format": "png",
        "dpi_val": 300,
        "fig_size": (10, 6),
        "regions": ["global"],
        "fps": fps,
    }
    base_dir = Path(eval_cfg.get("runplot_base_dir", eval_cfg.get("results_base_dir", ".")))
    output_dir = base_dir if base_dir.name == run_id else base_dir / run_id

    plotter = Plotter(plotter_cfg, output_dir, stream=stream)

    _logger.info(
        "Loading WG data: stream=%s, %d forecast steps, %d channels",
        stream, len(wg_raw_steps), len(channels),
    )
    wg_data = wg_reader.get_data(stream=stream, fsteps=wg_raw_steps, channels=channels)
    for fstep_key in list(wg_data.target):
        wg_data.target[fstep_key] = wg_data.target[fstep_key].load()
        wg_data.prediction[fstep_key] = wg_data.prediction[fstep_key].load()
    _logger.info("WG data loaded into memory.")

    wg_rmse_per_channel: dict[str, list[float]] = {ch: [] for ch in channels}
    cams_rmse_per_channel: dict[str, list[float]] = {ch: [] for ch in channels}
    valid_fsteps: list[int] = []

    # Phase 1: verify timestep alignment (sequential)
    _logger.info("Verifying timestep alignment for %d forecast hours …", len(forecast_steps))
    _hr_info: dict[int, tuple] = {}
    for hr in forecast_steps:
        raw = wg_map[hr]
        if raw not in wg_data.target:
            _logger.warning(f"Forecast step {hr}h (raw {raw}) not found in WG output – skipping.")
            continue
        wg_target = wg_data.target[raw]
        wg_pred = wg_data.prediction[raw]
        valid_time = _verify_timestep_alignment(wg_target, wg_pred, cams_reader, hr)
        init_time = valid_time - np.timedelta64(int(hr), "h")
        _hr_info[hr] = (raw, wg_target, wg_pred, init_time)
        valid_fsteps.append(hr)
        _logger.info(
            "  Forecast hour %dh: verified alignment at %s",
            hr, _format_datetime64(valid_time),
        )

    # Phase 2: batch-load CAMS data
    cams_cache = _preload_and_interpolate_cams(
        cams_reader, channels, valid_fsteps, _hr_info,
    )

    cams_native_rmse: dict[tuple[str, int], float] | None = None
    cams_analysis_filename = cams_cfg.get("cams_analysis_filename")
    if cams_analysis_filename:
        cams_analysis_path = Path(cams_cfg.get("cams_base_dir")) / cams_analysis_filename
        if cams_analysis_path.exists():
            cams_native_rmse = _compute_cams_native_rmse(
                cams_reader, cams_analysis_path, channels, valid_fsteps, _hr_info,
            )
        else:
            _logger.warning(
                "CAMS analysis file %s not found; falling back to WG targets for CAMS RMSE.",
                cams_analysis_path,
            )

    _logger.info(
        "Computing RMSE for %d forecast hours × %d channels …",
        len(valid_fsteps), len(channels),
    )
    for hr in valid_fsteps:
        _raw, wg_target, wg_pred, _init_time = _hr_info[hr]
        for ch in channels:
            tar = wg_target.sel(channel=ch)
            pred = wg_pred.sel(channel=ch)

            ppm_factor = _channel_ppm_factor(ch) if convert_to_ppm else None
            pred_scaled = pred * ppm_factor if ppm_factor else pred
            tar_scaled = tar * ppm_factor if ppm_factor else tar

            pred_flat = np.asarray(pred_scaled.values).ravel()
            tar_flat_wg = np.asarray(tar_scaled.values).ravel()
            wg_rmse_per_channel[ch].append(
                float(np.sqrt(((pred_flat - tar_flat_wg) ** 2).mean()))
            )

            if cams_native_rmse is not None and (ch, hr) in cams_native_rmse:
                cams_rmse_val = cams_native_rmse[(ch, hr)]
                if ppm_factor is not None:
                    cams_rmse_val *= ppm_factor
                cams_rmse_per_channel[ch].append(cams_rmse_val)
            else:
                cams_vals = cams_cache[(ch, hr)]
                tar_flat = np.asarray(tar.values).ravel()
                cams_vals_sc = cams_vals * ppm_factor if ppm_factor else cams_vals
                tar_flat_sc = tar_flat * ppm_factor if ppm_factor else tar_flat
                cams_rmse_per_channel[ch].append(
                    float(np.sqrt(((cams_vals_sc - tar_flat_sc) ** 2).mean()))
                )
    _logger.info("RMSE computation complete.")

    # Post-RMSE verification
    _logger.info("Verifying RMSE results …")
    for ch in channels:
        wg_vals = wg_rmse_per_channel[ch]
        c_vals = cams_rmse_per_channel[ch]
        if len(c_vals) > 1:
            cams_arr = np.array(c_vals)
            cams_mean = float(np.mean(cams_arr))
            cams_cv = float(np.std(cams_arr) / cams_mean) if cams_mean > 0 else 0.0
            if cams_cv < 0.01:
                _logger.warning(
                    "SUSPICIOUS RMSE: CAMS %s RMSE is nearly constant across %d "
                    "forecast steps (CV=%.4f). Values: %s. "
                    "This may indicate the same CAMS data is being used for every step.",
                    ch, len(c_vals), cams_cv,
                    [f"{v:.4e}" for v in c_vals],
                )
        for i, hr in enumerate(valid_fsteps):
            if i < len(wg_vals) and i < len(c_vals):
                wg_v, cams_v = wg_vals[i], c_vals[i]
                if cams_v > 0 and wg_v > 0:
                    ratio = wg_v / cams_v
                    if ratio > 10 or ratio < 0.1:
                        _logger.warning(
                            "LARGE RMSE ratio for %s at %dh: WG=%.4e vs CAMS=%.4e "
                            "(ratio=%.2g). Check data selection and units.",
                            ch, hr, wg_v, cams_v, ratio,
                        )
    _logger.info("RMSE verification complete.")

    if plot_bias_maps_flag:
        _logger.info("Generating bias maps for common forecast steps")
        bias_dir = plotter.out_plot_basedir / stream / "maps" / "bias_compare"
        _plot_bias_maps(
            plotter, wg_reader, cams_reader, stream, channels,
            forecast_steps, wg_map, run_id,
            convert_to_ppm=convert_to_ppm, color_max_ppb=color_max_ppb,
            wg_data=wg_data, cams_cache=cams_cache,
        )
        if create_video_flag:
            _logger.info("Building bias map animations")
            _build_animation_from_frames(bias_dir, channels, forecast_steps, run_id, fps=fps, prefix="bias")

    if plot_value_maps_flag:
        _logger.info("Generating value comparison maps (CAMS vs WG)")
        value_dir = _plot_value_maps(
            plotter, wg_reader, cams_reader, stream, channels,
            forecast_steps, wg_map, run_id,
            convert_to_ppm=convert_to_ppm, color_max_ppb=color_max_ppb,
            wg_data=wg_data, cams_cache=cams_cache,
        )
        if create_video_flag:
            _logger.info("Building value map animations")
            _build_animation_from_frames(value_dir, channels, forecast_steps, run_id, fps=fps, prefix="values")

    if plot_rmse_flag or write_scorecard_flag:
        rmse_dir = output_dir / stream / "rmse"
        rmse_dir.mkdir(parents=True, exist_ok=True)

        if plot_rmse_flag:
            _plot_rmse_curves(rmse_dir, valid_fsteps, wg_rmse_per_channel, cams_rmse_per_channel, run_id)

        if write_scorecard_flag:
            _rmse_scorecard(
                rmse_dir, valid_fsteps, wg_rmse_per_channel, cams_rmse_per_channel,
                run_id, rel_pct_clip=rmse_rel_pct_clip,
            )