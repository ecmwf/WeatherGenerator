# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Index orchestration: per-sample index computation, stream aggregation, JSON output.

Structurally parallel to ``scores/score_orchestration.py`` but parallelizes
over *sample* rather than *forecast step*: an index (e.g. NAM) needs one
sample's full forecast-lead-time trajectory as a single population to fit its
EOF, whereas scores are independent per (fstep, region).
"""

import json
import logging

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from joblib import delayed

from weathergen.evaluate.indices.nam import compute_nam_index
from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel, get_num_workers
from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.utils.clim_utils import get_climatology
from weathergen.evaluate.utils.regions import RegionBoundingBox

_logger = logging.getLogger(__name__)

# index name -> pure computation callable, analogous to Scores.det_metrics_dict
# but scoped to this module (does not touch the scores registry).
INDEX_REGISTRY = {"nam": compute_nam_index}


def _select_sample(da: xr.DataArray, sample: int) -> xr.DataArray:
    """Select one sample's data, whether ``sample`` is a dim or a per-ipoint coord."""
    if "sample" in da.dims:
        return da.sel(sample=sample)
    return da.sel(ipoint=da["sample"] == sample)


def _index_single_sample(
    sample: int,
    index_fn,
    tars_by_fstep: dict[int, xr.DataArray],
    climatology_by_fstep: dict[int, xr.DataArray] | None,
    bbox: RegionBoundingBox,
    fsteps: list[int],
    channels: list[str],
    parameters: dict,
) -> tuple[int, dict[str, np.typing.NDArray], dict] | None:
    """Compute one index for one sample, across all its forecast steps.

    Stateless, thread-safe (mirrors ``_score_single_fstep``'s contract).

    Returns
    -------
    (sample, {channel: index_array}, attrs) or None if no channel could be
    scored (e.g. too few valid forecast steps/points).
    """
    lat = None
    channel_series: dict[str, list[np.typing.NDArray]] = {ch: [] for ch in channels}

    for fstep in fsteps:
        tars_fs = tars_by_fstep.get(fstep)
        if tars_fs is None:
            continue

        tars_fs = bbox.apply_mask(tars_fs)
        if tars_fs.sizes.get("ipoint", 0) == 0:
            return None

        sample_da = _select_sample(tars_fs, sample)
        if lat is None:
            lat = sample_da["lat"].values

        if climatology_by_fstep is not None:
            clim_fs = bbox.apply_mask(climatology_by_fstep[fstep])
            clim_sample = _select_sample(clim_fs, sample)
            if "statistic" in clim_sample.dims:
                clim_sample = clim_sample.sel(statistic="mean", drop=True)
            values = (sample_da - clim_sample).transpose("ipoint", "channel").values
        else:
            values = sample_da.transpose("ipoint", "channel").values

        for i, ch in enumerate(channels):
            channel_series[ch].append(values[:, i])

    if lat is None:
        return None

    index_by_channel: dict[str, np.typing.NDArray] = {}
    attrs: dict = {}
    significance_level = parameters.get("significance_level", 0.05)

    for ch in channels:
        stacked = np.stack(channel_series[ch], axis=0)  # (n_fsteps, n_points)
        if climatology_by_fstep is None:
            # Own-trajectory anomaly fallback: subtract this sample's own
            # time-mean over its forecast steps (matches analyze_nam.py).
            # NOTE: baseline is sample-specific, so resulting index values are
            # not comparable in sign/magnitude across different samples.
            stacked = stacked - stacked.mean(axis=0, keepdims=True)

        result = index_fn(stacked, lat, significance_level=significance_level)
        if result is None:
            continue

        index, pattern, explained_variance_ratio = result
        index_by_channel[ch] = index
        attrs[f"{ch}/pattern"] = pattern.tolist()
        attrs[f"{ch}/explained_variance_ratio"] = explained_variance_ratio

    if not index_by_channel:
        return None

    return sample, index_by_channel, attrs


def calc_indices_per_stream(
    reader: Reader,
    stream: str,
    indices_dict: dict,
    output_data: ReaderOutput | None = None,
) -> dict:
    """Calculate climate indices for a given run and stream.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about a particular run.
    stream : str
        Stream name to calculate indices for.
    indices_dict : dict
        Index name -> parameters dict (e.g. ``{"nam": {"min_lat": 20.0}}``).
    output_data : ReaderOutput | None
        Pre-loaded data; when provided ``reader.get_data()`` is skipped.

    Returns
    -------
    dict
        Indices for this stream, keyed ``index_name -> stream -> run_id -> DataArray``.
    """
    local_indices: dict = {}

    available_data = reader.check_availability(stream, mode="evaluation")
    if not available_data.score_availability:
        _logger.warning(f"RUN {reader.run_id} - {stream}: Skipping index computation.")
        return {}

    fsteps = available_data.fsteps
    samples = available_data.samples
    channels = available_data.channels
    ensemble = available_data.ensemble

    if output_data is None:
        output_data = reader.get_data(
            stream, fsteps=fsteps, samples=samples, channels=channels, ensemble=ensemble
        )

    da_tars = output_data.target
    fsteps = sorted(da_tars.keys())
    all_samples = sorted({int(s) for s in samples})

    max_workers = reader.eval_cfg.get("max_workers", None)
    n_workers = get_num_workers(max_workers=max_workers)

    for index_name, parameters in indices_dict.items():
        index_fn = INDEX_REGISTRY.get(index_name)
        if index_fn is None:
            _logger.warning(f"Unknown index '{index_name}', skipping.")
            continue

        bbox = RegionBoundingBox(
            lat_min=parameters.get("min_lat", 20.0),
            lat_max=90.0,
            lon_min=-180.0,
            lon_max=180.0,
            projection=ccrs.PlateCarree(),
        )

        use_climatology = parameters.get("use_climatology", True)
        aligned_clim_data = get_climatology(reader, da_tars, stream) if use_climatology else None

        _logger.info(
            f"RUN {reader.run_id} - {stream}: Calculating index {index_name} "
            f"across {len(all_samples)} samples and channels {channels}..."
        )

        calls = [
            delayed(_index_single_sample)(
                sample, index_fn, da_tars, aligned_clim_data, bbox, fsteps, channels, parameters
            )
            for sample in all_samples
        ]
        raw_results = dispatch_parallel(
            calls,
            n_workers=n_workers,
            backend="threading",
            desc=f"Computing {index_name} for {reader.run_id} - {stream}",
        )
        results = sorted([r for r in raw_results if r is not None], key=lambda r: r[0])

        if not results:
            _logger.warning(
                f"RUN {reader.run_id} - {stream}: No valid {index_name} results for any sample."
            )
            continue

        store_index_results(
            local_indices,
            reader.run_id,
            stream,
            index_name,
            results,
            all_samples,
            fsteps,
            channels,
            ensemble,
            parameters,
        )

    return local_indices


def store_index_results(
    local_indices: dict,
    run_id: str,
    stream: str,
    index_name: str,
    results: list,
    samples: list[int],
    fsteps: list[int],
    channels: list[str],
    ensemble: list[str],
    parameters: dict,
) -> None:
    """Populate the result data structure from computed per-sample indices.

    Parameters
    ----------
    local_indices : dict
        Output dict, mutated in-place (index_name -> stream -> run_id -> DataArray).
    run_id, stream, index_name : str
        Identifiers for this result.
    results : list
        Sorted list of ``(sample, {channel: index_array}, attrs)`` tuples.
    samples, fsteps, channels, ensemble
        Coordinate arrays for the output DataArray (``ensemble`` unused in v1 —
        NAM operates on the target/anomaly only, no ensemble handling).
    parameters : dict
        Index parameters, stored as attrs (mirrors the scores JSON convention).
    """
    index_stream = xr.DataArray(
        np.full((len(samples), len(fsteps), len(channels)), np.nan),
        dims=["sample", "forecast_step", "channel"],
        coords={"sample": samples, "forecast_step": fsteps, "channel": channels},
    )

    for sample, index_by_channel, attrs in results:
        for ch, index in index_by_channel.items():
            index_stream.loc[{"sample": sample, "channel": ch}] = index
        for k, v in attrs.items():
            index_stream.attrs[f"sample_{sample}/{k}"] = v

    index_stream = index_stream.assign_attrs(parameters)
    local_indices.setdefault(index_name, {}).setdefault(stream, {})[run_id] = index_stream


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def index_list_to_json(reader: Reader, stream: str, indices_dict: dict) -> None:
    """Write index results to their own JSON files, separate from scores/*.json.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about the run_id.
    stream : str
        Stream name.
    indices_dict : dict
        Indices for this stream (index_name -> stream -> run_id -> DataArray),
        as produced by ``calc_indices_per_stream``.
    """
    indices_dir = reader.metrics_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    eval_settings = reader.get_eval_settings(stream)

    for index_name, index_by_stream in indices_dict.items():
        for run_id, index_data in index_by_stream[stream].items():
            save_path = (
                indices_dir / f"{run_id}_{stream}_{index_name}_chkpt{reader.mini_epoch:05d}.json"
            )
            data_dict = {"eval_settings": eval_settings, "indices": [index_data.to_dict()]}
            with open(save_path, "w") as f:
                json.dump(data_dict, f, indent=4)

    _logger.info(
        f"Saved all index results of inference run {reader.run_id} - "
        f"mini_epoch {reader.mini_epoch:d} successfully to {indices_dir}."
    )
