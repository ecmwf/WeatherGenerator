# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Score orchestration: per-fstep scoring, stream aggregation, and JSON output."""

import json
import logging

import numpy as np
import xarray as xr
from joblib import delayed

from weathergen.evaluate.io.data.io_orchestration import dispatch_parallel, get_num_workers
from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.scores.score import VerifiedData, get_score
from weathergen.evaluate.utils.array_utils import scalar_coord_to_dim
from weathergen.evaluate.utils.clim_utils import get_climatology
from weathergen.evaluate.utils.regions import RegionBoundingBox

_logger = logging.getLogger(__name__)


def get_next_fstep_data(fstep, da_preds, da_tars, fsteps):
    """Get the next forecast step data for the given forecast step."""
    fstep_idx = fsteps.index(fstep)
    next_fstep = fsteps[fstep_idx + 1] if fstep_idx + 1 < len(fsteps) else None
    if next_fstep is not None:
        preds_next = da_preds.get(next_fstep, None)
        tars_next = da_tars.get(next_fstep, None)
    else:
        preds_next = None
        tars_next = None
    return preds_next, tars_next


def _score_single_fstep(
    fstep: int,
    tars: xr.DataArray,
    preds: xr.DataArray,
    preds_next: xr.DataArray | None,
    tars_next: xr.DataArray | None,
    climatology: xr.DataArray | None,
    bbox: "RegionBoundingBox",
    metrics: dict,
    group_by_coord: str | None,
    agg_dims: str | list[str] = "ipoint",
) -> tuple[int, xr.DataArray, dict[tuple[int, str], dict]] | None:
    """Score all metrics for one fstep in one region. Stateless, thread-safe.

    Parameters
    ----------
    fstep : int
        Forecast step index.
    tars, preds : xr.DataArray
        Target and prediction data for this fstep.
    preds_next, tars_next : xr.DataArray | None
        Next-step data for froct/troct metrics.
    climatology : xr.DataArray | None
        Aligned climatology for this fstep.
    bbox : RegionBoundingBox
        Region bounding box to apply.
    metrics : dict
        Metric name → parameters dict.
    group_by_coord : str | None
        Coordinate to group by (None for gridded, "sample" for scatter).

    Returns
    -------
    (fstep, combined_metrics, metric_attrs) or None if no valid scores.
    """
    if preds.sizes.get("ipoint") == 0:
        return None

    tars, preds, tars_next, preds_next = [
        bbox.apply_mask(x) if x is not None else None for x in (tars, preds, tars_next, preds_next)
    ]

    score_data = VerifiedData(preds, tars, preds_next, tars_next, climatology)

    valid_scores = []
    valid_metric_names = []

    for metric, parameters in metrics.items():
        score = get_score(
            score_data,
            metric,
            agg_dims=agg_dims,
            group_by_coord=group_by_coord,
            parameters=parameters,
        )
        if score is not None:
            valid_scores.append(score)
            valid_metric_names.append(metric)

    if not valid_scores:
        return None

    metric_attrs = {}
    for metric_name, score in zip(valid_metric_names, valid_scores, strict=False):
        if score.attrs:
            metric_attrs[(int(fstep), metric_name)] = score.attrs.copy()

    combined = xr.concat(
        valid_scores,
        dim="metric",
        coords="minimal",
        combine_attrs="drop_conflicts",
    )
    combined = combined.assign_coords(metric=valid_metric_names)
    combined = combined.compute()

    for coord in ["channel", "sample", "ens"]:
        combined = scalar_coord_to_dim(combined, coord)

    return fstep, combined, metric_attrs


def calc_scores_per_stream(
    reader: Reader,
    stream: str,
    regions: list[str],
    metrics_dict: dict,
    output_data: ReaderOutput | None = None,
):
    """Calculate scores for a given run and stream using the specified metrics.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about a particular run.
    stream :
        Stream name to calculate scores for.
    regions :
        List of regions to calculate scores on.
    metrics_dict :
        Dictionary mapping regions to lists of metric names to calculate.
    output_data : ReaderOutput | None
        Pre-loaded data.  When provided, reader.get_data() is skipped — this
        avoids the double-load when data is already loaded for plotting.

    Returns
    -------
    dict
        Scores for each metric and stream.
    """
    local_scores = {}

    available_data = reader.check_availability(stream, mode="evaluation")
    if not available_data.score_availability:
        _logger.warning(f"RUN {reader.run_id} - {stream}: Skipping scoring.")
        return {}
    fsteps = available_data.fsteps
    samples = available_data.samples
    channels = available_data.channels
    ensemble = available_data.ensemble
    is_gridded_data = reader.is_gridded_data(stream)
    group_by_coord = None if is_gridded_data else "sample"

    if output_data is None:
        output_data = reader.get_data(
            stream,
            fsteps=fsteps,
            samples=samples,
            channels=channels,
            ensemble=ensemble,
        )

    da_preds = output_data.prediction
    da_tars = output_data.target
    fsteps = sorted(list(da_preds.keys()))
    aligned_clim_data = get_climatology(reader, da_tars, stream)

    max_workers = reader.eval_cfg.get("max_workers", None)
    agg_dims = reader.eval_cfg.get("agg_dims", "ipoint")

    for region in regions:
        bbox = RegionBoundingBox.from_region_name(region)
        metrics = metrics_dict[region]

        fstep_results, all_metric_attrs = compute_scores_for_region(
            reader.run_id,
            stream,
            region,
            da_preds,
            da_tars,
            fsteps,
            aligned_clim_data,
            is_gridded_data,
            group_by_coord,
            bbox,
            metrics,
            max_workers,
            agg_dims,
        )

        store_metrics_for_region(
            local_scores,
            reader.run_id,
            stream,
            region,
            fstep_results,
            all_metric_attrs,
            metrics,
            samples,
            fsteps,
            channels,
            ensemble,
            da_preds,
        )

    return local_scores


def compute_scores_for_region(
    run_id: str,
    stream: str,
    region: str,
    da_preds: dict,
    da_tars: dict,
    fsteps: list[int],
    aligned_clim_data: dict | None,
    is_gridded_data: bool,
    group_by_coord: str | None,
    bbox: "RegionBoundingBox",
    metrics: dict,
    max_workers: int | None,
    agg_dims: str | list[str] = "ipoint",
) -> tuple[list, dict]:
    """Dispatch parallel scoring for all fsteps in one region.

    Parameters
    ----------
    run_id : str
        Run identifier (used for logging).
    stream : str
        Stream name.
    region : str
        Region name.
    da_preds, da_tars : dict
        Prediction and target dicts keyed by forecast step.
    fsteps : list[int]
        Sorted forecast steps.
    aligned_clim_data : dict | None
        Climatology aligned to forecast steps, or None.
    is_gridded_data : bool
        Whether the stream is gridded.
    group_by_coord : str | None
        Coordinate to group by (None for gridded, "sample" for scatter).
    bbox : RegionBoundingBox
        Region bounding box.
    metrics : dict
        Metric name → parameters dict.
    max_workers : int | None
        Hard cap on parallel workers from config.

    Returns
    -------
    tuple[list, dict]
        ``(fstep_results, all_metric_attrs)`` — sorted list of
        ``(fstep, combined_metrics, fstep_attrs)`` tuples plus the
        merged attribute dict.
    """
    _logger.info(
        f"RUN {run_id} - {stream}: Calculating scores for region {region}"
        f" across {len(fsteps)} fsteps and metrics {list(metrics.keys())}..."
    )

    fstep_tasks = []
    for fstep in fsteps:
        tars_fs = da_tars[fstep]
        preds_fs = da_preds[fstep]
        if is_gridded_data:
            preds_next, tars_next = get_next_fstep_data(fstep, da_preds, da_tars, fsteps)
        else:
            preds_next, tars_next = None, None
        climatology = aligned_clim_data[fstep] if aligned_clim_data else None
        fstep_tasks.append((fstep, tars_fs, preds_fs, preds_next, tars_next, climatology))

    calls = [
        delayed(_score_single_fstep)(
            fstep,
            tars_fs,
            preds_fs,
            preds_next,
            tars_next,
            climatology,
            bbox,
            metrics,
            group_by_coord,
            agg_dims,
        )
        for fstep, tars_fs, preds_fs, preds_next, tars_next, climatology in fstep_tasks
    ]
    n_workers = get_num_workers(max_workers=max_workers)
    all_results = dispatch_parallel(
        calls,
        n_workers=n_workers,
        backend="threading",
        desc=f"Scoring {run_id} - {stream} {region}",
    )
    fstep_results = sorted(
        [r for r in all_results if r is not None],
        key=lambda r: r[0],
    )

    all_metric_attrs: dict = {}
    for _, _, fstep_attrs in fstep_results:
        all_metric_attrs.update(fstep_attrs)

    return fstep_results, all_metric_attrs


def store_metrics_for_region(
    local_scores: dict,
    run_id: str,
    stream: str,
    region: str,
    fstep_results: list,
    all_metric_attrs: dict,
    metrics: dict,
    samples: list,
    fsteps: list[int],
    channels: list[str],
    ensemble: list[str],
    da_preds: dict,
) -> None:
    """Populate the result data structure from computed per-fstep scores.

    Parameters
    ----------
    local_scores : dict
        Output dict, mutated in-place (metric → region → stream → run_id → DataArray).
    run_id : str
        Run identifier.
    stream : str
        Stream name.
    region : str
        Region name.
    fstep_results : list
        Sorted list of ``(fstep, combined_metrics, fstep_attrs)`` tuples.
    all_metric_attrs : dict
        Merged attribute dict from all fsteps.
    metrics : dict
        Metric name → parameters dict.
    samples, fsteps, channels, ensemble
        Coordinate arrays for the output DataArray.
    da_preds : dict
        Prediction dict (used only to check for ``lead_time`` coord).
    """
    metric_stream = xr.DataArray(
        np.full(
            (len(samples), len(fsteps), len(channels), len(metrics), len(ensemble)),
            np.nan,
        ),
        coords={
            "sample": samples,
            "forecast_step": fsteps,
            "channel": channels,
            "metric": list(metrics.keys()),
            "ens": ensemble,
        },
    )

    if "lead_time" in da_preds[fsteps[0]].coords:
        metric_stream = metric_stream.assign_coords(
            lead_time=("forecast_step", np.full(len(fsteps), -1, dtype=int))
        )

    for fstep, combined_metrics, _fstep_attrs in fstep_results:
        criteria = {
            "forecast_step": int(fstep),
            "sample": combined_metrics.sample.values,
            "channel": combined_metrics.channel.values,
            "metric": combined_metrics.metric.values,
        }
        if "ens" in combined_metrics.dims:
            criteria["ens"] = combined_metrics.ens.values

        metric_stream.loc[criteria] = combined_metrics

        for coord_name in combined_metrics.coords:
            if coord_name in combined_metrics.dims or coord_name in metric_stream.dims:
                continue
            if coord_name == "lead_time":
                metric_stream.coords["lead_time"].loc[{"forecast_step": int(fstep)}] = (
                    combined_metrics.coords["lead_time"].values.astype("timedelta64[h]").astype(int)
                )
            else:
                coord_dims = combined_metrics.coords[coord_name].dims
                if not all(dim in metric_stream.dims for dim in coord_dims):
                    _logger.debug(
                        f"Skipping coordinate '{coord_name}' with incompatible "
                        f"dimensions {coord_dims} (metric_stream has {metric_stream.dims})"
                    )
                    continue

                if coord_name not in metric_stream.coords:
                    coord_shape = tuple(len(metric_stream.coords[dim]) for dim in coord_dims)
                    metric_stream = metric_stream.assign_coords(
                        {
                            coord_name: xr.DataArray(
                                np.full(coord_shape, "", dtype=object),
                                dims=coord_dims,
                                coords={dim: metric_stream.coords[dim] for dim in coord_dims},
                            )
                        }
                    )

                indexers = {dim: criteria[dim] for dim in coord_dims if dim in criteria}
                metric_stream.coords[coord_name].loc[indexers] = combined_metrics.coords[coord_name]

    _logger.info(f"Scores for run {run_id} - {stream} calculated successfully.")
    _logger.debug(f"all_metric_attrs keys: {list(all_metric_attrs.keys())}")

    for metric, parameters in metrics.items():
        metric_data = metric_stream.sel({"metric": metric}).assign_attrs(parameters)
        for (_stored_fstep, stored_metric), attrs in all_metric_attrs.items():
            if stored_metric == metric and attrs:
                _logger.debug(f"Restoring {len(attrs)} attributes for {metric}")
                metric_data.attrs.update(attrs)
                break

        local_scores.setdefault(metric, {}).setdefault(region, {}).setdefault(stream, {})[
            run_id
        ] = metric_data


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def metric_list_to_json(
    reader: Reader, stream: str, metrics_dict: list[xr.DataArray], regions: list[str]
):
    """Write evaluation results to stream- and metric-specific JSON files.

    Parameters
    ----------
    reader : Reader
        Reader object containing all info about the run_id.
    stream : str
        Stream name.
    metrics_dict : dict
        Metrics per stream (metric → region → stream → run_id → DataArray).
    regions : list[str]
        Region names.
    """
    reader.metrics_dir.mkdir(parents=True, exist_ok=True)

    for metric, metric_stream in metrics_dict.items():
        for region in regions:
            for run_id, metric_data in metric_stream[region][stream].items():
                save_path = (
                    reader.metrics_dir
                    / f"{run_id}_{stream}_{region}_{metric}_chkpt{reader.mini_epoch:05d}.json"
                )
                metric_data_dict = metric_data.to_dict()

                if save_path.exists():
                    _logger.debug(f"{save_path} already present")
                    with save_path.open("r") as f:
                        data_dict = json.load(f)
                    if "scores" not in data_dict:
                        data_dict = {"scores": [data_dict]}
                    scores = data_dict.get("scores")
                    for i, existing_score in enumerate(scores):
                        if existing_score["attrs"] == metric_data.attrs:
                            _logger.warning("Metric with same parameters found, replacing")
                            scores[i] = metric_data_dict
                            break
                    else:
                        scores.append(metric_data_dict)
                        _logger.debug(f"Appending results to {save_path}")
                else:
                    _logger.debug(f"Saving results to new file {save_path}")
                    data_dict = {"scores": [metric_data_dict]}

                with open(save_path, "w") as f:
                    json.dump(data_dict, f, indent=4)

    _logger.info(
        f"Saved all results of inference run {reader.run_id} - mini_epoch {reader.mini_epoch:d} "
        f"successfully to {reader.metrics_dir}."
    )
