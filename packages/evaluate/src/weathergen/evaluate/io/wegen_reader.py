# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Standard library
import json
import logging
from collections import defaultdict
from pathlib import Path

# Third-party
import numpy as np
import omegaconf as oc
import xarray as xr
from tqdm import tqdm

# Local application / package
from weathergen.common.config import (
    get_path_run,
    load_merge_configs,
    load_run_config,
)
from weathergen.common.io import zarrio_reader
from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.scores.score_utils import to_list
from weathergen.evaluate.utils.derived_channels import DeriveChannels

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class WeatherGenReader(Reader):
    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict | None = None):
        super().__init__(eval_cfg, run_id, private_paths)

        # TODO: remove backwards compatibility to "epoch" in Feb. 2026
        self.mini_epoch = eval_cfg.get("mini_epoch", 0)
        self.rank = eval_cfg.get("rank", 0)
        # Load model configuration and set (run-id specific) directories
        self.inference_cfg = self.get_inference_config()

        if not self.results_base_dir:
            self.results_base_dir = get_path_run(self.inference_cfg)
            _logger.info(f"Results directory obtained from private config: {self.results_base_dir}")
        else:
            _logger.info(f"Results directory parsed: {self.results_base_dir}")

        self.runplot_base_dir = Path(
            self.eval_cfg.get("runplot_base_dir", self.results_base_dir)
        )  # base directory where map plots and histograms will be stored

        self.metrics_base_dir = Path(
            self.eval_cfg.get("metrics_base_dir", self.results_base_dir)
        )  # base directory where score files will be stored

        self.step_hrs = self.inference_cfg.get("step_hrs", 1)

        self.results_dir, self.runplot_dir = (
            Path(self.results_base_dir),
            Path(self.runplot_base_dir),
        )
        # for backward compatibility allow metric_dir to be specified in the run config
        self.metrics_dir = Path(
            self.eval_cfg.get("metrics_dir", self.metrics_base_dir / "evaluation")
        )

    def get_inference_config(self):
        """
        load the config associated to the inference run (different from the eval_cfg which
        contains plot and evaluaiton options.)

        Returns
        -------
        dict
            configuration file from the inference run
        """
        if self.private_paths:
            _logger.info(
                f"Loading config for run {self.run_id} from private paths: {self.private_paths}"
            )
            config = load_merge_configs(self.private_paths, self.run_id, self.mini_epoch)
        else:
            _logger.info(
                f"Loading config for run {self.run_id} from model directory: {self.model_base_dir}"
            )
            config = load_run_config(self.run_id, self.mini_epoch, self.model_base_dir)

        if type(config) not in [dict, oc.DictConfig]:
            _logger.warning("Model config not found. inference config will be empty.")
            config = {}
        return config

    def get_climatology_filename(self, stream: str) -> str | None:
        """
        Get the climatology filename for a given stream from the inference configuration.
        Parameters
        ----------
        stream :
            Name of the data stream.
        Returns
        -------
            Climatology filename if specified, otherwise None.
        """

        stream_dict = self.get_stream(stream)

        clim_data_path = stream_dict.get("climatology_path", None)
        if not clim_data_path:
            clim_base_dir = self.inference_cfg.get("data_path_aux", None)

            clim_fn = next(
                (
                    item.get("climatology_filename")
                    for item in self.inference_cfg["streams"]
                    if item.get("name") == stream
                ),
                None,
            )

            if clim_base_dir and clim_fn:
                clim_data_path = Path(clim_base_dir).join(clim_fn)
            else:
                _logger.warning(
                    f"No climatology path specified for stream {stream}. Setting climatology to "
                    "NaN. Add 'climatology_path' to evaluation config to use metrics like ACC."
                )

        return clim_data_path

    def get_channels(self, stream: str) -> list[str]:
        """
        Get the list of channels for a given stream from the config.

        Parameters
        ----------
        stream :
            The name of the stream to get channels for.

        Returns
        -------
            A list of channel names.
        """
        _logger.debug(f"Getting channels for stream {stream}...")
        all_channels = self.get_inference_stream_attr(stream, "val_target_channels")
        _logger.debug(f"Channels found in config: {all_channels}")
        return all_channels

    def load_scores(
        self, stream: str, regions: list[str], metrics: dict[str, object]
    ) -> xr.DataArray | None:
        """
        Load multiple pre-computed scores for a given run, stream and metric and epoch.

        Parameters
        ----------
        reader :
            Reader object containing all info for a specific run_id
        stream :
            Stream name.
        regions :
            Region names.
        metrics :
            Metric names.

        Returns
        -------
        xr.DataArray
            The metric DataArray.
        computable_metrics:
            dictionary of regions and metrics that can be recomputed
            (empty for JSONreader).
        """

        local_scores = {}
        missing_metrics = {}
        for region in regions:
            for metric, parameters in metrics.items():
                score = self.load_single_score(stream, region, metric, parameters)
                if score is not None:
                    available_data = self.check_availability(stream, score, mode="evaluation")
                    if available_data.score_availability:
                        score = score.sel(
                            sample=available_data.samples,
                            channel=available_data.channels,
                            forecast_step=available_data.fsteps,
                        )
                        local_scores.setdefault(metric, {}).setdefault(region, {}).setdefault(
                            stream, {}
                        )[self.run_id] = score
                        continue

                # all other cases: recompute scores
                missing_metrics.setdefault(region, {}).update({metric: parameters})
                continue
        recomputable_missing_metrics = self.get_recomputable_metrics(missing_metrics)
        return local_scores, recomputable_missing_metrics

    def load_single_score(
        self, stream: str, region: str, metric: str, parameters: dict | None = None
    ) -> xr.DataArray | None:
        """
        Load a single pre-computed score for a given run, stream and metric
        """
        if parameters is None:
            parameters = {}
        score_path = (
            Path(self.metrics_dir)
            / f"{self.run_id}_{stream}_{region}_{metric}_chkpt{self.mini_epoch:05d}.json"
        )
        _logger.debug(f"Looking for: {score_path}")
        score = None
        if score_path.exists():
            with open(score_path) as f:
                data_dict = json.load(f)
                if "scores" not in data_dict:
                    data_dict = {"scores": [data_dict]}
                for score_version in data_dict["scores"]:
                    if score_version["attrs"] == parameters:
                        score = xr.DataArray.from_dict(score_version)
                        break
        return score

    def get_recomputable_metrics(self, metrics):
        """determine whether given metrics can be re-computed."""
        return metrics

    def get_inference_stream_attr(self, stream_name: str, key: str, default=None):
        """
        Get the value of a key for a specific stream from the a model config.

        Parameters:
        ------------
            config:
                The full configuration dictionary.
            stream_name:
                The name of the stream (e.g. 'ERA5').
            key:
                The key to look up (e.g. 'tokenize_spacetime').
            default: Optional
                Value to return if not found (default: None).

        Returns:
            The parameter value if found, otherwise the default.
        """
        for stream in self.inference_cfg.get("streams", []):
            if stream.get("name") == stream_name:
                return stream.get(key, default)
        return default


class WeatherGenJSONReader(WeatherGenReader):
    def __init__(
        self,
        eval_cfg: dict,
        run_id: str,
        private_paths: dict | None = None,
        regions: list[str] | None = None,
        metrics: dict[str, object] | None = None,
    ):
        super().__init__(eval_cfg, run_id, private_paths)
        # goes looking for the coordinates available for all streams, regions, metrics
        streams = list(self.eval_cfg.streams.keys())
        coord_names = ["sample", "forecast_step", "ens"]
        all_coords = {name: [] for name in coord_names}  # collect all available coordinates
        provenance = {
            name: defaultdict(list) for name in coord_names
        }  # remember who had which coords, so we can warn about it later.
        for stream in streams:
            for region in regions:
                for metric, parameters in metrics.items():
                    score = self.load_single_score(stream, region, metric, parameters)
                    if score is not None:
                        for name in coord_names:
                            vals = set(score[name].values)
                            all_coords[name].append(vals)
                            for val in vals:
                                provenance[name][val].append((stream, region, metric))
        self.common_coords = {name: set.intersection(*all_coords[name]) for name in coord_names}
        # issue warnings for skipped coords
        for name in coord_names:
            skipped = set.union(*all_coords[name]) - self.common_coords[name]
            if skipped:
                message = [f"Some {name}(s) were not common among streams, regions and metrics:"]
                for val in skipped:
                    message.append(f" {val} only in {provenance[name][val]}")
                _logger.warning("\n".join(message))

    def get_samples(self) -> set[int]:
        return self.common_coords["sample"]

    def get_forecast_steps(self) -> set[int]:
        return self.common_coords["forecast_step"]

    def get_ensemble(self, stream: str | None = None) -> list[str]:
        return self.common_coords["ens"]

    def get_data(self, *args, **kwargs):
        # TODO this should not be needed, the reader should not even be created if this is the case
        # it can still happen when a particular score was available for a different channel
        raise ValueError(f"Missing JSON data for run {self.run_id}.")

    def get_recomputable_metrics(self, metrics):
        _logger.info(
            f"The following metrics have not yet been computed:{metrics}. Use type: zarr for that."
        )
        return {}


class WeatherGenZarrReader(WeatherGenReader):
    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict | None = None):
        """Data reader class for WeatherGenerator model outputs stored in Zarr format."""
        super().__init__(eval_cfg, run_id, private_paths)

        zarr_ext = self.inference_cfg.get("zarr_store", "zarr")
        # for backwards compatibility assume zarr store is local i.e. .zarr format

        fname_zarr = self.results_dir.joinpath(
            f"validation_chkpt{self.mini_epoch:05d}_rank{self.rank:04d}.{zarr_ext}"
        )
        if fname_zarr.exists():
            if (zarr_ext == "zarr" and fname_zarr.is_dir()) or (
                zarr_ext == "zip" and fname_zarr.is_file()
            ):
                self.fname_zarr = fname_zarr
        else:
            _logger.error(f"Zarr file {fname_zarr} does not exist.")
            raise FileNotFoundError(f"Zarr file {fname_zarr} does not exist")

    def get_data(
        self,
        stream: str,
        samples: list[int] | None = None,
        fsteps: list[str] | None = None,
        channels: list[str] | None = None,
        ensemble: list[str] | None = None,
    ) -> ReaderOutput:
        """
        Retrieve prediction and target data for a given run from the Zarr store.

        Parameters
        ----------
        cfg :
            Configuration dictionary containing all information for the evaluation.
        results_dir : Path
            Directory where the inference results are stored.
            Expected scheme `<results_base_dir>/<run_id>`.
        stream :
            Stream name to retrieve data for.
        samples :
            List of sample indices to retrieve. If None, all samples are retrieved.
        fsteps :
            List of forecast steps to retrieve. If None, all forecast steps are retrieved.
        channels :
            List of channel names to retrieve. If None, all channels are retrieved.

        Returns
        -------
        ReaderOutput
            A dataclass containing:
            - target: Dictionary of xarray DataArrays for targets, indexed by forecast step.
            - prediction: Dictionary of xarray DataArrays for predictions, indexed by forecast step.
        """
        # get type of zarr store

        with zarrio_reader(self.fname_zarr) as zio:
            stream_cfg = self.get_stream(stream)
            all_channels = self.get_channels(stream)
            _logger.info(f"RUN {self.run_id}: Processing stream {stream}...")

            fsteps = self.get_forecast_steps() if fsteps is None else fsteps

            # TODO: Avoid conversion of fsteps and sample to integers (as obtained from the ZarrIO)
            fsteps = sorted([int(fstep) for fstep in fsteps])
            samples = samples or sorted([int(sample) for sample in self.get_samples()])
            channels = channels or stream_cfg.get("channels", all_channels)
            channels = to_list(channels)

            ensemble = ensemble or self.get_ensemble(stream)
            ensemble = to_list(ensemble)

            da_tars, da_preds = [], []

            fsteps_final = []

            for fstep in fsteps:
                _logger.info(f"RUN {self.run_id} - {stream}: Processing fstep {fstep}...")
                da_tars_fs, da_preds_fs, valid_times_fs = [], [], []

                for sample in tqdm(samples, desc=f"Processing {self.run_id} - {stream} - {fstep}"):
                    out = zio.get_data(sample, stream, fstep)

                    if out.target is None or out.prediction is None:
                        _logger.info(
                            f"Skipping {stream} sample {sample} forecast step: {fstep}. "
                            "No data found."
                        )
                        continue

                    target, pred = out.target.as_xarray(), out.prediction.as_xarray()

                    npoints = len(target.ipoint)

                    if npoints == 0:
                        _logger.info(
                            f"Skipping {stream} sample {sample} forecast step: {fstep}. "
                            "Dataset is empty."
                        )
                        continue

                    if ensemble == ["mean"]:
                        _logger.debug("Averaging over ensemble members.")
                        pred = pred.mean("ens", keepdims=True)
                    else:
                        _logger.debug(f"Selecting ensemble members {ensemble}.")
                        pred = pred.sel(ens=ensemble)

                    pred = pred.squeeze()
                    target = target.squeeze()

                    if self.is_regular(stream):
                        vt_list = np.unique(target.valid_time.values).tolist()
                        valid_times_fs.append(vt_list)
                    else:
                        valid_times_fs.append(fstep)

                    da_tars_fs.append(target.persist())
                    da_preds_fs.append(pred.persist())

                if not da_tars_fs:
                    _logger.info(
                        f"[{self.run_id} - {stream}] No valid data found for fstep {fstep}."
                    )
                    continue

                # fsteps_final.extend(valid_times_fs)
                fsteps_final.append(valid_times_fs)

                _logger.debug(
                    f"Concatenating targets and predictions for stream {stream}, "
                    f"forecast_step {fstep}..."
                )

                # faster processing
                if self.is_regular(stream):
                    # Efficient concatenation for regular grid
                    da_preds_fs = _split_by_valid_time(da_preds_fs)
                    da_tars_fs = _split_by_valid_time(da_tars_fs)

                    da_tars_fs = _force_consistent_grids(da_tars_fs)
                    da_preds_fs = _force_consistent_grids(da_preds_fs)
                else:
                    # Irregular (scatter) case. concatenate over ipoint
                    da_tars_fs = [xr.concat(da_tars_fs, dim="ipoint", coords="minimal")]
                    da_preds_fs = [xr.concat(da_preds_fs, dim="ipoint", coords="minimal")]

                da_tars.append([da for da in da_tars_fs])
                da_preds.append([da for da in da_preds_fs])

            # Safer than a list
            da_tars_dict, da_preds_dict = {}, {}
            i = 1

            for _, (fstep, da_t, da_p) in enumerate(
                zip(fsteps_final, da_tars, da_preds, strict=True)
            ):
                if isinstance(fstep, list):  # regular grid with lead times (1 or multiple)
                    for t, p in zip(da_t, da_p, strict=True):
                        t, p = _select_channels(t, p, stream, channels, stream_cfg)
                        t = t.assign_coords(forecast_step=i)
                        p = p.assign_coords(forecast_step=i)
                        t = _add_lead_time_coord(
                            t
                        )  # TODO: move somewhere else into another loop maybe. but 2 loops is slow?
                        p = _add_lead_time_coord(p)
                        p = _scale_z_channels(p, stream)
                        t = _scale_z_channels(t, stream)
                        da_tars_dict[i] = t
                        da_preds_dict[i] = p
                        i += 1
                else:
                    da_t, da_p = _select_channels(da_t, da_p, stream, channels, stream_cfg)
                    da_tars_dict[int(fstep)] = da_t
                    da_preds_dict[int(fstep)] = da_p

            return ReaderOutput(target=da_tars_dict, prediction=da_preds_dict)

    ######## reader utils ########

    def get_stream(self, stream: str):
        """
        returns the dictionary associated to a particular stream.
        Returns an empty dictionary if the stream does not exist in the Zarr file.

        Parameters
        ----------
        stream:
            the stream name

        Returns
        -------
            The config dictionary associated to that stream
        """
        stream_dict = {}

        with zarrio_reader(self.fname_zarr) as zio:
            if stream in zio.streams:
                stream_dict = self.eval_cfg.streams.get(stream, {})
        return stream_dict

    def get_samples(self) -> set[int]:
        """Get the set of sample indices from the Zarr file."""
        with zarrio_reader(self.fname_zarr) as zio:
            return set(int(s) for s in zio.samples)

    def get_forecast_steps(self) -> set[int]:
        """Get the set of forecast steps from the Zarr file."""
        with zarrio_reader(self.fname_zarr) as zio:
            return set(int(f) for f in zio.forecast_steps)

    def get_forecast_substep_valid_times(self, stream: str) -> set[str]:
        """Get the set of forecast times from the Zarr file."""
        if not self.is_regular(stream):
            _logger.warning(f"Stream {stream} is not regular. Forecast times cannot be retrieved.")
            return set()

        with zarrio_reader(self.fname_zarr) as zio:
            dummy = zio.get_data(0, stream, zio.forecast_steps[0])
            unique_lead = np.unique(dummy.valid_time.data)
        return set(str(lt) for lt in unique_lead)

    def get_ensemble(self, stream: str | None = None) -> list[str]:
        """Get the list of ensemble member names for a given stream from the config.
        Parameters
        ----------
        stream :
            The name of the stream to get channels for.

        Returns
        -------
            A list of ensemble members.
        """
        _logger.debug(f"Getting ensembles for stream {stream}...")

        # TODO: improve this to get ensemble from io class
        with zarrio_reader(self.fname_zarr) as zio:
            dummy = zio.get_data(0, stream, zio.forecast_steps[0])
        return list(dummy.prediction.as_xarray().coords["ens"].values)

    # TODO: improve this
    def is_regular(self, stream: str) -> bool:
        """Check if the latitude and longitude coordinates are regularly spaced for a given stream.
        Parameters
        ----------
        stream :
            The name of the stream to get channels for.

        Returns
        -------
            True if the stream is regularly spaced. False otherwise.
        """
        _logger.debug(f"Checking regular spacing for stream {stream}...")

        with zarrio_reader(self.fname_zarr) as zio:
            dummy = zio.get_data(0, stream, zio.forecast_steps[0])

            sample_idx = zio.samples[1] if len(zio.samples) > 1 else zio.samples[0]
            fstep_idx = (
                zio.forecast_steps[1] if len(zio.forecast_steps) > 1 else zio.forecast_steps[0]
            )
            dummy1 = zio.get_data(sample_idx, stream, fstep_idx)

        da = dummy.prediction.as_xarray()
        da1 = dummy1.prediction.as_xarray()

        if (
            da["lat"].shape != da1["lat"].shape
            or da["lon"].shape != da1["lon"].shape
            or not (
                np.allclose(sorted(da["lat"].values), sorted(da1["lat"].values))
                and np.allclose(sorted(da["lon"].values), sorted(da1["lon"].values))
            )
        ):
            _logger.debug("Latitude and/or longitude coordinates are not regularly spaced.")
            return False

        _logger.debug("Latitude and longitude coordinates are regularly spaced.")
        return True


################### Helper functions ########################


def _select_channels(
    da_tar: xr.DataArray, da_pred: xr.DataArray, stream, channels, stream_cfg
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Preprocess the data by scaling z channels if needed and adding lead_time coordinate.

    Parameters
    ----------
    da_tar :
        Input DataArray to preprocess.
    da_pred :
        Input DataArray to preprocess.
    stream:
        Stream name, used to determine if z channels need to be scaled.
    channels:
        List of channels to select.
    stream_cfg:
        Stream configuration dictionary, used to determine if derived channels need to be computed.
    Returns
    -------
        Data arrays with selected channels and added derived channels if applicable.
    """
    # Ensure channel is a dimension, not a scalar coordinate (can happen after squeeze)
    if "channel" not in da_tar.dims:
        da_tar = da_tar.expand_dims("channel")
    if "channel" not in da_pred.dims:
        da_pred = da_pred.expand_dims("channel")

    assert da_pred.channel.values.tolist() == da_tar.channel.values.tolist(), (
        "Channels in prediction and target do not match."
    )

    all_channels = da_tar.channel.values.tolist()

    if set(channels) != set(all_channels):
        _logger.debug(
            f"Restricting targets and predictions to channels {channels} for stream {stream}..."
        )

        dc = DeriveChannels(
            all_channels,
            channels,
            stream_cfg,
        )

        da_tar, da_pred, channels = dc.get_derived_channels(da_tar, da_pred)

        da_tar = da_tar.sel(channel=channels)
        da_pred = da_pred.sel(channel=channels)

    return da_pred, da_tar


def _scale_z_channels(data: xr.DataArray, stream: str) -> xr.DataArray:
    """
    Check scale all channels.

    Parameters
    ----------
    data :
        Input dataset
    stream :
        Stream name.
    Returns
    -------
        Returns a Dataset where channels have been scaled if needed
    """
    if stream is None or not str(stream).startswith("ERA5"):
        return data

    channels_z = [ch for ch in np.atleast_1d(data.channel.values) if str(ch).startswith("z_")]
    factor = 9.80665

    if channels_z:
        channels = data.channel.astype(str)
        mask = channels.str.startswith("z_")
        data = data.where(~mask, data / factor)
    return data


def _split_by_valid_time(arrays: list[xr.DataArray]) -> list[xr.DataArray]:
    """
    Split arrays by valid_time and stack by sample, creating separate
    arrays for each unique lead_time.

    Lead_time is calculated as: valid_time - source_interval_start

    Parameters
    ----------
    arrays : list[xr.DataArray]
        List of DataArrays, each containing multiple valid_times per sample

    Returns
    -------
    list[xr.DataArray]
        List of DataArrays, one per unique lead_time, with samples
        stacked along 'sample' dimension
    """
    # Pre-compute all lead times and build index in single pass
    lead_time_groups = {}  # lead_time -> list of (arr_idx, ipoint_indices)

    unique_valid_times = [np.unique(da.valid_time.values) for da in arrays]
    if len(unique_valid_times) == len(arrays) and all(len(uvt) == 1 for uvt in unique_valid_times):
        _logger.debug(
            "All arrays have a single unique valid_time. Skipping splitting by valid_time."
        )
        return arrays

    for arr_idx, da in tqdm(enumerate(arrays), total=len(arrays), desc="Splitting by valid time"):
        vt = da.valid_time.values
        sis = da.source_interval_start.values

        # Calculate lead_time once
        if vt.ndim > 1:
            lead_times = vt - (sis[:, np.newaxis] if sis.ndim == 1 else sis)
            # Flatten and get unique lead times with their ipoint indices
            valid_mask = ~np.isnat(lead_times)
            for i in range(lead_times.shape[0]):
                row_leads = lead_times[i][valid_mask[i]]
                row_ipoints = np.where(valid_mask[i])[0]
                for lead, ipoint in zip(row_leads, row_ipoints, strict=False):
                    lead_time_groups.setdefault(lead, []).append((arr_idx, i, ipoint))
        else:
            lead_times = vt - sis
            valid_mask = ~np.isnat(lead_times)
            valid_leads = lead_times[valid_mask]
            valid_ipoints = np.where(valid_mask)[0]
            for lead, ipoint in zip(valid_leads, valid_ipoints, strict=False):
                lead_time_groups.setdefault(lead, []).append((arr_idx, 0, ipoint))

    # Get reference grid from first array for alignment
    ref_lat = arrays[0].lat.values
    ref_lon = arrays[0].lon.values
    ref_sort_idx = np.lexsort((ref_lon, ref_lat))
    ref_lat_sorted = ref_lat[ref_sort_idx]
    ref_lon_sorted = ref_lon[ref_sort_idx]

    # Process each lead time
    sorted_leads = sorted(lead_time_groups.keys())
    out = []

    for forecast_step, lead in enumerate(sorted_leads, start=1):
        # Group by array index to minimize selections
        array_groups = {}
        for arr_idx, sample_idx, ipoint in lead_time_groups[lead]:
            array_groups.setdefault(arr_idx, {}).setdefault(sample_idx, []).append(ipoint)

        per_sample = []
        for arr_idx, sample_dict in array_groups.items():
            da = arrays[arr_idx]

            for sample_idx, ipoint_list in sample_dict.items():
                # Single selection operation
                ipoint_arr = np.array(ipoint_list)
                da_subset = da.isel(ipoint=ipoint_arr)

                # Align to reference grid
                sort_idx = np.lexsort((da_subset.lon.values, da_subset.lat.values))
                da_subset = da_subset.isel(ipoint=sort_idx).assign_coords(
                    ipoint=np.arange(len(ipoint_arr)),
                    lat=("ipoint", ref_lat_sorted[: len(ipoint_arr)]),
                    lon=("ipoint", ref_lon_sorted[: len(ipoint_arr)]),
                )

                # Ensure sample dimension
                if "sample" not in da_subset.dims:
                    sample_val = da.sample.values.item() if da.sample.ndim == 0 else sample_idx
                    da_subset = da_subset.expand_dims(sample=[sample_val])

                per_sample.append(da_subset)

        if per_sample:
            # Single concat operation
            combined = xr.concat(per_sample, dim="sample", coords="different", compat="equals")
            combined = combined.assign_coords(
                ipoint=np.arange(combined.sizes["ipoint"]), forecast_step=forecast_step
            )
            out.append(combined)

    return out


def _add_lead_time_coord(da: xr.DataArray, sample_dim="sample") -> xr.DataArray:
    """
    Add lead_time coordinate computed as:
    valid_time - source_interval_start

    lead_time has dims (sample, ipoint) and dtype timedelta64[ns].

    Parameters
    ----------
    da :
        Input DataArray
    sample_dim :
        The name of the sample dimension (default is "sample") which should be kept.
        Collapse over the others.
    Returns
    -------
        Returns a DataArray with the lead_time coordinate added.

    NB. Need to be used AFTER splitting by valid_time and stacking by sample,
    so that all valid_times within a sample are the same and we can assign a
    single lead_time per sample.

    """
    vt = da["valid_time"].values
    sis = da["source_interval_start"].values
    # Compute lead_time: valid_time - source_interval_start
    if vt.ndim > 1:
        sis_expanded = sis[:, np.newaxis] if sis.ndim == 1 else sis
        lead_time_values = vt - sis_expanded
        # Get unique lead_time per sample, verify consistency
        lead_times = [
            np.unique(lead_time_values[i][~np.isnat(lead_time_values[i])])
            for i in range(lead_time_values.shape[0])
        ]
        if any(len(lt) != 1 for lt in lead_times):
            raise ValueError(
                "Inconsistent lead_time values within samples for "
                f"forecast_step {da.forecast_step.values}"
            )
        lead_time_per_sample = np.array([lt[0] for lt in lead_times])
    else:
        lead_time_values = vt - sis
        lead_time_per_sample = np.unique(lead_time_values[~np.isnat(lead_time_values)])

    # Verify all samples have same lead_time for this forecast_step
    unique_lead = np.unique(lead_time_per_sample)
    if len(unique_lead) != 1:
        raise ValueError(
            "Multiple lead_time values across samples for "
            f"forecast_step {da.forecast_step.values}: {unique_lead}"
        )

    da = da.assign_coords(lead_time=unique_lead[0])
    return da


def _force_consistent_grids(ref: list[xr.DataArray]) -> xr.DataArray:
    """
    Force all samples to share the same ipoint order.

    Parameters
    ----------
    ref:
       Input dataset
    Returns
    -------
        Returns a Dataset where all samples have the same lat lon and ipoint ordering
    """

    # Pick first sample as reference
    ref_lat = ref[0].lat
    ref_lon = ref[0].lon

    sort_idx = np.lexsort((ref_lon.values, ref_lat.values))
    npoints = sort_idx.size
    aligned = []
    samples = []
    for i, a in enumerate(ref):
        a_sorted = a.isel(ipoint=sort_idx)
        samples.append(a_sorted.sample.values)
        a_sorted = a_sorted.assign_coords(
            ipoint=np.arange(npoints),
            lat=("ipoint", ref_lat.values[sort_idx]),
            lon=("ipoint", ref_lon.values[sort_idx]),
        )

        if "sample" not in a_sorted.dims:
            a_sorted = a_sorted.expand_dims(sample=[i])

        aligned.append(a_sorted)

    return aligned  # xr.concat(aligned, dim="sample")
