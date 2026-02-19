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
            _logger.info(
                f"Results directory obtained from private config: "
                f"{self.results_base_dir}"
            )
        else:
            _logger.info(f"Results directory parsed: {self.results_base_dir}")

        self.runplot_base_dir = Path(
            self.eval_cfg.get("runplot_base_dir", self.results_base_dir)
        )  # base directory where map plots and histograms will be stored

        self.metrics_base_dir = Path(
            self.eval_cfg.get("metrics_base_dir", self.results_base_dir)
        )  # base directory where score files will be stored

        self.step_hrs = self.inference_cfg.get("step_hrs", 1)

        # for backward compatibility allow metric_dir to be specified in the run config
        self.results_dir = Path(self.results_base_dir)
        self.runplot_dir = Path(self.runplot_base_dir)
        self.metrics_dir = Path(
            self.eval_cfg.get(
                "metrics_dir", self.metrics_base_dir / "evaluation"
            )
        )

    def get_inference_config(self):
        """
        Load the config associated to the inference run (different from the
        eval_cfg which contains plot and evaluation options.)

        Returns
        -------
        dict
            configuration file from the inference run
        """
        config = {}
        try:
            if self.private_paths:
                _logger.info(
                    f"Loading config for run {self.run_id} from private paths: "
                    f"{self.private_paths}"
                )
                config = load_merge_configs(self.private_paths, self.run_id,
                                            self.mini_epoch)
            else:
                _logger.info(
                    f"Loading config for run {self.run_id} from model directory: "
                    f"{self.model_base_dir}"
                )
                config = load_run_config(self.run_id, self.mini_epoch,
                                         self.model_base_dir)
        except Exception as e:
            _logger.warning(
                f"Failed to load inference config: {e}. Defaulting to empty dict."
            )

        if not isinstance(config, (dict, oc.DictConfig)):
            _logger.warning("Model config not found. inference config will be empty.")
            config = {}
        return config

    def get_climatology_filename(self, stream: str) -> str | None:
        """
        Get the climatology filename for a given stream from the inference
        configuration.

        Parameters
        ----------
        stream : str
            Name of the data stream.

        Returns
        -------
        str | None
            Full climatology path if available, otherwise None.
        """
        stream_dict = self.get_stream(stream)

        clim_data_path = stream_dict.get("climatology_path", None)
        if not clim_data_path:
            clim_base_dir = self.inference_cfg.get("data_path_aux", None)
            clim_fn = next(
                (
                    item.get("climatology_filename")
                    for item in self.inference_cfg.get("streams", [])
                    if item.get("name") == stream
                ),
                None,
            )
            if clim_base_dir and clim_fn:
                clim_data_path = Path(clim_base_dir) / clim_fn
            else:
                _logger.warning(
                    f"No climatology path specified for stream {stream}. Setting climatology to "
                    "NaN. Add 'climatology_path' to evaluation config to use metrics like ACC."
                )

        return str(clim_data_path) if clim_data_path else None

    def get_channels(self, stream: str) -> list[str]:
        """
        Get the list of channels for a given stream from the config.

        Parameters
        ----------
        stream : str
            The name of the stream to get channels for.

        Returns
        -------
        list[str]
            A list of channel names.
        """
        _logger.debug(f"Getting channels for stream {stream}...")
        all_channels = self.get_inference_stream_attr(
            stream, "val_target_channels"
        )
        _logger.debug(f"Channels found in config: {all_channels}")
        return all_channels

    def load_scores(
        self, stream: str, regions: list[str], metrics: list[str]
    ) -> tuple[dict, dict]:
        """
        Load multiple pre-computed scores for a given run, stream and metric
        and epoch.

        Parameters
        ----------
        stream : str
            Stream name.
        regions : list[str]
            Region names.
        metrics : list[str]
            Metric names.

        Returns
        -------
        tuple[dict, dict]
            - local_scores: dictionary of available scores.
            - recomputable_missing_metrics: dictionary of regions and metrics
              that must be recomputed (empty for JSON reader).
        """
        local_scores = {}
        missing_metrics = {}
        for region in regions:
            for metric in metrics:
                score = self.load_single_score(stream, region, metric)
                if score is None:
                    missing_metrics.setdefault(region, []).append(metric)
                else:
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
        recomputable_missing_metrics = self.get_recomputable_metrics(missing_metrics)
        return local_scores, recomputable_missing_metrics

    def load_single_score(
        self, stream: str, region: str, metric: str
    ) -> xr.DataArray | None:
        """
        Load a single pre-computed score for a given run, stream and metric.

        Returns
        -------
        xr.DataArray or None
            DataArray of the score if found, else None.
        """
        score_path = (
            Path(self.metrics_dir)
            / f"{self.run_id}_{stream}_{region}_{metric}_chkpt{self.mini_epoch:05d}.json"
        )
        _logger.debug(f"Looking for: {score_path}")

        score = None
        if score_path.exists():
            with open(score_path) as f:
                data_dict = json.load(f)
                score = xr.DataArray.from_dict(data_dict)

        return score

    def get_recomputable_metrics(self, metrics):
        """
        Determine which metrics can be recomputed.
        Parameters
        ----------
        metrics : dict
            Dictionary mapping regions to missing metrics.

        Returns
        -------
        dict
            Same as input
        """
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
        metrics: list[str] | None = None,
    ):
        super().__init__(eval_cfg, run_id, private_paths)
        self.common_coords: dict = self._compute_common_coords(regions, metrics)

    def _compute_common_coords(self, regions: list[str], metrics: list[str]) -> dict:
        # Find common coordinates across streams, regions, metrics.
        streams = list(self.streams)
        coord_names = ["sample", "forecast_step", "ens"]
        all_coords = {name: [] for name in coord_names}
        provenance = {name: defaultdict(list) for name in coord_names}
        
        for stream in streams:
            for region in regions:
                for metric in metrics:
                    score = self.load_single_score(stream, region, metric)
                    if score is not None:
                        for name in coord_names:
                            vals = set(score[name].values)
                            all_coords[name].append(vals)
                            for val in vals:
                                provenance[name][val].append((stream, region, metric))

        common_coords = {
            name: set.intersection(*all_coords[name])
            for name in coord_names
        }

        # Warn about any skipped coordinates
        for name in coord_names:
            skipped = set.union(*all_coords[name]) - common_coords[name]
            if skipped:
                msg_lines = [
                    f"Some {name}(s) were not common across streams, "
                    f"regions, and metrics:"
                ]
                for val in skipped:
                    msg_lines.append(
                        f"  {val} only present in {provenance[name][val]}"
                    )
                _logger.warning("\n".join(msg_lines))
    
        return common_coords

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
        # For backwards compatibility, assume zarr store is local (.zarr format).

        fname_zarr = self.results_dir.joinpath(
            f"validation_chkpt{self.mini_epoch:05d}_rank{self.rank:04d}.{zarr_ext}"
        )

        if fname_zarr.exists():
            if (zarr_ext == "zarr" and fname_zarr.is_dir()) or (
                zarr_ext == "zip" and fname_zarr.is_file()
            ):
                self.fname_zarr = fname_zarr
            else:
                _logger.error(
                    f"Zarr file {fname_zarr} exists but has unexpected format "
                    f"({zarr_ext}). Expected directory for 'zarr' or file for 'zip'."
                )
                raise FileNotFoundError(
                    f"Zarr file {fname_zarr} has unexpected format."
                )
        else:
            _logger.error(
                f"Zarr file {fname_zarr} does not exist."
            )
            raise FileNotFoundError(
                f"Zarr file {fname_zarr} does not exist."
            )

    def get_data(
        self,
        stream: str,
        samples: list[int] | None = None,
        fsteps: list[str] | None = None,
        channels: list[str] | None = None,
        ensemble: list[str] | None = None,
        return_counts: bool = False,
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
        return_counts :
            If True, also return the number of points per sample.

        Returns
        -------
        ReaderOutput
            A dataclass containing:
            - target: Dictionary of xarray DataArrays for targets, indexed by forecast step.
            - prediction: Dictionary of xarray DataArrays for predictions, indexed by forecast step.
            - points_per_sample: xarray DataArray containing the number of points per sample,
              if `return_counts` is True.
        """

        stream_cfg = self.get_stream(stream)
        all_channels = self.get_channels(stream)
        _logger.info(
            f"RUN {self.run_id}: Processing stream {stream}..."
        )

        fsteps = self.get_forecast_steps() if fsteps is None else fsteps

        # TODO: Avoid conversion of fsteps and sample to integers (as obtained from the ZarrIO)
        fsteps = sorted([int(fstep) for fstep in fsteps])
        samples = samples or sorted([int(sample) for sample in self.get_samples()])
        channels = channels or stream_cfg.get("channels", all_channels)
        channels = to_list(channels)

        ensemble = ensemble or self.get_ensemble(stream)
        ensemble = to_list(ensemble)

        dc = DeriveChannels(all_channels, channels, stream_cfg)

        da_tars, da_preds = [], []

        if return_counts:
            points_per_sample = xr.DataArray(
                np.full((len(fsteps), len(samples)), np.nan),
                coords={"forecast_step": fsteps, "sample": samples},
                dims=("forecast_step", "sample"),
                name=f"points_per_sample_{stream}",
            )
        else:
            points_per_sample = None

        fsteps_final = []

        with zarrio_reader(self.fname_zarr) as zio:
            for fstep in fsteps:
                _logger.info(f"RUN {self.run_id} - {stream}: Processing fstep {fstep}...")
                da_tars_fs, da_preds_fs, pps = [], [], []

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
                    pps.append(npoints)

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

                    da_tars_fs.append(target.squeeze())
                    da_preds_fs.append(pred.squeeze())

                if not da_tars_fs:
                    _logger.info(
                        f"[{self.run_id} - {stream}] No valid data found for fstep {fstep}."
                    )
                    continue

                fsteps_final.append(fstep)

                _logger.debug(
                    f"Concatenating targets and predictions for stream {stream}, "
                    f"forecast_step {fstep}..."
                )

                # faster processing
                if self.is_regular(stream):
                    # Efficient concatenation for regular grid
                    da_preds_fs = _force_consistent_grids(da_preds_fs)
                    da_tars_fs = _force_consistent_grids(da_tars_fs)

                    # add lead time coordinate
                    da_tars_fs = self.add_lead_time_coord(da_tars_fs)
                    da_preds_fs = self.add_lead_time_coord(da_preds_fs)
                else:
                    # Irregular (scatter) case. concatenate over ipoint
                    da_tars_fs = xr.concat(da_tars_fs, dim="ipoint")
                    da_preds_fs = xr.concat(da_preds_fs, dim="ipoint")

                if len(samples) == 1:
                    _logger.debug("Repeating sample coordinate for single-sample case.")
                    for da in (da_tars_fs, da_preds_fs):
                        da.assign_coords(
                            sample=(
                                "ipoint",
                                np.repeat(da.sample.values, da.sizes["ipoint"]),
                            )
                        )

                if set(channels) != set(all_channels):
                    _logger.debug(
                        f"Restricting targets and predictions to channels {channels} "
                        f"for stream {stream}..."
                    )

                    da_tars_fs, da_preds_fs, channels = dc.get_derived_channels(
                        da_tars_fs, da_preds_fs
                    )

                    da_tars_fs = da_tars_fs.sel(channel=channels)
                    da_preds_fs = da_preds_fs.sel(channel=channels)

                # apply z scaling if needed
                da_tars_fs = self.scale_z_channels(da_tars_fs, stream)
                da_preds_fs = self.scale_z_channels(da_preds_fs, stream)

                da_tars.append(da_tars_fs)
                da_preds.append(da_preds_fs)
                if return_counts:
                    points_per_sample.loc[{"forecast_step": fstep}] = np.array(pps)

            # Safer than a list
            da_tars = {fstep: da for fstep, da in zip(fsteps_final, da_tars, strict=True)}
            da_preds = {fstep: da for fstep, da in zip(fsteps_final, da_preds, strict=True)}

            return ReaderOutput(
                target=da_tars, prediction=da_preds, points_per_sample=points_per_sample
            )

    ######## reader utils ########

    def add_lead_time_coord(self, da: xr.DataArray, sample_dim="sample") -> xr.DataArray:
        """
        Add lead_time coordinate computed as:
        valid_time - source_interval_end

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
            Returns a Dataset with an added lead_time coordinate.
        """

        vt = da["valid_time"]
        sis = da["source_interval_start"]

        vt_reduced = vt.min(dim=[d for d in vt.dims if d != sample_dim])

        lead_time = vt_reduced - sis

        return da.assign_coords(lead_time=lead_time)

    def scale_z_channels(self, data: xr.DataArray, stream: str) -> xr.DataArray:
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
        if stream not in ["ERA5"]:
            return data

        channels_z = [ch for ch in np.atleast_1d(data.channel.values) if str(ch).startswith("z_")]
        factor = 9.80665

        if channels_z:
            channels = data.channel.astype(str)
            mask = channels.str.startswith("z_")
            data = data.where(~mask, data / factor)
        return data

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

    def is_regular(self, stream: str) -> bool:
        """
        Determine if the stream’s spatial grid is regular (lat/lon evenly spaced).

        Parameters
        ----------
        stream :
            The name of the stream.

        Returns
        -------
        bool
            True if lat/lon grids are consistent and regularly spaced across all samples,
            False otherwise.
        """
        _logger.debug(f"Checking regular spacing for stream '{stream}'...")

        # Early exit: if stream not found or no samples/forecast steps
        try:
            with zarrio_reader(self.fname_zarr) as zio:
                if stream not in zio.streams:
                    _logger.debug(f"Stream '{stream}' not found in Zarr. Treating as irregular.")
                    return False
                if not zio.samples or not zio.forecast_steps:
                    _logger.debug("No samples or forecast steps found. Treating as irregular.")
                    return False

                sample_idx = zio.samples[0]
                fstep_idx = zio.forecast_steps[0]
                dummy = zio.get_data(sample_idx, stream, fstep_idx)
                da = dummy.prediction.as_xarray()

                # Extract lat/lon; exit if missing or not 1D
                lat = da.get("lat")
                lon = da.get("lon")
                if lat is None or lon is None:
                    _logger.debug("Missing lat/lon coordinates in prediction data.")
                    return False
                lat = lat.squeeze()
                lon = lon.squeeze()
                if lat.ndim != 1 or lon.ndim != 1:
                    _logger.debug("Lat/lon not 1D. Irregular scatter grid.")
                    return False
        except Exception as e:
            _logger.debug(f"Exception during initial lat/lon check: {e}")
            return False

        # Verify regular spacing for lat and lon (monotonic, uniform step size)
        try:
            lat_vals = np.asarray(lat.values)
            lon_vals = np.asarray(lon.values)

            # Monotonicity check
            if not (np.all(np.diff(lat_vals) > 0) or np.all(np.diff(lat_vals) < 0)):
                _logger.debug("Latitude is not monotonically increasing or decreasing.")
                return False
            if not (np.all(np.diff(lon_vals) > 0) or np.all(np.diff(lon_vals) < 0)):
                _logger.debug("Longitude is not monotonically increasing or decreasing.")
                return False

            # Uniform spacing check for lat (allow tolerance for floating point noise)
            dlat = np.diff(lat_vals)
            if not np.allclose(dlat, dlat[0], rtol=1e-5, atol=1e-8):
                _logger.debug("Latitude spacing is non-uniform.")
                return False

            # Uniform spacing check for lon
            dlon = np.diff(lon_vals)
            if not np.allclose(dlon, dlon[0], rtol=1e-5, atol=1e-8):
                _logger.debug("Longitude spacing is non-uniform.")
                return False

            # Optional: verify consistency across a second sample/forecast step
            sample_idx2 = zio.samples[1] if len(zio.samples) > 1 else zio.samples[0]
            fstep_idx2 = zio.forecast_steps[1] if len(zio.forecast_steps) > 1 else zio.forecast_steps[0]
            if sample_idx2 == sample_idx and fstep_idx2 == fstep_idx:
                # Only one unique sample/step; assume consistency
                _logger.debug("Only one sample and one forecast step; using it for grid check.")
            else:
                dummy2 = zio.get_data(sample_idx2, stream, fstep_idx2)
                da2 = dummy2.prediction.as_xarray()
                lat2 = da2.get("lat").squeeze()
                lon2 = da2.get("lon").squeeze()
                if lat2 is None or lon2 is None or lat2.ndim != 1 or lon2.ndim != 1:
                    _logger.debug("Second sample/step missing lat/lon or not 1D.")
                    return False

                if not (np.allclose(lat2.values, lat_vals, rtol=1e-5, atol=1e-8) and
                        np.allclose(lon2.values, lon_vals, rtol=1e-5, atol=1e-8)):
                    _logger.debug("Lat/lon grids differ between samples.")
                    return False

            _logger.debug(f"Stream '{stream}' has a regular grid.")
            return True

        except Exception as e:
            _logger.debug(f"Exception during regular-spacing validation: {e}")
            return False


################### Helper functions ########################


def _force_consistent_grids(ref: list[xr.DataArray]) -> xr.DataArray:
    """
    Force all samples to share the same ipoint order.

    This function aligns the spatial ordering (lat/lon/ipoint) of all samples to that of the first sample,
    ensuring consistent spatial coordinates for subsequent concatenation. It is essential for regular-grid
    (gridded) data where spatial order matters but may differ across samples.

    Parameters
    ----------
    ref: list[xr.DataArray]
        List of xarray DataArrays, each representing one sample. Must have at least one element.
    Returns
    -------
    xr.DataArray
        A concatenated DataArray across the 'sample' dimension, where each sample's ipoint indices
        have been reordered to match the sorted lat/lon order of the first sample.

    Notes
    -----
    - All input DataArrays must share identical lat/lon values (though possibly in different orders).
    - Enforces consistent ipoint indexing after alignment (0..N-1).
    - Preserves and aligns all other coordinates and data variables.
    """
    if not ref:
        raise ValueError("_force_consistent_grids requires at least one input DataArray.")

    # Determine the reference sorting using the first sample's lat/lon
    ref_lat = ref[0]["lat"].values
    ref_lon = ref[0]["lon"].values
    sort_idx = np.lexsort((ref_lon, ref_lat))

    # Precompute aligned coordinates for efficiency
    n_points = len(sort_idx)
    aligned_lat = ref_lat[sort_idx]
    aligned_lon = ref_lon[sort_idx]
    aligned_ipoint = np.arange(n_points)

    # Reorder and align each sample
    aligned_samples = []
    for idx, sample in enumerate(ref):
        # Sort ipoint dimension by reference order
        sorted_sample = sample.isel(ipoint=sort_idx)

        # Reassign coordinates to enforce consistent ipoint and spatial coord values
        aligned_sample = sorted_sample.assign_coords(
            ipoint=aligned_ipoint,
            lat=("ipoint", aligned_lat),
            lon=("ipoint", aligned_lon),
        )

        # Ensure 'sample' dimension exists for concat; if missing, expand
        if "sample" not in aligned_sample.dims:
            aligned_sample = aligned_sample.expand_dims(sample=[idx])

        # Explicitly update the sample coordinate
        aligned_sample = aligned_sample.assign_coords(sample=[idx])
        aligned_samples.append(aligned_sample)

    # Concatenate along the sample dimension
    return xr.concat(aligned_samples, dim="sample")
