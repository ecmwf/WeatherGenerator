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

# Local application / package
from weathergen.common.config import (
    get_path_run,
    load_merge_configs,
    load_run_config,
)
from weathergen.common.io import zarrio_reader
from weathergen.evaluate.io.data.dataarray_builders import EnsembleSelect
from weathergen.evaluate.io.data.io_orchestration import (
    _build_io_state,
    get_data_dirstore,
    get_data_zipstore,
    get_num_workers,
)
from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.scores.score_utils import to_list

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

        # for backward compatibility allow metric_dir to be specified in the run config
        self.results_dir = Path(self.results_base_dir)
        self.runplot_dir = Path(self.runplot_base_dir)
        self.metrics_dir = Path(
            self.eval_cfg.get("metrics_dir", self.metrics_base_dir / "evaluation")
        )

    def get_inference_config(self):
        """
        Load the config associated to the inference run (different from the
        eval_cfg which contains plot and evaluation options.)

        Returns
        -------
        config: dict
            Configuration file from the inference run
        """
        config = {}

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

        if not isinstance(config, dict | oc.DictConfig):
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
        path: str | None
            Full climatology path if available, otherwise None.
        """
        stream_dict = self.get_stream(stream)
        explicit_path = stream_dict.get("climatology_path", None)
        if explicit_path:
            return str(explicit_path)

        clim_base_dir = self.inference_cfg.get("data_path_aux", None)
        if not clim_base_dir:
            _logger.warning(
                "No 'data_path_aux' defined in inference config."
                " Cannot infer climatology path for stream %s.",
                stream,
            )
            return None

        streams = self.inference_cfg.get("streams", {})
        if isinstance(streams, list | oc.ListConfig):
            streams = {s["name"]: s for s in streams}
        streams = oc.OmegaConf.create(streams)
        clim_fn = next(streams.get("filenames"), None)

        try:
            clim_fn = streams[stream].get("filenames")
        except KeyError:
            clim_fn = None

        if isinstance(clim_fn, oc.ListConfig) and len(clim_fn) == 1:
            climatology_partial_filename = clim_fn[0]
        else:
            _logger.warning(
                f"Many source filenames found for stream {stream} in model config."
                " In that case the climatology filename should be specified"
                " explicitly via 'climatology_path' in the evaluation config."
            )
            return None

        clim_data_path = (
            Path(clim_base_dir)
            / "climatology"
            / climatology_partial_filename.replace(".zarr", "_climatology.zarr")
        )

        if not clim_data_path.exists():
            _logger.warning(
                f"Climatology file {clim_data_path} does not exist or configuration is invalid."
                " Setting climatology to NaN."
                " Please check that the path is correct and that the file exists."
            )
            return None
        else:
            _logger.info(f"Using climatology file: {clim_data_path}")

        return str(clim_data_path)

    def get_channels(self, stream: str) -> list[str]:
        """
        Get the list of channels for a given stream from the config.

        Parameters
        ----------
        stream : str
            The name of the stream to get channels for.

        Returns
        -------
        all_channels: list[str]
            A list of channel names.
        """
        _logger.debug(f"Getting channels for stream {stream}...")
        all_channels = self.get_inference_stream_attr(stream, "val_target_channels")
        _logger.debug(f"Channels found in config: {all_channels}")
        return all_channels

    def load_scores(
        self, stream: str, regions: list[str], metrics: dict[str, object]
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
            for metric, parameters in metrics.items():
                score = self.load_single_score(stream, region, metric, parameters)
                if score is None:
                    # all other cases: recompute scores
                    missing_metrics.setdefault(region, {}).update({metric: parameters})
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
                    else:
                        # JSON exists but doesn't cover the requested data — recompute.
                        missing_metrics.setdefault(region, {}).update({metric: parameters})

        recomputable_missing_metrics = self.get_recomputable_metrics(missing_metrics)
        return local_scores, recomputable_missing_metrics

    def load_single_score(
        self, stream: str, region: str, metric: str, parameters: dict | None = None
    ) -> xr.DataArray | None:
        """
        Load a single pre-computed score for a given run, stream and metric.

        Returns
        -------
        score: xr.DataArray or None
            DataArray of the score if found, else None.
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

    def get_recomputable_metrics(self, metrics: dict) -> dict:
        """
        Determine which metrics can be recomputed.

        Parameters
        ----------
        metrics : dict
            Dictionary mapping regions to missing metrics.

        Returns
        -------
        metrics: dict
            Same as input
        """
        return metrics

    def get_inference_stream_attr(self, stream_name: str, key: str, default=None):
        """
        Get the value of a key for a specific stream from the a model config.

        Parameters:
        ------------
            stream_name: str
                The name of the stream (e.g. 'ERA5').
            key: str
                The key to look up (e.g. 'tokenize_spacetime').
            default: Optional
                Value to return if not found (default: None).

        Returns:
        ------------
            The parameter value if found, otherwise the default.
        """

        streams = self.inference_cfg.get("streams", {})
        if isinstance(streams, list | oc.ListConfig):
            for stream in streams:
                if stream.get("name") == stream_name:
                    return stream.get(key, default)
        else:
            return streams.get(stream_name, {}).get(key, default)

        return default


class WeatherGenJsonReader(WeatherGenReader):
    def __init__(
        self,
        eval_cfg: dict,
        run_id: str,
        private_paths: dict | None = None,
        regions: list[str] | None = None,
        metrics: dict[str, object] | None = None,
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
                for metric, parameters in metrics.items():
                    score = self.load_single_score(stream, region, metric, parameters)
                    if score is not None:
                        for name in coord_names:
                            vals = set(score[name].values)
                            all_coords[name].append(vals)
                            for val in vals:
                                provenance[name][val].append((stream, region, metric))

        common_coords = {name: set.intersection(*all_coords[name]) for name in coord_names}

        # Warn about any skipped coordinates
        for name in coord_names:
            skipped = set.union(*all_coords[name]) - common_coords[name]
            if skipped:
                msg_lines = [
                    f"Some {name}(s) were not common across streams, regions, and metrics:"
                ]
                for val in skipped:
                    msg_lines.append(f"  {val} only present in {provenance[name][val]}")
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
        assert False, f"Missing JSON data for run {self.run_id}."

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

        assert fname_zarr.exists(), f"Zarr file {fname_zarr} does not exist."

        assert (zarr_ext == "zarr" and fname_zarr.is_dir()) or (
            zarr_ext == "zip" and fname_zarr.is_file()
        ), (
            f"Zarr file {fname_zarr} has unexpected format. ({zarr_ext}). "
            f"Expected directory for 'zarr' or file for 'zip'."
        )
        self.fname_zarr = fname_zarr

        # Metadata caches — populated lazily on first access
        self._cached_samples: set[int] | None = None
        self._cached_fsteps: set[int] | None = None
        self._cached_streams: set[str] | None = None
        self._cached_ensemble: dict[str, list[str]] = {}
        self._cached_is_gridded: dict[str, bool] = {}

        # Raw I/O worker config (direct zarr access)
        self._max_workers: int | None = eval_cfg.get("max_workers")
        self._num_io_workers: int = get_num_workers(max_workers=self._max_workers)

    def get_data(
        self,
        stream: str,
        samples: list[int] | None = None,
        fsteps: list[int] | None = None,
        channels: list[str] | None = None,
        ensemble: list[str] | None = None,
    ) -> ReaderOutput:
        """Load prediction and target data via direct zarr array access.

        Parameters
        ----------
        stream : str
            Stream name to retrieve data for.
        samples, fsteps, channels, ensemble
            Optional filters; ``None`` means "all".

        Returns
        -------
        ReaderOutput
            target/prediction dicts of xarray DataArrays keyed by forecast step.
        """
        resolved_ensemble = to_list(ensemble or self.get_ensemble(stream))
        ens_select = EnsembleSelect.from_names(resolved_ensemble, self.get_ensemble(stream))
        state = _build_io_state(
            self.run_id,
            self.fname_zarr,
            stream,
            self.get_stream(stream),
            self.get_channels(stream),
            self.is_gridded_data(stream),
            sorted(int(f) for f in (fsteps or self.get_forecast_steps())),
            sorted(int(s) for s in (samples or self.get_samples())),
            to_list(channels or self.get_stream(stream).get("channels", self.get_channels(stream))),
            resolved_ensemble,
            self._num_io_workers,
            ens_select,
        )
        get_data = get_data_zipstore if state.is_zip else get_data_dirstore
        return get_data(state)

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
        if self._cached_streams is None:
            with zarrio_reader(self.fname_zarr) as zio:
                self._cached_streams = set(zio.streams)

        if stream in self._cached_streams:
            return self.eval_cfg.streams.get(stream, {})
        return {}

    def get_samples(self) -> set[int]:
        """Get the set of sample indices from the Zarr file."""
        if self._cached_samples is None:
            with zarrio_reader(self.fname_zarr) as zio:
                self._cached_samples = set(int(s) for s in zio.samples)
        return self._cached_samples

    def get_forecast_steps(self) -> set[int]:
        """Get the set of forecast steps from the Zarr file."""
        if self._cached_fsteps is None:
            with zarrio_reader(self.fname_zarr) as zio:
                self._cached_fsteps = set(int(f) for f in zio.forecast_steps)
        return self._cached_fsteps

    def get_forecast_substep_valid_times(self, stream: str) -> set[str]:
        """Get the set of forecast times from the Zarr file."""
        if not self.is_gridded_data(stream):
            _logger.warning(f"Stream {stream} is not gridded. Forecast times cannot be retrieved.")
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

        if stream not in self._cached_ensemble:
            # TODO: improve this to get ensemble from io class
            with zarrio_reader(self.fname_zarr) as zio:
                dummy = zio.get_data(0, stream, zio.forecast_steps[0])
            self._cached_ensemble[stream] = list(dummy.prediction.as_xarray().coords["ens"].values)
        return self._cached_ensemble[stream]

    def is_gridded_data(self, stream: str) -> bool:
        """Check if the latitude and longitude coordinates are regularly spaced for a given stream.
        Parameters
        ----------
        stream :
            The name of the stream to get channels for.

        Returns
        -------
            True if the stream is regularly spaced. False otherwise.
        """
        if stream not in self._cached_is_gridded:
            self._cached_is_gridded[stream] = self._compute_is_gridded(stream)
        return self._cached_is_gridded[stream]

    def _compute_is_gridded(self, stream: str) -> bool:
        """is_gridded_data logic, called once per stream and cached."""
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
        else:
            _logger.debug("Latitude and longitude coordinates are regularly spaced.")
            return True
