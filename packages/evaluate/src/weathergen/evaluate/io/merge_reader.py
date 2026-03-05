# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Standard library
import logging
from pathlib import Path

# Third-party
import xarray as xr

# Local application / package
from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.io.wegen_reader import WeatherGenJSONReader, WeatherGenZarrReader
from weathergen.evaluate.utils.utils import merge

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class WeatherGenMergeReader(Reader):
    def __init__(
        self,
        eval_cfg: dict,
        run_id: str,
        private_paths: dict | None = None,
        regions: list[str] | None = None,
        metrics: list[str] | None = None,
        reader_type: str = "zarr",
    ):
        """
        Data reader class for merging WeatherGenerator model outputs stored in Zarr or JSON format.

        Parameters
        ----------
        eval_cfg: dict
           config with plotting and evaluation options for that run id
        run_id: str
            run id of the model
        private_paths: dict
            dictionary of private paths for the supported HPC
        regions: list[str]
            names of predefined bounding box for a region (only used for WeatherGenJSONReader)
        metrics: list[str]
            names of the metric scores to compute (only used for WeatherGenJSONReader)
        reader_type: str
            The type of the internal reader. If zarr, WeatherGenZarrReader is used,
            WeatherGenJSONReader otherwise. Default: zarr
        """
        super().__init__(eval_cfg, run_id, private_paths)
        self.run_ids = eval_cfg.get("merge_run_ids", [])
        self.metrics_dir = Path(eval_cfg.get("merge_metrics_dir"))
        self.mini_epoch = eval_cfg.get("mini_epoch", 0)

        assert self.run_ids, (
            f"'merge_run_ids' must be non-empty in eval_cfg, but got: {self.run_ids}"
        )

        _logger.info(f"Initialising merge reader with {len(self.run_ids)} run(s): {self.run_ids}")

        self.readers: list[Reader] = []

        for i, run_id in enumerate(self.run_ids):
            _logger.debug(
                f"Creating internal reader {i + 1}/{len(self.run_ids)} for run_id '{run_id}' ..."
            )
            if reader_type == "zarr":
                reader = WeatherGenZarrReader(self.eval_cfg, run_id, self.private_paths)
            else:
                reader = WeatherGenJSONReader(
                    self.eval_cfg, run_id, self.private_paths, regions, metrics
                )
            self.readers.append(reader)
            _logger.debug(
                f"Instantiated reader for run_id '{run_id}' with {reader_type}"
                f""
            )

        _logger.info(
            f"Instantiated {len(self.readers)} internal readers of type {reader_type}."
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
            - prediction: Dictionary of xarray DataArrays for predictions, indexed by forecast
                          step.
            - points_per_sample: xarray DataArray containing the number of points per sample,
              if `return_counts` is True.
        """

        da_tars_merge, da_preds_merge, fsteps_merge = [], [], []

        points_per_sample = None

        for reader in self.readers:
            da_tars, da_preds, da_fsteps = [], [], []
            _logger.info(f"MERGE READERS: Processing run_id {reader.run_id}...")

            out = reader.get_data(
                stream,
                samples,
                fsteps,
                channels,
                ensemble="mean",
            )

            for fstep in out.target.keys():
                _logger.debug(f"MERGE READERS: Processing fstep {fstep}...")

                da_tars.append(out.target[fstep])
                da_preds.append(out.prediction[fstep])
                da_fsteps.append(fstep)

                if return_counts:
                    if points_per_sample is None:
                        points_per_sample = out.points_per_sample
                    else:
                        points_per_sample += out.points_per_sample

            da_tars_merge.append(da_tars)
            da_preds_merge.append(da_preds)
            fsteps_merge.append(da_fsteps)

        da_tars_merge = self._concat_over_ens(da_tars_merge, fsteps_merge)
        da_preds_merge = self._concat_over_ens(da_preds_merge, fsteps_merge)

        return ReaderOutput(
            target=da_tars_merge, prediction=da_preds_merge, points_per_sample=points_per_sample
        )

    def _concat_over_ens(self, da_merge, fsteps_merge):
        """
        Parameters
        ----------
        da_merge : list[list[xr.DataArray]]
            Outer list over readers, inner list over forecast steps.
        fsteps_merge : list[list[int]]
            Forecast steps per reader (must be identical across readers).

        Returns
        -------
        dict[int, xr.DataArray]
            DataArrays concatenated over new 'ens' dimension, keyed by fstep.
        """
        n_readers = len(da_merge)

        # use fsteps from first reader as reference
        fsteps = fsteps_merge[0]

        da_ens = {}
        for k, fstep in enumerate(fsteps):
            da_list = [da_merge[i][k] for i in range(n_readers)]
            da_ens[fstep] = xr.concat(da_list, dim="ens").assign_coords(ens=range(n_readers))

        return da_ens

    def load_scores(
        self, stream: str, regions: list[str], metrics: list[str]
    ) -> xr.DataArray | None:
        """
        Load the pre-computed scores for a given run, stream and metric and epoch.

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
        missing_metrics:
            dictionary of missing regions and metrics that need to be recomputed.
        """
        local_scores = {}
        missing_metrics = {}

        if isinstance(self.readers[0], WeatherGenZarrReader):
            # TODO: implement this properly. Not it is skipping loading scores
            for region in regions:
                for metric, parameters in metrics.items():
                    # all other cases: recompute scores
                    missing_metrics.setdefault(region, {}).update({metric: parameters})
        else:
            local_scores, missing_metrics = self._load_scores_json(stream, regions, metrics)
        return local_scores, missing_metrics

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
        for reader in self.readers:
            clim_data_path = reader.get_climatology_filename(stream)
            if clim_data_path:
                return clim_data_path
        return None

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
        stream_dict = self.eval_cfg.streams.get(stream, {})
        return stream_dict

    def get_samples(self) -> set[int]:
        """Get the set of sample indices from the Zarr file."""
        samples = []
        for reader in self.readers:
            samples.append(reader.get_samples())
        return set.intersection(*map(set, samples))

    def get_forecast_steps(self) -> set[int]:
        """Get the set of forecast steps from the Zarr file."""
        forecast_steps = []
        for reader in self.readers:
            forecast_steps.append(reader.get_forecast_steps())
        return set.intersection(*map(set, forecast_steps))

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
        all_channels = []

        for reader in self.readers:
            all_channels.append(reader.get_channels(stream))

        return set.intersection(*map(set, all_channels))

    def get_ensemble(self, stream: str | None = None) -> list[str]:
        """Get the list of ensemble member names for a given stream from the config.
        Parameters
        ----------
        stream :
            The name of the stream to get channels for.

        Returns
        -------
            A range of ensemble members equal to the number of merged readers.
        """
        _logger.debug(f"Getting ensembles for stream {stream}...")
        all_ensembles = []
        for reader in self.readers:
            all_ensembles.append(reader.get_ensemble(stream))

        assert all(e == ["0"] or e == [0] or e == {0} for e in all_ensembles), (
            "Merging readers with multiple ensemble members is not supported yet."
        )
        
        return set(range(len(self.readers)))

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
        return all(reader.is_regular(stream) for reader in self.readers)

    def _load_scores_json(self, stream, regions, metrics):
        "Concatenate the scores of all JSON readers"

        local_scores = {}
        missing_metrics = {}

        # deep merge dicts
        for reader in self.readers:
            scores, missing = reader.load_scores(stream, regions, metrics)
            merge(local_scores, scores)
            merge(missing_metrics, missing)

        # merge runs into one with all scores concatenated
        for metric in local_scores.keys():
            for region in local_scores[metric].keys():
                for stream in local_scores[metric][region].keys():
                    assert len(local_scores[metric][region][stream].keys()) == len(self.run_ids), (
                        f"Not all run ids are distinct or have the requested precomputed "
                        f"scores for metric: {metric}, region: {region}, stream: {stream}"
                    )

                    scores = (
                        local_scores[metric][region][stream].pop(run_id) for run_id in self.run_ids
                    )

                    local_scores[metric][region][stream].setdefault(
                        self.run_id,
                        xr.concat(scores, dim="ens").assign_coords(ens=range(len(self.readers))),
                    )

        return local_scores, missing_metrics
