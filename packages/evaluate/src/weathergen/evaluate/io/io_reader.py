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
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import omegaconf as oc

# Third-party
import xarray as xr

from weathergen.evaluate.utils.derived_channels import is_derivable_channel

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


@dataclass
class ReaderOutput:
    """
    Dataclass to hold the output of the Reader.get_data method.
    Attributes
    ----------
    target : dict[str, xr.Dataset]
        Dictionary of xarray Datasets for targets, indexed by forecast step.
    prediction : dict[str, xr.Dataset]
        Dictionary of xarray Datasets for predictions, indexed by forecast step.
    """

    target: dict[str, xr.Dataset]
    prediction: dict[str, xr.Dataset]


@dataclass
class DataAvailability:
    """
    Dataclass to hold information about data availability in the input files.
    Attributes
    ----------
    score_availability: bool
        True if the metric file contains the requested combination.
    channels:
        List of channels requested
    fsteps:
        List of forecast steps requested
    samples:
        List of samples requested
    ensemle:
        List of ensemble member identifiers
    """

    score_availability: bool
    channels: list[str] | None
    fsteps: list[int] | None
    samples: list[int] | None
    ensemble: list[str] | None = None


class Reader(ABC):
    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict[str, str] | None = None):
        """
        Generic data reader class.

        Parameters
        ----------
        eval_cfg : dict
            Config with plotting and evaluation options for that run id.
        run_id : str
            Run identifier of the model
        private_paths:
            Dictionary of private paths for the supported HPC
        """
        self.eval_cfg = eval_cfg
        self.run_id = run_id
        self.private_paths = private_paths
        self.streams = list(eval_cfg.streams.keys())
        # TODO: propagate it to the other functions using global plotting opts
        self.global_plotting_options = eval_cfg.get("global_plotting_options", {})

        # Default paths if not provided
        self.model_base_dir = eval_cfg.get("model_base_dir")
        self.results_base_dir = eval_cfg.get("results_base_dir")

    def get_eval_settings(self, stream: str) -> dict:
        """Extract evaluation settings that affect score computation for a stream.

        These settings are saved into JSON score files so that cached scores can
        be invalidated when settings change (e.g. regridding, ensemble selection,
        rank files, aggregation dimensions).

        Parameters
        ----------
        stream : str
            The stream name.

        Returns
        -------
        dict
            A dictionary of plain Python types (JSON-serializable).  Only
            non-default / non-empty values are included to keep the JSON compact
            and avoid spurious mismatches.
        """
        stream_cfg = self.get_stream(stream)

        # Collect raw config values; skip falsy / default entries
        entries = {
            "regrid": stream_cfg.get("regrid"),
            "ensemble": stream_cfg.get("evaluation", {}).get("ensemble"),
            "rank": self.eval_cfg.get("rank"),
            "agg_dims": self.eval_cfg.get("agg_dims"),
            "derived_channels": stream_cfg.get("derived_channels"),
        }
        # Defaults that should not trigger cache invalidation
        defaults = {"ensemble": "all", "rank": "all", "agg_dims": "ipoint"}

        settings: dict = {}
        for key, val in entries.items():
            if not val or val == defaults.get(key):
                continue
            settings[key] = (
                oc.OmegaConf.to_container(val, resolve=True) if oc.OmegaConf.is_config(val) else val
            )

        # Store the source (zarr) forecast steps so the cache is invalidated
        # when the underlying data changes (e.g. new rollout length).
        settings["forecast_steps"] = sorted(int(f) for f in self.get_forecast_steps())

        return settings

    def get_stream(self, stream: str):
        """
        returns the dictionary associated to a particular stream

        Parameters
        ----------
        stream: str
            the stream name

        Returns
        -------
        dict
            the config dictionary associated to that stream
        """
        return self.eval_cfg.streams.get(stream, {})

    @abstractmethod
    def get_samples(self) -> set[int]:
        """Placeholder implementation of sample getter. Override in subclass."""
        pass

    @abstractmethod
    def get_forecast_steps(self) -> set[int]:
        """Placeholder implementation forecast step getter. Override in subclass."""
        pass

    # TODO: get this from config
    @abstractmethod
    def get_channels(self, stream: str | None = None) -> list[str]:
        """Placeholder implementation channel names getter. Override in subclass."""
        pass

    @abstractmethod
    def get_ensemble(self, stream: str | None = None) -> list[str]:
        """Placeholder implementation ensemble member names getter. Override in subclass."""
        pass

    def is_gridded_data(self, stream: str) -> bool:
        """
        Placeholder implementation to check if lat/lon are regularly spaced.
        Override in subclass.
        """
        return True

    @abstractmethod
    def load_scores(self, stream: str, region: str, metric: str) -> xr.DataArray:
        """Placeholder to load pre-computed scores for a given run, stream, metric"""
        pass

    def check_availability(
        self,
        stream: str,
        available_data: dict | None = None,
        mode: str = "evaluation",
    ) -> DataAvailability:
        """
        Check if requested channels, forecast steps and samples are
        i) available in the previously saved metric file if specified (return False otherwise)
        ii) available in the source file (e.g. the Zarr file, return error otherwise)
        Additionally, if channels, forecast steps or samples is None/'all', it will
        i) set the variable to all available vars in source file
        ii) return True only if the respective variable contains the same indices in
            metric file and source file (return False otherwise)

        Parameters
        ----------
        stream : str
            The stream considered.
        available_data : dict or None
            Available data loaded from metric file.
        mode : str
            Mode string. Can be 'evaluation' or 'plotting'.

        Returns
        -------
        DataAvailability
            A dataclass containing:
            - channels: list of channels or None if 'all'
            - fsteps: list of forecast steps or None if 'all'
            - samples: list of samples or None if 'all'
            - ensemble: list of ensembleor None if 'all'
        """

        # Fill requested info for channels, fsteps, samples, ensemble
        requested_data = self._get_channels_fsteps_samples(stream, mode)

        if not requested_data.score_availability:
            return requested_data

        channels = requested_data.channels
        fsteps = requested_data.fsteps
        samples = requested_data.samples
        ensemble = requested_data.ensemble
        requested = {
            "channel": set(channels) if channels is not None else None,
            "fstep": set(fsteps) if fsteps is not None else None,
            "sample": set(samples) if samples is not None else None,
            "ensemble": set(ensemble) if ensemble is not None else None,
        }

        # Extract available info from metric file (if provided)
        available = {
            "channel": (
                set(available_data["channel"].values.ravel())
                if available_data is not None
                else set()
            ),
            "fstep": (
                set(available_data["forecast_step"].values.ravel())
                if available_data is not None
                else set()
            ),
            "sample": (
                set(available_data.coords["sample"].values.ravel())
                if available_data is not None
                else set()
            ),
            "ensemble": (
                set(available_data["ens"].values.ravel())
                if (available_data is not None and "ens" in available_data.coords)
                else set()
            ),
        }

        # Extract actual reader data (from source)
        reader_data = {
            "fstep": set(int(f) for f in self.get_forecast_steps()),
            "sample": set(int(s) for s in self.get_samples()),
            "channel": set(self.get_channels(stream)),
            "ensemble": set(self.get_ensemble(stream)),
        }

        check_score = True
        corrected = False

        for name in ["channel", "fstep", "sample", "ensemble"]:
            if requested[name] is None:
                # Default to all in Zarr (or to cached score coords if available)
                if available_data is not None and available[name]:
                    # Trust the cached score's coordinates (may include expanded
                    # sub-steps beyond zarr base fsteps).  Validity is guaranteed
                    # by the eval_settings check in load_single_score.
                    requested[name] = available[name]
                else:
                    requested[name] = reader_data[name]

            # Must be subset of source data (skip when validating a cached score,
            # since scores may contain expanded sub-steps beyond zarr base fsteps)
            if available_data is not None:
                pass
            elif not requested[name] <= reader_data[name]:
                missing = requested[name] - reader_data[name]

                # Special handling for ensemble mean and std
                if name == "ensemble" and "mean" in missing:
                    missing.remove("mean")
                if name == "ensemble" and "std" in missing:
                    missing.remove("std")

                # Derivable channels (e.g. 10ff) will be computed later by
                # DeriveChannels — keep them in the requested set.
                if name == "channel":
                    derivable = {ch for ch in missing if is_derivable_channel(ch)}
                    if derivable:
                        _logger.debug(
                            f"Channels {derivable} not in source but recognised "
                            f"as derivable – keeping them."
                        )
                    missing -= derivable

                if missing:
                    _logger.info(
                        f"Requested {name}(s) {missing} is unavailable. "
                        f"Removing missing {name}(s) for {mode}."
                    )
                    requested[name] = requested[name] - missing
                    corrected = True

            # Must be a subset of available_data (if provided)
            if available_data is not None and not requested[name] <= available[name]:
                missing = requested[name] - available[name]
                _logger.info(
                    f"{name.capitalize()}(s) {missing} missing in previous "
                    "evaluation. Recomputation required."
                )
                check_score = False

        if check_score and not corrected:
            scope = "metric file" if available_data is not None else "Zarr file"
            _logger.info(
                f"All checks passed – All channels, samples, fsteps requested for {mode} are "
                f"present in {scope}..."
            )

        return DataAvailability(
            score_availability=check_score,
            channels=sorted(list(requested["channel"])),
            fsteps=sorted(list(requested["fstep"])),
            samples=sorted(list(requested["sample"])),
            ensemble=sorted(list(requested["ensemble"])),
        )

    def _get_channels_fsteps_samples(self, stream: str, mode: str) -> DataAvailability:
        """
        Get channels, fsteps and samples for a given run and stream from the config.
        Replace 'all' with None.

        Parameters
        ----------
        stream: str
            The stream considered.
        mode: str
            if plotting or evaluation mode

        Returns
        -------
        DataAvailability
            A dataclass containing:
            - channels: list of channels or None if 'all'
            - fsteps: list of forecast steps or None if 'all'
            - samples: list of samples or None if 'all'
        """

        # Helper function to process range strings like '1-3' into lists [1,2,3]
        def _parse_range_list(value, name):
            if isinstance(value, str) and value != "all":
                assert re.match(r"^\d+-\d+$", value), (
                    f"String format for {name} in config must be "
                    f"'digit-digit' or 'all'. "
                    f"Got '{value}'."
                )
                start, end = map(int, value.split("-"))
                return list(range(start, end + 1))
            return value

        # Normalize None vs "all"
        def normalize(val):
            return (
                None
                if (val == "all" or val is None)
                else list(val)
                if isinstance(val, list)
                else val
            )

        assert mode in ("plotting", "evaluation"), (
            f"Mode must be either 'plotting' or 'evaluation'. Got '{mode}' instead."
        )

        stream_cfg = self.get_stream(stream)
        if not stream_cfg.get(mode, False):
            _logger.warning(
                f"Mode '{mode}' does not exist in stream config for '{stream}'. Skipping."
            )
            return DataAvailability(
                score_availability=False,
                channels=None,
                fsteps=None,
                samples=None,
                ensemble=None,
            )

        samples = stream_cfg[mode].get("sample", None)
        fsteps = stream_cfg[mode].get("forecast_step", None)
        channels = stream_cfg.get("channels", None)
        ensemble = stream_cfg[mode].get("ensemble", None)

        if ensemble == "mean":
            ensemble = ["mean"]
        elif ensemble == "std":
            ensemble = ["std"]

        fsteps = _parse_range_list(fsteps, "forecast_step")
        samples = _parse_range_list(samples, "sample")

        return DataAvailability(
            score_availability=True,
            channels=normalize(channels),
            fsteps=normalize(fsteps),
            samples=normalize(samples),
            ensemble=normalize(ensemble),
        )
