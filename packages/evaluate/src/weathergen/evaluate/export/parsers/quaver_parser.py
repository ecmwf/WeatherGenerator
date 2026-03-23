# pylint: disable=bad-builtin

import logging
from pathlib import Path

import earthkit.data as ekd
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf

from weathergen.evaluate.export.cf_utils import CfParser

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

"""
Usage: 

uv run export --run-id ciga1p9c --stream ERA5 
--output-dir ./test_output1 
--format quaver --type prediction target  
--samples 2 --fsteps 2 
--quaver-template-folder "<path to quaver templates> --quaver-template-grid-type o96 
--expver test 

NOTE: check if it is o96 or O96 in the template.
"""


class QuaverParser(CfParser):
    """
    Child class for handling Quaver output format.
    """

    def __init__(self, config: OmegaConf, **kwargs):
        """
        Initialize Quaver parser with configuration and additional parameters.
        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        if not hasattr(self, "quaver_template_folder") or self.quaver_template_folder is None:
            raise ValueError("Template folder must be provided for Quaver format.")
        if not hasattr(self, "quaver_template_grid_type") or self.quaver_template_grid_type is None:
            raise ValueError("Template grid type must be provided for Quaver format.")
        if not hasattr(self, "channels") or self.channels is None:
            raise ValueError("Channels must be provided for Quaver format.")
        if not hasattr(self, "expver") or self.expver is None:
            raise ValueError("Expver must be provided for Quaver format.")
        super().__init__(config, **kwargs)

        self.template_cache = []

        self.template = str(
            Path(self.quaver_template_folder)
            / f"aifs_{{level_type}}_{self.quaver_template_grid_type}_data.grib"
        )

        self.pl_template = ekd.from_source("file", self.template.format(level_type="pl"))
        self.sf_template = ekd.from_source("file", self.template.format(level_type="sfc"))

        self.encoder = ekd.create_encoder("grib")

        self.pl_file = ekd.create_target("file", self.get_output_filename("pl"))
        self.sf_file = ekd.create_target("file", self.get_output_filename("sfc"))

        self.template_cache = self.cache_templates()

        # Cached sort order for assign_coords — computed on first call.
        self._coord_order = None

        # Pre-compute per-variable metadata that is constant across all samples/fsteps.
        self._var_info = {}  # var -> (level, level_type, scale_factor, add_offset)
        for var in self.channels:
            _, level, level_type = self.extract_var_info(var)
            var_short = var.split("_")[0] if "_" in var else var
            var_config = self.mapping.get(var_short, {})
            raw = var_config.get("scale_factor", "1.0")
            if isinstance(raw, str) and "/" in raw:
                parts = raw.split("/")
                scale_factor = float(parts[0]) / float(parts[1])
            else:
                scale_factor = float(raw)
            add_offset = float(var_config.get("add_offset", 0.0))
            self._var_info[var] = (level, level_type, scale_factor, add_offset)

    def process_sample(
        self,
        fstep_iterator_results: iter,
        ref_time: np.datetime64,
    ):
        """
        Process results from get_data_worker: reshape, concatenate, add metadata, and save.
        Parameters
        ----------
            fstep_iterator_results : Iterator over results from get_data_worker.
            ref_time : Forecast reference time for the sample.
        Returns
        -------
            None
        """
        for result in fstep_iterator_results:
            if result is None:
                continue

            # result is already a materialized xarray DataArray (built in the worker).
            if not isinstance(result, xr.DataArray):
                result = result.as_xarray().squeeze()
            result = result.sel(channel=self.channels)
            da_fs = self.assign_coords(result)

            # forecast_step may be a scalar (from lightweight worker) or array
            fs_val = result.forecast_step.values
            step = int(fs_val) if np.ndim(fs_val) == 0 else int(np.unique(fs_val).item())

            # Extract the full numpy data and channel list once to avoid
            # repeated xarray .sel() overhead in the inner loop.
            all_values = da_fs.values  # (npoints, nchannels)
            channel_list = list(da_fs.channel.values)
            channel_to_col = {ch: i for i, ch in enumerate(channel_list)}

            sf_fields = []
            pl_fields = []
            for var in self.channels:
                level, level_type, scale_factor, add_offset = self._var_info[var]

                col = channel_to_col[var]
                field_values = all_values[:, col]

                if scale_factor != 1.0 or add_offset != 0.0:
                    field_values = field_values * scale_factor + add_offset

                template_field = self.template_cache.get((var, level), None)
                if template_field is None:
                    continue

                metadata = self.get_metadata(ref_time=ref_time, step=step, level=level)

                encoded = self.encoder.encode(
                    values=field_values, template=template_field, metadata=metadata
                )

                field_list = pl_fields if level_type == "pl" else sf_fields
                field_list.append(encoded.to_field())

            self.save(pl_fields, "pl")
            self.save(sf_fields, "sfc")

        _logger.info(f"Saved sample data to {self.output_format} in {self.output_dir}.")

    def extract_var_info(self, var: str) -> tuple[str, str, str]:
        """
        Extract variable short name, level, and level type from variable string.
        Parameters
        ----------
            var : str
                Variable string (e.g., 'temperature_850').
        Returns
        -------
            tuple[str, str, str]
                Variable short name, level, and level type.
        """
        var_short = var.split("_")[0] if "_" in var else var
        level = int(var.split("_")[-1]) if "_" in var else "sfc"

        var_config = self.mapping.get(var_short, {})
        if not var_config:
            raise ValueError(
                f"Variable '{var} (using: {var_short})' not found in configuration mapping."
            )

        level_type = var_config.get("level_type", "None")

        return var_short, level, level_type

    def cache_templates(self) -> dict[tuple[str, str], object]:
        """
        Get the index of the template field for a given variable and level.

        Returns
        -------
            Template field matching the variable and level.

        """
        template_cache = {}
        for var in self.channels:
            var_short, level, level_type = self.extract_var_info(var)
            template = self.pl_template if level_type != "sfc" else self.sf_template

            criteria = {"shortName": var_short}
            if level_type != "sfc":
                criteria["level"] = level  # , "step": step}

            matching_messages = template.sel(**criteria)

            if matching_messages:
                template_cache[(var, level)] = matching_messages[0]
            else:
                _logger.error(f"Template field for variable '{var}' at level '{level}' not found.")

        return template_cache

    def get_output_filename(self, level_type: str) -> Path:
        """
        Generate output filename.
        Parameters
        ----------
            data_type : str
                Type of data (e.g., 'prediction' or 'target').
            level_type : str
                Level type (e.g., 'sfc', 'pl', etc.).
        Returns
        -------
            Path
                Output filename as a Path object.
        """
        return (
            Path(self.output_dir)
            / f"{self.data_type}_{level_type}_{self.run_id}_{self.expver}.{self.file_extension}"
        )

    def assign_coords(self, data: xr.DataArray) -> xr.DataArray:
        """
        Assign forecast reference time coordinate to the dataset.
        Parameters
        ----------
            data : xr.DataArray
                Input data array.
        Returns
        -------
            xr.DataArray
                Data array with assigned coordinates.
        """

        if {"lon", "lat"}.issubset(data.coords):
            lons = (data.lon.values + 360) % 360
            data = data.assign_coords(lon=("ipoint", lons))

            # Cache the sort order — the grid is identical across all fsteps,
            # so lexsort only needs to run once.
            if not hasattr(self, "_coord_order") or self._coord_order is None:
                self._coord_order = np.lexsort((data.lon.values, -data.lat.values))

            data = data.isel(ipoint=self._coord_order)
        return data

    def get_metadata(
        self,
        ref_time: pd.Timestamp,
        step: int,
        level: str,
    ):
        """
        Add metadata to the dataset attributes.
        """

        metadata = {
            "date": ref_time,
            "step": step * self.fstep_hours.astype(int),
            "expver": self.expver,
            "marsClass": "rd",
        }
        if level != "sfc":
            metadata["level"] = level
        return metadata

    def save(self, encoded_fields: list, level_type: str):
        """
        Save the dataset to a file.
        Parameters
        ----------
            encoded_fields : List
                List of encoded fields to write.
            level_type : str
                Level type ('pl' or 'sfc').
        Returns
        -------
            None
        """

        file = self.pl_file if level_type == "pl" else self.sf_file

        for field in encoded_fields:
            file.write(field)
