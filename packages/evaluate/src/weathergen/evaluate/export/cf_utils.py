import logging
from pathlib import Path

import numpy as np
import xarray as xr

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class CfParser:
    """
    Base class for CF parsers.
    """

    def __init__(self, config, **kwargs):
        """
        CF-compliant parser that handles both regular and Gaussian grids.
        Parameters
        ----------
        config : OmegaConf
            Configuration defining variable mappings and dimension metadata.
        grid_type : str
            Type of grid ('regular' or 'gaussian').
        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.config = config
        self.file_extension = _get_file_extension(self.output_format)
        self.fstep_hours = np.timedelta64(self.fstep_hours, "h")
        self.mapping = config.get("variables", {})

    def get_output_filename(self) -> Path:
        """
        Generate output filename based on run_id and output directory.
        """
        return Path(self.output_dir) / f"{self.run_id}.{self.file_extension}"

    def process_sample(self, fstep_iterator_results: iter, ref_time: np.datetime64):
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
        pass

    def scale_data(self, data: xr.DataArray, var_short: str) -> xr.DataArray:
        """
        Scale data based on variable configuration.
        Parameters
        ----------
            data : xr.DataArray
                Input data array.
            var_short : str
                Variable name.
        Returns
        -------
            xr.DataArray
                Scaled data array.
        """
        var_config = self.mapping.get(var_short, {})
        raw = var_config.get("scale_factor", "1.0")
        parts = raw.split("/")
        scale_factor = float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(parts[0])

        add_offset = var_config.get("add_offset", 0.0)

        scaled_data = data * scale_factor + add_offset
        return scaled_data


##########################################


# Helpers
def _get_file_extension(output_format: str) -> str:
    """
    Get file extension based on output format.

    Parameters
    ----------
        output_format : Output file format (currently only 'netcdf' supported).

    Returns
    -------
        File extension as a string.
    """
    if output_format == "netcdf":
        return "nc"
    elif output_format == "quaver":
        return "grib"
    else:
        raise ValueError(
            f"Unsupported output format: {output_format},"
            "supported formats are ['netcdf', 'DWD', 'quaver']"
        )
