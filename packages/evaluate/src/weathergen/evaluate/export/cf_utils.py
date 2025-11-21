import logging
from pathlib import Path

import numpy as np

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