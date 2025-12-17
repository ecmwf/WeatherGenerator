from omegaconf import OmegaConf

from weathergen.evaluate.export.cf_utils import CfParser
from weathergen.evaluate.export.parsers.netcdf_parser import NetcdfParser
from weathergen.evaluate.export.parsers.quaver_parser import QuaverParser


class CfParserFactory:
    """
    Factory class to get appropriate CF parser based on output format.
    """

    @staticmethod
    def get_parser(config: OmegaConf, **kwargs) -> CfParser:
        """
        Get the appropriate CF parser based on the output format.

        Parameters
        ----------
            config : OmegaConf
                Configuration defining variable mappings and dimension metadata.
            grid_type : str
                Type of grid ('regular' or 'gaussian').

        Returns
        -------
            Instance of a CF_Parser subclass.
        """

        _parser_map = {
            "netcdf": (NetcdfParser, ["grid_type"]),
            "quaver": (QuaverParser, ["grid_type", "channels", "template"]),
        }

        fmt = kwargs.get("output_format")

        parser_class = _parser_map.get(fmt)
        parser = parser_class[0]

        # allowed_keys = parser_class[1]
        # filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

        if parser_class is None:
            raise ValueError(f"Unsupported format: {fmt}")

        return parser(config, **kwargs)
