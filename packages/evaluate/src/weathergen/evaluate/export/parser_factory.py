from omegaconf import OmegaConf
from weathergen.evaluate.export.cf_utils import CF_Parser
from weathergen.evaluate.export.parsers.netcdf_parser import NetCDF_Parser
from weathergen.evaluate.export.parsers.quaver_parser import Quaver_Parser


class CF_ParserFactory(object):
    """
    Factory class to get appropriate CF parser based on output format.
    """

    @staticmethod
    def get_parser(
        config: OmegaConf,
        **kwargs
    ) -> CF_Parser:
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
            "netcdf": (NetCDF_Parser, ["grid_type"]),
            "quaver": (Quaver_Parser, ["grid_type", "channels", "template"])
            }

        fmt = kwargs.get("output_format")
        
        parser_class = _parser_map.get(fmt)
        parser = parser_class[0]
        
        # allowed_keys = parser_class[1]
        # filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        
        if parser_class is None:
            raise ValueError(f"Unsupported format: {fmt}")
        
        return parser(config, **kwargs)