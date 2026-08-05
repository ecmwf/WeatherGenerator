# pylint: disable=bad-builtin

import logging
import xarray as xr
import numpy as np
from cfgrib.xarray_to_grib import to_grib
from weathergen.evaluate.export.parsers.netcdf_parser import NetcdfParser

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class GribParser(NetcdfParser):
    """
    Child class for handling GRIB output format.
    Important to note it must be used with regridding to regular_ll
    """

    def gribify(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Convert dataset to use GRIB data variable names.
        """
        # change pressure to isobaricInhPa for GRIB compliance
        if "pressure" in ds.coords:
            ds = ds.rename({"pressure": "isobaricInhPa"})
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        if "forecast_period" in ds.coords:
            ds = ds.rename({"forecast_period": "step"})
        return ds

    def _attrs_gaussian_grid(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Assign CF-compliant attributes to variables in a Gaussian grid dataset.
        Parameters
        ----------
            ds : xr.Dataset
                Input dataset.
        Returns
        -------
            xr.Dataset
                Dataset with CF-compliant variable attributes.
        """
        variables = {}
        dims_cfg = self.config.get("dimensions", {})
        ds, ds_attrs = self._assign_dim_attrs(ds, dims_cfg)
        for var_name, da in ds.data_vars.items():
            mapped_info = self.mapping.get(var_name, {})
            mapped_name = mapped_info.get("var", var_name)

            coords = self._build_coordinate_mapping(ds, mapped_info, ds_attrs)

            # parse grib specifc short names
            grib_shortnames = {
                "t2m": "2t",
                "u10": "10u",
                "v10": "10v",
                "d2m": "2d",
            }
            grib_levels_special = {"t2m": 2, "u10": 10, "v10": 10, "d2m": 2}

            attributes = {
                "GRIB_shortName": grib_shortnames.get(mapped_name, mapped_name),  # if GRIB
                "standard_name": mapped_info.get("std", var_name),
                "units": mapped_info.get("std_unit", "unknown"),
            }
            if mapped_name in grib_shortnames.keys():
                attributes.update(
                    {
                        "GRIB_typeOfLevel": "heightAboveGround",
                        "GRIB_level": grib_levels_special[mapped_name],
                    }
                )
            if "long" in mapped_info:
                attributes["long_name"] = mapped_info["long"]
            variables[mapped_name] = xr.DataArray(
                data=da.values,
                dims=da.dims,
                coords=coords,
                attrs=attributes,
                name=mapped_name,
            )

        return variables

    def save(self, ds: xr.Dataset, forecast_ref_time: np.datetime64) -> None:
        """
        Save the dataset to a GRIB file.

        Parameters
        ----------
            ds : xarray Dataset to save.
            data_type : Type of data ('pred' or 'targ') to include in the filename.
            forecast_ref_time : Forecast reference time to include in the filename.

        Returns
        -------
            None
        """
        out_fname = self.get_output_filename(forecast_ref_time)
        _logger.info(f"Saving to {out_fname}.")
        ds = self.gribify(ds)
        to_grib(ds, out_fname)
        _logger.info(f"Saved GRIB file to {out_fname}.")
