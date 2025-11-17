import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from omegaconf import OmegaConf

from weathergen.evaluate.export.cf_utils import CfParser
from weathergen.evaluate.export.reshape import find_pl

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

"""
Usage:

uv run export --run-id ciga1p9c --stream ERA5 
--output-dir ./test_output1 
--format netcdf --samples 1 2  --fsteps 1 2 3
"""


class NetcdfParser(CfParser):
    """
    Child class for handling NetCDF output format.
    """

    def __init__(self, config: OmegaConf, **kwargs):
        """
        CF-compliant parser that handles both regular and Gaussian grids.

        Parameters
        ----------
        config : OmegaConf
            Configuration defining variable mappings and dimension metadata.
        ds : xr.Dataset
            Input dataset.

        Returns
        -------
        xr.Dataset
            CF-compliant dataset with consistent naming and attributes.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        super().__init__(config=config, grid_type=self.grid_type)

        self.mapping = config.get("variables", {})

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
        da_fs = []

        for result in fstep_iterator_results:
            if result is None:
                continue

            result = result.as_xarray().squeeze()
            result = result.sel(channel=self.channels)
            result = self.reshape(result)
            da_fs.append(result)

        _logger.info(f"Retrieved {len(da_fs)} forecast steps for type {self.data_type}.")
        _logger.info(f"Saved sample data to {self.output_format} in {self.output_dir}.")

        if da_fs:
            da_fs = self.concatenate(da_fs)
            da_fs = self.assign_coords(da_fs, ref_time)
            da_fs = self.add_attrs(da_fs)
            da_fs = self.add_metadata(da_fs)
            self.save(da_fs, ref_time)

    def get_output_filename(self, forecast_ref_time: np.datetime64) -> Path:
        """
        Generate output filename based on prefix (should refer to type e.g. pred/targ),
        run_id, sample index, output directory, format and forecast_ref_time.

        Parameters
        ----------
            forecast_ref_time : Forecast reference time to include in the filename.

        Returns
        -------
            Full path to the output file.
        """

        frt = np.datetime_as_string(forecast_ref_time, unit="h")
        out_fname = (
            Path(self.output_dir) / f"{self.data_type}_{frt}_{self.run_id}.{self.file_extension}"
        )
        return out_fname

    def reshape(self, data: xr.DataArray) -> xr.Dataset:
        """
        Reshape dataset while preserving grid structure (regular or Gaussian).

        Parameters
        ----------
        data : xr.DataArray
            Input data with dimensions (ipoint, channel)

        Returns
        -------
        xr.Dataset
            Reshaped dataset appropriate for the grid type
        """
        grid_type = self.grid_type

        # Original logic
        var_dict, pl = find_pl(data.channel.values)
        data_vars = {}

        for new_var, old_vars in var_dict.items():
            if len(old_vars) > 1:
                data_vars[new_var] = xr.DataArray(
                    data.sel(channel=old_vars).values,
                    dims=["ipoint", "pressure_level"],
                )
            else:
                data_vars[new_var] = xr.DataArray(
                    data.sel(channel=old_vars[0]).values,
                    dims=["ipoint"],
                )

        reshaped_dataset = xr.Dataset(data_vars)
        reshaped_dataset = reshaped_dataset.assign_coords(
            ipoint=data.coords["ipoint"],
            pressure_level=pl,
        )

        if grid_type == "regular":
            # Use original reshape logic for regular grids
            # This is safe for regular grids
            reshaped_dataset = reshaped_dataset.set_index(
                ipoint=("valid_time", "lat", "lon")
            ).unstack("ipoint")
        else:
            # Use new logic for Gaussian/unstructured grids
            reshaped_dataset = reshaped_dataset.set_index(ipoint2=("ipoint", "valid_time")).unstack(
                "ipoint2"
            )
            # rename ipoint to ncells
            reshaped_dataset = reshaped_dataset.rename_dims({"ipoint": "ncells"})
            reshaped_dataset = reshaped_dataset.rename_vars({"ipoint": "ncells"})

        return reshaped_dataset

    def concatenate(
        self,
        array_list,
        dim="valid_time",
        data_vars="minimal",
        coords="different",
        compat="equals",
        combine_attrs="drop",
        sortby_dim="valid_time",
    ) -> xr.Dataset:
        """
        Uses list of pred/target xarray DataArrays to save one sample to a NetCDF file.

        Parameters
        ----------
        type_str : str
            Type of data ('pred' or 'targ') to include in the filename.
        array_list : list of xr.DataArray
            List of DataArrays to concatenate.
        dim : str, optional
            Dimension along which to concatenate. Default is 'valid_time'.
        data_vars : str, optional
            How to handle data variables during concatenation. Default is 'minimal'.
        coords : str, optional
            How to handle coordinates during concatenation. Default is 'different'.
        compat : str, optional
            Compatibility check for variables. Default is 'equals'.
        combine_attrs : str, optional
            How to combine attributes. Default is 'drop'.
        sortby_dim : str, optional
            Dimension to sort the final dataset by. Default is 'valid_time'.

        Returns
        -------
        xr.Dataset
            Concatenated xarray Dataset.
        """

        data = xr.concat(
            array_list,
            dim=dim,
            data_vars=data_vars,
            coords=coords,
            compat=compat,
            combine_attrs=combine_attrs,
        ).sortby(sortby_dim)

        return data

    def assign_coords(self, ds: xr.Dataset, reference_time: np.datetime64) -> xr.Dataset:
        """
        Assign forecast reference time coordinate to the dataset.

        Parameters
        ----------
            ds : xarray Dataset to assign coordinates to.
            reference_time : Forecast reference time to assign.

        Returns
        -------
            xarray Dataset with assigned forecast reference time coordinate.
        """
        ds = ds.assign_coords(forecast_ref_time=reference_time)

        if "sample" in ds.coords:
            ds = ds.drop_vars("sample")

        n_hours = self.fstep_hours.astype("int64")
        ds["forecast_period"] = ds["forecast_step"] * n_hours

        return ds

    def add_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add CF-compliant attributes to the dataset variables.

        Parameters
        ----------
            ds : xarray Dataset to add attributes to.
        Returns
        -------
            xarray Dataset with CF-compliant variable attributes.
        """

        ds["forecast_period"].attrs = {
            "standard_name": "forecast_period",
            "long_name": "time since forecast_reference_time",
            "units": "hours",
        }

        if self.grid_type == "gaussian":
            variables = self._attrs_gaussian_grid(ds)
        else:
            variables = self._attrs_regular_grid(ds)

        dataset = xr.merge(variables.values())
        dataset.attrs = ds.attrs

        return dataset

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

        for var_name, da in ds.data_vars.items():
            if var_name in ["lat", "lon"]:
                continue

            mapped_info = self.mapping.get(var_name, {})
            mapped_name = mapped_info.get("var", var_name)

            attributes = {
                "standard_name": mapped_info.get("std", var_name),
                "units": mapped_info.get("std_unit", "unknown"),
                "coordinates": "lat lon",
            }

            variables[mapped_name] = xr.DataArray(
                data=da.values,
                dims=list(da.dims),
                coords={coord: ds.coords[coord] for coord in da.coords if coord in ds.coords},
                attrs=attributes,
                name=mapped_name,
            )

        self._assign_latlon_attrs(ds)

        return variables

    def _attrs_regular_grid(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Assign CF-compliant attributes to variables in a regular grid dataset.
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
        dims = self.config.get("dimensions", {})
        ds_attrs = self._assign_dim_attrs(ds, dims)
        mapping = self.mapping

        for var_name, da in ds.data_vars.items():
            var_cfg = mapping.get(var_name)
            if var_cfg is None:
                continue

            dims = ["pressure", "valid_time", "latitude", "longitude"]
            if var_cfg.get("level_type") == "sfc":
                dims.remove("pressure")

            coords = self._build_coordinate_mapping(ds, var_cfg, ds_attrs)

            attrs = {
                "standard_name": var_cfg.get("std", var_name),
                "units": var_cfg.get("std_unit", "unknown"),
            }

            mapped_name = var_cfg.get("var", var_name)
            variables[mapped_name] = xr.DataArray(
                data=da.values,
                dims=dims,
                coords={**coords, "valid_time": ds["valid_time"].values},
                attrs=attrs,
                name=mapped_name,
            )

        return variables

    def _assign_latlon_attrs(self, ds: xr.Dataset) -> None:
        """Add CF-compliant attributes to lat/lon coordinates if they exist.
        Parameters
        ----------
            ds : xr.Dataset
                Input dataset.
        Returns
        -------
            None
        """
        if "lat" in ds.coords:
            ds.coords["lat"].attrs.update(
                {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                }
            )
        if "lon" in ds.coords:
            ds.coords["lon"].attrs.update(
                {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                }
            )

    def _assign_dim_attrs(
        self, ds: xr.Dataset, dim_cfg: dict[str, Any]
    ) -> dict[str, dict[str, str]]:
        """
        Assign CF attributes from given config file.
        Parameters
        ----------
            ds : xr.Dataset
                Input dataset.
            dim_cfg : Dict[str, Any]
                Dimension configuration from mapping.
        Returns
        -------
            Dict[str, Dict[str, str]]:
                Attributes for each dimension.
        """
        ds_attrs = {}

        for dim_name, meta in dim_cfg.items():
            wg_name = meta.get("wg", dim_name)
            if dim_name in ds.dims and dim_name != wg_name:
                ds = ds.rename_dims({dim_name: wg_name})

            dim_attrs = {"standard_name": meta.get("std", wg_name)}
            if meta.get("std_unit"):
                dim_attrs["units"] = meta["std_unit"]
            ds_attrs[wg_name] = dim_attrs

        return ds_attrs

    def _build_coordinate_mapping(
        self, ds: xr.Dataset, var_cfg: dict[str, Any], attrs: dict[str, dict[str, str]]
    ) -> dict[str, Any]:
        """Create coordinate mapping for a given variable.
        Parameters
        ----------
            ds : xr.Dataset
                Input dataset.
            var_cfg : Dict[str, Any]
                Variable configuration from mapping.
            attrs : Dict[str, Dict[str, str]]
                Attributes for dimensions.
        Returns
        -------
            Dict[str, Any]:
                Coordinate mapping for the variable.
        """
        coords = {}
        coord_map = self.config.get("coordinates", {}).get(var_cfg.get("level_type"), {})

        for coord, new_name in coord_map.items():
            coords[new_name] = (
                ds.coords[coord].dims,
                ds.coords[coord].values,
                attrs[new_name],
            )

        return coords

    def _add_grid_attrs(self, ds: xr.Dataset, grid_info: dict | None = None) -> xr.Dataset:
        """
        Add Gaussian grid metadata following CF conventions.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to add metadata to
        grid_info : dict, optional
            Dictionary with grid information:
            - 'N': Gaussian grid number (e.g., N320)
            - 'reduced': Whether it's a reduced Gaussian grid

        Returns
        -------
        xr.Dataset
            Dataset with added grid metadata
        """

        if self.grid_type != "gaussian":
            return ds

        # ds = ds.copy()
        # Add grid mapping information
        ds.attrs["grid_type"] = "gaussian"

        # If grid info provided, add it
        if grid_info:
            ds.attrs["gaussian_grid_number"] = grid_info.get("N", "unknown")
            ds.attrs["gaussian_grid_type"] = (
                "reduced" if grid_info.get("reduced", False) else "regular"
            )

        return ds

    def add_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add CF conventions to the dataset attributes.

        Parameters
        ----------
            ds : Input xarray Dataset to add conventions to.
        Returns
        -------
            xarray Dataset with CF conventions added to attributes.
        """
        # ds = ds.copy()
        ds.attrs["title"] = f"WeatherGenerator Output for {self.run_id} using stream {self.stream}"
        ds.attrs["institution"] = "WeatherGenerator Project"
        ds.attrs["source"] = "WeatherGenerator v0.0"
        ds.attrs["history"] = (
            "Created using the export_inference.py script on "
            + np.datetime_as_string(np.datetime64("now"), unit="s")
        )
        ds.attrs["Conventions"] = "CF-1.12"
        return ds

    def save(self, ds: xr.Dataset, forecast_ref_time: np.datetime64) -> None:
        """
        Save the dataset to a NetCDF file.

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
        ds.to_netcdf(out_fname)
        _logger.info(f"Saved NetCDF file to {out_fname}.")
