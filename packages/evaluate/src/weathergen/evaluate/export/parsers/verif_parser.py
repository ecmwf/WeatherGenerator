# pylint: disable=bad-builtin

import contextlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from omegaconf import OmegaConf

from weathergen.evaluate.export.cf_utils import CfParser
from weathergen.evaluate.export.reshape import (
    find_pl,
    get_grid_points,
    get_obs_coordinates,
)
from weathergen.evaluate.export.verif_interpolator import InterpolatorFactory
from weathergen.evaluate.utils.derived_channels import compute_mslp, compute_precip

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

"""
Usage:

uv run export --run-id wgp6fowx --stream ERA5 \
--output-dir ../test_output1 \
--format verif --samples 1 2  --fsteps 1 2 3 \
--obs /p/project1/weatherai/myhre1/metno_observations_v3.nc \
--method 2d
"""


class VerifParser(CfParser):
    """
    Child class for handling NetCDF output format for MetNor Verif software.
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

        super().__init__(config=config)

        if not hasattr(self, "obs"):
            raise ValueError("Observation data required for creating verif compliant NetCDFs")

        self.mapping = config.get("variables", {})

        # add extra attributes
        self.obs = xr.open_dataset(self.obs)
        lat, lon, _ = get_obs_coordinates(self.obs)
        self.obs_coords = np.column_stack((lat.values, lon.values))
        self.zarr_coords = None
        obs_data_channels = ["10u", "10v", "sp", "2t", "msl", "tp"]
        self.channels = list(set(self.channels) & set(obs_data_channels))
        self.zarr_dt: np.timedelta64 | None = None

    def process_sample(
        self,
        fstep_iterator_results: iter,
        ref_time: np.datetime64,
        source_interval_start: np.datetime64 = None,
        source_interval_end: np.datetime64 = None,
    ):
        """
        Process results from get_data_worker: reshape, concatenate, add metadata, and save.
        Parameters
        ----------
            fstep_iterator_results : Iterator over results from get_data_worker.
            ref_time : Forecast reference time for the sample.
            source_interval_start : Start of the source (conditioning) window.
            source_interval_end : End of the source (conditioning) window.
        Returns
        -------
            None
        """
        # check ref_time exists in the obs data
        if ref_time not in self.obs.time.values:
            _logger.warning(
                f"Reference time {ref_time} not found in observation data. Skipping sample."
            )
            return

        da_fs = []
        for result in fstep_iterator_results:
            if result is None:
                continue
            # result is already a materialized xarray DataArray (built in the worker).
            if not isinstance(result, xr.DataArray):
                result = result.as_xarray().squeeze()
            result = result.sel(channel=self.channels)
            result = self.preprocess(result)
            result = self.reshape(result)
            da_fs.append(result)

        _logger.info(f"Retrieved {len(da_fs)} forecast steps for type {self.data_type}.")

        if da_fs:
            if self.zarr_coords is None:
                self.zarr_coords = get_grid_points(da_fs[0])
                self.zarr_dt = self.get_zarr_dt(source_interval_start, source_interval_end)
            # check consistency of grid points across forecast steps
            if len(da_fs) > 1:
                assert np.array_equal(get_grid_points(da_fs[1]), get_grid_points(da_fs[0])), (
                    "Grid points between forecast steps are not consistent."
                    "Check that inference was not performed with masking"
                )
            da_fs = self.concatenate(da_fs)
            da_fs = self.assign_frt(da_fs, ref_time)
            da_fs = self.add_attrs(da_fs)
            vars_to_merge = {verif_var: None for verif_var in self.mapping.keys()}

            for verif_var in self.mapping.keys():
                da_var = self.regrid(da_fs, verif_var)
                if da_var is None:
                    continue
                da_var = self.add_encoding(da_var)
                obs_result = self.obs_preprocess(da_var, verif_var)
                obs_result = self.add_encoding(obs_result)
                merged = self.merge(da_var, obs_result)
                merged = self.add_metadata(merged, verif_var)
                vars_to_merge[verif_var] = merged
        return vars_to_merge

    def get_zarr_dt(
        self,
        source_interval_start: np.datetime64,
        source_interval_end: np.datetime64,
    ) -> np.timedelta64:
        """
        Compute the time difference between source interval start and end in hours.
        Parameters
        ----------
            source_interval_start : np.datetime64
                Start of the source (conditioning) window.
            source_interval_end : np.datetime64
                End of the source (conditioning) window.
        Returns
        -------
            np.timedelta64
                Time difference between source interval start and end in hours.
        """
        zarr_dt = (source_interval_end - source_interval_start).astype("timedelta64[h]")

        return zarr_dt

    def get_output_filename(self, variable: str) -> Path:
        """
        Create output directories for the verif files
        and return path to output file
        Args:
            variables (list[string])
            outfiles (string): template for the output files
        Outputs:
            None
        """
        outfile = Path(
            self.verif_template.replace("%S", self.stream)
            .replace("%V", variable)
            .replace("%M", self.method)
            .replace("%D", self.data_type)
            .replace("%R", self.run_id)
        )
        outfile = Path(self.output_dir) / outfile
        pathdir = outfile.parent
        _logger.info(f"Output directory: {pathdir}")
        pathdir.mkdir(exist_ok=True, parents=True)
        return outfile

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
        var_dict = find_pl(data.channel.values)
        data_vars = {}

        for new_var, pls in var_dict.items():
            if pls[0] is not None:
                old_vars = [f"{new_var}_{p}" for p in pls]
                data_vars[new_var] = xr.DataArray(
                    data.sel(channel=old_vars).values,
                    dims=["ipoint", "pressure_level"],
                    coords={"pressure_level": pls},
                )
            else:
                data_vars[new_var] = xr.DataArray(
                    data.sel(channel=new_var).values,
                    dims=["ipoint"],
                )

        reshaped_dataset = xr.Dataset(data_vars)
        reshaped_dataset = reshaped_dataset.assign_coords(
            ipoint=data.coords["ipoint"],
        )

        # order using pressure_level coord
        if "pressure_level" in reshaped_dataset.coords:
            reshaped_dataset = reshaped_dataset.sortby("pressure_level")

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

    def obs_preprocess(self, ds_var, verif_var: str) -> xr.DataArray:
        """
        Preprocess the observation data for the given variable and valid times.
        This includes computing derived variables like MSLP and total precipitation if needed.

        Parameters
        ----------
            obs_data : xr.Dataset
                The original observation dataset.
            ds_var : xr.DataArray
                The forecast data array to which the observation data should be regridded.

        Returns
        -------
            xr.DataArray
                Regridded observation data matching the forecast grid.
        """
        obs_data = self.obs
        mapped_info = self.mapping.get(verif_var, {})
        obs_name = mapped_info.get("obs_name", {})

        original_shape = ds_var.shape
        new_shape = list(original_shape)

        obs_dataarray = np.empty(new_shape, dtype=np.float32)

        for i, leadtime in enumerate(ds_var.coords["leadtime"].values):
            valid_time = ds_var.coords["time"] + np.timedelta64(int(leadtime), "h")
            if verif_var == "mslp":
                obs_dataarray[:, i, :] = compute_mslp(obs_data, valid_time)
            if verif_var == "tp":
                obs_dataarray[:, i, :] = compute_precip(obs_data, self.zarr_dt, valid_time)
            else:
                obs_dataarray[:, i, :] = obs_data.data_vars[obs_name].sel(time=valid_time)

        obs_dataarray = ds_var.copy(data=obs_dataarray)
        obs_dataarray.name = "obs"

        return obs_dataarray

    def preprocess(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Preprocess variables and only keep relevant ones for WG output
        Parameters
        ----------
            ds : xr.Dataset


        Returns
        -------
            xr.Dataset
        """
        if set(["10u", "10v"]).issubset(self.channels):
            u = ds.sel(channel="10u")
            v = ds.sel(channel="10v")
            # hypotenuese
            wind_speed = xr.apply_ufunc(
                np.hypot, u, v, dask="parallelized", output_dtypes=[ds.dtype]
            ).astype("float32")
            wind_speed = wind_speed.expand_dims(channel=["10si"])
            if ds.chunks:
                wind_speed = wind_speed.chunk(
                    {"ipoint": ds.chunks[ds.get_axis_num("ipoint")][0], "channel": 1}
                )
            new_ds = xr.concat([ds, wind_speed], dim="channel")
            new_ds.attrs = ds.attrs

            # remove unnecessary
            new_ds = new_ds.drop_sel(channel=["10u", "10v"])
            return new_ds
        else:
            return ds

    def regrid(self, ds: xr.Dataset, verif_var: str) -> xr.Dataset:
        """
        Regrid a single xarray Dataset using specific method.
        Parameters
        ----------
            ds: native xarray Dataset
        Returns
        -------
            Regridded xarray Dataset.
        """
        mapped_info = self.mapping.get(verif_var, {})
        wg_var = mapped_info.get("var", None)
        try:
            ds_var = ds[wg_var]
        except KeyError as e:
            _logger.info(f"{wg_var} not available in WeatherGenerator output: {e}")
            return
        # set coords
        # TODO: tidy this up
        new_coords = {
            "time": (["time"], np.atleast_1d(ds_var.coords["time"].values), ds_var["time"].attrs),
            "location": (
                ["location"],
                self.obs.location.values,
                {"long_name": "Norwegian station ID"},
            ),
            "leadtime": (
                ["leadtime"],
                np.atleast_1d(ds_var.coords["leadtime"].values.astype("float32")),
                ds_var["leadtime"].attrs,
            ),
        }
        # set variable attrs
        attrs = ds_var.attrs.copy()
        with contextlib.suppress(KeyError):
            del attrs["ncells"]  #

        original_shape = ds_var.shape
        new_shape = list(original_shape)
        pos = ds_var.dims.index("ncells")
        new_shape[pos] = self.obs.location.shape[0]
        # rearrange to be time,location
        order = [1, 0]
        new_shape = [new_shape[x] for x in order]

        fcstdata = np.empty(new_shape, dtype=np.float32)

        # set interpolation method
        method_factory = InterpolatorFactory(self.method)
        interpolator = method_factory.get_interpolator(self.zarr_coords, self.obs_coords)

        num_leadtimes = np.atleast_1d(ds_var.coords["leadtime"].values).shape[0]

        for idx in range(num_leadtimes):
            regrid_values = interpolator.interpolate(ds_var.values[:, idx])
            fcstdata[idx, :] = regrid_values

        regridded_var = xr.DataArray(
            np.array([fcstdata]),
            dims=["time", "leadtime", "location"],
            coords={**new_coords},
            name="fcst",
            attrs=attrs,
        )
        return regridded_var

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

    def assign_frt(self, ds: xr.Dataset, reference_time: np.datetime64) -> xr.Dataset:
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
        ds = ds.assign_coords(forecast_reference_time=reference_time)

        if "sample" in ds.coords:
            ds = ds.drop_vars("sample")
        n_hours = self.fstep_hours.astype("int64")
        ds["forecast_step"] = ds["forecast_step"] * n_hours
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
        variables = self._attrs_gaussian_grid(ds)
        dataset = xr.merge(variables.values(), compat="no_conflicts")
        return dataset

    def add_encoding(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add time encoding to the dataset variables.
        Add aux coordinates to leadtime

        Parameters
        ----------
            ds : xarray Dataset to add time encoding to.
        Returns
        -------
            xarray Dataset with time encoding added.
        """
        time_encoding = {
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }

        if "time" in ds.coords:
            ds["time"].encoding.update(time_encoding)

        if "forecast_reference_time" in ds.coords:
            ds["forecast_reference_time"].encoding.update(time_encoding)

        if "leadtime" in ds.coords:
            ds["leadtime"].encoding.update({"coordinates": "forecast_reference_time"})

        return ds

    def add_metadata(self, ds: xr.Dataset, verif_var) -> xr.Dataset:
        """
        Add CF conventions to the dataset attributes.

        Parameters
        ----------
            ds : Input xarray Dataset to add conventions to.
        Returns
        -------
            xarray Dataset with CF conventions added to attributes.
        """
        ds.attrs["title"] = (
            f"WeatherGenerator Output for {self.run_id}, variable {verif_var} "
            f"using stream {self.stream}"
        )
        ds.attrs["institution"] = "WeatherGenerator Collaboration"
        ds.attrs["source"] = "WeatherGenerator v0.0"
        ds.attrs["history"] = "Created using the verif_parser on " + np.datetime_as_string(
            np.datetime64("now"), unit="s"
        )
        ds.attrs["conventions"] = "verif_1.0.0"

        return ds

    def _attrs_gaussian_grid(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Assign CF-compliant attributes to variables in a gaussian grid dataset.
        Parameters
        ----------
            ds : xr.Dataset
                Input dataset.
        Returns
        -------
            xr.Dataset
                Dataset with CF-compliant variable attributes.
        """
        unit_conversion = {"kg/m^2": 1.0, "Pa": 1.0, "K": 1.0, "m/s": 1.0, "m": 1000.0}

        variables = {}
        dims_cfg = self.config.get("dimensions", {})
        ds, ds_attrs = self._assign_dim_attrs(ds, dims_cfg)
        for var_name, da in ds.data_vars.items():
            mapped_info = self.mapping.get(var_name, {})
            mapped_name = mapped_info.get("var", var_name)
            mapped_units = mapped_info.get("wg_unit", {})

            coords = self._build_coordinate_mapping(ds, mapped_info, ds_attrs)

            wg_unit = mapped_units.get(self.stream, "DEFAULT")
            verif_unit = mapped_info.get("verif_unit", None)
            if wg_unit != verif_unit:
                # perform unit conversion
                da.values = da.values * unit_conversion[wg_unit]

            attributes = {
                "units": verif_unit,
            }

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

    def _assign_dim_attrs(
        self, ds: xr.Dataset, dim_cfg: dict[str, Any]
    ) -> tuple[xr.Dataset, dict[str, dict[str, str]]]:
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
            xr.Dataset:
                Dataset with renamed dimensions.
        """
        ds_attrs = {}

        for dim_name, meta in dim_cfg.items():
            verif_name = meta.get("verif", dim_name)
            if dim_name in ds.dims and dim_name != verif_name:
                ds = ds.rename_dims({dim_name: verif_name})

            dim_attrs = {"standard_name": meta.get("std", verif_name)}
            if meta.get("verif_unit"):
                dim_attrs["units"] = meta["verif_unit"]
            if meta.get("long"):
                dim_attrs["long_name"] = meta["long"]
            ds_attrs[verif_name] = dim_attrs
        return ds, ds_attrs

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

    def merge(self, ds, obs_ds):
        lat, lon, alt = get_obs_coordinates(self.obs)
        merged = xr.merge([ds, obs_ds, lat, lon, alt], compat="minimal")
        # may need join=inner if some leadtimes missing in obs
        return merged

    def save(self, list_samples: list) -> None:
        """
        Save the dataset to a NetCDF file.

        Parameters
        ----------
            list_samples : list of dictionary containing variables to merge and save.
             Each dictionary corresponds to a sample and contains variables for that sample.

        Returns
        -------
            None
        """
        for verif_var in self.mapping.keys():
            var_list = [sample[verif_var] for sample in list_samples if verif_var in sample]
            if all(v is None for v in var_list):
                _logger.warning(f"No data to save for variable {verif_var}. Skipping.")
                continue
            ds = xr.concat(
                var_list, dim="time", data_vars="minimal", coords="minimal", join="exact"
            )
            out_fname = self.get_output_filename(verif_var)
            _logger.info(f"Saving to {out_fname}.")
            ds.to_netcdf(out_fname)
            _logger.info(f"Saved NetCDF file to {out_fname}.")
            _logger.info(f"Saved {verif_var} data to {self.output_format} in {self.output_dir}.")
