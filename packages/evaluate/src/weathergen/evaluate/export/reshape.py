import logging
import re

import numpy as np
import xarray as xr
from earthkit.regrid import interpolate
from itertools import product

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

"""
Enhanced functions to handle Gaussian grids when converting from Zarr to NetCDF.
"""


def detect_grid_type(data: xr.DataArray) -> str:
    """
    Detect whether data is on a regular lat/lon grid or Gaussian grid.

    Parameters
    ----------
    data:
        input dataset.

    Returns
    -------
    str:
        String with the grid type.
        Supported options at the moment: "unknown", "regular", "gaussian"
    """
    if "lat" not in data.coords or "lon" not in data.coords:
        return "unknown"

    lats = data.coords["lat"].values
    lons = data.coords["lon"].values

    unique_lats = np.unique(lats)
    unique_lons = np.unique(lons)

    # Check if all (lat, lon) combinations exist (regular grid)
    if len(lats) == len(unique_lats) * len(unique_lons):
        lat_lon_pairs = set(zip(lats, lons, strict=False))
        expected_pairs = {(lat, lon) for lat in unique_lats for lon in unique_lons}
        if lat_lon_pairs == expected_pairs:
            return "regular"

    # Otherwise it's Gaussian (irregular spacing or reduced grid)
    return "gaussian"


def find_pl(vars: list) -> tuple[dict[str, list[str]], list[int]]:
    """
    Find all the pressure levels for each variable using regex and returns a dictionary
    mapping variable names to their corresponding pressure levels.

    Parameters
    ----------
        vars : list of variable names with pressure levels (e.g.,'q_500','t_2m').

    Returns
    -------
        A tuple containing:
        - var_dict: dict
            Dictionary mapping variable names to lists of their corresponding pressure levels.
        - pl: list of int
            List of unique pressure levels found in the variable names.
    """
    var_dict = {}
    pl = []
    for var in vars:
        match = re.search(r"^([a-zA-Z0-9_]+)_(\d+)$", var)
        if match:
            var_name = match.group(1)
            pressure_level = int(match.group(2))
            pl.append(pressure_level)
            var_dict.setdefault(var_name, []).append(var)
        else:
            var_dict.setdefault(var, []).append(var)
    pl = sorted(set(pl))
    return var_dict, pl


class Regridder:
    """
    Class to handle regridding of xarray Datasets using earthkit regrid options available.
    """
    def __init__(self, ds, output_grid_type: str, degree: float):
        self.output_grid_type = output_grid_type
        self.degree = degree
        self.dataset = ds
        self.indices = None  # to store lat/lon ordering indices

    def find_lat_lon_ordering(self) -> list[int]:
        """
        Find all the the latitude and longitude ordering for unparsed WeatherGenerator data
        Ordering from North West to South East.
        Returns the indices required to reorder the data.

        Parameters
        ----------
            ds : Input xarray Dataset containing the inference data,

        Returns
        -------
            indices: list of indices to reorder the data from original to lat/lon ordered.
        """
        ds = self.dataset
        x = ds["longitude"].values[:, 0]
        y = ds["latitude"].values[:, 0]
        tuples = list(zip(x, y, strict=False))
        ordered_tuples = sorted(tuples, key=lambda t: (-t[1], t[0]))
        indices = [tuples.index(t) for t in ordered_tuples]
        self.indices = indices

    def detect_input_grid_type(self) -> str:
        """
        Detect whether data is on a regular lat/lon grid or Gaussian grid.
        Parameters
        ----------
            data : xr.Dataset
                input dataset.
        Returns
        -------
            str
                String with the grid type.
                Supported options at the moment: "regular", "gaussian"
        """
        data = self.dataset
        #check dataset attributes first
        if "grid_type" in data.attrs:
            return data.attrs["grid_type"]
        elif "ncells" in data.dims:
            return "gaussian"
        elif "latitude" in data.coords and "longitude" in data.coords: # skeptical- check!
            return "regular"
        else:
            raise ValueError("Unable to detect grid type from data attributes or dimensions.")

    #def define_earthkit_input(self):


    def define_earthkit_output(self):
        """
        Define the output grid type and shape based on desired output grid type and degree.
        Returns
        -------
            output_grid_type : str
                Type of grid to regrid to (e.g., 'regular_ll').
            grid_shape : list
                Shape of the output grid.
        """
        if self.output_grid_type == "regular_ll":
            degree = int(self.degree)
            earthkit_output = [degree, degree]
            grid_shape = [180 // degree + 1, 360 // degree]
            return earthkit_output, grid_shape
        else:
            earthkit_output = self.output_grid_type + str(self.degree)
            grid_shape = None
            #TODO add other grid types if needed
            return earthkit_output, grid_shape


    def regrid_to_regular_da(
        self, data: xr.DataArray
    ) -> xr.DataArray:
        """
        Regrid a single xarray Dataset to regular lat/lon grid. 
        Requires a change in number of dimensions (not just size), so handled separately.

        Parameters
        ----------
            data : Input xarray DataArray containing the inference data on native grid.
            degree : Degree of the regular lat/lon grid (e.g., 2 for 2.0 degree grid)/96 for O96 grid
            grid_shape : Shape of the output grid.
        Returns
        -------
            Regridded xarray DataArray.
        """

        # set coords
        new_coords = data.coords.copy()
        new_coords.update({
            "valid_time": data["valid_time"].values,
            "latitude": np.linspace(-90, 90, self.grid_shape[0]),
            "longitude": np.linspace(0, 360 - self.degree, self.grid_shape[1]),
        })
        print(new_coords, type(new_coords))
        print(dir(new_coords))
        print(xr._version)
        new_coords._drop_coords(["ncells"])
        print(new_coords)

        # set attrs
        attrs = data.attrs.copy()
        print(attrs)
        try:
            del attrs["ncells"]
        except KeyError:
            pass
        print(attrs)

        
        # if data.ndim == 3:
        #     values = np.empty((data.shape[0], grid_shape[0], grid_shape[1], data.shape[2]))
        #     x = 0
        # else:
        #     values = np.empty((grid_shape[0], grid_shape[1], data.shape[1]))
        #     x = 1
        # for i in range(data.shape[x]):
        #     if data.ndim == 3:
        #         for j in range(data.shape[2]):
        #             data_reg_var = interpolate(
        #                 data.values[i, :, j], {"grid": "O96"}, {"grid": output_grid_type}
        #             )
        #             values[i, :, :, j] = data_reg_var
        #     elif data.ndim == 2:
        #         data_reg_var = interpolate(
        #             data.values[:, i], {"grid": "O96"}, {"grid": output_grid_type}
        #         )
        #         values[:, :, i] = data_reg_var
        #     else:
        #         raise ValueError(
        #             f"Unsupported data dimension: {data.ndim}, supported dimensions are 2 and 3."
        #         )
            
        # find new dims and loop through extra dimensions
        original_shape = data.shape
        print(original_shape)
        new_shape = list(original_shape)
        pos = data.dims.index("ncells")
        new_shape[pos : pos + 1] = [self.grid_shape[0], self.grid_shape[1]]
        new_shape = tuple(new_shape)
        print(new_shape)

        original_index = [list(range(original_shape_i)) for original_shape_i in original_shape]
        original_index[pos] = [slice(None)]  # :placeholder
        print(original_index)

        regridded_values = np.empty(new_shape)
        result = product(*original_index)
        for item in result:
            print('chosen indices:', item)
            original_data_slice = data.values[item]
            print(original_data_slice.shape)
            print(self.earthkit_input, self.earthkit_output)
            regridded_slice = interpolate(
                original_data_slice, 
                {"grid": self.earthkit_input}, 
                {"grid": self.earthkit_output}
            )
            # sSet in regridded_values
            new_index = list(item)
            new_index[pos : pos + 1] = [slice(None), slice(None)]
            print(new_index)
            regridded_values[tuple(new_index)] = regridded_slice

        dims = list(data.dims)
        pos = dims.index("ncells")
        dims[pos : pos + 1] = ["latitude", "longitude"]
        dims = tuple(dims)

        regrid_data = xr.DataArray(
            data=regridded_values, dims=dims, coords=new_coords, attrs=attrs, name=data.name
        )

        return regrid_data

    def regrid_ds(self,
    ) -> xr.Dataset:
        """
        Regrid an xarray Dataset from Gaussian grid to regular lat/lon grid.

        Parameters
        ----------
            ds : Input xarray Dataset containing the inference data on Gaussian grid.
            output_grid_type : Type of grid to regrid to (e.g., 'regular_ll').
            degree : Degree of the regular lat/lon grid (e.g., 2 for 2.0 degree grid)/96 for O96 grid

        Returns
        -------
            Regridded xarray Dataset.
        """
        ds = self.dataset
        self.earthkit_output, self.grid_shape = self.define_earthkit_output()
        self.degree = int(self.degree)
        self.input_grid_type = self.detect_input_grid_type()
        if self.indices is None:
            self.find_lat_lon_ordering()
            _logger.info("Determined lat/lon ordering indices, saved for reuse.")

        if self.input_grid_type == "gaussian":
            # find type of Gaussian grid
            n_lats = len(set(ds["latitude"].values[:, 0]))//2
            print(n_lats)
            num_cells = len(ds["ncells"])
            if num_cells == 4 * n_lats ** 2:
                self.earthkit_input = f'N{n_lats}'
            else:
                self.earthkit_input = f'O{n_lats}'
            _logger.info(f"Detected Gaussian grid type: {self.earthkit_input}")
            # reorder everything except ncells
            original_ncells = ds["ncells"]
            ds = ds.isel(ncells=self.indices)
            ds["ncells"] = original_ncells

        regrid_vars = {}
        for var in ds.data_vars:
            if self.input_grid_type == "gaussian":
                regrid_vars[var] = self.regrid_to_regular_da(ds[var])
            else:
                raise NotImplementedError(
                    "Regridding from non-Gaussian grids is not implemented yet."
                )
        regrid_ds = xr.Dataset(regrid_vars)
        for coord in ds.coords:
            if coord not in ["latitude", "longitude"]:
                if "ncells" not in ds[coord].dims:
                    regrid_ds.coords[coord] = ds[coord]
            else:
                # preserve CF attributes
                regrid_ds.coords[coord].attrs = ds[coord].attrs
        # keep global attrs
        regrid_ds.attrs = ds.attrs
        # change grid_type
        regrid_ds.attrs["grid_type"] = self.output_grid_type
        regrid_ds.attrs["history"] += (
            f" and regridded from {self.earthkit_input} to {self.earthkit_output} using earthkit"
        )
        return regrid_ds
