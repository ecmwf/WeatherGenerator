import contextlib
import logging
import re
from itertools import product

import numpy as np
import xarray as xr
from earthkit.regrid import interpolate

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
        self.indices = self.find_lat_lon_ordering()  # to store lat/lon ordering indices

        self.earthkit_input: str = ""
        self.earthkit_output: str = ""
        self.grid_shape: tuple[int] = []
        self.input_grid_type: str = ""

    def find_lat_lon_ordering(self) -> list[int]:
        """
        Find all the the latitude and longitude ordering for CF-parsed WeatherGenerator data
        Ordering from North West to South East.
        Returns the indices required to reorder the data.
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
        return indices

    def detect_input_grid_type(self) -> str:
        """
        Detect whether data is on a regular lat/lon grid or Gaussian grid.
        Returns
        -------
            str
                String with the grid type.
                Supported options at the moment: "regular", "gaussian"
        """
        data = self.dataset
        # check dataset attributes first
        if "grid_type" in data.attrs:
            return data.attrs["grid_type"]
        elif "ncells" in data.dims:
            return "gaussian"
        elif "latitude" in data.coords and "longitude" in data.coords:  # skeptical- check!
            return "regular_ll"
        else:
            raise ValueError("Unable to detect grid type from data attributes or dimensions.")

    def define_earthkit_input(self):
        """
        Define the input grid type for earthkit regrid based on detected input grid type."""
        ds = self.dataset
        if self.input_grid_type == "gaussian":
            # fix all other indices except ncells
            lat_ds_dims = len(ds["latitude"].shape)
            pos = ds["latitude"].dims.index("ncells")
            selected_indices = np.zeros(lat_ds_dims, dtype=int).tolist()
            selected_indices[pos] = slice(None)
            lat_ds = ds["latitude"].values[tuple(selected_indices)]

            # find type of Gaussian grid
            n_lats = len(set(lat_ds)) // 2  ## UNEXPECTED LOGIC
            num_cells = len(ds["ncells"])
            if num_cells == 4 * n_lats**2:
                return f"N{n_lats}"
            else:
                return f"O{n_lats}"
            _logger.info(f"Detected Gaussian grid type: {self.earthkit_input}")
        if self.input_grid_type == "regular_ll":
            ## Needs to be tested properly when there are regular grids
            _logger.warning("Regular lat/lon grid input detection not fully tested yet.")
            n_lats = len(ds["latitude"].shape)
            degree = int(180 / (n_lats - 1))
            return [degree, degree]

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
            earthkit_output = [self.degree, self.degree]
            grid_shape = [int(180 // self.degree + 1), int(360 // self.degree)]
            return earthkit_output, grid_shape
        elif self.output_grid_type in ["N", "O"]:
            earthkit_output = self.output_grid_type + str(int(self.degree))
            grid_shape = self.find_num_cells()
            return earthkit_output, grid_shape
        else:
            raise ValueError(f"Unsupported output grid type: {self.output_grid_type}")
        # TODO add other grid types if needed

    def gaussian_regular_da(self, data: xr.DataArray) -> xr.DataArray:
        """
        Regrid a single xarray Dataset to regular lat/lon grid.
        Requires a change in number of dimensions (not just size), so handled separately.

        Parameters
        ----------
            data : Input xarray DataArray containing the inference data on native grid.
        Returns
        -------
            Regridded xarray DataArray.
        """

        # set coords
        new_coords = data.coords.copy()
        new_coords.update(
            {
                "valid_time": data["valid_time"].values,
                "latitude": np.linspace(-90, 90, self.grid_shape[0]),
                "longitude": np.linspace(0, 360 - self.degree, self.grid_shape[1]),
            }
        )
        new_coords._drop_coords(["ncells"])

        # set attrs
        attrs = data.attrs.copy()
        with contextlib.suppress(KeyError):
            del attrs["ncells"]

        # find new dims and loop through extra dimensions
        original_shape = data.shape
        new_shape = list(original_shape)
        pos = data.dims.index("ncells")
        new_shape[pos : pos + 1] = [self.grid_shape[0], self.grid_shape[1]]
        new_shape = tuple(new_shape)

        original_index = [list(range(original_shape_i)) for original_shape_i in original_shape]
        original_index[pos] = [slice(None)]  # :placeholder

        regridded_values = np.empty(new_shape)
        result = product(*original_index)
        for item in result:
            original_data_slice = data.values[item]
            regridded_slice = interpolate(
                original_data_slice, {"grid": self.earthkit_input}, {"grid": self.earthkit_output}
            )
            # sSet in regridded_values
            new_index = list(item)
            new_index[pos : pos + 1] = [slice(None), slice(None)]
            regridded_values[tuple(new_index)] = regridded_slice

        dims = list(data.dims)
        pos = dims.index("ncells")
        dims[pos : pos + 1] = ["latitude", "longitude"]
        dims = tuple(dims)

        regrid_data = xr.DataArray(
            data=regridded_values, dims=dims, coords=new_coords, attrs=attrs, name=data.name
        )

        return regrid_data

    def regular_gaussian_da(self, data: xr.DataArray) -> xr.DataArray:
        """
        Regrid a single xarray Dataset to Gaussian grid.
        Requires a change in number of dimensions (not just size), so handled separately.

        Parameters
        ----------
            data : Input xarray DataArray containing the inference data on native grid.
        Returns
        -------
            Regridded xarray DataArray.
        """
        raise NotImplementedError(
            "Regridding from regular lat/lon grids to Gaussian grids is not implemented yet."
        )

        # set coords
        new_coords = data.coords.copy()
        new_coords.update(
            {
                "ncells": np.arange(self.find_num_cells()),
                # "valid_time": data["valid_time"].values,
            }
        )
        ####THIS IS GOING TO BE COMPLICATED AS LAT LON SHOULD BE DEFINED BY NCELLS####
        # set attrs
        attrs = data.attrs.copy()

        # find lat, lon position
        original_shape = data.shape
        new_shape = list(original_shape)
        lat_pos = data.dims.index("latitude")
        lon_pos = data.dims.index("longitude")
        ####COULD BE RISKY IF LAT/LON NOT NEXT TO EACH OTHER####
        new_shape[lat_pos : lon_pos + 1] = [self.find_num_cells()]
        new_shape = tuple(new_shape)
        # find indices
        original_index = [list(range(original_shape_i)) for original_shape_i in original_shape]
        original_index[lat_pos, lon_pos] = [slice(None), slice(None)]  # :placeholder

        regridded_values = np.empty(new_shape)
        result = product(*original_index)
        for item in result:
            original_data_slice = data.values[item]
            regridded_slice = interpolate(
                original_data_slice, {"grid": self.earthkit_input}, {"grid": self.earthkit_output}
            )
            # sSet in regridded_values
            new_index = list(item)
            new_index[lat_pos] = slice(None)
            new_index[lon_pos] = slice(None)
            regridded_values[tuple(new_index)] = regridded_slice

        dims = list(data.dims)
        dims[lat_pos : lon_pos + 1] = ["ncells"]
        dims = tuple(dims)

        regrid_data = xr.DataArray(
            data=regridded_values, dims=dims, coords=new_coords, attrs=attrs, name=data.name
        )

        return regrid_data

    def regular_regular_da(self, data: xr.DataArray) -> xr.DataArray:
        _logger.warning("Regridding between different regular grids has not been tested.")

        """
        Regrid a single xarray Dataset to regular lat/lon grid. 
        Parameters
        ----------
            data : Input xarray DataArray containing the inference data on native grid.
        Returns
        -------
            Regridded xarray DataArray.
        """
        # set coords
        new_coords = data.coords.copy()
        new_coords.update(
            {
                "valid_time": data["valid_time"].values,
                "latitude": np.linspace(-90, 90, self.grid_shape[0]),
                "longitude": np.linspace(0, 360 - self.degree, self.grid_shape[1]),
            }
        )

        # set attrs
        attrs = data.attrs.copy()

        # find new dims and loop through extra dimensions
        original_shape = data.shape
        new_shape = list(original_shape)
        lat_pos = data.dims.index("latitude")
        lon_pos = data.dims.index("longitude")
        new_shape[lat_pos] = self.grid_shape[0]
        new_shape[lon_pos] = self.grid_shape[1]
        new_shape = tuple(new_shape)

        original_index = [list(range(original_shape_i)) for original_shape_i in original_shape]
        original_index[lat_pos, lon_pos] = [slice(None), slice(None)]  # :placeholder

        regridded_values = np.empty(new_shape)
        result = product(*original_index)
        for item in result:
            original_data_slice = data.values[item]
            regridded_slice = interpolate(
                original_data_slice, {"grid": self.earthkit_input}, {"grid": self.earthkit_output}
            )
            # sSet in regridded_values
            new_index = list(item)
            new_index[lat_pos] = slice(None)
            new_index[lon_pos] = slice(None)
            regridded_values[tuple(new_index)] = regridded_slice

        regrid_data = xr.DataArray(
            data=regridded_values, dims=data.dims, coords=new_coords, attrs=attrs, name=data.name
        )

        return regrid_data

    def find_num_cells(self) -> int:
        """
        Find number of cells in the (output) Gaussian grid based on N or O number.
        Returns
        -------
            num_cells : int
                Number of cells in the Gaussian grid.
        """
        if self.output_grid_type[0] == "N":
            n_lats = int(re.findall(r"\d+", self.earthkit_input)[0])
            num_cells = 4 * n_lats**2
            return num_cells
        elif self.output_grid_type[0] == "O":
            n_lats = int(re.findall(r"\d+", self.earthkit_input)[0])
            num_cells = 2 * n_lats * (n_lats + 1)
            return num_cells
        else:
            raise ValueError("Input grid type is not Gaussian, cannot find number of cells.")

    def gaussian_gaussian_da(self, data: xr.DataArray) -> xr.DataArray:
        """
        Regrid a single xarray Dataset to Gaussian grid.
        Parameters
        ----------
            data : Input xarray DataArray containing the inference data on native grid.
        Returns
        -------
            Regridded xarray DataArray.
        """
        _logger.warning("Regridding between different Gaussian grids has not been tested.")
        # set coords
        new_coords = data.coords.copy()
        new_coords.update(
            {
                "ncells": np.arange(self.grid_shape),
                # "valid_time": data["valid_time"].values,
            }
        )
        # set attrs
        attrs = data.attrs.copy()

        # find ncells position
        original_shape = data.shape
        new_shape = list(original_shape)
        pos = data.dims.index("ncells")
        new_shape[pos] = self.grid_shape
        new_shape = tuple(new_shape)
        # find indices
        original_index = [list(range(original_shape_i)) for original_shape_i in original_shape]
        original_index[pos] = [slice(None)]  # :placeholder

        regridded_values = np.empty(new_shape)
        result = product(*original_index)
        for item in result:
            original_data_slice = data.values[item]
            regridded_slice = interpolate(
                original_data_slice, {"grid": self.earthkit_input}, {"grid": self.earthkit_output}
            )
            # sSet in regridded_values
            new_index = list(item)
            new_index[pos] = slice(None)
            regridded_values[tuple(new_index)] = regridded_slice

        regrid_data = xr.DataArray(
            data=regridded_values, dims=data.dims, coords=new_coords, attrs=attrs, name=data.name
        )

        return regrid_data

    def prepare_data(
        self,
    ) -> None:
        """
        Prepare data for regridding.
        """
        if self.input_grid_type == "gaussian":
            ds = self.dataset
            # reorder everything except ncells
            original_ncells = ds["ncells"]
            ds = ds.isel(ncells=self.indices)
            ds["ncells"] = original_ncells
            self.dataset = ds
        else:
            pass

    def add_attrs(self, regrid_ds: xr.Dataset) -> xr.Dataset:
        """
        Preserve original coordinates after regridding.
        Parameters
        ----------
            regrid_ds : xr.Dataset
                Regridded xarray Dataset.
        Returns
        -------
            regrid_ds : xr.Dataset
                xarray Dataset with coordinates.
        """
        ds = self.dataset

        if self.input_grid_type == "gaussian" and self.output_grid_type == "regular_ll":
            for coord in ds.coords:
                if coord not in ["latitude", "longitude"]:
                    if "ncells" not in ds[coord].dims:
                        regrid_ds.coords[coord] = ds[coord]
                else:
                    # preserve CF attributes
                    regrid_ds.coords[coord].attrs = ds[coord].attrs
        if self.input_grid_type == "regular_ll" and self.output_grid_type == "gaussian":
            raise NotImplementedError(
                "Preserving coordinates when regridding from regular lat/lon grids "
                "to Gaussian grids is not implemented yet."
            )

        # keep global attrs
        regrid_ds.attrs = ds.attrs
        # change grid_type
        regrid_ds.attrs["grid_type"] = self.output_grid_type
        regrid_ds.attrs["history"] += (
            f" and regridded from {self.earthkit_input} to {self.earthkit_output} using earthkit"
        )

        return regrid_ds

    def regrid_ds(
        self,
    ) -> xr.Dataset:
        """
        Regrids an xarray Dataset from native grid to chosen grid.
        Returns
        -------
            Regridded xarray Dataset.
        """
        self.input_grid_type = self.detect_input_grid_type()
        self.earthkit_input = self.define_earthkit_input()
        self.earthkit_output, self.grid_shape = self.define_earthkit_output()
        _logger.info(f"Attempting to regrid from {self.earthkit_input} to {self.earthkit_output}")
        # No regridding needed if both input and output are same degree
        if self.input_grid_type == self.output_grid_type:
            if self.earthkit_input == self.earthkit_output:
                _logger.info("Input and output grid types are the same, skipping regridding step.")
                return self.dataset
        self.prepare_data()

        ds = self.dataset

        regrid_vars = {}
        for var in ds.data_vars:
            regrid_vars[var] = self.regrid_da(ds[var])
        regrid_ds = xr.Dataset(regrid_vars)
        regrid_ds = self.add_attrs(regrid_ds)

        return regrid_ds

    def regrid_da(self, da: xr.DataArray) -> xr.DataArray:
        """
        Regrid a single xarray DataArray from input grid to output grid.

        Parameters
        ----------
            da : Input xarray DataArray containing the inference data on native grid.
        Returns
            Regridded xarray DataArray.
        -------
        """
        if self.input_grid_type == "gaussian" and self.output_grid_type == "regular_ll":
            regrid_da = self.gaussian_regular_da(da)
        elif self.input_grid_type == "regular_ll" and self.output_grid_type == "gaussian":
            regrid_da = self.regular_gaussian_da(da)
        elif self.input_grid_type == self.output_grid_type:
            regrid_da = self.same_grid_da(da)
        else:
            raise NotImplementedError(
                f"""Regridding from {self.earthkit_input} to {self.earthkit_output} grid 
                is not implemented yet."""
            )
        return regrid_da
