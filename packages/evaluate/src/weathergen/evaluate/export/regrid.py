import pathlib
import xarray as xr
import numpy as np
import logging
from earthkit.regrid import interpolate
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def find_lat_lon_ordering(ds: xr.Dataset) -> list[int]:
    """
    Find all the the latitude and longitude ordering for the WeatherGenerator data
    Ordering from North West to South East.
    Returns the indices required to reorder the data.

    Parameters
    ----------
        ds : Input xarray Dataset containing the inference data,

    Returns
    -------
        indices: list of indices to reorder the data from original to lat/lon ordered.
    """
    x = ds['longitude'].values[:,0]
    y = ds['latitude'].values[:,0]
    tuples = list(zip(x, y))
    ordered_tuples = sorted(tuples, key=lambda t: (-t[1], t[0]))
    indices = [tuples.index(t) for t in ordered_tuples]
    return indices

def regrid_gaussian_ds(data: xr.DataArray, grid_type: str, degree: float, indices: list) -> xr.DataArray:
    """
    Regrid a single xarray DataArray from O96 grid to regular lat/lon grid.

    Parameters
    ----------
        indices: list of indices to reorder the data from original to lat/lon ordered.
        data : Input xarray DataArray containing the inference data on Gaussian grid.
        grid_type : Type of grid to regrid to (e.g., 'regular_ll').
        degree : Degree of the regular lat/lon grid (e.g., 2 for 2.0 degree grid)/96 for O96 grid

    Returns
    -------
        Regridded xarray DataArray.
    """
    #grid_type logic
    if grid_type == 'regular_ll':
        grid_type = [degree, degree]
        grid_shape = (180 // degree + 1, 360 // degree)
    else:
        raise ValueError(f'Unsupported grid_type: {grid_type}, supported types are ["regular"]')
        # to be implemented:
        grid_type = grid_type + str(degree)
        grid_shape = None # ????? to be determined

    # reorder everything except ncells
    og_ncells = data['ncells'].values
    indices = find_lat_lon_ordering(data)
    data = data.isel(ncells = indices)
    data['ncells'] = (og_ncells)

    # create empty xarray ds for regridded data
    values = np.empty((data.shape[0], grid_shape[0], grid_shape[1], data.shape[2]))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            data_reg_var = interpolate(
                data.values[i,:,j],
                {'grid': 'O96'},
                {'grid': grid_type}
                )
            values[i,:,:,j] = data_reg_var
    dims = list(data.dims)
    pos = dims.index('ncells')
    dims[pos:pos+1] = ['latitude', 'longitude']
    dims = tuple(dims)
    regrid_data = xr.DataArray(
        data = values,
        dims = dims,
        coords = {
            'valid_time': data['valid_time'].values,
            'latitude': np.linspace(-90, 90, grid_shape[0]),
            'longitude': np.linspace(0, 360 - degree, grid_shape[1]),
        },
        attrs = data.attrs,
        name = data.name
    )
    return regrid_data

def regrid_gaussian(directory: str, output_directory: str, grid_type: str, degree: float) -> None:
    """
    Regrid all the files in a directory from O96 grid to regular lat/lon grid.
    Parameters
    ----------
        directory : Input directory containing the inference data on Gaussian grid.
        output_directory : Output directory to save the regridded data.
        grid_type : Type of grid to regrid to (e.g., 'regular_ll').
        degree : Degree of the regular lat/lon grid (e.g., 2 for 2.0 degree grid)/96 for O96 grid
    Returns
    -------
        None
    """
    #grid_type logic
    if grid_type == 'regular_ll':
        grid_type = [degree, degree]
        grid_shape = (180 // degree + 1, 360 // degree)
    else:
        raise ValueError(f'Unsupported grid_type: {grid_type}, supported types are ["regular"]')
        # to be implemented:
        grid_type = grid_type + str(degree)
        grid_shape = None # ????? to be determined

    #make output directory if it doesn't exist
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    # list of all inputs
    input_paths = os.listdir(pathlib.Path(directory))
    input_paths = [pathlib.Path(directory, f) for f in input_paths]
    # find ordering (should be same across all files)
    test_ds = xr.open_dataset(input_paths[0])
    indices = find_lat_lon_ordering(test_ds)
    for path in input_paths:
        print(f'Regridding file: {path}')
        new_ds = xr.open_dataset(path)
        # reorder everything except ncells
        og_ncells = new_ds['ncells'].values
        new_ds = new_ds.isel(ncells = indices)
        new_ds['ncells'] = (og_ncells)
        # create empty xarray ds for regridded data
        vars = {}

        for var in new_ds.data_vars:
            if new_ds[var].ndim == 3:
                values = np.empty((new_ds[var].shape[0], grid_shape[0], grid_shape[1], new_ds[var].shape[2]))
                x = 0
            else:
                values = np.empty((grid_shape[0], grid_shape[1], new_ds[var].shape[1]))
                x = 1
            for i in range(new_ds[var].shape[x]):
                if new_ds[var].ndim == 3:
                    for j in range(new_ds[var].shape[2]):
                        data_reg_var = interpolate(
                            new_ds[var].values[i,:,j],
                            {'grid': 'O96'},
                            {'grid': grid_type}
                            )
                        values[i,:,:,j] = data_reg_var
                else:
                    data_reg_var = interpolate(
                        new_ds[var].values[:,i],
                        {'grid': 'O96'},
                        {'grid': grid_type}
                        )
                    values[:,:,i] = data_reg_var
            dims = list(new_ds[var].dims)
            pos = dims.index('ncells')
            dims[pos:pos+1] = ['latitude', 'longitude']
            dims = tuple(dims)
            vars[var] = xr.DataArray(
                data = values,
                dims = dims,
                coords = {
                    'valid_time': new_ds['valid_time'].values,
                    'latitude': np.linspace(-90, 90, grid_shape[0]),
                    'longitude': np.linspace(0, 360 - degree, grid_shape[1]),
                },
                attrs = new_ds[var].attrs,
                name = var
            )
        regrid_ds = xr.Dataset(vars)
        for coord in new_ds.coords:
            print(coord)
            if coord not in ['latitude', 'longitude']:
                if 'ncells' not in new_ds[coord].dims:
                    regrid_ds.coords[coord] = new_ds[coord]
            else:
                #preserve units
                regrid_ds.coords[coord].attrs = new_ds[coord].attrs

        regrid_ds.attrs = new_ds.attrs
        #change grid_type
        regrid_ds.attrs['grid_type'] = grid_type
        regrid_ds.attrs['history'] += f'Regridded from O96 to {degree} degree {grid_type} using earthkit'
                        
        # filenames
        output_name = path.stem + '_regridded.nc'
        output_path = pathlib.Path(output_directory) / output_name
        print(f'Saving regridded file to: {output_path}')
        regrid_ds.to_netcdf(output_path)