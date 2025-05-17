import os
import json
import numpy as np
import xarray as xr
from datetime import datetime

class RadklimDataset:
    def __init__(self, start_time, end_time, len_hrs, step_hrs, data_path, normalization_path):
        self.start_time = datetime.strptime(str(start_time), '%Y%m%d%H%M')
        self.end_time = datetime.strptime(str(end_time), '%Y%m%d%H%M')
        self.len_hrs = len_hrs
        self.step_hrs = step_hrs

        # Needs changes after discussing the stats file
        stats = json.load(open(normalization_path))
        self.mean = np.array(stats['mean'], dtype=np.float32).reshape((1, -1, 1, 1))
        self.std  = np.array(stats['std'],  dtype=np.float32).reshape((1, -1, 1, 1))

        # collect files per month window
        # I believe this is more simple
        sy = self.start_time.year * 100 + self.start_time.month
        ey = self.end_time.year   * 100 + self.end_time.month
        files = []
        for root, _, fs in os.walk(data_path):
            for f in fs:
                if f.endswith('.nc'):
                    digs = ''.join(filter(str.isdigit, f))
                    if len(digs) >= 6 and sy <= int(digs[:6]) <= ey:
                        files.append(os.path.join(root, f))
        files = sorted(files)

        # open dataset lazily (no parallel to avoid segfaults)
        # I really need your help for the open_mfdataset init
        self.ds = xr.open_mfdataset(
            files,
            combine='by_coords',
            chunks={'time': step_hrs, 'y': 200, 'x': 200},
            engine='netcdf4',
            parallel=False,
        )
        self.ds = self.ds.sel(time=slice(self.start_time, self.end_time))

        # keep only float variables (exclude 'crs')
        # I am not sure how to handle the crs
        self.variables = [v for v in self.ds.data_vars if self.ds[v].dtype.kind == 'f' and v != 'crs']
        self.ds = self.ds[self.variables]

        # stack into 4D DataArray
        self.data_array = self.ds.to_array('var').transpose('time', 'var', 'y', 'x')
        self.times = self.ds['time'].values


    # check it please
    def __len__(self):
        return max(0, (len(self.times) - self.len_hrs) // self.step_hrs + 1)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise IndexError('Index out of range')
        si = idx * self.step_hrs
        ei = si + self.len_hrs
        arr = self.data_array.isel(time=slice(si, ei)).compute().values
        # normalize: arr shape (len_hrs, nvar, y, x)
        return (arr - self.mean) / self.std

    def get_source(self, idx):
        return self._get(idx)

    def get_target(self, idx):
        return self._get(idx)

    # check it please
    def _get(self, idx):
        block = self[idx]  # shape (len_hrs, nvar, y, x)
        t, nvar, ny, nx = block.shape
        # coords
        lats = self.ds['y'].values
        lons = self.ds['x'].values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        base_coords = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)
        coords = np.tile(base_coords, (t, 1))
        # geoinfos
        geoinfos = np.zeros((coords.shape[0], 0), dtype=np.float32)
        # data flattened
        data = block.transpose(0, 2, 3, 1).reshape(-1, nvar)
        # times: slice start to end-1
        si = idx * self.step_hrs
        ei = si + self.len_hrs - 1
        times = np.repeat(self.times[si:ei+1], ny * nx)
        return coords, geoinfos, data, times

    def get_source_num_channels(self):
        return len(self.variables)

    def get_target_num_channels(self):
        return len(self.variables)

    def get_coords_size(self):
        return 2

    def get_geoinfo_size(self):
        return 0

    def normalize_source_channels(self, data):
        # data shape (..., nvar)
        return (data - self.mean.reshape(-1)) / self.std.reshape(-1)

    def normalize_target_channels(self, data):
        return (data - self.mean.reshape(-1)) / self.std.reshape(-1)

    def denormalize_source_channels(self, data):
        return data * self.std.reshape(-1) + self.mean.reshape(-1)

    def denormalize_target_channels(self, data):
        return data * self.std.reshape(-1) + self.mean.reshape(-1)

    def time_window(self, idx):
        if not (0 <= idx < len(self)):
            return (np.array([], dtype='datetime64'), np.array([], dtype='datetime64'))
        si = idx * self.step_hrs
        ei = si + self.len_hrs - 1
        return (self.times[si], self.times[ei])