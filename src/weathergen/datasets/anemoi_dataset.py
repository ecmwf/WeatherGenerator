# (C) Copyright 2024 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import code

import numpy as np

from anemoi.datasets import open_dataset


class AnemoiDataset():
  "Wrapper for Anemoi dataset"

  def __init__(
        self,
        filename: str,
        start: int,
        end: int,
        len_hrs: int,
        step_hrs: int = None,
        normalize: bool = True,
        select: list[str] = None,
    ) -> None :

    assert len_hrs == step_hrs, 'Currently only step_hrs=len_hrs is supported'

    self.ds = open_dataset( filename, frequency=str(step_hrs) + 'h', 
                                      start=str(start)[:-4], end=str(end)[:-4] )
    # caches lats and lons
    self.latitudes = self.ds.latitudes.astype( np.float32)
    self.longitudes = self.ds.longitudes.astype( np.float32)

    self.colnames = ['lat', 'lon'] + self.ds.variables

    self.properties = { 'obs_id' : 0,
                        'means' : self.ds.statistics['mean'], 
                        'vars' : np.square(self.ds.statistics['stdev']), }

  def __len__(self) :
    "Length of dataset"
    return len(self.ds)

  def __getitem__( self, idx: int) -> tuple :
    "Get (data,datetime) for given index"
    
    # prepend lat and lon to data; squeeze out ensemble dimension (for the moment)
    data = np.concatenate( [np.expand_dims( self.latitudes, 0), 
                            np.expand_dims( self.longitudes, 0),
                            self.ds[idx].squeeze()], 0).transpose()

    # date time matching #data points of data
    datetimes = np.full( data.shape[0], self.ds.dates[idx])

    return (data, datetimes)

  def time_window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
    return (self.ds.dates[idx], self.ds.dates[idx])
