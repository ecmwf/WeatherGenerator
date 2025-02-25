# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from weathergen.datasets.utils import arc_alpha


class DataNormalizer:
    ###################################################
    def __init__(self, stream_info, geoinfo_offset, stats_offset, ds, geoinfo_idx, data_idx, do):
        # obs_id, year, day_of_year, day
        self.geoinfo_offset = geoinfo_offset
        self.stats_offset = stats_offset

        self.stream_info = stream_info
        self.geoinfo_idx = np.array(geoinfo_idx)
        self.data_idx = np.array(data_idx)
        self.geoinfo_size = len(geoinfo_idx)

        self.source_chs = np.arange(len(data_idx))
        self.loss_chs = np.arange(len(data_idx))

        self.mean = np.array(ds.properties["means"])[do:]
        self.var = np.array(ds.properties["vars"])[do:]

    ###################################################
    def normalize_data(self, data, with_offset=True):
        go = self.geoinfo_size + self.geoinfo_offset
        so = self.stats_offset
        for i, ch in enumerate(self.data_idx):
            data[..., go + i] = (data[..., go + i] - self.mean[ch - so]) / (
                self.var[ch - so] ** 0.5
            )

        return data

    ###################################################
    def denormalize_data(self, data, with_offset=True):
        go = self.geoinfo_size + self.geoinfo_offset if with_offset else 0
        so = self.stats_offset
        for i, ch in enumerate(self.data_idx):
            data[..., go + i] = (data[..., go + i] * (self.var[ch - so] ** 0.5)) + self.mean[
                ch - so
            ]

        return data

    ###################################################
    def normalize_coords(self, data, normalize_latlon=True):
        so = self.stats_offset

        # TODO: geoinfo_offset should be derived from the code below and the corresponding code in
        #       multi_obs_data_sampler
        # obs_id, year, day of the year, minute of the day
        assert self.geoinfo_offset == 6
        data[..., 0] /= 256.0
        data[..., 1] /= 2100.0
        data[..., 2] = data[..., 2] / 365.0
        data[..., 3] = data[..., 3] / 1440.0
        data[..., 4] = np.sin(data[..., 4] / (12.0 * 3600.0) * 2.0 * np.pi)
        data[..., 5] = np.cos(data[..., 5] / (12.0 * 3600.0) * 2.0 * np.pi)

        go = self.geoinfo_offset
        for i, ch in enumerate(self.geoinfo_idx):
            if i == 0:  # lats
                if normalize_latlon:
                    data[..., go + i] = np.sin(np.deg2rad(data[..., go + i]))
                pass
            elif i == 1:  # lons
                if normalize_latlon:
                    data[..., go + i] = np.sin(0.5 * np.deg2rad(data[..., go + i]))
            else:
                data[..., go + i] = (data[..., go + i] - self.mean[ch - so]) / (
                    (self.var[ch - so] ** 0.5) if self.var[ch - so] > 0.0 else 1.0
                )

        return data

    ###################################################
    def normalize_targets(self, data):
        so = self.stats_offset

        # TODO: geoinfo_offset should be derived from the code below and the corresponding code in
        #       multi_obs_data_sampler
        # obs_id, year, day of the year, minute of the day
        assert self.geoinfo_offset == 6
        data[..., 0] /= 256.0
        data[..., 1] = np.sin(data[..., 1] / (12.0 * 3600.0) * 2.0 * np.pi)
        data[..., 2] = np.cos(data[..., 2] / (12.0 * 3600.0) * 2.0 * np.pi)
        data[..., 3] = np.sin(data[..., 3] / (12.0 * 3600.0) * 2.0 * np.pi)
        data[..., 4] = np.cos(data[..., 4] / (12.0 * 3600.0) * 2.0 * np.pi)
        data[..., 5] = np.sin(data[..., 5] / (12.0 * 3600.0) * 2.0 * np.pi)

        go = self.geoinfo_offset
        for i, ch in enumerate(self.geoinfo_idx):
            if i > 1:  # skip lat/lon
                data[..., go + i] = (data[..., go + i] - self.mean[ch - so]) / (
                    (self.var[ch - so] ** 0.5) if self.var[ch - so] > 0.0 else 1.0
                )

        return data

    ###################################################
    def denormalize_coords(self, data):
        # obs_id, year, day of the year, minute of the day
        assert self.geoinfo_offset == 6
        data[..., 0] *= 256.0
        data[..., 1] = (arc_alpha(data[..., 1], data[..., 2]) / (2.0 * np.pi)) * (12.0 * 3600.0)
        data[..., 2] = data[..., 1]
        data[..., 3] = data[..., 1]
        data[..., 4] = data[..., 1]
        data[..., 5] = data[..., 1]

        # go = self.geoinfo_offset
        # for i, ch in enumerate( self.geoinfo_idx) :
        #   if 0 == i : # lats
        #     data[...,go+i] = torch.rad2deg( torch.arcsin( data[...,go+i]))
        #   elif 1 == i :  # lons
        #     data[...,go+i] = torch.rad2deg( 2.0 * torch.arcsin( data[...,go+i]))
        #   else :
        #     data[...,go+i] = (data[...,go+i] * (self.var[ch]**0.5)) + self.mean[ch]

        return data
