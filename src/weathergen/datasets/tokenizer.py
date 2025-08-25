# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings
from abc import ABC, abstractmethod

import astropy_healpix as hp
import numpy as np
import torch

from weathergen.datasets.utils import (
    healpix_verts_rots,
    locs_to_cell_coords_ctrs,
)

class Tokenizer(ABC):
    """
    Base class for tokenizers.
    """
    def __init__(self, healpix_level: int):
        ref = torch.tensor([1.0, 0.0, 0.0])

        self.hl_source = healpix_level
        self.hl_target = healpix_level

        self.num_healpix_cells_source = 12 * 4**self.hl_source
        self.num_healpix_cells_target = 12 * 4**self.hl_target

        verts00, verts00_rots = healpix_verts_rots(self.hl_source, 0.0, 0.0)
        verts10, verts10_rots = healpix_verts_rots(self.hl_source, 1.0, 0.0)
        verts11, verts11_rots = healpix_verts_rots(self.hl_source, 1.0, 1.0)
        verts01, verts01_rots = healpix_verts_rots(self.hl_source, 0.0, 1.0)
        vertsmm, vertsmm_rots = healpix_verts_rots(self.hl_source, 0.5, 0.5)
        self.hpy_verts = [
            verts00.to(torch.float32),
            verts10.to(torch.float32),
            verts11.to(torch.float32),
            verts01.to(torch.float32),
            vertsmm.to(torch.float32),
        ]
        self.hpy_verts_rots_source = [
            verts00_rots.to(torch.float32),
            verts10_rots.to(torch.float32),
            verts11_rots.to(torch.float32),
            verts01_rots.to(torch.float32),
            vertsmm_rots.to(torch.float32),
        ]

        verts00, verts00_rots = healpix_verts_rots(self.hl_target, 0.0, 0.0)
        verts10, verts10_rots = healpix_verts_rots(self.hl_target, 1.0, 0.0)
        verts11, verts11_rots = healpix_verts_rots(self.hl_target, 1.0, 1.0)
        verts01, verts01_rots = healpix_verts_rots(self.hl_target, 0.0, 1.0)
        vertsmm, vertsmm_rots = healpix_verts_rots(self.hl_target, 0.5, 0.5)
        self.hpy_verts = [
            verts00.to(torch.float32),
            verts10.to(torch.float32),
            verts11.to(torch.float32),
            verts01.to(torch.float32),
            vertsmm.to(torch.float32),
        ]
        self.hpy_verts_rots_target = [
            verts00_rots.to(torch.float32),
            verts10_rots.to(torch.float32),
            verts11_rots.to(torch.float32),
            verts01_rots.to(torch.float32),
            vertsmm_rots.to(torch.float32),
        ]

        self.verts_local = []
        verts = torch.stack([verts10, verts11, verts01, vertsmm])
        temp = ref - torch.stack(locs_to_cell_coords_ctrs(verts00_rots, verts.transpose(0, 1)))
        self.verts_local.append(temp.flatten(1, 2))

        verts = torch.stack([verts00, verts11, verts01, vertsmm])
        temp = ref - torch.stack(locs_to_cell_coords_ctrs(verts10_rots, verts.transpose(0, 1)))
        self.verts_local.append(temp.flatten(1, 2))

        verts = torch.stack([verts00, verts10, verts01, vertsmm])
        temp = ref - torch.stack(locs_to_cell_coords_ctrs(verts11_rots, verts.transpose(0, 1)))
        self.verts_local.append(temp.flatten(1, 2))

        verts = torch.stack([verts00, verts11, verts10, vertsmm])
        temp = ref - torch.stack(locs_to_cell_coords_ctrs(verts01_rots, verts.transpose(0, 1)))
        self.verts_local.append(temp.flatten(1, 2))

        verts = torch.stack([verts00, verts10, verts11, verts01])
        temp = ref - torch.stack(locs_to_cell_coords_ctrs(vertsmm_rots, verts.transpose(0, 1)))
        self.verts_local.append(temp.flatten(1, 2))

        self.hpy_verts_local_target = torch.stack(self.verts_local).transpose(0, 1)

        # add local coords wrt to center of neighboring cells
        # (since the neighbors are used in the prediction)
        num_healpix_cells = 12 * 4**self.hl_target
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(
                np.arange(num_healpix_cells), 2**self.hl_target, order="nested"
            ).transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        self.hpy_nctrs_target = (
            vertsmm[temp.flatten()]
            .reshape((num_healpix_cells, 8, 3))
            .transpose(1, 0)
            .to(torch.float32)
        )

        self.size_time_embedding = 6

    def get_size_time_embedding(self) -> int:
        """
        Get size of time embedding
        """
        return self.size_time_embedding
    
    @abstractmethod
    def reset_rng(self, rng) -> None:
        """
        Abstract method that all subclasses must implement.
        Reset rng after epoch to ensure proper randomization
        """
        pass

    @abstractmethod
    def batchify_source(
        self,
        stream_info: dict,
        coords: np.array,
        geoinfos: np.array,
        source: np.array,
        times: np.array,
        time_win: tuple,
        normalizer,  # dataset
    ):
        """
        Abstract method that all subclasses must implement.
        """
        pass

    @abstractmethod
    def batchify_target(
        self,
        stream_info: dict,
        sampling_rate_target: float,
        coords: np.array,
        geoinfos: np.array,
        source: np.array,
        times: np.array,
        time_win: tuple,
        normalizer,  # dataset
    ):
        """
        Abstract method that all subclasses must implement.
        """
        pass