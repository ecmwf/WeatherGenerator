# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from weathergen.datasets.domain import Domain
from weathergen.datasets.utils import healpix_verts_rots


class HealpixGeometry:
    """Precomputed HEALPix geometry shared by source and target coordinate encoders."""

    def __init__(self, healpix_level: int, domain: Domain | None = None):
        self.healpix_level = healpix_level
        self.domain = domain if domain is not None else Domain(healpix_level)
        self.num_healpix_cells = len(self.domain)

        vertices, rotations = self._compute_vertices_and_rotations()
        self.hpy_verts = [vertex.to(torch.float32) for vertex in vertices]
        self.hpy_verts_rots = [rotation.to(torch.float32) for rotation in rotations]
        self.hpy_verts_local = self._compute_local_vertices(vertices, rotations)
        self.hpy_nctrs = self._compute_neighbor_centers(vertices[-1])

    def _compute_vertices_and_rotations(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        cells = self.domain.active_cells
        verts00, verts00_rots = healpix_verts_rots(self.healpix_level, 0.0, 0.0, cells)
        verts10, verts10_rots = healpix_verts_rots(self.healpix_level, 1.0, 0.0, cells)
        verts11, verts11_rots = healpix_verts_rots(self.healpix_level, 1.0, 1.0, cells)
        verts01, verts01_rots = healpix_verts_rots(self.healpix_level, 0.0, 1.0, cells)
        vertsmm, vertsmm_rots = healpix_verts_rots(self.healpix_level, 0.5, 0.5, cells)
        return (
            [verts00, verts10, verts11, verts01, vertsmm],
            [verts00_rots, verts10_rots, verts11_rots, verts01_rots, vertsmm_rots],
        )

    @staticmethod
    def _compute_local_vertices(
        vertices: list[torch.Tensor], rotations: list[torch.Tensor]
    ) -> torch.Tensor:
        ref = torch.tensor([1.0, 0.0, 0.0])
        verts00, verts10, verts11, verts01, vertsmm = vertices
        verts00_rots, verts10_rots, verts11_rots, verts01_rots, vertsmm_rots = rotations

        transforms = [
            ([verts10, verts11, verts01, vertsmm], verts00_rots),
            ([verts00, verts11, verts01, vertsmm], verts10_rots),
            ([verts00, verts10, verts01, vertsmm], verts11_rots),
            ([verts00, verts11, verts10, vertsmm], verts01_rots),
            ([verts00, verts10, verts11, verts01], vertsmm_rots),
        ]

        verts_local = []
        for _verts, rot in transforms:
            # Compute local coordinates
            verts = torch.stack(_verts)
            # shape: <healpix, 4, 3>
            verts = verts.transpose(0, 1)
            # Batch multiplication by the 3x3 rotation matrices.
            # shape: <healpix, 3, 3> @ <healpix, 4, 3> -> <healpix, 4, 3>
            # Needs to transpose first to <healpix, 3, 4> then transpose back.
            t1 = torch.bmm(rot, verts.transpose(-1, -2)).transpose(-2, -1)
            t2 = ref - t1
            verts_local.append(t2.flatten(1, 2))

        return torch.stack(verts_local).transpose(0, 1)

    def _compute_neighbor_centers(self, vertsmm: torch.Tensor) -> torch.Tensor:
        # add local coords wrt to center of neighboring cells
        # (since the neighbors are used in the prediction)
        temp = self.domain.neighbours_compact()
        return (
            vertsmm[temp.flatten()]
            .reshape((self.num_healpix_cells, 8, 3))
            .transpose(1, 0)
            .to(torch.float32)
        )
