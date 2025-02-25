# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings

import astropy_healpix as hp
import numpy as np
import torch
from astropy_healpix.healpy import ang2pix


####################################################################################################
def arc_alpha(sin_alpha, cos_alpha):
    """Invert cosine/sine for alpha \in [0,2pi] using both functions"""
    t = torch.arccos(cos_alpha)
    mask = sin_alpha < 0.0
    t[mask] = (2.0 * np.pi) - t[mask]
    return t


####################################################################################################
def merge_cells(s_list, num_healpix_cells):
    if torch.tensor([len(s) for s in s_list]).sum() == 0:
        return torch.tensor([])

    ret = torch.cat(
        [
            torch.cat([s_list[i_s][i] for i_s in range(len(s_list)) if len(s_list[i_s]) > 0])
            for i in range(num_healpix_cells)
        ]
    )

    return ret


####################################################################################################
def vecs_to_rots(vecs):
    """
    Convert vectors to rotations that align with (1,0,0) ie coordinate origin in geophysical
    spherical coordinates. A variant of Rodrigues formula is used
    """

    Rs = torch.zeros((vecs.shape[0], 3, 3), dtype=torch.float64)
    c1 = vecs[:, 0]
    c2 = vecs[:, 1]
    c3 = vecs[:, 2]
    s = torch.square(c2) + torch.square(c3)
    Rs[:, 0, 0] = c1
    Rs[:, 0, 1] = c2
    Rs[:, 0, 2] = c3
    Rs[:, 1, 0] = -c2
    Rs[:, 1, 1] = (c1 * torch.square(c2) + torch.square(c3)) / s
    Rs[:, 1, 2] = (-1.0 + c1) * c2 * c3 / s
    Rs[:, 2, 0] = -c3
    Rs[:, 2, 1] = (-1.0 + c1) * c2 * c3 / s
    Rs[:, 2, 2] = (torch.square(c2) + c1 * torch.square(c3)) / s

    return Rs


####################################################################################################
def s2tor3(lats, lons):
    """
    Convert from spherical to Cartesion R^3 coordinates

    Note: mathematics convention with lats \in [0,pi] and lons \in [0,2pi] is used
          (which is not problematic for lons but for lats care is required)
    """
    x = torch.sin(lats) * torch.cos(lons)
    y = torch.sin(lats) * torch.sin(lons)
    z = torch.cos(lats)
    out = torch.stack([x, y, z])
    return out.permute([*list(np.arange(len(out.shape))[:-1] + 1), 0])


####################################################################################################
def r3tos2(pos):
    """
    Convert from spherical to Cartesion R^3 coordinates

    Note: mathematics convention with lats \in [0,pi] and lons \in [0,2pi] is used
          (which is not problematic for lons but for lats care is required)
    """
    norm2 = torch.square(pos[..., 0]) + torch.square(pos[..., 1])
    # r = torch.sqrt(norm2 + torch.square(pos[..., 2]))
    lats = torch.atan2(pos[..., 2], torch.sqrt(norm2))
    lons = torch.atan2(pos[..., 1], pos[..., 0])
    out = torch.stack([lats, lons])
    return out.permute([*list(torch.arange(len(out.shape))[:-1] + 1), 0])


####################################################################################################
def locs_to_cell_coords(hl: int, locs: list, dx=0.5, dy=0.5) -> list:
    """
    Map a list of locations per cell to spherical local coordinates centered
    at the healpix cell center
    """

    assert locs[13].shape[-1] == 3 if len(locs[13]) > 0 else True

    # centroids of healpix cells
    num_healpix_cells = 12 * 4**hl
    assert len(locs) == num_healpix_cells

    lons, lats = hp.healpix_to_lonlat(
        np.arange(0, num_healpix_cells), 2**hl, dx=dx, dy=dy, order="nested"
    )
    healpix_centers = s2tor3(
        torch.from_numpy(np.pi / 2.0 - lats.value), torch.from_numpy(lons.value)
    )
    healpix_centers_Rs = vecs_to_rots(healpix_centers)

    # express each centroid in local coordinates w.r.t to healpix center by rotating center to origin
    local_locs = [
        torch.matmul(R, s.transpose(-1, -2)).transpose(-2, -1) if len(s) > 0 else torch.tensor([])
        for i, (R, s) in enumerate(zip(healpix_centers_Rs, locs, strict=False))
    ]

    return local_locs


####################################################################################################
def locs_to_ctr_coords(ctrs_r3, locs: list) -> list:
    """
    Map a list of locations per cell to spherical local coordinates centered
    at the healpix cell center
    """

    ctrs_Rs = vecs_to_rots(ctrs_r3).to(torch.float32)

    # express each centroid in local coordinates w.r.t to healpix center by rotating center to origin
    local_locs = [
        torch.matmul(R, s.transpose(-1, -2)).transpose(-2, -1)
        if len(s) > 0
        else torch.zeros([0, 3])
        for i, (R, s) in enumerate(zip(ctrs_Rs, locs, strict=False))
    ]

    return local_locs


####################################################################################################
def healpix_verts(hl: int, dx=0.5, dy=0.5):
    """
    healpix cell center
    """

    # centroids of healpix cells
    num_healpix_cells = 12 * 4**hl
    lons, lats = hp.healpix_to_lonlat(
        np.arange(0, num_healpix_cells), 2**hl, dx=dx, dy=dy, order="nested"
    )
    verts = s2tor3(torch.from_numpy(np.pi / 2.0 - lats.value), torch.from_numpy(lons.value))

    return verts


####################################################################################################
def healpix_verts_rots(hl: int, dx=0.5, dy=0.5):
    """
    healpix cell center
    """

    # centroids of healpix cells
    num_healpix_cells = 12 * 4**hl
    lons, lats = hp.healpix_to_lonlat(
        np.arange(0, num_healpix_cells), 2**hl, dx=dx, dy=dy, order="nested"
    )
    verts = s2tor3(torch.from_numpy(np.pi / 2.0 - lats.value), torch.from_numpy(lons.value))
    verts_R3 = vecs_to_rots(verts)

    return verts, verts_R3


####################################################################################################
def locs_to_cell_coords_ctrs(healpix_centers_Rs, locs: list) -> list:
    """
    Map a list of locations per cell to spherical local coordinates centered
    at the healpix cell center
    """

    # express each centroid in local coordinates w.r.t to healpix center by rotating center to origin
    local_locs = [
        torch.matmul(R, s.transpose(-1, -2)).transpose(-2, -1) if len(s) > 0 else torch.tensor([])
        for i, (R, s) in enumerate(zip(healpix_centers_Rs, locs, strict=False))
    ]

    return local_locs


####################################################################################################
def coords_to_hpyidxs(hl, thetas, phis):
    thetas = ((90.0 - thetas) / 180.0) * np.pi
    phis = ((180.0 + phis) / 360.0) * 2.0 * np.pi
    hpyidxs = ang2pix(2**hl, thetas, phis, nest=True)

    return hpyidxs


####################################################################################################
def add_local_vert_coords(hl, a, verts, tcs, zi, dx, dy, geoinfo_offset):
    ref = torch.tensor([1.0, 0.0, 0.0])
    aa = locs_to_cell_coords(hl, verts.unsqueeze(1), dx, dy)
    aa = ref - torch.cat(
        [
            aaa.repeat([*tt.shape[:-1], 1]) if len(tt) > 0 else torch.tensor([])
            for tt, aaa in zip(tcs, aa, strict=False)
        ]
    )
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = aa
    return a


####################################################################################################
def add_local_vert_coords_ctrs2(ctrs, verts, tcs, a, zi, geoinfo_offset):
    ref = torch.tensor([1.0, 0.0, 0.0])
    aa = locs_to_cell_coords_ctrs(ctrs, verts.transpose(0, 1))
    aa = ref - torch.cat(
        [
            aaa.unsqueeze(0).repeat([*tt.shape[:-1], 1, 1]) if len(tt) > 0 else torch.tensor([])
            for tt, aaa in zip(tcs, aa, strict=False)
        ],
        0,
    )
    aa = aa.flatten(1, 2)
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + aa.shape[-1])] = aa
    return a


####################################################################################################
def add_local_vert_coords_ctrs2(verts_local, tcs_lens, a, zi, geoinfo_offset):
    ref = torch.tensor([1.0, 0.0, 0.0])
    # aa = locs_to_cell_coords_ctrs(ctrs, verts.transpose(0, 1))
    aa = ref - torch.cat(
        [
            aaa.unsqueeze(0).repeat([*tcs_lens, 1, 1]) if len(tt) > 0 else torch.tensor([])
            for tt, aaa in zip(tcs_lens, verts_local, strict=False)
        ],
        0,
    )
    aa = aa.flatten(1, 2)
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + aa.shape[-1])] = aa
    return a


# ####################################################################################################
# def add_local_vert_coords_ctrs3( ctrs, verts, tcs, a, zi, geoinfo_offset) :

#   ref = torch.tensor( [1., 0., 0.])

#   local_locs = [torch.matmul( R, s.transpose( -1, -2)).transpose( -2, -1)
#                               for i,(R,s) in enumerate(zip(healpix_centers_Rs,locs)) if len(s)>0]
#   aa = locs_to_cell_coords_ctrs( ctrs, verts.transpose(0,1))
#   aa = ref - torch.cat( [aaa.unsqueeze(0).repeat( [*tt.shape[:-1],1,1])
#                                                               if len(tt)>0 else torch.tensor([])
#                                                               for tt,aaa in zip(tcs,aa)] if tt>, 0 )
#   aa = aa.flatten(1,2)
#   a[...,(geoinfo_offset+zi):(geoinfo_offset+zi+aa.shape[-1])] = aa
#   return a


####################################################################################################
def get_target_coords_local(hlc, target_coords, geoinfo_offset):
    """Generate local coordinates for target coords w.r.t healpix cell vertices and
    and for healpix cell vertices themselves
    """

    # target_coords_lens = [len(t) for t in target_coords]
    tcs = [
        s2tor3(
            torch.deg2rad(90.0 - t[..., geoinfo_offset].to(torch.float64)),
            torch.deg2rad(180.0 + t[..., geoinfo_offset + 1].to(torch.float64)),
        )
        if len(t) > 0
        else torch.tensor([])
        for t in target_coords
    ]
    target_coords = torch.cat(target_coords)
    if target_coords.shape[0] == 0:
        return torch.tensor([])

    verts00 = healpix_verts(hlc, 0.0, 0.0)
    verts10 = healpix_verts(hlc, 1.0, 0.0)
    verts11 = healpix_verts(hlc, 1.0, 1.0)
    verts01 = healpix_verts(hlc, 0.0, 1.0)
    vertsmm = healpix_verts(hlc, 0.5, 0.5)

    a = torch.zeros(
        [*target_coords.shape[:-1], (target_coords.shape[-1] - 2) + 5 * (3 * 5) + 3 * 8]
    )
    a[..., :geoinfo_offset] = target_coords[..., :geoinfo_offset]
    ref = torch.tensor([1.0, 0.0, 0.0])

    zi = 0
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords(hlc, tcs, 0.0, 0.0)
    )
    a = add_local_vert_coords(hlc, a, verts10, tcs, 3, 0.0, 0.0, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, verts11, tcs, 6, 0.0, 0.0, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, verts01, tcs, 9, 0.0, 0.0, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, vertsmm, tcs, 12, 0.0, 0.0, geoinfo_offset)

    zi = 15
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords(hlc, tcs, 1.0, 0.0)
    )
    a = add_local_vert_coords(hlc, a, verts00, tcs, 18, 1.0, 0.0, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, verts11, tcs, 21, 1.0, 0.0, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, verts01, tcs, 24, 1.0, 0.0, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, vertsmm, tcs, 27, 1.0, 0.0, geoinfo_offset)

    zi = 30
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords(hlc, tcs, 1.0, 1.0)
    )
    a = add_local_vert_coords(hlc, a, verts00, tcs, 33, 1.0, 1.0, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, verts10, tcs, 36, 1.0, 1.0, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, verts01, tcs, 39, 1.0, 1.0, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, vertsmm, tcs, 42, 1.0, 1.0, geoinfo_offset)

    zi = 45
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords(hlc, tcs, 0.0, 1.0)
    )
    a = add_local_vert_coords(hlc, a, verts00, tcs, 48, 0.0, 1.0, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, verts11, tcs, 51, 0.0, 1.0, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, verts10, tcs, 54, 0.0, 1.0, geoinfo_offset)
    # a = add_local_vert_coords( hlc, a, verts10, tcs, 51, 0.0, 1.0, geoinfo_offset)
    # a = add_local_vert_coords( hlc, a, verts01, tcs, 54, 0.0, 1.0, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, vertsmm, tcs, 57, 0.0, 1.0, geoinfo_offset)

    zi = 60
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords(hlc, tcs, 0.5, 0.5)
    )
    a = add_local_vert_coords(hlc, a, verts00, tcs, 63, 0.5, 0.5, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, verts10, tcs, 66, 0.5, 0.5, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, verts11, tcs, 69, 0.5, 0.5, geoinfo_offset)
    a = add_local_vert_coords(hlc, a, verts01, tcs, 72, 0.5, 0.5, geoinfo_offset)

    # add centroids to neighboring cells wrt to cell center
    num_healpix_cells = 12 * 4**hlc
    with warnings.catch_warnings(action="ignore"):
        temp = hp.neighbours(np.arange(num_healpix_cells), 2**hlc, order="nested").transpose()
    # fix missing nbors with references to self
    for i, row in enumerate(temp):
        temp[i][row == -1] = i
    # coords of centers of all centers
    lons, lats = hp.healpix_to_lonlat(
        np.arange(0, num_healpix_cells), 2**hlc, dx=0.5, dy=0.5, order="nested"
    )
    ctrs = s2tor3(torch.from_numpy(np.pi / 2.0 - lats.value), torch.from_numpy(lons.value))
    ctrs = ctrs[temp.flatten()].reshape((num_healpix_cells, 8, 3)).transpose(1, 0)
    # local coords with respect to all neighboring centers
    tcs_ctrs = torch.cat([ref - torch.cat(locs_to_ctr_coords(c, tcs)) for c in ctrs], -1)
    zi = 75
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + (3 * 8))] = tcs_ctrs

    # remaining geoinfos (zenith angle etc)
    zi = 99
    a[..., (geoinfo_offset + zi) :] = target_coords[..., (geoinfo_offset + 2) :]

    return a


####################################################################################################
def get_target_coords_local_fast(hlc, target_coords, geoinfo_offset):
    """Generate local coordinates for target coords w.r.t healpix cell vertices and
    and for healpix cell vertices themselves
    """

    # target_coords_lens = [len(t) for t in target_coords]
    tcs = [
        s2tor3(
            torch.deg2rad(90.0 - t[..., geoinfo_offset].to(torch.float64)),
            torch.deg2rad(180.0 + t[..., geoinfo_offset + 1].to(torch.float64)),
        )
        if len(t) > 0
        else torch.tensor([])
        for t in target_coords
    ]
    target_coords = torch.cat(target_coords)
    if target_coords.shape[0] == 0:
        return torch.tensor([])

    verts00, verts00_Rs = healpix_verts_rots(hlc, 0.0, 0.0)
    verts10, verts10_Rs = healpix_verts_rots(hlc, 1.0, 0.0)
    verts11, verts11_Rs = healpix_verts_rots(hlc, 1.0, 1.0)
    verts01, verts01_Rs = healpix_verts_rots(hlc, 0.0, 1.0)
    vertsmm, vertsmm_Rs = healpix_verts_rots(hlc, 0.5, 0.5)

    a = torch.zeros(
        [*target_coords.shape[:-1], (target_coords.shape[-1] - 2) + 5 * (3 * 5) + 3 * 8]
    )
    # a = torch.zeros( [*target_coords.shape[:-1], (target_coords.shape[-1]-2) + 5*(3*5) + 3*8])
    # a = torch.zeros( [*target_coords.shape[:-1], 148]) #(target_coords.shape[-1]-2) + 5*(3*5) + 3*8])
    a[..., :geoinfo_offset] = target_coords[..., :geoinfo_offset]
    ref = torch.tensor([1.0, 0.0, 0.0])

    zi = 0
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords_ctrs(verts00_Rs, tcs)
    )
    verts = torch.stack([verts10, verts11, verts01, vertsmm])
    a = add_local_vert_coords_ctrs2(verts00_Rs, verts, tcs, a, 3, geoinfo_offset)

    zi = 15
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords_ctrs(verts10_Rs, tcs)
    )
    verts = torch.stack([verts00, verts11, verts01, vertsmm])
    a = add_local_vert_coords_ctrs2(verts10_Rs, verts, tcs, a, 18, geoinfo_offset)

    zi = 30
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords_ctrs(verts11_Rs, tcs)
    )
    verts = torch.stack([verts00, verts10, verts01, vertsmm])
    a = add_local_vert_coords_ctrs2(verts11_Rs, verts, tcs, a, 33, geoinfo_offset)

    zi = 45
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords_ctrs(verts01_Rs, tcs)
    )
    verts = torch.stack([verts00, verts11, verts10, vertsmm])
    a = add_local_vert_coords_ctrs2(verts01_Rs, verts, tcs, a, 48, geoinfo_offset)

    zi = 60
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords_ctrs(vertsmm_Rs, tcs)
    )
    verts = torch.stack([verts00, verts10, verts11, verts01])
    a = add_local_vert_coords_ctrs2(vertsmm_Rs, verts, tcs, a, 63, geoinfo_offset)

    # add local coords wrt to center of neighboring cells
    # (since the neighbors are used in the prediction)
    num_healpix_cells = 12 * 4**hlc
    with warnings.catch_warnings(action="ignore"):
        temp = hp.neighbours(np.arange(num_healpix_cells), 2**hlc, order="nested").transpose()
    # fix missing nbors with references to self
    for i, row in enumerate(temp):
        temp[i][row == -1] = i
    nctrs = vertsmm[temp.flatten()].reshape((num_healpix_cells, 8, 3)).transpose(1, 0)
    # local coords with respect to all neighboring centers
    tcs_ctrs = torch.cat([ref - torch.cat(locs_to_ctr_coords(c, tcs)) for c in nctrs], -1)
    zi = 75
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + (3 * 8))] = tcs_ctrs
    # a = add_local_vert_coords_ctrs2( vertsmm_Rs, nctrs, tcs, a, 99, geoinfo_offset)

    # remaining geoinfos (zenith angle etc)
    # zi=99+3*8;
    zi = 99
    # assert target_coords.shape[-1] + zi < a.shape[-1]
    a[..., (geoinfo_offset + zi) :] = target_coords[..., (geoinfo_offset + 2) :]

    return a


####################################################################################################
def get_target_coords_local_ffast(hlc, target_coords, geoinfo_offset, verts_Rs, verts_local, nctrs):
    """Generate local coordinates for target coords w.r.t healpix cell vertices and
    and for healpix cell vertices themselves
    """

    # target_coords_lens = [len(t) for t in target_coords]
    tcs = [
        s2tor3(
            torch.deg2rad(90.0 - t[..., geoinfo_offset]),
            torch.deg2rad(180.0 + t[..., geoinfo_offset + 1]),
        )
        if len(t) > 0
        else torch.tensor([])
        for t in target_coords
    ]
    target_coords = torch.cat(target_coords)
    if target_coords.shape[0] == 0:
        return torch.tensor([])

    verts00_Rs, verts10_Rs, verts11_Rs, verts01_Rs, vertsmm_Rs = verts_Rs

    a = torch.zeros(
        [*target_coords.shape[:-1], (target_coords.shape[-1] - 2) + 5 * (3 * 5) + 3 * 8]
    )
    a[..., :geoinfo_offset] = target_coords[..., :geoinfo_offset]
    ref = torch.tensor([1.0, 0.0, 0.0])

    tcs_lens = torch.tensor([tt.shape[0] for tt in tcs], dtype=torch.int32)
    tcs_lens_mask = tcs_lens > 0
    tcs_lens = tcs_lens[tcs_lens_mask]

    vls = torch.cat(
        [vl.repeat([tt, 1, 1]) for tt, vl in zip(tcs_lens, verts_local[tcs_lens_mask], strict=False)], 0
    )
    vls = vls.transpose(0, 1)

    zi = 0
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords_ctrs(verts00_Rs, tcs)
    )
    zi = 3
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[0]

    zi = 15
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords_ctrs(verts10_Rs, tcs)
    )
    zi = 18
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[1]

    zi = 30
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords_ctrs(verts11_Rs, tcs)
    )
    zi = 33
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[2]

    zi = 45
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords_ctrs(verts01_Rs, tcs)
    )
    zi = 48
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[3]

    zi = 60
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - torch.cat(
        locs_to_cell_coords_ctrs(vertsmm_Rs, tcs)
    )
    zi = 63
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[4]

    tcs_ctrs = torch.cat([ref - torch.cat(locs_to_ctr_coords(c, tcs)) for c in nctrs], -1)
    zi = 75
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + (3 * 8))] = tcs_ctrs
    # a = add_local_vert_coords_ctrs2( vertsmm_Rs, nctrs, tcs, a, 99, geoinfo_offset)

    # remaining geoinfos (zenith angle etc)
    # zi=99+3*8;
    zi = 99
    a[..., (geoinfo_offset + zi) :] = target_coords[..., (geoinfo_offset + 2) :]

    return a


####################################################################################################
if __name__ == "__main__":
    vecs = torch.nn.functional.normalize(torch.rand((10, 3), dtype=torch.float64))
    Rs = vecs_to_rots(vecs)
    res = torch.stack([torch.matmul(R, vec) for R, vec in zip(Rs, vecs, strict=False)])
    ref = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    passed = torch.allclose(res, ref)
    print(f"passed = {passed}")
