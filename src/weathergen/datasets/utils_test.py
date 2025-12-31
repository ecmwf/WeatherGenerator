import torch
from torch import tensor

from weathergen.datasets.utils import (
    locs_to_cell_coords_ctrs,
    locs_to_ctr_coords,
    vecs_to_rots,
)


def _locs_to_cell_coords_ctrs(
    healpix_centers_rots: torch.Tensor, locs: list[torch.Tensor]
) -> torch.Tensor:
    return torch.cat(
        [
            torch.matmul(R, s.transpose(-1, -2)).transpose(-2, -1)
            if len(s) > 0
            else torch.tensor([])
            for _, (R, s) in enumerate(zip(healpix_centers_rots, locs, strict=False))
        ]
    )


def test_locs_to_cell_coords_ctrs():
    locs = [
        tensor(
            [
                [0.7235, -0.6899, -0.0245],
                [0.7178, -0.6951, -0.0408],
                [0.7288, -0.6835, -0.0408],
                [0.7229, -0.6886, -0.0571],
            ]
        ),
        tensor(
            [
                [0.6899, -0.7235, -0.0245],
                [0.6835, -0.7288, -0.0408],
                [0.6951, -0.7178, -0.0408],
                [0.6886, -0.7229, -0.0571],
            ]
        ),
        tensor([]),
    ]
    hp_centers_rots = tensor(
        [
            [
                [7.0711e-01, 7.0711e-01, 6.1232e-17],
                [-7.0711e-01, 7.0711e-01, -2.5363e-17],
                [-6.1232e-17, -2.5363e-17, 1.0000e00],
            ],
            [
                [6.8939e-01, 7.2409e-01, 2.0833e-02],
                [-7.2409e-01, 6.8965e-01, -8.9294e-03],
                [-2.0833e-02, -8.9294e-03, 9.9974e-01],
            ],
            [
                [7.2409e-01, 6.8939e-01, 2.0833e-02],
                [-6.8939e-01, 7.2434e-01, -8.3304e-03],
                [-2.0833e-02, -8.3304e-03, 9.9975e-01],
            ],
            [
                [7.0649e-01, 7.0649e-01, 4.1667e-02],
                [-7.0649e-01, 7.0751e-01, -1.7250e-02],
                [-4.1667e-02, -1.7250e-02, 9.9898e-01],
            ],
        ]
    )
    torch.testing.assert_close(
        _locs_to_cell_coords_ctrs(hp_centers_rots, locs),
        locs_to_cell_coords_ctrs(hp_centers_rots, locs),
    )


# def _tcs_simpled(target_coords: list[Tensor]) -> tuple[list[Tensor], Tensor]:
#     tcs = [
#         (
#             s2tor3(
#                 torch.deg2rad(90.0 - t[..., 0]),
#                 torch.deg2rad(180.0 + t[..., 1]),
#             )
#             if len(t) > 0
#             else torch.tensor([])
#         )
#         for t in target_coords
#     ]
#     cat_target_coords = torch.cat(target_coords)
#     return tcs, cat_target_coords


# def test_tcs():
#     target_coords = [
#         tensor(
#             [[2.3377, -135.0000], [1.4026, -135.4545], [1.4026, -134.5455], [0.4675, -135.0000]]
#         ),
#         tensor(
#             [[3.2727, -133.6082], [2.3377, -134.0816], [2.3377, -133.1633], [1.4026, -133.6364]]
#         ),
#     ]
#     tcs_ref, cat_tcs_ref = _tcs_simpled(target_coords)
#     tcs_opt, cat_tcs_opt = tcs_optimized(target_coords)
#     assert len(tcs_ref) == len(tcs_opt)
#     torch.testing.assert_close(cat_tcs_ref, cat_tcs_opt)
#     torch.testing.assert_close(tcs_ref, tcs_opt, atol=1e-8, rtol=1e-5)


def _locs_to_ctr_coords(ctrs_r3, locs: list[torch.Tensor]) -> list[torch.Tensor]:
    ctrs_rots = vecs_to_rots(ctrs_r3).to(torch.float32)

    ## express each centroid in local coordinates w.r.t to healpix center
    #  by rotating center to origin
    return [
        (
            torch.matmul(R, s.transpose(-1, -2)).transpose(-2, -1)
            if len(s) > 0
            else torch.zeros([0, 3])
        )
        for i, (R, s) in enumerate(zip(ctrs_rots, locs, strict=False))
    ]


def test_locs_to_ctr_coords():
    locs = [
        tensor(
            [
                [0.7235, -0.6899, -0.0245],
                [0.7178, -0.6951, -0.0408],
                [0.7288, -0.6835, -0.0408],
                [0.7229, -0.6886, -0.0571],
            ]
        ),
        tensor(
            [
                [0.6899, -0.7235, -0.0245],
                [0.6835, -0.7288, -0.0408],
                [0.6951, -0.7178, -0.0408],
                [0.6886, -0.7229, -0.0571],
            ]
        ),
        tensor([]),
    ]
    ctrs_r3 = tensor(
        [
            [7.2425e-01, 6.8954e-01, 6.1232e-17],
            [7.0695e-01, 7.0695e-01, 2.0833e-02],
            [7.4079e-01, 6.7141e-01, 2.0833e-02],
        ]
    )
    torch.testing.assert_close(
        locs_to_ctr_coords(ctrs_r3, locs),
        _locs_to_ctr_coords(ctrs_r3, locs),
    )
