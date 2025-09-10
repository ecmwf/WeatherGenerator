import torch
from torch import tensor

from weathergen.datasets.utils import locs_to_ctr_coords, vecs_to_rots


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
    ctrs_r3 = []
    torch.testing.assert_close(
        locs_to_ctr_coords(ctrs_r3, locs),
        _locs_to_ctr_coords(ctrs_r3, locs),
    )
