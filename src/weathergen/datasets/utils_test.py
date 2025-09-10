import torch
from torch import Tensor, tensor

from weathergen.datasets.utils import s2tor3, tcs_optimized


def _tcs_simpled(target_coords: list[Tensor]) -> tuple[list[Tensor], Tensor]:
    tcs = [
        (
            s2tor3(
                torch.deg2rad(90.0 - t[..., 0]),
                torch.deg2rad(180.0 + t[..., 1]),
            )
            if len(t) > 0
            else torch.tensor([])
        )
        for t in target_coords
    ]
    cat_target_coords = torch.cat(target_coords)
    return tcs, cat_target_coords


def test_tcs():
    target_coords = [
        tensor(
            [[2.3377, -135.0000], [1.4026, -135.4545], [1.4026, -134.5455], [0.4675, -135.0000]]
        ),
        tensor(
            [[3.2727, -133.6082], [2.3377, -134.0816], [2.3377, -133.1633], [1.4026, -133.6364]]
        ),
    ]
    tcs_ref, cat_tcs_ref = _tcs_simpled(target_coords)
    tcs_opt, cat_tcs_opt = tcs_optimized(target_coords)
    assert len(tcs_ref) == len(tcs_opt)
    torch.testing.assert_close(cat_tcs_ref, cat_tcs_opt)
    torch.testing.assert_close(tcs_ref, tcs_opt, atol=1e-8, rtol=1e-5)
