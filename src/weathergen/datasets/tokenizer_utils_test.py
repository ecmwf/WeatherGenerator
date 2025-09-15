import torch
from torch import Tensor, tensor

from weathergen.datasets.tokenizer_utils import CoordNormalizer, _coords_local, r3tos2

_pos3r = tensor(
    [
        [-1.2492e-02, -1.0921e-09, 9.9992e-01],
        [-1.1881e-02, 9.9992e-01, -3.8603e-03],
        [-1.0106e-02, -7.3428e-03, 9.9992e-01],
        [-7.3428e-03, -1.0106e-02, 9.9992e-01],
        [-3.8603e-03, -1.1881e-02, 9.9992e-01],
        [1.4897e-10, -1.2492e-02, 9.9992e-01],
        [3.8603e-03, -1.1881e-02, 9.9992e-01],
        [7.3428e-03, -1.0106e-02, 9.9992e-01],
        [1.0106e-02, -7.3428e-03, 9.9992e-01],
        [1.1881e-02, -3.8603e-03, 9.9992e-01],
        [1.2492e-02, 0.0000e00, 9.9992e-01],
        [1.1881e-02, 3.8603e-03, 9.9992e-01],
        [1.0106e-02, 7.3428e-03, 9.9992e-01],
        [7.3428e-03, 1.0106e-02, 9.9992e-01],
        [3.8603e-03, 1.1881e-02, 9.9992e-01],
        [-5.4606e-10, 1.2492e-02, 9.9992e-01],
        [-3.8603e-03, 1.1881e-02, 9.9992e-01],
        [-7.3428e-03, 1.0106e-02, 9.9992e-01],
        [-1.0106e-02, 7.3428e-03, 9.9992e-01],
    ]
)

_idxs_ord = [
    tensor([6, 4, 5, 7, 0, 0, 0, 0]),
    tensor([1, 2, 3, 8, 0, 0, 0, 0]),
    tensor([9, 10, 11, 0, 0, 0, 0, 0]),
]

_hpy_verts_rots = tensor(
    [
        [[0.7070, 0.7070, 0.0208], [-0.7070, 0.7072, -0.0086], [-0.0208, -0.0086, 0.9997]],
        [[0.6889, 0.7236, 0.0417], [-0.7236, 0.6900, -0.0179], [-0.0417, -0.0179, 0.9990]],
        [[0.7236, 0.6889, 0.0417], [-0.6889, 0.7246, -0.0167], [-0.0417, -0.0167, 0.9990]],
    ]
)


def simple_coords_local(
    posr3: Tensor, hpy_verts_rots: Tensor, idxs_ord: list[Tensor], n_coords: CoordNormalizer
) -> list[Tensor]:
    fp32 = torch.float32
    posr3 = torch.cat([torch.zeros_like(posr3[0]).unsqueeze(0), posr3])  # prepend zero
    """Compute simple local coordinates for a set of 3D positions on the unit sphere."""
    return [
        n_coords(r3tos2(torch.matmul(R, posr3[idxs].transpose(1, 0)).transpose(1, 0)).to(fp32))
        for R, idxs in zip(hpy_verts_rots, idxs_ord, strict=True)
    ]


def test_coords_local():
    n_coords = lambda x: x
    coords_local = simple_coords_local(_pos3r, _hpy_verts_rots, _idxs_ord, n_coords)
    coords_local_ref = _coords_local(_pos3r, _hpy_verts_rots, _idxs_ord, n_coords)
    torch.testing.assert_close(coords_local, coords_local_ref, atol=1e-6, rtol=0)


test_coords_local()
