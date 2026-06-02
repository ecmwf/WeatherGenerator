from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.train.loss_modules.loss_module_physical import LossPhysical
from weathergen.train.utils import TRAIN, VAL


@pytest.mark.parametrize("stage", [TRAIN, VAL])
def test_compute_loss_keeps_logged_avg_for_valid_correspondences(stage):
    cf = OmegaConf.create(
        {
            "streams": [
                {
                    "name": "SurfaceCombined",
                    "train_target_channels": ["obs"],
                    "val_target_channels": ["obs"],
                    "loss_weight": 1.0,
                }
            ]
        }
    )
    mode_cfg = OmegaConf.create({"forecast": {}})
    loss_module = LossPhysical(cf, mode_cfg, stage, "cpu", mse={})

    preds = SimpleNamespace(
        physical=[
            {
                "SurfaceCombined": [
                    torch.tensor([[[2.0]]], dtype=torch.float32),
                    torch.tensor([[[5.0]]], dtype=torch.float32),
                ]
            }
        ]
    )
    targets = SimpleNamespace(
        output_idxs=[0],
        physical=[
            {
                "SurfaceCombined": {
                    "target": [
                        torch.tensor([[1.0]], dtype=torch.float32),
                        torch.tensor([[float("nan")]], dtype=torch.float32),
                    ],
                    "target_coords": [
                        torch.tensor([[0.0, 0.0]], dtype=torch.float32),
                        torch.tensor([[0.0, 0.0]], dtype=torch.float32),
                    ],
                    "target_times": [np.array([0]), np.array([0])],
                    "target_metda_data": [
                        {"SurfaceCombined": SimpleNamespace(global_params={"idx": 0})},
                        {"SurfaceCombined": SimpleNamespace(global_params={"idx": 1})},
                    ],
                    "is_spoof": [False, False],
                }
            }
        ],
    )
    metadata = (
        None,
        [
            SimpleNamespace(global_params={"correspondence": 0, "loss": ["mse"]}),
            SimpleNamespace(global_params={"correspondence": 1, "loss": ["mse"]}),
        ],
        None,
        None,
    )

    loss_values = loss_module.compute_loss(preds, targets, metadata)

    assert loss_values.loss.item() == pytest.approx(1.0)
    assert (
        loss_values.losses_all["SurfaceCombined"]["mse"]["avg"].item()
        == pytest.approx(1.0)
    )
    assert (
        loss_values.losses_all["SurfaceCombined"]["mse"]["obs"]["0"].item()
        == pytest.approx(1.0)
    )