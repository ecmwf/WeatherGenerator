# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from omegaconf import OmegaConf

from weathergen.common.config import _check_diffusion
from weathergen.model.diffusion import validate_diffusion_config
from weathergen.train.loss_modules import LossFlowMatching, LossLatentDiffusion


@pytest.mark.parametrize(
    "diff_cf",
    [
        {"enabled": True},  # no type
        {"type": "ddpm"},
        {"type": "edm", "edm": {}, "flow_matching": {}},  # inactive section present
        {"type": "flow_matching", "sampler": {"num_steps": 10, "rho": 7}},  # misplaced key
    ],
)
def test_validate_diffusion_config_rejects(diff_cf):
    with pytest.raises(AssertionError):
        validate_diffusion_config(OmegaConf.create(diff_cf))


def test_validate_diffusion_config():
    diff_cf = {"type": "flow_matching", "flow_matching": {}, "sampler": {"num_steps": 10}}
    assert validate_diffusion_config(OmegaConf.create(diff_cf)) == "flow_matching"


def test_legacy_diffusion_config_is_migrated():
    legacy = {
        "diffusion": {
            "enabled": True,
            "conditioning": "forecast",
            "sigma_data": 0.78,
            "edm_preconditioning": True,
            "sampler": {"num_steps": 10, "rho": 7, "s_churn": 1.0},
        }
    }
    cf = _check_diffusion(OmegaConf.create(legacy))
    expected = {
        "enabled": True,
        "conditioning": "forecast",
        "type": "edm",
        "sampler": {"num_steps": 10},
        "edm": {"sigma_data": 0.78, "preconditioning": True, "sampler": {"rho": 7, "s_churn": 1.0}},
    }
    assert OmegaConf.to_container(cf.diffusion) == expected
    assert validate_diffusion_config(cf.diffusion) == "edm"
    # configs in the current format are left unchanged
    assert _check_diffusion(cf) == cf


@pytest.mark.parametrize(
    "loss_cls,diffusion_type",
    [(LossLatentDiffusion, "flow_matching"), (LossFlowMatching, "edm")],
)
def test_loss_requires_matching_diffusion_type(loss_cls, diffusion_type):
    cf = OmegaConf.create(
        {"diffusion": {"type": diffusion_type, diffusion_type: {"sigma_data": 1.0}}}
    )
    with pytest.raises(AssertionError):
        loss_cls(cf, None, "train", "cpu", mse={})
