import torch

from weathergen.train.wd_scheduler import WeightDecayScheduler


def _optimizer():
    p1, p2 = torch.nn.Parameter(torch.ones(2, 2)), torch.nn.Parameter(torch.ones(2))
    groups = [{"params": [p1], "weight_decay": 0.1}, {"params": [p2], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=1e-3)


def test_linear_ramp_scales_groups():
    opt = _optimizer()
    sched = WeightDecayScheduler([opt], 0.1, {"policy": "linear"}, n_steps=100, istep=0)
    assert opt.param_groups[0]["weight_decay"] == 0.0
    sched.step(50)
    assert abs(opt.param_groups[0]["weight_decay"] - 0.05) < 1e-12
    assert opt.param_groups[1]["weight_decay"] == 0.0
    sched.step(200)
    assert abs(opt.param_groups[0]["weight_decay"] - 0.1) < 1e-12


def test_cosine_and_ramp_fraction():
    opt = _optimizer()
    cfg = {"policy": "cosine", "wd_start": 0.01, "ramp_fraction": 0.5}
    sched = WeightDecayScheduler([opt], 0.1, cfg, n_steps=100, istep=0)
    assert abs(sched.get_wd(0) - 0.01) < 1e-12
    assert abs(sched.get_wd(25) - 0.055) < 1e-12
    assert abs(sched.get_wd(50) - 0.1) < 1e-12
    assert abs(sched.get_wd(90) - 0.1) < 1e-12
