# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from weathergen.train.utils import NoOpGradScaler


def test_noop_grad_scaler_runs_an_unscaled_step():
    """With grad_scaling off the training step calls the scaler API like GradScaler's."""
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=0.5)
    scaler = NoOpGradScaler()

    loss = (2.0 * param).sum()
    scaled = scaler.scale(loss)
    assert scaled is loss
    scaled.backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()

    assert param.grad is not None and param.grad.item() == 2.0
    assert param.item() == 0.0
