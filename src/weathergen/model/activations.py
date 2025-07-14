# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, chunk_dim=1):
        super(SwiGLU, self).__init__()
        self.chunk_dim = chunk_dim

    def forward(self, x):
        x_gate, x_latent = x.chunk(2, dim=self.chunk_dim)
        return F.silu(x_gate) * x_latent

