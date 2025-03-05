# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch


#########################################
def get_num_parameters(block):
    nps = filter(lambda p: p.requires_grad, block.parameters())
    return sum([torch.prod(torch.tensor(p.size())) for p in nps])


#########################################
def freeze_weights(block):
    for p in block.parameters():
        p.requires_grad = False
