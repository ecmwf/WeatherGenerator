# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from torch.utils.checkpoint import checkpoint

class StreamEmbedLinear( torch.nn.Module) :

  def __init__(self, dim_in, dim_out) :
    '''Constructor'''
 
    super( StreamEmbedLinear, self).__init__()

    self.layer = torch.nn.Linear( dim_in, dim_out)

  def forward( self, x) :

    # x = checkpoint( self.layer, x.flatten( -2, -1), use_reentrant=True)
    x = self.layer( x.flatten( -2, -1))

    return x
