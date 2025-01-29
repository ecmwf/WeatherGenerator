
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
