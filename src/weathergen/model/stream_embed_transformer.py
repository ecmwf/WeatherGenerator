# (C) Copyright 2024 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math
import code

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from weathergen.model.attention import MultiSelfAttentionHead
from weathergen.model.mlp import MLP
from weathergen.model.norms import RMSNorm
from  weathergen.model.positional_encoding import positional_encoding_harmonic
from weathergen.model.positional_encoding import positional_encoding_harmonic_coord

from weathergen.model.utils import get_num_parameters

class StreamEmbedTransformer( torch.nn.Module) :

  def __init__(self, mode, num_tokens, token_size, num_channels, dim_embed, dim_out,
                     num_blocks, num_heads, norm_type = 'LayerNorm', embed_size_centroids=64,
                     unembed_mode = 'full') :
    '''Constructor

      unembed_mode : { 'full' , 'block'}
        full : monolithic (and correspondingly large) unembedding network that maps from
               (num_tokens x dim_embed) to dim_out, allowing for mixing between channels/columns
        block : per-channel/column unembedding network (which is hence a block-sparse form of full)
    '''
 
    super( StreamEmbedTransformer, self).__init__()

    self.num_tokens = num_tokens
    self.num_channels = num_channels
    self.dim_in = token_size if mode=='channels' else num_channels
    self.dim_embed = dim_embed
    self.dim_out = dim_out
    self.num_blocks = num_blocks
    self.num_heads = num_heads
    self.embed_size_centroids = embed_size_centroids
    self.unembed_mode = unembed_mode

    norm = torch.nn.LayerNorm if norm_type == 'LayerNorm' else RMSNorm

    self.embed = torch.nn.Linear( self.dim_in, self.dim_embed)
    
    self.layers = torch.nn.ModuleList()
    for _ in range( self.num_blocks) :
      self.layers.append( MultiSelfAttentionHead( self.dim_embed, self.num_heads, dropout_rate=0.1,
                                                  with_qk_lnorm=True, with_flash=True))
      self.layers.append( MLP( self.dim_embed, self.dim_embed, hidden_factor=2, dropout_rate=0.1,
                               with_residual=True))

    if mode == 'channels' :

      if self.unembed_mode == 'full' :
        self.ln_final = norm( num_channels*self.dim_embed)
        self.unembed = torch.nn.Linear( num_channels*self.dim_embed,
                                        self.num_tokens*self.dim_out - embed_size_centroids)

      elif self.unembed_mode == 'block' :
        # modify embed_size_centroids to ensure no additional padding is needed
        rem = (self.num_tokens*self.dim_out - embed_size_centroids) % num_channels
        embed_size_centroids += rem
        dim_out = (self.num_tokens*self.dim_out - embed_size_centroids) // num_channels
        Linear = torch.nn.Linear
        self.unembed = torch.nn.ModuleList([Linear(dim_embed,dim_out) for _ in range(num_channels)])
        self.ln_final = torch.nn.ModuleList( [norm( dim_embed) for _ in range(num_channels)])

      else :
        assert False

      self.forward = self.forward_channels

    elif mode == 'columns' :
      assert self.unembed_mode == 'block' # only supported mode at the moment
      # padding needed if the unembedded columns cannot be concatenated to dim_out (e.g GPSRO)
      self.pad = (self.dim_out-embed_size_centroids) % token_size
      self.out_pad = torch.nn.Parameter( torch.zeros( self.pad))
      self.unembed = torch.nn.Linear( self.dim_embed, 
                                self.num_tokens * ((self.dim_out-embed_size_centroids)//token_size))
      self.ln_final = norm( dim_out)
      self.forward = self.forward_columns

    else :
      assert False

    self.dropout_final = torch.nn.Dropout( 0.1)
    self.embed_centroids = torch.nn.Linear( 5, embed_size_centroids)

  def forward_channels( self, x_in, centroids) :

    peh = positional_encoding_harmonic

    # embed provided input data
    x = peh( checkpoint( self.embed, x_in.transpose( -2, -1), use_reentrant=False))

    for layer in self.layers :
      x = checkpoint( layer, x, use_reentrant=False)

    # read out
    if self.unembed_mode == 'full' :
      out = checkpoint( self.unembed, self.ln_final( x.flatten( -2,-1)), use_reentrant=False)
    elif self.unembed_mode == 'block' :
      out = [checkpoint( ue, ln(x[:,i]), use_reentrant=False)
                                      for i,(ue,ln) in enumerate(zip(self.unembed,self.ln_final))]
      out = torch.stack( out, dim=1).flatten( -2, -1)
    else :
      assert False

    # append centroids
    if self.embed_size_centroids > 0 :
      out = torch.cat([ out, self.embed_centroids(centroids)], -1)
    # final reshape
    out = self.dropout_final( out.reshape(-1,self.num_tokens,self.dim_out))

    return out

  # @torch.compile( dynamic=True)
  def forward_columns( self, x_in, centroids) :

    # embed provided input data
    x = positional_encoding_harmonic( checkpoint( self.embed, x_in, use_reentrant=False))

    for layer in self.layers :
      x = checkpoint( layer, x, use_reentrant=False)

    # append centroids
    # unembed and reshape
    out = checkpoint( self.unembed, x, use_reentrant=False)
    out = out.flatten(-2,-1).reshape(x.shape[0],self.num_tokens,-1)
    # TODO: unsqueeze will not work with num_tokens > 1
    out = torch.cat( [out, self.embed_centroids(centroids).unsqueeze(1)], -1)
    # pad to uniform dim_out (that has to be uniform across streams) 
    if self.pad > 0 :
      out = torch.cat( (out, self.out_pad.repeat( (x.shape[0],self.num_tokens,1))), -1)
    # also encode centroids with overlayed positional encoding
    out = self.dropout_final( self.ln_final( out))

    return out
