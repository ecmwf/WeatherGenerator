# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
import time
import code
from functools import partial

import numpy as np

from flash_attn import flash_attn_varlen_func, flash_attn_func

from weathergen.model.norms import RMSNorm
from weathergen.model.norms import AdaLayerNorm

from torch.nn.attention.flex_attention import flex_attention, create_mask, create_block_mask

####################################################################################################
class MultiSelfAttentionHead_Varlen(torch.nn.Module):

  #########################################
  def __init__(self, dim_embed, num_heads, dim_head_proj=None, dropout_rate=0., 
                     with_qk_lnorm=True, with_flash=True, norm_type = 'LayerNorm', 
                     softcap=0.0, dim_aux=None) :
    
    super(MultiSelfAttentionHead_Varlen, self).__init__()

    self.num_heads = num_heads
    self.dropout_rate = dropout_rate
    self.with_flash = with_flash
    self.softcap = softcap

    assert 0 == dim_embed % num_heads
    self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

    if norm_type == 'LayerNorm' :
      norm = partial( torch.nn.LayerNorm, elementwise_affine=False) 
    else :
      norm = RMSNorm

    if dim_aux is not None :
      self.lnorm = AdaLayerNorm( dim_embed, dim_aux)
    else :
      self.lnorm = norm( dim_embed)
    self.proj_heads_q = torch.nn.Linear( dim_embed, num_heads*self.dim_head_proj, bias = False)
    self.proj_heads_k = torch.nn.Linear( dim_embed, num_heads*self.dim_head_proj, bias = False)
    self.proj_heads_v = torch.nn.Linear( dim_embed, num_heads*self.dim_head_proj, bias = False)
    self.proj_out = torch.nn.Linear( dim_embed, dim_embed, bias = False)
    self.dropout = torch.nn.Dropout( p=dropout_rate) if dropout_rate > 0. else torch.nn.Identity()

    lnorm = norm if with_qk_lnorm else torch.nn.Identity
    self.lnorm_q = lnorm( self.dim_head_proj)
    self.lnorm_k = lnorm( self.dim_head_proj)
    
    assert with_flash, 'Only flash attention supported at the moment'

  #########################################
  def forward( self, x, x_lens, ada_ln_aux=None) :

    x_in = x
    x = self.lnorm( x) if ada_ln_aux is None else self.lnorm( x, ada_ln_aux)

    # project onto heads and q,k,v and ensure these are 4D tensors as required for flash attention
    s = [ x.shape[0], self.num_heads, -1]
    qs = self.lnorm_q( self.proj_heads_q( x).reshape(s)).to(torch.float16)
    ks = self.lnorm_k( self.proj_heads_k( x).reshape(s)).to(torch.float16)
    vs = self.proj_heads_v( x).reshape(s)

    cum_x_lens = torch.cumsum( x_lens, 0, dtype=torch.int32)
    # ordering of tensors (seq, heads, embed) (which differs from torch's flash attention implt)
    outs = flash_attn_varlen_func( qs, ks, vs, cum_x_lens, cum_x_lens, x_lens.max(), x_lens.max(),
                                    softcap=self.softcap, dropout_p=self.dropout_rate)

    # return x_in + self.dropout( self.proj_out( outs.flatten( -2, -1)) )
    return x_in + self.proj_out( outs.flatten( -2, -1))

####################################################################################################
class MultiSelfAttentionHead_Varlen_Flex(torch.nn.Module):

  #########################################
  def __init__(self, dim_embed, num_heads, dim_head_proj=None, dropout_rate=0.,
                     with_qk_lnorm=True, with_flash=True, norm_type = 'LayerNorm',
                     softcap=0.0) :

    super(MultiSelfAttentionHead_Varlen_Flex, self).__init__()

    self.num_heads = num_heads
    self.with_flash = with_flash
    self.softcap = softcap

    assert 0 == dim_embed % num_heads
    self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

    if norm_type == 'LayerNorm' :
      norm = partial( torch.nn.LayerNorm, elementwise_affine=False) 
    else :
      norm = RMSNorm

    self.lnorm = norm( dim_embed)
    self.proj_heads_q = torch.nn.Linear( dim_embed, num_heads*self.dim_head_proj, bias = False)
    self.proj_heads_k = torch.nn.Linear( dim_embed, num_heads*self.dim_head_proj, bias = False)
    self.proj_heads_v = torch.nn.Linear( dim_embed, num_heads*self.dim_head_proj, bias = False)
    self.proj_out = torch.nn.Linear( dim_embed, dim_embed, bias = False)
    self.dropout = torch.nn.Dropout( p=dropout_rate) if dropout_rate > 0. else torch.nn.Identity()

    lnorm = norm if with_qk_lnorm else torch.nn.Identity
    self.lnorm_q = lnorm( self.dim_head_proj)
    self.lnorm_k = lnorm( self.dim_head_proj)

    assert with_flash, 'Only flash attention supported at the moment'

    def att( qs, ks, vs, x_mask) : 

      def sparsity_mask( score, b, h, q_idx, kv_idx):
        # return x_mask[q_idx] == x_mask[kv_idx]
        return (q_idx // 16) == (kv_idx % 16)

      return flex_attention( qs, ks, vs, score_mod=sparsity_mask)
    self.compiled_flex_attention = torch.compile( att, dynamic=False)

  #########################################
  def forward( self, x, x_lens = None) :

    x_in = x
    x = self.lnorm( x)

    # project onto heads and q,k,v and ensure these are 4D tensors as required for flash attention
    s = [ x.shape[0], 1, self.num_heads, -1]
    qs = self.lnorm_q( self.proj_heads_q( x).reshape(s)).to(torch.float16).permute( [1, 2, 0, 3])
    ks = self.lnorm_k( self.proj_heads_k( x).reshape(s)).to(torch.float16).permute( [1, 2, 0, 3])
    vs = self.proj_heads_v( x).reshape(s).permute( [1, 2, 0, 3])

    outs = self.compiled_flex_attention( qs, ks, vs).transpose(1,2).squeeze()

    return x_in + self.dropout( self.proj_out( outs.flatten( -2, -1)) )

####################################################################################################
class MultiSelfAttentionHead_Local(torch.nn.Module):

  #########################################
  def __init__(self, dim_embed, num_heads, qkv_len, block_factor, 
                     dim_head_proj=None, dropout_rate=0.,
                     with_qk_lnorm=True, with_flash=True, norm_type = 'LayerNorm',
                     softcap=0.0, dim_aux=None) :

    super(MultiSelfAttentionHead_Local, self).__init__()

    self.num_heads = num_heads
    self.with_flash = with_flash
    self.softcap = softcap

    assert 0 == dim_embed % num_heads
    self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

    if norm_type == 'LayerNorm' :
      norm = partial( torch.nn.LayerNorm, elementwise_affine=False) 
    else :
      norm = RMSNorm

    if dim_aux is not None :
      self.lnorm = AdaLayerNorm( dim_embed, dim_aux)
    else :
      self.lnorm = norm( dim_embed)
    self.proj_heads_q = torch.nn.Linear( dim_embed, num_heads*self.dim_head_proj, bias = False)
    self.proj_heads_k = torch.nn.Linear( dim_embed, num_heads*self.dim_head_proj, bias = False)
    self.proj_heads_v = torch.nn.Linear( dim_embed, num_heads*self.dim_head_proj, bias = False)
    self.proj_out = torch.nn.Linear( dim_embed, dim_embed, bias = False)
    self.dropout = torch.nn.Dropout( p=dropout_rate) if dropout_rate > 0. else torch.nn.Identity()

    lnorm = norm if with_qk_lnorm else torch.nn.Identity
    self.lnorm_q = lnorm( self.dim_head_proj)
    self.lnorm_k = lnorm( self.dim_head_proj)

    assert with_flash, 'Only flash attention supported.'

    # define block mask
    def mask_block_local( batch, head, idx_q, idx_kv):
      return (idx_q // block_factor) == (idx_kv // block_factor)
    self.block_mask = create_block_mask( mask_block_local, B=None, H=None, 
                                         Q_LEN=qkv_len, KV_LEN=qkv_len)
    # compile for efficiency
    self.flex_attention = torch.compile( flex_attention, dynamic=False)

  #########################################
  def forward( self, x, ada_ln_aux=None) :

    x_in = x
    x = self.lnorm( x) if ada_ln_aux is None else self.lnorm( x, ada_ln_aux)

    # project onto heads
    s = [ x.shape[0], x.shape[1], self.num_heads, -1]
    qs = self.lnorm_q( self.proj_heads_q( x).reshape(s)).to(torch.float16).permute( [0, 2, 1, 3])
    ks = self.lnorm_k( self.proj_heads_k( x).reshape(s)).to(torch.float16).permute( [0, 2, 1, 3])
    vs = self.proj_heads_v( x).reshape(s).permute( [0, 2, 1, 3])

    outs = self.flex_attention( qs, ks, vs, block_mask=self.block_mask).transpose(1,2)

    return x_in + self.proj_out( self.dropout( outs.flatten( -2, -1)) )

####################################################################################################
class MultiCrossAttentionHead_Varlen(torch.nn.Module):

  #########################################
  def __init__(self, dim_embed_q, dim_embed_kv, num_heads, dim_head_proj=None,
                     dropout_rate=0., with_residual=True, with_qk_lnorm=True, with_flash=True,  
                     norm_type = 'LayerNorm', softcap=0., dim_aux=None) :
    
    super(MultiCrossAttentionHead_Varlen, self).__init__()

    self.num_heads = num_heads
    self.dropout_rate = dropout_rate
    self.with_residual = with_residual
    self.with_flash = with_flash
    self.softcap = softcap

    if norm_type == 'LayerNorm' :
      norm = partial( torch.nn.LayerNorm, elementwise_affine=False) 
    elif norm_type == 'RMSNorm' :
      norm = RMSNorm
    else :
      assert False, 'Unsupported norm type.'

    self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

    if dim_aux is not None :
      self.lnorm_in_q = AdaLayerNorm( dim_embed_q, dim_aux)
    else :
      self.lnorm_in_q = norm( dim_embed_q)
    self.lnorm_in_kv = norm( dim_embed_kv)

    self.proj_heads_q = torch.nn.Linear( dim_embed_q, num_heads*self.dim_head_proj, bias=False)
    self.proj_heads_k = torch.nn.Linear( dim_embed_kv, num_heads*self.dim_head_proj, bias=False)
    self.proj_heads_v = torch.nn.Linear( dim_embed_kv, num_heads*self.dim_head_proj, bias=False)
    
    self.proj_out = torch.nn.Linear( self.dim_head_proj*num_heads, dim_embed_q, bias=False)
    self.dropout = torch.nn.Dropout( p=dropout_rate) if dropout_rate > 0. else torch.nn.Identity()

    lnorm = norm if with_qk_lnorm else torch.nn.Identity
    self.lnorm_q = lnorm( self.dim_head_proj)
    self.lnorm_k = lnorm( self.dim_head_proj)
    
    assert with_flash, 'Only flash attention supported at the moment'

  #########################################
  def forward( self, x_q, x_kv, x_q_lens=None, x_kv_lens=None, ada_ln_aux=None) :

    if self.with_residual :
      x_q_in = x_q
    x_q = self.lnorm_in_q( x_q) if ada_ln_aux is None else self.lnorm_in_q( x_q, ada_ln_aux)
    x_kv = self.lnorm_in_kv( x_kv)

    # project onto heads and q,k,v and ensure these are 4D tensors as required for flash attention
    s = [ x_q.shape[0], self.num_heads, self.dim_head_proj]
    qs = self.lnorm_q( self.proj_heads_q( x_q).reshape(s)).to(torch.float16)
    s = [ x_kv.shape[0], self.num_heads, self.dim_head_proj ]
    ks = self.lnorm_k( self.proj_heads_k( x_kv).reshape(s)).to(torch.float16)
    vs = self.proj_heads_v( x_kv).reshape(s)

    if x_kv_lens is not None :
      cum_x_q_lens = torch.cumsum( x_q_lens, 0, dtype=torch.int32)
      cum_x_kv_lens = torch.cumsum( x_kv_lens, 0, dtype=torch.int32)
      outs = flash_attn_varlen_func( qs, ks, vs, cum_x_q_lens, cum_x_kv_lens, 
                                                 x_q_lens.max(),  x_kv_lens.max(),
                                     softcap=self.softcap,
                                     dropout_p=self.dropout_rate)
    else :
      assert False

    # outs = self.dropout( self.proj_out( outs.flatten( -2, -1)) )
    outs = self.proj_out( outs.flatten( -2, -1))
    if self.with_residual :
      outs = x_q_in + outs

    return outs

  #########################################
  def attention( self, q, k, v) :
    scaling = 1. / torch.sqrt( torch.tensor(q.shape[-1]))
    return torch.matmul( self.softmax( scaling * self.score( q, k)), v)
      
  #########################################
  def score( self, q, k) :
    return torch.matmul( q, torch.transpose( k, -2, -1))

####################################################################################################
class MultiCrossAttentionHead_Varlen_SlicedQ(torch.nn.Module):

  #########################################
  def __init__(self, dim_embed_q, dim_embed_kv, num_slices_q, num_heads, dim_head_proj=None,
                     dropout_rate=0., with_residual=True, with_qk_lnorm=True, with_flash=True,  
                     norm_type = 'LayerNorm', softcap=0., dim_aux=None) :

    super(MultiCrossAttentionHead_Varlen_SlicedQ, self).__init__()

    self.num_slices_q = num_slices_q
    self.num_heads = num_heads
    self.dropout_rate = dropout_rate
    self.with_residual = with_residual
    self.with_flash = with_flash
    self.softcap = softcap

    if norm_type == 'LayerNorm' :
      norm = partial( torch.nn.LayerNorm, elementwise_affine=False) 
    elif norm_type == 'RMSNorm' :
      norm = RMSNorm
    else :
      assert False, 'Unsupported norm type.'

    self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

    if dim_aux is not None :
      self.lnorm_in_q = AdaLayerNorm( dim_embed_q, dim_aux)
    else :
      self.lnorm_in_q = norm( dim_embed_q)
    self.lnorm_in_kv = norm( dim_embed_kv)

    assert num_heads % num_slices_q == 0
    num_heads_r = num_heads
    self.proj_heads_q = torch.nn.ModuleList()
    for _ in range( num_slices_q) :
      self.proj_heads_q.append( torch.nn.Linear( dim_embed_q, num_heads_r*self.dim_head_proj, bias=False))                          
    self.proj_heads_k = torch.nn.Linear( dim_embed_kv, num_heads_r*self.dim_head_proj, bias=False)
    self.proj_heads_v = torch.nn.Linear( dim_embed_kv, num_heads_r*self.dim_head_proj, bias=False)

    self.proj_out = torch.nn.Linear( self.dim_head_proj*num_heads, dim_embed_q, bias=False)
    self.dropout = torch.nn.Dropout( p=dropout_rate) if dropout_rate > 0. else torch.nn.Identity()

    lnorm = norm if with_qk_lnorm else torch.nn.Identity
    self.lnorm_q = lnorm( self.dim_head_proj)
    self.lnorm_k = lnorm( self.dim_head_proj)

    assert with_flash, 'Only flash attention supported at the moment'

  #########################################
  def forward( self, x_q, x_kv, x_q_lens=None, x_kv_lens=None, ada_ln_aux=None) :

    if self.with_residual :
      x_q_in = x_q
    x_q = self.lnorm_in_q( x_q) if ada_ln_aux is None else self.lnorm_in_q( x_q, ada_ln_aux)
    x_kv = self.lnorm_in_kv( x_kv)

    # project onto heads and q,k,v and ensure these are 4D tensors as required for flash attention
    s = [ x_q.shape[0], self.num_heads, self.dim_head_proj]
    qs = [self.lnorm_q( head_proj( x_q_i).reshape(s)).to(torch.float16)
                                for head_proj,x_q_i in zip( self.proj_heads_q, x_q.transpose(1,0))]
    s = [ x_kv.shape[0], self.num_heads, self.dim_head_proj ]
    ks = self.lnorm_k( self.proj_heads_k( x_kv).reshape(s)).to(torch.float16)
    vs = self.proj_heads_v( x_kv).reshape(s)

    cum_x_q_lens = torch.cumsum( x_q_lens, 0, dtype=torch.int32)
    cum_x_kv_lens = torch.cumsum( x_kv_lens, 0, dtype=torch.int32)
    outs = []
    for i,qs_i in enumerate(qs) :
      outs += [ flash_attn_varlen_func( qs_i, ks, vs, cum_x_q_lens, cum_x_kv_lens, 
                                                      x_q_lens.max(), x_kv_lens.max(),
                                                      softcap=self.softcap,
                                                      dropout_p=self.dropout_rate) ]

    # outs = self.dropout( self.proj_out( torch.stack(outs).transpose(1,0).flatten( -2, -1)) )
    outs = self.proj_out( torch.stack(outs).transpose(1,0).flatten( -2, -1))
    if self.with_residual :
      outs = x_q_in + outs.reshape(x_q_in.shape)

    return outs

####################################################################################################
class MultiSelfAttentionHead(torch.nn.Module):

  #########################################
  def __init__(self, dim_embed, num_heads, dim_head_proj=None, dropout_rate=0., 
                     with_qk_lnorm=True, with_flash=True, norm_type = 'LayerNorm', dim_aux=None) :
    
    super(MultiSelfAttentionHead, self).__init__()

    self.num_heads = num_heads
    self.with_flash = with_flash
    self.dropout_rate = dropout_rate

    assert 0 == dim_embed % num_heads
    self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

    if norm_type == 'LayerNorm' :
      norm = partial( torch.nn.LayerNorm, elementwise_affine=False) 
    else :
      norm = RMSNorm

    if dim_aux is not None :
      self.lnorm = AdaLayerNorm( dim_embed, dim_aux)
    else :
      self.lnorm = norm( dim_embed)
    self.proj_heads_q = torch.nn.Linear( dim_embed, num_heads*self.dim_head_proj, bias = False)
    self.proj_heads_k = torch.nn.Linear( dim_embed, num_heads*self.dim_head_proj, bias = False)
    self.proj_heads_v = torch.nn.Linear( dim_embed, num_heads*self.dim_head_proj, bias = False)
    self.proj_out = torch.nn.Linear( dim_embed, dim_embed, bias = False)
    self.dropout = torch.nn.Dropout( p=dropout_rate) if dropout_rate > 0. else torch.nn.Identity()

    lnorm = norm if with_qk_lnorm else torch.nn.Identity
    self.lnorm_q = lnorm( self.dim_head_proj)
    self.lnorm_k = lnorm( self.dim_head_proj)
    
    if with_flash :
      self.att = torch.nn.functional.scaled_dot_product_attention
    else :
      self.att = self.attention
      self.softmax = torch.nn.Softmax(dim=-1)

  #########################################
  def forward( self, x, ada_ln_aux=None) :

    x_in = x
    # x = self.lnorm( x)
    x = self.lnorm( x) if ada_ln_aux is None else self.lnorm( x, ada_ln_aux)

    # project onto heads and q,k,v and ensure these are 4D tensors as required for flash attention
    s = [ *([x.shape[0],1] if len(x.shape)==2 else x.shape[:-1]), self.num_heads, -1]
    qs = self.lnorm_q( self.proj_heads_q( x).reshape(s)).to(torch.float16)
    ks = self.lnorm_k( self.proj_heads_k( x).reshape(s)).to(torch.float16)
    vs = self.proj_heads_v( x).reshape(s)
    
    # ordering of tensors (seq, heads, embed) (which differs from torch's flash attention implt)
    outs = flash_attn_func( qs, ks, vs, dropout_p=self.dropout_rate)

    # return x_in + self.dropout( self.proj_out( outs.flatten( -2, -1)) )
    return x_in + self.proj_out( outs.flatten( -2, -1))

  #########################################
  def attention( self, q, k, v) :
    scaling = 1. / torch.sqrt( torch.tensor(q.shape[-1]))
    return torch.matmul( self.softmax( scaling * self.score( q, k)), v)
      
  #########################################
  def score( self, q, k) :
    return torch.matmul( q, torch.transpose( k, -2, -1))

####################################################################################################
class MultiCrossAttentionHead(torch.nn.Module):

  #########################################
  def __init__(self, dim_embed_q, dim_embed_kv, num_heads, dim_head_proj=None,
                     dropout_rate=0., with_residual=True, with_qk_lnorm=True, with_flash=True,  
                     norm_type = 'LayerNorm') :
    
    super(MultiCrossAttentionHead, self).__init__()

    self.num_heads = num_heads
    self.with_residual = with_residual
    self.with_flash = with_flash

    if norm_type == 'LayerNorm' :
      norm = partial( torch.nn.LayerNorm, elementwise_affine=False) 
    else :
      norm = RMSNorm

    assert 0 == dim_embed_q % num_heads
    self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

    self.lnorm_in_q = norm( dim_embed_q)
    self.lnorm_in_kv = norm( dim_embed_kv)

    self.proj_heads_q = torch.nn.Linear( dim_embed_q, num_heads*self.dim_head_proj, bias=False)
    self.proj_heads_k = torch.nn.Linear( dim_embed_kv, num_heads*self.dim_head_proj, bias=False)
    self.proj_heads_v = torch.nn.Linear( dim_embed_kv, num_heads*self.dim_head_proj, bias=False)
    
    self.proj_out = torch.nn.Linear( self.dim_head_proj*num_heads, dim_embed_q, bias=False)
    self.dropout = torch.nn.Dropout( p=dropout_rate) if dropout_rate > 0. else torch.nn.Identity()

    lnorm = norm if with_qk_lnorm else torch.nn.Identity
    self.lnorm_q = lnorm( self.dim_head_proj)
    self.lnorm_k = lnorm( self.dim_head_proj)
    
    self.att = torch.nn.functional.scaled_dot_product_attention
    self.softmax = torch.nn.Softmax(dim=-1)

  #########################################
  def forward( self, x_q, x_kv) :

    if self.with_residual :
      x_q_in = x_q
    x_q, x_kv = self.lnorm_in_q( x_q), self.lnorm_in_kv( x_kv)

    # project onto heads and q,k,v and ensure these are 4D tensors as required for flash attention
    s = [ x_q.shape[0], -1, self.num_heads, self.dim_head_proj]
    qs = self.lnorm_q( self.proj_heads_q( x_q).reshape(s)).to(torch.float16).transpose( -3, -2)
    s = [ x_kv.shape[0], -1, self.num_heads, self.dim_head_proj ]
    ks = self.lnorm_k( self.proj_heads_k( x_kv).reshape(s)).to(torch.float16).transpose( -3, -2)
    vs = self.proj_heads_v( x_kv).reshape(s).transpose( -3, -2)

    # correct ordering of tensors with seq dimension second but last is critical
    with torch.nn.attention.sdpa_kernel( torch.nn.attention.SDPBackend.FLASH_ATTENTION) :
      outs = self.att( qs, ks, vs).transpose( 2, 1)

    outs = self.dropout( self.proj_out( outs.flatten( -2, -1)) )
    if self.with_residual :
      outs = x_q_in + outs

    return outs

  #########################################
  def attention( self, q, k, v) :
    scaling = 1. / torch.sqrt( torch.tensor(q.shape[-1]))
    return torch.matmul( self.softmax( scaling * self.score( q, k)), v)
      
  #########################################
  def score( self, q, k) :
    return torch.matmul( q, torch.transpose( k, -2, -1))
