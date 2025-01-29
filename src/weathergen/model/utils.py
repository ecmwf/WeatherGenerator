
import torch

#########################################
def get_num_parameters( block) :
  nps = filter(lambda p: p.requires_grad, block.parameters())
  return sum([torch.prod(torch.tensor(p.size())) for p in nps])

#########################################
def freeze_weights( block) :
  for p in block.parameters() :
    p.requires_grad = False
