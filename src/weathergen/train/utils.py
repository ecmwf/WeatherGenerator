
import string
import torch
import random


def get_run_id() :
  s1 = string.ascii_lowercase
  s2 = string.ascii_lowercase + string.digits
  return ''.join(random.sample(s1, 1)) + ''.join(random.sample(s2, 7))

def str_to_tensor(modelid):
  return torch.tensor([ord(c) for c in modelid], dtype=torch.int32)

def tensor_to_str(tensor):
  return ''.join([chr(x) for x in tensor])

def json_to_dict( fname) :
  json_str = open( fname, 'r').readlines()
  return json.loads( ''.join([s.replace('\n','') for s in json_str]))
