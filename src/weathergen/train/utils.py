# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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
