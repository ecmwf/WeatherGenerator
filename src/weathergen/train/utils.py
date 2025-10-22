# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import random
import string

import torch


from weathergen.common import config

# TODO: remove this definition, it should directly using common.
get_run_id = config.get_run_id


def str_to_tensor(modelid):
    return torch.tensor([ord(c) for c in modelid], dtype=torch.int32)


def tensor_to_str(tensor):
    return "".join([chr(x) for x in tensor])


def json_to_dict(fname):
    with open(fname) as f:
        json_str = f.readlines()
    return json.loads("".join([s.replace("\n", "") for s in json_str]))
