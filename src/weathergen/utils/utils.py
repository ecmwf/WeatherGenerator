# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy

import torch

from weathergen.common.config import Config


def apply_overrides_to_dict(cf: Config, overrides: dict) -> Config:
    copied_cf = copy.deepcopy(cf)
    for key, val in overrides.items():
        copied_cf[key] = val
    return copied_cf


def get_dtype(value: str) -> torch.dtype:
    """
    changes the conf value to a torch dtype
    """
    if value == "bf16":
        return torch.bfloat16
    elif value == "fp16":
        return torch.float16
    elif value == "fp32":
        return torch.float32
    else:
        raise NotImplementedError(
            f"Dtype {value} is not recognized, choose either, bf16, fp16, or fp32"
        )


def get_batch_size(cf: Config, world_size: int) -> int:
    return world_size * cf.batch_size_per_gpu
