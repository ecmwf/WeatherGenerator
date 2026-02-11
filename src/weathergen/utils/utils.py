# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch

from weathergen.utils.train_logger import TRAIN, Stage


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


def is_stream_forcing(stream_cfg: dict, stage: Stage | None = None) -> bool:
    """
    Determine if stream is forcing, i.e. does not produce (physical) predictions
    """
    is_forcing = stream_cfg.get("forcing", False)
    if stage is not None:
        is_forcing = is_forcing or (
            (len(stream_cfg.get("train_target_channels", [])) == 0)
            if stage == TRAIN
            else (len(stream_cfg.get("val_target_channels", [])) == 0)
        )
    else:
        is_forcing = is_forcing or (
            len(stream_cfg.get("train_target_channels", [])) == 0
            and len(stream_cfg.get("val_target_channels", [])) == 0
        )

    return is_forcing


def is_stream_diagnostic(stream_cfg: dict, stage: Stage | None = None) -> bool:
    """
    Determine if stream is diagnostic, i.e. does not contribute to model input
    """
    is_diagnostic = stream_cfg.get("diagnostic", False)
    if stage is not None:
        is_diagnostic = is_diagnostic or (
            (len(stream_cfg.get("train_source_channels", [])) == 0)
            if stage == TRAIN
            else (len(stream_cfg.get("val_source_channels", [])) == 0)
        )
    else:
        is_diagnostic = is_diagnostic or (
            len(stream_cfg.get("train_source_channels", [])) == 0
            and len(stream_cfg.get("val_source_channels", [])) == 0
        )

    return is_diagnostic
