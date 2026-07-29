# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import re

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_num_parameters(block):
    nps = filter(lambda p: p.requires_grad, block.parameters())
    return sum([torch.prod(torch.tensor(p.size())) for p in nps])


def freeze_weights(block):
    if hasattr(block, "name"):
        logger.info(f"Freeze block {block.name}")
    for p in block.parameters():
        p.requires_grad = False


def reset_weights(block):
    block_name = getattr(block, "name", type(block).__name__)
    if hasattr(block, "reset_parameters"):
        logger.info(f"Reset weights of block {block_name}")
        block.reset_parameters()
    else:
        logger.info(f"Skip reset for block {block_name} (no reset_parameters)")


def set_to_eval(block):
    if hasattr(block, "name"):
        logger.info(f"Set block {block.name} to eval mode")
    block.eval()


def apply_fct_to_blocks(model, blocks, fct):
    """
    Apply a function to specific blocks of a model.
    Args:
        model : model instance with attribute named_modules
        blocks : regex pattern to match block names
        fct : function to apply to matching blocks
    """

    for name, module in model.named_modules():
        name = module.name if hasattr(module, "name") else name
        # avoid the whole model element which has name ''
        if (re.fullmatch(blocks, name) is not None) and (name != ""):
            fct(module)
            logger.debug(f"Applied function {fct.__name__} to block {name}")
        else:
            logger.debug(f"Did not apply function {fct.__name__} to block {name}")


def broadcast_matching_params(model, blocks, src=0):
    """
    Broadcast parameters and buffers of blocks matching the regex from rank src to
    all ranks. Needed after reset_parameters() under DDP: the reset draws from each
    rank's own RNG, and DDP only syncs parameters at wrap time, so without a
    broadcast the ranks train permanently diverged weights.
    Args:
        model : model instance with attribute named_modules
        blocks : regex pattern to match block names
        src : rank whose values are broadcast
    """

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    seen = set()
    tensors = []
    for name, module in model.named_modules():
        name = module.name if hasattr(module, "name") else name
        if (re.fullmatch(blocks, name) is not None) and (name != ""):
            for t in list(module.parameters()) + list(module.buffers()):
                if id(t) not in seen:
                    seen.add(id(t))
                    tensors.append(t)
    for t in tensors:
        torch.distributed.broadcast(t.data, src=src)
    logger.info(f"Broadcast {len(tensors)} reset tensors from rank {src}")


def check_reset_not_frozen(model, reset_blocks):
    """
    Verify that no parameter about to be reset is frozen. A parameter that is reset
    to random values but has requires_grad=False can never train, leaving random
    dead weights in the model (almost never intended).
    Args:
        model : model instance with attribute named_modules
        reset_blocks : regex pattern of block names that will be reset
    Raises:
        ValueError listing the frozen parameters that match the reset pattern.
    """

    frozen = []
    for name, module in model.named_modules():
        name = module.name if hasattr(module, "name") else name
        if (re.fullmatch(reset_blocks, name) is not None) and (name != ""):
            frozen += [f"{name}.{pn}" for pn, p in module.named_parameters() if not p.requires_grad]
    if frozen:
        frozen = sorted(set(frozen))
        raise ValueError(
            "reset_modules overlaps with frozen parameters; these would be reset to random "
            "values but never trained. Remove them from freeze_modules or reset_modules: "
            + ", ".join(frozen[:16])
            + (" ..." if len(frozen) > 16 else "")
        )


def log_trainable_summary(model):
    """
    Log per-top-level-block parameter counts and trainable fractions.
    """

    # unwrap DDP for readable block names
    block = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    logger.info("Trainable parameter summary:")
    for name, child in block.named_children():
        n_total = sum(p.numel() for p in child.parameters())
        if n_total == 0:
            continue
        n_train = sum(p.numel() for p in child.parameters() if p.requires_grad)
        logger.info(
            f"  {name}: {n_train:,} / {n_total:,} trainable ({100 * n_train / n_total:.1f}%)"
        )
    n_total = sum(p.numel() for p in block.parameters())
    n_train = sum(p.numel() for p in block.parameters() if p.requires_grad)
    if n_total > 0:
        logger.info(
            f"  total: {n_train:,} / {n_total:,} trainable ({100 * n_train / n_total:.1f}%)"
        )


class ActivationFactory:
    _registry = {
        "identity": nn.Identity,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "prelu": nn.PReLU,
        "softplus": nn.Softplus,
        "linear": nn.Linear,
        "logsoftmax": nn.LogSoftmax,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
    }

    @classmethod
    def get(cls, name: str, **kwargs):
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unsupported activation type: '{name}'")
        fn = cls._registry[name]
        return fn(**kwargs) if callable(fn) else fn
