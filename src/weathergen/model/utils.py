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


def resample_latent(
    tokens: torch.Tensor,
    source_level: int,
    target_level: int,
    num_aux_tokens: int,
    num_queries: int,
) -> torch.Tensor:
    """Pool nested children or repeat parents without mixing queries or auxiliaries."""
    if source_level == target_level:
        return tokens
    spatial = tokens[..., num_aux_tokens:, :].reshape(
        *tokens.shape[:-2], 12 * 4**source_level, num_queries, tokens.shape[-1]
    )
    if source_level > target_level:
        spatial = spatial.reshape(
            *tokens.shape[:-2],
            12 * 4**target_level,
            4 ** (source_level - target_level),
            num_queries,
            tokens.shape[-1],
        )
        dtype = torch.float32 if tokens.dtype in (torch.float16, torch.bfloat16) else tokens.dtype
        spatial = spatial.mean(dim=-3, dtype=dtype).to(tokens.dtype)
    else:
        spatial = spatial.repeat_interleave(4 ** (target_level - source_level), dim=-3)
    return torch.cat((tokens[..., :num_aux_tokens, :], spatial.flatten(-3, -2)), dim=-2)


def exchange_latents(
    latents: dict[str, torch.Tensor],
    levels: dict[str, int],
    common_level: int,
    num_aux_tokens: int,
    num_queries: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Replace each native coarse component by consensus, returning the common sum."""
    if len(latents) == 1:
        return latents, next(iter(latents.values()))
    common_dtype = next(iter(latents.values())).dtype
    coarse = {}
    for name, tokens in latents.items():
        common_dtype = torch.promote_types(common_dtype, tokens.dtype)
        work = tokens.float() if tokens.dtype in (torch.float16, torch.bfloat16) else tokens
        coarse[name] = resample_latent(
            work, levels[name], common_level, num_aux_tokens, num_queries
        )
    common = sum(coarse.values())
    mean = common / len(latents)
    # ponytail: coarse-only exchange; learned fine-scale attention would share native residuals.
    exchanged = {
        name: (
            tokens
            + resample_latent(
                mean - coarse[name], common_level, levels[name], num_aux_tokens, num_queries
            )
        ).to(tokens.dtype)
        for name, tokens in latents.items()
    }
    return exchanged, common.to(common_dtype)


def get_num_parameters(block):
    nps = filter(lambda p: p.requires_grad, block.parameters())
    return sum([torch.prod(torch.tensor(p.size())) for p in nps])


def freeze_weights(block):
    if hasattr(block, "name"):
        logger.info(f"Freeze block {block.name}")
    for p in block.parameters():
        p.requires_grad = False


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
