# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
import torch.nn.functional as F


def compute_cos_sim_to_prev(
    tokens: torch.Tensor, prev_tokens: torch.Tensor, num_aux_tokens: int
) -> torch.Tensor:
    """
    Per-token cosine similarity between the current and previous patch tokens.

    Auxiliary tokens (register/class) are dropped and the remaining patch tokens are
    flattened so each token contributes one similarity value. The previous tokens are
    detached so the loss only flows through the current forecast step.

    Args:
        tokens: Current latent tokens, shape ``(batch, num_tokens, dim)``.
        prev_tokens: Previous-step latent tokens, same shape as ``tokens``.
        num_aux_tokens: Number of leading auxiliary tokens to skip.

    Returns:
        1D tensor of cosine similarities, one per patch token.
    """
    cur = tokens[:, num_aux_tokens:].reshape(-1, tokens.shape[-1])
    prv = prev_tokens[:, num_aux_tokens:].reshape(-1, prev_tokens.shape[-1])
    return F.cosine_similarity(cur, prv.detach(), dim=-1)
