# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch


#########################################
def get_num_parameters(block):
    nps = filter(lambda p: p.requires_grad, block.parameters())
    return sum([torch.prod(torch.tensor(p.size())) for p in nps])


#########################################
def freeze_weights(block):
    for p in block.parameters():
        p.requires_grad = False

#########################################
def get_activation(name: str):
    """
    Returns a PyTorch activation function based on the provided name.
    Args:
        name (str): The name of the activation function.
    Returns:
        torch.nn.Module: The corresponding activation function.
    """
    name = name.lower()
    if name == "tanh":
        return torch.nn.Tanh()
    elif name == "softmax":
        return torch.nn.Softmax(dim=-1)
    elif name == "sigmoid":
        return torch.nn.Sigmoid()
    elif name == "gelu":
        return torch.nn.GELU()
    elif name == "identity":
        return torch.nn.Identity()
    elif name == "relu":
        return torch.nn.ReLU()
    elif name == "leakyrelu":
        return torch.nn.LeakyReLU()
    elif name == "elu":
        return torch.nn.ELU()
    elif name == "selu":
        return torch.nn.SELU()
    elif name == "prelu":
        return torch.nn.PReLU()
    elif name == "softplus":
        return torch.nn.Softplus()
    elif name == "linear":
        return None
    else:
        raise ValueError(f"Unsupported activation type: {name}")