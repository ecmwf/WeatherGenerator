# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json

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


def flatten_dictionary(d, parent_key="", sep="."):
    """
    Flattens a nested dictionary into a single-level dictionary
    with concatenated keys.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key string for the current level (used in recursion).
        sep (str): The separator to use between keys (e.g., '_', '.').

    Returns:
        dict: The flattened dictionary.
    """
    items = []

    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)

        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_dictionary(v, new_key, sep=sep).items())
        elif isinstance(v, list | tuple):
            # Handle lists/tuples, treating items inside as indexed elements
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    # If list item is a dict, flatten it with index in key
                    list_key = new_key + sep + str(i)
                    items.extend(flatten_dictionary(item, list_key, sep=sep).items())
                else:
                    # If list item is a scalar, store it with index in key
                    items.append((new_key + sep + str(i), item))
        else:
            # Base case: value is a scalar, store key-value pair
            items.append((new_key, v))

    return dict(items)
