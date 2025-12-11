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


def flatten_dict(d, parent_key="", sep="."):
    """
    Flattens a nested dictionary, keeping lists of scalar values intact.

    :param d: The dictionary to flatten.
    :param parent_key: The base key for recursion (used internally).
    :param sep: The separator to join keys.
    :return: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        # Construct the new key
        new_key = parent_key + sep + k if parent_key else k

        # 1. Handle Dictionaries (Recursion)
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_dict(v, new_key, sep=sep).items())

        # 2. Handle Lists
        elif isinstance(v, list):
            # Check if the list contains non-scalar/non-empty values (i.e., nested dicts/lists)
            # A value is considered a scalar if it's NOT a dict or a list.
            is_scalar_list = all(not isinstance(item, (dict | list)) for item in v)

            if is_scalar_list:
                # Requirement: Keep lists of scalar values as is
                items.append((new_key, v))
            else:
                # If the list contains nested dicts/lists, we must iterate and flatten them
                for i, item in enumerate(v):
                    index_key = new_key + sep + str(i)
                    if isinstance(item, dict):
                        # Recursively flatten the dictionary inside the list
                        items.extend(flatten_dict(item, index_key, sep=sep).items())
                    elif isinstance(item, list):
                        # Treat list within a list as a scalar list *at that level* # and append it (to avoid overly complex list indexing)
                        items.append((index_key, item))
                    else:
                        # Append the scalar item
                        items.append((index_key, item))

        # 3. Handle Scalar Values
        else:
            # Append all other scalar values (str, int, float, bool, None, etc.)
            items.append((new_key, v))

    return dict(items)


def unflatten_dict(d, separator="."):
    """
    Unflattens a dictionary where nested keys were joined by a separator.

    :param d: The flattened dictionary.
    :param separator: The delimiter used to join nested keys.
    :return: The unflattened dictionary.
    """
    unflattened = {}
    for key, value in d.items():
        # Split the key into its components
        parts = key.split(separator)

        # Start at the root of the unflattened dictionary
        current_level = unflattened

        # Iterate over all parts of the key except the last one
        for part in parts[:-1]:
            # If the part is not a key in the current level, create a new dictionary
            if part not in current_level:
                current_level[part] = {}

            # Move down to the next level
            current_level = current_level[part]

        # Set the value for the final, innermost key
        current_level[parts[-1]] = value

    return unflattened
