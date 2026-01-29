# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Standard library
import logging
from typing import Any

# Third-party
from omegaconf.listconfig import ListConfig

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def to_list(obj: Any) -> list:
    """
    Convert given object to list if obj is not already a list. Sets are also transformed to a list.

    Parameters
    ----------
    obj : Any
        The object to transform into a list.
    Returns
    -------
    list
        A list containing the object, or the object itself if it was already a list.
    """
    if isinstance(obj, set | tuple | ListConfig):
        obj = list(obj)
    elif not isinstance(obj, list):
        obj = [obj]
    return obj


def arr_to_list(arr, idx):
    """
    Convert array values at specific index to a Python list for JSON serialization.

    Parameters
    ----------
    arr : xr.DataArray or np.ndarray
        Array containing values to convert.
    idx : tuple
        Index tuple for accessing multi-dimensional array.

    Returns
    -------
    list
        Python list of values at the specified index.
    """
    return arr.values[(...,) + idx].tolist() if arr.ndim > 1 else arr.values.tolist()


def arr_to_float(arr, idx):
    """
    Convert array value at specific index to a Python float for JSON serialization.

    Parameters
    ----------
    arr : xr.DataArray or np.ndarray
        Array containing the value to convert.
    idx : tuple
        Index for accessing the array.

    Returns
    -------
    float
        Python float value at the specified index.
    """
    return float(arr.values[idx]) if arr.ndim > 0 else float(arr.values)
