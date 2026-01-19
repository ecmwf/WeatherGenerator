from typing import Protocol, runtime_checkable

import torch

from weathergen.common.io import IOReaderData


@runtime_checkable
class Pinnable(Protocol):
    """
    Protocol that allows the pytorch content of a data structure
    to be pinned to the memory of the current accelerator.

    This extends the pin_memory() capability of a torch Tensor
    to other classes.

    It is blocking.
    """

    def pin_memory(self): ...


def pin_object(obj: Pinnable | torch.Tensor | IOReaderData | list | dict | None):
    if obj is None:
        return
    elif isinstance(obj, torch.Tensor | Pinnable):
        obj.pin_memory()
    elif isinstance(obj, IOReaderData):
        # Special case: IOReaderData is in common package and can't have torch deps
        # Note: These SHOULD be numpy arrays per the type hints, but might be tensors
        pin_object(obj.coords)
        pin_object(obj.data)
        pin_object(obj.geoinfos)

    elif isinstance(obj, list):
        # Assume the list is a list of potentially pinnable objects and traverse it.
        for e in obj:
            pin_object(e)
    elif isinstance(obj, dict):
        # Assume the values are pinnable.
        for e in obj.values():
            pin_object(e)
