"""
Coding recipe for abstract fields in Python.

Based on https://stackoverflow.com/questions/23831510/abstract-attribute-not-property
"""

from abc import ABCMeta as NativeABCMeta
from collections.abc import Callable
from typing import Any, TypeVar, cast

R = TypeVar("R")


class DummyAttribute:
    pass


def abstract_attribute(obj: Callable[[Any], R] | None = None) -> R:
    _obj = cast(Any, obj)
    if obj is None:
        _obj = DummyAttribute()
    _obj.__is_abstract_attribute__ = True
    return cast(R, _obj)


class ABCMeta(NativeABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = NativeABCMeta.__call__(cls, *args, **kwargs)
        abstract_attributes = {
            name
            for name in dir(instance)
            if hasattr(getattr(instance, name), "__is_abstract_attribute__")
        }
        if abstract_attributes:
            raise NotImplementedError(
                "Can't instantiate abstract class {} with abstract attributes: {}".format(
                    cls.__name__, ", ".join(abstract_attributes)
                )
            )
        return instance
