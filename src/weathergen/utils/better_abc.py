"""
Coding recipe for abstract fields in Python.

Based on https://stackoverflow.com/questions/23831510/abstract-attribute-not-property

It is necessary because of our coding style and python's limited support for abstract fields.
This code is a workaround to allow defining abstract attributes
in classes that use the `ABCMeta` metaclass.
"""

from abc import ABCMeta as NativeABCMeta
from collections.abc import Callable
from typing import Any, cast


class DummyAttribute:
    pass


def abstract_attribute[R](obj: Callable[[Any], R] | None = None) -> R:
    _obj = cast(Any, obj)
    if obj is None:
        _obj = DummyAttribute()
    _obj.__is_abstract_attribute__ = True  # pylint: disable=attribute-defined-outside-init
    return cast(R, _obj)


class ABCMeta(NativeABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = NativeABCMeta.__call__(cls, *args, **kwargs)
        # pylint: disable-next=attribute-defined-outside-init
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
