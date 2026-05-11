"""
Thin wrapper over `prefect.logging.get_run_logger` that hides Prefect's
`Logger | LoggerAdapter` return type.

Prefect's `get_run_logger` may return a `LoggerAdapter` that is structurally
compatible with `logging.Logger` (same `info`, `debug`, ... API). Callers in
this codebase work in terms of `logging.Logger` only, so the cast is
performed once here. Use `get_run_logger()` everywhere instead of importing
`prefect.logging.get_run_logger` directly.
"""

import logging
from typing import cast

from prefect.logging import get_run_logger as _get_run_logger

__all__ = ["get_run_logger"]

def get_run_logger() -> logging.Logger:
    """Return the active Prefect run logger as a `logging.Logger`.

    Must be called inside a Prefect flow or task run; raises otherwise
    (this is the same precondition as `prefect.logging.get_run_logger`).
    """
    return cast(logging.Logger, _get_run_logger())
