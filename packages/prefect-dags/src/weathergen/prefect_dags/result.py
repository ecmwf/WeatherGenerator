"""
Result[T] = T | OpError — the codebase-wide convention for fallible operations.

Primitives like `run_cmd` and `submit_slurm` return `Result[T]` instead of
raising. Callers narrow with `isinstance(r, OpError)` (or the `is_err`
TypeGuard helper) before using the success value.

The contract is: failures *of the operation itself* are captured and
returned as `OpError`. Programmer bugs (typos, AttributeError, ...) still
raise — conflating the two flattens useful tracebacks and hides bugs.

Why this shape (vs. `Ok(T) | Err(E)`):
    Scientists writing Prefect tasks/flows are the ultimate consumers.
    The success path is unwrapped (`result.stdout` works directly), the
    failure check is one isinstance, and the type checker enforces it
    without scientists having to learn `Ok` / `Err` / `.unwrap()`.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from typing_extensions import TypeIs  # Python 3.13+ has typing.TypeIs natively.


@dataclass
class OpError:
    """
    A captured failure from a fallible operation.

    `err` is the underlying exception. The remaining fields carry partial
    state when the operation produced output before failing (typically a
    shell command); leave them None when not applicable (e.g. validation
    errors that fail before any execution).
    """

    err: Exception
    stdout: str | None = None
    stderr: str | None = None
    return_code: int | None = None


type Result[T] = T | OpError


def is_err(r: object) -> TypeIs[OpError]:
    """
    Type-narrowing predicate for `Result[T]`.

    `TypeIs` (vs. `TypeGuard`) narrows in BOTH branches: after `if is_err(r):`
    the True branch sees `OpError`, and the False branch sees `T` — so
    `Result[CommandResult]` becomes `CommandResult` automatically without
    a follow-up assert. That's why `is_err` is preferred over a plain
    `isinstance(r, OpError)` at call sites.
    """
    return isinstance(r, OpError)


def try_run[T](
    fn: Callable[[], T],
    logger: logging.Logger | None = None,
) -> Result[T]:
    """
    Run a sync callable and capture any exception as an `OpError`.

    Use this to bridge code that raises (file I/O, parsing, third-party
    libraries) into the Result convention so that a larger function can
    stay in the Result world without nested try/except. Bind arguments
    via `lambda` or `functools.partial`.
    """
    try:
        return fn()
    except Exception as e:
        if logger is not None:
            logger.error(f"try_run captured exception: {e}")
        return OpError(err=e)


async def try_run_async[T](
    fn: Callable[[], Awaitable[T]],
    logger: logging.Logger | None = None,
) -> Result[T]:
    """Async variant of `try_run` for awaitable callables."""
    try:
        return await fn()
    except Exception as e:
        if logger is not None:
            logger.error(f"try_run_async captured exception: {e}")
        return OpError(err=e)


def unwrap[T](r: Result[T]) -> T:
    """
    Return the success value, or re-raise the captured exception.

    The escape hatch out of the Result convention into normal exception
    flow. Use this at boundaries where you'd rather propagate failure
    via `raise` — e.g. inside a Prefect task body, where an unhandled
    exception is the right way to fail the task.

    The original exception (with its original traceback) is re-raised.
    Python adds the unwrap call site as `__context__` so both locations
    are visible in the printed traceback.
    """
    if is_err(r):
        raise r.err
    return r
