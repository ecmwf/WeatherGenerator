"""
Wrapper around the prefect task and flow decorators to provide sensible 
defaults for machine learning.
"""

import functools
import hashlib
import inspect
import json
import types
from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Any, Union, get_args, get_origin, get_type_hints, overload

import prefect
import prefect.runtime.flow_run
from prefect import flow as p_flow, task as p_task
from prefect.context import TaskRunContext

from weathergen.prefect_dags.prefect_logging import get_run_logger


# ---------------------------------------------------------------------------
# Cache key: (rerun_token OR flow_run.id) :: task name :: hashed parameters.
#
# - No token  -> key embeds the *current* flow_run.id -> nothing in the cache
#                matches -> task runs -> result is persisted under this id.
# - With token -> key is stable across runs that share the token -> previously
#                completed tasks are returned from cache and skipped.
# ---------------------------------------------------------------------------
def _stable_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def rerun_aware_cache_key(
    context: TaskRunContext, parameters: dict[str, Any]
) -> str | None:
    flow_params = prefect.runtime.flow_run.parameters or {}
    token = flow_params.get("rerun_token") or prefect.runtime.flow_run.id
    return f"{token}::{context.task.name}::{_stable_hash(parameters)}"


# A thin wrapper around @task so scientists writing new tasks just use
# @task or @task(...) and get the right caching behaviour automatically. The
# signature mirrors the common subset of `prefect.task` so IDE autocomplete
# works; remaining prefect kwargs are forwarded via **extra_kwargs.
#
# Two overloads make type checkers (pyrefly / pyright) preserve the wrapped
# function's signature through the decorator: calling the decorated `get_pwd`
# stays `() -> Coroutine[..., str]` rather than degrading to `Any` / mismatched
# Task signatures. ParamSpec carries the args, R the return type.
@overload
def task[**P, R](__fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def task[**P, R](
    *,
    name: str | None = ...,
    description: str | None = ...,
    tags: Iterable[str] | None = ...,
    version: str | None = ...,
    cache_expiration: timedelta | None = ...,
    task_run_name: str | Callable[[], str] | None = ...,
    retries: int | None = ...,
    retry_delay_seconds: float | list[float] | None = ...,
    retry_jitter_factor: float | None = ...,
    timeout_seconds: float | None = ...,
    log_prints: bool | None = ...,
    refresh_cache: bool | None = ...,
    **extra_kwargs: Any,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def task(
    __fn: Any = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: Iterable[str] | None = None,
    version: str | None = None,
    cache_expiration: timedelta | None = None,
    task_run_name: str | Callable[[], str] | None = None,
    retries: int | None = None,
    retry_delay_seconds: float | list[float] | None = None,
    retry_jitter_factor: float | None = None,
    timeout_seconds: float | None = None,
    log_prints: bool | None = None,
    refresh_cache: bool | None = None,
    **extra_kwargs: Any,
) -> Any:
    # Framework defaults applied only when the caller didn't specify a value.
    if cache_expiration is None:
        cache_expiration = timedelta(days=14)
    if retries is None:
        retries = 0
    # Enforced contract: cache_key_fn and persist_result are required for
    # rerun_token-aware cache replay to work. Not user-configurable here.
    decorator = p_task(
        cache_key_fn=rerun_aware_cache_key,
        persist_result=True,
        name=name,
        description=description,
        tags=tags,
        version=version,
        cache_expiration=cache_expiration,
        task_run_name=task_run_name,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        retry_jitter_factor=retry_jitter_factor,
        timeout_seconds=timeout_seconds,
        log_prints=log_prints,
        refresh_cache=refresh_cache,
        **extra_kwargs,
    )
    return decorator(__fn) if __fn is not None else decorator

def make_flow_run_name() -> str:
    # Prefect runtime exposes the active flow's name (the @flow(name=...) value
    # or the function name). Fall back to "flow" if invoked outside a run.
    flow_name = (prefect.runtime.flow_run.flow_name or "flow").replace("_", "-")
    params = prefect.runtime.flow_run.parameters or {}
    if params.get("rerun_token"):
        tag = f"rerun-{params['rerun_token'][:8]}"
    else:
        tag = f"fresh-{prefect.runtime.flow_run.id[:8]}"
    return f"{flow_name}-{tag}"

def _is_str_or_none(tp: Any) -> bool:
    # Accept both `str | None` (PEP 604, types.UnionType) and `Optional[str]` /
    # `Union[str, None]` (typing.Union).
    if isinstance(tp, types.UnionType) or get_origin(tp) is Union:
        return set(get_args(tp)) == {str, type(None)}
    return False


def _validate_rerun_token_param(fn: Any) -> None:
    """Ensure the decorated flow function exposes the rerun_token contract.

    Required: `rerun_token` parameter must exist. If it carries a type
    annotation, that annotation must be `str | None`. Raises TypeError at
    decoration time so the mistake surfaces on import, not first run.
    """
    sig = inspect.signature(fn)
    if "rerun_token" not in sig.parameters:
        raise TypeError(
            f"Flow {fn.__name__!r} must accept a `rerun_token` parameter. "
            "Add `rerun_token: str | None = None` to its signature."
        )
    # get_type_hints evaluates string annotations and `from __future__ import
    # annotations` defers; signature.parameters[...].annotation can be a string.
    hints = get_type_hints(fn)
    if "rerun_token" in hints and not _is_str_or_none(hints["rerun_token"]):
        raise TypeError(
            f"Flow {fn.__name__!r}: `rerun_token` must be typed as `str | None`, "
            f"got {hints['rerun_token']!r}."
        )


def _log_rerun_info(rerun_token: str | None) -> None:
    """Log how to (re)run the flow.

    On a fresh run, prints the current flow_run.id so the user knows the
    value to pass back as `rerun_token` to skip already-completed tasks.
    On a resume, confirms which token is being used.
    """
    log = get_run_logger()
    if rerun_token is None:
        log.info(
            "Fresh run. To resume if interrupted, rerun with "
            f"rerun_token='{prefect.runtime.flow_run.id}'."
        )
    else:
        log.info(f"Resuming with rerun_token={rerun_token!r}.")


@overload
def flow[**P, R](__fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def flow[**P, R](
    *,
    name: str | None = ...,
    version: str | None = ...,
    retries: int | None = ...,
    retry_delay_seconds: float | None = ...,
    description: str | None = ...,
    timeout_seconds: float | None = ...,
    validate_parameters: bool = ...,
    log_prints: bool | None = ...,
    **extra_kwargs: Any,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def flow(
    __fn: Any = None,
    /,
    *,
    name: str | None = None,
    version: str | None = None,
    retries: int | None = None,
    retry_delay_seconds: float | None = None,
    description: str | None = None,
    timeout_seconds: float | None = None,
    validate_parameters: bool = True,
    log_prints: bool | None = None,
    **extra_kwargs: Any,
) -> Any:
    """
    Wrapper around @flow with sensible defaults for a machine learning workflow.
    - does not depend on the code to check if the flow needs a rerun. The code is typically
      called separately in a slurm script.
    - expects an additional (optional) parameter `rerun_token` in the flow signature, which is used to determine caching keys for tasks. See `rerun_aware_cache_key` for details.

    Usable as both `@flow` and `@flow(...)`. The signature mirrors the common
    subset of `prefect.flow`; remaining kwargs are forwarded via **extra_kwargs.
    `flow_run_name` is intentionally not exposed — the codebase enforces a
    single naming convention.
    """
    # Enforce our naming function on every flow in this codebase. Drop any
    # user-supplied flow_run_name silently — the design intent is one
    # consistent naming convention, not a per-flow opt-out.
    extra_kwargs.pop("flow_run_name", None)
    p_flow_decorator = p_flow(
        flow_run_name=make_flow_run_name,
        name=name,
        version=version,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        description=description,
        timeout_seconds=timeout_seconds,
        validate_parameters=validate_parameters,
        log_prints=log_prints,
        **extra_kwargs,
    )

    def wrapper(fn):
        _validate_rerun_token_param(fn)
        sig = inspect.signature(fn)

        # Wrap the user's function so we log the rerun token at the start of
        # every run. `bind_partial` handles both positional and keyword args
        # so `rerun_token` is extracted correctly regardless of how the flow
        # was invoked. functools.wraps preserves __wrapped__ so Prefect's
        # signature introspection still sees the original function shape.
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def _async_wrapped(*args: Any, **kwargs: Any) -> Any:
                bound = sig.bind_partial(*args, **kwargs).arguments
                _log_rerun_info(bound.get("rerun_token"))
                return await fn(*args, **kwargs)
            return p_flow_decorator(_async_wrapped)

        @functools.wraps(fn)
        def _sync_wrapped(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind_partial(*args, **kwargs).arguments
            _log_rerun_info(bound.get("rerun_token"))
            return fn(*args, **kwargs)
        return p_flow_decorator(_sync_wrapped)

    return wrapper(__fn) if __fn is not None else wrapper
