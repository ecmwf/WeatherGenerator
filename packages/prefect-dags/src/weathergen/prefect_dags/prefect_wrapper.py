"""
Wrapper around the prefect task and flow decorators to provide sensible
defaults for machine learning.
"""

import datetime
import functools
import hashlib
import inspect
import json
import types
from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Any, Literal, Optional, Union, get_args, get_origin, get_type_hints, overload

import prefect
import prefect.runtime.flow_run
from prefect import flow as p_flow
from prefect import task as p_task
from prefect.assets import Asset
from prefect.cache_policies import CachePolicy
from prefect.context import TaskRunContext
from prefect.flows import Flow, FlowStateHook
from prefect.futures import PrefectFuture
from prefect.results import ResultSerializer, ResultStorage
from prefect.task_runners import TaskRunner
from prefect.tasks import (
    RetryConditionCallable,
    StateHookCallable,
    Task,
    TaskRunNameValueOrCallable,
)
from prefect.utilities.annotations import NotSet

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
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def rerun_aware_cache_key(context: TaskRunContext, parameters: dict[str, Any]) -> str | None:
    flow_params = prefect.runtime.flow_run.parameters or {}
    token = flow_params.get("rerun_token") or prefect.runtime.flow_run.id
    return f"{token}::{context.task.name}::{_stable_hash(parameters)}"


# Thin wrapper around `prefect.task` so scientists writing new tasks just use
# `@task` or `@task(...)`. The decorator's signature mirrors `prefect.task`
# (same kwargs, same defaults, same return type `Task[P, R]`) so the IDE
# experience is identical — `.submit`, `.map`, `.with_options`, etc. are all
# visible on the decorated object. Two kwargs are *not* exposed because they
# carry codebase invariants:
#   - `cache_key_fn` is fixed to `rerun_aware_cache_key` (enables resumability)
#   - `persist_result` is forced True (required for cache replay)
# We also apply two non-default defaults: `cache_expiration=14d`, `retries=0`.
@overload
def task[**P, R](__fn: Callable[P, R], /) -> Task[P, R]: ...
@overload
def task[**P, R](
    __fn: Literal[None] = None,
    /,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    version: Optional[str] = None,
    cache_policy: Union[CachePolicy, type[NotSet]] = NotSet,
    cache_expiration: Optional[datetime.timedelta] = None,
    task_run_name: Optional[TaskRunNameValueOrCallable] = None,
    retries: Optional[int] = None,
    retry_delay_seconds: Union[
        float, int, list[float], Callable[[int], list[float]], None
    ] = None,
    retry_jitter_factor: Optional[float] = None,
    result_storage: Optional[ResultStorage] = None,
    result_storage_key: Optional[str] = None,
    result_serializer: Optional[ResultSerializer] = None,
    cache_result_in_memory: bool = True,
    timeout_seconds: Union[int, float, None] = None,
    log_prints: Optional[bool] = None,
    refresh_cache: Optional[bool] = None,
    on_completion: Optional[list[StateHookCallable]] = None,
    on_failure: Optional[list[StateHookCallable]] = None,
    on_running: Optional[list[StateHookCallable]] = None,
    retry_condition_fn: Optional[RetryConditionCallable] = None,
    viz_return_value: Any = None,
    asset_deps: Optional[list[Union[str, Asset]]] = None,
) -> Callable[[Callable[P, R]], Task[P, R]]: ...
def task(__fn: Any = None, /, **kwargs: Any) -> Any:
    # Apply framework defaults only when the caller didn't specify a value.
    kwargs.setdefault("cache_expiration", timedelta(days=14))
    kwargs.setdefault("retries", 0)
    # Enforced invariants — `cache_key_fn` + `persist_result` are required for
    # rerun_token-aware cache replay to work, regardless of what the caller
    # passed. Drop any user-supplied values silently.
    kwargs["cache_key_fn"] = rerun_aware_cache_key
    kwargs["persist_result"] = True
    decorator = p_task(**kwargs)
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


# Thin wrapper around `prefect.flow`. Like the `task` wrapper above, this
# mirrors `prefect.flow`'s full signature — kwargs, defaults, and `Flow[P, R]`
# return type — so IDEs see the real decorated object. One kwarg is reserved:
#   - `flow_run_name` is fixed to `make_flow_run_name` (consistent naming
#     convention across the codebase)
# In addition, every decorated flow must declare `rerun_token: str | None`;
# this is checked at decoration time and surfaces as TypeError on import.
@overload
def flow[**P, R](__fn: Callable[P, R], /) -> Flow[P, R]: ...
@overload
def flow[**P, R](
    __fn: Literal[None] = None,
    /,
    *,
    name: Optional[str] = None,
    version: Optional[str] = None,
    retries: Optional[int] = None,
    retry_delay_seconds: Optional[Union[int, float]] = None,
    task_runner: Optional[TaskRunner[PrefectFuture[Any]]] = None,
    description: Optional[str] = None,
    timeout_seconds: Union[int, float, None] = None,
    validate_parameters: bool = True,
    persist_result: Optional[bool] = None,
    result_storage: Optional[ResultStorage] = None,
    result_serializer: Optional[ResultSerializer] = None,
    cache_result_in_memory: bool = True,
    log_prints: Optional[bool] = None,
    on_completion: Optional[list[FlowStateHook[..., Any]]] = None,
    on_failure: Optional[list[FlowStateHook[..., Any]]] = None,
    on_cancellation: Optional[list[FlowStateHook[..., Any]]] = None,
    on_crashed: Optional[list[FlowStateHook[..., Any]]] = None,
    on_running: Optional[list[FlowStateHook[..., Any]]] = None,
) -> Callable[[Callable[P, R]], Flow[P, R]]: ...
def flow(__fn: Any = None, /, **kwargs: Any) -> Any:
    """
    Wrapper around @flow with sensible defaults for a machine learning workflow.
    - does not depend on the code to check if the flow needs a rerun. The code is typically
      called separately in a slurm script.
    - expects an additional (optional) parameter `rerun_token` in the flow signature, which is used to determine caching keys for tasks. See `rerun_aware_cache_key` for details.

    Usable as both `@flow` and `@flow(...)`. The signature mirrors `prefect.flow`;
    `flow_run_name` is reserved — the codebase enforces a single naming convention.
    """
    # Enforce our naming function on every flow in this codebase. Drop any
    # user-supplied flow_run_name silently.
    kwargs["flow_run_name"] = make_flow_run_name
    p_flow_decorator = p_flow(**kwargs)

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
            async def _async_wrapped(*args: Any, **kw: Any) -> Any:
                bound = sig.bind_partial(*args, **kw).arguments
                _log_rerun_info(bound.get("rerun_token"))
                return await fn(*args, **kw)

            return p_flow_decorator(_async_wrapped)

        @functools.wraps(fn)
        def _sync_wrapped(*args: Any, **kw: Any) -> Any:
            bound = sig.bind_partial(*args, **kw).arguments
            _log_rerun_info(bound.get("rerun_token"))
            return fn(*args, **kw)

        return p_flow_decorator(_sync_wrapped)

    return wrapper(__fn) if __fn is not None else wrapper
