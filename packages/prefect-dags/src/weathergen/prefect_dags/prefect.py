"""
Wrappers around prefect primitives (task / flow decorators, run logger) to
provide sensible defaults for machine learning.

The code is mostly boilerplate:
- `task` wrapper:
    sets cache_policy, cache_expiration, retries,
    enforces the presence of `cache_key_fn` and `persist_result` for cache replay to work.
- `flow` wrapper:
    sets flow_run_name to a consistent convention,
    enforces the presence of `rerun_token: str | None` in the signature for cache replay to work,
    logs the rerun token at the start of every run so users know what to pass to resume if
    interrupted.
- `get_run_logger` wrapper:
    hides Prefect's `Logger | LoggerAdapter` return type. Prefect's
    `get_run_logger` may return a `LoggerAdapter` that is structurally
    compatible with `logging.Logger` (same `info`, `debug`, ... API). Callers
    in this codebase work in terms of `logging.Logger` only, so the cast is
    performed once here. Use this `get_run_logger()` everywhere instead of
    importing `prefect.logging.get_run_logger` directly.
"""

import datetime
import functools
import hashlib
import inspect
import json
import logging
import types
from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import (
    Any,
    ParamSpec,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import prefect
import prefect.runtime.flow_run
from prefect import flow as p_flow
from prefect import task as p_task
from prefect.assets import Asset
from prefect.cache_policies import CachePolicy
from prefect.context import TaskRunContext
from prefect.flows import Flow, FlowStateHook
from prefect.futures import PrefectFuture
from prefect.logging import get_run_logger as _get_run_logger
from prefect.results import ResultSerializer, ResultStorage
from prefect.task_runners import TaskRunner
from prefect.tasks import (
    RetryConditionCallable,
    StateHookCallable,
    Task,
    TaskRunNameValueOrCallable,
)
from prefect.utilities.annotations import NotSet


def get_run_logger() -> logging.Logger:
    """Return the active Prefect run logger as a `logging.Logger`.

    Must be called inside a Prefect flow or task run; raises otherwise
    (this is the same precondition as `prefect.logging.get_run_logger`).
    """
    return cast(logging.Logger, _get_run_logger())


# Module-level ParamSpec / TypeVar declarations rather than PEP-695 generics
# (`def task[**P, R](...)`). The two are semantically equivalent, but Pylance
# fails to flow type vars through the two-step decorator-factory application
# in the second overload (`@task(...)` form) when they're declared per-function:
# the decorated function gets typed as a bare function, so `.submit` etc. are
# invisible. The classic ParamSpec/TypeVar form has been around longer and is
# resolved correctly.
_P = ParamSpec("_P")
_R = TypeVar("_R")


# ---------------------------------------------------------------------------
# Cache key: (rerun_token OR short flow id) :: task name :: hashed parameters.
#
# - No token  -> key embeds a short `flow_<8>` derived from the current
#                flow_run.id -> nothing in the cache matches -> task runs ->
#                result is persisted under that short id.
# - With token -> key is stable across runs that share the token -> previously
#                completed tasks are returned from cache and skipped.
# ---------------------------------------------------------------------------
def _stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _short_run_id() -> str:
    # Short, human-typable identifier derived from the current flow_run.id.
    # Used as the default rerun token so users can copy/paste it from the log.
    # 32 bits of entropy is enough for cache-key disambiguation in practice.
    return f"flo{prefect.runtime.flow_run.id[:8]}"


def rerun_aware_cache_key(context: TaskRunContext, parameters: dict[str, Any]) -> str | None:
    flow_params = prefect.runtime.flow_run.parameters or {}
    token = flow_params.get("rerun_token") or _short_run_id()
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
def task(__fn: Callable[_P, _R], /) -> Task[_P, _R]: ...
@overload
def task(
    __fn: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: Iterable[str] | None = None,
    version: str | None = None,
    cache_policy: CachePolicy | type[NotSet] = NotSet,
    cache_expiration: datetime.timedelta | None = None,
    task_run_name: TaskRunNameValueOrCallable | None = None,
    retries: int | None = None,
    retry_delay_seconds: float | int | list[float] | Callable[[int], list[float]] | None = None,
    retry_jitter_factor: float | None = None,
    result_storage: ResultStorage | None = None,
    result_storage_key: str | None = None,
    result_serializer: ResultSerializer | None = None,
    cache_result_in_memory: bool = True,
    timeout_seconds: int | float | None = None,
    log_prints: bool | None = None,
    refresh_cache: bool | None = None,
    on_completion: list[StateHookCallable] | None = None,
    on_failure: list[StateHookCallable] | None = None,
    on_running: list[StateHookCallable] | None = None,
    retry_condition_fn: RetryConditionCallable | None = None,
    viz_return_value: Any = None,
    asset_deps: list[str | Asset] | None = None,
) -> Callable[[Callable[_P, _R]], Task[_P, _R]]: ...
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
    # Strip the `flow_` prefix from the token when embedding into the run name
    # so we don't get redundant "fresh-flow_xxxx" / "rerun-flow_xxxx".
    token = params.get("rerun_token") or _short_run_id()
    short = token.removeprefix("flow_")
    kind = "rerun" if params.get("rerun_token") else "fresh"
    return f"{flow_name}-{kind}-{short}"


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

    On a fresh run, prints the short `flow_<8>` form of the current
    flow_run.id so the user knows the value to pass back as `rerun_token`
    to skip already-completed tasks. On a resume, confirms which token is
    being used.
    """
    log = get_run_logger()
    if rerun_token is None:
        log.info(
            f"Fresh run. To resume if interrupted, rerun with rerun_token='{_short_run_id()}'."
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
def flow(__fn: Callable[_P, _R], /) -> Flow[_P, _R]: ...
@overload
def flow(
    __fn: None = None,
    /,
    *,
    name: str | None = None,
    version: str | None = None,
    retries: int | None = None,
    retry_delay_seconds: int | float | None = None,
    task_runner: TaskRunner[PrefectFuture[Any]] | None = None,
    description: str | None = None,
    timeout_seconds: int | float | None = None,
    validate_parameters: bool = True,
    persist_result: bool | None = None,
    result_storage: ResultStorage | None = None,
    result_serializer: ResultSerializer | None = None,
    cache_result_in_memory: bool = True,
    log_prints: bool | None = None,
    on_completion: list[FlowStateHook[..., Any]] | None = None,
    on_failure: list[FlowStateHook[..., Any]] | None = None,
    on_cancellation: list[FlowStateHook[..., Any]] | None = None,
    on_crashed: list[FlowStateHook[..., Any]] | None = None,
    on_running: list[FlowStateHook[..., Any]] | None = None,
) -> Callable[[Callable[_P, _R]], Flow[_P, _R]]: ...
def flow(__fn: Any = None, /, **kwargs: Any) -> Any:
    """
    Wrapper around @flow with sensible defaults for a machine learning workflow.
    - does not depend on the code to check if the flow needs a rerun. The code is typically
      called separately in a slurm script.
    - expects an additional (optional) parameter `rerun_token` in the flow signature, which is
      used to determine caching keys for tasks. See `rerun_aware_cache_key` for details.

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
