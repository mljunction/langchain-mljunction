"""`@traceable`: make a plain Python function a span, LangSmith-style.

LangChain only reports runnables. A helper such as `def plan_day(...)` that
calls the model twice shows up as two loose model calls. Decorating it opens a
chain run around the body and makes that run the parent of every LangChain call
made inside, so the tree reads `plan_day > model, model`.

It goes through LangChain's own callback manager rather than the tracer, so it
nests under whatever run is active and also shows up in any other tracer
attached to the run (LangSmith included).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, overload

from langchain_core.runnables.config import (
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
    var_child_runnable_config,
)

__all__ = ["traceable"]

F = TypeVar("F", bound=Callable[..., Any])

# Never recorded as inputs: they are plumbing, not what the function was given.
_SKIPPED_ARGUMENTS = frozenset({"self", "cls", "config", "run_manager", "callbacks"})


def _inputs(signature: inspect.Signature, args: tuple, kwargs: dict) -> dict[str, Any]:
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        return {"args": list(args), "kwargs": kwargs}
    return {k: v for k, v in bound.arguments.items() if k not in _SKIPPED_ARGUMENTS}


def _outputs(result: Any) -> dict[str, Any]:
    return result if isinstance(result, dict) else {"output": result}


@overload
def traceable(func: F) -> F: ...
@overload
def traceable(
    *,
    name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]: ...


def traceable(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Record each call of a sync or async function as a named span."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        run_name = name or fn.__name__
        signature = inspect.signature(fn)

        def config() -> Any:
            # ensure_config reads the active run's config, so a call made inside
            # a graph node lands under that node.
            base = ensure_config()
            return {
                **base,
                "tags": [*(base.get("tags") or []), *(tags or [])],
                "metadata": {**(base.get("metadata") or {}), **(metadata or {})},
            }

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                parent = config()
                manager = get_async_callback_manager_for_config(parent)
                run = await manager.on_chain_start(
                    None, _inputs(signature, args, kwargs), name=run_name
                )
                token = var_child_runnable_config.set(
                    patch_config(parent, callbacks=run.get_child())
                )
                try:
                    result = await fn(*args, **kwargs)
                except BaseException as error:
                    await run.on_chain_error(error)
                    raise
                finally:
                    var_child_runnable_config.reset(token)
                await run.on_chain_end(_outputs(result))
                return result

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            parent = config()
            manager = get_callback_manager_for_config(parent)
            run = manager.on_chain_start(None, _inputs(signature, args, kwargs), name=run_name)
            token = var_child_runnable_config.set(patch_config(parent, callbacks=run.get_child()))
            try:
                result = fn(*args, **kwargs)
            except BaseException as error:
                run.on_chain_error(error)
                raise
            finally:
                var_child_runnable_config.reset(token)
            run.on_chain_end(_outputs(result))
            return result

        return wrapper

    return decorate(func) if func is not None else decorate
