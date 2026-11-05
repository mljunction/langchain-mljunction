"""Process-wide tracing, on by default, the way LangSmith's env switch works.

LangSmith traces every LangChain run without code changes because it registers
a configure hook: whenever LangChain builds a callback manager, the hook adds
its tracer. This module does the same for ML Junction. The first ML Junction
component built in the process supplies the credentials, and from then on every
chain, LangGraph node, tool and model call becomes a span without anyone
passing `callbacks=`.

An ML Junction key in the environment is not consent on its own: an app that
keeps the key for one feature while its LangChain models run on another
provider must not start sending those runs to ML Junction just because this
package was imported. MLJUNCTION_TRACING=true opts the whole process in from
the environment key alone.

Switches:

    MLJUNCTION_TRACING=false                    turn it off for the process
    MLJUNCTION_TRACING_CAPTURE_CONTENT=false    keep structure, drop inputs/outputs
    with tracing_disabled(): ...                turn it off for one block

Like the tracer itself this is best-effort: a missing OpenTelemetry install or
a broken exporter logs once and leaves the application untouched.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any

from langchain_core.tracers.context import register_configure_hook

__all__ = ["get_tracer", "tracing_disabled", "tracing_enabled", "use_tracer"]

logger = logging.getLogger(__name__)

_FALSE = frozenset({"0", "false", "no", "off"})

_lock = threading.Lock()
_tracer: Any | None = None
_unavailable = False
_disabled: ContextVar[bool] = ContextVar("mljunction_tracing_disabled", default=False)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() not in _FALSE


def tracing_enabled() -> bool:
    """Whether automatic tracing is switched on for this process."""
    return _env_flag("MLJUNCTION_TRACING", True)


def _explicitly_enabled() -> bool:
    raw = os.getenv("MLJUNCTION_TRACING")
    return raw is not None and raw.strip().lower() not in _FALSE


def capture_content_default() -> bool:
    return _env_flag("MLJUNCTION_TRACING_CAPTURE_CONTENT", True)


def get_tracer(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    app_name: str | None = None,
    force: bool = False,
) -> Any | None:
    """Return the process tracer, creating it from these credentials if needed.

    The first caller with credentials wins: one process exports through one
    pipeline. `force` builds it even when MLJUNCTION_TRACING is off, for a
    model constructed with `tracing=True`.
    """
    global _tracer, _unavailable
    if _tracer is not None or _unavailable:
        return _tracer
    if not (force or tracing_enabled()):
        return None
    api_key = api_key or os.getenv("MLJUNCTION_API_KEY")
    if not api_key:
        return None
    with _lock:
        if _tracer is not None or _unavailable:
            return _tracer
        try:
            from langchain_mljunction.tracer import MLJunctionTracer

            tracer = MLJunctionTracer(
                endpoint=base_url or os.getenv("MLJUNCTION_BASE_URL", "https://api.mljunction.com"),
                api_key=api_key,
                app_name=app_name or os.getenv("MLJUNCTION_APP_NAME"),
                environment=os.getenv("MLJUNCTION_ENVIRONMENT", "production"),
                capture_content=capture_content_default(),
            )
        except Exception:
            _unavailable = True
            logger.warning(
                "ML Junction tracing is unavailable; continuing without it", exc_info=True
            )
            return None
        _tracer = tracer
        atexit.register(_shutdown, tracer)
        return tracer


def use_tracer(tracer: Any) -> None:
    """Make an explicitly built tracer the process tracer.

    MLJunction calls this so its own `callbacks=[tracer]` and the automatic hook
    hand LangChain the same object, which it de-duplicates by identity. Two
    different tracers would export every span twice.
    """
    global _tracer
    with _lock:
        _tracer = tracer


def _shutdown(tracer: Any) -> None:
    with contextlib.suppress(Exception):
        tracer.flush(timeout=5.0)
        tracer.close()


@contextlib.contextmanager
def tracing_disabled() -> Iterator[None]:
    """Run a block without automatic tracing."""
    token = _disabled.set(True)
    try:
        yield
    finally:
        _disabled.reset(token)


class _ProcessSlot:
    """Stands in for the ContextVar a configure hook normally reads.

    LangChain only calls `.get()` on it. A real ContextVar set in one thread or
    task is invisible to threads started elsewhere (a web server's worker pool),
    so the tracer is held process-wide and only the off switch is contextual.
    """

    def get(self) -> Any | None:
        if _disabled.get():
            return None
        if _tracer is not None:
            return _tracer
        # Only an explicit MLJUNCTION_TRACING=true lets the hook start tracing
        # from the environment key; otherwise an ML Junction component must have
        # been built first (see the module docstring).
        return get_tracer() if _explicitly_enabled() else None


register_configure_hook(_ProcessSlot(), inheritable=True)  # type: ignore[arg-type]
