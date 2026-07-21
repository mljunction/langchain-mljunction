"""LangChain callback handler that reports agent structure to ML Junction.

A chat model only ever sees its own single call. The shape of an agent run -
which agent delegated to which subagent, which tool ran inside which step -
exists only in LangChain's callback stream, where every event carries a
`run_id` and a `parent_run_id`. This handler turns that stream into spans.

Two id systems, kept deliberately separate:

    execution     trace_id / span_id / parent_span_id  - builds the tree
    correlation   session_id / request_id / app_id     - finds the tree

Only the execution ids define parentage. Correlation ids say which session or
gateway request a span belongs to; they never imply a parent-child link.

Everything here is best-effort. Tracing must never crash, block, or slow the
agent it is observing: serialisation is defensive, export is off-thread, and
a dead endpoint costs a log line rather than a raised exception.
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import queue
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

__all__ = [
    "MLJunctionTracer",
    "jsonable",
    "new_id",
    "normalize_usage",
]

# Metadata namespace. Everything the SDK reads or writes on a RunnableConfig
# is prefixed, so it can never collide with the customer's own metadata.
NS = "mljunction"


def new_id() -> str:
    """UUIDv7 where available (time-ordered, nicer as a database key)."""
    factory = getattr(uuid, "uuid7", uuid.uuid4)
    return str(factory())


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


_REDACTED_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "password",
        "secret",
        "client_secret",
        "access_token",
        "refresh_token",
        "private_key",
        "mljunction_api_key",
    }
)


def _should_redact(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return (
        normalized in _REDACTED_KEYS
        or normalized.endswith("_api_key")
        or normalized.endswith("_password")
        or normalized.endswith("_secret")
        or normalized.endswith("_token")
    )


def jsonable(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 12,
    max_string_length: int = 20_000,
    max_collection_length: int = 500,
) -> Any:
    """Convert arbitrary Python objects into safe JSON values.

    Deliberately total: every branch either returns a JSON-safe value or falls
    through to a repr. Nothing raises. A customer's exotic object graph, a
    __repr__ that throws, a cycle deeper than max_depth - all of these produce
    a placeholder string rather than an exception inside a callback.
    """
    if depth > max_depth:
        return "<maximum-depth-reached>"

    if value is None or isinstance(value, bool | int | float):
        return value

    if isinstance(value, str):
        if len(value) > max_string_length:
            return value[:max_string_length] + "<truncated>"
        return value

    if isinstance(value, UUID | datetime):
        return str(value)

    if isinstance(value, bytes | bytearray):
        return f"<bytes:{len(value)}>"

    if isinstance(value, BaseMessage):
        try:
            return jsonable(value.model_dump(mode="json"), depth=depth + 1, max_depth=max_depth)
        except Exception:
            return {"type": value.type, "content": jsonable(value.content, depth=depth + 1)}

    if hasattr(value, "model_dump"):
        try:
            return jsonable(value.model_dump(mode="json"), depth=depth + 1, max_depth=max_depth)
        except Exception:
            pass

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return jsonable(dataclasses.asdict(value), depth=depth + 1, max_depth=max_depth)
        except Exception:
            pass

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= max_collection_length:
                output["<truncated>"] = f"{len(value) - max_collection_length} additional entries"
                break
            key = str(raw_key)
            output[key] = "<redacted>" if _should_redact(key) else jsonable(
                item, depth=depth + 1, max_depth=max_depth
            )
        return output

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        items = list(value)
        result: list[Any] = [
            jsonable(item, depth=depth + 1, max_depth=max_depth)
            for item in items[:max_collection_length]
        ]
        if len(items) > max_collection_length:
            result.append(f"<truncated:{len(items) - max_collection_length}>")
        return result

    try:
        text = repr(value)
    except Exception:
        text = f"<unserializable:{type(value).__name__}>"
    if len(text) > max_string_length:
        text = text[:max_string_length] + "<truncated>"
    return text


def merge_numeric(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Add token counts together, recursing into nested detail dictionaries."""
    for key, value in source.items():
        if isinstance(value, Mapping):
            child = target.setdefault(key, {})
            if isinstance(child, dict):
                merge_numeric(child, value)
        elif isinstance(value, int | float):
            current = target.get(key, 0)
            if isinstance(current, int | float):
                target[key] = current + value
        elif key not in target:
            target[key] = jsonable(value)


def normalize_usage(raw_usage: Mapping[str, Any]) -> dict[str, Any]:
    """Reconcile OpenAI-style and LangChain-style token names."""
    usage = dict(raw_usage)
    normalized: dict[str, Any] = {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
        "total_tokens": usage.get("total_tokens", 0),
    }
    if not normalized["total_tokens"]:
        normalized["total_tokens"] = normalized["input_tokens"] + normalized["output_tokens"]

    input_details = usage.get("input_token_details") or usage.get("prompt_tokens_details")
    output_details = usage.get("output_token_details") or usage.get("completion_tokens_details")
    if input_details:
        normalized["input_token_details"] = jsonable(input_details)
    if output_details:
        normalized["output_token_details"] = jsonable(output_details)
    return normalized


def extract_llm_result(
    response: LLMResult,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Pull generations, merged usage, and model metadata out of an LLMResult.

    ChatMLJunction populates `usage_metadata` on every AIMessage, so the
    message path is the one that normally fires; the llm_output fallback exists
    for other model classes sharing this tracer.
    """
    generations_payload: list[dict[str, Any]] = []
    combined_usage: dict[str, Any] = {}
    model_data: dict[str, Any] = {}

    for group in response.generations:
        for generation in group:
            message = getattr(generation, "message", None)
            payload: dict[str, Any] = {
                "text": getattr(generation, "text", None),
                "generation_info": jsonable(getattr(generation, "generation_info", None)),
            }
            if message is not None:
                payload["message"] = jsonable(message)
                usage_metadata = getattr(message, "usage_metadata", None)
                if usage_metadata:
                    merge_numeric(combined_usage, usage_metadata)
                metadata = dict(getattr(message, "response_metadata", None) or {})
                if metadata:
                    for source, target in (
                        ("model_name", "name"),
                        ("model", "name"),
                        ("provider", "provider"),
                        ("finish_reason", "finish_reason"),
                    ):
                        if metadata.get(source) and target not in model_data:
                            model_data[target] = metadata[source]
                    # UnifiedResponse.id IS the gateway request_id. This is the
                    # seam that ties a customer-side span to a billable request.
                    response_id = metadata.get("id") or metadata.get("response_id")
                    if response_id:
                        model_data["request_id"] = str(response_id)
                    for key in ("session_id", "task_id", "session_name", "task_name"):
                        if metadata.get(key) and key not in model_data:
                            model_data[key] = metadata[key]
            generations_payload.append(payload)

    llm_output = response.llm_output or {}
    combined_usage = (
        normalize_usage(combined_usage)
        if combined_usage
        else normalize_usage(llm_output.get("token_usage") or llm_output.get("usage") or {})
        if (llm_output.get("token_usage") or llm_output.get("usage"))
        else {}
    )
    if "name" not in model_data:
        name = llm_output.get("model_name") or llm_output.get("model")
        if name:
            model_data["name"] = name
    if provider := llm_output.get("provider"):
        model_data.setdefault("provider", provider)
    return generations_payload, combined_usage, model_data


class BatchExporter:
    """Non-blocking batch exporter.

    Callbacks fire on the agent's own thread, often inside a hot loop. Doing
    HTTP there would add a round trip to every tool call, so callbacks only
    enqueue and a daemon worker does the sending.

    The queue is bounded. When it fills, events are dropped and counted rather
    than blocking the agent - losing telemetry is always preferable to stalling
    the thing being observed.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        app_id: str | None = None,
        batch_size: int = 50,
        flush_interval_seconds: float = 1.0,
        queue_size: int = 10_000,
        timeout_seconds: float = 10.0,
        user_agent: str = "langchain-mljunction",
    ) -> None:
        self.endpoint = endpoint.rstrip("/") + "/v1/trace-events/batch"
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds

        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_size)
        self._closed = False
        self._dropped_events = 0
        self._failed_batches = 0

        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": user_agent,
            "Content-Type": "application/json",
        }
        if app_id:
            headers["X-App"] = app_id
        self._client = httpx.Client(timeout=timeout_seconds, headers=headers)

        self._worker = threading.Thread(
            target=self._run, name="mljunction-trace-exporter", daemon=True
        )
        self._worker.start()
        atexit.register(self.close)

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    @property
    def failed_batches(self) -> int:
        return self._failed_batches

    def emit(self, event: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped_events += 1

    def _run(self) -> None:
        stopping = False
        while not stopping:
            item = self._queue.get()
            if item is None:
                break
            batch = [item]
            deadline = time.monotonic() + self.flush_interval_seconds
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    stopping = True
                    break
                batch.append(item)
            self._send(batch)

    def _send(self, events: list[dict[str, Any]]) -> None:
        for attempt in range(3):
            try:
                response = self._client.post(
                    self.endpoint,
                    json={"schema_version": "1.0", "events": events},
                )
                response.raise_for_status()
                return
            except Exception as error:
                if attempt == 2:
                    self._failed_batches += 1
                    # Reported, never raised. An unreachable tracing endpoint
                    # must not become an exception inside the customer's agent.
                    print(
                        f"ML Junction tracing export failed after 3 attempts: "
                        f"{type(error).__name__}: {error}",
                        file=sys.stderr,
                    )
                    return
                time.sleep(0.25 * (2**attempt))

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until the queue drains. For tests and short-lived scripts."""
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        # The worker may still be mid-POST after the queue empties.
        time.sleep(0.05)
        return self._queue.empty()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(queue.Full):
            self._queue.put(None, timeout=1)
        self._worker.join(timeout=5)
        self._client.close()


@dataclass
class RunState:
    """In-flight span. Lives from the start callback to the end/error one."""

    trace_id: str
    parent_span_id: str | None
    started_at: str
    started_monotonic: float
    span_kind: str
    name: str
    context: dict[str, Any]
    first_token_monotonic: float | None = None
    stream_chunks: int = 0
    # Point-in-time occurrences, flushed with the terminal event. Sending them
    # cumulatively (rather than as their own events) is what lets the server
    # use replace-semantics and stay idempotent under retry.
    events: list[dict[str, Any]] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


class MLJunctionTracer(BaseCallbackHandler):
    """Callback handler that exports a span tree to ML Junction.

    Attach it once, at the top of a run, and LangChain propagates it to every
    child runnable automatically. The one case it cannot follow on its own is
    an agent invoked by hand inside a tool function - see
    `MLJunction.child_agent_config`.
    """

    # run_inline keeps ordering deterministic: span starts must be recorded
    # before any child callback looks up its parent.
    run_inline = True
    # A raising callback would take the customer's agent down with it.
    raise_error = False

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        app_id: str | None = None,
        app_name: str | None = None,
        environment: str = "production",
        capture_content: bool = True,
        exporter: BatchExporter | None = None,
    ) -> None:
        self.app_id = app_id
        self.app_name = app_name
        self.environment = environment
        self.capture_content = capture_content
        self.exporter = exporter or BatchExporter(
            endpoint=endpoint, api_key=api_key, app_id=app_id
        )
        self._runs: dict[str, RunState] = {}
        self._lock = threading.RLock()

    def close(self) -> None:
        self.exporter.close()

    def flush(self, timeout: float = 5.0) -> bool:
        return self.exporter.flush(timeout)

    # -- context -----------------------------------------------------------

    def _context(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        """Read the SDK's reserved keys off a RunnableConfig's metadata.

        Note what is absent: request_id. In ML Junction a request id is minted
        by the gateway per HTTP call, so one agent run produces many of them.
        It is attached per-LLM-span from the response, never declared up front.
        """
        metadata = metadata or {}
        raw_depth = metadata.get(f"{NS}.agent_depth")
        try:
            agent_depth = int(raw_depth) if raw_depth is not None else None
        except (TypeError, ValueError):
            agent_depth = None
        return {
            "app_id": metadata.get(f"{NS}.app_id", self.app_id),
            "session_id": metadata.get(f"{NS}.session_id"),
            "agent_name": metadata.get(f"{NS}.agent_name"),
            "agent_role": metadata.get(f"{NS}.agent_role"),
            "agent_instance_id": metadata.get(f"{NS}.agent_instance_id"),
            "agent_depth": agent_depth,
        }

    @staticmethod
    def _name(
        serialized: Mapping[str, Any] | None,
        fallback: str,
        explicit_name: str | None = None,
    ) -> str:
        if explicit_name:
            return str(explicit_name)
        if serialized:
            if name := serialized.get("name"):
                return str(name)
            identifier = serialized.get("id")
            if isinstance(identifier, list) and identifier:
                return str(identifier[-1])
        return fallback

    # -- span lifecycle ----------------------------------------------------

    def _start(
        self,
        *,
        run_id: UUID,
        parent_run_id: UUID | None,
        span_kind: str,
        name: str,
        input_data: Any,
        tags: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        span_id = str(run_id)
        parent_span_id = str(parent_run_id) if parent_run_id else None
        started_at = utc_now()
        started_monotonic = time.monotonic()

        with self._lock:
            parent = self._runs.get(parent_span_id) if parent_span_id else None
            context = self._context(metadata)
            # A span with no known parent starts its own trace. That is also
            # what happens when a subagent is invoked without the active
            # config - the tree silently splits in two. See child_agent_config.
            trace_id = parent.trace_id if parent else span_id

            # Inherit correlation downward: an inner tool span usually carries
            # no metadata of its own, but it belongs to the same session and
            # agent as whatever started it.
            if parent:
                for key, value in parent.context.items():
                    if context.get(key) is None:
                        context[key] = value

            effective_kind = span_kind
            if span_kind == "chain":
                current_agent = context.get("agent_instance_id")
                parent_agent = parent.context.get("agent_instance_id") if parent else None
                # Ownership changed, so this chain IS an agent. The comparison
                # is purely local (me vs my parent), which is exactly why it
                # keeps working at any depth without special-casing a level.
                if current_agent and current_agent != parent_agent:
                    effective_kind = "agent"
                    # Fall back to the auto-detected span name when the caller
                    # never supplied one. LangChain gives us the function,
                    # graph-node or class name for free, which beats showing an
                    # unlabelled agent in the tree.
                    if not context.get("agent_name"):
                        context["agent_name"] = name

            state = RunState(
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                started_at=started_at,
                started_monotonic=started_monotonic,
                span_kind=effective_kind,
                name=name,
                context=context,
                attributes=dict(attributes or {}),
            )
            self._runs[span_id] = state

        event: dict[str, Any] = {
            "event_type": "span.start",
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "span_kind": effective_kind,
            "name": name,
            "timestamp": started_at,
            "started_at": started_at,
            "attributes": jsonable(
                {
                    **(attributes or {}),
                    "tags": list(tags or []),
                    "environment": self.environment,
                    **({"app_name": self.app_name} if self.app_name else {}),
                }
            ),
            **{k: v for k, v in context.items() if v is not None},
        }
        if self.capture_content:
            event["input_payload"] = {"value": jsonable(input_data)}
        self.exporter.emit(event)

    def _finish(
        self,
        *,
        run_id: UUID,
        output: Any,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        span_id = str(run_id)
        ended_monotonic = time.monotonic()
        ended_at = utc_now()

        with self._lock:
            state = self._runs.pop(span_id, None)
        if state is None:
            return

        event: dict[str, Any] = {
            "event_type": "span.end",
            "trace_id": state.trace_id,
            "span_id": span_id,
            "parent_span_id": state.parent_span_id,
            "span_kind": state.span_kind,
            "name": state.name,
            "timestamp": ended_at,
            "started_at": state.started_at,
            "ended_at": ended_at,
            "duration_ms": round((ended_monotonic - state.started_monotonic) * 1000, 3),
            "events": jsonable(state.events),
            "attributes": jsonable(
                {
                    **state.attributes,
                    **(attributes or {}),
                    "stream_chunks": state.stream_chunks,
                }
            ),
            **{k: v for k, v in state.context.items() if v is not None},
        }
        if state.first_token_monotonic is not None:
            event["ttft_ms"] = round(
                (state.first_token_monotonic - state.started_monotonic) * 1000, 3
            )
        # An LLM span learns its gateway request_id only from the response, so
        # it overrides any inherited value here.
        if attributes and attributes.get("request_id"):
            event["request_id"] = str(attributes["request_id"])
        if self.capture_content:
            event["output_payload"] = {"value": jsonable(output)}
        self.exporter.emit(event)

    def _error(self, *, run_id: UUID, error: BaseException) -> None:
        span_id = str(run_id)
        ended_monotonic = time.monotonic()
        ended_at = utc_now()

        with self._lock:
            state = self._runs.pop(span_id, None)
        if state is None:
            return

        self.exporter.emit(
            {
                "event_type": "span.error",
                "trace_id": state.trace_id,
                "span_id": span_id,
                "parent_span_id": state.parent_span_id,
                "span_kind": state.span_kind,
                "name": state.name,
                "timestamp": ended_at,
                "started_at": state.started_at,
                "ended_at": ended_at,
                "duration_ms": round((ended_monotonic - state.started_monotonic) * 1000, 3),
                "events": jsonable(state.events),
                "attributes": jsonable(state.attributes),
                "error_message": f"{type(error).__name__}: {error}"[:2000],
                "error_payload": {
                    "type": type(error).__name__,
                    "message": str(error)[:4000],
                    "stack": "".join(
                        traceback.format_exception(type(error), error, error.__traceback__)
                    )[:20_000],
                },
                **{k: v for k, v in state.context.items() if v is not None},
            }
        )

    def _span_event(self, *, run_id: UUID, name: str, data: Any) -> None:
        """Record something that happened inside a span but has no duration."""
        span_id = str(run_id)
        with self._lock:
            state = self._runs.get(span_id)
            if state is None:
                return
            # Bounded: a retry storm must not grow this list without limit.
            if len(state.events) < 200:
                state.events.append({"name": name, "at": utc_now(), "data": jsonable(data)})

    # -- chains ------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_kind="chain",
            name=self._name(serialized, "chain", kwargs.get("name")),
            input_data=inputs,
            tags=tags,
            metadata=metadata,
        )

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id=run_id, output=outputs)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._error(run_id=run_id, error=error)

    # -- models ------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        invocation = kwargs.get("invocation_params") or {}
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_kind="llm",
            name=self._name(serialized, "chat model", invocation.get("model")),
            input_data=messages,
            tags=tags,
            metadata=metadata,
            attributes={"invocation_params": jsonable(invocation)},
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_kind="llm",
            name=self._name(serialized, "llm", kwargs.get("name")),
            input_data=prompts,
            tags=tags,
            metadata=metadata,
        )

    def on_llm_new_token(self, token: str, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            state = self._runs.get(str(run_id))
            if state is None:
                return
            state.stream_chunks += 1
            if state.first_token_monotonic is None:
                state.first_token_monotonic = time.monotonic()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        generations, usage, model = extract_llm_result(response)
        attributes: dict[str, Any] = {"usage": usage, "model": model}
        if request_id := model.get("request_id"):
            attributes["request_id"] = request_id
        self._finish(run_id=run_id, output=generations, attributes=attributes)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._error(run_id=run_id, error=error)

    # -- tools -------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_kind="tool",
            name=self._name(serialized, "tool", kwargs.get("name")),
            input_data=inputs if inputs is not None else input_str,
            tags=tags,
            metadata=metadata,
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id=run_id, output=output)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._error(run_id=run_id, error=error)

    # -- retrievers --------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_kind="retriever",
            name=self._name(serialized, "retriever", kwargs.get("name")),
            input_data=query,
            tags=tags,
            metadata=metadata,
        )

    def on_retriever_end(self, documents: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(
            run_id=run_id,
            output=documents,
            attributes={
                "document_count": len(documents) if hasattr(documents, "__len__") else None
            },
        )

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._error(run_id=run_id, error=error)

    # -- agent decisions ---------------------------------------------------

    def on_agent_action(self, action: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._span_event(
            run_id=run_id,
            name="agent_action",
            data={
                "tool": getattr(action, "tool", None),
                "tool_input": getattr(action, "tool_input", None),
            },
        )

    def on_agent_finish(self, finish: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._span_event(
            run_id=run_id,
            name="agent_finish",
            data={"return_values": getattr(finish, "return_values", None)},
        )

    def on_retry(self, retry_state: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._span_event(
            run_id=run_id,
            name="retry",
            data={"attempt": getattr(retry_state, "attempt_number", None)},
        )
