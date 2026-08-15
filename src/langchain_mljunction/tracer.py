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

import contextlib
import dataclasses
import json
import re
import threading
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

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


_SECRET_TEXT = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+|\b(sk-[A-Za-z0-9_-]{12,})\b")


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
        value = _SECRET_TEXT.sub(lambda match: f"{match.group(1) or ''}[REDACTED]", value)
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
            output[key] = (
                "<redacted>"
                if _should_redact(key)
                else jsonable(item, depth=depth + 1, max_depth=max_depth)
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


class _ManagedSpanProcessor:
    """A closable processor safe to leave attached to a global provider."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._closed = False

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        if not self._closed:
            self._delegate.on_start(span, parent_context=parent_context)

    def on_end(self, span: Any) -> None:
        if not self._closed:
            self._delegate.on_end(span)

    def _on_ending(self, span: Any) -> None:
        """Forward the SDK 1.44 pre-end hook used by multi-processors."""
        if not self._closed:
            self._delegate._on_ending(span)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return bool(self._delegate.force_flush(timeout_millis)) if not self._closed else True

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._delegate.shutdown()


class _StartSnapshotProcessor(_ManagedSpanProcessor):
    """Queue an immutable start snapshot for opt-in in-flight visibility."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        if self._closed:
            return
        # SpanProcessor.on_start receives the mutable SDK Span. Snapshot it now
        # so the background batch does not observe the later end state.
        self._delegate.on_end(span._readable_span())

    def on_end(self, span: Any) -> None:
        pass

    def _on_ending(self, span: Any) -> None:
        pass


def _attribute(value: Any) -> str | bool | int | float | list[str] | None:
    """Convert arbitrary callback data into an OTel-compatible attribute."""
    safe = jsonable(value)
    if safe is None or isinstance(safe, str | bool | int | float):
        return safe
    if isinstance(safe, list) and all(isinstance(item, str) for item in safe):
        return safe
    return json.dumps(safe, separators=(",", ":"), ensure_ascii=False)


@dataclass
class RunState:
    """In-flight span. Lives from the start callback to the end/error one."""

    span: Any
    context_token: Any
    started_monotonic: float
    span_kind: str
    name: str
    context: dict[str, Any]
    first_token_monotonic: float | None = None
    stream_chunks: int = 0
    event_count: int = 0
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
        capture_content: bool = False,
        export_inflight: bool = False,
        span_exporter: Any | None = None,
        tracer_provider: Any | None = None,
    ) -> None:
        try:
            from opentelemetry import context as otel_context
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:  # pragma: no cover - exercised without the extra installed
            raise ImportError(
                "Agent tracing requires the OpenTelemetry extra: "
                "pip install 'langchain-mljunction[otel]'"
            ) from exc

        self.app_id = app_id
        self.app_name = app_name
        self.environment = environment
        self.capture_content = capture_content
        headers = {"Authorization": f"Bearer {api_key}"}
        if app_id:
            headers["X-App"] = app_id
        exporter = span_exporter or OTLPSpanExporter(
            endpoint=endpoint.rstrip("/") + "/v1/traces",
            headers=headers,
            timeout=10,
        )
        processor = _ManagedSpanProcessor(
            BatchSpanProcessor(
                exporter,
                max_export_batch_size=50,
                schedule_delay_millis=1_000,
                max_queue_size=10_000,
                export_timeout_millis=10_000,
            )
        )
        provider = tracer_provider
        if provider is None:
            current = trace.get_tracer_provider()
            if hasattr(current, "add_span_processor"):
                provider = current
            else:
                resource_attributes: dict[str, Any] = {
                    "service.name": app_name or app_id or "langchain-mljunction",
                    "deployment.environment.name": environment,
                }
                if app_id:
                    resource_attributes["service.instance.id"] = app_id
                    resource_attributes["mlj.app.id"] = app_id
                provider = TracerProvider(resource=Resource.create(resource_attributes))
                trace.set_tracer_provider(provider)
        provider.add_span_processor(processor)
        inflight_processor = None
        if export_inflight:
            inflight_exporter = span_exporter or OTLPSpanExporter(
                endpoint=endpoint.rstrip("/") + "/v1/traces",
                headers=headers,
                timeout=10,
            )
            inflight_processor = _StartSnapshotProcessor(
                BatchSpanProcessor(
                    inflight_exporter,
                    max_export_batch_size=50,
                    schedule_delay_millis=1_000,
                    max_queue_size=10_000,
                    export_timeout_millis=10_000,
                )
            )
            provider.add_span_processor(inflight_processor)
        self._provider = provider
        self._processor = processor
        self._inflight_processor = inflight_processor
        self._export_inflight = export_inflight
        self._trace = trace
        self._otel_context = otel_context
        self._tracer = provider.get_tracer("langchain-mljunction", "0.1.0")
        self._runs: dict[str, RunState] = {}
        self._lock = threading.RLock()

    def close(self) -> None:
        self._processor.shutdown()
        if self._inflight_processor is not None:
            self._inflight_processor.shutdown()

    def flush(self, timeout: float = 5.0) -> bool:
        timeout_millis = max(1, int(timeout * 1000))
        complete = self._processor.force_flush(timeout_millis)
        inflight = (
            self._inflight_processor.force_flush(timeout_millis)
            if self._inflight_processor is not None
            else True
        )
        return complete and inflight

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
        run_key = str(run_id)
        parent_key = str(parent_run_id) if parent_run_id else None
        started_monotonic = time.monotonic()

        with self._lock:
            parent = self._runs.get(parent_key) if parent_key else None
            context = self._context(metadata)
            # A span with no known parent starts its own trace. That is also
            # what happens when a subagent is invoked without the active
            # config - the tree silently splits in two. See child_agent_config.
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

            initial_attributes: dict[str, Any] = {
                "gen_ai.operation.name": effective_kind,
                "deployment.environment.name": self.environment,
                "mlj.tags": list(tags or []),
                **({"mlj.in_flight": True} if self._export_inflight else {}),
                **({"service.name": self.app_name} if self.app_name else {}),
                **{
                    f"mlj.agent.{key.removeprefix('agent_')}": value
                    for key, value in context.items()
                    if key.startswith("agent_") and value is not None
                },
                **{
                    {
                        "app_id": "mlj.app.id",
                        "session_id": "mlj.session.id",
                    }[key]: value
                    for key, value in context.items()
                    if key in {"app_id", "session_id"} and value is not None
                },
            }
            for key, value in (attributes or {}).items():
                initial_attributes[f"mlj.attribute.{key}"] = value
            if self.capture_content:
                initial_attributes["mlj.capture.input"] = input_data
            otel_attributes = {
                key: converted
                for key, value in initial_attributes.items()
                if (converted := _attribute(value)) is not None
            }
            parent_context = (
                self._trace.set_span_in_context(parent.span)
                if parent
                else self._otel_context.get_current()
            )
            span = self._tracer.start_span(
                name,
                context=parent_context,
                attributes=otel_attributes,
                start_time=time.time_ns(),
            )
            token = self._otel_context.attach(self._trace.set_span_in_context(span))
            state = RunState(
                span=span,
                context_token=token,
                started_monotonic=started_monotonic,
                span_kind=effective_kind,
                name=name,
                context=context,
                attributes=dict(attributes or {}),
            )
            self._runs[run_key] = state

    def _finish(
        self,
        *,
        run_id: UUID,
        output: Any,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        span_id = str(run_id)
        with self._lock:
            state = self._runs.pop(span_id, None)
        if state is None:
            return

        span = state.span
        if self._export_inflight:
            span.set_attribute("mlj.in_flight", False)
        span.set_attribute("mlj.stream_chunks", state.stream_chunks)
        if state.first_token_monotonic is not None:
            span.set_attribute(
                "mlj.ttft_ms",
                round((state.first_token_monotonic - state.started_monotonic) * 1000, 3),
            )
        # An LLM span learns its gateway request_id only from the response, so
        # it overrides any inherited value here.
        if attributes and attributes.get("request_id"):
            span.set_attribute("mlj.request.id", str(attributes["request_id"]))
        self._set_result_attributes(span, attributes or {})
        if self.capture_content:
            span.set_attribute("mlj.capture.output", _attribute(output) or "null")
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.OK))
        span.end(end_time=time.time_ns())
        with contextlib.suppress(ValueError, RuntimeError):
            self._otel_context.detach(state.context_token)

    def _error(self, *, run_id: UUID, error: BaseException) -> None:
        span_id = str(run_id)
        with self._lock:
            state = self._runs.pop(span_id, None)
        if state is None:
            return

        span = state.span
        if self._export_inflight:
            span.set_attribute("mlj.in_flight", False)
        from opentelemetry.trace import Status, StatusCode

        description = (
            f"{type(error).__name__}: {error}"[:2000]
            if self.capture_content
            else type(error).__name__
        )
        span.set_status(Status(StatusCode.ERROR, description=description))
        if self.capture_content:
            span.set_attribute(
                "mlj.capture.error",
                _attribute(
                    {
                        "type": type(error).__name__,
                        "message": str(error)[:4000],
                        "stack": "".join(
                            traceback.format_exception(type(error), error, error.__traceback__)
                        )[:20_000],
                    }
                )
                or "{}",
            )
        span.end(end_time=time.time_ns())
        with contextlib.suppress(ValueError, RuntimeError):
            self._otel_context.detach(state.context_token)

    def _span_event(self, *, run_id: UUID, name: str, data: Any) -> None:
        """Record something that happened inside a span but has no duration."""
        span_id = str(run_id)
        with self._lock:
            state = self._runs.get(span_id)
            if state is None:
                return
            # Bounded: a retry storm must not grow this list without limit.
            if state.event_count < 200:
                state.span.add_event(
                    name,
                    attributes={"mlj.event.data": _attribute(data) or "null"},
                    timestamp=time.time_ns(),
                )
                state.event_count += 1

    @staticmethod
    def _set_result_attributes(span: Any, attributes: Mapping[str, Any]) -> None:
        usage = attributes.get("usage")
        if isinstance(usage, Mapping):
            for source, target in (
                ("input_tokens", "gen_ai.usage.input_tokens"),
                ("output_tokens", "gen_ai.usage.output_tokens"),
                ("total_tokens", "gen_ai.usage.total_tokens"),
            ):
                value = usage.get(source)
                if isinstance(value, int | float):
                    span.set_attribute(target, value)
        model = attributes.get("model")
        if isinstance(model, Mapping):
            if model.get("name"):
                span.set_attribute("gen_ai.request.model", str(model["name"]))
            if model.get("provider"):
                span.set_attribute("gen_ai.system", str(model["provider"]))
            if model.get("finish_reason"):
                span.set_attribute("gen_ai.response.finish_reasons", [str(model["finish_reason"])])
        for key, value in attributes.items():
            if key in {"usage", "model", "request_id"}:
                continue
            converted = _attribute(value)
            if converted is not None:
                span.set_attribute(f"mlj.attribute.{key}", converted)

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
