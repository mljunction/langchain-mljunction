"""Offline coverage for the LangChain-to-OpenTelemetry bridge."""

from __future__ import annotations

import json
import uuid
from typing import Any, ClassVar

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from langchain_mljunction.telemetry import MAX_AGENT_DEPTH, MLJunction
from langchain_mljunction.tracer import NS, MLJunctionTracer, jsonable, normalize_usage


class RecordingExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans: Any) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


@pytest.fixture
def exporter() -> RecordingExporter:
    return RecordingExporter()


@pytest.fixture
def tracer(exporter: RecordingExporter) -> MLJunctionTracer:
    instance = MLJunctionTracer(
        endpoint="http://localhost:8001",
        api_key="mlj_test",
        app_id="test-app",
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    yield instance
    instance.close()


@pytest.fixture
def telemetry(tracer: MLJunctionTracer) -> MLJunction:
    return MLJunction(api_key="mlj_test", app_id="test-app", tracer=tracer)


def uid() -> uuid.UUID:
    return uuid.uuid4()


def exported(tracer: MLJunctionTracer, exporter: RecordingExporter) -> list[Any]:
    assert tracer.flush()
    return exporter.spans


def by_name(tracer: MLJunctionTracer, exporter: RecordingExporter, name: str) -> Any:
    return next(span for span in exported(tracer, exporter) if span.name == name)


def test_child_span_inherits_trace_and_links_to_parent(tracer, exporter):
    root, child = uid(), uid()
    tracer.on_chain_start({"name": "root"}, {}, run_id=root)
    tracer.on_tool_start({"name": "search"}, "q", run_id=child, parent_run_id=root)
    tracer.on_tool_end("result", run_id=child)
    tracer.on_chain_end({}, run_id=root)

    root_span = by_name(tracer, exporter, "root")
    child_span = by_name(tracer, exporter, "search")
    assert child_span.context.trace_id == root_span.context.trace_id
    assert child_span.parent.span_id == root_span.context.span_id


def test_orphan_span_starts_its_own_trace(tracer, exporter):
    child = uid()
    tracer.on_tool_start({"name": "orphan"}, "q", run_id=child, parent_run_id=uid())
    tracer.on_tool_end("done", run_id=child)

    span = by_name(tracer, exporter, "orphan")
    assert span.parent is None


def test_agent_boundary_and_correlation_become_semantic_attributes(
    tracer, exporter, telemetry
):
    root_config = telemetry.context(agent_name="coordinator", session_id="sess-1").config
    child_config = telemetry.child_agent_config(root_config, agent_name="researcher")
    root, child = uid(), uid()
    tracer.on_chain_start(
        {"name": "coordinator"}, {}, run_id=root, metadata=root_config["metadata"]
    )
    tracer.on_chain_start(
        {"name": "researcher"},
        {},
        run_id=child,
        parent_run_id=root,
        metadata=child_config["metadata"],
    )
    tracer.on_chain_end({}, run_id=child)
    tracer.on_chain_end({}, run_id=root)

    child_span = by_name(tracer, exporter, "researcher")
    assert child_span.attributes["gen_ai.operation.name"] == "agent"
    assert child_span.attributes["mlj.agent.name"] == "researcher"
    assert child_span.attributes["mlj.agent.role"] == "subagent"
    assert child_span.attributes["mlj.agent.depth"] == 1
    assert child_span.attributes["mlj.session.id"] == "sess-1"


@pytest.mark.parametrize("depth", [4, 8, MAX_AGENT_DEPTH])
def test_nesting_works_at_arbitrary_depth(tracer, exporter, telemetry, depth):
    config = telemetry.context(agent_name="agent-0").config
    run_ids = [uid()]
    tracer.on_chain_start(
        {"name": "agent-0"}, {}, run_id=run_ids[0], metadata=config["metadata"]
    )
    for level in range(1, depth + 1):
        config = telemetry.child_agent_config(config, agent_name=f"agent-{level}")
        run_id = uid()
        tracer.on_chain_start(
            {"name": f"agent-{level}"},
            {},
            run_id=run_id,
            parent_run_id=run_ids[-1],
            metadata=config["metadata"],
        )
        run_ids.append(run_id)
    for run_id in reversed(run_ids):
        tracer.on_chain_end({}, run_id=run_id)

    spans = exported(tracer, exporter)
    assert len(spans) == depth + 1
    assert len({span.context.trace_id for span in spans}) == 1
    assert sorted(span.attributes["mlj.agent.depth"] for span in spans) == list(
        range(depth + 1)
    )


def test_active_otel_context_preserves_trace_without_langchain_parent(
    tracer, exporter, telemetry
):
    root_config = telemetry.context(agent_name="coordinator").config
    detached = telemetry.context(agent_name="researcher").config
    root, child = uid(), uid()
    tracer.on_chain_start(
        {"name": "coordinator"}, {}, run_id=root, metadata=root_config["metadata"]
    )
    tracer.on_chain_start(
        {"name": "researcher"}, {}, run_id=child, metadata=detached["metadata"]
    )
    tracer.on_chain_end({}, run_id=child)
    tracer.on_chain_end({}, run_id=root)

    spans = exported(tracer, exporter)
    assert len({span.context.trace_id for span in spans}) == 1
    coordinator = next(span for span in spans if span.name == "coordinator")
    researcher = next(span for span in spans if span.name == "researcher")
    assert researcher.parent.span_id == coordinator.context.span_id


def test_child_agent_config_rejects_a_config_without_identity(telemetry):
    with pytest.raises(ValueError, match="no ML Junction agent identity"):
        telemetry.child_agent_config({}, agent_name="researcher")


def test_depth_limit_is_enforced(telemetry):
    config = telemetry.context(agent_name="agent-0").config
    for level in range(1, MAX_AGENT_DEPTH + 1):
        config = telemetry.child_agent_config(config, agent_name=f"agent-{level}")
    with pytest.raises(RuntimeError, match="Maximum nested agent depth exceeded"):
        telemetry.child_agent_config(config, agent_name="one-too-many")


def test_error_is_privacy_safe_by_default(tracer, exporter):
    run_id = uid()
    tracer.on_tool_start({"name": "tool"}, "secret input", run_id=run_id)
    tracer.on_tool_error(ValueError("secret failure"), run_id=run_id)

    span = by_name(tracer, exporter, "tool")
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "ValueError"
    assert "mlj.capture.input" not in span.attributes
    assert "mlj.capture.error" not in span.attributes


def test_content_capture_is_explicit(exporter):
    tracer = MLJunctionTracer(
        endpoint="http://localhost:8001",
        api_key="mlj_test",
        capture_content=True,
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    try:
        run_id = uid()
        tracer.on_tool_start({"name": "tool"}, "secret input", run_id=run_id)
        tracer.on_tool_error(ValueError("boom"), run_id=run_id)
        span = by_name(tracer, exporter, "tool")
        assert span.attributes["mlj.capture.input"] == "secret input"
        assert json.loads(span.attributes["mlj.capture.error"])["type"] == "ValueError"
    finally:
        tracer.close()


def test_streaming_usage_model_and_request_id_use_gen_ai_attributes(tracer, exporter):
    run_id = uid()
    message = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        response_metadata={
            "id": "req_abc123",
            "model_name": "gpt-4o",
            "provider": "openai",
        },
    )
    tracer.on_chat_model_start({"name": "model"}, [[]], run_id=run_id)
    for token in "abc":
        tracer.on_llm_new_token(token, run_id=run_id)
    tracer.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id
    )

    span = by_name(tracer, exporter, "model")
    assert span.attributes["mlj.request.id"] == "req_abc123"
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 4
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["mlj.ttft_ms"] >= 0
    assert span.attributes["mlj.stream_chunks"] == 3


def test_agent_action_is_an_otel_span_event(tracer, exporter):
    run_id = uid()

    class Action:
        tool = "search"
        tool_input: ClassVar[dict[str, str]] = {"q": "x"}

    tracer.on_chain_start({"name": "chain"}, {}, run_id=run_id)
    tracer.on_agent_action(Action(), run_id=run_id)
    tracer.on_chain_end({}, run_id=run_id)
    span = by_name(tracer, exporter, "chain")
    assert span.events[0].name == "agent_action"
    assert json.loads(span.events[0].attributes["mlj.event.data"])["tool"] == "search"


def test_jsonable_redacts_and_survives_hostile_objects():
    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("no repr")

    result = jsonable({"api_key": "sk-live-123", "safe": Hostile()})
    assert result["api_key"] == "<redacted>"
    assert "unserializable" in result["safe"]


def test_normalize_usage_accepts_both_naming_conventions():
    openai_style = normalize_usage({"prompt_tokens": 10, "completion_tokens": 5})
    langchain_style = normalize_usage({"input_tokens": 10, "output_tokens": 5})
    assert openai_style == langchain_style
    assert openai_style["total_tokens"] == 15


def test_automatic_names_are_preserved(tracer, exporter):
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tools import tool

    @tool
    def search_the_web(query: str) -> str:
        """Search."""
        return "result"

    def summarize_document(text: str) -> str:
        return text.upper()

    RunnableLambda(summarize_document).invoke("hi", config={"callbacks": [tracer]})
    search_the_web.invoke({"query": "x"}, config={"callbacks": [tracer]})
    names = {span.name for span in exported(tracer, exporter)}
    assert {"summarize_document", "search_the_web"} <= names


def test_batching_defaults_match_the_previous_exporter(tracer):
    delegate = tracer._processor._delegate._batch_processor
    assert delegate._max_export_batch_size == 50
    assert delegate._schedule_delay_millis == 1_000
    assert delegate._max_queue_size == 10_000
    assert delegate._export_timeout_millis == 10_000


def test_inflight_export_is_opt_in_and_uses_the_same_span_identity(exporter):
    tracer = MLJunctionTracer(
        endpoint="http://localhost:8001",
        api_key="mlj_test",
        export_inflight=True,
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    try:
        run_id = uid()
        tracer.on_chain_start({"name": "agent"}, {}, run_id=run_id)
        assert tracer.flush()
        assert len(exporter.spans) == 1
        assert exporter.spans[0].end_time is None
        assert exporter.spans[0].attributes["mlj.in_flight"] is True

        tracer.on_chain_end({}, run_id=run_id)
        assert tracer.flush()
        assert len(exporter.spans) == 2
        assert exporter.spans[1].attributes["mlj.in_flight"] is False
        assert exporter.spans[0].context == exporter.spans[1].context
    finally:
        tracer.close()


def test_metadata_namespace_is_prefixed(telemetry):
    config = telemetry.context(agent_name="a", metadata={"agent_name": "theirs"}).config
    assert config["metadata"]["agent_name"] == "theirs"
    assert config["metadata"][f"{NS}.agent_name"] == "a"
