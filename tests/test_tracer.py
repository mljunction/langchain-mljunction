"""Tracer tests.

These run entirely offline against a recording exporter. What they check is
the thing that is genuinely hard: that the span tree comes out with the right
shape, at arbitrary depth, and that failures degrade instead of propagating.
"""

from __future__ import annotations

import itertools
import uuid
from typing import Any, ClassVar

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_mljunction.telemetry import MAX_AGENT_DEPTH, MLJunction
from langchain_mljunction.tracer import (
    NS,
    BatchExporter,
    MLJunctionTracer,
    jsonable,
    normalize_usage,
)


class RecordingExporter:
    """Stands in for BatchExporter; keeps events instead of sending them."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass

    def flush(self, timeout: float = 5.0) -> bool:
        return True


@pytest.fixture
def exporter() -> RecordingExporter:
    return RecordingExporter()


@pytest.fixture
def tracer(exporter: RecordingExporter) -> MLJunctionTracer:
    return MLJunctionTracer(
        endpoint="http://localhost:8001",
        api_key="mlj_test",
        app_id="test-app",
        exporter=exporter,  # type: ignore[arg-type]
    )


@pytest.fixture
def telemetry(tracer: MLJunctionTracer) -> MLJunction:
    return MLJunction(api_key="mlj_test", app_id="test-app", tracer=tracer)


def uid() -> uuid.UUID:
    return uuid.uuid4()


def starts(exporter: RecordingExporter) -> list[dict[str, Any]]:
    return [e for e in exporter.events if e["event_type"] == "span.start"]


def by_span(exporter: RecordingExporter, span_id: str) -> list[dict[str, Any]]:
    return [e for e in exporter.events if e["span_id"] == str(span_id)]


# -- structure ------------------------------------------------------------


def test_child_span_inherits_trace_and_links_to_parent(tracer, exporter):
    root, child = uid(), uid()
    tracer.on_chain_start({"name": "root"}, {}, run_id=root)
    tracer.on_tool_start({"name": "search"}, "q", run_id=child, parent_run_id=root)

    root_event, child_event = starts(exporter)
    assert child_event["parent_span_id"] == str(root)
    assert child_event["trace_id"] == root_event["trace_id"]
    # A root span's trace id is its own span id.
    assert root_event["trace_id"] == str(root)


def test_orphan_span_starts_its_own_trace(tracer, exporter):
    """A span whose parent was never seen must still be exportable."""
    child = uid()
    tracer.on_tool_start({"name": "orphan"}, "q", run_id=child, parent_run_id=uid())

    event = starts(exporter)[0]
    assert event["trace_id"] == str(child)


def test_agent_boundary_detected_when_instance_id_changes(tracer, exporter, telemetry):
    """A chain is promoted to an agent exactly when ownership changes."""
    root_config = telemetry.context(agent_name="coordinator").config
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

    root_event, child_event = starts(exporter)
    assert root_event["span_kind"] == "agent"
    assert child_event["span_kind"] == "agent"
    assert child_event["agent_name"] == "researcher"
    assert child_event["agent_role"] == "subagent"


def test_chain_under_same_agent_stays_a_chain(tracer, exporter, telemetry):
    """Only an ownership change promotes a chain - plain nesting does not."""
    config = telemetry.context(agent_name="solo").config
    root, inner = uid(), uid()
    tracer.on_chain_start({"name": "solo"}, {}, run_id=root, metadata=config["metadata"])
    tracer.on_chain_start(
        {"name": "inner"}, {}, run_id=inner, parent_run_id=root, metadata=config["metadata"]
    )

    assert [e["span_kind"] for e in starts(exporter)] == ["agent", "chain"]


@pytest.mark.parametrize("depth", [4, 8, MAX_AGENT_DEPTH])
def test_nesting_works_at_arbitrary_depth(tracer, exporter, telemetry, depth):
    """The boundary comparison is local, so no level needs special handling."""
    config = telemetry.context(agent_name="agent-0").config
    run_ids = [uid()]
    tracer.on_chain_start({"name": "agent-0"}, {}, run_id=run_ids[0], metadata=config["metadata"])

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

    events = starts(exporter)
    assert len(events) == depth + 1
    # Every level is an agent, and every level shares one trace.
    assert {e["span_kind"] for e in events} == {"agent"}
    assert len({e["trace_id"] for e in events}) == 1
    assert [e["agent_depth"] for e in events] == list(range(depth + 1))
    # The chain of parentage is unbroken.
    for parent, child in itertools.pairwise(events):
        assert child["parent_span_id"] == parent["span_id"]


# -- the silent failure mode ----------------------------------------------


def test_subagent_without_config_silently_starts_a_new_root(tracer, exporter, telemetry):
    """The single most likely integration bug, pinned as a test.

    Invoking a subagent without threading the active config does not raise. It
    produces a second root and a flat tree - which is exactly why the docs warn
    about it and why expose_agent_as_tool exists.
    """
    root_config = telemetry.context(agent_name="coordinator").config
    root = uid()
    tracer.on_chain_start(
        {"name": "coordinator"}, {}, run_id=root, metadata=root_config["metadata"]
    )

    # The mistake: a fresh context instead of a derived child config.
    detached = telemetry.context(agent_name="researcher").config
    tracer.on_chain_start({"name": "researcher"}, {}, run_id=uid(), metadata=detached["metadata"])

    root_event, detached_event = starts(exporter)
    assert detached_event["parent_span_id"] is None
    assert detached_event["trace_id"] != root_event["trace_id"]


def test_child_agent_config_rejects_a_config_without_identity(telemetry):
    """Deriving from a bare config is refused rather than silently flattening."""
    with pytest.raises(ValueError, match="no ML Junction agent identity"):
        telemetry.child_agent_config({}, agent_name="researcher")


def test_depth_limit_is_enforced(telemetry):
    config = telemetry.context(agent_name="agent-0").config
    for level in range(1, MAX_AGENT_DEPTH + 1):
        config = telemetry.child_agent_config(config, agent_name=f"agent-{level}")

    with pytest.raises(RuntimeError, match="Maximum nested agent depth exceeded"):
        telemetry.child_agent_config(config, agent_name="one-too-many")


def test_child_config_preserves_callbacks_and_drops_run_id(telemetry):
    root = telemetry.context(agent_name="root").config
    root["run_id"] = uid()
    child = telemetry.child_agent_config(root, agent_name="child")

    assert child["callbacks"] == root["callbacks"]
    # Reusing the parent's run id would collapse parent and child into one span.
    assert "run_id" not in child


# -- lifecycle ------------------------------------------------------------


def test_end_event_carries_duration_and_status(tracer, exporter):
    run_id = uid()
    tracer.on_chain_start({"name": "c"}, {}, run_id=run_id)
    tracer.on_chain_end({"result": "ok"}, run_id=run_id)

    end = by_span(exporter, run_id)[-1]
    assert end["event_type"] == "span.end"
    assert end["duration_ms"] >= 0
    assert end["ended_at"]


def test_error_event_captures_type_and_message(tracer, exporter):
    run_id = uid()
    tracer.on_tool_start({"name": "t"}, "x", run_id=run_id)
    tracer.on_tool_error(ValueError("boom"), run_id=run_id)

    error = by_span(exporter, run_id)[-1]
    assert error["event_type"] == "span.error"
    assert "boom" in error["error_message"]
    assert error["error_payload"]["type"] == "ValueError"


def test_end_without_start_is_ignored(tracer, exporter):
    """Callbacks can arrive for runs we never saw start. That must not raise."""
    tracer.on_chain_end({}, run_id=uid())
    assert exporter.events == []


def test_streaming_records_ttft_and_chunk_count(tracer, exporter):
    run_id = uid()
    tracer.on_chat_model_start({"name": "m"}, [[]], run_id=run_id)
    for token in "abc":
        tracer.on_llm_new_token(token, run_id=run_id)
    tracer.on_llm_end(LLMResult(generations=[[]]), run_id=run_id)

    end = by_span(exporter, run_id)[-1]
    assert end["ttft_ms"] >= 0
    assert end["attributes"]["stream_chunks"] == 3


def test_llm_end_extracts_usage_and_gateway_request_id(tracer, exporter):
    """The correlation seam: UnifiedResponse.id is the gateway request id."""
    run_id = uid()
    message = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        response_metadata={"id": "req_abc123", "model_name": "gpt-4o", "session_id": "s1"},
    )
    tracer.on_chat_model_start({"name": "m"}, [[]], run_id=run_id)
    tracer.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id
    )

    end = by_span(exporter, run_id)[-1]
    assert end["request_id"] == "req_abc123"
    assert end["attributes"]["usage"]["total_tokens"] == 14
    assert end["attributes"]["model"]["name"] == "gpt-4o"


def test_agent_action_is_recorded_on_the_span(tracer, exporter):
    run_id = uid()

    class Action:
        tool = "search"
        tool_input: ClassVar[dict[str, str]] = {"q": "x"}

    tracer.on_chain_start({"name": "c"}, {}, run_id=run_id)
    tracer.on_agent_action(Action(), run_id=run_id)
    tracer.on_chain_end({}, run_id=run_id)

    end = by_span(exporter, run_id)[-1]
    assert end["events"][0]["name"] == "agent_action"
    assert end["events"][0]["data"]["tool"] == "search"


def test_correlation_is_inherited_by_child_spans(tracer, exporter, telemetry):
    """A tool span carries no metadata of its own but belongs to the session."""
    config = telemetry.context(agent_name="a", session_id="sess-1").config
    root, tool = uid(), uid()
    tracer.on_chain_start({"name": "a"}, {}, run_id=root, metadata=config["metadata"])
    tracer.on_tool_start({"name": "t"}, "x", run_id=tool, parent_run_id=root)

    assert starts(exporter)[1]["session_id"] == "sess-1"


# -- defensive behaviour ---------------------------------------------------


def test_jsonable_redacts_secrets_at_any_nesting():
    payload = {
        "api_key": "sk-live-123",
        "Authorization": "Bearer abc",
        "nested": {"openai_api_key": "sk-2", "access_token": "t"},
        "safe": "visible",
    }
    result = jsonable(payload)

    assert result["api_key"] == "<redacted>"
    assert result["Authorization"] == "<redacted>"
    assert result["nested"]["openai_api_key"] == "<redacted>"
    assert result["nested"]["access_token"] == "<redacted>"
    assert result["safe"] == "visible"


def test_jsonable_survives_objects_that_refuse_to_serialize():
    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("no repr for you")

    assert "unserializable" in jsonable(Hostile())


def test_jsonable_bounds_depth_and_length():
    deep: dict[str, Any] = {}
    node = deep
    for _ in range(40):
        node["next"] = {}
        node = node["next"]

    assert "maximum-depth-reached" in repr(jsonable(deep))
    assert jsonable("x" * 50_000).endswith("<truncated>")
    assert len(jsonable(list(range(2000)))) == 501  # 500 items + marker


def test_tracer_never_raises_into_the_agent(tracer, exporter):
    """A callback that cannot serialise its input still must not raise."""

    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("nope")

    tracer.on_chain_start({"name": "c"}, {"bad": Hostile()}, run_id=uid())
    assert len(exporter.events) == 1


def test_normalize_usage_accepts_both_naming_conventions():
    openai_style = normalize_usage({"prompt_tokens": 10, "completion_tokens": 5})
    langchain_style = normalize_usage({"input_tokens": 10, "output_tokens": 5})

    assert openai_style == langchain_style
    # total is derived when the provider omits it
    assert openai_style["total_tokens"] == 15


def test_exporter_drops_rather_than_blocks_when_the_queue_is_full():
    exporter = BatchExporter(
        endpoint="http://127.0.0.1:1", api_key="k", queue_size=1, flush_interval_seconds=0.01
    )
    try:
        for _ in range(200):
            exporter.emit({"event_type": "span.start", "trace_id": "t", "span_id": "s"})
        # Dropping is the correct behaviour: losing telemetry beats stalling
        # the agent being observed.
        assert exporter.dropped_events > 0
    finally:
        exporter.close()


def test_unreachable_endpoint_does_not_raise():
    """The endpoint being down must cost a log line, not an exception."""
    exporter = BatchExporter(
        endpoint="http://127.0.0.1:1", api_key="k", flush_interval_seconds=0.01, timeout_seconds=0.1
    )
    try:
        exporter.emit({"event_type": "span.start", "trace_id": "t", "span_id": "s"})
        exporter.flush(timeout=3)
    finally:
        exporter.close()


def test_metadata_namespace_is_prefixed(telemetry):
    """Reserved keys must never collide with the customer's own metadata."""
    config = telemetry.context(agent_name="a", metadata={"agent_name": "theirs"}).config

    assert config["metadata"]["agent_name"] == "theirs"
    assert config["metadata"][f"{NS}.agent_name"] == "a"


# -- automatic naming ------------------------------------------------------


def test_names_are_detected_automatically_from_langchain(tracer, exporter):
    """No annotation required: LangChain hands us the name it knows.

    This is the behaviour that makes tracing feel like it works by magic - the
    function name, tool name or graph-node name appears in the tree without the
    customer labelling anything.
    """
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

    names = {(e["span_kind"], e["name"]) for e in starts(exporter)}
    assert ("chain", "summarize_document") in names
    assert ("tool", "search_the_web") in names


def test_explicit_run_name_wins_over_the_detected_one(tracer, exporter):
    tracer.on_chain_start({"name": "detected"}, {}, run_id=uid(), name="explicit")
    assert starts(exporter)[0]["name"] == "explicit"


def test_class_name_is_the_last_resort(tracer, exporter):
    """serialized.id is a dotted path; its tail is the class name."""
    tracer.on_chain_start({"id": ["langchain", "schema", "RunnableSequence"]}, {}, run_id=uid())
    assert starts(exporter)[0]["name"] == "RunnableSequence"


def test_agent_span_borrows_the_detected_name_when_none_was_given(tracer, exporter):
    """An agent boundary with no declared name still gets a readable label."""
    tracer.on_chain_start(
        {"name": "research_workflow"},
        {},
        run_id=uid(),
        metadata={f"{NS}.agent_instance_id": "agent-1"},
    )

    event = starts(exporter)[0]
    assert event["span_kind"] == "agent"
    assert event["agent_name"] == "research_workflow"
