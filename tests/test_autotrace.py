"""Automatic, LangSmith-style tracing: no callbacks passed anywhere."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableLambda
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from langchain_mljunction import ChatMLJunction, autotrace, traceable, tracing_disabled
from langchain_mljunction.tracer import MLJunctionTracer


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
        capture_content=True,
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    autotrace.use_tracer(instance)
    yield instance
    instance.close()


def spans(tracer: MLJunctionTracer, exporter: RecordingExporter) -> dict[str, Any]:
    assert tracer.flush()
    return {span.name: span for span in exporter.spans}


def test_runs_are_traced_without_passing_callbacks(tracer, exporter):
    model = FakeListChatModel(responses=["hi"])
    chain = RunnableLambda(lambda text: model.invoke(text), name="answer")

    chain.invoke("hello")

    named = spans(tracer, exporter)
    assert named["FakeListChatModel"].parent.span_id == named["answer"].context.span_id


def test_tracing_disabled_block_records_nothing(tracer, exporter):
    with tracing_disabled():
        RunnableLambda(lambda x: x, name="quiet").invoke(1)
    assert "quiet" not in spans(tracer, exporter)


def test_hidden_plumbing_runs_are_skipped_and_children_reattach(tracer, exporter):
    root, hidden, child = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    tracer.on_chain_start({"name": "graph"}, {}, run_id=root)
    tracer.on_chain_start(
        {"name": "ChannelWrite"}, {}, run_id=hidden, parent_run_id=root, tags=["langsmith:hidden"]
    )
    tracer.on_tool_start({"name": "search"}, "q", run_id=child, parent_run_id=hidden)
    tracer.on_tool_end("r", run_id=child)
    tracer.on_chain_end({}, run_id=hidden)
    tracer.on_chain_end({}, run_id=root)

    named = spans(tracer, exporter)
    assert "ChannelWrite" not in named
    assert named["search"].parent.span_id == named["graph"].context.span_id


def test_model_session_reaches_the_enclosing_graph_spans(tracer, exporter):
    graph, node, llm = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    tracer.on_chain_start({"name": "graph"}, {}, run_id=graph)
    tracer.on_chain_start(
        {"name": "planner"},
        {},
        run_id=node,
        parent_run_id=graph,
        metadata={"langgraph_node": "planner", "langgraph_step": 2},
    )
    tracer.on_chat_model_start(
        {"name": "ChatMLJunction"},
        [[]],
        run_id=llm,
        parent_run_id=node,
        metadata={"langgraph_node": "planner", "langgraph_step": 2},
        invocation_params={"model": "gpt-5.6-luna", "session_id": "sess-1"},
    )
    tracer.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]]), run_id=llm
    )
    tracer.on_chain_end({}, run_id=node)
    tracer.on_chain_end({}, run_id=graph)

    named = spans(tracer, exporter)
    for name in ("graph", "planner", "gpt-5.6-luna"):
        assert named[name].attributes["mlj.session.id"] == "sess-1"
    assert named["gpt-5.6-luna"].attributes["mlj.graph.node"] == "planner"
    assert named["gpt-5.6-luna"].attributes["mlj.graph.step"] == 2


def test_traceable_function_parents_the_calls_inside_it(tracer, exporter):
    @traceable(name="plan_day")
    def plan_day(city: str, config: Any = None) -> str:
        return RunnableLambda(lambda x: x.upper(), name="shout").invoke(city)

    assert plan_day("goa") == "GOA"

    named = spans(tracer, exporter)
    assert named["shout"].parent.span_id == named["plan_day"].context.span_id
    assert '"city":"goa"' in named["plan_day"].attributes["mlj.capture.input"]


def test_traceable_async_function_nests_under_the_active_run(tracer, exporter):
    @traceable
    async def fetch(x: int) -> int:
        return await RunnableLambda(lambda v: v + 1, name="inc").ainvoke(x)

    outer = RunnableLambda(lambda x: x, name="outer")

    async def run() -> int:
        return await outer.ainvoke(1) + await fetch(1)

    assert asyncio.run(run()) == 3
    named = spans(tracer, exporter)
    assert named["inc"].parent.span_id == named["fetch"].context.span_id


def test_traceable_records_errors_and_reraises(tracer, exporter):
    @traceable
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()
    assert spans(tracer, exporter)["boom"].status.status_code.name == "ERROR"


def test_env_switch_off_builds_no_tracer():
    ChatMLJunction(model="m", api_key="mlj_test")
    assert autotrace._tracer is None


def test_model_tracing_flag_overrides_the_env_switch(monkeypatch):
    built: list[dict[str, Any]] = []

    class Stub:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

    monkeypatch.setattr("langchain_mljunction.tracer.MLJunctionTracer", Stub)
    monkeypatch.setattr(autotrace.atexit, "register", lambda *a: None)
    ChatMLJunction(model="m", api_key="mlj_test", base_url="http://gw", tracing=True)
    assert built and built[0]["endpoint"] == "http://gw"
    assert built[0]["capture_content"] is True


def test_env_key_alone_does_not_trace_other_providers(monkeypatch):
    """A key kept for one feature must not trace runs of other providers."""
    monkeypatch.delenv("MLJUNCTION_TRACING", raising=False)
    monkeypatch.setenv("MLJUNCTION_API_KEY", "mlj_test")
    assert autotrace._ProcessSlot().get() is None
    assert autotrace._tracer is None


def test_explicit_env_switch_traces_from_the_env_key(monkeypatch):
    monkeypatch.setenv("MLJUNCTION_TRACING", "true")
    monkeypatch.setenv("MLJUNCTION_API_KEY", "mlj_test")
    monkeypatch.setattr(autotrace.atexit, "register", lambda *a: None)
    built: list[dict[str, Any]] = []

    class Stub:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

    monkeypatch.setattr("langchain_mljunction.tracer.MLJunctionTracer", Stub)
    assert isinstance(autotrace._ProcessSlot().get(), Stub)
    assert built[0]["api_key"] == "mlj_test"
