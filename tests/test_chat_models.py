import asyncio
import base64
import struct

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

from langchain_mljunction._client import (
    MLJunctionAPIError,
    MLJunctionStreamError,
    _araise_for_status,
    _raise_for_status,
)
from langchain_mljunction.chat_models import ChatMLJunction, _ai_message, _content, _message
from langchain_mljunction.embeddings import MLJunctionEmbeddings


def test_message_conversion_preserves_tool_calls() -> None:
    message = AIMessage(
        content="",
        tool_calls=[{"name": "weather", "args": {"city": "Paris"}, "id": "call_1"}],
    )
    converted = _message(message)
    assert converted["tool_calls"][0]["function"]["name"] == "weather"


def test_message_conversion_deduplicates_langchain_v1_tool_blocks() -> None:
    message = AIMessage(
        content=[
            {"type": "text", "text": "checking"},
            {
                "type": "tool_call",
                "name": "weather",
                "args": {"city": "Paris"},
                "id": "call_1",
            },
        ],
        tool_calls=[{"name": "weather", "args": {"city": "Paris"}, "id": "call_1"}],
    )

    converted = _message(message)

    assert converted["content"] == [{"type": "text", "text": "checking"}]
    assert len(converted["tool_calls"]) == 1


def test_content_normalizes_standard_multimodal_blocks() -> None:
    converted = _content(
        [
            {"type": "image", "base64": "abc", "mime_type": "image/png"},
            {"type": "audio", "base64": "def", "mime_type": "audio/wav"},
            {"type": "file", "base64": "ghi", "mime_type": "application/pdf"},
        ]
    )
    assert converted[0] == {
        "type": "image_url",
        "image_url": "data:image/png;base64,abc",
    }
    assert converted[1] == {
        "type": "input_audio",
        "data": "def",
        "media_type": "audio/wav",
    }
    assert converted[2]["data"] == "data:application/pdf;base64,ghi"


def test_native_response_conversion_preserves_usage_and_receipt() -> None:
    converted = _ai_message(
        {
            "id": "resp_1",
            "model": "test-model",
            "session_id": "sess_1",
            "output": [{"type": "json", "object": {"ok": True}}],
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            "routing": {"provider": "openai"},
            "receipt": {"actual_charge_usd": 0.1},
        }
    )
    assert converted.content == '{"ok":true}'
    assert converted.usage_metadata and converted.usage_metadata["total_tokens"] == 5
    assert converted.response_metadata["receipt"]["actual_charge_usd"] == 0.1


def test_reasoning_details_and_token_round_trip_without_becoming_text() -> None:
    block = {"type": "reasoning", "encrypted_content": "opaque"}
    converted = _ai_message(
        {
            "id": "resp_1",
            "model": "claude-test",
            "output": [{"type": "text", "text": "answer"}],
            "reasoning": {
                "details": [block],
                "continuation_token": "gwrt_v3.a-valid-long-token",
                "phase": "completed",
                "reusable": True,
            },
        }
    )
    assert converted.content == "answer"
    assert _message(converted) == {
        "role": "assistant",
        "content": "answer",
        "reasoning_details": [block],
    }

    model = ChatMLJunction(model="test-model", api_key="test-key")
    payload = model._payload(
        [HumanMessage(content="first"), converted, HumanMessage(content="continue")],
        stream=False,
        stop=None,
    )
    assert payload["reasoning"]["continuation_token"] == "gwrt_v3.a-valid-long-token"


def test_explicit_null_continuation_disables_automatic_replay() -> None:
    previous = AIMessage(
        content="answer",
        response_metadata={"reasoning": {"continuation_token": "gwrt_v3.a-valid-long-token"}},
    )
    model = ChatMLJunction(model="test-model", api_key="test-key")
    payload = model._payload(
        [HumanMessage(content="first"), previous, HumanMessage(content="fork")],
        stream=False,
        stop=None,
        reasoning={"enabled": True, "continuation_token": None},
    )
    assert payload["reasoning"]["continuation_token"] is None


def test_tool_result_continuation_sets_input_type() -> None:
    previous = AIMessage(
        content="",
        tool_calls=[{"name": "weather", "args": {}, "id": "call_1"}],
        response_metadata={"reasoning": {"continuation_token": "gwrt_v3.a-valid-long-token"}},
    )
    model = ChatMLJunction(model="test-model", api_key="test-key")
    payload = model._payload(
        [
            HumanMessage(content="weather?"),
            previous,
            ToolMessage(content="sunny", tool_call_id="call_1"),
        ],
        stream=False,
        stop=None,
    )
    assert payload["reasoning"]["input_type"] == "tool_results"


def test_tool_result_continuation_survives_trailing_workflow_context() -> None:
    previous = AIMessage(
        content="",
        tool_calls=[{"name": "weather", "args": {}, "id": "call_1"}],
        response_metadata={"reasoning": {"continuation_token": "gwrt_v3.a-valid-long-token"}},
    )
    model = ChatMLJunction(model="test-model", api_key="test-key")
    payload = model._payload(
        [
            HumanMessage(content="weather?"),
            previous,
            ToolMessage(content="sunny", tool_call_id="call_1"),
            HumanMessage(content="Current workflow state: enquiry complete"),
        ],
        stream=False,
        stop=None,
    )
    assert payload["reasoning"]["input_type"] == "tool_results"


def test_payload_uses_native_routing_controls() -> None:
    model = ChatMLJunction(
        model="test-model",
        api_key="test-key",
        routing={"strategy": "latency", "require_zdr": True},
        reasoning={"enabled": True, "effort": "high"},
        requirements={"tools": "preferred"},
        context={"mode": "strict"},
        metadata={"trace": "abc"},
        idempotency_key="idem-1",
        top_p=0.8,
        seed=7,
    )
    payload = model._payload([HumanMessage(content="hello")], stream=False, stop=None)
    assert payload["routing"] == {"strategy": "latency", "require_zdr": True}
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["reasoning"]["effort"] == "high"
    assert payload["requirements"]["tools"] == "preferred"
    assert payload["context"]["mode"] == "strict"
    assert payload["metadata"] == {"trace": "abc"}
    assert payload["idempotency_key"] == "idem-1"
    assert payload["sampling"]["top_p"] == 0.8
    assert payload["sampling"]["seed"] == 7


def test_constructor_stop_sequences_are_sent_to_gateway() -> None:
    model = ChatMLJunction(
        model="test-model",
        api_key="test-key",
        stop=["DONE"],
    )
    payload = model._payload([HumanMessage(content="hello")], stream=False, stop=None)
    assert payload["sampling"]["stop"] == ["DONE"]

    overridden = model._payload([HumanMessage(content="hello")], stream=False, stop=["STOP"])
    assert overridden["sampling"]["stop"] == ["STOP"]


def test_bind_tools_forwards_parallel_control_as_native_field() -> None:
    model = ChatMLJunction(model="test-model", api_key="test-key")
    bound = model.bind_tools(
        [
            {
                "name": "add",
                "description": "Add values",
                "parameters": {"type": "object"},
            }
        ],
        parallel_tool_calls=True,
    )
    assert bound.kwargs["parallel_tool_calls"] is True
    assert "compatibility" not in bound.kwargs


def test_structured_json_stream_produces_a_langchain_generation() -> None:
    model = ChatMLJunction(model="test-model", api_key="test-key")
    # As the gateway sends it: the object, then the terminal frame.
    model._client.stream = lambda *_args, **_kwargs: iter(
        [
            ("response.output_json.done", {"object": {"ok": True}}),
            ("response.completed", {"id": "resp_1", "model": "test-model", "output": []}),
        ]
    )

    chunks = list(model._stream([HumanMessage(content="Return JSON")]))

    assert len(chunks) == 2
    assert chunks[0].message.content == '{"ok":true}'


def test_bound_structured_stream_uses_one_complete_non_streaming_response() -> None:
    model = ChatMLJunction(model="test-model", api_key="test-key", buffered_stream=False)
    model._client.stream = lambda *_args, **_kwargs: pytest.fail(
        "structured output must not use the streaming transport"
    )
    model._client.post = lambda *_args, **_kwargs: {
        "id": "resp_1",
        "model": "test-model",
        "output": [{"type": "json", "object": {"ok": True}}],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }

    chunks = list(
        model._stream(
            [HumanMessage(content="Return JSON")],
            output={
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object"},
                }
            },
        )
    )

    assert len(chunks) == 1
    assert chunks[0].message.content == '{"ok":true}'
    assert chunks[0].message.usage_metadata
    assert chunks[0].message.usage_metadata["total_tokens"] == 5


def test_stream_terminal_chunk_contains_usage_and_model_metadata() -> None:
    model = ChatMLJunction(model="test-model", api_key="test-key")
    model._client.stream = lambda *_args, **_kwargs: iter(
        [
            ("response.output_text.delta", {"delta": "hello"}),
            (
                "response.completed",
                {
                    "id": "resp_1",
                    "model": "test-model",
                    "output": [{"type": "text", "text": "hello"}],
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                    },
                },
            ),
        ]
    )

    chunks = list(model._stream([HumanMessage(content="Say hello")]))

    assert len(chunks) == 2
    assert chunks[-1].message.content == ""
    assert chunks[-1].message.usage_metadata
    assert chunks[-1].message.usage_metadata["total_tokens"] == 3
    assert chunks[-1].message.response_metadata["model_name"] == "test-model"


def test_structured_function_calling_stream_is_atomic() -> None:
    class Answer(BaseModel):
        answer: str

    model = ChatMLJunction(model="test-model", api_key="test-key")
    structured = model.with_structured_output(Answer, method="function_calling")
    bound = structured.first
    assert bound.kwargs["_mljunction_structured_output"] is True
    assert bound.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "Answer"},
    }


def test_bind_tools_normalizes_langchain_any_choice() -> None:
    model = ChatMLJunction(model="test-model", api_key="test-key")
    bound = model.bind_tools(
        [{"name": "answer", "description": "Answer", "parameters": {}}],
        tool_choice="any",
    )
    assert bound.kwargs["tool_choice"] == "required"


def test_with_structured_output_rejects_unknown_method() -> None:
    model = ChatMLJunction(model="test-model", api_key="test-key")
    with pytest.raises(ValueError, match="method must be one of"):
        model.with_structured_output({"type": "object"}, method="made_up")


def test_with_structured_output_accepts_pydantic_v1_schema() -> None:
    class LegacyAnswer(BaseModelV1):
        answer: str

    model = ChatMLJunction(model="test-model", api_key="test-key")
    runnable = model.with_structured_output(LegacyAnswer, method="json_schema")
    assert runnable is not None


def test_with_structured_output_include_raw_uses_langchain_result_contract() -> None:
    class Answer(BaseModel):
        answer: str

    model = ChatMLJunction(model="test-model", api_key="test-key", buffered_stream=False)
    model._client.post = lambda *_args, **_kwargs: {
        "id": "resp_1",
        "model": "test-model",
        "output": [{"type": "json", "object": {"answer": "yes"}}],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    result = model.with_structured_output(Answer, include_raw=True).invoke("Answer")

    assert result["raw"].content == '{"answer":"yes"}'
    assert result["parsed"] == Answer(answer="yes")
    assert result["parsing_error"] is None


@pytest.mark.asyncio
async def test_async_structured_json_stream_produces_a_langchain_generation() -> None:
    model = ChatMLJunction(model="test-model", api_key="test-key")

    async def events():
        yield "response.output_json.done", {"object": {"ok": True}}
        await asyncio.sleep(0)
        yield "response.completed", {"id": "resp_1", "model": "test-model", "output": []}

    model._client.astream = lambda *_args, **_kwargs: events()

    chunks = [chunk async for chunk in model._astream([HumanMessage(content="Return JSON")])]

    assert len(chunks) == 2
    assert chunks[0].message.content == '{"ok":true}'


@pytest.mark.asyncio
async def test_async_bound_structured_stream_uses_complete_response() -> None:
    model = ChatMLJunction(model="test-model", api_key="test-key", buffered_stream=False)

    async def fail_stream():
        pytest.fail("structured output must not use the streaming transport")
        yield

    async def post(*_args, **_kwargs):
        return {
            "id": "resp_1",
            "model": "test-model",
            "output": [{"type": "json", "object": {"ok": True}}],
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        }

    model._client.astream = lambda *_args, **_kwargs: fail_stream()
    model._client.apost = post

    chunks = [
        chunk
        async for chunk in model._astream(
            [HumanMessage(content="Return JSON")],
            output={
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object"},
                }
            },
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].message.content == '{"ok":true}'
    assert chunks[0].message.usage_metadata
    assert chunks[0].message.usage_metadata["total_tokens"] == 5


def test_structured_transport_error_preserves_gateway_details() -> None:
    response = httpx.Response(
        400,
        headers={"x-request-id": "resp_1"},
        json={
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_request",
                "message": "bad route",
                "details": {"reason": "opaque_reasoning_provider_mismatch"},
            }
        },
    )
    with pytest.raises(MLJunctionAPIError) as exc_info:
        _raise_for_status(response)
    assert exc_info.value.request_id == "resp_1"
    assert exc_info.value.details["reason"] == "opaque_reasoning_provider_mismatch"


@pytest.mark.asyncio
async def test_async_streaming_transport_error_is_read_before_parsing() -> None:
    response = httpx.Response(
        400,
        headers={"x-request-id": "resp_stream_1"},
        stream=httpx.ByteStream(
            b'{"error":{"code":"invalid_request","message":"bad streamed request"}}'
        ),
    )

    with pytest.raises(MLJunctionAPIError) as exc_info:
        await _araise_for_status(response)

    assert exc_info.value.request_id == "resp_stream_1"
    assert exc_info.value.code == "invalid_request"
    assert "bad streamed request" in str(exc_info.value)


def test_base64_embedding_is_decoded_to_langchain_float_vector() -> None:
    encoded = base64.b64encode(struct.pack("<2f", 1.25, -2.5)).decode()
    assert MLJunctionEmbeddings._vector(encoded) == [1.25, -2.5]


def _buffered_model(events: list) -> ChatMLJunction:
    model = ChatMLJunction(model="test-model", api_key="test-key")
    model._client.post = lambda *_args, **_kwargs: pytest.fail(
        "a buffered call must read the answer as a stream"
    )
    model._client.stream = lambda *_args, **_kwargs: iter(events)
    return model


def test_invoke_reads_a_stream_and_returns_one_complete_message() -> None:
    usage = {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4}
    model = _buffered_model(
        [
            ("response.created", {"id": "resp_1"}),
            ("response.output_text.delta", {"delta": "hel"}),
            ("response.output_text.delta", {"delta": "lo"}),
            (
                "response.completed",
                {
                    "id": "resp_1",
                    "model": "test-model",
                    "finish_reason": "stop",
                    "routing": {"route": "r1"},
                    "output": [{"type": "text", "text": "hello"}],
                    "usage": usage,
                },
            ),
        ]
    )

    message = model.invoke("Say hello")

    assert type(message) is AIMessage
    assert message.content == "hello"
    assert message.usage_metadata["total_tokens"] == 4
    assert message.response_metadata["id"] == "resp_1"
    assert message.response_metadata["finish_reason"] == "stop"
    result = model._generate([HumanMessage(content="Say hello")])
    assert result.llm_output == {"route": "r1"}


def test_buffered_invoke_merges_streamed_tool_call_arguments() -> None:
    model = _buffered_model(
        [
            (
                "response.tool_call.delta",
                {"index": 0, "id": "call_1", "name": "weather", "arguments": '{"city":'},
            ),
            ("response.tool_call.delta", {"index": 0, "arguments": '"Paris"}'}),
            ("response.completed", {"id": "resp_1", "model": "test-model", "output": []}),
        ]
    )

    message = model.invoke("Weather?")

    assert message.tool_calls == [
        {"name": "weather", "args": {"city": "Paris"}, "id": "call_1", "type": "tool_call"}
    ]


def test_buffered_structured_output_parses_the_streamed_object() -> None:
    class Answer(BaseModel):
        answer: str

    model = _buffered_model(
        [
            ("response.output_json.done", {"object": {"answer": "yes"}}),
            ("response.completed", {"id": "resp_1", "model": "test-model", "output": []}),
        ]
    )

    assert model.with_structured_output(Answer).invoke("Answer") == Answer(answer="yes")


def test_buffered_invoke_raises_on_a_failed_stream() -> None:
    model = _buffered_model(
        [
            ("response.output_text.delta", {"delta": "partial"}),
            ("response.failed", {"id": "resp_1", "error": {"code": "upstream_error"}}),
        ]
    )

    with pytest.raises(MLJunctionStreamError, match="upstream_error"):
        model.invoke("Hi")


def test_buffered_invoke_never_returns_a_truncated_answer() -> None:
    model = _buffered_model([("response.output_text.delta", {"delta": "partial"})])

    with pytest.raises(MLJunctionStreamError) as error:
        model.invoke("Hi")
    assert error.value.code == "stream_incomplete"
    assert error.value.partial is True


@pytest.mark.asyncio
async def test_async_invoke_reads_a_stream_and_returns_one_complete_message() -> None:
    model = ChatMLJunction(model="test-model", api_key="test-key")

    async def post(*_args, **_kwargs):
        pytest.fail("a buffered call must read the answer as a stream")

    async def events():
        yield "response.output_text.delta", {"delta": "o"}
        await asyncio.sleep(0)
        yield "response.output_text.delta", {"delta": "k"}
        yield "response.completed", {"id": "resp_1", "model": "test-model", "output": []}

    model._client.apost = post
    model._client.astream = lambda *_args, **_kwargs: events()

    message = await model.ainvoke("Reply OK")

    assert type(message) is AIMessage
    assert message.content == "ok"


def test_provider_prefix_is_kept_by_default() -> None:
    model = ChatMLJunction(model="openai/gpt-4.1-mini", api_key="test-key")
    payload = model._payload([HumanMessage(content="hi")], stream=False, stop=None)
    assert payload["model"] == "openai/gpt-4.1-mini"


def test_provider_prefix_is_stripped_when_enabled() -> None:
    model = ChatMLJunction(
        model="openai/gpt-4.1-mini", api_key="test-key", strip_provider_prefix=True
    )
    messages = [HumanMessage(content="hi")]

    assert model._payload(messages, stream=False, stop=None)["model"] == "gpt-4.1-mini"
    # A per-call override is stripped too.
    override = model._payload(messages, stream=False, stop=None, model="google/gemini-3.5-flash")
    assert override["model"] == "gemini-3.5-flash"
    # A bare name is left alone.
    bare = ChatMLJunction(model="gpt-5", api_key="test-key", strip_provider_prefix=True)
    assert bare._payload(messages, stream=False, stop=None)["model"] == "gpt-5"


def test_embeddings_strip_provider_prefix_only_when_enabled() -> None:
    kept = MLJunctionEmbeddings(model="openai/text-embedding-3-small", api_key="test-key")
    stripped = MLJunctionEmbeddings(
        model="openai/text-embedding-3-small", api_key="test-key", strip_provider_prefix=True
    )
    assert kept._payload(["x"])["model"] == "openai/text-embedding-3-small"
    assert stripped._payload(["x"])["model"] == "text-embedding-3-small"


@pytest.mark.parametrize("where", ["constructor", "per_call"])
def test_idempotency_key_keeps_a_plain_post_so_replay_protection_holds(where: str) -> None:
    kwargs = {"idempotency_key": "ticket-1"} if where == "constructor" else {}
    model = ChatMLJunction(model="test-model", api_key="test-key", **kwargs)
    model._client.stream = lambda *_args, **_kwargs: pytest.fail(
        "an idempotent request must not be streamed"
    )
    sent = {}

    def post(_path, payload):
        sent.update(payload)
        return {"id": "resp_1", "model": "test-model", "output": [{"type": "text", "text": "ok"}]}

    model._client.post = post
    call_kwargs = {"idempotency_key": "ticket-1"} if where == "per_call" else {}

    assert model.invoke("Hi", **call_kwargs).content == "ok"
    assert sent["stream"] is False
    assert sent["idempotency_key"] == "ticket-1"
