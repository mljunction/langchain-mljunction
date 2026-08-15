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
    model._client.stream = lambda *_args, **_kwargs: iter(
        [("response.output_json.done", {"object": {"ok": True}})]
    )

    chunks = list(model._stream([HumanMessage(content="Return JSON")]))

    assert len(chunks) == 1
    assert chunks[0].message.content == '{"ok":true}'


def test_bound_structured_stream_uses_one_complete_non_streaming_response() -> None:
    model = ChatMLJunction(model="test-model", api_key="test-key")
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

    model = ChatMLJunction(model="test-model", api_key="test-key")
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

    model._client.astream = lambda *_args, **_kwargs: events()

    chunks = [chunk async for chunk in model._astream([HumanMessage(content="Return JSON")])]

    assert len(chunks) == 1
    assert chunks[0].message.content == '{"ok":true}'


@pytest.mark.asyncio
async def test_async_bound_structured_stream_uses_complete_response() -> None:
    model = ChatMLJunction(model="test-model", api_key="test-key")

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
