import base64
import struct

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_mljunction._client import MLJunctionAPIError, _raise_for_status
from langchain_mljunction.chat_models import ChatMLJunction, _ai_message, _message
from langchain_mljunction.embeddings import MLJunctionEmbeddings


def test_message_conversion_preserves_tool_calls() -> None:
    message = AIMessage(
        content="",
        tool_calls=[{"name": "weather", "args": {"city": "Paris"}, "id": "call_1"}],
    )
    converted = _message(message)
    assert converted["tool_calls"][0]["function"]["name"] == "weather"


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
        response_metadata={
            "reasoning": {"continuation_token": "gwrt_v3.a-valid-long-token"}
        },
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
        response_metadata={
            "reasoning": {"continuation_token": "gwrt_v3.a-valid-long-token"}
        },
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


def test_base64_embedding_is_decoded_to_langchain_float_vector() -> None:
    encoded = base64.b64encode(struct.pack("<2f", 1.25, -2.5)).decode()
    assert MLJunctionEmbeddings._vector(encoded) == [1.25, -2.5]
