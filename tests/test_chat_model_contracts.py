"""Contract checks for the chat model: stream endings, request shape, cache
identity, response conversion and message roles. No network."""

import asyncio

import pytest
from langchain_core.messages import ChatMessage, HumanMessage

from langchain_mljunction._client import MLJunctionStreamError
from langchain_mljunction.chat_models import ChatMLJunction, _ai_message, _content, _message

COMPLETED = ("response.completed", {"id": "resp_1", "model": "test-model", "output": []})


def _model(**kwargs) -> ChatMLJunction:
    return ChatMLJunction(model="test-model", api_key="test-key", **kwargs)


# ---- stream endings (SDK-02) ------------------------------------------------


def _stream(events):
    model = _model()
    model._client.stream = lambda *_a, **_k: iter(events)
    return list(model._stream([HumanMessage(content="hi")]))


def test_a_failure_after_partial_output_raises_instead_of_returning_it() -> None:
    with pytest.raises(MLJunctionStreamError) as raised:
        _stream(
            [
                ("response.created", {"id": "resp_9"}),
                ("response.output_text.delta", {"delta": "half an ans"}),
                (
                    "response.failed",
                    {
                        "id": "resp_9",
                        "partial": True,
                        "error": {"code": "provider_timeout", "message": "slow", "status": 504},
                    },
                ),
            ]
        )
    assert raised.value.partial is True
    assert raised.value.request_id == "resp_9"
    assert raised.value.code == "provider_timeout"
    assert raised.value.status_code == 504


def test_a_stream_that_just_stops_is_not_a_complete_answer() -> None:
    with pytest.raises(MLJunctionStreamError) as raised:
        _stream([("response.output_text.delta", {"delta": "cut"})])
    assert raised.value.code == "stream_incomplete"
    assert raised.value.partial is True


def test_frames_after_the_terminal_one_are_ignored() -> None:
    usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    completed = ("response.completed", {"id": "r", "model": "m", "output": [], "usage": usage})
    chunks = _stream([("response.output_text.delta", {"delta": "ok"}), completed, completed])
    assert len(chunks) == 2  # the duplicate terminal would double the usage


async def test_the_async_stream_raises_on_failure_too() -> None:
    model = _model()

    async def events():
        yield "response.output_text.delta", {"delta": "x"}
        await asyncio.sleep(0)
        yield "response.failed", {"id": "r", "partial": True, "error": {"code": "boom"}}

    model._client.astream = lambda *_a, **_k: events()
    with pytest.raises(MLJunctionStreamError):
        [chunk async for chunk in model._astream([HumanMessage(content="hi")])]


# ---- request shape (SDK-06, SDK-07) -----------------------------------------


def test_per_call_sampling_and_limits_go_where_the_gateway_expects_them() -> None:
    payload = _model(temperature=0.2)._payload(
        [HumanMessage(content="hi")], stream=False, stop=None, temperature=0.9, max_tokens=30
    )
    assert payload["sampling"]["temperature"] == 0.9
    assert payload["output"]["max_tokens"] == 30
    assert "temperature" not in payload and "max_tokens" not in payload


def test_structured_output_keeps_the_constructor_output_limit() -> None:
    payload = _model(max_tokens=50)._payload(
        [HumanMessage(content="hi")],
        stream=False,
        stop=None,
        output={"format": {"type": "json_object"}},
    )
    assert payload["output"] == {"max_tokens": 50, "format": {"type": "json_object"}}


def test_an_unset_max_tokens_does_not_erase_a_native_output_limit() -> None:
    payload = _model(output={"max_tokens": 7})._payload(
        [HumanMessage(content="hi")], stream=False, stop=None
    )
    assert payload["output"]["max_tokens"] == 7


def test_a_per_call_none_keeps_the_configured_value() -> None:
    payload = _model(temperature=0.3)._payload(
        [HumanMessage(content="hi")], stream=False, stop=None, temperature=None
    )
    assert payload["sampling"]["temperature"] == 0.3


# ---- cache identity (SDK-01) ------------------------------------------------


def test_cache_identity_separates_keys_sessions_and_settings() -> None:
    base = _model()._identifying_params
    assert ChatMLJunction(model="test-model", api_key="other-key")._identifying_params != base
    assert _model(session_id="s2")._identifying_params != base
    assert _model(temperature=0.9)._identifying_params != base
    assert _model(max_tokens=10)._identifying_params != base
    assert _model()._identifying_params == base


def test_cache_identity_never_contains_the_key() -> None:
    params = ChatMLJunction(model="m", api_key="sk-very-secret")._identifying_params
    assert "sk-very-secret" not in repr(params)


# ---- response conversion (SDK-08) ------------------------------------------


def test_malformed_tool_arguments_become_an_invalid_call_not_a_crash() -> None:
    message = _ai_message(
        {
            "id": "resp_1",
            "finish_reason": "length",
            "output": [
                {"type": "text", "text": "partial"},
                {
                    "type": "tool_call",
                    "tool_call": {"id": "c1", "function": {"name": "f", "arguments": '{"a": '}},
                },
                {
                    "type": "tool_call",
                    "tool_call": {"id": "c2", "function": {"name": "g", "arguments": '{"b": 1}'}},
                },
            ],
            "usage": {"input_tokens": 3, "cached_input_tokens": 2, "reasoning_tokens": 5},
        }
    )
    assert message.content == "partial"
    assert [call["name"] for call in message.tool_calls] == ["g"]
    assert message.invalid_tool_calls[0]["name"] == "f"
    assert message.invalid_tool_calls[0]["args"] == '{"a": '
    assert message.response_metadata["id"] == "resp_1"
    assert message.response_metadata["finish_reason"] == "length"
    assert message.usage_metadata["input_token_details"] == {"cache_read": 2}
    assert message.usage_metadata["output_token_details"] == {"reasoning": 5}


# ---- message roles and documents (SDK-12) -----------------------------------


def test_a_generic_chat_message_uses_its_role() -> None:
    assert _message(ChatMessage(role="user", content="hi"))["role"] == "user"
    with pytest.raises(ValueError, match="not supported"):
        _message(ChatMessage(role="narrator", content="hi"))


def test_a_document_block_is_sent_as_a_file() -> None:
    [block] = _content(
        [
            {
                "type": "document",
                "title": "report.pdf",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "QUJD"},
            }
        ]
    )
    assert block == {
        "type": "file",
        "data": "data:application/pdf;base64,QUJD",
        "filename": "report.pdf",
        "media_type": "application/pdf",
    }
