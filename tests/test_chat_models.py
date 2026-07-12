from langchain_core.messages import AIMessage, HumanMessage

from langchain_mljunction.chat_models import ChatMLJunction, _ai_message, _message


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


def test_payload_uses_native_routing_controls() -> None:
    model = ChatMLJunction(model="test-model", api_key="test-key", routing={"strategy": "latency"})
    payload = model._payload([HumanMessage(content="hello")], stream=False, stop=None)
    assert payload["routing"] == {"strategy": "latency"}
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
