"""A tool held back from the prompt must stay held back through bind_tools."""

from langchain_mljunction.chat_models import _format_tool


def test_defer_loading_survives_formatting() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object", "properties": {}},
        },
        "defer_loading": True,
    }
    assert _format_tool(tool, strict=None)["defer_loading"] is True


def test_an_ordinary_tool_does_not_gain_the_flag() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "f",
            "description": "F",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    assert "defer_loading" not in _format_tool(tool, strict=None)


def test_a_server_tool_passes_through_with_the_flag() -> None:
    assert _format_tool({"type": "calculator", "defer_loading": True}, strict=None) == {
        "type": "calculator",
        "defer_loading": True,
    }
