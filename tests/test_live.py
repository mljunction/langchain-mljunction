import json
import os

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from langchain_mljunction import ChatMLJunction, MLJunctionEmbeddings

RUN = os.getenv("RUN_LIVE_PROVIDER_TESTS") == "1"
KEY = os.getenv("LIVE_API_KEY", "")

pytestmark = pytest.mark.skipif(not RUN or not KEY, reason="live test disabled")


class Place(BaseModel):
    city: str
    country: str


def test_native_langchain_invoke_stream_tools_and_structured_output() -> None:
    model = ChatMLJunction(
        model=os.getenv("LIVE_OPENAI_MODEL", "gpt-4o-mini"),
        api_key=KEY,
        base_url=os.getenv("LIVE_API_BASE", "http://localhost:8001"),
        max_tokens=512,
        routing={"strategy": "latency"},
        session_id="langchain-live",
        session_name="LangChain session",
        task_id="langchain-task",
        task_name="LangChain live test",
        app_name="langchain-live-app",
    )
    response = model.invoke([HumanMessage(content="Reply only PONG")])
    assert response.content
    assert response.response_metadata["routing"]
    assert response.response_metadata["app_name"] == "langchain-live-app"
    assert "".join(chunk.content for chunk in model.stream("Count 1, 2, 3"))

    tool_model = model.bind_tools(
        [
            {
                "name": "multiply",
                "description": "Multiply numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            }
        ],
        tool_choice="required",
    )
    assert tool_model.invoke("Use multiply for 7 times 8").tool_calls
    streamed_tool_chunks = list(tool_model.stream("Use multiply for 9 times 6"))
    assert any(chunk.tool_call_chunks for chunk in streamed_tool_chunks)

    parallel_model = model.bind_tools(
        [
            {
                "name": "add",
                "description": "Add numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            },
            {
                "name": "multiply",
                "description": "Multiply numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            },
        ],
        tool_choice="required",
        parallel_tool_calls=True,
    )
    parallel = parallel_model.invoke("Call add for 2+3 and multiply for 4*5 in the same turn.")
    assert {call["name"] for call in parallel.tool_calls} == {"add", "multiply"}

    place = model.with_structured_output(Place).invoke("Return Paris, France")
    assert place.city and place.country
    function_place = model.with_structured_output(Place, method="function_calling").invoke(
        "Use the required response function for Paris, France"
    )
    assert function_place.city and function_place.country
    json_mode_place = model.with_structured_output(Place, method="json_mode").invoke(
        "Return a JSON object with city and country for Paris, France"
    )
    assert json_mode_place.city and json_mode_place.country
    raw_place = model.with_structured_output(Place, include_raw=True).invoke("Return Paris, France")
    assert raw_place["raw"].content
    assert raw_place["parsed"].city and raw_place["parsed"].country
    assert raw_place["parsing_error"] is None

    structured_stream = model.bind(
        output={
            "streaming_mode": "raw_compat",
            "format": {
                "type": "json_schema",
                "name": "checks",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "checks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 8,
                            "maxItems": 8,
                        }
                    },
                    "required": ["checks"],
                    "additionalProperties": False,
                },
            },
        }
    )
    structured_chunks = list(
        structured_stream.stream("Return exactly eight detailed API reliability checks.")
    )
    structured = json.loads("".join(chunk.content for chunk in structured_chunks))
    assert len(structured["checks"]) == 8
    assert len([chunk for chunk in structured_chunks if chunk.content]) == 1


async def test_native_langchain_async_and_embeddings() -> None:
    model = ChatMLJunction(
        model=os.getenv("LIVE_OPENAI_MODEL", "gpt-4o-mini"),
        api_key=KEY,
        base_url=os.getenv("LIVE_API_BASE", "http://localhost:8001"),
        max_tokens=64,
    )
    response = await model.ainvoke("Reply only OK")
    assert response.content
    chunks = [chunk async for chunk in model.astream("Count 1, 2, 3")]
    assert "".join(chunk.content for chunk in chunks)

    embeddings = MLJunctionEmbeddings(
        model=os.getenv("LIVE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=KEY,
        base_url=os.getenv("LIVE_API_BASE", "http://localhost:8001"),
    )
    vector = await embeddings.aembed_query("ML Junction")
    assert len(vector) > 100
