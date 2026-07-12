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
        max_tokens=64,
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

    place = model.with_structured_output(Place).invoke("Return Paris, France")
    assert place.city and place.country


async def test_native_langchain_async_and_embeddings() -> None:
    model = ChatMLJunction(
        model=os.getenv("LIVE_OPENAI_MODEL", "gpt-4o-mini"),
        api_key=KEY,
        base_url=os.getenv("LIVE_API_BASE", "http://localhost:8001"),
        max_tokens=16,
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
