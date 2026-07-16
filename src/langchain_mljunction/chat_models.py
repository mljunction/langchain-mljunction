from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr

from langchain_mljunction._client import MLJunctionClient


def _message(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    if isinstance(message, AIMessage):
        content: Any = message.content or None
        value: dict[str, Any] = {"role": "assistant", "content": content}
        reasoning_details = message.additional_kwargs.get("reasoning_details") or []
        if reasoning_details:
            value["reasoning_details"] = reasoning_details
        if message.tool_calls:
            value["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call.get("args") or {}),
                    },
                }
                for call in message.tool_calls
            ]
        return value
    return {"role": message.type, "content": message.content}


def _ai_message(body: dict[str, Any]) -> AIMessage:
    reasoning = body.get("reasoning") or {}
    reasoning_details = reasoning.get("details") or [
        item.get("reasoning")
        for item in body.get("output", [])
        if item.get("type") == "reasoning" and item.get("reasoning")
    ]
    texts = [
        item.get("text", "")
        if item["type"] == "text"
        else json.dumps(item.get("object"), separators=(",", ":"))
        for item in body.get("output", [])
        if item["type"] in {"text", "json"}
    ]
    calls = []
    for item in body.get("output", []):
        if item["type"] != "tool_call":
            continue
        call = item["tool_call"]
        function = call["function"]
        args = function.get("arguments") or "{}"
        calls.append(
            {
                "name": function["name"],
                "args": json.loads(args) if isinstance(args, str) else args,
                "id": call.get("id"),
                "type": "tool_call",
            }
        )
    usage = body.get("usage") or {}
    return AIMessage(
        content="".join(texts),
        tool_calls=calls,
        additional_kwargs={"reasoning_details": reasoning_details},
        usage_metadata={
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        response_metadata={
            "id": body.get("id"),
            "model": body.get("model"),
            "routing": body.get("routing"),
            "receipt": body.get("receipt"),
            "warnings": body.get("warnings"),
            "session_id": body.get("session_id"),
            "session_name": body.get("session_name"),
            "task_id": body.get("task_id"),
            "task_name": body.get("task_name"),
            "app_name": body.get("app_name"),
            "reasoning": reasoning,
        },
    )


def _latest_continuation_token(messages: list[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        reasoning = message.response_metadata.get("reasoning") or {}
        token = reasoning.get("continuation_token")
        if isinstance(token, str) and token:
            return token
    return None


def _has_tool_result_after_latest_assistant(messages: list[BaseMessage]) -> bool:
    latest_assistant_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AIMessage):
            latest_assistant_index = index
            break
    if latest_assistant_index is None:
        return False
    return any(
        isinstance(message, ToolMessage)
        for message in messages[latest_assistant_index + 1 :]
    )


class ChatMLJunction(BaseChatModel):
    """Native LangChain chat model for ML Junction's rich Responses API."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("MLJUNCTION_API_KEY", "")), repr=False
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv("MLJUNCTION_BASE_URL", "http://localhost:8001")
    )
    timeout: float = 120
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    reasoning: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    requirements: dict[str, Any] = Field(default_factory=dict)
    routing: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None
    compatibility: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    session_name: str | None = None
    task_id: str | None = None
    task_name: str | None = None
    app_name: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    _client: MLJunctionClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        if not self.api_key.get_secret_value():
            raise ValueError("api_key or MLJUNCTION_API_KEY is required")
        self._client = MLJunctionClient(
            api_key=self.api_key.get_secret_value(),
            base_url=self.base_url,
            timeout=self.timeout,
            app_name=self.app_name,
        )

    @property
    def _llm_type(self) -> str:
        return "mljunction-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url, "routing": self.routing}

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        return LangSmithParams(
            ls_provider="mljunction",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=self.temperature,
            ls_max_tokens=self.max_tokens,
            ls_stop=stop,
        )

    def _payload(
        self, messages: list[BaseMessage], *, stream: bool, stop: list[str] | None, **kwargs: Any
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message(message) for message in messages],
            "stream": stream,
            "sampling": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "stop": stop,
            },
            "reasoning": self.reasoning,
            "output": {**self.output, "max_tokens": self.max_tokens},
            "requirements": self.requirements,
            "routing": self.routing,
            "context": self.context,
            "metadata": self.metadata,
            "idempotency_key": self.idempotency_key,
            "session_id": self.session_id,
            "session_name": self.session_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "compatibility": self.compatibility,
        }
        payload.update(self.model_kwargs)
        payload.update(kwargs)
        reasoning = dict(payload.get("reasoning") or {})
        if "continuation_token" not in reasoning:
            token = _latest_continuation_token(messages)
            if token is not None:
                reasoning["continuation_token"] = token
        if (
            reasoning.get("continuation_token")
            and "input_type" not in reasoning
            and _has_tool_result_after_latest_assistant(messages)
        ):
            reasoning["input_type"] = "tool_results"
        payload["reasoning"] = reasoning
        payload["sampling"] = {
            key: value for key, value in payload.get("sampling", {}).items() if value is not None
        }
        payload["output"] = {
            key: value for key, value in payload.get("output", {}).items() if value is not None
        }
        return {key: value for key, value in payload.items() if value is not None}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        body = self._client.post(
            "/v1/responses", self._payload(messages, stream=False, stop=stop, **kwargs)
        )
        return ChatResult(
            generations=[ChatGeneration(message=_ai_message(body))], llm_output=body.get("routing")
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        body = await self._client.apost(
            "/v1/responses", self._payload(messages, stream=False, stop=stop, **kwargs)
        )
        return ChatResult(
            generations=[ChatGeneration(message=_ai_message(body))], llm_output=body.get("routing")
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        events = self._client.stream(
            "/v1/responses", self._payload(messages, stream=True, stop=stop, **kwargs)
        )
        for event, data in events:
            if event == "response.output_text.delta":
                message = AIMessageChunk(content=data.get("delta", ""))
            elif event == "response.output_json.done":
                message = AIMessageChunk(
                    content=json.dumps(
                        data.get("object"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            elif event == "response.tool_call.delta":
                message = AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": data.get("name"),
                            "args": data.get("arguments", ""),
                            "id": data.get("id"),
                            "index": data.get("index", 0),
                            "type": "tool_call_chunk",
                        }
                    ],
                )
            else:
                continue
            chunk = ChatGenerationChunk(message=message)
            if run_manager:
                run_manager.on_llm_new_token(data.get("delta", ""), chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        events = self._client.astream(
            "/v1/responses", self._payload(messages, stream=True, stop=stop, **kwargs)
        )
        async for event, data in events:
            if event == "response.output_text.delta":
                message = AIMessageChunk(content=data.get("delta", ""))
            elif event == "response.output_json.done":
                message = AIMessageChunk(
                    content=json.dumps(
                        data.get("object"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            elif event == "response.tool_call.delta":
                message = AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": data.get("name"),
                            "args": data.get("arguments", ""),
                            "id": data.get("id"),
                            "index": data.get("index", 0),
                            "type": "tool_call_chunk",
                        }
                    ],
                )
            else:
                continue
            chunk = ChatGenerationChunk(message=message)
            if run_manager:
                await run_manager.on_llm_new_token(data.get("delta", ""), chunk=chunk)
            yield chunk

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        *,
        tool_choice: dict[str, Any] | str | bool | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable:
        formatted = [convert_to_openai_tool(tool, strict=strict) for tool in tools]
        if tool_choice is True:
            if len(formatted) != 1:
                raise ValueError("tool_choice=True requires exactly one tool")
            tool_choice = formatted[0]["function"]["name"]
        if isinstance(tool_choice, str) and tool_choice not in {"auto", "none", "required"}:
            tool_choice = {"type": "function", "function": {"name": tool_choice}}
        return self.bind(
            tools=formatted,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            **kwargs,
        )

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        if isinstance(schema, type):
            json_schema = convert_to_openai_tool(schema, strict=True)["function"]["parameters"]
            parser = lambda message: schema.model_validate_json(message.content)  # noqa: E731
        else:
            json_schema = schema
            parser = lambda message: json.loads(message.content)  # noqa: E731
        runnable = self.bind(
            output={
                "format": {
                    "type": "json_schema",
                    "name": getattr(schema, "__name__", "response"),
                    "schema": json_schema,
                    "strict": True,
                }
            },
            **kwargs,
        )
        return runnable if include_raw else runnable | parser

    def close(self) -> None:
        """Close the synchronous transport owned by this model."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the asynchronous transport owned by this model."""
        await self._client.aclose()
