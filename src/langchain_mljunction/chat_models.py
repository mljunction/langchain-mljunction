from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator, Sequence
from operator import itemgetter
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
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr

from langchain_mljunction._client import MLJunctionClient


def _data_url(data: str, media_type: str) -> str:
    return data if data.startswith("data:") else f"data:{media_type};base64,{data}"


def _content(content: Any) -> Any:
    """Normalize LangChain content blocks to the ML Junction native contract."""
    if not isinstance(content, list):
        return content
    normalized: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            normalized.append({"type": "text", "text": str(block)})
            continue
        kind = block.get("type")
        if kind == "text":
            value = {"type": "text", "text": block.get("text", "")}
            if block.get("cache_control"):
                value["cache_control"] = block["cache_control"]
            normalized.append(value)
        elif kind == "image_url":
            image = block.get("image_url")
            url = image.get("url", "") if isinstance(image, dict) else image or ""
            normalized.append({"type": "image_url", "image_url": url})
        elif kind == "image":
            source = block.get("source") if isinstance(block.get("source"), dict) else {}
            media_type = block.get("mime_type") or source.get("media_type") or "image/png"
            data = block.get("base64") or source.get("data")
            url = block.get("url")
            if data:
                url = _data_url(str(data), str(media_type))
            normalized.append({"type": "image_url", "image_url": url or ""})
        elif kind in {"audio", "input_audio"}:
            audio = block.get("input_audio") if isinstance(block.get("input_audio"), dict) else {}
            media_type = block.get("mime_type") or f"audio/{audio.get('format', 'wav')}"
            normalized.append(
                {
                    "type": "input_audio",
                    "data": block.get("base64") or audio.get("data") or "",
                    "media_type": media_type,
                }
            )
        elif kind == "file":
            file = block.get("file") if isinstance(block.get("file"), dict) else {}
            media_type = block.get("mime_type") or "application/octet-stream"
            data = block.get("base64") or file.get("file_data")
            normalized.append(
                {
                    "type": "file",
                    "data": _data_url(str(data), str(media_type)) if data else None,
                    "file_id": block.get("file_id") or file.get("file_id"),
                    "file_url": block.get("url") or file.get("file_url"),
                    "filename": block.get("filename") or file.get("filename"),
                    "media_type": media_type,
                }
            )
        elif kind == "tool_result":
            nested = block.get("content", "")
            if isinstance(nested, str):
                normalized.append({"type": "text", "text": nested})
            else:
                normalized.extend(_content(nested) or [])
        elif kind in {"thinking", "redacted_thinking"}:
            normalized.append(block)
        elif kind not in {"tool_call", "tool_use"}:
            normalized.append({"type": "text", "text": json.dumps(block, ensure_ascii=False)})
    return [
        {key: value for key, value in block.items() if value is not None} for block in normalized
    ]


def _message(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, SystemMessage):
        return {
            "role": "system",
            "content": _content(message.content),
            **({"name": message.name} if message.name else {}),
        }
    if isinstance(message, HumanMessage):
        return {
            "role": "user",
            "content": _content(message.content),
            **({"name": message.name} if message.name else {}),
        }
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": _content(message.content),
            "tool_call_id": message.tool_call_id,
            **({"name": message.name} if message.name else {}),
        }
    if isinstance(message, AIMessage):
        content: Any = _content(message.content) or None
        value: dict[str, Any] = {"role": "assistant", "content": content}
        if message.name:
            value["name"] = message.name
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
            "model_name": body.get("model"),
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
        isinstance(message, ToolMessage) for message in messages[latest_assistant_index + 1 :]
    )


def _ai_message_chunk(message: AIMessage, *, include_output: bool = True) -> AIMessageChunk:
    """Convert a complete response to one merge-safe LangChain chunk."""
    tool_call_chunks = []
    if include_output:
        tool_call_chunks = [
            {
                "name": call["name"],
                "args": json.dumps(call.get("args") or {}, ensure_ascii=False),
                "id": call.get("id"),
                "index": index,
                "type": "tool_call_chunk",
            }
            for index, call in enumerate(message.tool_calls)
        ]
    return AIMessageChunk(
        content=message.content if include_output else "",
        additional_kwargs=message.additional_kwargs,
        response_metadata=message.response_metadata,
        usage_metadata=message.usage_metadata,
        tool_call_chunks=tool_call_chunks,
    )


class ChatMLJunction(BaseChatModel):
    """Native LangChain chat model for ML Junction's rich Responses API."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("MLJUNCTION_API_KEY", "")), repr=False
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv("MLJUNCTION_BASE_URL", "https://api.mljunction.com")
    )
    timeout: float = 120
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
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
            ls_model_name=kwargs.get("model", self.model),
            ls_model_type="chat",
            ls_temperature=self.temperature,
            ls_max_tokens=self.max_tokens,
            ls_stop=stop if stop is not None else self.stop,
        )

    def _structured_output_bound(self, **kwargs: Any) -> bool:
        """Return whether native JSON Schema output is active for this call."""
        call_output = kwargs.get("output")
        merged_output = {
            **self.output,
            **(call_output if isinstance(call_output, dict) else {}),
        }
        output_format = merged_output.get("format")
        return isinstance(output_format, dict) and output_format.get("type") == "json_schema"

    def _atomic_structured_stream(self, **kwargs: Any) -> bool:
        return bool(kwargs.get("_mljunction_structured_output")) or self._structured_output_bound(
            **kwargs
        )

    @staticmethod
    def _without_internal_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(kwargs)
        cleaned.pop("_mljunction_structured_output", None)
        return cleaned

    def _payload(
        self, messages: list[BaseMessage], *, stream: bool, stop: list[str] | None, **kwargs: Any
    ) -> dict[str, Any]:
        kwargs = self._without_internal_kwargs(kwargs)
        effective_stop = stop if stop is not None else self.stop
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
                "stop": effective_stop,
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
        if self._atomic_structured_stream(**kwargs):
            result = self._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            generation = result.generations[0]
            message = _ai_message_chunk(generation.message)
            chunk = ChatGenerationChunk(
                message=message,
                generation_info=generation.generation_info,
            )
            yield chunk
            return
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
            elif event == "response.completed":
                complete = _ai_message(data)
                if not complete.response_metadata.get("model_name"):
                    model_name = kwargs.get("model", self.model)
                    complete.response_metadata["model"] = model_name
                    complete.response_metadata["model_name"] = model_name
                message = _ai_message_chunk(complete, include_output=False)
            else:
                continue
            chunk = ChatGenerationChunk(message=message)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self._atomic_structured_stream(**kwargs):
            result = await self._agenerate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            generation = result.generations[0]
            message = _ai_message_chunk(generation.message)
            chunk = ChatGenerationChunk(
                message=message,
                generation_info=generation.generation_info,
            )
            yield chunk
            return
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
            elif event == "response.completed":
                complete = _ai_message(data)
                if not complete.response_metadata.get("model_name"):
                    model_name = kwargs.get("model", self.model)
                    complete.response_metadata["model"] = model_name
                    complete.response_metadata["model_name"] = model_name
                message = _ai_message_chunk(complete, include_output=False)
            else:
                continue
            chunk = ChatGenerationChunk(message=message)
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
        if tool_choice is False:
            tool_choice = "none"
        if tool_choice is True:
            if len(formatted) != 1:
                raise ValueError("tool_choice=True requires exactly one tool")
            tool_choice = formatted[0]["function"]["name"]
        if tool_choice == "any":
            tool_choice = "required"
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
        method: str = "json_schema",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable:
        if method not in {"json_schema", "function_calling", "json_mode"}:
            raise ValueError(
                "method must be one of 'json_schema', 'function_calling', or 'json_mode'"
            )
        if method == "json_mode" and strict is not None:
            raise ValueError("strict is not supported with method='json_mode'")

        is_pydantic = isinstance(schema, type) and is_basemodel_subclass(schema)
        formatted_tool = convert_to_openai_tool(schema, strict=strict)
        function = formatted_tool["function"]
        schema_name = function["name"]

        def parse_content(message: AIMessage) -> Any:
            if is_pydantic:
                if hasattr(schema, "model_validate_json"):
                    return schema.model_validate_json(message.content)
                return schema.parse_raw(message.content)
            return json.loads(message.content)

        def parse_tool_call(message: AIMessage) -> Any:
            if not message.tool_calls:
                raise ValueError("The model did not return the required structured tool call")
            args = message.tool_calls[0]["args"]
            if not is_pydantic:
                return args
            if hasattr(schema, "model_validate"):
                return schema.model_validate(args)
            return schema.parse_obj(args)

        trace_format = {
            "kwargs": {"method": method},
            "schema": schema,
        }
        if method == "function_calling":
            runnable = self.bind_tools(
                [formatted_tool],
                tool_choice=schema_name,
                strict=strict,
                _mljunction_structured_output=True,
                ls_structured_output_format=trace_format,
                **kwargs,
            )
            parser = parse_tool_call
        else:
            output_format: dict[str, Any] = {"type": "json_object"}
            if method == "json_schema":
                output_format = {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": function["parameters"],
                    "strict": True if strict is None else strict,
                }
            runnable = self.bind(
                output={"format": output_format},
                _mljunction_structured_output=True,
                ls_structured_output_format=trace_format,
                **kwargs,
            )
            parser = parse_content
        parser_runnable = RunnableLambda(parser)
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parser_runnable,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=runnable) | parser_with_fallback
        return runnable | parser_runnable

    def close(self) -> None:
        """Close the synchronous transport owned by this model."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the asynchronous transport owned by this model."""
        await self._client.aclose()
