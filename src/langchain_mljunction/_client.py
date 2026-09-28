from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx


class MLJunctionAPIError(RuntimeError):
    """Structured gateway failure with status, request ID, code, and details."""

    def __init__(self, response: httpx.Response) -> None:
        self.status_code = response.status_code
        self.request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        try:
            body = response.json()
        except ValueError:
            body = {}
        error = body.get("error", body) if isinstance(body, dict) else {}
        self.code = error.get("code") if isinstance(error, dict) else None
        self.error_type = error.get("type") if isinstance(error, dict) else None
        self.details = error.get("details", {}) if isinstance(error, dict) else {}
        message = (
            error.get("message") if isinstance(error, dict) else None
        ) or response.reason_phrase
        super().__init__(
            f"ML Junction request failed ({self.status_code}, {self.code or 'unknown'}): {message}"
        )


class MLJunctionStreamError(RuntimeError):
    """A stream that did not end in a successful ``response.completed``.

    Raised when the gateway reports ``response.failed``, or when the connection
    ends before any terminal event. ``partial`` is True when some output was
    already streamed: that text is real, but it is not a complete answer.
    """

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        code: str | None = None,
        error_type: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
        partial: bool = False,
    ) -> None:
        self.request_id = request_id
        self.code = code
        self.error_type = error_type
        self.status_code = status_code
        self.retryable = retryable
        self.partial = partial
        super().__init__(message)

    @classmethod
    def from_failed_event(cls, data: dict[str, Any]) -> MLJunctionStreamError:
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        code = error.get("code")
        return cls(
            f"ML Junction stream failed ({code or 'unknown'}): "
            f"{error.get('message') or 'the request did not complete'}",
            request_id=data.get("id"),
            code=code,
            error_type=error.get("type"),
            status_code=error.get("status"),
            retryable=bool(error.get("retryable")),
            partial=bool(data.get("partial")),
        )


def bare_model_name(model: str) -> str:
    """``openai/gpt-4.1-mini`` -> ``gpt-4.1-mini``, for OpenRouter-style names."""
    return model.split("/", 1)[1] if "/" in model else model


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_error:
        if not response.is_closed:
            response.read()
        raise MLJunctionAPIError(response)


async def _araise_for_status(response: httpx.Response) -> None:
    if response.is_error:
        if not response.is_closed:
            await response.aread()
        raise MLJunctionAPIError(response)


class MLJunctionClient:
    """Small native transport shared by LangChain integrations."""

    def __init__(
        self, *, api_key: str, base_url: str, timeout: float, app_name: str | None = None
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if app_name:
            headers["X-App"] = app_name
        self.sync = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        self.async_ = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.sync.post(path, json=payload)
        _raise_for_status(response)
        return response.json()

    async def apost(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.async_.post(path, json=payload)
        _raise_for_status(response)
        return response.json()

    def put(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Upsert. Used by outcome-definition registration, which is idempotent
        by name rather than creating a new row per call."""
        response = self.sync.put(path, json=payload)
        _raise_for_status(response)
        return response.json()

    async def aput(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.async_.put(path, json=payload)
        await _araise_for_status(response)
        return response.json()

    def stream(self, path: str, payload: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
        with self.sync.stream("POST", path, json=payload) as response:
            _raise_for_status(response)
            yield from _parse_sse(response.iter_lines())

    async def astream(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        async with self.async_.stream("POST", path, json=payload) as response:
            await _araise_for_status(response)
            event: str | None = None
            data: list[str] = []
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
                elif not line and event and data:
                    yield event, json.loads("\n".join(data))
                    event, data = None, []

    def close(self) -> None:
        self.sync.close()

    async def aclose(self) -> None:
        await self.async_.aclose()


def _parse_sse(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    event: str | None = None
    data: list[str] = []
    for line in lines:
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif not line and event and data:
            yield event, json.loads("\n".join(data))
            event, data = None, []
