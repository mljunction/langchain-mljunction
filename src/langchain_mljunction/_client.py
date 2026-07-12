from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx


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
        response.raise_for_status()
        return response.json()

    async def apost(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.async_.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    def stream(self, path: str, payload: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
        with self.sync.stream("POST", path, json=payload) as response:
            response.raise_for_status()
            yield from _parse_sse(response.iter_lines())

    async def astream(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        async with self.async_.stream("POST", path, json=payload) as response:
            response.raise_for_status()
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
