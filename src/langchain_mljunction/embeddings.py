from __future__ import annotations

import base64
import os
import struct
from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, PrivateAttr, SecretStr

from langchain_mljunction._client import MLJunctionClient


class MLJunctionEmbeddings(BaseModel, Embeddings):
    model: str
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("MLJUNCTION_API_KEY", "")), repr=False
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv("MLJUNCTION_BASE_URL", "http://localhost:8001")
    )
    timeout: float = 120
    dimensions: int | None = None
    encoding_format: str = "float"
    user: str | None = None
    routing: dict[str, Any] = Field(default_factory=dict)
    app_name: str | None = None
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

    def _payload(self, texts: list[str]) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
            "encoding_format": self.encoding_format,
            "user": self.user,
            "routing": self.routing,
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        body = self._client.post("/v1/embeddings", self._payload(texts))
        return [self._vector(item["embedding"]) for item in body["data"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        body = await self._client.apost("/v1/embeddings", self._payload(texts))
        return [self._vector(item["embedding"]) for item in body["data"]]

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    @staticmethod
    def _vector(value: list[float] | str) -> list[float]:
        if isinstance(value, list):
            return value
        raw = base64.b64decode(value, validate=True)
        if len(raw) % 4:
            raise ValueError("Invalid base64 embedding byte length")
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._client.aclose()
