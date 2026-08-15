import os
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_mljunction import ChatMLJunction

RUN_LIVE = os.getenv("RUN_LIVE_PROVIDER_TESTS") == "1" and bool(os.getenv("LIVE_API_KEY"))

pytestmark = pytest.mark.skipif(not RUN_LIVE, reason="live integration test disabled")


class TestChatMLJunctionIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatMLJunction

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {
            "model": os.getenv(
                "LIVE_LANGCHAIN_MODEL",
                os.getenv("LIVE_OPENAI_MODEL", "gemini-2.5-flash-lite"),
            ),
            "api_key": os.environ.get("LIVE_API_KEY", "live-test-key"),
            "base_url": os.getenv("LIVE_API_BASE", "http://localhost:8001"),
            "max_tokens": 256,
        }

    @property
    def structured_output_kwargs(self) -> dict[str, Any]:
        return {"method": "json_schema"}

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def model_override_value(self) -> str:
        return os.getenv("LIVE_MODEL_OVERRIDE", "gemini-2.5-flash")

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return True

    @property
    def supports_image_tool_message(self) -> bool:
        return True

    @property
    def supports_pdf_tool_message(self) -> bool:
        return True

    @property
    def enable_vcr_tests(self) -> bool:
        return True
