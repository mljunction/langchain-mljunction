from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_mljunction import ChatMLJunction


class TestChatMLJunctionUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatMLJunction

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {"model": "test-model", "api_key": "test-key"}

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"MLJUNCTION_API_KEY": "env-test-key"},
            {"model": "test-model"},
            {"api_key": "env-test-key"},
        )

    @property
    def structured_output_kwargs(self) -> dict[str, Any]:
        return {"method": "json_schema"}

    @property
    def model_override_value(self) -> str:
        return "test-model-override"
