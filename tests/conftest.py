from typing import Any

import pytest
from langchain_tests.conftest import base_vcr_config


def _sanitize_response(response: dict[str, Any]) -> dict[str, Any]:
    """Keep cassette payloads deterministic without retaining server headers."""
    response["headers"] = {}
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    """Use LangChain's credential filters and ML Junction-specific redaction."""
    config = base_vcr_config()
    config.setdefault("filter_headers", []).extend(
        [
            ("user-agent", "PLACEHOLDER"),
            ("x-app", "PLACEHOLDER"),
        ]
    )
    config["before_record_response"] = _sanitize_response
    return config
