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


@pytest.fixture(autouse=True)
def _no_process_tracer(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Keep automatic tracing off unless a test installs a tracer itself.

    Every ChatMLJunction built with a key would otherwise start a real OTLP
    exporter, and a tracer installed by one test would leak into the next.
    """
    from langchain_mljunction import autotrace

    monkeypatch.setenv("MLJUNCTION_TRACING", "false")
    monkeypatch.setattr(autotrace, "_tracer", None)
    monkeypatch.setattr(autotrace, "_unavailable", False)
    yield
