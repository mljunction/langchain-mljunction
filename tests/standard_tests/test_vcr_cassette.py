from pathlib import Path

import yaml

CASSETTE = (
    Path(__file__).parents[1] / "cassettes" / "TestChatMLJunctionIntegration.test_stream_time.yaml"
)


def test_stream_timing_cassette_is_sanitized() -> None:
    raw = CASSETTE.read_text(encoding="utf-8")
    assert "astro_test_" not in raw
    assert "Bearer " not in raw
    assert "password" not in raw.lower()
    assert "secret" not in raw.lower()

    cassette = yaml.safe_load(raw)
    assert cassette["interactions"]
    for interaction in cassette["interactions"]:
        request = interaction["request"]
        headers = request["headers"]
        assert headers["authorization"] == ["PLACEHOLDER"]
        assert headers["user-agent"] == ["PLACEHOLDER"]
        assert request["uri"].startswith("http://127.0.0.1:8001/")
        assert interaction["response"]["headers"] == {}
