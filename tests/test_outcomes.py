"""Outcome reporting: the addressing rules and the idempotency key."""

from datetime import UTC, datetime
from typing import ClassVar

import pytest

import langchain_mljunction.outcomes as outcomes_module
from langchain_mljunction.outcomes import (
    OutcomeReceipt,
    Outcomes,
    _payload,
    request_id_of,
)


def _valid(**overrides):
    body = {
        "metrics": {"resolved": True},
        "event_id": "e1",
        "request_id": None,
        "session_id": "ticket_8842",
        "task_id": None,
        "observed_at": None,
    }
    body.update(overrides)
    return body


class TestAddressing:
    def test_a_report_must_say_what_it_is_about(self) -> None:
        with pytest.raises(ValueError, match="request_id, session_id or task_id"):
            _payload(**_valid(session_id=None))

    def test_two_targets_are_refused_rather_than_resolved_by_precedence(self) -> None:
        """Two keys usually means the caller is unsure what the outcome covers,
        and guessing attributes a ticket-level result to a single message."""
        with pytest.raises(ValueError, match="exactly one target"):
            _payload(**_valid(request_id="req_1"))

    @pytest.mark.parametrize("field", ["request_id", "session_id", "task_id"])
    def test_each_target_works_alone(self, field: str) -> None:
        args = _valid(session_id=None)
        args[field] = "x"
        body = _payload(**args)
        assert body[field] == "x"
        # Unused targets are omitted entirely rather than sent as null.
        assert len([k for k in body if k.endswith("_id") and k != "event_id"]) == 1


class TestMetrics:
    def test_at_least_one_metric_is_required(self) -> None:
        with pytest.raises(ValueError, match="at least one metric"):
            _payload(**_valid(metrics={}))

    def test_several_facts_ride_one_event(self) -> None:
        """A closing ticket settles resolved AND rated AND not escalated. Three
        calls for one event is three chances to half-fail."""
        body = _payload(**_valid(metrics={"resolved": True, "csat": 4.8, "escalated": False}))
        assert body["metrics"] == {"resolved": True, "csat": 4.8, "escalated": False}


class TestObservedAt:
    def test_it_is_omitted_when_not_supplied(self) -> None:
        assert "observed_at" not in _payload(**_valid())

    def test_it_is_sent_as_iso_when_the_event_predates_the_call(self) -> None:
        """A ticket that closed at 11:30 belongs in the 11:30 window even if the
        queue only flushed at 13:42."""
        when = datetime(2026, 8, 27, 11, 30, tzinfo=UTC)
        assert _payload(**_valid(observed_at=when))["observed_at"] == when.isoformat()


class TestReceipt:
    def test_a_clean_flush(self) -> None:
        receipt = OutcomeReceipt.from_body({"accepted": 12, "duplicates": 0, "discarded": {}})
        assert (receipt.accepted, receipt.duplicates) == (12, 0)

    def test_a_replayed_queue_reports_duplicates_rather_than_failing(self) -> None:
        receipt = OutcomeReceipt.from_body({"accepted": 0, "duplicates": 12, "discarded": {}})
        assert receipt.duplicates == 12

    def test_discards_are_surfaced_not_hidden(self) -> None:
        """A rising discard count says something real about the integration."""
        receipt = OutcomeReceipt.from_body(
            {"accepted": 9, "duplicates": 0, "discarded": {"spanned_candidates": 3}}
        )
        assert receipt.discarded["spanned_candidates"] == 3

    def test_a_sparse_body_does_not_crash(self) -> None:
        receipt = OutcomeReceipt.from_body({})
        assert (receipt.accepted, receipt.duplicates, receipt.discarded) == (0, 0, {})


class TestRequestIdHelper:
    def test_it_reads_the_id_off_a_reply(self) -> None:
        message = type("M", (), {"response_metadata": {"id": "req_abc"}})()
        assert request_id_of(message) == "req_abc"

    def test_a_message_from_elsewhere_returns_none(self) -> None:
        assert request_id_of(type("M", (), {})()) is None


class _FakeClient:
    instances: ClassVar[list["_FakeClient"]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.put_calls: list[tuple[str, dict]] = []
        self.closed = False
        self.instances.append(self)

    def put(self, path: str, payload: dict) -> dict:
        self.put_calls.append((path, payload))
        return {"name": path.rsplit("/", 1)[-1], **payload}

    def close(self) -> None:
        self.closed = True


class TestDefinitionRegistrationCredentials:
    def setup_method(self) -> None:
        _FakeClient.instances.clear()

    def test_registration_refuses_an_inference_key(self, monkeypatch) -> None:
        monkeypatch.setattr(outcomes_module, "MLJunctionClient", _FakeClient)
        client = Outcomes(api_key="mlj_live_inference")

        with pytest.raises(ValueError, match=r"management key.*adaptive:write"):
            client.register("resolved")

    def test_registration_uses_the_separate_management_plane(self, monkeypatch) -> None:
        monkeypatch.setattr(outcomes_module, "MLJunctionClient", _FakeClient)
        client = Outcomes(
            api_key="mlj_live_inference",
            management_api_key="mlj_mgmt_control",
            base_url="http://localhost:8000",
        )

        body = client.register("resolved")

        assert body["name"] == "resolved"
        assert len(_FakeClient.instances) == 2
        inference, management = _FakeClient.instances
        assert inference.kwargs["api_key"] == "mlj_live_inference"
        assert management.kwargs["api_key"] == "mlj_mgmt_control"
        assert management.put_calls[0][0] == (
            "/v1/management/adaptive/outcome-definitions/resolved"
        )

        client.close()
        assert inference.closed is True
        assert management.closed is True
