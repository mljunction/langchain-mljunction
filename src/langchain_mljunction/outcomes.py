"""Report what happened after a request.

An outcome is the second source of evidence behind an adaptive model. The first
is an offline evaluation suite you upload; this one is your own application
telling ML Junction whether the work actually succeeded - a ticket closing, a
patch passing its tests, a meeting getting booked.

The platform never interprets what a name means. It compares each name against
itself across the models in a pool, so it can rank on your business metric
without knowing your business. The only thing it stores about a name is whether
a bigger number is better.

Three rules worth knowing before you wire this up:

* **Register a name before you send it.** An unregistered name is refused, which
  is what stops a typo becoming a second half-populated metric that nothing ever
  surfaces.
* **`event_id` is yours, and it is required.** Retries and at-least-once queues
  are normal; without an idempotency key a redelivered flush inflates your
  numbers and nothing errors.
* **Address the unit the outcome is about.** A support ticket is many requests
  and one outcome, so report it against the `session_id` you sent with those
  requests, not against a single `request_id`.

    from langchain_mljunction import ChatMLJunction, Outcomes

    chat = ChatMLJunction(model="support-agent", session_id="ticket_8842")
    reply = chat.invoke("I want a refund")

    outcomes = Outcomes()
    outcomes.report(
        session_id="ticket_8842",
        event_id="ticket_8842_closed",
        metrics={"resolved": True, "csat": 4.8},
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_mljunction._client import MLJunctionClient

DEFAULT_BASE_URL = "https://api.mljunction.com"


@dataclass(frozen=True, slots=True)
class OutcomeReceipt:
    """What one report call did.

    ``discarded`` is not an error. An outcome whose session was answered by more
    than one candidate belongs to neither, and one whose request has aged past
    your retention window can no longer be attributed - both are stored and
    counted rather than silently dropped, so a rising number here tells you
    something real about the integration.
    """

    accepted: int
    duplicates: int
    discarded: dict[str, int]

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> OutcomeReceipt:
        return cls(
            accepted=int(body.get("accepted", 0)),
            duplicates=int(body.get("duplicates", 0)),
            discarded=dict(body.get("discarded") or {}),
        )


def _payload(
    *,
    metrics: dict[str, bool | float],
    event_id: str,
    request_id: str | None,
    session_id: str | None,
    task_id: str | None,
    observed_at: datetime | None,
) -> dict[str, Any]:
    targets = {"request_id": request_id, "session_id": session_id, "task_id": task_id}
    supplied = {key: value for key, value in targets.items() if value}
    if not supplied:
        raise ValueError("one of request_id, session_id or task_id is required")
    if len(supplied) > 1:
        # Refused rather than resolved by precedence: two keys usually means the
        # caller is unsure what the outcome covers, and guessing would attribute
        # a ticket-level result to a single message or the reverse.
        raise ValueError(f"supply exactly one target, got {', '.join(sorted(supplied))}")
    if not metrics:
        raise ValueError("at least one metric is required")

    body: dict[str, Any] = {"event_id": event_id, "metrics": metrics, **supplied}
    if observed_at is not None:
        body["observed_at"] = observed_at.isoformat()
    return body


class Outcomes:
    """Client for ``POST /v1/outcomes``.

    Reporting reuses the inference credential. Definition registration is a
    control-plane mutation and therefore requires a separate management key
    carrying ``adaptive:write``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        management_api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        app_name: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("MLJUNCTION_API_KEY")
        if not resolved_key:
            raise ValueError(
                "an API key is required: pass api_key= or set MLJUNCTION_API_KEY"
            )
        resolved_base_url = base_url or os.environ.get("MLJUNCTION_BASE_URL") or DEFAULT_BASE_URL
        self._client = MLJunctionClient(
            api_key=resolved_key,
            base_url=resolved_base_url,
            timeout=timeout,
            app_name=app_name,
        )
        resolved_management_key = management_api_key or os.environ.get(
            "MLJUNCTION_MANAGEMENT_API_KEY"
        )
        self._management_client = (
            MLJunctionClient(
                api_key=resolved_management_key,
                base_url=resolved_base_url,
                timeout=timeout,
                app_name=app_name,
            )
            if resolved_management_key
            else None
        )

    # -------------------------------------------------------------- reporting

    def report(
        self,
        *,
        metrics: dict[str, bool | float],
        event_id: str,
        request_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        observed_at: datetime | None = None,
    ) -> OutcomeReceipt:
        """Report one business event, carrying every metric it settled.

        ``metrics`` is a mapping because a closing ticket settles several facts
        at once - resolved AND rated AND not escalated. One call writes them all;
        forcing three calls for one event is three chances to half-fail.

        Pass ``observed_at`` when the event happened earlier than this call. A
        ticket that closed at 11:30 belongs in the 11:30 window even if the queue
        only flushed at 13:42.
        """
        body = _payload(
            metrics=metrics,
            event_id=event_id,
            request_id=request_id,
            session_id=session_id,
            task_id=task_id,
            observed_at=observed_at,
        )
        return OutcomeReceipt.from_body(self._client.post("/v1/outcomes", body))

    async def areport(
        self,
        *,
        metrics: dict[str, bool | float],
        event_id: str,
        request_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        observed_at: datetime | None = None,
    ) -> OutcomeReceipt:
        """Async :meth:`report`."""
        body = _payload(
            metrics=metrics,
            event_id=event_id,
            request_id=request_id,
            session_id=session_id,
            task_id=task_id,
            observed_at=observed_at,
        )
        return OutcomeReceipt.from_body(await self._client.apost("/v1/outcomes", body))

    def report_batch(self, reports: list[dict[str, Any]]) -> OutcomeReceipt:
        """Send many reports in one call.

        Outcomes are usually flushed from a queue rather than sent one at a time
        from a hot path, and a batch is one round trip instead of five hundred.
        Each entry takes the same keyword arguments as :meth:`report`.
        """
        if not reports:
            raise ValueError("report_batch needs at least one report")
        payload = {"outcomes": [_payload(**report) for report in _normalized(reports)]}
        return OutcomeReceipt.from_body(self._client.post("/v1/outcomes", payload))

    async def areport_batch(self, reports: list[dict[str, Any]]) -> OutcomeReceipt:
        """Async :meth:`report_batch`."""
        if not reports:
            raise ValueError("report_batch needs at least one report")
        payload = {"outcomes": [_payload(**report) for report in _normalized(reports)]}
        return OutcomeReceipt.from_body(await self._client.apost("/v1/outcomes", payload))

    # ------------------------------------------------------------ definitions

    def register(
        self,
        name: str,
        *,
        direction: str = "maximize",
        display_name: str | None = None,
        description: str | None = None,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
    ) -> dict[str, Any]:
        """Register a metric name so reports carrying it are accepted.

        ``direction`` is the entire vocabulary: ``maximize`` or ``minimize``.
        Leave the bounds at 0-1 for a yes/no metric; a 1-5 rating should declare
        its real range so the confidence interval is computed on the scale you
        actually report.
        """
        if self._management_client is None:
            raise ValueError(
                "register() requires a management key with adaptive:write scope: "
                "pass management_api_key= or set MLJUNCTION_MANAGEMENT_API_KEY"
            )
        return self._management_client.put(
            f"/v1/management/adaptive/outcome-definitions/{name}",
            {
                "direction": direction,
                "display_name": display_name,
                "description": description,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        )

    async def aregister(
        self,
        name: str,
        *,
        direction: str = "maximize",
        display_name: str | None = None,
        description: str | None = None,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
    ) -> dict[str, Any]:
        """Async :meth:`register`."""
        if self._management_client is None:
            raise ValueError(
                "aregister() requires a management key with adaptive:write scope: "
                "pass management_api_key= or set MLJUNCTION_MANAGEMENT_API_KEY"
            )
        return await self._management_client.aput(
            f"/v1/management/adaptive/outcome-definitions/{name}",
            {
                "direction": direction,
                "display_name": display_name,
                "description": description,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        )

    def close(self) -> None:
        self._client.close()
        if self._management_client is not None:
            self._management_client.close()

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._management_client is not None:
            await self._management_client.aclose()


def _normalized(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill in the keys `_payload` expects, so callers may omit the unused ones."""
    defaults = {
        "request_id": None,
        "session_id": None,
        "task_id": None,
        "observed_at": None,
    }
    return [{**defaults, **report} for report in reports]


def request_id_of(message: Any) -> str | None:
    """The ML Junction request id from a LangChain message.

    Convenience for the one-shot case: ``report(request_id=request_id_of(reply))``.
    Returns ``None`` for a message that did not come from this integration.
    """
    metadata = getattr(message, "response_metadata", None) or {}
    return metadata.get("id")
