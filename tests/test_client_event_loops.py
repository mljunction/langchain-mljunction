"""The async pool must follow the running event loop.

Callers cache chat models across event loops (one ``asyncio.run`` per script
step, per test, per worker thread). An ``httpx.AsyncClient`` keeps connections
that belong to the loop that opened them, so a single shared pool failed with
``RuntimeError: Event loop is closed`` on the second loop. A real local socket
is used on purpose: a ``MockTransport`` owns no connection and hides the bug.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from langchain_mljunction._client import MLJunctionAPIError, MLJunctionClient


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        failing = self.path == "/fail"
        body = json.dumps(
            {"error": {"code": "bad_request", "message": "nope"}} if failing else {"ok": True}
        ).encode()
        self.send_response(400 if failing else 200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def base_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_one_client_serves_sequential_event_loops(base_url: str) -> None:
    client = MLJunctionClient(api_key="k", base_url=base_url, timeout=5)

    assert asyncio.run(client.apost("/x", {})) == {"ok": True}
    assert asyncio.run(client.apost("/x", {})) == {"ok": True}


def test_aclose_is_idempotent_and_the_client_reopens(base_url: str) -> None:
    client = MLJunctionClient(api_key="k", base_url=base_url, timeout=5)

    async def run() -> dict:
        await client.apost("/x", {})
        await client.aclose()
        await client.aclose()
        return await client.apost("/x", {})

    assert asyncio.run(run()) == {"ok": True}


def test_async_post_error_raises_structured_api_error(base_url: str) -> None:
    client = MLJunctionClient(api_key="k", base_url=base_url, timeout=5)

    with pytest.raises(MLJunctionAPIError) as caught:
        asyncio.run(client.apost("/fail", {}))

    assert caught.value.status_code == 400
    assert caught.value.code == "bad_request"
