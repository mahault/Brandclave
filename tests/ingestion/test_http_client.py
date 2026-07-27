"""Tests for the shared resilient HTTP helper (ingestion/http_client.py)."""

import httpx
import pytest

from ingestion import http_client
from ingestion.http_client import resilient_get, resilient_request


@pytest.fixture
def sleeps(monkeypatch):
    """Record backoff sleeps instead of actually sleeping."""
    recorded: list[float] = []
    monkeypatch.setattr(http_client, "_sleep", recorded.append)
    return recorded


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def test_success_passthrough(sleeps):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text="ok")

    with make_client(handler) as client:
        response = resilient_get("https://example.com/feed", client=client)

    assert response is not None
    assert response.status_code == 200
    assert response.text == "ok"
    assert len(calls) == 1
    assert sleeps == []


def test_retry_then_success_on_500(sleeps):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(500)
        return httpx.Response(200, text="recovered")

    with make_client(handler) as client:
        response = resilient_get("https://example.com/flaky", retries=3, client=client)

    assert response is not None
    assert response.status_code == 200
    assert response.text == "recovered"
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_none_after_exhausted_retries(sleeps):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    with make_client(handler) as client:
        response = resilient_get("https://example.com/dead", retries=3, client=client)

    assert response is None
    assert len(calls) == 3
    # No sleep after the final attempt
    assert len(sleeps) == 2


def test_retry_after_honored_on_429(sleeps):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, text="ok")

    with make_client(handler) as client:
        response = resilient_get("https://example.com/rate", retries=3, client=client)

    assert response is not None
    assert response.status_code == 200
    assert len(calls) == 2
    assert sleeps == [2.0]


def test_retry_after_capped(sleeps):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "3600"})
        return httpx.Response(200)

    with make_client(handler) as client:
        response = resilient_get("https://example.com/rate", retries=3, client=client)

    assert response is not None
    assert sleeps == [http_client.MAX_RETRY_AFTER]


def test_timeout_produces_retry(sleeps):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("read timed out", request=request)
        return httpx.Response(200, text="ok")

    with make_client(handler) as client:
        response = resilient_get("https://example.com/slow", retries=3, client=client)

    assert response is not None
    assert response.status_code == 200
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_all_timeouts_return_none_without_raising(sleeps):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectTimeout("connect timed out", request=request)

    with make_client(handler) as client:
        response = resilient_get("https://example.com/black-hole", retries=3, client=client)

    assert response is None
    assert len(calls) == 3


def test_non_retryable_status_passes_through(sleeps):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404)

    with make_client(handler) as client:
        response = resilient_get("https://example.com/missing", retries=3, client=client)

    assert response is not None
    assert response.status_code == 404
    assert len(calls) == 1
    assert sleeps == []


def test_resilient_request_post(sleeps):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(502)
        return httpx.Response(201, json={"ok": True})

    with make_client(handler) as client:
        response = resilient_request(
            "POST",
            "https://example.com/api",
            retries=3,
            client=client,
            json={"key": "value"},
        )

    assert response is not None
    assert response.status_code == 201
    assert calls[0].method == "POST"
    assert len(calls) == 2


def test_params_and_headers_forwarded(sleeps):
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["header"] = request.headers.get("X-Test")
        return httpx.Response(200)

    with make_client(handler) as client:
        response = resilient_get(
            "https://example.com/search",
            params={"q": "hotels"},
            headers={"X-Test": "yes"},
            client=client,
        )

    assert response is not None
    assert captured["url"] == "https://example.com/search?q=hotels"
    assert captured["header"] == "yes"
