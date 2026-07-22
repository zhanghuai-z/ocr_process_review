from __future__ import annotations

import requests
import pytest

from app.integrations.paddle.vl_client import (
    PaddleVLClient,
    PaddleVLClientError,
    PaddleVLQueueFullError,
)


class _Response:
    def __init__(self, status_code: int, body: object, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self) -> object:
        return self._body

    def raise_for_status(self) -> None:
        return None


def test_direct_network_mode_disables_environment_proxies(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    response = object()
    monkeypatch.setattr(
        requests,
        "get",
        lambda _url, **kwargs: calls.append(kwargs) or response,
    )

    client = PaddleVLClient(network_mode="direct")

    assert client._request("GET", "https://example.test") is response
    assert calls == [{"proxies": {"http": None, "https": None, "all": None}}]


def test_auto_network_mode_retries_with_environment_proxy(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_get(_url: str, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise requests.exceptions.ConnectionError("direct failed")
        return "response"

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.test:8080")
    monkeypatch.setattr(requests, "get", fake_get)
    client = PaddleVLClient(network_mode="auto")

    assert client._request("GET", "https://example.test") == "response"
    assert calls[0]["proxies"] == {"http": None, "https": None, "all": None}
    assert "proxies" not in calls[1]


def test_submit_retries_only_remote_queue_full_and_reuploads_full_content(monkeypatch) -> None:
    responses = iter(
        (
            _Response(400, {"code": 10010}, '{"code":10010,"msg":"任务提交队列已满"}'),
            _Response(200, {"data": {"jobId": "job-1"}}),
        )
    )
    uploads: list[bytes] = []
    sleeps: list[float] = []
    client = PaddleVLClient(queue_retry_delays_s=(2.0,), sleep=sleeps.append)

    def fake_request(_self, _method: str, _url: str, **kwargs):
        uploads.append(kwargs["files"]["file"][1].read())
        return next(responses)

    monkeypatch.setattr(PaddleVLClient, "_request", fake_request)

    body = client._submit_with_queue_retry(b"complete-image", filename="page.png", data={})

    assert body == {"data": {"jobId": "job-1"}}
    assert uploads == [b"complete-image", b"complete-image"]
    assert sleeps == [2.0]


def test_submit_does_not_retry_other_bad_requests(monkeypatch) -> None:
    calls = 0
    client = PaddleVLClient(queue_retry_delays_s=(0.0, 0.0), sleep=lambda _delay: None)

    def fake_request(_self, _method: str, _url: str, **_kwargs):
        nonlocal calls
        calls += 1
        return _Response(400, {"code": 10008}, '{"code":10008,"msg":"请求参数错误"}')

    monkeypatch.setattr(PaddleVLClient, "_request", fake_request)

    with pytest.raises(PaddleVLClientError, match="HTTP 400"):
        client._submit_with_queue_retry(b"image", filename="page.png", data={})

    assert calls == 1


def test_submit_reports_queue_full_after_bounded_retries(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []
    client = PaddleVLClient(queue_retry_delays_s=(1.0, 3.0), sleep=sleeps.append)

    def fake_request(_self, _method: str, _url: str, **_kwargs):
        nonlocal calls
        calls += 1
        return _Response(400, {"code": "10010"}, "queue full")

    monkeypatch.setattr(PaddleVLClient, "_request", fake_request)

    with pytest.raises(PaddleVLQueueFullError, match="10010"):
        client._submit_with_queue_retry(b"image", filename="page.png", data={})

    assert calls == 3
    assert sleeps == [1.0, 3.0]
