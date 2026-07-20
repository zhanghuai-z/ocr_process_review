from __future__ import annotations

import requests

from app.integrations.paddle.vl_client import PaddleVLClient


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
