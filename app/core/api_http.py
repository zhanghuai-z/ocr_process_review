"""HTTP helpers for Paddle API calls."""

from __future__ import annotations

from typing import Any

import requests


def post_json_without_env_proxy(
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
) -> requests.Response:
    """POST JSON without inheriting proxy variables from the host process."""
    proxy_overrides = {"http": None, "https": None, "all": None}
    return requests.post(
        url,
        json=json,
        headers=headers,
        timeout=timeout,
        proxies=proxy_overrides,
    )
