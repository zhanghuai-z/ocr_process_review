"""PaddleOCR-VL-1.6 official jobs API adapter."""
from __future__ import annotations

import json
import os
import time
from io import BytesIO
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import requests

from app.core.api_image_codec import encode_image_bytes_for_paddle

PADDLE_V16_MODEL = "PaddleOCR-VL-1.6"
PADDLE_V16_JOBS_PATH = "/api/v2/ocr/jobs"
PADDLE_V16_OFFICIAL_JOBS_URL = f"https://paddleocr.aistudio-app.com{PADDLE_V16_JOBS_PATH}"
PADDLE_V16_NETWORK_MODES = {"direct", "env_proxy", "auto"}
_DIRECT_PROXY_OVERRIDES = {"http": None, "https": None, "all": None}


def is_paddle_v16_endpoint(endpoint_url: str | None) -> bool:
    return (endpoint_url or "").strip().rstrip("/").endswith(PADDLE_V16_JOBS_PATH)


def build_paddle_v16_optional_payload(
    *,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_chart_recognition: bool = False,
) -> dict[str, object]:
    return {
        "useDocOrientationClassify": use_doc_orientation_classify,
        "useDocUnwarping": use_doc_unwarping,
        "useChartRecognition": use_chart_recognition,
    }


def normalize_paddle_v16_network_mode(value: str | None) -> str:
    mode = str(value or "direct").strip().lower()
    return mode if mode in PADDLE_V16_NETWORK_MODES else "direct"


def _has_env_proxy() -> bool:
    return any(
        os.environ.get(key)
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    )


def normalize_paddle_v16_jsonl_response(
    jsonl_text: str,
    *,
    job_id: str = "",
    job_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert the VL1.6 JSONL result into the legacy layout result envelope."""
    raw_pages: list[dict[str, Any]] = []
    layout_results: list[dict[str, Any]] = []
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            continue
        raw_pages.append(parsed)
        result = parsed.get("result", parsed)
        if not isinstance(result, dict):
            continue
        values = result.get("layoutParsingResults")
        if isinstance(values, list):
            layout_results.extend(value for value in values if isinstance(value, dict))

    return {
        "errorCode": 0,
        "errorMsg": "",
        "result": {
            "layoutParsingResults": layout_results,
        },
        "paddle_v16": {
            "jobId": job_id,
            "job": job_data or {},
            "rawJsonl": raw_pages,
        },
    }


@dataclass
class PaddleV16LayoutClient:
    jobs_url: str = PADDLE_V16_OFFICIAL_JOBS_URL
    token: str = ""
    request_timeout: int = 180
    poll_timeout: int = 600
    poll_interval_s: float = 5.0
    sleep: Callable[[float], None] = field(default=time.sleep)
    network_mode: str = "direct"
    telemetry: dict[str, Any] = field(default_factory=dict, init=False)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"bearer {self.token}"
        return headers

    def _network_attempts(self) -> list[bool]:
        mode = normalize_paddle_v16_network_mode(self.network_mode)
        if mode == "env_proxy":
            return [True]
        if mode == "auto" and _has_env_proxy():
            return [True, False]
        return [False]

    def _request_proxy_kwargs(self, use_env_proxy: bool) -> dict[str, Any]:
        if use_env_proxy:
            return {}
        return {"proxies": _DIRECT_PROXY_OVERRIDES}

    def _rewind_request_files(self, files: Any) -> None:
        if isinstance(files, dict):
            values = files.values()
        elif isinstance(files, (list, tuple)):
            values = [item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item for item in files]
        else:
            values = []
        for value in values:
            file_obj = value[1] if isinstance(value, (list, tuple)) and len(value) > 1 else value
            if hasattr(file_obj, "seek"):
                try:
                    file_obj.seek(0)
                except Exception:
                    pass

    def _request_with_network_fallback(self, method: str, url: str, *, label: str, **kwargs) -> requests.Response:
        attempts = self._network_attempts()
        retryable_errors = (
            requests.exceptions.ProxyError,
            requests.exceptions.ConnectionError,
            requests.exceptions.SSLError,
        )
        last_exc: BaseException | None = None
        for index, use_env_proxy in enumerate(attempts):
            try:
                request_kwargs = {**kwargs, **self._request_proxy_kwargs(use_env_proxy)}
                if method.upper() == "POST":
                    response = requests.post(url, **request_kwargs)
                elif method.upper() == "GET":
                    response = requests.get(url, **request_kwargs)
                else:
                    response = requests.request(method, url, **request_kwargs)
                self.telemetry[f"{label}_network_mode"] = "env_proxy" if use_env_proxy else "direct"
                return response
            except retryable_errors as exc:
                last_exc = exc
                if index >= len(attempts) - 1:
                    raise
                self._rewind_request_files(kwargs.get("files"))
                self.telemetry[f"{label}_fallback"] = "env_proxy_to_direct"
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"PaddleOCR-VL-1.6 request failed before sending: {method} {url}")

    def _raise_submit_error(self, resp) -> None:
        try:
            body = resp.text[:1000]
        except Exception:
            body = ""
        detail = f": {body}" if body else ""
        raise RuntimeError(f"PaddleOCR-VL-1.6 submit failed: HTTP {resp.status_code}{detail}")

    def submit_image(
        self,
        image_bgr: np.ndarray,
        *,
        model: str = PADDLE_V16_MODEL,
        optional_payload: dict[str, object] | None = None,
        batch_id: str = "",
    ) -> str:
        image_bytes = encode_image_bytes_for_paddle(image_bgr)
        if not image_bytes:
            raise RuntimeError("Cannot encode image for PaddleOCR-VL-1.6")
        return self.submit_image_bytes(
            image_bytes,
            model=model,
            optional_payload=optional_payload,
            batch_id=batch_id,
        )

    def submit_image_bytes(
        self,
        image_bytes: bytes,
        *,
        model: str = PADDLE_V16_MODEL,
        optional_payload: dict[str, object] | None = None,
        batch_id: str = "",
        filename: str = "page.png",
    ) -> str:
        started = time.perf_counter()
        payload = optional_payload or build_paddle_v16_optional_payload()
        data = {
            "model": model,
            "optionalPayload": json.dumps(payload, ensure_ascii=False),
        }
        if batch_id:
            data["batchId"] = batch_id
        file_obj = BytesIO(image_bytes)
        file_obj.name = filename
        files = {"file": file_obj}
        resp = self._request_with_network_fallback(
            "POST",
            self.jobs_url,
            label="submit",
            data=data,
            files=files,
            headers=self._headers(),
            timeout=self.request_timeout,
        )
        self.telemetry["submit_seconds"] = time.perf_counter() - started
        if getattr(resp, "status_code", 200) >= 400:
            self._raise_submit_error(resp)
        resp.raise_for_status()
        body = resp.json()
        try:
            job_id = body["data"]["jobId"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"PaddleOCR-VL-1.6 submit response missing jobId: {body!r}") from exc
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError(f"PaddleOCR-VL-1.6 submit response has invalid jobId: {body!r}")
        return job_id

    def wait_for_result_json_url(self, job_id: str) -> tuple[str, dict[str, Any]]:
        deadline = time.monotonic() + max(1, self.poll_timeout)
        poll_url = f"{self.jobs_url.rstrip('/')}/{job_id}"
        last_body: dict[str, Any] = {}
        started = time.perf_counter()
        poll_count = 0
        while time.monotonic() <= deadline:
            resp = self._request_with_network_fallback(
                "GET",
                poll_url,
                label="poll",
                headers=self._headers(),
                timeout=self.request_timeout,
            )
            poll_count += 1
            resp.raise_for_status()
            body = resp.json()
            if isinstance(body, dict):
                last_body = body
            data = body.get("data", {}) if isinstance(body, dict) else {}
            state = data.get("state") if isinstance(data, dict) else None
            if state == "done":
                result_url = data.get("resultUrl", {}) if isinstance(data, dict) else {}
                json_url = result_url.get("jsonUrl") if isinstance(result_url, dict) else None
                if isinstance(json_url, str) and json_url:
                    self.telemetry["wait_seconds"] = time.perf_counter() - started
                    self.telemetry["poll_count"] = poll_count
                    return json_url, data
                raise RuntimeError(f"PaddleOCR-VL-1.6 job done without jsonUrl: {body!r}")
            if state in {"failed", "canceled", "cancelled"}:
                error_msg = data.get("errorMsg", "") if isinstance(data, dict) else ""
                self.telemetry["wait_seconds"] = time.perf_counter() - started
                self.telemetry["poll_count"] = poll_count
                raise RuntimeError(f"PaddleOCR-VL-1.6 job {state}: {error_msg}")
            self.sleep(self.poll_interval_s)
        self.telemetry["wait_seconds"] = time.perf_counter() - started
        self.telemetry["poll_count"] = poll_count
        raise TimeoutError(f"PaddleOCR-VL-1.6 job polling timed out: {job_id}, last={last_body!r}")

    def download_jsonl(self, json_url: str) -> str:
        started = time.perf_counter()
        resp = self._request_with_network_fallback(
            "GET",
            json_url,
            label="download",
            headers={},
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        text = resp.text
        self.telemetry["download_seconds"] = time.perf_counter() - started
        self.telemetry["jsonl_bytes"] = len(text.encode("utf-8"))
        return text

    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        url = f"{self.jobs_url.rstrip('/')}/batch/{batch_id}"
        resp = self._request_with_network_fallback(
            "GET",
            url,
            label="batch_status",
            headers=self._headers(),
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"PaddleOCR-VL-1.6 batch status response is invalid: {body!r}")
        return body

    def analyze_image(
        self,
        image_bgr: np.ndarray,
        *,
        optional_payload: dict[str, object] | None = None,
        batch_id: str = "",
    ) -> dict[str, Any]:
        self.telemetry.clear()
        self.telemetry["network_mode"] = normalize_paddle_v16_network_mode(self.network_mode)
        if batch_id:
            self.telemetry["batch_id"] = batch_id
        total_started = time.perf_counter()
        job_id = self.submit_image(image_bgr, optional_payload=optional_payload, batch_id=batch_id)
        json_url, job_data = self.wait_for_result_json_url(job_id)
        jsonl_text = self.download_jsonl(json_url)
        self.telemetry["total_seconds"] = time.perf_counter() - total_started
        job_with_telemetry = dict(job_data)
        job_with_telemetry["clientTelemetry"] = dict(self.telemetry)
        return normalize_paddle_v16_jsonl_response(
            jsonl_text,
            job_id=job_id,
            job_data=job_with_telemetry,
        )

    def analyze_image_bytes(
        self,
        image_bytes: bytes,
        *,
        optional_payload: dict[str, object] | None = None,
        batch_id: str = "",
        filename: str = "page.png",
    ) -> dict[str, Any]:
        self.telemetry.clear()
        self.telemetry["network_mode"] = normalize_paddle_v16_network_mode(self.network_mode)
        if batch_id:
            self.telemetry["batch_id"] = batch_id
        total_started = time.perf_counter()
        job_id = self.submit_image_bytes(
            image_bytes,
            optional_payload=optional_payload,
            batch_id=batch_id,
            filename=filename,
        )
        json_url, job_data = self.wait_for_result_json_url(job_id)
        jsonl_text = self.download_jsonl(json_url)
        self.telemetry["total_seconds"] = time.perf_counter() - total_started
        job_with_telemetry = dict(job_data)
        job_with_telemetry["clientTelemetry"] = dict(self.telemetry)
        return normalize_paddle_v16_jsonl_response(
            jsonl_text,
            job_id=job_id,
            job_data=job_with_telemetry,
        )
