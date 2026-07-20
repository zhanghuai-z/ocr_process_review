"""HTTP client for the PaddleOCR-VL jobs API.

The client returns the vendor response envelope as mappings.  It does not
construct application records or know about project/session repositories.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
import time
from typing import Any

import requests


PADDLE_VL_JOBS_PATH = "/api/v2/ocr/jobs"
PADDLE_VL_JOBS_URL = f"https://paddleocr.aistudio-app.com{PADDLE_VL_JOBS_PATH}"
PADDLE_VL_MODEL = "PaddleOCR-VL-1.6"


class PaddleVLClientError(RuntimeError):
    """The vendor request failed before a usable response was returned."""


class PaddleVLRequestCancelled(PaddleVLClientError):
    """The caller cancelled an in-flight vendor request."""


def _success_envelope(jsonl_text: str, *, job_id: str, job_data: dict[str, Any]) -> dict[str, Any]:
    raw_lines: list[dict[str, Any]] = []
    layout_results: list[dict[str, Any]] = []
    error_code: object = 0
    error_message = "Success"
    for line_number, line in enumerate(jsonl_text.splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PaddleVLClientError(
                f"Paddle result JSONL line {line_number} is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise PaddleVLClientError(
                f"Paddle result JSONL line {line_number} must be an object"
            )
        raw_lines.append(value)
        if value.get("errorCode") not in (None, 0, "0"):
            error_code = value.get("errorCode")
            error_message = str(value.get("errorMsg") or error_code)
        elif value.get("errorMsg") not in (None, "", "Success"):
            error_message = str(value["errorMsg"])
        result = value.get("result", value)
        if not isinstance(result, dict):
            continue
        values = result.get("layoutParsingResults")
        if isinstance(values, list):
            layout_results.extend(item for item in values if isinstance(item, dict))

    telemetry = dict(job_data)
    telemetry["jobId"] = job_id
    return {
        "errorCode": error_code,
        "errorMsg": error_message,
        "result": {"layoutParsingResults": layout_results},
        "paddle_vl": {"jobId": job_id, "job": telemetry, "rawJsonl": raw_lines},
    }


@dataclass(slots=True)
class PaddleVLClient:
    """Call PaddleOCR-VL-1.6 and return one vendor response mapping."""

    jobs_url: str = PADDLE_VL_JOBS_URL
    token: str = ""
    request_timeout: float = 30.0
    poll_timeout: float = 600.0
    poll_interval_s: float = 5.0
    sleep: Any = field(default=time.sleep, repr=False)
    cancel_callback: Any = field(default=None, repr=False)

    def _raise_if_cancelled(self) -> None:
        if self.cancel_callback is not None and self.cancel_callback():
            raise PaddleVLRequestCancelled("Paddle VL request cancelled")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"bearer {self.token}"} if self.token else {}

    @staticmethod
    def _raise_http(response: requests.Response, operation: str) -> None:
        if response.status_code >= 400:
            body = response.text[:1000]
            detail = f": {body}" if body else ""
            raise PaddleVLClientError(
                f"Paddle VL {operation} failed: HTTP {response.status_code}{detail}"
            )
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise PaddleVLClientError(f"Paddle VL {operation} failed") from exc

    def analyze_image_bytes(
        self,
        image_bytes: bytes,
        *,
        filename: str = "page.png",
        page_uid: str = "",
        source_run_id: str = "",
    ) -> dict[str, Any]:
        """Submit one page, wait for completion, and download its JSONL result."""
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise PaddleVLClientError("Paddle VL requires non-empty image bytes")
        self._raise_if_cancelled()
        optional_payload = {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }
        data: dict[str, str] = {
            "model": PADDLE_VL_MODEL,
            "optionalPayload": json.dumps(optional_payload, separators=(",", ":")),
        }
        if source_run_id:
            data["batchId"] = source_run_id
        file_obj = _NamedBytes(image_bytes, filename)
        try:
            response = requests.post(
                self.jobs_url,
                data=data,
                files={"file": (filename, file_obj, "application/octet-stream")},
                headers=self._headers(),
                timeout=self.request_timeout,
            )
            self._raise_http(response, "submit")
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise PaddleVLClientError("Paddle VL submit response is invalid") from exc
        job_id = self._job_id(body)
        json_url, job_data = self._wait_for_result(job_id)
        self._raise_if_cancelled()
        try:
            result_response = requests.get(
                json_url,
                headers={},
                timeout=self.request_timeout,
            )
            self._raise_http(result_response, "download")
            jsonl_text = result_response.text
        except requests.RequestException as exc:
            raise PaddleVLClientError("Paddle VL result download failed") from exc
        return _success_envelope(jsonl_text, job_id=job_id, job_data=job_data)

    @staticmethod
    def _job_id(body: object) -> str:
        if not isinstance(body, dict):
            raise PaddleVLClientError("Paddle VL submit response must be an object")
        data = body.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("jobId"), str) or not data["jobId"]:
            raise PaddleVLClientError("Paddle VL submit response is missing data.jobId")
        return data["jobId"]

    def _wait_for_result(self, job_id: str) -> tuple[str, dict[str, Any]]:
        deadline = time.monotonic() + max(1.0, float(self.poll_timeout))
        poll_url = f"{self.jobs_url.rstrip('/')}/{job_id}"
        last_body: dict[str, Any] = {}
        while time.monotonic() <= deadline:
            self._raise_if_cancelled()
            try:
                response = requests.get(
                    poll_url,
                    headers=self._headers(),
                    timeout=self.request_timeout,
                )
                self._raise_http(response, "poll")
                body = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise PaddleVLClientError("Paddle VL poll response is invalid") from exc
            if not isinstance(body, dict):
                raise PaddleVLClientError("Paddle VL poll response must be an object")
            last_body = body
            data = body.get("data")
            if not isinstance(data, dict):
                raise PaddleVLClientError("Paddle VL poll response is missing data")
            state = data.get("state")
            if state == "done":
                result_url = data.get("resultUrl")
                if not isinstance(result_url, dict) or not isinstance(result_url.get("jsonUrl"), str) or not result_url["jsonUrl"]:
                    raise PaddleVLClientError("Paddle VL completed job has no resultUrl.jsonUrl")
                return result_url["jsonUrl"], data
            if state in {"failed", "canceled", "cancelled"}:
                raise PaddleVLClientError(
                    f"Paddle VL job {state}: {data.get('errorMsg') or 'unknown error'}"
                )
            self.sleep(max(0.0, float(self.poll_interval_s)))
        raise TimeoutError(f"Paddle VL polling timed out: {job_id}; last={last_body!r}")


class _NamedBytes:
    """Small file-like wrapper used to give requests a stable filename."""

    def __init__(self, value: bytes, name: str) -> None:
        from io import BytesIO

        self._stream = BytesIO(value)
        self.name = name or "page.png"

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._stream.seek(offset, whence)

    def tell(self) -> int:
        return self._stream.tell()


__all__ = [
    "PADDLE_VL_JOBS_URL",
    "PaddleVLClient",
    "PaddleVLClientError",
    "PaddleVLRequestCancelled",
]
