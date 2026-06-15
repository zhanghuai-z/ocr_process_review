"""PaddleOCR-VL-1.6 official jobs API adapter."""
from __future__ import annotations

import json
import time
from io import BytesIO
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.core.api_http import (
    get_without_env_proxy,
    post_multipart_without_env_proxy,
)
from app.core.api_image_codec import encode_image_bytes_for_paddle

PADDLE_V16_MODEL = "PaddleOCR-VL-1.6"
PADDLE_V16_JOBS_PATH = "/api/v2/ocr/jobs"
PADDLE_V16_OFFICIAL_JOBS_URL = f"https://paddleocr.aistudio-app.com{PADDLE_V16_JOBS_PATH}"


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

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"bearer {self.token}"
        return headers

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
    ) -> str:
        image_bytes = encode_image_bytes_for_paddle(image_bgr)
        if not image_bytes:
            raise RuntimeError("Cannot encode image for PaddleOCR-VL-1.6")
        return self.submit_image_bytes(
            image_bytes,
            model=model,
            optional_payload=optional_payload,
        )

    def submit_image_bytes(
        self,
        image_bytes: bytes,
        *,
        model: str = PADDLE_V16_MODEL,
        optional_payload: dict[str, object] | None = None,
        filename: str = "page.png",
    ) -> str:
        payload = optional_payload or build_paddle_v16_optional_payload()
        data = {
            "model": model,
            "optionalPayload": json.dumps(payload, ensure_ascii=False),
        }
        file_obj = BytesIO(image_bytes)
        file_obj.name = filename
        files = {"file": file_obj}
        resp = post_multipart_without_env_proxy(
            self.jobs_url,
            data=data,
            files=files,
            headers=self._headers(),
            timeout=self.request_timeout,
        )
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
        while time.monotonic() <= deadline:
            resp = get_without_env_proxy(
                poll_url,
                headers=self._headers(),
                timeout=self.request_timeout,
            )
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
                    return json_url, data
                raise RuntimeError(f"PaddleOCR-VL-1.6 job done without jsonUrl: {body!r}")
            if state in {"failed", "canceled", "cancelled"}:
                error_msg = data.get("errorMsg", "") if isinstance(data, dict) else ""
                raise RuntimeError(f"PaddleOCR-VL-1.6 job {state}: {error_msg}")
            self.sleep(self.poll_interval_s)
        raise TimeoutError(f"PaddleOCR-VL-1.6 job polling timed out: {job_id}, last={last_body!r}")

    def download_jsonl(self, json_url: str) -> str:
        resp = get_without_env_proxy(
            json_url,
            headers={},
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        return resp.text

    def analyze_image(
        self,
        image_bgr: np.ndarray,
        *,
        optional_payload: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        job_id = self.submit_image(image_bgr, optional_payload=optional_payload)
        json_url, job_data = self.wait_for_result_json_url(job_id)
        return normalize_paddle_v16_jsonl_response(
            self.download_jsonl(json_url),
            job_id=job_id,
            job_data=job_data,
        )
