"""Benchmark PaddleOCR-VL jobs API timing for single or multi-page files.

Examples:
    PYTHONPATH=. python scripts/benchmark_paddle_v16_jobs.py /tmp/pages.pdf \
        --token-from-sample-script --poll-interval 1 --output /tmp/paddle_job.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_JOBS_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_MODEL = "PaddleOCR-VL-1.6"


def _token_from_sample_script(path: Path) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"""TOKEN\s*=\s*["']([^"']+)["']""", text)
    return match.group(1).strip() if match else ""


def _resolve_token(*, token: str, token_from_sample_script: bool) -> str:
    value = (token or os.environ.get("OCR_API_TOKEN") or os.environ.get("PADDLE_API_TOKEN") or "").strip()
    if value:
        return value
    try:
        from app.core.app_config import get_config

        value = str(get_config().get("api_token") or "").strip()
    except Exception:
        value = ""
    if value:
        return value
    if token_from_sample_script:
        return _token_from_sample_script(ROOT / "scripts" / "PaddleOCR-VL-1.6.sh")
    return ""


def _optional_payload() -> dict[str, bool]:
    return {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,
    }


def _submit_file(
    session: requests.Session,
    *,
    jobs_url: str,
    token: str,
    file_path: Path,
    model: str,
    timeout: int,
) -> tuple[str, dict[str, Any], float]:
    data = {
        "model": model,
        "optionalPayload": json.dumps(_optional_payload(), ensure_ascii=False),
    }
    headers = {"Authorization": f"bearer {token}"}
    started = time.perf_counter()
    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh)}
        resp = session.post(jobs_url, data=data, files=files, headers=headers, timeout=timeout)
    elapsed = time.perf_counter() - started
    if getattr(resp, "status_code", 200) >= 400:
        raise RuntimeError(f"submit failed: HTTP {resp.status_code}: {resp.text[:1000]}")
    resp.raise_for_status()
    body = resp.json()
    try:
        job_id = body["data"]["jobId"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"submit response missing jobId: {body!r}") from exc
    return str(job_id), body, elapsed


def _wait_done(
    session: requests.Session,
    *,
    jobs_url: str,
    token: str,
    job_id: str,
    request_timeout: int,
    poll_timeout: int,
    poll_interval: float,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], float]:
    headers = {"Authorization": f"bearer {token}"}
    poll_url = f"{jobs_url.rstrip('/')}/{job_id}"
    deadline = time.monotonic() + max(1, poll_timeout)
    started = time.perf_counter()
    polls: list[dict[str, Any]] = []
    last_body: dict[str, Any] = {}
    while time.monotonic() <= deadline:
        poll_started = time.perf_counter()
        resp = session.get(poll_url, headers=headers, timeout=request_timeout)
        poll_elapsed = time.perf_counter() - poll_started
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body, dict):
            last_body = body
        data = body.get("data", {}) if isinstance(body, dict) else {}
        state = data.get("state") if isinstance(data, dict) else None
        progress = data.get("extractProgress", {}) if isinstance(data, dict) else {}
        polls.append({
            "elapsed_since_wait_start": time.perf_counter() - started,
            "poll_seconds": poll_elapsed,
            "state": state,
            "totalPages": progress.get("totalPages") if isinstance(progress, dict) else None,
            "extractedPages": progress.get("extractedPages") if isinstance(progress, dict) else None,
        })
        if state == "done":
            result_url = data.get("resultUrl", {}) if isinstance(data, dict) else {}
            json_url = result_url.get("jsonUrl") if isinstance(result_url, dict) else None
            if isinstance(json_url, str) and json_url:
                return json_url, data, polls, time.perf_counter() - started
            raise RuntimeError(f"job done without jsonUrl: {body!r}")
        if state in {"failed", "canceled", "cancelled"}:
            error_msg = data.get("errorMsg", "") if isinstance(data, dict) else ""
            raise RuntimeError(f"job {state}: {error_msg}")
        time.sleep(max(0.1, poll_interval))
    raise TimeoutError(f"polling timed out: {job_id}, last={last_body!r}")


def _download_jsonl(
    session: requests.Session,
    *,
    json_url: str,
    timeout: int,
) -> tuple[str, float]:
    started = time.perf_counter()
    resp = session.get(json_url, timeout=timeout)
    resp.raise_for_status()
    return resp.text, time.perf_counter() - started


def _parse_jsonl(jsonl_text: str) -> tuple[int, int, float]:
    started = time.perf_counter()
    raw_pages = 0
    layout_results = 0
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        raw_pages += 1
        parsed = json.loads(line)
        result = parsed.get("result", parsed) if isinstance(parsed, dict) else {}
        values = result.get("layoutParsingResults") if isinstance(result, dict) else None
        if isinstance(values, list):
            layout_results += len(values)
    return raw_pages, layout_results, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PaddleOCR-VL jobs API timing.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--jobs-url", default=DEFAULT_JOBS_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token", default="")
    parser.add_argument("--token-from-sample-script", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--poll-timeout", type=int, default=1200)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    file_path = args.file
    if not file_path.is_file():
        raise SystemExit(f"file not found: {file_path}")
    token = _resolve_token(token=args.token, token_from_sample_script=bool(args.token_from_sample_script))
    if not token:
        raise SystemExit("Paddle token not found. Set OCR_API_TOKEN/PADDLE_API_TOKEN or pass --token-from-sample-script.")

    total_started = time.perf_counter()
    with requests.Session() as session:
        session.trust_env = False
        job_id, submit_body, submit_seconds = _submit_file(
            session,
            jobs_url=args.jobs_url,
            token=token,
            file_path=file_path,
            model=args.model,
            timeout=args.request_timeout,
        )
        json_url, job_data, polls, wait_seconds = _wait_done(
            session,
            jobs_url=args.jobs_url,
            token=token,
            job_id=job_id,
            request_timeout=args.request_timeout,
            poll_timeout=args.poll_timeout,
            poll_interval=args.poll_interval,
        )
        jsonl_text, download_seconds = _download_jsonl(
            session,
            json_url=json_url,
            timeout=args.request_timeout,
        )
    raw_pages, layout_results, parse_seconds = _parse_jsonl(jsonl_text)
    total_seconds = time.perf_counter() - total_started
    payload = {
        "mode": "paddle_v16_jobs_timing",
        "file": str(file_path),
        "file_size_bytes": file_path.stat().st_size,
        "jobs_url": args.jobs_url,
        "model": args.model,
        "job_id": job_id,
        "submit_seconds": submit_seconds,
        "wait_seconds": wait_seconds,
        "download_seconds": download_seconds,
        "parse_seconds": parse_seconds,
        "total_seconds": total_seconds,
        "raw_jsonl_pages": raw_pages,
        "layout_results": layout_results,
        "poll_count": len(polls),
        "polls": polls,
        "job_data": job_data,
        "submit_response_keys": list(submit_body.keys()) if isinstance(submit_body, dict) else [],
        "jsonl_bytes": len(jsonl_text.encode("utf-8")),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
