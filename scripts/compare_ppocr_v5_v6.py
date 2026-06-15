#!/usr/bin/env python3
"""Compare PP-OCRv5 sync `/ocr` output with PP-OCRv6 async jobs output.

The main app still uses PP-OCRv5 for proof line geometry.  This script is a
controlled probe for the new PP-OCRv6 API: it keeps the protocol differences
visible and writes raw payloads plus a compact shape summary for review.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

try:
    from app.core.api_profiles import (
        PADDLE_COORD_STABILITY_FLAGS,
        PADDLE_OCR_TEXT_DET_PARAMS,
    )
except Exception:  # pragma: no cover - lets the script run outside repo root.
    PADDLE_COORD_STABILITY_FLAGS = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
    }
    PADDLE_OCR_TEXT_DET_PARAMS = {
        "textDetLimitSideLen": 1536,
        "textDetLimitType": "max",
        "textDetThresh": 0.3,
        "textDetBoxThresh": 0.6,
        "textDetUnclipRatio": 2.0,
        "textRecScoreThresh": 0.0,
    }


DEFAULT_V5_ENDPOINT = "https://n6z9feddjca4l7b5.aistudio-app.com/ocr"
DEFAULT_V6_JOBS_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_V6_MODEL = "PP-OCRv6"


def _file_type(path_or_url: str) -> int:
    suffix = Path(path_or_url.split("?", 1)[0]).suffix.lower()
    return 0 if suffix == ".pdf" else 1


def _read_b64(path: Path) -> str:
    with path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _v5_payload(file_path: Path, *, include_text_det_params: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "file": _read_b64(file_path),
        "fileType": _file_type(str(file_path)),
    }
    payload.update(PADDLE_COORD_STABILITY_FLAGS)
    if include_text_det_params:
        payload.update(PADDLE_OCR_TEXT_DET_PARAMS)
    return payload


def run_v5(
    file_path: Path,
    *,
    endpoint: str,
    token: str,
    timeout: int,
    include_text_det_params: bool,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        endpoint,
        json=_v5_payload(file_path, include_text_det_params=include_text_det_params),
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def submit_v6(
    file_arg: str,
    *,
    jobs_url: str,
    token: str,
    model: str,
    timeout: int,
) -> str:
    headers = {"Authorization": f"bearer {token}"}
    optional_payload = dict(PADDLE_COORD_STABILITY_FLAGS)
    if file_arg.startswith(("http://", "https://")):
        headers["Content-Type"] = "application/json"
        payload = {
            "fileUrl": file_arg,
            "model": model,
            "optionalPayload": optional_payload,
        }
        resp = requests.post(jobs_url, json=payload, headers=headers, timeout=timeout)
    else:
        file_path = Path(file_arg)
        data = {
            "model": model,
            "optionalPayload": json.dumps(optional_payload, ensure_ascii=False),
        }
        with file_path.open("rb") as fh:
            files = {"file": (file_path.name, fh)}
            resp = requests.post(jobs_url, data=data, files=files, headers=headers, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    try:
        job_id = body["data"]["jobId"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"PP-OCRv6 submit response missing jobId: {body!r}") from exc
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError(f"PP-OCRv6 submit response has invalid jobId: {body!r}")
    return job_id


def wait_v6_jsonl_url(
    job_id: str,
    *,
    jobs_url: str,
    token: str,
    request_timeout: int,
    poll_timeout: int,
    poll_interval: float,
) -> tuple[str, dict[str, Any]]:
    headers = {"Authorization": f"bearer {token}"}
    deadline = time.monotonic() + max(1, poll_timeout)
    poll_url = f"{jobs_url.rstrip('/')}/{job_id}"
    last_body: dict[str, Any] = {}
    while time.monotonic() <= deadline:
        resp = requests.get(poll_url, headers=headers, timeout=request_timeout)
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
            raise RuntimeError(f"PP-OCRv6 job done without jsonUrl: {body!r}")
        if state in {"failed", "canceled", "cancelled"}:
            error_msg = data.get("errorMsg", "") if isinstance(data, dict) else ""
            raise RuntimeError(f"PP-OCRv6 job {state}: {error_msg}")
        time.sleep(poll_interval)
    raise TimeoutError(f"PP-OCRv6 polling timed out: {job_id}, last={last_body!r}")


def download_text(url: str, *, timeout: int) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_jsonl(jsonl_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _ocr_results_from_v5(raw: dict[str, Any]) -> list[dict[str, Any]]:
    result = raw.get("result", {}) if isinstance(raw, dict) else {}
    values = result.get("ocrResults", []) if isinstance(result, dict) else []
    return [value for value in values if isinstance(value, dict)]


def _ocr_results_from_v6(raw_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for page in raw_pages:
        result = page.get("result", page)
        if not isinstance(result, dict):
            continue
        page_values = result.get("ocrResults", [])
        if isinstance(page_values, list):
            values.extend(value for value in page_values if isinstance(value, dict))
    return values


def _pruned(result: dict[str, Any]) -> Any:
    return result.get("prunedResult", result)


def _text_preview(pruned: Any, limit: int = 180) -> str:
    if isinstance(pruned, dict):
        for key in ("rec_texts", "recTexts", "texts", "textLines", "rec_text"):
            value = pruned.get(key)
            if isinstance(value, list):
                text = " ".join(str(item) for item in value if item is not None)
                return text[:limit]
            if isinstance(value, str):
                return value[:limit]
    if isinstance(pruned, str):
        return pruned[:limit]
    return ""


def _line_count(pruned: Any) -> int:
    if not isinstance(pruned, dict):
        return 0
    candidates = []
    for key in (
        "rec_texts",
        "recTexts",
        "texts",
        "rec_polys",
        "dt_polys",
        "rec_boxes",
        "textlineOrientationAngles",
    ):
        value = pruned.get(key)
        if isinstance(value, list):
            candidates.append(len(value))
    return max(candidates, default=0)


def _field_shape(pruned: Any) -> dict[str, str]:
    if not isinstance(pruned, dict):
        return {"$type": type(pruned).__name__}
    shape: dict[str, str] = {}
    for key, value in sorted(pruned.items()):
        if isinstance(value, list):
            item_type = type(value[0]).__name__ if value else "empty"
            shape[key] = f"list[{len(value)}:{item_type}]"
        elif isinstance(value, dict):
            shape[key] = f"dict[{len(value)}]"
        else:
            shape[key] = type(value).__name__
    return shape


def summarize_ocr_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    pages = []
    field_union: set[str] = set()
    total_lines = 0
    for index, result in enumerate(results):
        pruned = _pruned(result)
        if isinstance(pruned, dict):
            field_union.update(str(key) for key in pruned.keys())
        line_count = _line_count(pruned)
        total_lines += line_count
        pages.append(
            {
                "index": index,
                "line_count": line_count,
                "fields": _field_shape(pruned),
                "text_preview": _text_preview(pruned),
                "has_ocr_image": isinstance(result.get("ocrImage"), str) and bool(result.get("ocrImage")),
            }
        )
    return {
        "page_count": len(results),
        "total_line_count": total_lines,
        "field_union": sorted(field_union),
        "pages": pages,
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PP-OCRv5 vs PP-OCRv6 Compare",
        "",
        "## Protocol",
        "",
        "- PP-OCRv5: sync POST `/ocr`, base64 file JSON, `Authorization: token ...`.",
        "- PP-OCRv6: async POST `/api/v2/ocr/jobs`, multipart or URL input, `Authorization: bearer ...`, then poll and download JSONL.",
        "",
        "## Sent Options",
        "",
        "```json",
        json.dumps(summary["sent_options"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    for key, title in (("v5", "PP-OCRv5"), ("v6", "PP-OCRv6")):
        result = summary.get(key)
        lines.extend([f"## {title}", ""])
        if not result:
            lines.extend(["未运行。", ""])
            continue
        lines.extend(
            [
                f"- pages: {result['page_count']}",
                f"- total_lines: {result['total_line_count']}",
                f"- fields: {', '.join(result['field_union']) or '(none)'}",
                "",
            ]
        )
        for page in result["pages"]:
            lines.extend(
                [
                    f"### Page {page['index']}",
                    "",
                    f"- line_count: {page['line_count']}",
                    f"- has_ocr_image: {page['has_ocr_image']}",
                    f"- text_preview: {page['text_preview']}",
                    "",
                    "```json",
                    json.dumps(page["fields"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PP-OCRv5 sync OCR with PP-OCRv6 async jobs OCR.")
    parser.add_argument("file", help="Local image/PDF path or HTTP(S) URL. v5 requires a local file.")
    parser.add_argument("--token", default=os.getenv("PADDLE_OCR_TOKEN", ""), help="Shared Paddle token; defaults to PADDLE_OCR_TOKEN.")
    parser.add_argument("--v5-token", default="", help="Override PP-OCRv5 token.")
    parser.add_argument("--v6-token", default="", help="Override PP-OCRv6 token.")
    parser.add_argument("--v5-endpoint", default=DEFAULT_V5_ENDPOINT)
    parser.add_argument("--v6-jobs-url", default=DEFAULT_V6_JOBS_URL)
    parser.add_argument("--v6-model", default=DEFAULT_V6_MODEL)
    parser.add_argument("--skip-v5", action="store_true")
    parser.add_argument("--skip-v6", action="store_true")
    parser.add_argument("--official-v5-options", action="store_true", help="Send only the three official stability flags to v5.")
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--poll-timeout", type=int, default=900)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--output-dir", default="debug/ppocr_v5_v6_compare")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    file_arg = args.file
    local_path = Path(file_arg) if not file_arg.startswith(("http://", "https://")) else None
    if local_path is not None and not local_path.exists():
        print(f"File not found: {local_path}", file=sys.stderr)
        return 2
    if not args.skip_v5 and local_path is None:
        print("PP-OCRv5 compare needs a local file. Use --skip-v5 for URL input.", file=sys.stderr)
        return 2

    v5_token = args.v5_token or args.token
    v6_token = args.v6_token or args.token
    if not args.skip_v5 and not v5_token:
        print("Missing v5 token. Set PADDLE_OCR_TOKEN or --v5-token.", file=sys.stderr)
        return 2
    if not args.skip_v6 and not v6_token:
        print("Missing v6 token. Set PADDLE_OCR_TOKEN or --v6-token.", file=sys.stderr)
        return 2

    stem = Path(file_arg.split("?", 1)[0]).stem or "url_input"
    run_dir = Path(args.output_dir) / f"{stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "input": file_arg,
        "sent_options": {
            "v5": dict(PADDLE_COORD_STABILITY_FLAGS)
            if args.official_v5_options
            else {**PADDLE_COORD_STABILITY_FLAGS, **PADDLE_OCR_TEXT_DET_PARAMS},
            "v6": dict(PADDLE_COORD_STABILITY_FLAGS),
        },
    }

    if not args.skip_v5 and local_path is not None:
        v5_raw = run_v5(
            local_path,
            endpoint=args.v5_endpoint,
            token=v5_token,
            timeout=args.request_timeout,
            include_text_det_params=not args.official_v5_options,
        )
        _write_json(run_dir / "ppocrv5.raw.json", v5_raw)
        summary["v5"] = summarize_ocr_results(_ocr_results_from_v5(v5_raw))

    if not args.skip_v6:
        job_id = submit_v6(
            file_arg,
            jobs_url=args.v6_jobs_url,
            token=v6_token,
            model=args.v6_model,
            timeout=args.request_timeout,
        )
        jsonl_url, job_data = wait_v6_jsonl_url(
            job_id,
            jobs_url=args.v6_jobs_url,
            token=v6_token,
            request_timeout=args.request_timeout,
            poll_timeout=args.poll_timeout,
            poll_interval=args.poll_interval,
        )
        jsonl_text = download_text(jsonl_url, timeout=args.request_timeout)
        raw_pages = parse_jsonl(jsonl_text)
        _write_text(run_dir / "ppocrv6.raw.jsonl", jsonl_text)
        _write_json(run_dir / "ppocrv6.raw-pages.json", raw_pages)
        _write_json(run_dir / "ppocrv6.job.json", {"jobId": job_id, "job": job_data, "jsonUrl": jsonl_url})
        summary["v6"] = summarize_ocr_results(_ocr_results_from_v6(raw_pages))

    _write_json(run_dir / "summary.json", summary)
    _write_text(run_dir / "summary.md", build_markdown(summary))
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
