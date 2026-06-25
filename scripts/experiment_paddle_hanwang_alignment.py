#!/usr/bin/env python3
"""Offline Paddle-to-Hanwang alignment experiment.

This script uses checked-in fixtures only. It does not call Paddle or Hanwang.
It verifies whether Paddle layout geometry/content can be aligned with the
Hanwang text-slice routing that the application already uses.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.layout_analyzer import LayoutAnalyzer
from app.core.paddle_line_routing import (
    LAYOUT_LINE_ROUTES_FIELD,
    PaddleRouteLineHint,
    _formula_spans,
    _infer_formula_spans_by_line,
    attach_page_ocr_line_routes,
    block_text,
    is_formula_label,
    is_formula_style_position_block,
    is_table_label,
    route_subblocks_for_block,
    route_authority_label,
    text_slice_routes_for_block,
    vertical_overlap_ratio,
)
from app.models import BBox, Line, Page


XYXY = tuple[int, int, int, int]


@dataclass(frozen=True)
class FormulaCandidate:
    text: str
    label: str
    bbox: XYXY
    line_index: int
    segment_index: int
    source: str


@dataclass(frozen=True)
class ManualProbe:
    name: str
    bbox: XYXY
    expected: str


@dataclass(frozen=True)
class RecordScan:
    fixture: str
    record_index: int
    label: str
    bbox: XYXY
    parent_formula_spans: int
    geometry_formula_boxes: int
    route_formula_segments: int
    text_slices: int
    parent_span_geometry_gap: int
    unresolved_formula_spans: int
    detected_probe_results: Counter[str]
    unresolved_probe_results: Counter[str]


def _repo_root() -> Path:
    return REPO_ROOT


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _xyxy(value: Any) -> XYXY:
    x1, y1, x2, y2 = value
    return int(x1), int(y1), int(x2), int(y2)


def _area(box: XYXY) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersect(a: XYXY, b: XYXY) -> XYXY | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _iou(a: XYXY, b: XYXY) -> float:
    overlap = _intersect(a, b)
    if overlap is None:
        return 0.0
    inter_area = _area(overlap)
    union_area = _area(a) + _area(b) - inter_area
    return inter_area / max(1, union_area)


def _coverage(target: XYXY, candidate: XYXY) -> float:
    """Return how much of candidate is covered by target."""
    overlap = _intersect(target, candidate)
    if overlap is None:
        return 0.0
    return _area(overlap) / max(1, _area(candidate))


def _center_inside(inner: XYXY, outer: XYXY) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _expand_box(box: XYXY, pad: int, width: int, height: int) -> XYXY:
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(width, box[2] + pad),
        min(height, box[3] + pad),
    )


def _match_score(manual: XYXY, candidate: XYXY) -> float:
    coverage = _coverage(manual, candidate)
    return (
        coverage * 0.55
        + _iou(manual, candidate) * 0.25
        + vertical_overlap_ratio(manual, candidate) * 0.15
        + (0.05 if _center_inside(candidate, manual) else 0.0)
    )


def _nearest_line_index(box: XYXY, lines: list[Line]) -> int:
    best_index = 0
    best_score = float("-inf")
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    for index, line in enumerate(lines):
        line_box = line.bbox.to_xyxy()
        line_cx = (line_box[0] + line_box[2]) / 2
        line_cy = (line_box[1] + line_box[3]) / 2
        center_inside = line_box[1] <= cy <= line_box[3]
        horizontal_distance = 0.0 if line_box[0] <= cx <= line_box[2] else min(abs(cx - line_box[0]), abs(cx - line_box[2]))
        score = (
            vertical_overlap_ratio(box, line_box) * 1000.0
            + (100.0 if center_inside else 0.0)
            - abs(cy - line_cy)
            - horizontal_distance * 0.05
        )
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _load_fixture_page(layout_fixture: Path, ocr_lines_fixture: Path) -> tuple[Page, list[Line]]:
    raw = _load_json(layout_fixture)
    page = _page_from_layout_payload(raw)

    ocr_raw = _load_json(ocr_lines_fixture)
    page_ocr_lines = [
        Line(
            text=str(row["text"]),
            confidence=float(row["score"]),
            bbox=BBox.from_xyxy(*row["bbox"]),
        )
        for row in ocr_raw["lines"]
    ]
    attach_page_ocr_line_routes(
        page.ppvl_parsing_res_list,
        page_ocr_lines,
        page.width,
        page.height,
    )
    return page, page_ocr_lines


def _page_from_layout_payload(raw: dict[str, Any]) -> Page:
    page_info = raw["page"]
    image_path_value = str(page_info["display_image_path"])
    image_path = Path(image_path_value)
    if not image_path.is_absolute():
        image_path = _repo_root() / image_path
    page = Page(
        image_path=str(image_path),
        width=int(page_info["width"]),
        height=int(page_info["height"]),
        page_number=int(page_info["page_number"]),
    )
    page.source_path = str(page_info.get("source_path") or "")
    LayoutAnalyzer()._extract_api_blocks(page, raw["response"])
    return page


def _load_layout_cache_page(layout_cache: Path, ocr_lines_fixture: Path | None = None) -> tuple[Page, list[Line]]:
    raw = _load_json(layout_cache)
    page = _page_from_layout_payload(raw)
    page_ocr_lines: list[Line] = []
    if ocr_lines_fixture is not None and ocr_lines_fixture.exists():
        ocr_raw = _load_json(ocr_lines_fixture)
        page_ocr_lines = [
            Line(
                text=str(row["text"]),
                confidence=float(row["score"]),
                bbox=BBox.from_xyxy(*row["bbox"]),
            )
            for row in ocr_raw.get("lines", [])
        ]
        attach_page_ocr_line_routes(
            page.ppvl_parsing_res_list,
            page_ocr_lines,
            page.width,
            page.height,
        )
    return page, page_ocr_lines


def _layout_cache_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".layout-api.json")


def _matching_ocr_lines_fixture(layout_fixture: Path) -> Path:
    if layout_fixture.name.endswith("-layout-api-fixture.json"):
        return layout_fixture.with_name(layout_fixture.name.replace("-layout-api-fixture.json", "-page-ocr-lines.json"))
    return layout_fixture.with_name(layout_fixture.stem + "-page-ocr-lines.json")


def _main_text_record(page: Page) -> dict[str, Any]:
    for record in page.ppvl_parsing_res_list:
        if record.get("block_label") == "text" and "$ Y_{ct} $" in str(record.get("block_content") or ""):
            return record
    raise RuntimeError("Could not find 120166 main formula text block")


def _formula_candidates(record: dict[str, Any]) -> list[FormulaCandidate]:
    candidates: list[FormulaCandidate] = []
    for line_index, route in enumerate(record.get(LAYOUT_LINE_ROUTES_FIELD, [])):
        for segment_index, segment in enumerate(route.get("segments", [])):
            if segment.get("kind") != "formula":
                continue
            candidates.append(
                FormulaCandidate(
                    text=str(segment.get("text") or ""),
                    label=str(segment.get("label") or ""),
                    bbox=_xyxy(segment["bbox"]),
                    line_index=line_index,
                    segment_index=segment_index,
                    source="paddle_geometry+parent_text",
                )
            )
    return candidates


def _record_index(page: Page, record: dict[str, Any]) -> int:
    for index, item in enumerate(page.ppvl_parsing_res_list):
        if item is record:
            return index
    return -1


def _is_parent_formula_record(record: dict[str, Any]) -> bool:
    label = route_authority_label(record)
    return is_formula_label(label) or is_formula_style_position_block(record)


def _is_parent_table_record(record: dict[str, Any]) -> bool:
    return is_table_label(route_authority_label(record))


def _unresolved_spans_by_line(
    record: dict[str, Any],
    page_ocr_lines: list[Line],
    formula_candidates: list[FormulaCandidate],
    page_width: int,
    page_height: int,
) -> dict[int, list[str]]:
    line_hints = [
        PaddleRouteLineHint(text=line.text, bbox=line.bbox.to_xyxy())
        for line in page_ocr_lines
    ]
    formula_box_counts_by_line: dict[int, int] = defaultdict(int)
    for subblock in route_subblocks_for_block(record, page_width, page_height):
        if "formula" not in subblock["label"]:
            continue
        formula_box_counts_by_line[_nearest_line_index(subblock["bbox"], page_ocr_lines)] += 1

    inferred = _infer_formula_spans_by_line(
        block_text(record),
        line_hints,
        dict(formula_box_counts_by_line),
    )
    resolved_counts_by_line: dict[int, Counter[str]] = defaultdict(Counter)
    for candidate in formula_candidates:
        if candidate.text:
            resolved_counts_by_line[candidate.line_index][candidate.text] += 1

    unresolved: dict[int, list[str]] = {}
    for line_index, spans in inferred.items():
        remaining: list[str] = []
        for span in spans:
            if resolved_counts_by_line[line_index][span] > 0:
                resolved_counts_by_line[line_index][span] -= 1
            else:
                remaining.append(span)
        if remaining:
            unresolved[line_index] = remaining
    return unresolved


def _manual_probe_for_line(
    line: Line,
    *,
    width: int,
    height: int,
) -> XYXY:
    line_box = line.bbox.to_xyxy()
    probe_width = max(80, min(360, (line_box[2] - line_box[0]) // 5))
    return _expand_box(
        (
            line_box[0],
            line_box[1],
            min(line_box[2], line_box[0] + probe_width),
            line_box[3],
        ),
        6,
        width,
        height,
    )


def _classify_probe(
    probe: ManualProbe,
    candidates: list[FormulaCandidate],
    unresolved: dict[int, list[str]],
    page_ocr_lines: list[Line],
) -> tuple[str, list[str]]:
    scored = [
        (candidate, _match_score(probe.bbox, candidate.bbox))
        for candidate in candidates
    ]
    hits = [
        (candidate, score)
        for candidate, score in sorted(scored, key=lambda item: item[1], reverse=True)
        if score >= 0.35 and _coverage(probe.bbox, candidate.bbox) >= 0.35
    ]
    if len(hits) == 1:
        candidate, score = hits[0]
        return (
            "PADDLE_GEOMETRY_HIT",
            [
                f"匹配公式 {candidate.text}，line={candidate.line_index + 1}，segment={candidate.segment_index}，score={score:.3f}",
                "可直接使用 Paddle 父文本恢复的公式内容，并从 Hanwang text slice 中挖掉该区域。",
            ],
        )
    if len(hits) > 1:
        details = [
            f"{candidate.text or candidate.label}@line{candidate.line_index + 1}: score={score:.3f}"
            for candidate, score in hits[:4]
        ]
        return (
            "AMBIGUOUS_GEOMETRY_HIT",
            [
                "人工框覆盖多个 Paddle 公式候选，不能静默绑定为一个公式。",
                "候选: " + "; ".join(details),
            ],
        )

    line_index = _nearest_line_index(probe.bbox, page_ocr_lines)
    line_spans = unresolved.get(line_index, [])
    if len(line_spans) == 1:
        return (
            "PARENT_TEXT_LINE_INFERRED",
            [
                f"无 Paddle 几何小框，但同一物理行存在唯一未解析公式 {line_spans[0]}。",
                "可以把人工框绑定到父级 Paddle block_content 的公式文本；需要打 review flag，因为几何来自人工。",
            ],
        )
    if len(line_spans) > 1:
        return (
            "AMBIGUOUS_PARENT_TEXT",
            [
                "无 Paddle 几何小框，且同一物理行存在多个未解析公式。",
                "需要人工选择公式文本，或接入专门公式 OCR。",
            ],
        )
    return (
        "EMPTY_FORMULA_REVIEW_BOX",
        [
            "没有几何命中，也没有可唯一召回的父级公式真值。",
            "保持为空公式校验框；后续由人工补录或外部公式 OCR 填写。",
        ],
    )


def _records_with_formula_context(page: Page) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in page.ppvl_parsing_res_list:
        spans = _formula_spans(block_text(record))
        subblocks = route_subblocks_for_block(record, page.width, page.height)
        has_formula_geometry = any("formula" in subblock["label"] for subblock in subblocks)
        if spans or has_formula_geometry:
            records.append(record)
    return records


def _scan_record(
    *,
    fixture_name: str,
    page: Page,
    page_ocr_lines: list[Line],
    record: dict[str, Any],
) -> RecordScan:
    spans = _formula_spans(block_text(record))
    subblocks = route_subblocks_for_block(record, page.width, page.height)
    parent_formula_record = _is_parent_formula_record(record)
    parent_table_record = _is_parent_table_record(record)
    candidates = [] if parent_formula_record else _formula_candidates(record)
    parent_span_geometry_gap = (
        0
        if parent_formula_record or parent_table_record
        else max(0, len(spans) - len(candidates))
    )
    unresolved = (
        {}
        if parent_formula_record or parent_table_record or not page_ocr_lines
        else _unresolved_spans_by_line(record, page_ocr_lines, candidates, page.width, page.height)
    )
    text_slices = [] if parent_formula_record or parent_table_record else text_slice_routes_for_block(record, page.width, page.height)

    detected_probe_results: Counter[str] = Counter()
    if parent_table_record:
        detected_probe_results["PADDLE_PARENT_TABLE_HIT"] += 1
    elif parent_formula_record and spans:
        detected_probe_results["PADDLE_PARENT_FORMULA_HIT"] += 1
    else:
        for candidate in candidates:
            status, _details = _classify_probe(
                ManualProbe(
                    name=f"auto detected probe {candidate.text or candidate.label}",
                    bbox=_expand_box(candidate.bbox, 4, page.width, page.height),
                    expected="detected formula should bind to Paddle geometry",
                ),
                candidates,
                unresolved,
                page_ocr_lines,
            )
            detected_probe_results[status] += 1

    unresolved_probe_results: Counter[str] = Counter()
    for line_index, line_spans in unresolved.items():
        if line_index >= len(page_ocr_lines):
            continue
        status, _details = _classify_probe(
            ManualProbe(
                name=f"auto unresolved probe line {line_index + 1}",
                bbox=_manual_probe_for_line(
                    page_ocr_lines[line_index],
                    width=page.width,
                    height=page.height,
                ),
                expected="unresolved formula should recall parent truth if line is unique",
            ),
            candidates,
            unresolved,
            page_ocr_lines,
        )
        unresolved_probe_results[status] += len(line_spans)

    return RecordScan(
        fixture=fixture_name,
        record_index=_record_index(page, record),
        label=str(record.get("block_label") or record.get("label") or ""),
        bbox=_xyxy(record.get("block_bbox") or record.get("coordinate") or [0, 0, page.width, page.height]),
        parent_formula_spans=len(spans),
        geometry_formula_boxes=sum(1 for item in subblocks if "formula" in item["label"]),
        route_formula_segments=len(candidates),
        text_slices=len(text_slices),
        parent_span_geometry_gap=parent_span_geometry_gap,
        unresolved_formula_spans=sum(len(values) for values in unresolved.values()),
        detected_probe_results=detected_probe_results,
        unresolved_probe_results=unresolved_probe_results,
    )


def _scan_fixture_dir(fixture_dir: Path) -> list[RecordScan]:
    scans: list[RecordScan] = []
    for layout_fixture in sorted(fixture_dir.glob("*-layout-api-fixture.json")):
        ocr_fixture = _matching_ocr_lines_fixture(layout_fixture)
        if not ocr_fixture.exists():
            continue
        page, page_ocr_lines = _load_fixture_page(layout_fixture, ocr_fixture)
        for record in _records_with_formula_context(page):
            scans.append(
                _scan_record(
                    fixture_name=layout_fixture.name,
                    page=page,
                    page_ocr_lines=page_ocr_lines,
                    record=record,
                )
            )
    return scans


def _scan_layout_cache_dir(cache_dir: Path) -> list[RecordScan]:
    scans: list[RecordScan] = []
    for layout_cache in sorted(cache_dir.glob("*.layout-api.json")):
        page, page_ocr_lines = _load_layout_cache_page(layout_cache)
        for record in _records_with_formula_context(page):
            scans.append(
                _scan_record(
                    fixture_name=str(layout_cache.relative_to(_repo_root()) if layout_cache.is_relative_to(_repo_root()) else layout_cache),
                    page=page,
                    page_ocr_lines=page_ocr_lines,
                    record=record,
                )
            )
    return scans


def _token_from_sample_script(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"""TOKEN\s*=\s*["']([^"']+)["']""", text)
    return match.group(1).strip() if match else ""


def _resolve_paddle_token(*, token_from_sample_script: bool) -> str:
    token = (
        os.environ.get("OCR_API_TOKEN", "")
        or os.environ.get("PADDLE_API_TOKEN", "")
    ).strip()
    if token:
        return token
    try:
        from app.core.ocr_config import get_config

        token = str(get_config().get("api_token") or "").strip()
    except Exception:
        token = ""
    if token:
        return token
    if token_from_sample_script:
        return _token_from_sample_script(_repo_root() / "scripts/PaddleOCR-VL-1.6.sh")
    return ""


def _write_layout_cache_for_image(
    *,
    image_path: Path,
    cache_path: Path,
    token: str,
    request_timeout: int,
    poll_timeout: int,
) -> None:
    import cv2

    from app.core.paddle_v16_client import (
        PaddleV16LayoutClient,
        build_paddle_v16_optional_payload,
    )

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    height, width = image_bgr.shape[:2]
    client = PaddleV16LayoutClient(
        token=token,
        request_timeout=request_timeout,
        poll_timeout=poll_timeout,
    )
    data = client.analyze_image(
        image_bgr,
        optional_payload=build_paddle_v16_optional_payload(),
    )
    payload = {
        "page": {
            "display_image_path": str(image_path),
            "source_path": str(image_path),
            "width": width,
            "height": height,
            "page_number": int(image_path.stem) if image_path.stem.isdigit() else 1,
        },
        "response": data,
    }
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _generate_file_layout_caches(
    *,
    file_dir: Path,
    limit: int,
    force: bool,
    token_from_sample_script: bool,
    request_timeout: int,
    poll_timeout: int,
) -> None:
    token = _resolve_paddle_token(token_from_sample_script=token_from_sample_script)
    if not token:
        raise RuntimeError(
            "Paddle token not found. Set OCR_API_TOKEN/PADDLE_API_TOKEN, configure app API token, "
            "or pass --token-from-sample-script."
        )

    images = sorted(file_dir.glob("*.tif"))
    if limit > 0:
        images = images[:limit]
    total = len(images)
    for index, image_path in enumerate(images, 1):
        cache_path = _layout_cache_for_image(image_path)
        if cache_path.exists() and not force:
            print(f"[skip] {index}/{total} {image_path.name}: cache exists")
            continue
        print(f"[paddle] {index}/{total} {image_path.name}: generating {cache_path.name}")
        _write_layout_cache_for_image(
            image_path=image_path,
            cache_path=cache_path,
            token=token,
            request_timeout=request_timeout,
            poll_timeout=poll_timeout,
        )


def _print_scan_summary(scans: list[RecordScan]) -> None:
    print("## 扩大实验面汇总")
    if not scans:
        print("- 没有找到可扫描的 layout fixture。")
        print()
        return

    fixture_count = len({scan.fixture for scan in scans})
    detected_results: Counter[str] = Counter()
    unresolved_results: Counter[str] = Counter()
    for scan in scans:
        detected_results.update(scan.detected_probe_results)
        unresolved_results.update(scan.unresolved_probe_results)

    print(f"- layout fixture 数: {fixture_count}")
    print(f"- 含公式上下文的父级记录数: {len(scans)}")
    print(f"- 父级公式真值 span 总数: {sum(scan.parent_formula_spans for scan in scans)}")
    print(f"- Paddle 几何公式框总数: {sum(scan.geometry_formula_boxes for scan in scans)}")
    print(f"- route formula segment 总数: {sum(scan.route_formula_segments for scan in scans)}")
    print(f"- 父级公式真值与几何 segment 缺口总数: {sum(scan.parent_span_geometry_gap for scan in scans)}")
    print(f"- 未被几何小框覆盖但可进入父级召回检查的 span 总数: {sum(scan.unresolved_formula_spans for scan in scans)}")
    print(f"- 已检公式自动 probe: {dict(detected_results)}")
    print(f"- 漏几何公式自动 probe: {dict(unresolved_results)}")
    print()
    print("### 逐记录扫描")
    for scan in scans:
        print(
            "- "
            f"{scan.fixture} record={scan.record_index} label={scan.label} bbox={list(scan.bbox)} "
            f"parent_spans={scan.parent_formula_spans} geometry_boxes={scan.geometry_formula_boxes} "
            f"route_segments={scan.route_formula_segments} unresolved={scan.unresolved_formula_spans} "
            f"gap={scan.parent_span_geometry_gap} "
            f"detected_probe={dict(scan.detected_probe_results)} unresolved_probe={dict(scan.unresolved_probe_results)}"
        )
    print()


def _print_report(page: Page, page_ocr_lines: list[Line], record: dict[str, Any]) -> None:
    parent_text = block_text(record)
    spans = _formula_spans(parent_text)
    routes = record.get(LAYOUT_LINE_ROUTES_FIELD, [])
    subblocks = route_subblocks_for_block(record, page.width, page.height)
    text_slices = text_slice_routes_for_block(record, page.width, page.height)
    candidates = _formula_candidates(record)
    unresolved = _unresolved_spans_by_line(record, page_ocr_lines, candidates, page.width, page.height)

    print("# Paddle/Hanwang 对齐离线实验")
    print()
    print(f"- fixture: page={Path(page.image_path).name}, size={page.width}x{page.height}")
    print(f"- 目标父级 text block: bbox={record.get('block_bbox')}")
    print(f"- 父级 Paddle block_content 公式 span 数: {len(spans)}")
    print(f"- Paddle 几何 inline_formula 小框数: {sum(1 for item in subblocks if 'formula' in item['label'])}")
    print(f"- Paddle OCR 行提示数: {len(page_ocr_lines)}")
    print(f"- Hanwang 将收到的 text slice 数: {len(text_slices)}")
    print()

    print("## 父级公式 span")
    for index, span in enumerate(spans, 1):
        print(f"{index}. {span}")
    print()

    print("## Paddle OCR 行提示")
    for index, line in enumerate(page_ocr_lines, 1):
        print(f"{index}. bbox={list(line.bbox.to_xyxy())}, score={line.confidence:.3f}, text={line.text}")
    print()

    print("## 当前 route 行")
    for line_index, route in enumerate(routes, 1):
        print(f"- line {line_index}: bbox={route.get('bbox')}")
        for segment in route.get("segments", []):
            kind = segment.get("kind")
            bbox = segment.get("bbox")
            text = segment.get("text") or ""
            if kind == "formula":
                print(f"  - formula bbox={bbox}, text={text}")
            else:
                print(f"  - text bbox={bbox} -> Hanwang")
    print()

    print("## 未被几何小框覆盖的父级公式")
    if unresolved:
        for line_index, line_spans in unresolved.items():
            print(f"- line {line_index + 1}: {', '.join(line_spans)}")
    else:
        print("- none")
    print()

    probes = [
        ManualProbe(
            name="loose box over detected Incentive x Post",
            bbox=(1030, 1888, 1165, 1960),
            expected="应命中 Paddle 几何小框",
        ),
        ManualProbe(
            name="manual box over missed standalone Incentive_c",
            bbox=(292, 1958, 620, 2038),
            expected="Paddle 漏几何框，但父级 block_content 应能反查唯一公式",
        ),
        ManualProbe(
            name="wide box over delta and varphi",
            bbox=(610, 2278, 790, 2348),
            expected="应判定为歧义，不应静默合并",
        ),
        ManualProbe(
            name="manual formula box with no Paddle parent truth",
            bbox=(1880, 2045, 2050, 2105),
            expected="无父级可召回公式真值时保留空公式校验框",
        ),
    ]

    print("## 人工框 probe")
    for probe in probes:
        status, details = _classify_probe(probe, candidates, unresolved, page_ocr_lines)
        print(f"- {probe.name}: bbox={list(probe.bbox)}")
        print(f"  - expected: {probe.expected}")
        print(f"  - result: {status}")
        for detail in details:
            print(f"  - {detail}")
    print()

    print("## 实验结论")
    print("- 对已被 Paddle 检出的行内公式，可以用几何 overlap/coverage 绑定到 route formula segment；Hanwang 只识别左右 text slice。")
    print("- 对 Paddle 漏几何框但父级 block_content 含公式的情况，可以做第二层 line-level 反查；120166 的 `$ Incentive_{c} $` 属于这种情况。")
    print("- 第二层反查必须带 review flag：几何来自人工，文本来自 Paddle 父级内容，不应伪装成 Paddle 已检框。")
    print("- 如果人工框覆盖多个公式候选，或同一行有多个未解析公式，必须进入人工选择/补录，不能自动合并。")
    print("- 如果父级真值无内容，则不再强行识别；保留空公式校验框等待人工/后续公式 OCR。")
    print("- 该 fixture 没有表格/图片块，因此本实验只能确认公式对齐；表格和图片需要补真实样例 fixture。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout-fixture",
        type=Path,
        default=_repo_root() / "tests/fixtures/layout/120166-layout-api-fixture.json",
    )
    parser.add_argument(
        "--ocr-lines-fixture",
        type=Path,
        default=_repo_root() / "tests/fixtures/layout/120166-page-ocr-lines.json",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=_repo_root() / "tests/fixtures/layout",
    )
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Only print the detailed target fixture report.",
    )
    parser.add_argument(
        "--file-dir",
        type=Path,
        help="Directory containing real .tif pages and optional .layout-api.json caches.",
    )
    parser.add_argument(
        "--generate-file-layouts",
        action="store_true",
        help="Call PaddleOCR-VL-1.6 for .tif pages under --file-dir and write .layout-api.json caches.",
    )
    parser.add_argument(
        "--file-limit",
        type=int,
        default=0,
        help="Limit real .tif pages processed under --file-dir; 0 means all.",
    )
    parser.add_argument(
        "--force-file-layouts",
        action="store_true",
        help="Regenerate existing .layout-api.json caches under --file-dir.",
    )
    parser.add_argument(
        "--token-from-sample-script",
        action="store_true",
        help="If no configured token is found, read TOKEN from scripts/PaddleOCR-VL-1.6.sh.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=180,
        help="Paddle request timeout in seconds for real file generation.",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=600,
        help="Paddle polling timeout in seconds for real file generation.",
    )
    args = parser.parse_args()

    if args.file_dir is not None:
        file_dir = args.file_dir
        if args.generate_file_layouts:
            _generate_file_layout_caches(
                file_dir=file_dir,
                limit=max(0, args.file_limit),
                force=bool(args.force_file_layouts),
                token_from_sample_script=bool(args.token_from_sample_script),
                request_timeout=max(1, args.request_timeout),
                poll_timeout=max(1, args.poll_timeout),
            )
        _print_scan_summary(_scan_layout_cache_dir(file_dir))
        return 0

    page, page_ocr_lines = _load_fixture_page(args.layout_fixture, args.ocr_lines_fixture)
    record = _main_text_record(page)
    _print_report(page, page_ocr_lines, record)
    print()
    if not args.no_scan:
        _print_scan_summary(_scan_fixture_dir(args.fixture_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
