"""Paddle normalization -> Inspector IR seam.

This module is intentionally UI-free. It owns:
- raw Paddle response normalization
- bbox/polygon canonicalization
- OCR IR construction
- availability diagnostics for char/token boxes

The current implementation stays in Python, but the functions and dataclasses
are the boundary that can later be replaced by a native module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.api_profiles import API_MODEL_PROFILES, infer_api_model_profile_from_endpoint
from tools.ocr_inspector.models.ir import (
    BBox,
    BlockNode,
    CharNode,
    DocumentNode,
    LineNode,
    PageNode,
    Polygon,
    classify_ir_text,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoreDiagnostic:
    code: str
    severity: str
    layer: str
    message: str
    row_index: Optional[int] = None

    def to_log(self) -> str:
        prefix = f"{self.severity}: [{self.code}] {self.layer}"
        if self.row_index is not None:
            prefix += f" row={self.row_index}"
        return f"{prefix}: {self.message}"


@dataclass
class PaddleCoreResult:
    document: DocumentNode
    diagnostics: list[CoreDiagnostic] = field(default_factory=list)


def canonical_bbox(box: Any) -> Optional[BBox]:
    if box is None:
        return None
    try:
        if isinstance(box, dict):
            if {"x", "y", "w", "h"} <= set(box.keys()):
                return BBox(float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"]))
            for key in ("coordinate", "bbox", "box", "points", "polygon", "poly", "block_bbox"):
                if key in box:
                    return canonical_bbox(box[key])
        if isinstance(box, (list, tuple)):
            if len(box) == 1 and isinstance(box[0], (list, tuple)):
                return canonical_bbox(box[0])
            if len(box) >= 8 and all(isinstance(v, (int, float)) for v in box[:8]):
                pts = [(box[i], box[i + 1]) for i in range(0, 8, 2)]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return BBox.from_xyxy(min(xs), min(ys), max(xs), max(ys))
            if len(box) >= 4 and all(isinstance(v, (int, float)) for v in box[:4]):
                return BBox.from_list(box)
            if len(box) >= 4 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in box[:4]):
                xs = [p[0] for p in box[:4]]
                ys = [p[1] for p in box[:4]]
                return BBox.from_xyxy(min(xs), min(ys), max(xs), max(ys))
    except Exception as exc:
        logger.debug("bbox canonicalization failed: %s - %r", exc, box)
    return None


def canonical_polygon(raw: Any) -> Optional[Polygon]:
    return Polygon.from_raw(raw)


def raw_contains_word_regions(value: Any) -> bool:
    if isinstance(value, dict):
        for key in ("text_word_region", "textWordRegion", "text_word_boxes", "textWordBoxes"):
            if isinstance(value.get(key), list) and value.get(key):
                return True
        return any(raw_contains_word_regions(v) for v in value.values())
    if isinstance(value, list):
        return any(raw_contains_word_regions(v) for v in value)
    return False


def _diag(diagnostics: list[CoreDiagnostic], code: str, severity: str, layer: str, message: str, row_index: int | None = None) -> None:
    diagnostics.append(CoreDiagnostic(code=code, severity=severity, layer=layer, message=message, row_index=row_index))


def _pruned_result(item: dict) -> dict:
    pruned = item.get("prunedResult", {}) if isinstance(item, dict) else {}
    return pruned if isinstance(pruned, dict) else {}


def _overall_ocr_res(item: dict) -> dict:
    pruned = _pruned_result(item)
    ocr_res = pruned.get("overall_ocr_res")
    if isinstance(ocr_res, dict):
        return ocr_res
    direct = item.get("overall_ocr_res") if isinstance(item, dict) else None
    if isinstance(direct, dict):
        return direct
    if isinstance(item, dict) and any(key in item for key in ("rec_texts", "rec_boxes", "rec_polys", "dt_polys")):
        return item
    return {}


def _first_list_from_sources(sources: list[dict], keys: tuple[str, ...]) -> list:
    first_empty: list | None = None
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, list):
                if value:
                    return value
                if first_empty is None:
                    first_empty = value
    return first_empty or []


def _word_box_rows(item: dict) -> tuple[list, list]:
    pruned = _pruned_result(item)
    ocr_res = _overall_ocr_res(item)
    direct = item if isinstance(item, dict) else {}
    sources = [pruned, direct, ocr_res]
    text_words = _first_list_from_sources(sources, ("text_word", "textWord"))
    word_regions = _first_list_from_sources(sources, ("text_word_region", "textWordRegion", "text_word_boxes", "textWordBoxes"))
    return text_words, word_regions


def _iter_result_items(data: dict) -> list[dict]:
    items: list[dict] = []
    for key in ("layoutParsingResults", "ocrResults"):
        value = data.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    return items or [data]


def _detect_profile(raw: Any, data: dict) -> str | None:
    profile = raw.get("api_model_profile") if isinstance(raw, dict) else None
    if isinstance(profile, str) and profile in API_MODEL_PROFILES:
        return profile
    for key in ("api_url", "endpoint", "url"):
        value = raw.get(key) if isinstance(raw, dict) else None
        detected = infer_api_model_profile_from_endpoint(value)
        if detected:
            return detected
    if isinstance(data.get("ocrResults"), list):
        return "pp-ocrv5"
    if isinstance(data.get("layoutParsingResults"), list):
        return "pp-structurev3"
    return None


def _runtime_meta(raw: Any) -> dict:
    if isinstance(raw, dict):
        meta = raw.get("_inspector_meta")
        if isinstance(meta, dict):
            return meta
    return {}


def _request_return_word_box(meta: dict) -> bool | None:
    for key in ("request_summary", "api_request_summary"):
        summary = meta.get(key)
        if isinstance(summary, dict) and "returnWordBox" in summary:
            value = summary.get("returnWordBox")
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.lower()
                if lowered in ("true", "1", "yes"):
                    return True
                if lowered in ("false", "0", "no"):
                    return False
    return None


def _append_runtime_metadata(raw: Any, doc: DocumentNode) -> None:
    meta = _runtime_meta(raw)
    if not meta:
        return
    source = meta.get("source", "-")
    pipeline = meta.get("pipeline", "-")
    profile = meta.get("api_model_profile", "-")
    endpoint = meta.get("api_url", "")
    doc.parse_log.append(f"INFO: [runtime] source={source} pipeline={pipeline} profile={profile} endpoint={endpoint or '-'}")

    request = meta.get("request_summary") or meta.get("api_request_summary") or {}
    if isinstance(request, dict) and request:
        parts = ", ".join(f"{key}={request.get(key)!r}" for key in sorted(request) if key != "file")
        doc.parse_log.append(f"INFO: [request] {parts}")

    response_fields = meta.get("response_field_summary") or {}
    flattened_fields = meta.get("flattened_field_summary") or {}
    if isinstance(response_fields, dict):
        doc.parse_log.append(f"INFO: [response-fields] {response_fields}")
    if isinstance(flattened_fields, dict):
        doc.parse_log.append(f"INFO: [flattened-fields] {flattened_fields}")


def build_paddle_document(raw: Any, *, source_path: str = "", image_path: str = "") -> PaddleCoreResult:
    diagnostics: list[CoreDiagnostic] = []
    doc = DocumentNode.make(source_path=source_path, engine="paddle", raw=raw)
    _append_runtime_metadata(raw, doc)

    data = raw.get("result", raw) if isinstance(raw, dict) else raw
    if not isinstance(data, dict):
        _diag(diagnostics, "top_level_not_dict", "ERROR", "normalization", "top-level data is not a dict")
        _attach_diagnostics(doc, diagnostics)
        return PaddleCoreResult(document=doc, diagnostics=diagnostics)

    page = PageNode.make(page_number=1, image_path=image_path, raw=data)
    doc.pages.append(page)
    profile = _detect_profile(raw, data)
    if profile:
        spec = API_MODEL_PROFILES[profile]
        doc.parse_log.append(
            "INFO: profile="
            f"{spec['label']} endpoint={spec['endpoint_suffix']} "
            f"max_text_bbox_granularity={spec['max_text_bbox_granularity']}"
        )
        if spec.get("max_text_bbox_granularity") == "line":
            _diag(
                diagnostics,
                "unsupported_model_granularity",
                "INFO",
                "availability",
                f"{spec['label']} is treated as line/block granularity; char/token boxes are not promised",
            )

    items = _iter_result_items(data)
    _build_blocks(page, items, diagnostics)
    _build_lines(page, items, diagnostics)
    _summarize_availability(raw, doc, diagnostics)
    _attach_diagnostics(doc, diagnostics)
    return PaddleCoreResult(document=doc, diagnostics=diagnostics)


def _build_blocks(page: PageNode, items: list[dict], diagnostics: list[CoreDiagnostic]) -> None:
    order = 0
    for item in items:
        pruned = _pruned_result(item)
        for block_raw in pruned.get("parsing_res_list") or item.get("parsing_res_list") or []:
            if not isinstance(block_raw, dict):
                _diag(diagnostics, "invalid_block", "WARNING", "normalization", "parsing_res_list item is not a dict")
                continue
            label = str(block_raw.get("block_label", "text"))
            bbox = canonical_bbox(block_raw.get("block_bbox") or block_raw.get("block_polygon_points"))
            content = str(block_raw.get("block_content") or "")
            page.blocks.append(BlockNode.make(
                label=label,
                bbox=bbox,
                content=content,
                order=order,
                source_field="parsing_res_list",
                raw=block_raw,
            ))
            order += 1

    det_order = 0
    for item in items:
        pruned = _pruned_result(item)
        layout_det = pruned.get("layout_det_res") or item.get("layout_det_res") or {}
        if not isinstance(layout_det, dict):
            continue
        for det_box in layout_det.get("boxes") or []:
            if not isinstance(det_box, dict):
                _diag(diagnostics, "invalid_layout_det", "WARNING", "geometry", "layout_det_res.boxes item is not a dict")
                continue
            det_label = str(
                det_box.get("label")
                or det_box.get("type")
                or det_box.get("category")
                or det_box.get("category_name")
                or det_box.get("cls_name")
                or det_box.get("layout_label")
                or "unknown"
            )
            det_bbox = canonical_bbox(det_box)
            try:
                det_score = float(det_box.get("score") or det_box.get("confidence") or det_box.get("layout_score") or det_box.get("cls_score") or 0.0)
            except (TypeError, ValueError):
                det_score = 0.0
            page.layout_det_blocks.append(BlockNode.make(
                label=det_label,
                bbox=det_bbox,
                content=f"[layout_det score={det_score:.3f}]",
                order=det_order,
                source_field="layout_det_res",
                raw=det_box,
            ))
            det_order += 1

    page_count = len(page.blocks)
    det_count = len(page.layout_det_blocks)
    _diag(diagnostics, "blocks_built", "INFO", "OCR_IR", f"parsing_res_list -> {page_count} block(s)")
    _diag(diagnostics, "layout_det_built", "INFO", "OCR_IR", f"layout_det_res.boxes -> {det_count} det-box(es)")


def _build_lines(page: PageNode, items: list[dict], diagnostics: list[CoreDiagnostic]) -> None:
    global_row = 0
    for item in items:
        ocr_res = _overall_ocr_res(item)
        rec_texts = ocr_res.get("rec_texts") or []
        rec_boxes = ocr_res.get("rec_boxes") or []
        rec_polys = ocr_res.get("rec_polys") or ocr_res.get("rec_polygons") or ocr_res.get("dt_polys") or []
        rec_scores = ocr_res.get("rec_scores") or []
        text_words, word_regions = _word_box_rows(item)

        if not rec_texts:
            _diag(diagnostics, "missing_rec_texts", "WARNING", "normalization", "overall_ocr_res.rec_texts is empty or missing")

        for row_idx, text in enumerate(rec_texts):
            text = str(text)
            conf = float(rec_scores[row_idx]) if row_idx < len(rec_scores) else 1.0
            box = rec_boxes[row_idx] if row_idx < len(rec_boxes) else None
            poly = rec_polys[row_idx] if row_idx < len(rec_polys) else None
            bbox = canonical_bbox(box)
            polygon = canonical_polygon(poly)
            if bbox is None and polygon is not None:
                bbox = polygon.to_bbox()
                _diag(diagnostics, "line_bbox_from_polygon", "INFO", "geometry", "line bbox estimated from polygon", global_row)

            line = LineNode.make(
                text=text,
                confidence=conf,
                bbox=bbox,
                polygon=polygon,
                source_field="overall_ocr_res.rec_texts",
                raw={"rec_text": text, "rec_box": box, "rec_poly": poly, "rec_score": conf},
            )

            tok_row = text_words[row_idx] if row_idx < len(text_words) else None
            region_row = word_regions[row_idx] if row_idx < len(word_regions) else None
            line.chars = _build_chars(
                text=text,
                confidence=conf,
                token_row=tok_row,
                region_row=region_row,
                row_idx=global_row,
                diagnostics=diagnostics,
            )
            matched = _match_block(line, page.blocks)
            if matched is not None:
                matched.lines.append(line)
            else:
                page.orphan_lines.append(line)
            global_row += 1

    _diag(diagnostics, "lines_built", "INFO", "OCR_IR", f"overall_ocr_res -> {len(page.all_lines)} line(s), {len(page.all_chars)} char(s)")
    if page.orphan_lines:
        _diag(diagnostics, "orphan_lines", "WARNING", "OCR_IR", f"{len(page.orphan_lines)} orphan line(s) not matched to any block")


def _build_chars(*, text: str, confidence: float, token_row: Any, region_row: Any, row_idx: int, diagnostics: list[CoreDiagnostic]) -> list[CharNode]:
    chars = [
        CharNode.make(
            char=glyph,
            bbox=None,
            confidence=confidence,
            kind=classify_ir_text(glyph),
            bbox_source="unavailable",
            bbox_granularity="unavailable",
            token_text=glyph,
            collection_kind="line",
        )
        for glyph in text
    ]
    if not isinstance(token_row, (list, tuple)) or not isinstance(region_row, (list, tuple)):
        _diag(diagnostics, "missing_text_word_region", "INFO", "availability", "row has no token/word regions; chars keep unavailable bbox", row_idx)
        return chars

    tokens = [str(t) for t in token_row if t is not None]
    cursor = 0
    for tok_idx, (tok_text, raw_region) in enumerate(zip(tokens, region_row)):
        tok_text = tok_text.strip()
        if not tok_text:
            continue
        bbox = canonical_bbox(raw_region)
        if bbox is None:
            _diag(diagnostics, "invalid_region_format", "WARNING", "geometry", f"token {tok_idx} region is unparsable", row_idx)
            continue
        poly = canonical_polygon(raw_region)
        start = text.find(tok_text, cursor)
        if start < 0:
            start = text.find(tok_text, max(0, cursor - 1))
        if start < 0:
            _diag(diagnostics, "token_text_not_found", "WARNING", "OCR_IR", f"token {tok_idx} text {tok_text!r} not found in row text", row_idx)
            continue
        end = min(len(text), start + len(tok_text))
        granularity = "char" if len(tok_text) == 1 else "word"
        kind = classify_ir_text(tok_text)
        for ci in range(start, end):
            chars[ci] = CharNode.make(
                char=text[ci],
                bbox=bbox,
                polygon=poly,
                confidence=confidence,
                kind=kind,
                bbox_source="ocr",
                bbox_granularity=granularity,
                token_text=tok_text,
                collection_kind="char" if len(tok_text) == 1 else "token",
                raw={"token_text": tok_text, "region": raw_region},
            )
        cursor = end
    return chars


def _summarize_availability(raw: Any, doc: DocumentNode, diagnostics: list[CoreDiagnostic]) -> None:
    raw_has_regions = raw_contains_word_regions(raw)
    meta = _runtime_meta(raw)
    return_word_box = _request_return_word_box(meta)
    ocr_chars = sum(1 for page in doc.pages for char in page.all_chars if char.bbox_source == "ocr" and char.bbox is not None)
    unavailable = sum(1 for page in doc.pages for char in page.all_chars if char.bbox_source == "unavailable")
    if raw_has_regions and ocr_chars:
        _diag(diagnostics, "word_regions_preserved", "INFO", "availability", f"text_word_region/text_word_boxes preserved into {ocr_chars} OCR char/token node(s)")
    elif raw_has_regions and not ocr_chars:
        _diag(diagnostics, "word_regions_not_consumed", "ERROR", "availability", "raw response has text_word_region/text_word_boxes but no OCR char/token nodes were produced")
    elif unavailable:
        if return_word_box is True:
            _diag(
                diagnostics,
                "server_missing_text_word_region",
                "WARNING",
                "availability",
                "returnWordBox=true was sent, but response has no text_word_region/text_word_boxes; failure is at Paddle/server response layer",
            )
        elif return_word_box is False:
            _diag(
                diagnostics,
                "request_word_box_disabled",
                "INFO",
                "availability",
                "returnWordBox=false for this run; char/token boxes are unavailable because the request did not ask for them",
            )
        else:
            _diag(diagnostics, "server_missing_text_word_region", "INFO", "availability", "response has no text_word_region/text_word_boxes; char/token boxes are unavailable")


def _attach_diagnostics(doc: DocumentNode, diagnostics: list[CoreDiagnostic]) -> None:
    for diagnostic in diagnostics:
        doc.parse_log.append(diagnostic.to_log())
    if doc.pages:
        page = doc.pages[0]
        doc.parse_log.append(f"INFO: parse complete - coord space: image pixel space ({page.width}x{page.height})")


def _match_block(line: LineNode, blocks: list[BlockNode]) -> Optional[BlockNode]:
    if not line.bbox or not blocks:
        return None
    lx1, ly1, lx2, ly2 = line.bbox.x, line.bbox.y, line.bbox.x2, line.bbox.y2
    best_block, best_area = None, 0.0
    for block in blocks:
        if block.bbox is None:
            continue
        bx1, by1, bx2, by2 = block.bbox.x, block.bbox.y, block.bbox.x2, block.bbox.y2
        area = max(0.0, min(lx2, bx2) - max(lx1, bx1)) * max(0.0, min(ly2, by2) - max(ly1, by1))
        if area > best_area:
            best_area = area
            best_block = block
    return best_block if best_area > 0 else None
