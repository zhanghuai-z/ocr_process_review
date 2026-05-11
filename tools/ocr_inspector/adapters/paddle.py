"""PaddleOCR adapter: converts raw OCR JSON → IR DocumentNode."""
from __future__ import annotations
import logging
from typing import Any, List, Optional

from app.core.api_profiles import (
    API_MODEL_PROFILES,
    infer_api_model_profile_from_endpoint,
)
from tools.ocr_inspector.models.ir import (
    BBox, Polygon, CharNode, LineNode, BlockNode, PageNode, DocumentNode,
    classify_ir_text,
)

logger = logging.getLogger(__name__)


def _bbox_from_box(box: Any) -> Optional[BBox]:
    if box is None:
        return None
    try:
        if isinstance(box, dict):
            if {"x", "y", "w", "h"} <= set(box.keys()):
                return BBox(float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"]))
            for key in ("coordinate", "bbox", "box", "points", "polygon", "poly", "block_bbox"):
                if key in box:
                    return _bbox_from_box(box[key])
        if isinstance(box, (list, tuple)):
            if len(box) == 1 and isinstance(box[0], (list, tuple)):
                return _bbox_from_box(box[0])
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
        logger.debug("bbox parse failed: %s — %r", exc, box)
    return None


def _pruned_result(item: dict) -> dict:
    pruned = item.get("prunedResult", {}) if isinstance(item, dict) else {}
    return pruned if isinstance(pruned, dict) else {}


def _overall_ocr_res(item: dict) -> dict:
    pruned = _pruned_result(item)
    ocr_res = pruned.get("overall_ocr_res")
    if isinstance(ocr_res, dict):
        return ocr_res
    direct = item.get("overall_ocr_res") if isinstance(item, dict) else None
    return direct if isinstance(direct, dict) else {}


def _iter_result_items(data: dict) -> list[dict]:
    if not isinstance(data, dict):
        return []
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


class PaddleAdapter:
    @classmethod
    def detect(cls, raw: Any) -> bool:
        if not isinstance(raw, dict):
            return False
        result = raw.get("result", raw)
        return isinstance(result, dict) and (
            "overall_ocr_res" in result
            or "parsing_res_list" in result
            or "rec_texts" in result
            or "layoutParsingResults" in result
            or "ocrResults" in result
        )

    def parse(self, raw: Any, *, source_path: str = "", image_path: str = "") -> DocumentNode:
        doc = DocumentNode.make(source_path=source_path, engine="paddle", raw=raw)
        log = doc.parse_log

        data = raw.get("result", raw) if isinstance(raw, dict) else raw
        if not isinstance(data, dict):
            log.append("ERROR: top-level data is not a dict")
            return doc

        page = PageNode.make(page_number=1, image_path=image_path, raw=data)
        doc.pages.append(page)
        profile = _detect_profile(raw, data)
        if profile:
            spec = API_MODEL_PROFILES[profile]
            log.append(
                "INFO: profile="
                f"{spec['label']} endpoint={spec['endpoint_suffix']} "
                f"max_text_bbox_granularity={spec['max_text_bbox_granularity']}"
            )

        items = _iter_result_items(data)

        # Blocks
        order = 0
        for item in items:
            pruned = _pruned_result(item)
            for block_raw in pruned.get("parsing_res_list") or item.get("parsing_res_list") or []:
                if not isinstance(block_raw, dict):
                    continue
                label = str(block_raw.get("block_label", "text"))
                bbox = _bbox_from_box(block_raw.get("block_bbox") or block_raw.get("block_polygon_points"))
                content = str(block_raw.get("block_content") or "")
                block = BlockNode.make(label=label, bbox=bbox, content=content,
                                       order=order, source_field="parsing_res_list",
                                       raw=block_raw)
                page.blocks.append(block)
                order += 1

        det_order = 0
        for item in items:
            pruned = _pruned_result(item)
            layout_det = pruned.get("layout_det_res") or item.get("layout_det_res") or {}
            if not isinstance(layout_det, dict):
                continue
            for det_box in layout_det.get("boxes") or []:
                if not isinstance(det_box, dict):
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
                det_bbox = _bbox_from_box(det_box)
                try:
                    det_score = float(
                        det_box.get("score")
                        or det_box.get("confidence")
                        or det_box.get("layout_score")
                        or det_box.get("cls_score")
                        or 0.0
                    )
                except (TypeError, ValueError):
                    det_score = 0.0
                det_block = BlockNode.make(
                    label=det_label,
                    bbox=det_bbox,
                    content=f"[layout_det score={det_score:.3f}]",
                    order=det_order,
                    source_field="layout_det_res",
                    raw=det_box,
                )
                page.layout_det_blocks.append(det_block)
                det_order += 1

        log.append(f"INFO: parsing_res_list → {len(page.blocks)} block(s)")
        log.append(f"INFO: layout_det_res.boxes → {len(page.layout_det_blocks)} det-box(es)")

        # Lines
        global_row = 0
        for item in items:
            pruned = _pruned_result(item)
            ocr_res = _overall_ocr_res(item)
            rec_texts  = ocr_res.get("rec_texts")  or []
            rec_boxes  = ocr_res.get("rec_boxes")  or []
            rec_polys  = ocr_res.get("rec_polys") or ocr_res.get("rec_polygons") or ocr_res.get("dt_polys") or []
            rec_scores = ocr_res.get("rec_scores") or []
            text_words = pruned.get("text_word") or ocr_res.get("text_word") or []
            word_regions = pruned.get("text_word_region") or ocr_res.get("text_word_region") or []

            if not rec_texts:
                log.append("WARNING: overall_ocr_res.rec_texts is empty or missing")

            for row_idx, text in enumerate(rec_texts):
                text = str(text)
                conf  = float(rec_scores[row_idx]) if row_idx < len(rec_scores) else 1.0
                box   = rec_boxes[row_idx] if row_idx < len(rec_boxes) else None
                poly  = rec_polys[row_idx] if row_idx < len(rec_polys) else None
                bbox    = _bbox_from_box(box)
                polygon = Polygon.from_raw(poly)
                if bbox is None and polygon is not None:
                    bbox = polygon.to_bbox()
                    log.append(f"INFO: row {global_row} bbox estimated from polygon")

                line = LineNode.make(
                    text=text, confidence=conf, bbox=bbox, polygon=polygon,
                    source_field="overall_ocr_res.rec_texts",
                    raw={"rec_text": text, "rec_box": box, "rec_poly": poly, "rec_score": conf},
                )

                tok_row    = text_words[row_idx]   if row_idx < len(text_words)   else None
                region_row = word_regions[row_idx] if row_idx < len(word_regions) else None
                line.chars = self._build_chars(
                    text=text, confidence=conf, line_bbox=bbox,
                    token_row=tok_row, region_row=region_row, row_idx=global_row, log=log,
                )

                matched = self._match_block(line, page.blocks)
                if matched is not None:
                    matched.lines.append(line)
                else:
                    page.orphan_lines.append(line)
                global_row += 1

        total_lines = len(page.all_lines)
        total_chars = len(page.all_chars)
        total_orphan = len(page.orphan_lines)
        log.append(f"INFO: overall_ocr_res → {total_lines} line(s), {total_chars} char(s)")
        if total_orphan:
            log.append(f"WARNING: {total_orphan} orphan line(s) not matched to any block")
        log.append(f"INFO: parse complete — coord space: image pixel space ({page.width}×{page.height})")

        return doc

    def _build_chars(self, *, text, confidence, line_bbox, token_row, region_row, row_idx, log):
        chars: List[CharNode] = []
        if not text:
            return chars
        for glyph in text:
            chars.append(CharNode.make(
                char=glyph, bbox=None, confidence=confidence,
                kind=classify_ir_text(glyph), bbox_source="unavailable",
                bbox_granularity="unavailable", token_text=glyph,
                collection_kind="line",
            ))
        if not isinstance(token_row, (list, tuple)) or not isinstance(region_row, (list, tuple)):
            log.append(f"INFO: row {row_idx} has no token/word regions; chars keep unavailable bbox")
            return chars
        tokens = [str(t) for t in token_row if t is not None]
        cursor = 0
        for tok_idx, (tok_text, raw_region) in enumerate(zip(tokens, region_row)):
            tok_text = tok_text.strip()
            if not tok_text:
                continue
            bbox = _bbox_from_box(raw_region)
            if bbox is None:
                log.append(f"WARNING: row {row_idx} token {tok_idx} unparsable region")
                continue
            poly = Polygon.from_raw(raw_region)
            start = text.find(tok_text, cursor)
            if start < 0:
                start = text.find(tok_text, max(0, cursor - 1))
            if start < 0:
                log.append(f"WARNING: row {row_idx} token '{tok_text}' not found")
                continue
            end = min(len(text), start + len(tok_text))
            gran = "char" if len(tok_text) == 1 else "word"
            kind = classify_ir_text(tok_text)
            for ci in range(start, end):
                chars[ci] = CharNode.make(
                    char=text[ci], bbox=bbox, polygon=poly, confidence=confidence,
                    kind=kind, bbox_source="ocr", bbox_granularity=gran,
                    token_text=tok_text,
                    collection_kind="char" if len(tok_text) == 1 else "token",
                    raw={"token_text": tok_text, "region": raw_region},
                )
            cursor = end
        return chars

    @staticmethod
    def _match_block(line: LineNode, blocks: List[BlockNode]) -> Optional[BlockNode]:
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
