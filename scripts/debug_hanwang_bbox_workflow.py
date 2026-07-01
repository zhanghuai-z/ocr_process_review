#!/usr/bin/env python3
"""Replay the app's PP-OCRv5 -> route -> Hanwang bbox workflow for one page.

This is a diagnostic harness, not an alternate OCR implementation.  It loads a
real ``.ocrproj`` page, runs the same PP-OCRv5 prepass used by OcrPipeline when
needed, calls the same routing helpers, then calls HanwangMicroRecBlockEngine.

The script fails closed when PP-OCRv5 line hints are unavailable.  It must not
silently fall back to block-sized route boxes, because that hides the bbox bug
we are trying to inspect.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.ocr_dispatch_policy import should_dispatch_to_text_ocr
from app.core.ocr_line_hints import is_ppocr_page_line_hint, mark_ppocr_page_line_hint
from app.core.paddle_line_routing import (
    LAYOUT_LINE_ROUTES_FIELD,
    attach_page_ocr_line_routes,
    block_bbox_xyxy,
    block_text as paddle_block_text,
    route_authority_label,
    route_subblocks_for_block,
    text_slice_routes_for_block,
)
from app.core.project_store import ProjectStore
from app.engines.hanwang.micro_recblock import (
    HanwangMicroRecBlockEngine,
    _is_text_label,
    _page_blocks_from_layout,
    _page_layout_has_user_edits,
    _page_ocr_lines_from_layout,
    _strip_cached_layout_line_routes_from_page,
)
from app.engines.real_ocr_adapter import ApiOcrEngine
from app.models import BBox, Block, Line, Page
from app.services.ocr_pipeline import OcrPipeline


XYXY = tuple[int, int, int, int]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _windows_path_to_wsl(path: str) -> str:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if not match:
        return path
    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def _resolve_path(path_value: str | Path) -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        return Path()
    raw = _windows_path_to_wsl(raw).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _patch_page_image_paths(page: Page) -> Path:
    image_path = _resolve_path(page.display_image_path)
    if not image_path.exists():
        source_path = _resolve_path(page.source_path)
        if source_path.exists():
            image_path = source_path
    if not image_path.exists():
        raise FileNotFoundError(f"Cannot locate page image: {page.display_image_path!r}")
    page.cache_image_path = str(image_path)
    page.image_path = str(image_path)
    return image_path


def _bbox_to_xyxy(bbox: BBox | None) -> XYXY | None:
    if bbox is None or bbox.area <= 0:
        return None
    return tuple(int(v) for v in bbox.to_xyxy())


def _as_list(box: XYXY) -> list[int]:
    return [int(v) for v in box]


def _select_page(project_pages: list[Page], args: argparse.Namespace) -> tuple[int, Page]:
    if args.page_index is not None:
        if not (0 <= args.page_index < len(project_pages)):
            raise IndexError(f"--page-index out of range: {args.page_index}")
        return args.page_index, project_pages[args.page_index]
    if args.page_number is not None:
        for index, page in enumerate(project_pages):
            if page.page_number == args.page_number:
                return index, page
        raise ValueError(f"No page_number={args.page_number}")
    if args.page_name:
        needle = args.page_name.lower()
        for index, page in enumerate(project_pages):
            haystack = " ".join(
                [
                    str(page.display_image_path),
                    str(page.source_path),
                    str(page.page_number),
                ]
            ).lower()
            if needle in haystack:
                return index, page
        raise ValueError(f"No page matching --page-name={args.page_name!r}")
    if not project_pages:
        raise ValueError("Project has no pages")
    return 0, project_pages[0]


def _line_payload(line: Line, index: int | None = None) -> dict[str, Any]:
    bbox = _bbox_to_xyxy(line.bbox)
    return {
        "index": index,
        "uid": line.uid,
        "text": line.text,
        "ocr_text": line.ocr_text,
        "confidence": line.confidence,
        "bbox": _as_list(bbox) if bbox else None,
        "review_flags": list(line.review_flags),
        "is_ppocr_page_line_hint": is_ppocr_page_line_hint(line),
        "char_count": len(line.chars),
    }


def _block_payload(block: Block, index: int, width: int, height: int) -> dict[str, Any]:
    return {
        "index": index,
        "uid": block.uid,
        "type": block.block_type.value,
        "source": block.source.value,
        "source_label": block.source_label,
        "recognizable": block.recognizable,
        "bbox": _as_list(block.bbox.to_xyxy()),
        "line_count": len(block.lines),
        "ppocr_hint_count": sum(1 for line in block.lines if is_ppocr_page_line_hint(line)),
        "text_dispatch": should_dispatch_to_text_ocr(block),
        "raw_label": route_authority_label(block.raw_payload) if block.raw_payload else "",
        "raw_bbox": (
            _as_list(block_bbox_xyxy(block.raw_payload, width, height))
            if block.raw_payload
            else None
        ),
    }


def _load_project(project_path: Path, project_id: int) -> Any:
    store = ProjectStore(project_path)
    store.open()
    try:
        project = store.load_project(project_id)
    finally:
        store.close()
    if project is None:
        raise RuntimeError(f"Cannot load project_id={project_id} from {project_path}")
    return project


def _ensure_ppocr_hints(
    page: Page,
    image_bgr: np.ndarray,
    page_idx: int,
    mode: str,
    progress_events: list[dict[str, Any]],
    ppocr_lines_json: Path | None,
) -> tuple[list[Line], str]:
    if ppocr_lines_json is not None:
        lines = _load_ppocr_lines_json(ppocr_lines_json)
        for line in lines:
            mark_ppocr_page_line_hint(line)
        OcrPipeline(engine=HanwangMicroRecBlockEngine())._assign_page_ocr_line_hints_to_blocks(page, lines)
        hints = _page_ocr_lines_from_layout(page)
        progress_events.append(
            {
                "stage": "ppocrv5_prepass",
                "mode": "fixture",
                "source": str(ppocr_lines_json),
                "hint_count": len(hints),
            }
        )
        if not hints:
            raise RuntimeError(f"PP-OCRv5 fixture produced zero usable hints: {ppocr_lines_json}")
        return hints, f"fixture:{ppocr_lines_json}"

    existing = _page_ocr_lines_from_layout(page)
    if mode == "never":
        if not existing:
            raise RuntimeError(
                "PP-OCRv5 page-line hints are missing and --prepass=never was used"
            )
        return existing, "existing"
    if existing and mode != "always":
        return existing, "existing"

    _validate_ppocr_api_config()
    pipeline = OcrPipeline(engine=HanwangMicroRecBlockEngine(), hybrid_prepass_engine=ApiOcrEngine())
    pipeline._process_page_with_page_ocr(
        image_bgr,
        page,
        page_idx,
        engine=ApiOcrEngine(),
        mark_page_line_hints=True,
    )
    hints = _page_ocr_lines_from_layout(page)
    progress_events.append(
        {
            "stage": "ppocrv5_prepass",
            "mode": mode,
            "hint_count": len(hints),
        }
    )
    if not hints:
        raise RuntimeError("PP-OCRv5 prepass finished but produced zero page-line hints")
    return hints, "prepass"


def _validate_ppocr_api_config() -> None:
    from app.core.api_profiles import FIXED_OCR_PROFILE, resolve_api_endpoint_for_role
    from app.core.app_config import get_config

    cfg = get_config()
    endpoint = resolve_api_endpoint_for_role(
        cfg.get("api_url", ""),
        profile=FIXED_OCR_PROFILE,
        role="ocr",
    )
    if not endpoint:
        raise RuntimeError(
            "PP-OCRv5 API URL is empty. Pass --api-url or set OCR_API_URL before running this debug workflow."
        )
    if not str(cfg.get("api_token") or "").strip():
        raise RuntimeError(
            "PP-OCRv5 API token is empty. Pass --api-token or set OCR_API_TOKEN before running this debug workflow."
        )


def _load_ppocr_lines_json(path: Path) -> list[Line]:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("--ppocr-lines-json must point to a list of PP-OCRv5 line records")
    lines: list[Line] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        bbox_raw = item.get("bbox")
        if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in bbox_raw]
        if x2 <= x1 or y2 <= y1:
            continue
        text = str(item.get("text") or item.get("ocr_text") or "")
        confidence = float(item.get("confidence") or 0.0)
        line = Line(
            text=text,
            confidence=confidence,
            bbox=BBox.from_xyxy(x1, y1, x2, y2),
            ocr_text=text,
        )
        line.review_flags.append(f"ppocr_fixture_line_index:{index}")
        lines.append(line)
    return lines


def _prepare_route_blocks(page: Page) -> list[dict[str, Any]]:
    routing_page = copy.deepcopy(page)
    _strip_cached_layout_line_routes_from_page(routing_page)
    layout_blocks = (
        _page_blocks_from_layout(routing_page)
        if _page_layout_has_user_edits(routing_page)
        else (routing_page.ppvl_parsing_res_list or _page_blocks_from_layout(routing_page))
    )
    if not layout_blocks:
        raise RuntimeError("No layout blocks available for Hanwang route replay")
    return copy.deepcopy(layout_blocks)


def _route_debug_payload(
    blocks: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for block_idx, block in enumerate(blocks):
        label = route_authority_label(block)
        routes = block.get(LAYOUT_LINE_ROUTES_FIELD)
        subblocks = route_subblocks_for_block(block, width, height)
        text_slices = text_slice_routes_for_block(block, width, height) if _is_text_label(label) else []
        payload.append(
            {
                "block_idx": block_idx,
                "label": label,
                "block_bbox": _as_list(block_bbox_xyxy(block, width, height)),
                "block_text": paddle_block_text(block),
                "subblocks": [
                    {
                        "label": str(item.get("label") or ""),
                        "bbox": _as_list(tuple(item["bbox"])),
                        "text": str(item.get("text") or ""),
                    }
                    for item in subblocks
                ],
                "line_routes": copy.deepcopy(routes or []),
                "text_slices": [
                    {
                        "line_idx": int(route.get("line_idx", -1)),
                        "segment_idx": int(route.get("segment_idx", -1)),
                        "bbox": [int(v) for v in route.get("bbox", [])],
                        "carved": bool(route.get("carved")),
                    }
                    for route in text_slices
                ],
            }
        )
    return payload


def _latest_hook_json(out_dir: Path, before: set[Path]) -> Path | None:
    candidates = sorted(out_dir.glob("micro_recblock_*.json"), key=lambda item: item.stat().st_mtime)
    new_candidates = [path for path in candidates if path not in before]
    return new_candidates[-1] if new_candidates else (candidates[-1] if candidates else None)


def _run_hanwang_workflow(
    page: Page,
    image_bgr: np.ndarray,
    out_dir: Path,
    progress_events: list[dict[str, Any]],
) -> tuple[Page, Path | None]:
    page_for_engine = copy.deepcopy(page)
    before = set(out_dir.glob("micro_recblock_*.json"))
    old_hook_dir = os.environ.get("HANWANG_MICRO_RECBLOCK_HOOK_DIR")
    os.environ["HANWANG_MICRO_RECBLOCK_HOOK_DIR"] = str(out_dir)
    try:
        engine = HanwangMicroRecBlockEngine()
        engine.recognize_page_blocks(
            image_bgr,
            page_for_engine,
            progress_callback=lambda current, total, message: progress_events.append(
                {
                    "stage": "hanwang",
                    "current": current,
                    "total": total,
                    "message": message,
                }
            ),
        )
    finally:
        if old_hook_dir is None:
            os.environ.pop("HANWANG_MICRO_RECBLOCK_HOOK_DIR", None)
        else:
            os.environ["HANWANG_MICRO_RECBLOCK_HOOK_DIR"] = old_hook_dir
    return page_for_engine, _latest_hook_json(out_dir, before)


def _draw_boxes(
    image_bgr: np.ndarray,
    boxes: Iterable[tuple[XYXY, tuple[int, int, int], int, str]],
) -> np.ndarray:
    canvas = image_bgr.copy()
    for bbox, color, thickness, label in boxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(
                canvas,
                label[:18],
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def _focus_bbox_from_text(
    page: Page,
    route_blocks: list[dict[str, Any]],
    focus_text: str,
    width: int,
    height: int,
) -> XYXY | None:
    needle = focus_text.strip()
    if not needle:
        return None
    for block in page.blocks:
        haystack = " ".join(
            [block.source_label, block.note, " ".join(line.text for line in block.lines)]
        )
        if needle in haystack:
            return block.bbox.to_xyxy()
    for block in route_blocks:
        if needle in str(block.get("block_text") or ""):
            return block_bbox_xyxy(block, width, height)
    return None


def _parse_focus_bbox(raw: str) -> XYXY | None:
    if not raw:
        return None
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != 4:
        raise ValueError("--focus-bbox must be x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--focus-bbox must use xyxy with positive size")
    return x1, y1, x2, y2


def _expand_focus(box: XYXY, width: int, height: int, pad: int = 80) -> XYXY:
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(width, box[2] + pad),
        min(height, box[3] + pad),
    )


def _write_focus_crop(path: Path, image: np.ndarray, focus: XYXY | None) -> None:
    if focus is None:
        return
    x1, y1, x2, y2 = focus
    _write_image(path, image[y1:y2, x1:x2].copy())


def _make_contact_sheet(images: list[tuple[str, Path]], out_path: Path) -> None:
    thumbs: list[np.ndarray] = []
    labels: list[str] = []
    for label, path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        target_w = 520
        scale = target_w / max(1, w)
        thumb = cv2.resize(img, (target_w, max(1, int(h * scale))))
        label_band = np.full((28, target_w, 3), 255, dtype=np.uint8)
        cv2.putText(label_band, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
        thumbs.append(np.vstack([label_band, thumb]))
        labels.append(label)
    if not thumbs:
        return
    max_h = max(img.shape[0] for img in thumbs)
    padded = []
    for img in thumbs:
        if img.shape[0] < max_h:
            pad = np.full((max_h - img.shape[0], img.shape[1], 3), 245, dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)
    rows = []
    for i in range(0, len(padded), 2):
        row_imgs = padded[i:i + 2]
        if len(row_imgs) == 1:
            row_imgs.append(np.full_like(row_imgs[0], 245))
        rows.append(np.hstack(row_imgs))
    _write_image(out_path, np.vstack(rows))


def _write_overlays(
    out_dir: Path,
    image_bgr: np.ndarray,
    page_before: Page,
    route_blocks: list[dict[str, Any]],
    hook_payload: dict[str, Any] | None,
    page_after: Page | None,
    focus: XYXY | None,
) -> None:
    width = page_before.width
    height = page_before.height
    outputs: list[tuple[str, Path]] = []

    layout_boxes = [
        (block.bbox.to_xyxy(), (0, 180, 0), 2, f"b{idx}:{block.block_type.value}")
        for idx, block in enumerate(page_before.blocks)
    ]
    img = _draw_boxes(image_bgr, layout_boxes)
    path = out_dir / "01_layout_blocks_green.png"
    _write_image(path, img)
    _write_focus_crop(out_dir / "crop_01_layout_blocks_green.png", img, focus)
    outputs.append(("01 layout blocks", path))

    hint_boxes = []
    for idx, line in enumerate(_page_ocr_lines_from_layout(page_before)):
        bbox = _bbox_to_xyxy(line.bbox)
        if bbox is not None:
            hint_boxes.append((bbox, (0, 255, 255), 2, f"p{idx}"))
    img = _draw_boxes(image_bgr, hint_boxes)
    path = out_dir / "02_ppocrv5_line_hints_yellow.png"
    _write_image(path, img)
    _write_focus_crop(out_dir / "crop_02_ppocrv5_line_hints_yellow.png", img, focus)
    outputs.append(("02 ppocrv5 lines", path))

    route_boxes = []
    segment_boxes = []
    for block_idx, block in enumerate(route_blocks):
        for route in block.get(LAYOUT_LINE_ROUTES_FIELD) or []:
            bbox = tuple(int(v) for v in route.get("bbox") or [])
            if len(bbox) == 4:
                route_boxes.append((bbox, (255, 0, 0), 2, f"r{block_idx}"))
            for segment in route.get("segments") or []:
                seg_bbox = tuple(int(v) for v in segment.get("bbox") or [])
                if len(seg_bbox) != 4:
                    continue
                kind = str(segment.get("kind") or "")
                color = (255, 0, 255) if kind == "formula" else (255, 128, 0)
                segment_boxes.append((seg_bbox, color, 1, kind[:1] or "s"))
    img = _draw_boxes(image_bgr, route_boxes)
    path = out_dir / "03_bound_ppocr_route_lines_blue.png"
    _write_image(path, img)
    _write_focus_crop(out_dir / "crop_03_bound_ppocr_route_lines_blue.png", img, focus)
    outputs.append(("03 route lines", path))

    img = _draw_boxes(image_bgr, segment_boxes)
    path = out_dir / "04_route_segments_text_orange_formula_magenta.png"
    _write_image(path, img)
    _write_focus_crop(out_dir / "crop_04_route_segments_text_orange_formula_magenta.png", img, focus)
    outputs.append(("04 route segments", path))

    rows = hook_payload.get("rows", []) if hook_payload else []
    route_slice_boxes = []
    recog_group_boxes = []
    line_boxes = []
    char_boxes = []
    for row_idx, row in enumerate(rows):
        for bbox_raw in row.get("route_text_slice_bboxes") or []:
            bbox = tuple(int(v) for v in bbox_raw)
            route_slice_boxes.append((bbox, (255, 0, 0), 2, f"s{row_idx}"))
        for bbox_raw in row.get("recog_group_bboxes") or []:
            bbox = tuple(int(v) for v in bbox_raw)
            recog_group_boxes.append((bbox, (0, 255, 0), 1, f"g{row_idx}"))
        for line in row.get("lines") or []:
            bbox_raw = line.get("bbox")
            if bbox_raw:
                line_boxes.append((tuple(int(v) for v in bbox_raw), (0, 165, 255), 2, "l"))
            for char in line.get("chars") or []:
                bbox_raw = char.get("bbox")
                if bbox_raw:
                    char_boxes.append((tuple(int(v) for v in bbox_raw), (0, 0, 255), 1, str(char.get("text") or "")[:1]))

    img = _draw_boxes(image_bgr, route_slice_boxes)
    path = out_dir / "05_actual_hanwang_route_slices_blue.png"
    _write_image(path, img)
    _write_focus_crop(out_dir / "crop_05_actual_hanwang_route_slices_blue.png", img, focus)
    outputs.append(("05 actual route slices", path))

    img = _draw_boxes(image_bgr, recog_group_boxes)
    path = out_dir / "06_actual_hanwang_recog_groups_green.png"
    _write_image(path, img)
    _write_focus_crop(out_dir / "crop_06_actual_hanwang_recog_groups_green.png", img, focus)
    outputs.append(("06 actual recog groups", path))

    img = _draw_boxes(image_bgr, line_boxes + char_boxes)
    path = out_dir / "07_hanwang_output_lines_orange_chars_red.png"
    _write_image(path, img)
    _write_focus_crop(out_dir / "crop_07_hanwang_output_lines_orange_chars_red.png", img, focus)
    outputs.append(("07 hanwang output", path))

    if page_after is not None:
        final_boxes = []
        for block_idx, block in enumerate(page_after.blocks):
            final_boxes.append((block.bbox.to_xyxy(), (0, 180, 0), 2, f"b{block_idx}"))
            for line in block.lines:
                bbox = _bbox_to_xyxy(line.bbox)
                if bbox is not None:
                    final_boxes.append((bbox, (0, 165, 255), 2, "l"))
                for char in line.chars:
                    cb = _bbox_to_xyxy(char.bbox)
                    if cb is not None:
                        color = (255, 0, 0) if char.bbox_source == "paddle_inline_formula" else (0, 0, 255)
                        final_boxes.append((cb, color, 1, str(char.char or char.token_text or "")[:1]))
        img = _draw_boxes(image_bgr, final_boxes)
        path = out_dir / "08_final_model_blocks_lines_chars.png"
        _write_image(path, img)
        _write_focus_crop(out_dir / "crop_08_final_model_blocks_lines_chars.png", img, focus)
        outputs.append(("08 final model", path))

    _make_contact_sheet(outputs, out_dir / "contact_sheet_full.png")
    crop_outputs = [
        (label, out_dir / f"crop_{path.name}")
        for label, path in outputs
        if (out_dir / f"crop_{path.name}").exists()
    ]
    if crop_outputs:
        _make_contact_sheet(crop_outputs, out_dir / "contact_sheet_crop.png")


def _diagnostics_from_hook(hook_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not hook_payload:
        return {"hook_present": False}
    route_heights = []
    recog_cross_route = 0
    for row in hook_payload.get("rows", []) or []:
        route_boxes = [tuple(int(v) for v in bbox) for bbox in row.get("route_text_slice_bboxes") or []]
        route_heights.extend(max(0, box[3] - box[1]) for box in route_boxes)
        for audit in row.get("segimg_group_audits") or []:
            if audit.get("clipped"):
                recog_cross_route += 1
    median_height = 0
    if route_heights:
        ordered = sorted(route_heights)
        median_height = ordered[len(ordered) // 2]
    return {
        "hook_present": True,
        "route_slice_count": len(route_heights),
        "route_slice_height_min": min(route_heights) if route_heights else 0,
        "route_slice_height_median": median_height,
        "route_slice_height_max": max(route_heights) if route_heights else 0,
        "segimg_groups_clipped_to_route": recog_cross_route,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "file/未命名项目.ocrproj")
    parser.add_argument("--project-id", type=int, default=1)
    parser.add_argument("--page-index", type=int, default=None)
    parser.add_argument("--page-number", type=int, default=None)
    parser.add_argument("--page-name", default="120193")
    parser.add_argument("--prepass", choices=["auto", "always", "never"], default="auto")
    parser.add_argument(
        "--ppocr-lines-json",
        type=Path,
        default=None,
        help="Replay saved PP-OCRv5 line records instead of calling the network prepass.",
    )
    parser.add_argument("--api-url", default="", help="Override OCR_API_URL for the PP-OCRv5 prepass.")
    parser.add_argument("--api-token", default="", help="Override OCR_API_TOKEN for the PP-OCRv5 prepass.")
    parser.add_argument("--skip-hanwang", action="store_true")
    parser.add_argument("--focus-text", default="")
    parser.add_argument("--focus-bbox", default="")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "tmp/hanwang_bbox_workflow_replay")
    args = parser.parse_args()

    if args.api_url:
        os.environ["OCR_API_URL"] = args.api_url
    if args.api_token:
        os.environ["OCR_API_TOKEN"] = args.api_token

    out_dir = args.out_dir / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    project = _load_project(args.project, args.project_id)
    page_idx, selected_page = _select_page(project.pages, args)
    page = copy.deepcopy(selected_page)
    image_path = _patch_page_image_paths(page)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"cv2 cannot read page image: {image_path}")
    page.height, page.width = image_bgr.shape[:2]

    progress_events: list[dict[str, Any]] = []
    ppocr_hints, ppocr_source = _ensure_ppocr_hints(
        page,
        image_bgr,
        page_idx,
        args.prepass,
        progress_events,
        args.ppocr_lines_json,
    )

    route_blocks = _prepare_route_blocks(page)
    attach_page_ocr_line_routes(route_blocks, ppocr_hints, page.width, page.height)
    if not any(block.get(LAYOUT_LINE_ROUTES_FIELD) for block in route_blocks):
        raise RuntimeError("Route attach produced zero _layout_line_routes; aborting instead of falling back")

    page_after: Page | None = None
    hook_payload: dict[str, Any] | None = None
    hook_json_path: Path | None = None
    if not args.skip_hanwang:
        page_after, hook_json_path = _run_hanwang_workflow(page, image_bgr, out_dir, progress_events)
        if hook_json_path is not None:
            hook_payload = json.loads(hook_json_path.read_text(encoding="utf-8"))

    focus = _parse_focus_bbox(args.focus_bbox)
    if focus is None and args.focus_text:
        focus = _focus_bbox_from_text(page, route_blocks, args.focus_text, page.width, page.height)
    if focus is not None:
        focus = _expand_focus(focus, page.width, page.height)

    route_payload = _route_debug_payload(route_blocks, page.width, page.height)
    trace = {
        "schema": "debug_hanwang_bbox_workflow.v1",
        "project": str(args.project),
        "project_name": project.name,
        "page_index": page_idx,
        "page_number": page.page_number,
        "image_path": str(image_path),
        "image_shape": [int(v) for v in image_bgr.shape],
        "prepass": {
            "mode": args.prepass,
            "source": ppocr_source,
            "hint_count": len(ppocr_hints),
        },
        "input_blocks_before_hanwang": [
            _block_payload(block, index, page.width, page.height)
            for index, block in enumerate(page.blocks)
        ],
        "ppocrv5_line_hints": [
            _line_payload(line, index)
            for index, line in enumerate(ppocr_hints)
        ],
        "route_blocks_after_attach": route_payload,
        "hanwang_hook_json": str(hook_json_path) if hook_json_path else "",
        "diagnostics": _diagnostics_from_hook(hook_payload),
        "progress_events": progress_events,
    }
    if page_after is not None:
        trace["final_model_blocks_after_hanwang"] = [
            {
                **_block_payload(block, index, page_after.width, page_after.height),
                "lines": [_line_payload(line, line_idx) for line_idx, line in enumerate(block.lines)],
            }
            for index, block in enumerate(page_after.blocks)
        ]
    _write_json(out_dir / "workflow_trace.json", trace)
    _write_overlays(out_dir, image_bgr, page, route_blocks, hook_payload, page_after, focus)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "workflow_trace": str(out_dir / "workflow_trace.json"),
                "contact_sheet_full": str(out_dir / "contact_sheet_full.png"),
                "contact_sheet_crop": str(out_dir / "contact_sheet_crop.png"),
                "ppocr_hint_count": len(ppocr_hints),
                "ppocr_source": ppocr_source,
                "hanwang_hook_json": str(hook_json_path) if hook_json_path else "",
                "diagnostics": trace["diagnostics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
