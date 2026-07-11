"""Write exact pre-Hanwang CharOCR routing artifacts when diagnostics are enabled.

The writer observes an immutable ``PageRoutingPlan`` after it has been
compiled. It never changes the layout, OCR observations, or project file.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models.charocr_routing import PageRoutingPlan, RoutingSegment


_LOGGER = logging.getLogger(__name__)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_TEXT_KINDS = frozenset({"text_other", "text_latin"})
_COLORS: dict[str, tuple[int, int, int]] = {
    "line": (255, 180, 0),
    "text_other": (0, 165, 255),
    "text_latin": (255, 0, 255),
    "formula": (0, 190, 0),
    "skip": (120, 120, 120),
}


def route_debug_output_root(
    image_path: str | Path,
    *,
    debug_enabled: bool,
) -> Path | None:
    """Resolve an explicit hook directory or the project's diagnostic directory."""
    configured = os.environ.get("CHAROCR_ROUTE_HOOK_DIR", "").strip()
    if configured:
        return Path(configured)
    if not debug_enabled:
        return None
    path = Path(image_path)
    if path.parent.name.lower() == "images" and path.parent.parent.name.endswith(".assets"):
        return path.parent.parent / "artifacts" / "charocr_route"
    return path.parent / ".cache" / "charocr_route"


def write_charocr_route_artifacts(
    *,
    image_bgr: np.ndarray,
    source_image_path: str | Path,
    routing_plan: PageRoutingPlan,
    output_root: Path,
) -> Path | None:
    """Render the immutable dispatch plan and route-boundary visualizations.

    A failed diagnostic write only emits a warning. The OCR path must not depend
    on debug output being writable.
    """
    try:
        output_dir = output_root / _run_name(routing_plan)
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        canvas = _as_bgr(image_bgr)
        records = _route_records(routing_plan, image_bgr, crops_dir)
        _draw_overlay(canvas, routing_plan)
        _write_png(output_dir / "route-overlay.png", canvas)
        payload = {
            "schema": "charocr-pre-dispatch-route.v1",
            "page_uid": routing_plan.page_uid,
            "prepass_run_id": routing_plan.prepass_run_id,
            "source_image_path": str(source_image_path),
            "image_shape": list(image_bgr.shape),
            "dispatchable": routing_plan.is_dispatchable,
            "validation_issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "line_index": issue.line_index,
                    "bbox": list(issue.bbox),
                }
                for issue in routing_plan.validation_issues
            ],
            "routes": records,
            "legend": {
                "blue": "PP-OCRv6 line geometry",
                "orange": "text_other -> LineCut",
                "magenta": "text_latin -> EngCut",
                "green": "formula -> excluded from Hanwang",
                "gray": "skip -> excluded from Hanwang",
            },
        }
        (output_dir / "route-plan.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "README.md").write_text(
            _readme_text(routing_plan, records),
            encoding="utf-8",
        )
        _LOGGER.info("Wrote CharOCR pre-dispatch route artifacts: %s", output_dir)
        return output_dir
    except (OSError, ValueError, TypeError, cv2.error) as exc:
        _LOGGER.warning("Failed to write CharOCR pre-dispatch route artifacts: %s", exc)
        return None


def _route_records(
    routing_plan: PageRoutingPlan,
    image_bgr: np.ndarray,
    crops_dir: Path,
) -> list[dict[str, Any]]:
    height, width = image_bgr.shape[:2]
    records: list[dict[str, Any]] = []
    for block_route in routing_plan.blocks:
        for routing_line in block_route.plan.lines:
            segments: list[dict[str, Any]] = []
            for segment_index, segment in enumerate(routing_line.segments):
                record = _segment_record(segment)
                if segment.kind in _TEXT_KINDS:
                    crop = _crop(image_bgr, segment.bbox)
                    crop_name = (
                        f"line_{routing_line.index:04d}_segment_{segment_index:02d}_"
                        f"{segment.kind}_{_bbox_name(segment.bbox)}.png"
                    )
                    _write_png(crops_dir / crop_name, crop)
                    record["native_branch"] = "engcut" if segment.kind == "text_latin" else "linecut"
                    record["crop_file"] = f"crops/{crop_name}"
                    record["crop_shape"] = list(crop.shape)
                    record["crop_role"] = "route_segment_visualization"
                else:
                    record["native_branch"] = "excluded"
                segments.append(record)
            records.append({
                "block_uid": block_route.block_uid,
                "line_index": routing_line.index,
                "line_bbox": list(_clamp(routing_line.bbox, width, height)),
                "source": routing_line.source,
                "segments": segments,
            })
    return records


def _segment_record(segment: RoutingSegment) -> dict[str, Any]:
    return {
        "kind": segment.kind,
        "bbox": list(segment.bbox),
        "label": segment.label,
        "text_hint": segment.text,
        "content_bbox": list(segment.content_bbox) if segment.content_bbox else None,
        "component_grouping": segment.component_grouping,
    }


def _draw_overlay(canvas: np.ndarray, routing_plan: PageRoutingPlan) -> None:
    height, width = canvas.shape[:2]
    for block_route in routing_plan.blocks:
        for routing_line in block_route.plan.lines:
            _draw_bbox(canvas, _clamp(routing_line.bbox, width, height), _COLORS["line"], 1)
            for segment_index, segment in enumerate(routing_line.segments):
                color = _COLORS.get(segment.kind, _COLORS["skip"])
                bbox = _clamp(segment.bbox, width, height)
                _draw_bbox(canvas, bbox, color, 2)
                label = f"L{routing_line.index}:{segment_index} {segment.kind}"
                _draw_label(canvas, bbox, label, color)


def _draw_bbox(canvas: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int], thickness: int) -> None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color, thickness)


def _draw_label(canvas: np.ndarray, bbox: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, _x2, _y2 = bbox
    cv2.putText(
        canvas,
        label,
        (max(0, x1), max(12, y1 - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
        cv2.LINE_AA,
    )


def _crop(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clamp(bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"route crop bbox is empty: {bbox!r}")
    return image_bgr[y1:y2, x1:x2].copy()


def _clamp(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"unsupported image shape for route artifact: {image.shape!r}")


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"cannot encode PNG: {path}")
    path.write_bytes(encoded.tobytes())


def _bbox_name(bbox: tuple[int, int, int, int]) -> str:
    return "_".join(str(int(value)) for value in bbox)


def _run_name(routing_plan: PageRoutingPlan) -> str:
    page = _SAFE_NAME.sub("_", routing_plan.page_uid).strip("_") or "page"
    run_id = _SAFE_NAME.sub("_", routing_plan.prepass_run_id).strip("_") or "prepass"
    return f"{int(time.time() * 1000)}_{page}_{run_id}"


def _readme_text(routing_plan: PageRoutingPlan, records: list[dict[str, Any]]) -> str:
    text_count = sum(
        1
        for line in records
        for segment in line["segments"]
        if segment["kind"] in _TEXT_KINDS
    )
    return "\n".join((
        "# CharOCR Hanwang 前路由回显",
        "",
        f"- Page UID: `{routing_plan.page_uid}`",
        f"- PP-OCRv6 run: `{routing_plan.prepass_run_id}`",
        f"- 可分派: `{routing_plan.is_dispatchable}`",
        f"- 文字路由段数: `{text_count}`",
        "",
        "`route-overlay.png` 是当前内存中的 `PageRoutingPlan`：",
        "- 蓝色：PP-OCRv6 行几何；",
        "- 橙色：送 LineCut；",
        "- 紫红色：送 EngCut；",
        "- 绿色：公式，明确不送 Hanwang；",
        "- 灰色：表格/图片等跳过区域。",
        "",
        "`crops/` 内每张图只是 `PageRoutingPlan` 的原始矩形路由段回显，不是",
        "对 native 调用的逐像素复刻。`text_other` 由 LineCut 在整页和 route bbox",
        "上执行 SegImg；`text_latin` 由 EngCut 物化为同一物理行的白底蒙版。",
        "启用 `HANWANG_MICRO_RECBLOCK_HOOK_DIR` 后，实际 EngCut 白底输入会写入",
        "该目录的 `engcut_masked_inputs/`。",
        "`route-plan.json` 记录路由段、bbox、所属 block、行号和原始回显文件。",
    )) + "\n"


__all__ = [
    "route_debug_output_root",
    "write_charocr_route_artifacts",
]
