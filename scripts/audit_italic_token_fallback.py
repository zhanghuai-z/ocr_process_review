#!/usr/bin/env python3
"""Audit token-local EngCut geometry and punctuation ownership.

This is an offline diagnostic. It uses cached Layout/PP-OCR observations, runs
the current production router and CharOCR engine, and writes evidence under a
debug directory. It never commits OCR results or edits a project file.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.adapters.paddle.ppocr_v6_prepass import (  # noqa: E402
    PpOcrV6LineHint,
    normalize_ppocr_v6_prepass_result,
    parse_ppocr_v6_prepass_jsonl,
)
import app.core.charocr_text_partition as partition_module  # noqa: E402
import app.core.ppocr_route_compiler as route_compiler_module  # noqa: E402
import app.engines.hanwang.micro_recblock as micro_module  # noqa: E402
from app.engines.hanwang.micro_recblock import HanwangMicroRecBlockEngine  # noqa: E402
from app.infrastructure.project_store import load_session  # noqa: E402
from app.models.project_session import PageRecord, ProjectRecord, ProjectSession  # noqa: E402
from app.services.layout_analysis_service import LayoutAnalysisService  # noqa: E402
from app.services.ocr_job_service import OcrJobService  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
POLICY_NAMES = ("baseline", "conservative", "balanced", "broad")


@dataclass(frozen=True)
class Dataset:
    name: str
    image_dir: Path
    layout_assets: Path
    ppocr_dirs: tuple[Path, ...]


@dataclass
class PageAudit:
    dataset: str
    page_id: str
    source_name: str
    source_image: str
    run_index: int
    status: str = "failed"
    seconds: float = 0.0
    error: str = ""
    tokens: list[dict[str, Any]] = field(default_factory=list)
    symbols: list[dict[str, Any]] = field(default_factory=list)


class _CachedLayoutClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    def analyze_image_bytes(self, *_args, **_kwargs) -> dict[str, Any]:
        return self._response


class _CachedPrepassClient:
    def __init__(self, path: Path) -> None:
        self._path = path

    def analyze_page(self, image_bgr: np.ndarray, *, page_uid: str = ""):
        if self._path.name.endswith(".raw-pages.json"):
            rows = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                rows = [rows]
            results = [
                item
                for row in rows
                if isinstance(row, dict)
                for item in row.get("result", {}).get("ocrResults", [])
                if isinstance(item, dict)
            ]
            if len(results) != 1:
                raise RuntimeError(
                    f"expected one cached PP-OCRv6 page in {self._path}, got {len(results)}"
                )
            height, width = image_bgr.shape[:2]
            return normalize_ppocr_v6_prepass_result(
                results[0],
                page_uid=page_uid,
                run_id=f"cached:{self._path.stem}",
                width=width,
                height=height,
            )
        return parse_ppocr_v6_prepass_jsonl(
            self._path.read_text(encoding="utf-8"),
            page_uid=page_uid,
            run_id=f"cached:{self._path.stem}",
        )


class _NoNetworkVlClient:
    def analyze_image(self, *_args, **_kwargs):
        raise RuntimeError("cached audit unexpectedly requires a fresh VL observation")


def _xyxy(value: Any) -> tuple[int, int, int, int]:
    return tuple(int(item) for item in value)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _intersects(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    return max(left[0], right[0]) < min(left[2], right[2]) and max(
        left[1], right[1]
    ) < min(left[3], right[3])


def _contains_center(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    x, y = _bbox_center(inner)
    return outer[0] <= x < outer[2] and outer[1] <= y < outer[3]


def _union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _token_key(text: str, bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, int, int, int]]:
    return str(text), _xyxy(bbox)


def _word_row(token: Any) -> dict[str, Any]:
    return {
        "token_index": int(token.token_index),
        "text": str(token.text),
        "bbox": list(_xyxy(token.bbox)),
        "is_symbol": not any(char.isalnum() for char in str(token.text or "")),
    }


class _Capture:
    def __init__(self, image_bgr: np.ndarray) -> None:
        self.image_bgr = image_bgr
        self.partitions: list[dict[str, Any]] = []
        self.engcut: list[dict[str, Any]] = []
        self.symbols: list[dict[str, Any]] = []
        self._active_partition: dict[str, Any] | None = None

    def capture_partition(
        self,
        original,
        image_bgr: np.ndarray | None,
        prepass_line: PpOcrV6LineHint,
        region_bbox,
        **kwargs,
    ):
        record: dict[str, Any] = {
            "line_text": str(prepass_line.text or ""),
            "line_bbox": list(_xyxy(prepass_line.bbox)),
            "region_bbox": list(_xyxy(region_bbox)),
            "tokens": [_word_row(token) for token in prepass_line.words],
            "ownership": None,
        }
        previous = self._active_partition
        self._active_partition = record
        try:
            result = original(image_bgr, prepass_line, region_bbox, **kwargs)
        finally:
            self._active_partition = previous
        record["segments"] = [
            {
                "kind": segment.kind,
                "bbox": list(segment.bbox),
                "text": segment.text,
                "tokens": [
                    {"text": token.text, "bbox": list(token.bbox)}
                    for token in segment.ppocr_latin_tokens
                ],
            }
            for segment in result.segments
        ]
        record["symbol_observations"] = [
            {
                "text": observation.text,
                "bbox": list(observation.bbox),
                "proposal_bbox": list(observation.proposal_bbox),
            }
            for observation in result.symbol_observations
        ]
        record["issues"] = [item.code for item in result.issues]
        self.partitions.append(record)
        return result

    def capture_ownership(
        self,
        original,
        components,
        tokens,
        region_bbox,
        **kwargs,
    ):
        result = original(components, tokens, region_bbox, **kwargs)
        owners = result[0]
        if self._active_partition is not None:
            self._active_partition["ownership"] = {
                "region_bbox": list(_xyxy(region_bbox)),
                "components": [
                    {
                        "bbox": list(component.bbox),
                        "area": int(component.area),
                        "owner_token_index": owners.get(component),
                    }
                    for component in components
                ],
                "tokens": [_word_row(token) for token in tokens],
            }
        return result

    def capture_engcut(self, original, chars, **kwargs):
        result = original(chars, **kwargs)
        self.engcut.append({
            "tokens": [
                {"text": token.text, "bbox": list(token.bbox)}
                for token in kwargs.get("ppocr_tokens", ())
            ],
            "foreground_word_bbox": (
                list(kwargs["foreground_word_bbox"])
                if kwargs.get("foreground_word_bbox") is not None
                else None
            ),
            "native_chars": [
                {
                    "text": str(char.text or ""),
                    "bbox": list(char.bbox) if char.bbox is not None else None,
                    "line_index": int(char.line_index),
                    "group_index": int(char.group_index),
                    "char_index": int(char.char_index),
                }
                for char in chars
            ],
            "result_text": result[0],
            "result_atoms": [
                {
                    "text": atom.text,
                    "bbox": list(atom.bbox) if atom.bbox is not None else None,
                    "source": atom.source,
                    "granularity": atom.bbox_granularity,
                }
                for atom in result[1]
            ],
            "text_disagreement": bool(result[2]),
        })
        return result

    def capture_symbols(self, original, lines, observations):
        rows: list[dict[str, Any]] = []
        for observation in observations:
            center_x, center_y = _bbox_center(observation.bbox)
            owners = [
                line
                for line in lines
                if line.bbox[0] <= center_x < line.bbox[2]
                and line.bbox[1] <= center_y < line.bbox[3]
            ]
            claimed: list[Any] = []
            intersecting: list[Any] = []
            if len(owners) == 1:
                line = owners[0]
                claimed = [
                    atom
                    for atom in line.chars
                    if atom.bbox is not None
                    and _contains_center(observation.proposal_bbox, atom.bbox)
                    and _intersects(atom.bbox, observation.bbox)
                ]
                intersecting = [
                    atom
                    for atom in line.chars
                    if atom.bbox is not None and _intersects(atom.bbox, observation.bbox)
                ]
            if len(owners) != 1:
                status = "symbol_line_owner_ambiguous"
            elif len(claimed) == 1:
                status = (
                    "symbol_native_exact"
                    if claimed[0].text == observation.text
                    else "symbol_candidate_bound"
                )
            elif claimed or intersecting:
                status = "symbol_unbound_by_atom_overlap"
            else:
                status = "symbol_atom_inserted"
            rows.append({
                "text": observation.text,
                "bbox": list(observation.bbox),
                "proposal_bbox": list(observation.proposal_bbox),
                "status": status,
                "intersecting_atoms": [
                    {
                        "text": atom.text,
                        "bbox": list(atom.bbox),
                        "source": atom.source,
                        "granularity": atom.bbox_granularity,
                    }
                    for atom in intersecting
                ],
            })
        result = original(lines, observations)
        self.symbols.extend(rows)
        return result


@contextmanager
def _capture_runtime(capture: _Capture) -> Iterator[None]:
    original_partition = partition_module.partition_charocr_text_region
    original_compiler_partition = route_compiler_module.partition_charocr_text_region
    original_ownership = partition_module._component_owner_token_indices
    original_engcut = micro_module._engcut_route_line_text_and_chars
    original_symbols = micro_module._apply_ppocr_symbol_observations

    def partition_wrapper(image_bgr, prepass_line, region_bbox, **kwargs):
        return capture.capture_partition(
            original_partition,
            image_bgr,
            prepass_line,
            region_bbox,
            **kwargs,
        )

    def ownership_wrapper(components, tokens, region_bbox, **kwargs):
        return capture.capture_ownership(
            original_ownership,
            components,
            tokens,
            region_bbox,
            **kwargs,
        )

    def engcut_wrapper(chars, **kwargs):
        return capture.capture_engcut(original_engcut, chars, **kwargs)

    def symbol_wrapper(lines, observations):
        return capture.capture_symbols(original_symbols, lines, observations)

    partition_module.partition_charocr_text_region = partition_wrapper
    route_compiler_module.partition_charocr_text_region = partition_wrapper
    partition_module._component_owner_token_indices = ownership_wrapper
    micro_module._engcut_route_line_text_and_chars = engcut_wrapper
    micro_module._apply_ppocr_symbol_observations = symbol_wrapper
    try:
        yield
    finally:
        partition_module.partition_charocr_text_region = original_partition
        route_compiler_module.partition_charocr_text_region = original_compiler_partition
        partition_module._component_owner_token_indices = original_ownership
        micro_module._engcut_route_line_text_and_chars = original_engcut
        micro_module._apply_ppocr_symbol_observations = original_symbols


def _foreground_labels(
    image_bgr: np.ndarray,
    region_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = region_bbox
    gray = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    threshold, _unused = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) <= threshold:
        binary = (gray > threshold).astype(np.uint8)
    else:
        binary = (gray <= threshold).astype(np.uint8)
    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    return binary, labels, stats


def _component_pixel_map(
    image_bgr: np.ndarray,
    region_bbox: tuple[int, int, int, int],
) -> dict[tuple[tuple[int, int, int, int], int], tuple[np.ndarray, np.ndarray]]:
    _binary, labels, stats = _foreground_labels(image_bgr, region_bbox)
    rx1, ry1, _rx2, _ry2 = region_bbox
    result: dict[tuple[tuple[int, int, int, int], int], tuple[np.ndarray, np.ndarray]] = {}
    for index in range(1, len(stats)):
        left, top, width, height, area = (int(value) for value in stats[index])
        if area < 3:
            continue
        ys, xs = np.where(labels == index)
        bbox = (rx1 + left, ry1 + top, rx1 + left + width, ry1 + top + height)
        result[(bbox, area)] = (ys + ry1, xs + rx1)
    return result


def _slant_score(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    height, width = image_bgr.shape[:2]
    x1 = max(0, bbox[0] - 2)
    y1 = max(0, bbox[1] - 2)
    x2 = min(width, bbox[2] + 2)
    y2 = min(height, bbox[3] + 2)
    gray = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    _threshold, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    ys, xs = np.where(binary > 0)
    if len(xs) < 8:
        return 0.0
    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    band = max(2, int((y_max - y_min + 1) * 0.35))
    upper = ys <= y_min + band
    lower = ys >= y_max - band
    if int(np.sum(upper)) < 4 or int(np.sum(lower)) < 4:
        return 0.0
    return round(
        (float(np.mean(xs[upper])) - float(np.mean(xs[lower])))
        / max(1.0, float(y_max - y_min + 1)),
        6,
    )


def _native_geometry_metrics(
    image_bgr: np.ndarray,
    partition: dict[str, Any],
    token: dict[str, Any],
    native_chars: list[dict[str, Any]],
) -> dict[str, Any]:
    ownership = partition.get("ownership") or {}
    region_bbox = _xyxy(ownership.get("region_bbox") or partition["region_bbox"])
    owned = [
        component
        for component in ownership.get("components") or []
        if component.get("owner_token_index") == token.get("token_index")
    ]
    char_boxes = [
        _xyxy(char["bbox"])
        for char in native_chars
        if char.get("bbox") is not None and str(char.get("text") or "")
    ]
    ordered_boxes = sorted(char_boxes)
    strict_overlap = any(
        right[0] < left[2] for left, right in zip(ordered_boxes, ordered_boxes[1:])
    )
    native_centers = [_bbox_center(box)[0] for box in char_boxes]
    nonmonotonic = any(right < left for left, right in zip(native_centers, native_centers[1:]))

    segment_bbox = next(
        (
            _xyxy(segment["bbox"])
            for segment in partition.get("segments") or []
            if segment["kind"] == "text_latin"
            and any(
                _token_key(item["text"], _xyxy(item["bbox"]))
                == _token_key(token["text"], _xyxy(token["bbox"]))
                for item in segment.get("tokens") or []
            )
        ),
        _union([_xyxy(component["bbox"]) for component in owned]),
    )
    center_outside = bool(segment_bbox) and any(
        not _contains_center(segment_bbox, box) for box in char_boxes
    )

    pixels = _component_pixel_map(image_bgr, region_bbox)
    total = 0
    uncovered = 0
    multi = 0
    split_area = 0
    split_count = 0
    reconstruction_missing = 0
    for component in owned:
        key = (_xyxy(component["bbox"]), int(component["area"]))
        point_set = pixels.get(key)
        if point_set is None:
            reconstruction_missing += 1
            continue
        ys, xs = point_set
        area = len(xs)
        total += area
        coverage = np.zeros(area, dtype=np.uint8)
        significant_hits = 0
        for box in char_boxes:
            hit = (xs >= box[0]) & (xs < box[2]) & (ys >= box[1]) & (ys < box[3])
            hit_count = int(np.sum(hit))
            coverage += hit.astype(np.uint8)
            if hit_count >= max(3, round(area * 0.08)):
                significant_hits += 1
        uncovered += int(np.sum(coverage == 0))
        multi += int(np.sum(coverage > 1))
        if significant_hits >= 2:
            split_count += 1
            split_area += area

    native_text = "".join(str(char.get("text") or "") for char in native_chars)
    token_text = str(token["text"])
    native_units = [char for char in native_text if not char.isspace()]
    token_units = [char for char in token_text if not char.isspace()]
    unexpected_symbols = [
        char
        for char in native_units
        if not char.isalnum() and char not in token_units
    ]
    return {
        "native_text": native_text,
        "native_atom_count": len(native_units),
        "token_char_count": len(token_units),
        "strict_bbox_overlap": strict_overlap,
        "nonmonotonic_centers": nonmonotonic,
        "char_center_outside_route": center_outside,
        "foreground_area": total,
        "foreground_uncovered_ratio": round(uncovered / total, 6) if total else None,
        "foreground_multi_covered_ratio": round(multi / total, 6) if total else None,
        "split_component_count": split_count,
        "split_component_area_ratio": round(split_area / total, 6) if total else None,
        "component_mask_reconstruction_missing": reconstruction_missing,
        "count_mismatch": len(native_units) != len(token_units),
        "text_mismatch": native_text != token_text,
        "unexpected_internal_symbols": unexpected_symbols,
        "slant_score": _slant_score(image_bgr, segment_bbox or _xyxy(token["bbox"])),
        "route_bbox": list(segment_bbox) if segment_bbox else None,
        "owned_components": owned,
    }


def _policy_results(metrics: dict[str, Any], symbol_conflict: bool) -> dict[str, str]:
    if symbol_conflict:
        return {name: "routing_symbol_conflict" for name in POLICY_NAMES}
    split = float(metrics.get("split_component_area_ratio") or 0.0)
    uncovered = float(metrics.get("foreground_uncovered_ratio") or 0.0)
    structural = bool(
        metrics.get("count_mismatch") or metrics.get("unexpected_internal_symbols")
    )
    hard = bool(
        metrics.get("strict_bbox_overlap")
        or metrics.get("nonmonotonic_centers")
        or metrics.get("char_center_outside_route")
    )
    decisions = {
        "baseline": bool(metrics.get("strict_bbox_overlap")),
        "conservative": hard or split >= 0.30 or (structural and split >= 0.10),
        "balanced": hard or split >= 0.20 or uncovered >= 0.10 or (structural and split > 0),
        "broad": hard or split >= 0.10 or uncovered >= 0.05 or structural,
    }
    return {
        name: "word_fallback" if decision else "keep_chars"
        for name, decision in decisions.items()
    }


def _matching_partition(
    partitions: list[dict[str, Any]],
    token_observation: dict[str, Any],
    foreground_bbox: list[int] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    key = _token_key(token_observation["text"], _xyxy(token_observation["bbox"]))
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for partition in partitions:
        ownership = partition.get("ownership") or {}
        matching_segments = [
            segment
            for segment in partition.get("segments") or []
            if segment["kind"] == "text_latin"
            and (foreground_bbox is None or _xyxy(segment["bbox"]) == _xyxy(foreground_bbox))
            and any(
                _token_key(item["text"], _xyxy(item["bbox"])) == key
                for item in segment.get("tokens") or []
            )
        ]
        if len(matching_segments) != 1:
            continue
        raw_tokens = [
            token
            for token in ownership.get("tokens") or []
            if str(token["text"]) == str(token_observation["text"])
            and _intersects(_xyxy(token["bbox"]), _xyxy(token_observation["bbox"]))
        ]
        for token in raw_tokens:
            if foreground_bbox is not None and not any(
                segment["kind"] == "text_latin"
                and _xyxy(segment["bbox"]) == _xyxy(foreground_bbox)
                and any(
                    str(item["text"]) == str(token["text"])
                    and _intersects(_xyxy(item["bbox"]), _xyxy(token["bbox"]))
                    for item in segment.get("tokens") or []
                )
                for segment in partition.get("segments") or []
            ):
                continue
            candidates.append((partition, token))
    return candidates[0] if len(candidates) == 1 else None


def _adjacent_symbol_conflicts(
    partition: dict[str, Any],
    token: dict[str, Any],
    native_text: str,
) -> list[dict[str, Any]]:
    tokens = sorted(partition.get("tokens") or [], key=lambda item: item["token_index"])
    position = next(
        (
            index
            for index, item in enumerate(tokens)
            if item["token_index"] == token["token_index"]
        ),
        None,
    )
    if position is None:
        return []
    observations = partition.get("symbol_observations") or []
    conflicts: list[dict[str, Any]] = []
    for neighbor_index in (position - 1, position + 1):
        if not 0 <= neighbor_index < len(tokens):
            continue
        neighbor = tokens[neighbor_index]
        if not neighbor.get("is_symbol"):
            continue
        glyphs = [char for char in str(neighbor["text"]) if not char.isspace()]
        swallowed = [
            glyph
            for glyph in glyphs
            if glyph in native_text and glyph not in str(token["text"])
        ]
        if not swallowed:
            continue
        emitted = [
            item
            for item in observations
            if item["text"] in swallowed
            and _intersects(_xyxy(item["proposal_bbox"]), _xyxy(neighbor["bbox"]))
        ]
        conflicts.append({
            "status": "symbol_swallowed_by_latin_route",
            "symbol_token": neighbor,
            "swallowed_glyphs": swallowed,
            "symbol_observation_emitted": bool(emitted),
        })
    return conflicts


def _assemble_tokens(capture: _Capture) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for engcut in capture.engcut:
        if len(engcut["tokens"]) != 1:
            continue
        matched = _matching_partition(
            capture.partitions,
            engcut["tokens"][0],
            engcut.get("foreground_word_bbox"),
        )
        if matched is None:
            rows.append({
                "text": engcut["tokens"][0]["text"],
                "pp_bbox": engcut["tokens"][0]["bbox"],
                "audit_error": "token_partition_binding_not_unique",
            })
            continue
        partition, token = matched
        metrics = _native_geometry_metrics(
            capture.image_bgr,
            partition,
            token,
            engcut["native_chars"],
        )
        symbol_conflicts = _adjacent_symbol_conflicts(
            partition, token, metrics["native_text"]
        )
        policies = _policy_results(metrics, bool(symbol_conflicts))
        current_word = any(
            atom.get("granularity") == "word" for atom in engcut["result_atoms"]
        )
        rows.append({
            "text": token["text"],
            "pp_bbox": token["bbox"],
            "route_bbox": metrics.pop("route_bbox"),
            "line_text": partition["line_text"],
            "line_bbox": partition["line_bbox"],
            "native_chars": engcut["native_chars"],
            "current_result": {
                "text": engcut["result_text"],
                "atoms": engcut["result_atoms"],
                "word_fallback": current_word,
            },
            "metrics": metrics,
            "symbol_conflicts": symbol_conflicts,
            "policies": policies,
        })
    return sorted(
        rows,
        key=lambda item: (
            (item.get("route_bbox") or item.get("pp_bbox") or [0, 0, 0, 0])[1],
            (item.get("route_bbox") or item.get("pp_bbox") or [0, 0, 0, 0])[0],
        ),
    )


def _layout_index(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for path in sorted((root / "artifacts").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        page = payload.get("page") if isinstance(payload, dict) else None
        response = payload.get("response") if isinstance(payload, dict) else None
        source_path = page.get("source_path") if isinstance(page, dict) else None
        if not isinstance(source_path, str) or not isinstance(response, dict):
            continue
        name = Path(source_path.replace("\\", "/")).name.casefold()
        if name in result:
            duplicates.add(name)
        else:
            result[name] = response
    if duplicates:
        raise RuntimeError(f"ambiguous layout cache basenames: {sorted(duplicates)[:5]}")
    return result


def _ppocr_path(stem: str, roots: tuple[Path, ...]) -> Path:
    candidates = [
        root / name
        for root in roots
        for name in (f"{stem}.jsonl", f"{stem}.raw.jsonl", f"{stem}.raw-pages.json")
        if (root / name).is_file()
    ]
    preferred = [path for path in candidates if path.name.endswith(".raw-pages.json")]
    matches = preferred or candidates
    if len(matches) != 1:
        raise RuntimeError(f"expected one cached PP-OCRv6 response for {stem}, got {matches}")
    return matches[0]


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    return image


def _page_session(
    image_path: Path,
    image: np.ndarray,
    layout_response: dict[str, Any],
    *,
    dataset: str,
) -> tuple[ProjectSession, PageRecord]:
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    project_uid = f"italic-audit-{dataset}"
    page = PageRecord(
        project_uid=project_uid,
        uid=f"italic-audit-{dataset}-{image_path.stem}",
        image_path=str(image_path),
        source_path=str(image_path),
        cache_image_path=str(image_path),
        thumbnail_path="",
        width=image.shape[1],
        height=image.shape[0],
        page_number=1,
        source_page_index=0,
        status="imported",
        error="",
        image_hash=image_hash,
        image_revision=1,
    )
    session = ProjectSession(ProjectRecord(project_uid, project_uid))
    session.page_repository.put(page, expected_revision=0)
    LayoutAnalysisService(_CachedLayoutClient(layout_response)).analyze_page(
        session,
        page.uid,
        source_run_id=f"cached-layout:{image_path.stem}",
    )
    return session, session.page_repository.get(page.uid)


def _run_session(
    *,
    dataset: str,
    source_name: str,
    source_image: Path,
    session: ProjectSession,
    page: PageRecord,
    prepass_path: Path,
    run_index: int,
) -> PageAudit:
    started = time.perf_counter()
    audit = PageAudit(
        dataset=dataset,
        page_id=page.uid,
        source_name=source_name,
        source_image=str(source_image),
        run_index=run_index,
    )
    try:
        image = _read_image(source_image)
        capture = _Capture(image)
        service = OcrJobService(
            prepass_client=_CachedPrepassClient(prepass_path),
            vl_client=_NoNetworkVlClient(),
            engine=HanwangMicroRecBlockEngine(),
        )
        with _capture_runtime(capture):
            service.execute_page(
                service.prepare_page(session, page.uid, image_bgr=image)
            )
        audit.tokens = _assemble_tokens(capture)
        audit.symbols = capture.symbols
        audit.status = "ok"
    except Exception as exc:
        audit.error = f"{type(exc).__name__}: {exc}"
    audit.seconds = time.perf_counter() - started
    return audit


def _run_project(
    project_path: Path,
    prepass_roots: tuple[Path, ...],
    *,
    repeat: int,
) -> list[PageAudit]:
    results: list[PageAudit] = []
    for run_index in range(1, repeat + 1):
        session = load_session(project_path)
        pages = session.page_repository.all()
        if len(pages) != 1:
            raise RuntimeError(f"audit project must contain one page: {project_path}")
        page = pages[0]
        source_name = Path(page.source_path.replace("\\", "/")).name
        image_path = project_path.parent / (page.cache_image_path or page.image_path)
        results.append(_run_session(
            dataset="test3",
            source_name=source_name,
            source_image=image_path,
            session=session,
            page=page,
            prepass_path=_ppocr_path(Path(source_name).stem, prepass_roots),
            run_index=run_index,
        ))
    return results


def _batch_tasks(
    datasets: tuple[Dataset, ...],
    *,
    pages: set[str],
    limit: int,
) -> list[tuple[Dataset, Path, dict[str, Any], Path]]:
    tasks: list[tuple[Dataset, Path, dict[str, Any], Path]] = []
    for dataset in datasets:
        layouts = _layout_index(dataset.layout_assets)
        for image_path in sorted(dataset.image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if pages and image_path.stem not in pages:
                continue
            response = layouts.get(image_path.name.casefold())
            if response is None:
                raise RuntimeError(f"missing cached layout for {image_path}")
            tasks.append((
                dataset,
                image_path,
                response,
                _ppocr_path(image_path.stem, dataset.ppocr_dirs),
            ))
    return tasks[:limit] if limit > 0 else tasks


def _run_batch(
    datasets: tuple[Dataset, ...],
    *,
    pages: set[str],
    limit: int,
) -> list[PageAudit]:
    tasks = _batch_tasks(datasets, pages=pages, limit=limit)
    results: list[PageAudit] = []
    for index, (dataset, image_path, layout, prepass_path) in enumerate(tasks, 1):
        image = _read_image(image_path)
        session, page = _page_session(
            image_path, image, layout, dataset=dataset.name
        )
        result = _run_session(
            dataset=dataset.name,
            source_name=image_path.name,
            source_image=image_path,
            session=session,
            page=page,
            prepass_path=prepass_path,
            run_index=1,
        )
        results.append(result)
        print(
            f"[{index:03d}/{len(tasks):03d}] {dataset.name}/{image_path.name}: "
            f"{result.status.upper()} tokens={len(result.tokens)} {result.seconds:.2f}s",
            flush=True,
        )
    return results


def _draw_bbox(
    image: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int,
    origin: tuple[int, int],
) -> None:
    x1, y1, x2, y2 = _xyxy(bbox)
    cv2.rectangle(
        image,
        (x1 - origin[0], y1 - origin[1]),
        (x2 - origin[0] - 1, y2 - origin[1] - 1),
        color,
        thickness,
        cv2.LINE_AA,
    )


def _token_panel(image: np.ndarray, token: dict[str, Any]) -> np.ndarray:
    bbox = _xyxy(token.get("route_bbox") or token["pp_bbox"])
    height, width = image.shape[:2]
    crop_bbox = (
        max(0, bbox[0] - 20),
        max(0, bbox[1] - 18),
        min(width, bbox[2] + 20),
        min(height, bbox[3] + 18),
    )
    x1, y1, x2, y2 = crop_bbox
    raw = image[y1:y2, x1:x2].copy()
    panels = [raw.copy() for _index in range(4)]
    origin = (x1, y1)
    _draw_bbox(panels[1], token["pp_bbox"], (255, 100, 0), 2, origin)
    if token.get("route_bbox"):
        _draw_bbox(panels[2], token["route_bbox"], (0, 160, 0), 2, origin)
    for component in token.get("metrics", {}).get("owned_components") or []:
        _draw_bbox(panels[2], component["bbox"], (200, 0, 200), 1, origin)
    for char in token.get("native_chars") or []:
        if char.get("bbox"):
            _draw_bbox(panels[3], char["bbox"], (0, 0, 220), 1, origin)
    if token.get("route_bbox"):
        _draw_bbox(panels[3], token["route_bbox"], (255, 100, 0), 2, origin)
    target_width = max(180, max(panel.shape[1] for panel in panels))
    target_height = max(70, max(panel.shape[0] for panel in panels))
    normalized: list[np.ndarray] = []
    for panel in panels:
        canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
        canvas[:panel.shape[0], :panel.shape[1]] = panel
        normalized.append(canvas)
    body = np.hstack(normalized)
    header = np.full((58, body.shape[1], 3), 255, dtype=np.uint8)
    metrics = token.get("metrics") or {}
    policies = token.get("policies") or {}
    label = (
        f"{token.get('text')} -> {metrics.get('native_text', '')} | "
        f"split={metrics.get('split_component_area_ratio')} "
        f"unc={metrics.get('foreground_uncovered_ratio')} | "
        f"B={policies.get('balanced')}"
    )
    cv2.putText(header, label[:150], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(
        header,
        "raw | PP proposal | owned foreground | native chars + route word",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, body))


def _write_contact_sheets(
    audit: PageAudit,
    output_dir: Path,
    *,
    max_tokens: int | None = None,
) -> list[str]:
    if audit.status != "ok":
        return []
    image = _read_image(Path(audit.source_image))
    tokens = [token for token in audit.tokens if not token.get("audit_error")]
    if max_tokens is not None:
        tokens = sorted(
            tokens,
            key=lambda token: (
                bool(token.get("symbol_conflicts")),
                float(token.get("metrics", {}).get("split_component_area_ratio") or 0),
                bool(token.get("metrics", {}).get("count_mismatch")),
            ),
            reverse=True,
        )[:max_tokens]
    page_dir = output_dir / "contact_sheets" / f"{audit.dataset}_{Path(audit.source_name).stem}_run{audit.run_index}"
    page_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for start in range(0, len(tokens), 8):
        panels = [_token_panel(image, token) for token in tokens[start:start + 8]]
        width = max(panel.shape[1] for panel in panels)
        normalized = []
        for panel in panels:
            canvas = np.full((panel.shape[0], width, 3), 255, dtype=np.uint8)
            canvas[:, :panel.shape[1]] = panel
            normalized.append(canvas)
        sheet = np.vstack(normalized)
        path = page_dir / f"tokens_{start // 8 + 1:02d}.png"
        cv2.imencode(".png", sheet)[1].tofile(path)
        paths.append(str(path))
    return paths


def _windows_path(path: Path) -> str:
    text = path.resolve().as_posix()
    if text.startswith("/mnt/d/"):
        text = "D:/" + text[len("/mnt/d/") :]
    return text.replace("/", "\\")


def _summary(pages: list[PageAudit]) -> dict[str, Any]:
    successful = [page for page in pages if page.status == "ok"]
    tokens = [token for page in successful for token in page.tokens if not token.get("audit_error")]
    policies = {
        name: dict(Counter(token.get("policies", {}).get(name, "missing") for token in tokens))
        for name in POLICY_NAMES
    }
    symbol_status = Counter(
        symbol["status"] for page in successful for symbol in page.symbols
    )
    return {
        "page_count": len(pages),
        "success": len(successful),
        "failed": len(pages) - len(successful),
        "token_count": len(tokens),
        "current_word_fallbacks": sum(
            bool(token.get("current_result", {}).get("word_fallback")) for token in tokens
        ),
        "text_mismatches": sum(
            bool(token.get("metrics", {}).get("text_mismatch")) for token in tokens
        ),
        "symbol_conflict_tokens": sum(bool(token.get("symbol_conflicts")) for token in tokens),
        "policies": policies,
        "symbol_status": dict(symbol_status),
    }


def _repeat_stability(project_pages: list[PageAudit]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
    for page in project_pages:
        if page.status != "ok":
            continue
        for token in page.tokens:
            key = (str(token.get("text")), tuple(token.get("route_bbox") or token.get("pp_bbox") or ()))
            grouped[key].append(token)
    rows: list[dict[str, Any]] = []
    for (text, bbox), tokens in sorted(grouped.items(), key=lambda item: (item[0][1][1], item[0][1][0])):
        native_texts = [token.get("metrics", {}).get("native_text") for token in tokens]
        current = [token.get("current_result", {}).get("word_fallback") for token in tokens]
        balanced = [token.get("policies", {}).get("balanced") for token in tokens]
        rows.append({
            "text": text,
            "bbox": list(bbox),
            "runs": len(tokens),
            "native_texts": native_texts,
            "current_word_fallbacks": current,
            "balanced_decisions": balanced,
            "native_stable": len(set(native_texts)) == 1,
            "current_stable": len(set(current)) == 1,
            "balanced_stable": len(set(balanced)) == 1,
        })
    return rows


def _write_report(
    output_dir: Path,
    project_pages: list[PageAudit],
    batch_pages: list[PageAudit],
    contact_sheets: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    project_run1_summary = _summary(project_pages[:1])
    payload = {
        "schema": "italic_token_fallback_study.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "project_run1_summary": project_run1_summary,
        "project_summary": _summary(project_pages),
        "batch_summary": _summary(batch_pages),
        "repeat_stability": _repeat_stability(project_pages),
        "contact_sheets": contact_sheets,
        "project_pages": [asdict(page) for page in project_pages],
        "batch_pages": [asdict(page) for page in batch_pages],
    }
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    test_summary = payload["project_summary"]
    test_run1_summary = payload["project_run1_summary"]
    batch_summary = payload["batch_summary"]
    failed_batch_pages = [
        page for page in batch_pages if page.status != "ok"
    ]
    unstable = [
        row
        for row in payload["repeat_stability"]
        if not row["native_stable"] or not row["current_stable"] or not row["balanced_stable"]
    ]
    lines = [
        "# Italic Token Fallback Study",
        "",
        "Diagnostic only. No OCR observation or project file was modified.",
        "",
        "## Summary",
        "",
        f"- test3 runs: {test_summary['page_count']} (success {test_summary['success']})",
        f"- test3 run 1 tokens: {test_run1_summary['token_count']}",
        f"- test3 run 1 current word fallbacks: {test_run1_summary['current_word_fallbacks']}",
        f"- test3 run 1 symbol-conflict tokens: {test_run1_summary['symbol_conflict_tokens']}",
        f"- batch pages: {batch_summary['page_count']} (success {batch_summary['success']}, failed {batch_summary['failed']})",
        f"- batch tokens: {batch_summary['token_count']}",
        f"- batch current word fallbacks: {batch_summary['current_word_fallbacks']}",
        "",
        "## Candidate Policies",
        "",
        "These threshold policies are diagnostic comparisons, not production recommendations.",
        "The study has no human-labelled ground truth for acceptable character boxes.",
        "",
        "| scope | baseline | conservative | balanced | broad | symbol conflict |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    batch_label = f"batch ({batch_summary['page_count']} pages)"
    for name, summary in (("test3 run 1", test_run1_summary), (batch_label, batch_summary)):
        lines.append(
            f"| {name} | "
            f"{summary['policies']['baseline'].get('word_fallback', 0)} | "
            f"{summary['policies']['conservative'].get('word_fallback', 0)} | "
            f"{summary['policies']['balanced'].get('word_fallback', 0)} | "
            f"{summary['policies']['broad'].get('word_fallback', 0)} | "
            f"{summary['symbol_conflict_tokens']} |"
        )
    lines.extend((
        "",
        "## Interpretation Guardrails",
        "",
        "- Failed pages are reported but excluded from token and policy counts.",
        "- Symbol ownership conflicts are reported separately and never converted into word fallback by a candidate policy.",
        "- Slant score is recorded for inspection only; it is not a policy trigger.",
        "- A component-split threshold catches target italic failures but also catches regular digits and Latin words; visual review is required before any production rule is selected.",
        "",
        "## Failed Batch Pages",
        "",
    ))
    if failed_batch_pages:
        for page in failed_batch_pages:
            lines.append(f"- `{page.source_name}`: {page.error}")
    else:
        lines.append("- none")
    lines.extend((
        "",
        "## Repeat Stability",
        "",
        f"- unstable token records: {len(unstable)}",
    ))
    for row in unstable[:30]:
        lines.append(
            f"- `{row['text']}` bbox={row['bbox']}: native={row['native_texts']} "
            f"current={row['current_word_fallbacks']} balanced={row['balanced_decisions']}"
        )
    lines.extend((
        "",
        "## Outputs",
        "",
        f"- JSON: `{_windows_path(json_path)}`",
        f"- Contact sheets: `{_windows_path(output_dir / 'contact_sheets')}`",
    ))
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        default=ROOT / "file/0723test/test3.ocrproj",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "debug/italic_token_fallback_study_20260723",
    )
    parser.add_argument("--repeat-project", type=int, default=3)
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--page", action="append", default=[])
    args = parser.parse_args()

    datasets = (
        Dataset(
            name="file1",
            image_dir=ROOT / "file/1",
            layout_assets=ROOT / "file/debug/all-page.assets",
            ppocr_dirs=(
                ROOT / "debug/ppocrv6_mask_stage_audit_244771/ppocrv6_raw",
                ROOT / "debug/ppocrv6_mask_stage_audit_13pages/ppocrv6_raw",
            ),
        ),
        Dataset(
            name="file2",
            image_dir=ROOT / "file/2",
            layout_assets=ROOT / "file/2/temp/68_page.assets",
            ppocr_dirs=(ROOT / "debug/ppocrv6_box_threshold_04_audit_20260714/raw",),
        ),
    )
    project_prepass_roots = datasets[0].ppocr_dirs
    project_pages = _run_project(
        args.project,
        project_prepass_roots,
        repeat=max(1, args.repeat_project),
    )
    batch_pages = [] if args.skip_batch else _run_batch(
        datasets,
        pages=set(args.page),
        limit=max(0, args.limit),
    )
    contact_sheets: list[str] = []
    if project_pages:
        contact_sheets.extend(_write_contact_sheets(project_pages[0], args.output_dir))
    for page in batch_pages:
        if page.status == "ok" and any(
            token.get("symbol_conflicts")
            or token.get("policies", {}).get("balanced") == "word_fallback"
            for token in page.tokens
        ):
            contact_sheets.extend(
                _write_contact_sheets(page, args.output_dir, max_tokens=16)
            )
    _write_report(args.output_dir, project_pages, batch_pages, contact_sheets)
    print(json.dumps({
        "schema": "italic_token_fallback_study.v1",
        "project": _summary(project_pages),
        "batch": _summary(batch_pages),
        "report": str(args.output_dir / "report.json"),
    }, ensure_ascii=False))
    return 0 if all(page.status == "ok" for page in project_pages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
