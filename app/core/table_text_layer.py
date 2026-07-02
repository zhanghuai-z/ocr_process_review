"""Hidden table text-layer geometry.

Paddle-VL gives us table semantics as HTML, but not reliable cell geometry.
This module keeps the semantic text from Paddle and infers cell text bboxes from
the page image.  The result is an internal export aid, not proof/UI content.
"""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

TABLE_TEXT_LAYER_CELLS_KEY = "table_text_layer_cells"


@dataclass(frozen=True)
class TableCell:
    text: str
    row_span: int = 1
    col_span: int = 1


@dataclass(frozen=True)
class PositionedTableCell:
    text: str
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1


class _TableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[TableCell]] = []
        self._current_row: list[TableCell] | None = None
        self._current_cell_text: list[str] | None = None
        self._current_cell_row_span = 1
        self._current_cell_col_span = 1

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            attrs_dict = {str(key).lower(): value for key, value in attrs}
            self._current_cell_text = []
            self._current_cell_row_span = positive_int(attrs_dict.get("rowspan"), default=1)
            self._current_cell_col_span = positive_int(attrs_dict.get("colspan"), default=1)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell_text is not None:
            cell_text = normalize_table_cell_text("".join(self._current_cell_text))
            if cell_text:
                self._current_row.append(TableCell(
                    text=cell_text,
                    row_span=self._current_cell_row_span,
                    col_span=self._current_cell_col_span,
                ))
            self._current_cell_text = None
            self._current_cell_row_span = 1
            self._current_cell_col_span = 1
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None
            self._current_cell_text = None

    def handle_data(self, data: str) -> None:
        if self._current_cell_text is not None:
            self._current_cell_text.append(data)


def table_rows_from_html(text: str) -> list[str]:
    parser = _parse_table_html(text)
    if parser is None:
        return []
    rows: list[str] = []
    for row in parser.rows:
        cells = [normalize_table_cell_text(cell.text) for cell in row]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append("  ".join(cells))
    return rows


def positioned_table_cells_from_html(text: str) -> tuple[list[PositionedTableCell], int, int]:
    parser = _parse_table_html(text)
    if parser is None:
        return [], 0, 0
    return position_table_cells(parser.rows)


def _parse_table_html(text: str) -> _TableTextParser | None:
    raw = str(text or "").strip()
    if "<tr" not in raw.lower() and "<table" not in raw.lower():
        return None
    parser = _TableTextParser()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        return None
    return parser


def position_table_cells(rows: list[list[TableCell]]) -> tuple[list[PositionedTableCell], int, int]:
    positioned: list[PositionedTableCell] = []
    occupied: set[tuple[int, int]] = set()
    max_col = 0
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row:
            while (row_idx, col_idx) in occupied:
                col_idx += 1
            row_span = max(1, int(cell.row_span))
            col_span = max(1, int(cell.col_span))
            positioned.append(PositionedTableCell(
                text=cell.text,
                row=row_idx,
                col=col_idx,
                row_span=row_span,
                col_span=col_span,
            ))
            for rr in range(row_idx, row_idx + row_span):
                for cc in range(col_idx, col_idx + col_span):
                    occupied.add((rr, cc))
            col_idx += col_span
            max_col = max(max_col, col_idx)
    return positioned, len(rows), max_col


def build_table_text_layer_cells(
    *,
    image_path: str,
    table_bbox: dict[str, Any],
    html: str,
    page_width: int,
    page_height: int,
) -> list[dict[str, Any]]:
    cells, row_count, col_count = positioned_table_cells_from_html(html)
    if not cells:
        return []
    inferred = infer_table_cell_bboxes_from_image(
        image_path=image_path,
        table_bbox=table_bbox,
        cells=cells,
        row_count=row_count,
        col_count=col_count,
        page_width=page_width,
        page_height=page_height,
    )
    result: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        bbox = inferred.get(index)
        bbox_source = "image_text_cluster"
        if bbox is None:
            bbox = split_bbox_cell(
                table_bbox,
                row_count=row_count,
                col_count=col_count,
                row=cell.row,
                col=cell.col,
                row_span=cell.row_span,
                col_span=cell.col_span,
            )
            bbox_source = "equal_grid_fallback"
        if bbox is None:
            continue
        result.append({
            "text": cell.text,
            "bbox": _float_bbox_to_dict(bbox),
            "row": cell.row,
            "col": cell.col,
            "row_span": cell.row_span,
            "col_span": cell.col_span,
            "bbox_source": bbox_source,
            "bbox_granularity": "table_formula_cell" if is_formula_cell_text(cell.text) else "table_cell",
        })
    return result


def infer_table_cell_bboxes_from_image(
    *,
    image_path: str,
    table_bbox: dict[str, Any],
    cells: list[PositionedTableCell],
    row_count: int,
    col_count: int,
    page_width: int,
    page_height: int,
) -> dict[int, dict[str, float]]:
    if row_count <= 0 or col_count <= 0 or not cells:
        return {}
    path = resolve_page_image_path(image_path)
    if not path.exists():
        return {}
    try:
        import cv2
        import numpy as np
    except Exception:
        return {}

    full_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if full_image is None:
        return {}
    image_height, image_width = full_image.shape[:2]
    max_width = page_width or image_width
    max_height = page_height or image_height
    left, top, right, bottom = bbox_dict_to_xyxy(table_bbox, max_width, max_height)
    if right <= left or bottom <= top:
        return {}
    left = max(0, min(image_width, left))
    right = max(left, min(image_width, right))
    top = max(0, min(image_height, top))
    bottom = max(top, min(image_height, bottom))
    crop = full_image[top:bottom, left:right]
    if crop.size == 0:
        return {}

    clusters_by_row = table_text_clusters(crop, cv2, np)
    if len(clusters_by_row) < max(2, row_count // 2):
        return {}
    if len(clusters_by_row) < row_count:
        return {}
    clusters_by_row = clusters_by_row[:row_count]

    column_centers = table_column_centers(clusters_by_row, col_count)
    if len(column_centers) != col_count:
        return {}
    row_bounds = bounds_from_centers([row["center_y"] for row in clusters_by_row], crop.shape[0])
    col_bounds = bounds_from_centers(column_centers, crop.shape[1])

    all_clusters: list[dict[str, float]] = []
    for row in clusters_by_row:
        all_clusters.extend(row["clusters"])

    result: dict[int, dict[str, float]] = {}
    for index, cell in enumerate(cells):
        if cell.row >= len(row_bounds) - 1 or cell.col >= len(col_bounds) - 1:
            continue
        row_end = min(len(row_bounds) - 1, cell.row + max(1, cell.row_span))
        col_end = min(len(col_bounds) - 1, cell.col + max(1, cell.col_span))
        search = (
            col_bounds[cell.col],
            row_bounds[cell.row],
            col_bounds[col_end],
            row_bounds[row_end],
        )
        matches = [
            cluster
            for cluster in all_clusters
            if point_in_rect(cluster["cx"], cluster["cy"], search, pad_x=8.0, pad_y=8.0)
        ]
        bbox = union_cluster_bboxes(matches, page_left=left, page_top=top)
        if bbox is not None:
            result[index] = bbox
    return result


def resolve_page_image_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    cache_candidate = Path(__file__).parent.parent.parent / ".cache" / "images" / path.name
    if cache_candidate.exists():
        return cache_candidate
    return path


def bbox_dict_to_xyxy(bbox: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    x = int(round(float(bbox.get("x") or 0)))
    y = int(round(float(bbox.get("y") or 0)))
    w = int(round(float(bbox.get("w") or 0)))
    h = int(round(float(bbox.get("h") or 0)))
    left = max(0, min(width, x))
    top = max(0, min(height, y))
    right = max(left, min(width, x + w))
    bottom = max(top, min(height, y + h))
    return left, top, right, bottom


def table_text_clusters(crop, cv2, np) -> list[dict[str, Any]]:
    blurred = cv2.GaussianBlur(crop, (3, 3), 0)
    _threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    width = crop.shape[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 12), 1))
    horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    text_mask = cv2.subtract(mask, horizontal_lines)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    projection_y = (text_mask > 0).sum(axis=1)
    threshold_y = max(2, int(width * 0.002))
    raw_bands = projection_bands(projection_y, threshold_y, max_gap=8, min_size=5)
    rows: list[dict[str, Any]] = []
    for y1, y2 in raw_bands:
        row_mask = text_mask[y1:y2, :]
        projection_x = (row_mask > 0).sum(axis=0)
        threshold_x = max(1, int(max(1, y2 - y1) * 0.06))
        x_bands = projection_bands(projection_x, threshold_x, max_gap=35, min_size=3)
        clusters: list[dict[str, float]] = []
        for x1, x2 in x_bands:
            local = row_mask[:, x1:x2]
            ys, xs = np.where(local > 0)
            if len(xs) == 0:
                continue
            left = float(x1 + int(xs.min()))
            right = float(x1 + int(xs.max()) + 1)
            top = float(y1 + int(ys.min()))
            bottom = float(y1 + int(ys.max()) + 1)
            if (right - left) < 3 or (bottom - top) < 5:
                continue
            clusters.append({
                "x1": left,
                "y1": top,
                "x2": right,
                "y2": bottom,
                "cx": (left + right) / 2.0,
                "cy": (top + bottom) / 2.0,
            })
        if clusters:
            rows.append({
                "center_y": (min(c["y1"] for c in clusters) + max(c["y2"] for c in clusters)) / 2.0,
                "clusters": clusters,
            })
    return rows


def projection_bands(values, threshold: int, *, max_gap: int, min_size: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    last = 0
    gap = 0
    for idx, value in enumerate(values):
        if int(value) > threshold:
            if start is None:
                start = idx
            last = idx + 1
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                end = max(start, last)
                if end - start >= min_size:
                    bands.append((start, end))
                start = None
                gap = 0
    if start is not None:
        end = max(start, last)
        if end - start >= min_size:
            bands.append((start, end))
    return bands


def table_column_centers(rows: list[dict[str, Any]], col_count: int) -> list[float]:
    candidates = [row["clusters"] for row in rows if len(row["clusters"]) == col_count]
    if not candidates:
        return []
    clusters = max(candidates, key=lambda item: sum(cluster["cy"] for cluster in item) / len(item))
    return [float(cluster["cx"]) for cluster in clusters]


def bounds_from_centers(centers: list[float], limit: int) -> list[float]:
    if not centers:
        return [0.0, float(limit)]
    ordered = sorted(float(center) for center in centers)
    bounds = [0.0]
    for prev, current in zip(ordered, ordered[1:]):
        bounds.append((prev + current) / 2.0)
    bounds.append(float(limit))
    return bounds


def split_bbox_cell(
    bbox: dict[str, Any] | None,
    *,
    row_count: int,
    col_count: int,
    row: int,
    col: int,
    row_span: int = 1,
    col_span: int = 1,
) -> dict[str, float] | None:
    if row_count <= 0 or col_count <= 0 or not isinstance(bbox, dict):
        return None
    x = float(bbox.get("x") or 0)
    y = float(bbox.get("y") or 0)
    w = max(0.0, float(bbox.get("w") or 0))
    h = max(0.0, float(bbox.get("h") or 0))
    if w <= 0 or h <= 0:
        return None
    cell_w = w / float(col_count)
    cell_h = h / float(row_count)
    return {
        "x": x + max(0, col) * cell_w,
        "y": y + max(0, row) * cell_h,
        "w": max(1, col_span) * cell_w,
        "h": max(1, row_span) * cell_h,
    }


def point_in_rect(x: float, y: float, rect: tuple[float, float, float, float], *, pad_x: float, pad_y: float) -> bool:
    left, top, right, bottom = rect
    return left - pad_x <= x <= right + pad_x and top - pad_y <= y <= bottom + pad_y


def union_cluster_bboxes(
    clusters: list[dict[str, float]],
    *,
    page_left: int,
    page_top: int,
) -> dict[str, float] | None:
    if not clusters:
        return None
    left = min(cluster["x1"] for cluster in clusters)
    top = min(cluster["y1"] for cluster in clusters)
    right = max(cluster["x2"] for cluster in clusters)
    bottom = max(cluster["y2"] for cluster in clusters)
    if right <= left or bottom <= top:
        return None
    return {
        "x": page_left + left,
        "y": page_top + top,
        "w": right - left,
        "h": bottom - top,
    }


def is_formula_cell_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if value.startswith("$") and value.endswith("$"):
        return True
    return value.startswith("\\(") and value.endswith("\\)")


def normalize_table_cell_text(text: str) -> str:
    return " ".join(str(text or "").split())


def positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _float_bbox_to_dict(bbox: dict[str, float]) -> dict[str, int]:
    return {
        "x": int(round(float(bbox.get("x") or 0))),
        "y": int(round(float(bbox.get("y") or 0))),
        "w": int(round(float(bbox.get("w") or 0))),
        "h": int(round(float(bbox.get("h") or 0))),
    }
