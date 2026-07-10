"""PDF 导出：fpdf2 + 系统/内置 CJK 字体。"""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Any

from app.export.base import ExporterBase
from app.core.logging import get_logger
from app.core.table_text_layer import TABLE_TEXT_LAYER_CELLS_KEY
from app.export.ir import ExportDocument, ExportElement, ExportPage
from app.export.ir_builder import build_export_ir
from app.models import OcrProject
from app.utils.image_io import read_cv_image

_RESOURCES_FONTS = Path(__file__).parent.parent.parent / "resources" / "fonts"
logger = get_logger(__name__)
PDF_DEFAULT_DPI = 300
# fpdf/PyMuPDF expose the embedded CJK font bbox with ~1.043 ascender.
# Use it to convert OCR bbox top/bottom into a text baseline instead of
# incorrectly treating the bbox bottom as the baseline itself.
PDF_TEXT_ASCENDER_RATIO = 1.043
# The PDF text extraction bbox is taller than the requested font size for the
# CJK fonts we embed. 0.80 keeps the invisible selection rectangle close to the
# OCR bbox while retaining a small lower padding for glyph descenders.
PDF_TEXT_FONT_SIZE_TO_BBOX_RATIO = 0.80
_FORMULA_SPAN_PATTERN = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\))")

# 字体搜索顺序：优先选择同时覆盖 CJK + Latin/digits 的字体。
_FONT_CANDIDATES = [
    _RESOURCES_FONTS / "NotoSansSC-Regular.ttf",
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("/mnt/c/Windows/Fonts/simsun.ttc"),
    Path("C:/Windows/Fonts/simsun.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    # DroidSansFallbackFull in this repository covers CJK but not Latin digits
    # reliably. Keep it as a last-resort CJK fallback, not the default dual-PDF
    # text-layer font.
    _RESOURCES_FONTS / "DroidSansFallbackFull.ttf",
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
]

_LATIN_FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
]


def _find_font() -> str | None:
    for p in _FONT_CANDIDATES:
        if p.exists():
            return str(p)
    return None


def _find_latin_font() -> str | None:
    for p in _LATIN_FONT_CANDIDATES:
        if p.exists():
            return str(p)
    return None


def _add_font(pdf, font_path: str) -> None:
    import fpdf

    if not str(getattr(fpdf, "__version__", "")).startswith("1."):
        pdf.add_font("CJK", "", font_path)
        return
    pdf.add_font("CJK", "", font_path, uni=True)


def _add_latin_fallback_font(pdf) -> None:
    latin_font = _find_latin_font()
    if not latin_font or not hasattr(pdf, "set_fallback_fonts"):
        return
    try:
        pdf.add_font("LatinFallback", "", latin_font)
        pdf.set_fallback_fonts(["LatinFallback"], exact_match=False)
    except Exception as e:
        logger.warning("PDF Latin fallback font setup failed: %s", e)
        return


@dataclass(frozen=True)
class PdfTextItem:
    text: str
    x: float
    y: float
    w: float
    h: float
    source: str = ""
    bbox_source: str = ""
    bbox_granularity: str = ""

    @property
    def font_size(self) -> float:
        return max(1.0, min(72.0, self.h * PDF_TEXT_FONT_SIZE_TO_BBOX_RATIO))


@dataclass(frozen=True)
class PdfTextSpan:
    text: str
    x: float
    y: float
    w: float
    h: float
    source: str = ""
    items: tuple[PdfTextItem, ...] = ()


@dataclass(frozen=True)
class PdfPagePlan:
    page_number: int
    image_path: str
    page_width_px: int
    page_height_px: int
    width_pt: float
    height_pt: float
    text_items: list[PdfTextItem]
    text_spans: list[PdfTextSpan]
    text_font_size_to_bbox_ratio: float = PDF_TEXT_FONT_SIZE_TO_BBOX_RATIO


@dataclass(frozen=True)
class _TableCell:
    text: str
    row_span: int = 1
    col_span: int = 1


@dataclass(frozen=True)
class _PositionedTableCell:
    text: str
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1


class PdfExporter(ExporterBase):
    def __init__(self, profile: str = "pdf-single"):
        self.profile = "pdf-single" if profile == "pdf" else profile

    def export(self, project: OcrProject, out_path: str) -> None:
        from fpdf import FPDF

        document = build_export_ir(project, self.profile)
        include_text = document.profile.format == "pdf-dual"
        plans = build_pdf_page_plans(
            document,
            include_text=include_text,
            dpi=_document_dpi(document),
            text_font_size_to_bbox_ratio=_document_text_font_size_to_bbox_ratio(document),
        )
        font_path = _find_font()
        needs_text_font = include_text and any(plan.text_spans for plan in plans)
        if needs_text_font and font_path is None:
            raise RuntimeError(
                "未找到 CJK 字体文件，无法导出 PDF。\n"
                "请将 NotoSansSC-Regular.ttf 放入 resources/fonts/ 目录，"
                "或在 Windows 系统上运行（使用系统 msyh.ttc）。"
            )

        pdf = FPDF(unit="pt")
        pdf.set_auto_page_break(auto=False)
        if font_path is not None:
            _add_font(pdf, font_path)
            _add_latin_fallback_font(pdf)
        if include_text:
            self._write_dual_pdf(pdf, plans)
        else:
            self._write_single_pdf(pdf, plans)

        pdf.output(out_path)

    def _write_single_pdf(self, pdf, plans: list[PdfPagePlan]) -> None:
        for plan in plans:
            _add_page_with_size(pdf, plan)
            _write_page_image(pdf, plan)

    def _write_dual_pdf(self, pdf, plans: list[PdfPagePlan]) -> None:
        for plan in plans:
            _add_page_with_size(pdf, plan)
            _write_page_image(pdf, plan)
            if plan.text_spans:
                _write_invisible_text_layer(pdf, plan)


def build_pdf_page_plans(
    document: ExportDocument,
    *,
    include_text: bool,
    dpi: int = PDF_DEFAULT_DPI,
    text_font_size_to_bbox_ratio: float | None = None,
) -> list[PdfPagePlan]:
    plans: list[PdfPagePlan] = []
    font_ratio = _clamp_text_font_size_to_bbox_ratio(text_font_size_to_bbox_ratio)
    for page in document.pages:
        width_px = max(1, int(page.size.get("w") or 1))
        height_px = max(1, int(page.size.get("h") or 1))
        text_items: list[PdfTextItem] = []
        text_spans: list[PdfTextSpan] = []
        if include_text:
            text_items, text_spans = _collect_pdf_text_layer(page, dpi=dpi)
        plans.append(PdfPagePlan(
            page_number=int(page.page_number),
            image_path=page.source_image,
            page_width_px=width_px,
            page_height_px=height_px,
            width_pt=px_to_pdf_points(width_px, dpi),
            height_pt=px_to_pdf_points(height_px, dpi),
            text_items=text_items,
            text_spans=text_spans,
            text_font_size_to_bbox_ratio=font_ratio,
        ))
    return plans


def px_to_pdf_points(value: float, dpi: int = PDF_DEFAULT_DPI) -> float:
    return float(value) * 72.0 / float(dpi)


def pixel_bbox_to_pdf_rect(
    bbox: dict[str, Any],
    page_height_px: int,
    *,
    dpi: int = PDF_DEFAULT_DPI,
) -> tuple[float, float, float, float]:
    x = float(bbox.get("x") or 0)
    y = float(bbox.get("y") or 0)
    w = max(0.0, float(bbox.get("w") or 0))
    h = max(0.0, float(bbox.get("h") or 0))
    scale = 72.0 / float(dpi)
    return (
        x * scale,
        (float(page_height_px) - y - h) * scale,
        w * scale,
        h * scale,
    )


def _document_dpi(document: ExportDocument) -> int:
    try:
        dpi = int(document.profile.options.get("dpi") or PDF_DEFAULT_DPI)
    except (TypeError, ValueError):
        dpi = PDF_DEFAULT_DPI
    return dpi if dpi > 0 else PDF_DEFAULT_DPI


def _document_text_font_size_to_bbox_ratio(document: ExportDocument) -> float:
    try:
        value = document.profile.options.get("pdf_text_font_size_to_bbox_ratio")
    except AttributeError:
        value = None
    return _clamp_text_font_size_to_bbox_ratio(value)


def _clamp_text_font_size_to_bbox_ratio(value: Any) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        ratio = PDF_TEXT_FONT_SIZE_TO_BBOX_RATIO
    return max(0.60, min(0.95, ratio))


def _collect_pdf_text_layer(page: ExportPage, *, dpi: int) -> tuple[list[PdfTextItem], list[PdfTextSpan]]:
    items: list[PdfTextItem] = []
    spans: list[PdfTextSpan] = []
    inline_formula_bboxes = _inline_formula_bboxes_from_page(page)
    for element in page.elements:
        if element.kind == "equation" and _duplicates_inline_formula_element(element, inline_formula_bboxes):
            continue
        payload = element.payload if isinstance(element.payload, dict) else {}
        lines = payload.get("lines")
        if isinstance(lines, list):
            if element.kind == "table":
                table_items, table_spans = _table_pdf_text_layer(payload, element, page, dpi=dpi)
                items.extend(table_items)
                spans.extend(table_spans)
                continue
            for line_idx, line in enumerate(lines):
                if not isinstance(line, dict):
                    continue
                line_items, line_spans = _line_pdf_text_layer(
                    line,
                    page,
                    line_idx=line_idx,
                    dpi=dpi,
                    element_kind=str(element.kind or ""),
                    element_bbox=element.bbox,
                )
                items.extend(line_items)
                spans.extend(line_spans)
        elif isinstance(payload.get("text"), str):
            if element.kind == "equation":
                formula_text = _formula_pdf_text(payload["text"])
                element_items = _single_text_item(
                    formula_text,
                    element.bbox,
                    page_height_px=int(page.size.get("h") or 1),
                    source=f"{element.id}:element",
                    dpi=dpi,
                    bbox_source="export_equation",
                    bbox_granularity="formula",
                )
            elif element.kind == "table":
                element_items = _single_text_item(
                    payload["text"],
                    element.bbox,
                    page_height_px=int(page.size.get("h") or 1),
                    source=f"{element.id}:table",
                    dpi=dpi,
                    bbox_source="export_table",
                    bbox_granularity="table",
                )
            else:
                element_items = _synthetic_text_items(
                    payload["text"],
                    element.bbox,
                    page_height_px=int(page.size.get("h") or 1),
                    source=f"{element.id}:element",
                    dpi=dpi,
                )
            items.extend(element_items)
            span_text = element_items[0].text if element_items else payload["text"]
            span = _span_from_items(span_text, element_items, source=f"{element.id}:element")
            if span is not None:
                spans.append(span)
    return items, spans


def _table_pdf_text_layer(
    payload: dict[str, Any],
    element: ExportElement,
    page: ExportPage,
    *,
    dpi: int,
) -> tuple[list[PdfTextItem], list[PdfTextSpan]]:
    page_height_px = int(page.size.get("h") or 1)
    items: list[PdfTextItem] = []
    spans: list[PdfTextSpan] = []
    payload_cell_items, payload_cell_spans = _table_text_layer_from_payload(
        payload,
        element,
        page_height_px=page_height_px,
        dpi=dpi,
    )
    if payload_cell_items:
        return payload_cell_items, payload_cell_spans

    html_rows = _table_rows_from_html(str(payload.get("text") or ""))
    if html_rows:
        bbox = element.bbox if isinstance(element.bbox, dict) else None
        table_cells, table_row_count, table_col_count = _table_cells_from_html(str(payload.get("text") or ""))
        if table_cells:
            inferred_cell_bboxes = _infer_table_cell_bboxes_from_image(
                page,
                element,
                table_cells,
                row_count=table_row_count,
                col_count=table_col_count,
            )
            for cell_idx, cell in enumerate(table_cells):
                cell_bbox = inferred_cell_bboxes.get(cell_idx) or _split_bbox_cell(
                    bbox,
                    row_count=table_row_count,
                    col_count=table_col_count,
                    row=cell.row,
                    col=cell.col,
                    row_span=cell.row_span,
                    col_span=cell.col_span,
                )
                cell_items = _single_text_item(
                    cell.text,
                    cell_bbox,
                    page_height_px=page_height_px,
                    source=f"{element.id}:table:html-cell:{cell_idx}",
                    dpi=dpi,
                    bbox_source="export_table",
                    bbox_granularity="table_formula_cell" if _is_formula_cell_text(cell.text) else "table_cell",
                )
                items.extend(cell_items)
                span_text = cell_items[0].text if cell_items else cell.text
                span = _span_from_items(
                    span_text,
                    cell_items,
                    source=f"{element.id}:table:html-cell:{cell_idx}",
                )
                if span is not None:
                    spans.append(span)
            return items, spans

        row_bboxes = _split_bbox_rows(bbox, len(html_rows))
        for row_idx, row_text in enumerate(html_rows):
            row_items = _single_text_item(
                row_text,
                row_bboxes[row_idx],
                page_height_px=page_height_px,
                source=f"{element.id}:table:html-row:{row_idx}",
                dpi=dpi,
                bbox_source="export_table",
                bbox_granularity="table_row",
            )
            items.extend(row_items)
            span_text = row_items[0].text if row_items else row_text
            span = _span_from_items(
                span_text,
                row_items,
                source=f"{element.id}:table:html-row:{row_idx}",
            )
            if span is not None:
                spans.append(span)
        return items, spans

    lines = payload.get("lines")
    if isinstance(lines, list):
        for line_idx, line in enumerate(lines):
            if not isinstance(line, dict):
                continue
            line_text = str(line.get("text") or "")
            row_items = _single_text_item(
                line_text,
                line.get("bbox") if isinstance(line.get("bbox"), dict) else None,
                page_height_px=page_height_px,
                source=f"{element.id}:table:line:{line_idx}",
                dpi=dpi,
                bbox_source="export_table",
                bbox_granularity="table_row",
            )
            items.extend(row_items)
            span_text = row_items[0].text if row_items else line_text
            span = _span_from_items(
                span_text,
                row_items,
                source=f"{element.id}:table:line:{line_idx}",
            )
            if span is not None:
                spans.append(span)
    if items:
        return items, spans

    table_text = str(payload.get("text") or "")
    items = _single_text_item(
        table_text,
        element.bbox,
        page_height_px=page_height_px,
        source=f"{element.id}:table",
        dpi=dpi,
        bbox_source="export_table",
        bbox_granularity="table",
    )
    span = _span_from_items(items[0].text if items else table_text, items, source=f"{element.id}:table")
    return items, [span] if span is not None else []


def _table_text_layer_from_payload(
    payload: dict[str, Any],
    element: ExportElement,
    *,
    page_height_px: int,
    dpi: int,
) -> tuple[list[PdfTextItem], list[PdfTextSpan]]:
    raw_cells = payload.get(TABLE_TEXT_LAYER_CELLS_KEY)
    if not isinstance(raw_cells, list):
        return [], []
    items: list[PdfTextItem] = []
    spans: list[PdfTextSpan] = []
    for idx, raw in enumerate(raw_cells):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "")
        bbox = raw.get("bbox")
        if not text or not isinstance(bbox, dict):
            continue
        granularity = str(raw.get("bbox_granularity") or "table_cell")
        cell_items = _single_text_item(
            text,
            bbox,
            page_height_px=page_height_px,
            source=f"{element.id}:table:payload-cell:{idx}",
            dpi=dpi,
            bbox_source=str(raw.get("bbox_source") or "table_text_layer"),
            bbox_granularity=granularity,
        )
        items.extend(cell_items)
        span = _span_from_items(
            cell_items[0].text if cell_items else text,
            cell_items,
            source=f"{element.id}:table:payload-cell:{idx}",
        )
        if span is not None:
            spans.append(span)
    return items, spans


class _TableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_TableCell]] = []
        self._current_row: list[_TableCell] | None = None
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
            self._current_cell_row_span = _positive_int(attrs_dict.get("rowspan"), default=1)
            self._current_cell_col_span = _positive_int(attrs_dict.get("colspan"), default=1)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell_text is not None:
            cell_text = _normalize_table_cell_text("".join(self._current_cell_text))
            if cell_text:
                self._current_row.append(_TableCell(
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


def _table_rows_from_html(text: str) -> list[str]:
    raw = str(text or "").strip()
    if "<tr" not in raw.lower() and "<table" not in raw.lower():
        return []
    parser = _TableTextParser()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        return []
    rows: list[str] = []
    for row in parser.rows:
        cells = [_normalize_table_cell_text(cell.text) for cell in row]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append("  ".join(cells))
    return rows


def _table_cells_from_html(text: str) -> tuple[list[_PositionedTableCell], int, int]:
    raw = str(text or "").strip()
    if "<tr" not in raw.lower() and "<table" not in raw.lower():
        return [], 0, 0
    parser = _TableTextParser()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        return [], 0, 0
    return _position_table_cells(parser.rows)


def _position_table_cells(rows: list[list[_TableCell]]) -> tuple[list[_PositionedTableCell], int, int]:
    positioned: list[_PositionedTableCell] = []
    occupied: set[tuple[int, int]] = set()
    max_col = 0
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row:
            while (row_idx, col_idx) in occupied:
                col_idx += 1
            row_span = max(1, int(cell.row_span))
            col_span = max(1, int(cell.col_span))
            positioned.append(_PositionedTableCell(
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


def _normalize_table_cell_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _split_bbox_rows(bbox: dict[str, Any] | None, row_count: int) -> list[dict[str, float] | None]:
    if row_count <= 0:
        return []
    if not isinstance(bbox, dict):
        return [None] * row_count
    x = float(bbox.get("x") or 0)
    y = float(bbox.get("y") or 0)
    w = max(0.0, float(bbox.get("w") or 0))
    h = max(0.0, float(bbox.get("h") or 0))
    if w <= 0 or h <= 0:
        return [None] * row_count
    row_h = h / row_count
    return [
        {"x": x, "y": y + idx * row_h, "w": w, "h": row_h}
        for idx in range(row_count)
    ]


def _split_bbox_cell(
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


def _infer_table_cell_bboxes_from_image(
    page: ExportPage,
    element: ExportElement,
    cells: list[_PositionedTableCell],
    *,
    row_count: int,
    col_count: int,
) -> dict[int, dict[str, float]]:
    if row_count <= 0 or col_count <= 0 or not cells or not isinstance(element.bbox, dict):
        return {}
    image_path = _resolve_page_image_path(page.source_image)
    if not image_path.exists():
        return {}
    try:
        import cv2
        import numpy as np
    except Exception:
        return {}

    full_image = read_cv_image(image_path, cv2.IMREAD_GRAYSCALE)
    if full_image is None:
        return {}
    page_height, page_width = full_image.shape[:2]
    left, top, right, bottom = _bbox_dict_to_xyxy(element.bbox, page_width, page_height)
    if right <= left or bottom <= top:
        return {}
    crop = full_image[top:bottom, left:right]
    if crop.size == 0:
        return {}

    clusters_by_row = _table_text_clusters(crop, cv2, np)
    if len(clusters_by_row) < max(2, row_count // 2):
        return {}
    if len(clusters_by_row) < row_count:
        return {}
    clusters_by_row = clusters_by_row[:row_count]

    column_centers = _table_column_centers(clusters_by_row, col_count)
    if len(column_centers) != col_count:
        return {}
    row_bounds = _bounds_from_centers([row["center_y"] for row in clusters_by_row], crop.shape[0])
    col_bounds = _bounds_from_centers(column_centers, crop.shape[1])

    all_clusters: list[dict[str, float]] = []
    for row in clusters_by_row:
        all_clusters.extend(row["clusters"])

    result: dict[int, dict[str, float]] = {}
    for idx, cell in enumerate(cells):
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
            if _point_in_rect(cluster["cx"], cluster["cy"], search, pad_x=8.0, pad_y=8.0)
        ]
        if not matches:
            continue
        bbox = _union_cluster_bboxes(matches, page_left=left, page_top=top)
        if bbox is not None:
            result[idx] = bbox
    return result


def _bbox_dict_to_xyxy(bbox: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    x = int(round(float(bbox.get("x") or 0)))
    y = int(round(float(bbox.get("y") or 0)))
    w = int(round(float(bbox.get("w") or 0)))
    h = int(round(float(bbox.get("h") or 0)))
    left = max(0, min(width, x))
    top = max(0, min(height, y))
    right = max(left, min(width, x + w))
    bottom = max(top, min(height, y + h))
    return left, top, right, bottom


def _table_text_clusters(crop, cv2, np) -> list[dict[str, Any]]:
    blurred = cv2.GaussianBlur(crop, (3, 3), 0)
    _threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    width = crop.shape[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 12), 1))
    horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    text_mask = cv2.subtract(mask, horizontal_lines)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    projection_y = (text_mask > 0).sum(axis=1)
    threshold_y = max(2, int(width * 0.002))
    raw_bands = _projection_bands(projection_y, threshold_y, max_gap=8, min_size=5)
    rows: list[dict[str, Any]] = []
    for y1, y2 in raw_bands:
        row_mask = text_mask[y1:y2, :]
        projection_x = (row_mask > 0).sum(axis=0)
        threshold_x = max(1, int(max(1, y2 - y1) * 0.06))
        x_bands = _projection_bands(projection_x, threshold_x, max_gap=35, min_size=3)
        clusters: list[dict[str, float]] = []
        for x1, x2 in x_bands:
            if x2 <= x1:
                continue
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


def _projection_bands(values, threshold: int, *, max_gap: int, min_size: int) -> list[tuple[int, int]]:
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


def _table_column_centers(rows: list[dict[str, Any]], col_count: int) -> list[float]:
    candidates = [row["clusters"] for row in rows if len(row["clusters"]) == col_count]
    if not candidates:
        return []
    # Prefer lower rows with full numeric columns; headings with colspan are less
    # reliable for column centers.
    clusters = max(candidates, key=lambda item: sum(cluster["cy"] for cluster in item) / len(item))
    return [float(cluster["cx"]) for cluster in clusters]


def _bounds_from_centers(centers: list[float], limit: int) -> list[float]:
    if not centers:
        return [0.0, float(limit)]
    ordered = sorted(float(center) for center in centers)
    bounds = [0.0]
    for prev, current in zip(ordered, ordered[1:]):
        bounds.append((prev + current) / 2.0)
    bounds.append(float(limit))
    return bounds


def _point_in_rect(x: float, y: float, rect: tuple[float, float, float, float], *, pad_x: float, pad_y: float) -> bool:
    left, top, right, bottom = rect
    return left - pad_x <= x <= right + pad_x and top - pad_y <= y <= bottom + pad_y


def _union_cluster_bboxes(clusters: list[dict[str, float]], *, page_left: int, page_top: int) -> dict[str, float] | None:
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


def _is_formula_cell_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if value.startswith("$") and value.endswith("$"):
        return True
    return value.startswith("\\(") and value.endswith("\\)")


def _line_pdf_text_layer(
    line: dict[str, Any],
    page: ExportPage,
    *,
    line_idx: int,
    dpi: int,
    element_kind: str = "",
    element_bbox: dict[str, Any] | None = None,
) -> tuple[list[PdfTextItem], list[PdfTextSpan]]:
    line_text = str(line.get("text") or "")
    if element_kind == "equation":
        line_text = _formula_pdf_text(line_text)
        bbox = line.get("bbox") if isinstance(line.get("bbox"), dict) else element_bbox
        items = _single_text_item(
            line_text,
            bbox,
            page_height_px=int(page.size.get("h") or 1),
            source=f"line:{line_idx}:equation",
            dpi=dpi,
            bbox_source="export_equation",
            bbox_granularity="formula",
        )
        span = _span_from_items(line_text, items, source=f"line:{line_idx}:equation")
        return items, [span] if span is not None else []

    chars = line.get("chars")
    if isinstance(chars, list) and chars:
        items: list[PdfTextItem] = []
        for char_idx, char_data in enumerate(chars):
            if not isinstance(char_data, dict):
                continue
            text = str(char_data.get("char") or "")
            bbox = char_data.get("bbox")
            if not text or text == "\n" or not isinstance(bbox, dict):
                continue
            rect = pixel_bbox_to_pdf_rect(bbox, int(page.size.get("h") or 1), dpi=dpi)
            items.append(PdfTextItem(
                text=text,
                x=rect[0],
                y=rect[1],
                w=rect[2],
                h=rect[3],
                source=f"line:{line_idx}:char:{char_idx}",
                bbox_source=str(char_data.get("bbox_source") or ""),
                bbox_granularity=str(char_data.get("bbox_granularity") or ""),
            ))
        if items:
            return items, _spans_from_positioned_items(items, source=f"line:{line_idx}:chars")
    items = _synthetic_text_items(
        line_text,
        line.get("bbox"),
        page_height_px=int(page.size.get("h") or 1),
        source=f"line:{line_idx}:synthetic",
        dpi=dpi,
    )
    span = _span_from_items(line_text, items, source=f"line:{line_idx}:synthetic")
    return items, [span] if span is not None else []


def _inline_formula_bboxes_from_page(page: ExportPage) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for element in page.elements:
        if element.kind == "equation":
            continue
        payload = element.payload if isinstance(element.payload, dict) else {}
        lines = payload.get("lines")
        if not isinstance(lines, list):
            continue
        for line in lines:
            if not isinstance(line, dict):
                continue
            chars = line.get("chars")
            if not isinstance(chars, list):
                continue
            for char_data in chars:
                if not isinstance(char_data, dict):
                    continue
                if not _is_formula_char_payload(char_data):
                    continue
                bbox = char_data.get("bbox")
                if isinstance(bbox, dict):
                    values.append(bbox)
    return values


def _is_formula_char_payload(char_data: dict[str, Any]) -> bool:
    marker = f"{char_data.get('bbox_source') or ''} {char_data.get('bbox_granularity') or ''}".lower()
    return "formula" in marker or "equation" in marker


def _duplicates_inline_formula_element(element: ExportElement, inline_formula_bboxes: list[dict[str, Any]]) -> bool:
    if not inline_formula_bboxes or not isinstance(element.bbox, dict):
        return False
    if not _is_inline_formula_element(element):
        return False
    for bbox in inline_formula_bboxes:
        if _bbox_overlap_ratio(element.bbox, bbox) >= 0.70:
            return True
    return False


def _is_inline_formula_element(element: ExportElement) -> bool:
    markers = [
        str(element.source.source_label or ""),
        str(element.source.semantic_label or ""),
        str(element.layout_attributes.get("semantic_label") or ""),
        str(element.layout_attributes.get("source_label") or ""),
    ]
    return any("inline_formula" in marker.lower() for marker in markers)


def _bbox_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_to_xyxy_unclamped(a)
    bx1, by1, bx2, by2 = _bbox_to_xyxy_unclamped(b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = min(area_a, area_b)
    return inter / denom if denom > 0 else 0.0


def _bbox_to_xyxy_unclamped(bbox: dict[str, Any]) -> tuple[float, float, float, float]:
    x = float(bbox.get("x") or 0)
    y = float(bbox.get("y") or 0)
    w = max(0.0, float(bbox.get("w") or 0))
    h = max(0.0, float(bbox.get("h") or 0))
    return x, y, x + w, y + h


def _formula_pdf_text(text: str) -> str:
    value = _single_line_pdf_text(text)
    spans = [match.group(0).strip() for match in _FORMULA_SPAN_PATTERN.finditer(value)]
    if len(spans) <= 1:
        return value
    remainder = _FORMULA_SPAN_PATTERN.sub("", value).strip()
    if remainder:
        return value
    first_key = _compact_formula_text(spans[0])
    if first_key and all(_compact_formula_text(span) == first_key for span in spans[1:]):
        return spans[0]
    return value


def _compact_formula_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _single_text_item(
    text: str,
    bbox: dict[str, Any] | None,
    *,
    page_height_px: int,
    source: str,
    dpi: int,
    bbox_source: str = "",
    bbox_granularity: str = "",
) -> list[PdfTextItem]:
    if not text or not isinstance(bbox, dict):
        return []
    rect = pixel_bbox_to_pdf_rect(bbox, page_height_px, dpi=dpi)
    if rect[2] <= 0 or rect[3] <= 0:
        return []
    return [PdfTextItem(
        text=_single_line_pdf_text(text),
        x=rect[0],
        y=rect[1],
        w=rect[2],
        h=rect[3],
        source=source,
        bbox_source=bbox_source,
        bbox_granularity=bbox_granularity,
    )]


def _single_line_pdf_text(text: str) -> str:
    return " ".join(str(text).replace("\r", "\n").split("\n"))


def _spans_from_positioned_items(items: list[PdfTextItem], *, source: str) -> list[PdfTextSpan]:
    spans: list[PdfTextSpan] = []
    pending: list[PdfTextItem] = []

    def flush_pending() -> None:
        if not pending:
            return
        span = _span_from_items("".join(item.text for item in pending), pending, source=f"{source}:text")
        if span is not None:
            spans.append(span)
        pending.clear()

    for index, item in enumerate(items):
        if _is_atomic_text_item(item):
            flush_pending()
            span = _span_from_items(item.text, [item], source=f"{source}:atom:{index}")
            if span is not None:
                spans.append(span)
            continue
        pending.append(item)
    flush_pending()
    return spans


def _synthetic_text_items(
    text: str,
    bbox: dict[str, Any] | None,
    *,
    page_height_px: int,
    source: str,
    dpi: int,
) -> list[PdfTextItem]:
    if not text or not isinstance(bbox, dict):
        return []
    chars = [ch for ch in text if ch != "\n"]
    if not chars:
        return []
    x = float(bbox.get("x") or 0)
    y = float(bbox.get("y") or 0)
    w = max(0.0, float(bbox.get("w") or 0))
    h = max(0.0, float(bbox.get("h") or 0))
    if w <= 0 or h <= 0:
        return []
    char_w = w / len(chars)
    items: list[PdfTextItem] = []
    for idx, ch in enumerate(chars):
        char_bbox = {"x": x + idx * char_w, "y": y, "w": char_w, "h": h}
        rect = pixel_bbox_to_pdf_rect(char_bbox, page_height_px, dpi=dpi)
        items.append(PdfTextItem(
            text=ch,
            x=rect[0],
            y=rect[1],
            w=rect[2],
            h=rect[3],
            source=f"{source}:{idx}",
        ))
    return items


def _span_from_items(text: str, items: list[PdfTextItem], *, source: str) -> PdfTextSpan | None:
    if not text or not items:
        return None
    x1 = min(item.x for item in items)
    y1 = min(item.y for item in items)
    x2 = max(item.x + item.w for item in items)
    y2 = max(item.y + item.h for item in items)
    return PdfTextSpan(
        text=text.replace("\n", ""),
        x=x1,
        y=y1,
        w=x2 - x1,
        h=y2 - y1,
        source=source,
        items=tuple(items),
    )


def _add_page_with_size(pdf, plan: PdfPagePlan) -> None:
    try:
        pdf.add_page(format=(plan.width_pt, plan.height_pt))
    except TypeError:
        pdf.add_page()


def _write_page_image(pdf, plan: PdfPagePlan) -> None:
    image_path = _resolve_page_image_path(plan.image_path)
    if not image_path.exists():
        logger.warning("PDF page image missing for page %s: %s", plan.page_number, image_path)
        return
    try:
        pdf.image(str(image_path), x=0, y=0, w=plan.width_pt, h=plan.height_pt)
    except Exception as e:
        logger.warning("PDF page image layer failed for %s: %s", image_path, e)


def _resolve_page_image_path(raw_path: str) -> Path:
    image_path = Path(raw_path)
    if image_path.exists():
        return image_path
    cache_candidate = Path(__file__).parent.parent.parent / ".cache" / "images" / image_path.name
    if cache_candidate.exists():
        return cache_candidate
    return image_path


def _write_invisible_text_layer(pdf, plan: PdfPagePlan) -> None:
    pdf.set_text_color(0, 0, 0)
    try:
        _set_text_rendering_mode(pdf, 3)
        default_font_size = _page_text_font_size(plan)
        for span in plan.text_spans:
            try:
                span_font_size = _span_font_size(plan, span, default_font_size)
                pdf.set_font("CJK", size=span_font_size)
                _write_positioned_span_text(pdf, plan, span, span_font_size)
            except Exception as e:
                logger.warning("PDF invisible text write failed for %s: %s", span.source, e)
    finally:
        _reset_text_stretching(pdf)
        _set_text_rendering_mode(pdf, 0)


def _page_text_font_size(plan: PdfPagePlan) -> float:
    font_items = _body_text_items(plan.text_items) or plan.text_items
    heights = sorted(item.h for item in font_items if item.h > 0)
    if not heights:
        return 1.0
    mid = len(heights) // 2
    median = heights[mid] if len(heights) % 2 else (heights[mid - 1] + heights[mid]) / 2.0
    return max(1.0, min(72.0, median * plan.text_font_size_to_bbox_ratio))


def _span_font_size(plan: PdfPagePlan, span: PdfTextSpan, default_font_size: float) -> float:
    if _is_atomic_span(span):
        item = span.items[0]
        return max(1.0, min(72.0, item.h * plan.text_font_size_to_bbox_ratio))
    return default_font_size


def _span_baseline_top_y(plan: PdfPagePlan, span: PdfTextSpan, font_size: float) -> float:
    baseline_items = _body_text_items(span.items) or list(span.items)
    if baseline_items:
        y2 = _median_float(item.y + item.h for item in baseline_items)
    else:
        y2 = span.y + span.h
    baseline_pdf_y = y2 - font_size * PDF_TEXT_ASCENDER_RATIO
    return plan.height_pt - baseline_pdf_y


def _median_float(values) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _body_text_items(items: list[PdfTextItem] | tuple[PdfTextItem, ...]) -> list[PdfTextItem]:
    return [item for item in items if not _is_formula_text_item(item) and not _is_table_text_item(item)]


def _is_formula_text_item(item: PdfTextItem) -> bool:
    marker = f"{item.bbox_source} {item.bbox_granularity}".lower()
    return "formula" in marker or "equation" in marker


def _is_table_text_item(item: PdfTextItem) -> bool:
    marker = f"{item.bbox_source} {item.bbox_granularity}".lower()
    return "table" in marker


def _is_atomic_text_item(item: PdfTextItem) -> bool:
    granularity = (item.bbox_granularity or "").strip().lower()
    return _is_formula_text_item(item) or _is_table_text_item(item) or (granularity == "word" and len(item.text) > 1)


def _is_atomic_span(span: PdfTextSpan) -> bool:
    return len(span.items) == 1 and _is_atomic_text_item(span.items[0])


def _write_span_text(pdf, plan: PdfPagePlan, span: PdfTextSpan, font_size: float) -> None:
    stretching = _span_text_stretching(pdf, span)
    try:
        _set_text_stretching(pdf, stretching)
        _write_text(pdf, span.x, _span_baseline_top_y(plan, span, font_size), span.text)
    finally:
        _reset_text_stretching(pdf)


def _span_text_stretching(pdf, span: PdfTextSpan) -> float:
    if span.w <= 0 or not span.text:
        return 100.0
    try:
        natural_width = float(pdf.get_string_width(span.text))
    except Exception:
        return 100.0
    if natural_width <= 0:
        return 100.0
    # Keep the searchable text's extracted bbox close to the OCR line bbox.
    # Without this, fpdf writes the line at natural font width and the hidden
    # text layer is visibly shorter than the scanned line image.
    min_stretching = 1.0 if _is_atomic_span(span) else 10.0
    return max(min_stretching, min(1000.0, span.w / natural_width * 100.0))


def _write_positioned_span_text(pdf, plan: PdfPagePlan, span: PdfTextSpan, font_size: float) -> None:
    items = [item for item in span.items if item.text and item.text != "\n"]
    if len(items) < 2 or not _supports_low_level_text_array(pdf):
        _write_span_text(pdf, plan, span, font_size)
        return

    try:
        content = _char_advance_text_object(pdf, plan, span, items, font_size)
    except Exception:
        _write_span_text(pdf, plan, span, font_size)
        return
    pdf._out(content)


def _supports_low_level_text_array(pdf) -> bool:
    current_font = getattr(pdf, "current_font", None)
    return (
        current_font is not None
        and hasattr(current_font, "encode_text")
        and hasattr(pdf, "normalize_text")
        and hasattr(pdf, "_out")
    )


def _char_advance_text_object(
    pdf,
    plan: PdfPagePlan,
    span: PdfTextSpan,
    items: list[PdfTextItem],
    font_size: float,
) -> str:
    _reset_text_stretching(pdf)
    _ensure_current_font_on_page(pdf)
    baseline_y = plan.height_pt - _span_baseline_top_y(plan, span, font_size)
    ops: list[str] = []
    for idx, item in enumerate(items):
        advance = _target_item_advance(item, items[idx + 1] if idx + 1 < len(items) else None)
        literal = _encoded_text_literal(pdf, item.text)
        natural_width = float(pdf.get_string_width(item.text))
        stretching = 100.0 if natural_width <= 0 else advance / natural_width * 100.0
        ops.append(f"{_clamp_text_stretching(stretching):.3f} Tz {literal} Tj")
    ops.append("100 Tz")
    return f"BT {span.x:.2f} {baseline_y:.2f} Td {' '.join(ops)} ET"


def _target_item_advance(item: PdfTextItem, next_item: PdfTextItem | None) -> float:
    if next_item is not None:
        advance = next_item.x - item.x
        if advance > 0:
            return advance
    return max(0.1, item.w)


def _encoded_text_literal(pdf, text: str) -> str:
    encoded = pdf.current_font.encode_text(pdf.normalize_text(text))
    suffix = " Tj"
    if not encoded.endswith(suffix):
        raise RuntimeError("unsupported encoded PDF text literal")
    return encoded[: -len(suffix)].strip()


def _clamp_text_stretching(stretching: float) -> float:
    return max(10.0, min(1000.0, stretching))


def _ensure_current_font_on_page(pdf) -> None:
    if getattr(pdf, "current_font_is_set_on_page", True):
        return
    pdf._out(pdf._set_font_for_page(pdf.current_font, pdf.font_size_pt))


def _set_text_stretching(pdf, stretching: float) -> None:
    if hasattr(pdf, "set_stretching"):
        pdf.set_stretching(stretching)


def _reset_text_stretching(pdf) -> None:
    _set_text_stretching(pdf, 100.0)


def _set_text_rendering_mode(pdf, mode: int) -> None:
    pdf._out(f"{int(mode)} Tr")


def _write_text(pdf, x: float, y: float, text: str) -> None:
    try:
        pdf.text(x, y, text=text)
    except TypeError:
        pdf.text(x, y, txt=text)
