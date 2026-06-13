"""PDF 导出：fpdf2 + 系统/内置 CJK 字体。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.export.base import ExporterBase
from app.core.logging import get_logger
from app.export.ir import ExportDocument, ExportPage
from app.export.ir_builder import build_export_ir
from app.models import OcrProject

_RESOURCES_FONTS = Path(__file__).parent.parent.parent / "resources" / "fonts"
logger = get_logger(__name__)
PDF_DEFAULT_DPI = 300
# fpdf/PyMuPDF expose the embedded CJK font bbox with ~1.043 ascender.
# Use it to convert OCR bbox top/bottom into a text baseline instead of
# incorrectly treating the bbox bottom as the baseline itself.
PDF_TEXT_ASCENDER_RATIO = 1.043

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

    @property
    def font_size(self) -> float:
        return max(1.0, min(72.0, self.h * 0.95))


@dataclass(frozen=True)
class PdfTextSpan:
    text: str
    x: float
    y: float
    w: float
    h: float
    source: str = ""


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


class PdfExporter(ExporterBase):
    def __init__(self, profile: str = "pdf-single"):
        self.profile = "pdf-single" if profile == "pdf" else profile

    def export(self, project: OcrProject, out_path: str) -> None:
        from fpdf import FPDF

        document = build_export_ir(project, self.profile)
        include_text = document.profile.format == "pdf-dual"
        plans = build_pdf_page_plans(document, include_text=include_text, dpi=_document_dpi(document))
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
) -> list[PdfPagePlan]:
    plans: list[PdfPagePlan] = []
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


def _collect_pdf_text_layer(page: ExportPage, *, dpi: int) -> tuple[list[PdfTextItem], list[PdfTextSpan]]:
    items: list[PdfTextItem] = []
    spans: list[PdfTextSpan] = []
    for element in page.elements:
        payload = element.payload if isinstance(element.payload, dict) else {}
        lines = payload.get("lines")
        if isinstance(lines, list):
            for line_idx, line in enumerate(lines):
                if not isinstance(line, dict):
                    continue
                line_items, span = _line_pdf_text_layer(line, page, line_idx=line_idx, dpi=dpi)
                items.extend(line_items)
                if span is not None:
                    spans.append(span)
        elif isinstance(payload.get("text"), str):
            element_items = _synthetic_text_items(
                payload["text"],
                element.bbox,
                page_height_px=int(page.size.get("h") or 1),
                source=f"{element.id}:element",
                dpi=dpi,
            )
            items.extend(element_items)
            span = _span_from_items(payload["text"], element_items, source=f"{element.id}:element")
            if span is not None:
                spans.append(span)
    return items, spans


def _line_pdf_text_layer(
    line: dict[str, Any],
    page: ExportPage,
    *,
    line_idx: int,
    dpi: int,
) -> tuple[list[PdfTextItem], PdfTextSpan | None]:
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
            ))
        if items:
            span = _span_from_items(
                str(line.get("text") or "") or "".join(item.text for item in items),
                items,
                source=f"line:{line_idx}:chars",
            )
            return items, span
    items = _synthetic_text_items(
        str(line.get("text") or ""),
        line.get("bbox"),
        page_height_px=int(page.size.get("h") or 1),
        source=f"line:{line_idx}:synthetic",
        dpi=dpi,
    )
    return items, _span_from_items(str(line.get("text") or ""), items, source=f"line:{line_idx}:synthetic")


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
    return PdfTextSpan(text=text.replace("\n", ""), x=x1, y=y1, w=x2 - x1, h=y2 - y1, source=source)


def _add_page_with_size(pdf, plan: PdfPagePlan) -> None:
    try:
        pdf.add_page(format=(plan.width_pt, plan.height_pt))
    except TypeError:
        pdf.add_page()


def _write_page_image(pdf, plan: PdfPagePlan) -> None:
    image_path = Path(plan.image_path)
    if not image_path.exists():
        logger.warning("PDF page image missing for page %s: %s", plan.page_number, image_path)
        return
    try:
        pdf.image(str(image_path), x=0, y=0, w=plan.width_pt, h=plan.height_pt)
    except Exception as e:
        logger.warning("PDF page image layer failed for %s: %s", image_path, e)


def _write_invisible_text_layer(pdf, plan: PdfPagePlan) -> None:
    pdf.set_text_color(0, 0, 0)
    try:
        _set_text_rendering_mode(pdf, 3)
        font_size = _page_text_font_size(plan)
        pdf.set_font("CJK", size=font_size)
        for span in plan.text_spans:
            try:
                _write_span_text(pdf, plan, span, font_size)
            except Exception as e:
                logger.warning("PDF invisible text write failed for %s: %s", span.source, e)
    finally:
        _reset_text_stretching(pdf)
        _set_text_rendering_mode(pdf, 0)


def _page_text_font_size(plan: PdfPagePlan) -> float:
    heights = sorted(item.h for item in plan.text_items if item.h > 0)
    if not heights:
        return 1.0
    mid = len(heights) // 2
    median = heights[mid] if len(heights) % 2 else (heights[mid - 1] + heights[mid]) / 2.0
    return max(1.0, min(72.0, median * 0.95))


def _span_baseline_top_y(plan: PdfPagePlan, span: PdfTextSpan, font_size: float) -> float:
    baseline_pdf_y = span.y + span.h - font_size * PDF_TEXT_ASCENDER_RATIO
    return plan.height_pt - baseline_pdf_y


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
    return max(10.0, min(1000.0, span.w / natural_width * 100.0))


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
