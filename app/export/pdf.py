"""PDF 导出：fpdf2 + 系统/内置 CJK 字体。"""
from pathlib import Path

from app.export.base import ExporterBase
from app.core.logging import get_logger
from app.models import OcrProject
from app.services.export_service import (
    get_block_style,
    get_export_text,
    iter_export_blocks,
    iter_export_lines,
    iter_export_pages,
)

_RESOURCES_FONTS = Path(__file__).parent.parent.parent / "resources" / "fonts"
logger = get_logger(__name__)

# 字体搜索顺序：项目内置优先，其次 Windows/Linux 系统字体
_FONT_CANDIDATES = [
    _RESOURCES_FONTS / "NotoSansSC-Regular.ttf",
    _RESOURCES_FONTS / "DroidSansFallbackFull.ttf",
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simsun.ttc"),
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
        pdf.set_fallback_fonts(["LatinFallback"])
    except Exception as e:
        logger.warning("PDF Latin fallback font setup failed: %s", e)
        return


def _write_cell(pdf, text: str, height: int) -> None:
    try:
        pdf.cell(0, height, text=text, new_x="LMARGIN", new_y="NEXT")
    except TypeError:
        pdf.cell(0, height, txt=text, ln=1)


def _write_multicell(pdf, text: str, height: int, align: str = "L") -> None:
    try:
        pdf.multi_cell(0, height, text=text, align=align)
    except TypeError:
        pdf.multi_cell(0, height, txt=text, align=align)


class PdfExporter(ExporterBase):

    def export(self, project: OcrProject, out_path: str) -> None:
        from fpdf import FPDF

        font_path = _find_font()
        if font_path is None:
            raise RuntimeError(
                "未找到 CJK 字体文件，无法导出 PDF。\n"
                "请将 NotoSansSC-Regular.ttf 放入 resources/fonts/ 目录，"
                "或在 Windows 系统上运行（使用系统 msyh.ttc）。"
            )

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        _add_font(pdf, font_path)
        _add_latin_fallback_font(pdf)

        for page in iter_export_pages(project):
            pdf.add_page()
            pdf.set_font("CJK", size=14)
            _write_cell(pdf, f"第 {page.page_number} 页", 10)
            pdf.ln(4)
            for block in iter_export_blocks(page):
                style = get_block_style(block)
                pdf.set_font("CJK", size=style.font_size_pt)
                align = "C" if block.block_type.value == "equation" else "L"
                for line in iter_export_lines(block):
                    _write_multicell(pdf, get_export_text(line), style.line_height_mm, align=align)
                pdf.ln(4)

        pdf.output(out_path)
