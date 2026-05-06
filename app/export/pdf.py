"""PDF 导出：fpdf2 + 系统/内置 CJK 字体。"""
from pathlib import Path

from app.export.base import ExporterBase
from app.models import OcrProject

_RESOURCES_FONTS = Path(__file__).parent.parent.parent / "resources" / "fonts"

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


def _find_font() -> str | None:
    for p in _FONT_CANDIDATES:
        if p.exists():
            return str(p)
    return None


class PdfExporter(ExporterBase):

    def export(self, project: OcrProject, out_path: str) -> None:
        from fpdf import FPDF, XPos, YPos

        font_path = _find_font()
        if font_path is None:
            raise RuntimeError(
                "未找到 CJK 字体文件，无法导出 PDF。\n"
                "请将 NotoSansSC-Regular.ttf 放入 resources/fonts/ 目录，"
                "或在 Windows 系统上运行（使用系统 msyh.ttc）。"
            )

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        # fpdf2 2.x 自动支持 Unicode，无需 uni=True 参数
        pdf.add_font("CJK", "", font_path)

        for page in project.pages:
            pdf.add_page()
            pdf.set_font("CJK", size=14)
            pdf.cell(0, 10, text=f"第 {page.page_number} 页",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("CJK", size=12)
            pdf.ln(4)
            for block in page.text_blocks:
                for line in block.lines:
                    pdf.multi_cell(0, 8, text=line.text)
                pdf.ln(4)

        pdf.output(out_path)
