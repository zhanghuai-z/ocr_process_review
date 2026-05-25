"""RTF 导出：手写 RTF，无需第三方库。支持 UTF-16 转义。"""
from app.export.base import ExporterBase
from app.export.ir_builder import build_export_ir
from app.export.rendering import element_lines
from app.models import OcrProject


def _rtf_escape(text: str) -> str:
    """将 Unicode 字符转为 RTF \\uN? 转义序列。"""
    out = []
    for ch in text:
        code = ord(ch)
        if code < 128:
            if ch in ("\\", "{", "}"):
                out.append(f"\\{ch}")
            else:
                out.append(ch)
        else:
            # RTF Unicode: \uN? — N 是有符号 16-bit 值
            signed = code if code <= 32767 else code - 65536
            out.append(f"\\u{signed}?")
    return "".join(out)


class RtfExporter(ExporterBase):

    def export(self, project: OcrProject, out_path: str) -> None:
        document = build_export_ir(project, "rtf")
        parts = [
            r"{\rtf1\ansi\ansicpg936\deff0"
            r"{\fonttbl{\f0\fnil\fcharset134 SimSun;}}"
            r"{\colortbl;\red0\green0\blue0;}"
            r"\widowctrl\wpaper12240\wpapr15840\margl1800\margr1800\margt1440\margb1440"
            r"\f0\fs24\cf1 "
        ]
        for page in document.pages:
            parts.append(r"\pard\sb200\b " + _rtf_escape(f"第 {page.page_number} 页") + r"\b0\par")
            for element in page.elements:
                prefix = r"\pard "
                suffix = r"\par"
                if element.kind == "title":
                    prefix = r"\pard\sb120\b\fs32 "
                    suffix = r"\b0\fs24\par"
                elif element.kind in {"figure_caption", "table_caption"}:
                    prefix = r"\pard\i\fs20 "
                    suffix = r"\i0\fs24\par"
                elif element.kind == "equation":
                    prefix = r"\pard\qc "
                    suffix = r"\par"
                for text in element_lines(element):
                    parts.append(prefix + _rtf_escape(text) + suffix)
                parts.append(r"\pard\par")
        parts.append("}")

        with open(out_path, "w", encoding="ascii", errors="replace") as f:
            f.write("".join(parts))
