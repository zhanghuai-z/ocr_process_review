"""RTF 导出：手写 RTF，无需第三方库。支持 UTF-16 转义。"""
from app.export.base import ExporterBase
from app.export.ir_builder import build_export_ir
from app.export.rendering import RichReflowBlock, rich_reflow_pages
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
        for page, blocks in rich_reflow_pages(document):
            parts.append(r"\pard\sb200\b " + _rtf_escape(f"第 {page.page_number} 页") + r"\b0\par")
            for block in blocks:
                prefix, suffix = _rtf_style(block)
                for text in block.lines:
                    parts.append(prefix + _rtf_escape(text) + suffix)
                parts.append(r"\pard\par")
        parts.append("}")

        with open(out_path, "w", encoding="ascii", errors="replace") as f:
            f.write("".join(parts))


def _rtf_style(block: RichReflowBlock) -> tuple[str, str]:
    if block.role == "heading":
        return r"\pard\sb120\b\fs32 ", r"\b0\fs24\par"
    if block.role == "caption":
        return r"\pard\i\fs20 ", r"\i0\fs24\par"
    if block.role == "equation":
        return r"\pard\qc ", r"\par"
    if block.role == "reference":
        return r"\pard\fs22 ", r"\fs24\par"
    return r"\pard ", r"\par"
