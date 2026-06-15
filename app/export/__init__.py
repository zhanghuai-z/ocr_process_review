from app.export.base import ExporterBase
from app.models import OcrProject


def get_exporter(fmt: str) -> ExporterBase:
    """工厂函数：根据格式字符串返回对应导出器。"""
    match fmt.lower():
        case "txt":
            from app.export.txt import TxtExporter
            return TxtExporter()
        case "rtf":
            from app.export.rtf import RtfExporter
            return RtfExporter()
        case "pdf":
            from app.export.pdf import PdfExporter
            return PdfExporter("pdf-single")
        case "pdf-single":
            from app.export.pdf import PdfExporter
            return PdfExporter("pdf-single")
        case "pdf-dual":
            from app.export.pdf import PdfExporter
            return PdfExporter("pdf-dual")
        case "xml":
            from app.export.xml import XmlExporter
            return XmlExporter()
        case "html":
            from app.export.html import HtmlExporter
            return HtmlExporter()
        case "json":
            from app.export.json_exporter import JsonExporter
            return JsonExporter()
        case "md" | "markdown":
            from app.export.markdown import MarkdownExporter
            return MarkdownExporter()
        case "docx":
            from app.export.docx_exporter import DocxExporter
            return DocxExporter()
        case _:
            raise ValueError(f"不支持的导出格式: {fmt}")


__all__ = ["get_exporter", "ExporterBase"]
