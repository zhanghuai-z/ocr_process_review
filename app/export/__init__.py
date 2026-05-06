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
            return PdfExporter()
        case "xml":
            from app.export.xml import XmlExporter
            return XmlExporter()
        case "html":
            from app.export.html import HtmlExporter
            return HtmlExporter()
        case "docx":
            from app.export.docx_exporter import DocxExporter
            return DocxExporter()
        case _:
            raise ValueError(f"不支持的导出格式: {fmt}")


__all__ = ["get_exporter", "ExporterBase"]
