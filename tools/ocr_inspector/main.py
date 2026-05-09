"""OCR Inspector entry point.

Usage:
    python -m tools.ocr_inspector [path/to/ocr_output.json]
    python tools/ocr_inspector/main.py [path/to/ocr_output.json]
"""
from __future__ import annotations
import sys
from PySide6.QtWidgets import QApplication
from tools.ocr_inspector.ui import OcrInspectorWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("OCR Inspector")
    window = OcrInspectorWindow()
    window.show()
    if len(sys.argv) > 1:
        window.load_file(sys.argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
