"""OCR Inspector — standalone launcher.

Source usage (from project root):
    python inspect_ocr.py [path/to/ocr_output.json]

This file also serves as the PyInstaller entry point for ocr_inspector.exe.
"""
from __future__ import annotations
import sys
import os

# Ensure project root is on sys.path when run as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tools.ocr_inspector.main import main

if __name__ == "__main__":
    sys.exit(main())
