# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for OCR Process
#
# Usage (Windows cmd/PowerShell):
#   pyinstaller ocr_process.spec --noconfirm
#
# Notes:
#   - onedir mode: dist/ocr_process/ is portable
#   - PaddleOCR model downloads to %%USERPROFILE%%\.paddleocr\ on first run
#   - UPX disabled by default; set upx=True after installing UPX

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = Path(SPECPATH)

# -- Data files -------------------------------------------------------
datas = []

# qt-material themes and resources
try:
    import qt_material
    qt_dir = Path(qt_material.__file__).parent
    for sub in ("themes", "resources"):
        d = qt_dir / sub
        if d.exists():
            datas.append((str(d), f"qt_material/{sub}"))
except ImportError:
    pass

# CJK fonts for fpdf2
fonts_dir = PROJECT_ROOT / "resources" / "fonts"
if fonts_dir.exists():
    for f in fonts_dir.iterdir():
        if f.suffix.lower() in (".ttf", ".ttc", ".otf"):
            datas.append((str(f), "resources/fonts"))

# Jinja2 / fpdf2 built-in data
datas += collect_data_files("jinja2")
datas += collect_data_files("fpdf")

# -- Hidden imports ---------------------------------------------------
hiddenimports = [
    # PySide6 core
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    # qt-material
    "qt_material",
    # Export dependencies
    "lxml",
    "lxml.etree",
    "lxml._elementpath",
    "lxml.builder",
    "fpdf",
    "fpdf.enums",
    "docx",
    "docx.oxml",
    "docx.oxml.ns",
    "docx.shared",
    "docx.enum.text",
    "jinja2",
    "jinja2.ext",
    # Image processing
    "PIL",
    "PIL.Image",
    "PIL.ImageQt",
    "cv2",
    "fitz",         # PyMuPDF
    # Core
    "sqlite3",
    "numpy",
    "numpy.core._multiarray_umath",
]

# Collect all app.* submodules
hiddenimports += collect_submodules("app")

# PaddleOCR (uncomment if needed, ~+1 GB)
# hiddenimports += collect_submodules("paddle")
# hiddenimports += collect_submodules("paddleocr")

# -- Analysis ---------------------------------------------------------
a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "scipy",
        "IPython",
        "notebook",
        "pytest",
        "unittest",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ocr_process",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "resources" / "icon.ico")
          if (PROJECT_ROOT / "resources" / "icon.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ocr_process",
)
