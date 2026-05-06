# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for OCR后处理
# 用法（在 Windows 的 cmd/PowerShell 下执行）:
#   pyinstaller ocr_process.spec --noconfirm --clean
#
# 说明：
#   - 使用 onedir 模式（dist/ocr_process/ 整个文件夹可复制分发）
#   - PaddleOCR 模型首次运行时自动下载到 %USERPROFILE%\.paddleocr\
#   - UPX 默认关闭（upx=False），安装 UPX 后可改为 True 减小体积

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = Path(SPECPATH)

# ── 数据文件 ──────────────────────────────────────────────────
datas = []

# qt-material 主题和资源文件
try:
    import qt_material
    qt_dir = Path(qt_material.__file__).parent
    for sub in ("themes", "resources"):
        d = qt_dir / sub
        if d.exists():
            datas.append((str(d), f"qt_material/{sub}"))
except ImportError:
    pass

# 项目字体（CJK 字体供 fpdf2 使用）
fonts_dir = PROJECT_ROOT / "resources" / "fonts"
if fonts_dir.exists():
    for f in fonts_dir.iterdir():
        if f.suffix.lower() in (".ttf", ".ttc", ".otf"):
            datas.append((str(f), "resources/fonts"))

# Jinja2 / fpdf2 内置数据
datas += collect_data_files("jinja2")
datas += collect_data_files("fpdf")

# ── 隐式导入 ─────────────────────────────────────────────────
hiddenimports = [
    # ── PySide6 核心 ──
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    # ── qt-material ──
    "qt_material",
    # ── 导出依赖 ──
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
    # ── 图像处理 ──
    "PIL",
    "PIL.Image",
    "PIL.ImageQt",
    "cv2",
    "fitz",         # PyMuPDF
    # ── 核心 ──
    "sqlite3",
    "numpy",
    "numpy.core._multiarray_umath",
]

# 收集所有 app.* 子模块（关键：避免面板/导出器被遗漏）
hiddenimports += collect_submodules("app")

# PaddleOCR（按需取消注释，体积约 +1 GB）
# hiddenimports += collect_submodules("paddle")
# hiddenimports += collect_submodules("paddleocr")

# ── 主程序分析 ─────────────────────────────────────────────────
a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],           # 无自定义 hooks
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
    upx=False,              # 不依赖外部 UPX；有 UPX 时改为 True
    console=False,          # 不弹出命令行窗口
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
