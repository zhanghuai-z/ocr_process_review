from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = Path(__file__).resolve().parent


def collect_shared_build_inputs(*, include_tools: bool = False):
    datas = []

    fonts_dir = PROJECT_ROOT / "resources" / "fonts"
    if fonts_dir.exists():
        for f in fonts_dir.iterdir():
            if f.suffix.lower() in (".ttf", ".ttc", ".otf"):
                datas.append((str(f), "resources/fonts"))

    # 汉王原生 OCR 资产（DLL / probe.exe / 字典）：整目录递归打包
    hanwang_native_dir = PROJECT_ROOT / "resources" / "hanwang_native"
    if hanwang_native_dir.exists():
        for f in hanwang_native_dir.rglob("*"):
            if f.is_file():
                rel_parent = f.parent.relative_to(PROJECT_ROOT).as_posix()
                datas.append((str(f), rel_parent))

    # 公式渲染资产：MathJax 以 node_modules 形式随包分发，运行时由
    # app.ui.proof.formula_renderer 通过 NODE_PATH 定位。
    formula_dir = PROJECT_ROOT / "resources" / "formula"
    if formula_dir.exists():
        for f in formula_dir.rglob("*"):
            if f.is_file():
                rel_parent = f.parent.relative_to(PROJECT_ROOT).as_posix()
                datas.append((str(f), rel_parent))

    datas += collect_data_files("jinja2")
    datas += collect_data_files("fpdf")

    hiddenimports = [
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtSvg",
        "PySide6.QtSvgWidgets",
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
        "PIL",
        "PIL.Image",
        "PIL.ImageQt",
        "cv2",
        "fitz",
        "sqlite3",
        "numpy",
        "numpy.core._multiarray_umath",
        "matplotlib.mathtext",
    ]
    hiddenimports += collect_submodules("app")
    if include_tools:
        hiddenimports += collect_submodules("tools")

    excludes = [
        "tkinter",
        "scipy",
        "IPython",
        "notebook",
        "pytest",
        "unittest",
    ]

    datas = sorted(dict.fromkeys(datas), key=lambda item: (item[1], item[0]))
    hiddenimports = sorted(dict.fromkeys(hiddenimports))

    return PROJECT_ROOT, datas, hiddenimports, excludes
