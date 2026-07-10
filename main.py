"""程序入口。"""
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from app.utils.font_installer import install_fonts

# 桌面应用默认启用 Hanwang batch-list；测试和脚本可用环境变量显式覆盖。
os.environ.setdefault("HANWANG_MICRO_RECBLOCK_BATCH", "1")

# 高 DPI 支持
QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
)


def main() -> None:
    from app.core.logging import setup_logging
    setup_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("OCR 后处理")
    app.setOrganizationName("ocr_process")

    # 应用浅色主题。
    from app.ui.styles import apply_theme
    apply_theme(app, "light")

    # Install fonts
    project_root = Path(__file__).parent
    install_fonts(project_root)

    from app.ui.main_window import MainWindow
    win = MainWindow()
    win.restore_working_project()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
