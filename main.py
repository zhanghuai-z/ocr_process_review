"""程序入口。"""
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# 高 DPI 支持
QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("OCR 后处理")
    app.setOrganizationName("ocr_process")

    # 应用浅色主题（参考 ui.jpg）
    from app.ui.style import apply_light_theme
    apply_light_theme(app)

    from app.ui.main_window import MainWindow
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
