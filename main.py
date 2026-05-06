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

    # 应用深色主题
    try:
        from qt_material import apply_stylesheet
        apply_stylesheet(app, theme="dark_teal.xml", invert_secondary=False)
    except ImportError:
        # qt_material 未安装时降级到简单样式
        app.setStyle("Fusion")
        app.setStyleSheet("""
            QWidget { background:#1e1e1e; color:#d4d4d4; }
            QMenuBar  { background:#252526; }
            QMenu     { background:#2d2d30; border:1px solid #3c3c3c; }
            QStatusBar{ background:#007acc; color:#fff; }
            QPushButton {
                background:#0e639c; color:#fff; border-radius:4px;
                padding:4px 12px; border:none;
            }
            QPushButton:hover   { background:#1177bb; }
            QPushButton:disabled{ background:#3c3c3c; color:#666; }
            QLineEdit, QPlainTextEdit, QTextEdit {
                background:#252526; border:1px solid #3c3c3c;
                border-radius:3px; color:#d4d4d4;
            }
            QListWidget, QTreeWidget {
                background:#252526; border:1px solid #3c3c3c;
                color:#d4d4d4;
            }
            QSplitter::handle { background:#3c3c3c; }
            QProgressBar {
                background:#3c3c3c; border-radius:3px; color:#fff;
            }
            QProgressBar::chunk { background:#007acc; border-radius:3px; }
        """)

    from app.ui.main_window import MainWindow
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
