"""全局浅色主题样式表。

参考 ui.jpg 的视觉语言：
- 浅灰背景 #f5f7fb，白色卡片 #ffffff
- 天蓝主色 #1a73e8，hover #1666cf，淡蓝填充 #e3f0ff
- 圆角 6~8px，边框 #e3e8ef
- 字体 Microsoft YaHei UI / PingFang SC / Source Han Sans CN

约定的 objectName：
- "card"          —— 白底圆角容器
- "sectionTitle"  —— 卡片标题（蓝色）
- "sectionDesc"   —— 卡片描述（浅灰）
- "fieldLabel"    —— 表单字段标签
- "noteLabel"     —— 注释/小字
- "primaryBtn"    —— 主操作按钮（蓝底白字）
- "ghostBtn"      —— 浅蓝 outlined 次要按钮
- "stepBtn"       —— 顶部 / 侧栏步骤按钮（checked = 蓝）
- "headerBar"     —— 顶部标题栏
- "sidebarBar"    —— 左侧导航栏
- "toolbar"       —— 面板内工具栏
"""

LIGHT_QSS = """
QWidget {
    background: #f5f7fb;
    color: #222;
    font-family: "Microsoft YaHei UI", "PingFang SC", "Source Han Sans CN", sans-serif;
    font-size: 13px;
}

QMainWindow, QDialog { background: #f5f7fb; }

QMenuBar {
    background: #ffffff;
    color: #222;
    border-bottom: 1px solid #e3e8ef;
    padding: 2px 6px;
}
QMenuBar::item { background: transparent; padding: 4px 10px; border-radius: 4px; }
QMenuBar::item:selected { background: #e3f0ff; color: #1a73e8; }

QMenu {
    background: #ffffff;
    border: 1px solid #e3e8ef;
    color: #222;
    padding: 4px;
}
QMenu::item { padding: 6px 18px; border-radius: 4px; }
QMenu::item:selected { background: #e3f0ff; color: #1a73e8; }

QStatusBar { background: #ffffff; color: #555; border-top: 1px solid #e3e8ef; }

/* ---------- 标题 / 区段 ---------- */
QLabel#sectionTitle { color: #1a73e8; font-size: 14px; font-weight: bold; }
QLabel#sectionDesc  { color: #888; font-size: 12px; }
QLabel#fieldLabel   { color: #555; font-size: 13px; }
QLabel#noteLabel    { color: #888; font-size: 12px; }
QLabel#pageTitle    { color: #1a73e8; font-size: 18px; font-weight: bold; }
QLabel#muted        { color: #888; font-size: 12px; }

/* ---------- 卡片 ---------- */
QFrame#card {
    background: #ffffff;
    border: 1px solid #e3e8ef;
    border-radius: 8px;
}
QFrame#card[selected="true"] {
    border: 1px solid #1a73e8;
    background: #f0f6ff;
}

/* ---------- 顶部 header / 侧栏 ---------- */
QWidget#headerBar {
    background: #ffffff;
    border-bottom: 1px solid #e3e8ef;
}
QWidget#sidebarBar {
    background: #ffffff;
    border-right: 1px solid #e3e8ef;
}
QWidget#toolbar {
    background: #ffffff;
    border-bottom: 1px solid #e3e8ef;
}

/* ---------- 步骤 / 导航按钮 ---------- */
QPushButton#stepBtn {
    border: none;
    border-radius: 6px;
    padding: 8px 14px;
    color: #555;
    background: transparent;
    text-align: left;
    font-size: 13px;
}
QPushButton#stepBtn:hover { background: #f0f6ff; color: #1a73e8; }
QPushButton#stepBtn:checked { background: #e3f0ff; color: #1a73e8; font-weight: bold; }
QPushButton#stepBtn:disabled { color: #ccc; }

/* ---------- 输入控件 ---------- */
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit {
    background: #ffffff;
    border: 1px solid #d6dde6;
    border-radius: 6px;
    padding: 4px 8px;
    color: #222;
    selection-background-color: #cfe2ff;
    selection-color: #1a73e8;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {
    border: 1px solid #1a73e8;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {
    background: #f1f3f6; color: #999;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #ffffff; border: 1px solid #d6dde6;
    selection-background-color: #e3f0ff; selection-color: #1a73e8;
}

QRadioButton, QCheckBox { spacing: 6px; padding: 4px; color: #333; font-size: 13px; }

/* ---------- 列表 / 树 ---------- */
QListWidget, QTreeWidget, QTableWidget {
    background: #ffffff;
    border: 1px solid #e3e8ef;
    border-radius: 6px;
    color: #222;
    outline: 0;
}
QListWidget::item, QTreeWidget::item {
    padding: 4px 6px;
    border-radius: 4px;
}
QListWidget::item:hover, QTreeWidget::item:hover { background: #f5f9ff; }
QListWidget::item:selected, QTreeWidget::item:selected {
    background: #e3f0ff; color: #1a73e8;
}
QHeaderView::section {
    background: #f5f7fb;
    color: #555;
    border: none;
    border-right: 1px solid #e3e8ef;
    padding: 4px 8px;
    font-weight: 500;
}

/* ---------- 按钮 ---------- */
QPushButton {
    background: #ffffff;
    border: 1px solid #d6dde6;
    border-radius: 6px;
    padding: 6px 14px;
    color: #333;
    font-size: 13px;
}
QPushButton:hover { border-color: #1a73e8; color: #1a73e8; }
QPushButton:pressed { background: #f0f6ff; }
QPushButton:disabled { color: #aaa; border-color: #e0e4eb; background: #f5f7fb; }

QPushButton#primaryBtn {
    background: #1a73e8; color: white;
    border: 1px solid #1a73e8; font-weight: 500;
}
QPushButton#primaryBtn:hover { background: #1666cf; border-color: #1666cf; color: white; }
QPushButton#primaryBtn:pressed { background: #1357a8; }
QPushButton#primaryBtn:disabled { background: #c7d4ea; border-color: #c7d4ea; color: #fff; }

QPushButton#ghostBtn {
    background: #f0f6ff; color: #1a73e8;
    border: 1px solid #b8d4ff;
}
QPushButton#ghostBtn:hover { background: #e0eeff; }
QPushButton#ghostBtn:checked {
    background: #1a73e8; color: white; border-color: #1a73e8;
}

QPushButton#dangerBtn {
    background: #ffffff; color: #d93025; border: 1px solid #f4c4c4;
}
QPushButton#dangerBtn:hover { background: #fff0f0; }

/* 版面分析绿色三角启动按钮 */
QPushButton#runBtn {
    background: #34a853;
    color: #ffffff;
    border: none;
    border-radius: 18px;
    font-size: 15px;
    font-weight: bold;
    padding: 0;
}
QPushButton#runBtn:hover { background: #2d9248; }
QPushButton#runBtn:pressed { background: #267040; }
QPushButton#runBtn:disabled { background: #b8d4bc; color: #888; }

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {
    background: transparent; width: 10px; margin: 2px;
}
QScrollBar::handle:vertical {
    background: #c8d0db; border-radius: 5px; min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #1a73e8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background: transparent; height: 10px; margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #c8d0db; border-radius: 5px; min-width: 30px;
}
QScrollBar::handle:horizontal:hover { background: #1a73e8; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ---------- 进度条 ---------- */
QProgressBar {
    background: #e9eef5;
    border: none;
    border-radius: 3px;
    text-align: center;
    color: #1a73e8;
    font-size: 12px;
    height: 6px;
    max-height: 6px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a73e8, stop:0.5 #4da3ff, stop:1 #1a73e8);
    border-radius: 3px;
}

/* ---------- 横校行对 ---------- */
QFrame#linePair {
    background: #ffffff;
    border-bottom: 1px solid #edf0f5;
}
QFrame#linePair[active="true"] {
    background: #f0f6ff;
    border-left: 3px solid #1a73e8;
}

/* ---------- Proof 空状态 ---------- */
QLabel#proofEmpty {
    color: #bbbec4;
    font-size: 15px;
    qproperty-alignment: 'AlignCenter';
}

/* ---------- 横校页面分隔条 ---------- */
QLabel#pageSep {
    color: #aab4c0;
    font-size: 11px;
    background: #f5f7fb;
    border-top: 1px solid #e3e8ef;
    border-bottom: 1px solid #e3e8ef;
    padding: 2px 0;
}

/* ---------- 标签辅助 ---------- */
QLabel#stepInfo { color: #1a73e8; font-size: 12px; font-weight: 500; }

/* ---------- Splitter ---------- */
QSplitter::handle { background: #e3e8ef; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }

/* ---------- Tab ---------- */
QTabWidget::pane { border: 1px solid #e3e8ef; border-radius: 6px; background: #fff; }
QTabBar::tab {
    background: transparent; padding: 6px 14px; color: #555;
    border: none; border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: #1a73e8; border-bottom: 2px solid #1a73e8; font-weight: 500; }
QTabBar::tab:hover:!selected { color: #1a73e8; }

/* ---------- ToolTip ---------- */
QToolTip {
    background: #2c3e50; color: #fff;
    border: none; border-radius: 4px;
    padding: 4px 8px;
}

/* ---------- GraphicsView 背景 ---------- */
QGraphicsView { background: #fafbfc; border: 1px solid #e3e8ef; border-radius: 6px; }
"""


def apply_light_theme(app) -> None:
    """在 QApplication 上应用全局浅色主题。"""
    app.setStyle("Fusion")
    app.setStyleSheet(LIGHT_QSS)
