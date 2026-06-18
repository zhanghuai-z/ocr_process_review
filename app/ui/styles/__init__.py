"""全局主题系统（tokens + QSS 模板 + dispatcher）。

设计原则：
- 颜色 / 间距 / 圆角 / 字号 全部从 token 字典生成 QSS，禁止再在各 widget 散落硬编码色值。
- Light / Dark 两套 token，结构完全一致，便于跨主题维护。
- 通过 `apply_theme(app, name)` 统一入口，name = "light" | "dark"。
- objectName 约定见 README.md。

旧的 `app/ui/style.py` 仍存在但已停用，新代码应优先使用本模块。
"""
from __future__ import annotations
from typing import Any


# ─────────────────────────────────────────────────────────────
# Tokens
# ─────────────────────────────────────────────────────────────

LIGHT_TOKENS: dict[str, str] = {
    # 背景层级
    "bg_root":      "#f0f2f5",
    "bg_panel":     "#ffffff",
    "bg_card":      "#ffffff",
    "bg_hover":     "#f5f5f5",
    "bg_selected":  "#e6f7ff",
    "bg_input":     "#ffffff",
    "bg_input_dis": "#f5f5f5",
    "bg_status":    "#f0f2f5",
    # 边框
    "border":         "#e8e8e8",
    "border_input":   "#d9d9d9",
    "border_focus":   "#1677ff",
    "border_subtle":  "#f0f0f0",
    # 文本
    "text_primary":   "#1f2937",
    "text_secondary": "#4b5563",
    "text_muted":     "#8c8c8c",
    "text_disabled":  "#bfbfbf",
    "text_on_brand":  "#ffffff",
    # 品牌
    "brand":            "#1677ff",
    "brand_hover":      "#4096ff",
    "brand_pressed":    "#0958d9",
    "brand_disabled":   "#69b1ff",
    "brand_border_lt":  "#91caff",
    # 状态色
    "accent_green":         "#34a853",
    "accent_green_hover":   "#2d9248",
    "accent_green_pressed": "#267040",
    "accent_green_disabled":"#b8d4bc",
    "danger":         "#d93025",
    "danger_bg":      "#fff0f0",
    "danger_border":  "#f4c4c4",
    "success":        "#52c41a",
    "warning":        "#faad14",
    "error":          "#ff4d4f",
    # 滚动条
    "scroll_handle":   "#d1d5db",
    "scroll_handle_h": "#9ca3af",
    # 进度条轨道
    "progress_track": "#e9eef5",
    # ToolTip
    "tooltip_bg":   "#2c3e50",
    "tooltip_text": "#ffffff",
    # GraphicsView
    "canvas_bg": "#f0f2f5",
    # 状态徽章 pill（done/running/warn 三档）
    "pill_done_bg":     "#e6f4ea",
    "pill_done_text":   "#137333",
    "pill_running_bg":  "#eff6ff",
    "pill_running_text":"#2563eb",
    "pill_warn_bg":     "#fef7e0",
    "pill_warn_text":   "#b06000",
    "pill_idle_bg":     "#eef1f5",
    "pill_idle_text":   "#5f6368",
    # 字体
    "font_family": '"Inter", "PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei", "Source Han Sans CN", sans-serif',
    "font_size":   "13px",
    "font_size_sm":"12px",
    "font_size_lg":"14px",
    "font_size_xl":"18px",
    # 圆角
    "radius_sm": "4px",
    "radius_md": "6px",
    "radius_lg": "8px",
}

DARK_TOKENS: dict[str, str] = {
    # 背景层级
    "bg_root":      "#0f172a",
    "bg_panel":     "#1e293b",
    "bg_card":      "#1e293b",
    "bg_hover":     "#1e3a5f",
    "bg_selected":  "#1e3a5f",  # 柔和深蓝，避免覆盖文本/缩略图
    "bg_input":     "#0b1220",
    "bg_input_dis": "#161e2e",
    "bg_status":    "#0f172a",
    # 边框
    "border":         "#334155",
    "border_input":   "#475569",
    "border_focus":   "#60a5fa",
    "border_subtle":  "#1f2a3a",
    # 文本
    "text_primary":   "#f1f5f9",
    "text_secondary": "#cbd5e1",
    "text_muted":     "#94a3b8",
    "text_disabled":  "#64748b",
    "text_on_brand":  "#ffffff",
    # 品牌
    "brand":            "#60a5fa",
    "brand_hover":      "#93c5fd",
    "brand_pressed":    "#3b82f6",
    "brand_disabled":   "#3a4b66",
    "brand_border_lt":  "#3b82f6",
    # 状态色
    "accent_green":         "#4ade80",
    "accent_green_hover":   "#22c55e",
    "accent_green_pressed": "#16a34a",
    "accent_green_disabled":"#345c40",
    "danger":         "#f87171",
    "danger_bg":      "#3b1f1f",
    "danger_border":  "#ef4444",
    # 滚动条
    "scroll_handle":   "#475569",
    "scroll_handle_h": "#60a5fa",
    # 进度条轨道
    "progress_track": "#1f2a3a",
    # ToolTip
    "tooltip_bg":   "#f1f5f9",
    "tooltip_text": "#0f172a",
    # GraphicsView
    "canvas_bg": "#0b1220",
    # 状态徽章 pill（done/running/warn 三档）
    "pill_done_bg":     "#13332080",
    "pill_done_text":   "#4ade80",
    "pill_running_bg":  "#1e3a5f80",
    "pill_running_text":"#60a5fa",
    "pill_warn_bg":     "#4d3a1080",
    "pill_warn_text":   "#fbbf24",
    "pill_idle_bg":     "#1f2a3a",
    "pill_idle_text":   "#94a3b8",
    # 字体（与 light 一致）
    "font_family": '"Microsoft YaHei UI", "PingFang SC", "Source Han Sans CN", sans-serif',
    "font_size":   "13px",
    "font_size_sm":"12px",
    "font_size_lg":"14px",
    "font_size_xl":"18px",
    # 圆角
    "radius_sm": "4px",
    "radius_md": "6px",
    "radius_lg": "8px",
}


# ─────────────────────────────────────────────────────────────
# QSS 模板（占位符 {token_name}，由 .format(**tokens) 生成）
# ─────────────────────────────────────────────────────────────

_QSS_TEMPLATE = """
QWidget {{
    background: {bg_root};
    color: {text_primary};
    font-family: {font_family};
    font-size: {font_size};
}}

QMainWindow {{ background: {bg_root}; }}
QDialog {{ background: {bg_panel}; }}

QMenuBar {{
    background: {bg_panel};
    color: {text_primary};
    border-bottom: 1px solid {border};
    padding: 2px 6px;
}}
QMenuBar::item {{ background: transparent; padding: 4px 10px; border-radius: {radius_sm}; }}
QMenuBar::item:selected {{ background: {bg_selected}; color: {brand}; }}

QMenu {{
    background: {bg_panel};
    border: 1px solid {border};
    color: {text_primary};
    padding: 4px;
}}
QMenu::item {{ padding: 6px 18px; border-radius: {radius_sm}; }}
QMenu::item:selected {{ background: {bg_selected}; color: {brand}; }}

QStatusBar {{ background: {bg_panel}; color: {text_secondary}; border-top: 1px solid {border}; }}

/* ---------- 标题 / 区段 ---------- */
QLabel#sectionTitle {{ color: {brand}; font-size: {font_size_lg}; font-weight: bold; }}
QLabel#sectionDesc  {{ color: {text_muted}; font-size: {font_size_sm}; }}
QLabel#fieldLabel   {{ color: {text_secondary}; font-size: {font_size}; }}
QLabel#noteLabel    {{ color: {text_muted}; font-size: {font_size_sm}; }}
QLabel#pageTitle    {{ color: {brand}; font-size: {font_size_xl}; font-weight: bold; }}
QLabel#muted        {{ color: {text_muted}; font-size: {font_size_sm}; }}
QLabel#stepInfo     {{ color: {brand}; font-size: {font_size_sm}; font-weight: 500; }}
QLabel#sidebarTitle {{ color: {text_primary}; font-size: {font_size}; font-weight: 600; }}

/* ---------- 版面分析项目统计 ---------- */
QFrame#layoutStatsPanel {{
    background: {bg_panel};
    border-top: 1px solid {border};
}}
QFrame#layoutSidebarHeader {{
    background: {bg_panel};
    border-bottom: 1px solid {border};
}}
QFrame#blockTypeGroup {{
    background: {bg_panel};
    border: none;
    border-radius: 0;
}}
QLabel#blockTypeGroupTitle {{
    color: {text_muted};
    font-size: {font_size_sm};
    font-weight: 600;
}}
QLabel#typeBadge {{
    padding: 2px 10px;
    border-radius: 12px;
    border: 1px solid {brand_border_lt};
    background: {bg_selected};
    color: {brand};
    font-weight: 600;
    font-size: {font_size_sm};
}}
QFrame#blockTypeDivider {{
    background: {border_subtle};
    border: none;
}}
QDialog#layoutFindDialog {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: {radius_lg};
}}
QFrame#layoutFindHeader {{
    background: {bg_panel};
    border-bottom: 1px solid {border};
}}

/* ---------- 面包屑（TopBar） ---------- */
QLabel#crumbBrand   {{ color: {brand}; font-weight: bold; font-size: {font_size_lg}; }}
QLabel#crumbSep     {{ color: {text_muted}; font-size: {font_size}; padding: 0 2px; }}
QLabel#crumbProject {{ color: {text_secondary}; font-size: {font_size}; }}
QLabel#crumbStep    {{ color: {text_primary}; font-size: {font_size}; font-weight: 500; }}

/* ---------- 状态 pill（TopBar / 块统计共用） ---------- */
QLabel#statusPill {{
    background: {pill_idle_bg};
    color: {pill_idle_text};
    border-radius: 10px;
    padding: 2px 10px;
    font-size: {font_size_sm};
    font-weight: 500;
    min-height: 18px;
}}
QLabel#statusPill[kind="done"]    {{ background: {pill_done_bg};    color: {pill_done_text}; }}
QLabel#statusPill[kind="running"] {{ background: {pill_running_bg}; color: {pill_running_text}; }}
QLabel#statusPill[kind="warn"]    {{ background: {pill_warn_bg};    color: {pill_warn_text}; }}
QLabel#statusPill[kind="idle"]    {{ background: {pill_idle_bg};    color: {pill_idle_text}; }}

/* ---------- PageDir 行（页面目录） ---------- */
QListWidget#pageDirectoryList {{
    background: {bg_panel};
    border: none;
    outline: 0;
    padding: 0;
}}
QListWidget#pageDirectoryList::item {{
    border: none;
    padding: 0;
    margin: 0;
    background: transparent;
}}
QListWidget#pageDirectoryList::item:hover    {{ background: {bg_hover}; }}
QListWidget#pageDirectoryList::item:selected {{ background: {bg_selected}; border-left: 3px solid {brand}; }}

QTabWidget#layoutLeftTabs::pane {{
    border: none;
    border-top: 1px solid {border};
    background: {bg_panel};
}}
QTabWidget#layoutLeftTabs QTabBar::tab {{
    background: transparent;
    color: {text_secondary};
    padding: 8px 16px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabWidget#layoutLeftTabs QTabBar::tab:selected {{
    color: {brand};
    border-bottom: 2px solid {brand};
}}
QComboBox#layoutSearchPreset {{
    min-height: 26px;
}}
QTreeWidget#headingOutlineTree,
QListWidget#layoutSearchResults {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: {radius_md};
    outline: 0;
}}
QTreeWidget#headingOutlineTree::item,
QListWidget#layoutSearchResults::item {{
    min-height: 22px;
    padding: 2px 4px;
}}
QTreeWidget#headingOutlineTree::item:selected,
QListWidget#layoutSearchResults::item:selected {{
    background: {bg_selected};
    color: {brand};
}}

QWidget#pageRow {{ background: transparent; }}
QLabel#pageThumb {{
    background: {bg_root};
    border: 1px solid {border};
    border-radius: {radius_sm};
    color: {text_muted};
    font-size: 18px;
}}
QLabel#pageRowTitle {{ color: {text_primary}; font-size: {font_size}; font-weight: 500; }}
QLabel#pageRowFile  {{ color: {text_muted};   font-size: {font_size_sm}; }}

/* PageDir 状态徽章（小设计：18高 圆角） */
QLabel#pageBadge {{
    background: {pill_idle_bg};
    color: {pill_idle_text};
    border-radius: 9px;
    font-size: 11px;
    font-weight: bold;
    padding: 0 4px;
}}
QLabel#pageBadge[kind="done"]    {{ background: {pill_done_bg};    color: {pill_done_text}; }}
QLabel#pageBadge[kind="running"] {{ background: {pill_running_bg}; color: {pill_running_text}; }}
QLabel#pageBadge[kind="warn"]    {{ background: {pill_warn_bg};    color: {pill_warn_text}; }}
QLabel#pageBadge[kind="err"]     {{ background: {danger_bg};       color: {danger}; }}

/* ---------- 卡片 ---------- */
QFrame#card {{
    background: {bg_card};
    border: 1px solid {border};
    border-radius: {radius_lg};
}}
QFrame#card[selected="true"] {{
    border: 1px solid {brand};
    background: {bg_hover};
}}

/* ---------- 顶部 header / 侧栏 / 工具栏 ---------- */
QWidget#headerBar {{
    background: {bg_panel};
    border-bottom: 1px solid {border};
}}
QFrame#topBarLeft,
QFrame#topBarActions {{
    background: transparent;
    border: none;
}}
QFrame#workflowSegment {{
    background: {bg_status};
    border: 1px solid {border};
    border-radius: 16px;
}}
QWidget#sidebarBar {{
    background: {bg_panel};
    border: none;
}}
QWidget#toolbar {{
    background: {bg_panel};
    border-bottom: 1px solid {border};
}}

/* ---------- 步骤 / 导航按钮 ---------- */
QPushButton#workflowStepBtn {{
    border: none;
    border-radius: 14px;
    padding: 4px 16px;
    color: {text_secondary};
    background: transparent;
    font-size: {font_size};
    font-weight: 500;
}}
QPushButton#workflowStepBtn:hover    {{ background: {bg_hover}; color: {brand}; }}
QPushButton#workflowStepBtn:checked  {{
    background: {bg_panel};
    color: {brand};
    border: 1px solid {border_input};
    font-weight: 600;
}}
QPushButton#workflowStepBtn:disabled {{ color: {text_disabled}; }}

/* 右侧框类型按钮 */
QPushButton#blockTypeButton {{
    background: {bg_panel};
    border: 1px solid {border_input};
    border-radius: {radius_md};
    padding: 0 8px;
    color: {text_secondary};
    font-size: {font_size};
}}
QPushButton#blockTypeButton:hover {{
    border-color: {brand_hover};
    color: {brand_hover};
}}
QPushButton#blockTypeButton:checked {{
    background: {bg_selected};
    border-color: {brand};
    color: {brand};
    font-weight: 600;
}}

/* 切换型工具按钮（如 ⊞ 字框） */
QPushButton#toolToggle {{
    border: 1px solid {border_input};
    border-radius: {radius_md};
    padding: 4px 10px;
    color: {text_secondary};
    background: {bg_panel};
}}
QPushButton#toolToggle:hover   {{ border-color: {brand}; color: {brand}; }}
QPushButton#toolToggle:checked {{ background: {bg_selected}; color: {brand}; border-color: {brand}; }}

/* viewer 顶部小工具条（字框 / 新建 / 选中） */
QFrame#viewerToolbar {{
    background: {bg_panel};
    border-bottom: 1px solid {border};
}}
QScrollArea#layoutToolScroll {{
    background: {bg_panel};
    border: none;
}}

/* 底栏翻页按钮 < > 与页码标签 */
QPushButton#pageNavBtn {{
    border: 1px solid {border_input};
    border-radius: {radius_md};
    background: {bg_panel};
    color: {text_secondary};
    font-size: {font_size_lg};
    font-weight: bold;
}}
QPushButton#pageNavBtn:hover     {{ border-color: {brand}; color: {brand}; }}
QPushButton#pageNavBtn:disabled  {{ color: {text_disabled}; background: {bg_input_dis}; }}
QLabel#pageNavLabel {{
    color: {text_primary};
    font-size: {font_size};
    font-weight: 500;
}}

/* 底栏次要按钮（完成/取消） */
QPushButton#secondaryBtn {{
    border: 1px solid {border_input};
    border-radius: {radius_md};
    padding: 6px 14px;
    background: {bg_panel};
    color: {text_primary};
}}
QPushButton#secondaryBtn:hover    {{ border-color: {brand}; color: {brand}; }}
QPushButton#secondaryBtn:disabled {{ color: {text_disabled}; background: {bg_input_dis}; }}
QPushButton#defaultBtn {{
    border: 1px solid {border_input};
    border-radius: {radius_md};
    padding: 6px 14px;
    background: {bg_panel};
    color: {text_primary};
}}
QPushButton#defaultBtn:hover    {{ border-color: {brand_hover}; color: {brand_hover}; }}
QPushButton#defaultBtn:pressed  {{ border-color: {brand_pressed}; color: {brand_pressed}; background: #fafafa; }}
QPushButton#defaultBtn:disabled {{ color: {text_disabled}; background: {bg_input_dis}; }}
QPushButton#iconBtn {{
    border: none;
    border-radius: {radius_sm};
    background: transparent;
    color: {text_muted};
    font-size: {font_size_lg};
    padding: 0;
}}
QPushButton#iconBtn:hover {{
    background: {bg_hover};
    color: {text_primary};
}}

/* ---------- 输入控件 ---------- */
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit {{
    background: {bg_input};
    border: 1px solid {border_input};
    border-radius: {radius_md};
    padding: 6px 12px;
    color: {text_primary};
    min-height: 24px;
    selection-background-color: {bg_selected};
    selection-color: {brand};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {brand};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
    background: {bg_input_dis}; color: {text_disabled};
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {bg_panel}; border: 1px solid {border_input};
    selection-background-color: {bg_selected}; selection-color: {brand};
}}

/* QDialog 内部独立覆盖 */
QDialog#layoutFindDialog QLineEdit, QDialog#layoutFindDialog QComboBox {{
    background: {bg_input};
    border: 1px solid {border_input};
    border-radius: {radius_md};
    padding: 6px 12px;
    color: {text_primary};
    min-height: 24px;
}}
QDialog#layoutFindDialog QLineEdit:focus, QDialog#layoutFindDialog QComboBox:focus {{
    border: 1px solid {brand};
}}

QRadioButton, QCheckBox {{
    spacing: 6px; padding: 4px;
    color: {text_primary}; font-size: {font_size};
}}

/* ---------- 列表 / 树 / 表 ---------- */
QListWidget, QTreeWidget, QTableWidget {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: {radius_md};
    color: {text_primary};
    outline: 0;
}}
QListWidget::item, QTreeWidget::item {{ padding: 4px 6px; border-radius: {radius_sm}; }}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: {bg_hover}; }}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {bg_selected}; color: {brand};
}}
QHeaderView::section {{
    background: {bg_root};
    color: {text_secondary};
    border: none;
    border-right: 1px solid {border};
    padding: 4px 8px;
    font-weight: 500;
}}

/* ---------- 通用按钮 ---------- */
QPushButton {{
    background: {bg_panel};
    border: 1px solid {border_input};
    border-radius: {radius_md};
    padding: 6px 14px;
    color: {text_primary};
    font-size: {font_size};
}}
QPushButton:hover    {{ border-color: {border_input}; background: #f9fafb; color: {text_primary}; }}
QPushButton:pressed  {{ background: {bg_root}; }}
QPushButton:disabled {{ color: {text_disabled}; border-color: {border}; background: {bg_root}; }}

QPushButton#primaryBtn {{
    background: {brand}; color: {text_on_brand};
    border: 1px solid {brand}; font-weight: 500;
}}
QPushButton#primaryBtn:hover    {{ background: {brand_hover}; border-color: {brand_hover}; color: {text_on_brand}; }}
QPushButton#primaryBtn:pressed  {{ background: {brand_pressed}; }}
QPushButton#primaryBtn:disabled {{ background: {brand_disabled}; border-color: {brand_disabled}; color: {text_on_brand}; }}

QPushButton#ghostBtn {{
    background: {bg_panel}; color: {brand};
    border: 1px solid {brand_border_lt};
}}
QPushButton#ghostBtn:hover    {{ background: {bg_selected}; }}
QPushButton#ghostBtn:checked  {{ background: {brand}; color: {text_on_brand}; border-color: {brand}; }}

QPushButton#dangerBtn {{
    background: {bg_panel}; color: {danger}; border: 1px solid {danger_border};
}}
QPushButton#dangerBtn:hover {{ background: {danger_bg}; }}

/* 版面分析绿色三角启动按钮 */
QPushButton#runBtn {{
    background: {accent_green};
    color: {text_on_brand};
    border: none;
    border-radius: 18px;
    font-size: 15px;
    font-weight: bold;
    padding: 0;
}}
QPushButton#runBtn:hover    {{ background: {accent_green_hover}; }}
QPushButton#runBtn:pressed  {{ background: {accent_green_pressed}; }}
QPushButton#runBtn:disabled {{ background: {accent_green_disabled}; color: {text_muted}; }}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {scroll_handle}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {scroll_handle_h}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {scroll_handle}; border-radius: 5px; min-width: 30px; }}
QScrollBar::handle:horizontal:hover {{ background: {scroll_handle_h}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ---------- 进度条 ---------- */
QProgressBar {{
    background: {progress_track};
    border: none;
    border-radius: 3px;
    text-align: center;
    color: {brand};
    font-size: {font_size_sm};
    height: 6px;
    max-height: 6px;
}}
QProgressBar::chunk {{ background: {brand}; border-radius: 3px; }}

/* ---------- 横校行对 ---------- */
QFrame#linePair {{
    background: {bg_panel};
    border-bottom: 1px solid {border_subtle};
}}
QFrame#linePair[active="true"] {{
    background: {bg_hover};
    border-left: 3px solid {brand};
}}

/* ---------- Proof 空状态 ---------- */
QLabel#proofEmpty {{
    color: {text_disabled};
    font-size: 15px;
    qproperty-alignment: 'AlignCenter';
}}

/* ---------- 横校页面分隔条 ---------- */
QLabel#pageSep {{
    color: {text_muted};
    font-size: 11px;
    background: {bg_root};
    border-top: 1px solid {border};
    border-bottom: 1px solid {border};
    padding: 2px 0;
}}

/* ---------- Splitter ---------- */
QSplitter::handle {{ background: {border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter::handle:hover {{ background: {brand_hover}; }}

/* ---------- Tab ---------- */
QTabWidget::pane {{ border: 1px solid {border}; border-radius: {radius_md}; background: {bg_panel}; }}
QTabBar::tab {{
    background: transparent; padding: 6px 14px; color: {text_secondary};
    border: none; border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {brand}; border-bottom: 2px solid {brand}; font-weight: 500; }}
QTabBar::tab:hover:!selected {{ color: {brand}; }}

/* ---------- ToolTip ---------- */
QToolTip {{
    background: {tooltip_bg}; color: {tooltip_text};
    border: none; border-radius: {radius_sm};
    padding: 4px 8px;
}}

/* ---------- GraphicsView 背景 ---------- */
QGraphicsView {{ background: {canvas_bg}; border: none; }}
"""


# ─────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────

_THEMES: dict[str, dict[str, str]] = {
    "light": LIGHT_TOKENS,
    "dark":  DARK_TOKENS,
}

# 旧配置兼容映射
_THEME_ALIASES: dict[str, str] = {
    "dark_teal": "dark",
    "light_blue": "light",
}


def _normalize(name: str | None) -> str:
    if not name:
        return "light"
    name = str(name).lower().strip()
    name = _THEME_ALIASES.get(name, name)
    return name if name in _THEMES else "light"


def build_qss(theme: str = "light") -> str:
    """根据主题名生成完整 QSS 文本。"""
    tokens = _THEMES[_normalize(theme)]
    return _QSS_TEMPLATE.format(**tokens)


def apply_theme(app: Any, theme: str | None = None) -> str:
    """在 QApplication 上应用主题，并返回实际生效的主题名。"""
    app.setStyle("Fusion")
    actual = _normalize(theme)
    app.setStyleSheet(build_qss(actual))
    return actual


def available_themes() -> list[str]:
    return list(_THEMES.keys())


__all__ = [
    "LIGHT_TOKENS",
    "DARK_TOKENS",
    "apply_theme",
    "build_qss",
    "available_themes",
]
