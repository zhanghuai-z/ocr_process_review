"""全局主题系统（tokens + QSS 模板 + dispatcher）。

设计原则：
- 颜色 / 间距 / 圆角 / 字号 全部从 token 字典生成 QSS，禁止再在各 widget 散落硬编码色值。
- 当前只保留浅色主题；未知主题名统一回落到 light。
- 通过 `apply_theme(app, name)` 统一入口。
- objectName 约定见 README.md。
"""
from pathlib import Path
from typing import Any
from PySide6.QtGui import QFontDatabase

# ─────────────────────────────────────────────────────────────
# OCR ProProof — Minimalist Studio 设计系统
# 所有色值直接提取自 Stitch 设计稿 HTML 源码
# ─────────────────────────────────────────────────────────────

LIGHT_TOKENS: dict[str, str] = {
    # ── 背景层级 ──
    "bg_root":        "#F5F3EE",   # warm workspace shell
    "bg_panel":       "#FFFDF8",   # soft white panels
    "bg_card":        "#FFFFFF",   # white card surfaces
    "bg_hover":       "#ECE8DF",   # primary-container — hover state
    "bg_active":      "#ECE8DF",
    "bg_selected":    "#2C2C2C",   # primary — selected state
    "bg_input":       "#FFFFFF",
    "bg_input_dis":   "#F5F3EE",
    "bg_success":     "#e6f4ea",
    "bg_warning":     "#fef7e0",

    # ── 边框 ──
    "border":         "#E7E2D8",   # subtle warm border
    "border_input":   "#D1CFCA",   # slightly stronger for inputs
    "border_focus":   "#2C2C2C",   # charcoal focus ring
    "border_subtle":  "transparent",

    # ── 文本 ──
    "text_primary":   "#2C2C2C",   # on-surface — charcoal
    "text_secondary": "#6B6B6B",   # on-surface-variant
    "text_muted":     "#A1A1A1",
    "text_disabled":  "#CCCCCC",
    "text_on_brand":  "#FFFFFF",

    # ── 品牌主色（Minimalist Studio = 碳黑） ──
    "brand":          "#2C2C2C",
    "brand_hover":    "#1A1A1A",
    "brand_pressed":  "#000000",
    "brand_disabled": "#8C8C8C",
    "brand_border_lt": "#4A4A4A",

    # ── 功能色 ──
    "accent_green":          "#5C6B58",   # secondary — muted sage
    "accent_green_hover":    "#4A5A46",
    "accent_green_pressed":  "#3A4A36",
    "accent_green_disabled": "#B0B8AD",
    "danger":                "#9E4444",   # error
    "danger_bg":             "#F5E0E0",
    "danger_border":         "#D4A0A0",
    "status_error":   "#9E4444",
    "status_success": "#5C6B58",
    "status_warn":    "#B8860B",
    "status_info":    "#2C2C2C",

    # ── 胶囊/徽章（直接使用主色系） ──
    "pill_done_bg":     "#E6F4EA",
    "pill_done_text":   "#3A5A36",
    "pill_run_bg":      "#ECE8DF",
    "pill_run_text":    "#2C2C2C",
    "pill_err_bg":      "#F5E0E0",
    "pill_err_text":    "#9E4444",
    "pill_warn_bg":     "#FEF7E0",
    "pill_warn_text":   "#8B6914",
    "pill_idle_bg":     "transparent",
    "pill_idle_text":   "#A1A1A1",

    # ── 字体 ──
    "font_family":      "Microsoft YaHei",
    "font_family_mono": "JetBrains Mono",
    "font_size_xs":  "12px",
    "font_size_sm":  "13px",
    "font_size":     "14px",
    "font_size_lg":  "16px",
    "font_size_xl":  "20px",

    # ── 圆角（设计稿: 2px default, 4px lg, 8px xl） ──
    "radius_sm":  "2px",
    "radius_md":  "4px",
    "radius_lg":  "8px",
    "radius_xl":  "16px",       # pills / rounded-full

    # ── 杂项 ──
    "scroll_handle":   "#D1CFCA",
    "scroll_handle_h": "#A0A0A0",
    "progress_track":  "#ECE8DF",
    "progress_fill":   "#9E4444",
    "tooltip_bg":      "#2C2C2C",
    "tooltip_text":    "#FFFFFF",
    "canvas_bg":       "#F5F3EE",
}

# ─────────────────────────────────────────────────────────────
# QSS 模板 — 对齐 Minimalist Studio 设计稿
# 注意：Python str.format 要求 {{ }} 来表示 CSS 花括号
# ─────────────────────────────────────────────────────────────

_QSS_TEMPLATE = """
/* ═══ Minimalist Studio — Global Reset ═══ */
* {{
    font-family: {font_family};
    font-size: {font_size};
    color: {text_primary};
}}

QMainWindow {{ background: {bg_root}; }}
QDialog     {{ background: {bg_root}; }}
QMessageBox {{
    background: {bg_root};
}}
QMessageBox QLabel {{
    background: transparent;
    color: {text_primary};
    font-size: {font_size};
}}
QMessageBox QPushButton {{
    min-width: 72px;
    min-height: 28px;
    padding: 6px 16px;
    border: 1px solid {border_input};
    border-radius: {radius_xl};
    background: transparent;
    color: {text_primary};
    font-size: {font_size};
}}
QMessageBox QPushButton:hover {{
    background: {bg_panel};
    border-color: {brand};
}}
QMessageBox QPushButton:pressed {{
    background: {bg_hover};
}}
QStatusBar {{
    background: {bg_root};
    color: {text_secondary};
    border: none;
}}

/* ═══ Menu Bar — 极简融入背景 ═══ */
QMenuBar {{
    background: {bg_root};
    border: none;
    font-size: {font_size_sm};
    padding: 2px 0;
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
    color: {text_secondary};
    border-radius: {radius_sm};
}}
QMenuBar::item:selected {{
    background: {bg_hover};
    color: {text_primary};
}}
QMenu {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: {radius_md};
    padding: 4px 0;
}}
QMenu::item {{
    padding: 6px 28px 6px 16px;
    color: {text_primary};
    font-size: {font_size_sm};
}}
QMenu::item:selected {{
    background: {brand};
    color: {text_on_brand};
}}

/* ═══ Scrollbar — 有明确槽位，避免看不见滚动区域 ═══ */
QScrollBar:vertical {{
    border: none;
    background: rgba(236, 232, 223, 170);
    width: 9px;
    margin: 0;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {scroll_handle};
    min-height: 34px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{ background: {scroll_handle_h}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QScrollBar:horizontal {{
    border: none;
    background: rgba(236, 232, 223, 170);
    height: 9px;
    margin: 0;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {scroll_handle};
    min-width: 34px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal:hover {{ background: {scroll_handle_h}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ═══ 核心结构容器 ═══ */
QFrame#whiteCard {{
    background: {bg_card};
    border-radius: {radius_lg};
    border: 1px solid rgba(255, 255, 255, 190);
}}

QFrame#layoutSidebarHeader {{
    background: transparent;
    border: none;
}}

QFrame#layoutStatsPanel {{
    background: {bg_card};
    border: 1px solid rgba(255, 255, 255, 180);
    border-radius: {radius_lg};
}}

QFrame#actionArea {{
    background: {bg_root};
    border: none;
}}

QFrame#layoutFindHeader {{
    background: {bg_root};
    border: none;
}}

QFrame#pillSwitch, QFrame#pillToolbar {{
    background: rgba(236, 232, 223, 220);
    border-radius: 12px;
    border: 1px solid {border};
    padding: 2px;
}}

QFrame#topBarLeft, QFrame#topBarActions {{
    background: transparent;
}}

QFrame#topMenuCluster {{
    background: transparent;
    border: none;
}}

QWidget#headerBar {{
    background: transparent;
    border: none;
}}

QWidget#sidebarBar {{
    background: transparent;
    border: none;
}}

QScrollArea#layoutToolScroll {{
    background: transparent;
    border: none;
}}

QWidget#layoutCanvasPane,
QWidget#layoutRightPane,
QWidget#layoutRightContent {{
    background: transparent;
}}

QWidget#proofCenterPane,
QWidget#proofContentPane,
QWidget#proofLeftPane,
QWidget#proofLeftStackPage,
QWidget#proofRightPane,
QWidget#settingsContent {{
    background: transparent;
}}

QWidget#proofRoot {{
    background: {bg_root};
}}

QWidget#importRoot {{
    background: {bg_root};
}}

QSplitter#hproofSplitter,
QSplitter#proofSplitter,
QSplitter#proofContentSplitter {{
    background: {bg_root};
}}

QScrollArea#proofScroll {{
    background: transparent;
    border: none;
}}

QScrollArea#proofScroll > QWidget > QWidget#proofLineList {{
    background: transparent;
}}

QWidget#proofCenterPane,
QWidget#proofLeftPane,
QWidget#proofRightPane,
QFrame#proofLeftPane,
QFrame#proofSidePanel,
QFrame#proofProgressPopup,
QFrame#proofCard,
QFrame#settingsNavPane,
QFrame#settingsDetailPane,
QFrame#exportDialogCard,
QFrame#importCard {{
    background: {bg_card};
    border: 1px solid {border};
    border-radius: {radius_lg};
}}

QFrame#importDropArea {{
    background: {bg_panel};
    border: 2px dashed {border_input};
    border-radius: {radius_lg};
}}
QFrame#importDropArea:hover {{
    background: {bg_hover};
    border-color: {text_secondary};
}}

QFrame#importFileRow {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: {radius_md};
}}
QFrame#importFileRow:hover {{
    background: {bg_hover};
}}

QWidget#proofToolbar,
QFrame#proofToolbar,
QWidget#proofStatusBar,
QFrame#bottomBar {{
    background: rgba(255, 253, 248, 220);
    border: none;
}}

QFrame#proofSection {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: {radius_lg};
}}

QWidget#candidatePanel {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: {radius_lg};
}}

QWidget#settingsCategoryList {{
    background: transparent;
}}

QDialog#apiSettingsDialog,
QDialog#exportDialog {{
    background: {bg_root};
}}

QTabWidget#layoutLeftTabs {{
    background: transparent;
    border: none;
}}

/* ═══ 左侧目录列表 ═══ */
QListWidget {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget::item {{
    background: transparent;
    border: none;
    padding: 0;
}}
QListWidget::item:selected {{
    background: transparent;
}}

QListWidget#pageDirectoryList {{
    background: {bg_root};
    border: none;
    outline: none;
}}
QListWidget#pageDirectoryList::item {{
    background: transparent;
    border: none;
    padding: 0;
}}
QWidget#pageRow {{
    background: transparent;
    border: none;
}}

QListWidget#importQueueList {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget#importQueueList::item {{
    padding: 3px 0;
    background: transparent;
    border: none;
}}
QListWidget#importQueueList::item:selected {{
    background: transparent;
}}
QWidget#pageRow:hover {{
    background: transparent;
    border-radius: {radius_lg};
}}
QFrame#pageThumbCard {{
    background: {bg_panel};
    border: 1px solid rgba(255, 255, 255, 180);
    border-radius: {radius_lg};
}}
QLabel#pageThumb {{
    background: transparent;
    border: none;
    border-radius: {radius_md};
    color: {text_muted};
    font-size: 10px;
    font-family: {font_family_mono};
}}
QLabel#pageRowTitle {{
    color: {text_primary};
    font-size: {font_size_sm};
    font-weight: 500;
}}
QLabel#pageRowFile {{
    color: {text_secondary};
    font-size: 10px;
    font-weight: 500;
}}
QLabel#pageBadge {{
    background: rgba(44, 44, 44, 205);
    color: {text_on_brand};
    border: none;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 500;
}}
QLabel#pageBadge[kind="done"] {{
    background: {pill_done_bg};
    color: {pill_done_text};
}}
QLabel#pageBadge[kind="running"] {{
    background: {pill_run_bg};
    color: {pill_run_text};
}}
QLabel#pageBadge[kind="warn"] {{
    background: {pill_warn_bg};
    color: {pill_warn_text};
}}
QLabel#pageBadge[kind="err"] {{
    background: {pill_err_bg};
    color: {pill_err_text};
}}

/* ═══ 模式标签（MODE: 正文） ═══ */
QLabel#typeBadge {{
    background: #E7EDE3;
    color: {accent_green};
    padding: 3px 12px;
    border-radius: 10px;
    font-size: {font_size_xs};
    font-weight: 600;
    letter-spacing: 0;
    border: 1px solid #C9D4C5;
}}

/* ═══ 标题与徽章 ═══ */
QLabel#superTitle {{
    color: {text_secondary};
    font-size: {font_size_xs};
    font-weight: 500;
    letter-spacing: 3px;
}}

QLabel#blockTypeGroupTitle {{
    color: {text_secondary};
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 2px;
}}

QLabel#brandTitle {{
    color: {text_primary};
    font-size: 16px;
    font-weight: 500;
    letter-spacing: 0.5px;
}}

QLabel#crumbProject {{
    color: {text_primary};
    font-size: {font_size_sm};
    font-weight: 500;
}}

QLabel#crumbSep, QLabel#crumbStep {{
    color: {text_secondary};
    font-size: 10px;
}}

QLabel#statusPill {{
    color: {text_secondary};
    font-family: {font_family_mono};
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 2px;
}}
QLabel#statusPill[kind="done"] {{ color: {status_success}; }}
QLabel#statusPill[kind="running"] {{ color: {text_primary}; }}
QLabel#statusPill[kind="warn"] {{ color: {status_warn}; }}
QLabel#statusPill[kind="error"] {{ color: {status_error}; }}

QLabel#sidebarTitle {{
    color: {text_primary};
    font-size: {font_size};
    font-weight: 500;
}}

QLabel#muted {{
    color: {text_muted};
    font-size: {font_size_xs};
}}

QLabel#sectionTitle {{
    color: {text_primary};
    font-size: {font_size_sm};
    font-weight: 500;
}}

QLabel#sectionDesc, QLabel#noteLabel {{
    color: {text_muted};
    font-size: {font_size_xs};
}}

QLabel#fieldLabel {{
    color: {text_secondary};
    font-size: {font_size_xs};
    font-weight: 500;
}}

QLabel#pageTitle {{
    color: {text_primary};
    font-size: {font_size_xl};
    font-weight: 500;
}}

QLabel#importTitle {{
    color: {text_primary};
    font-size: 20px;
    font-weight: 600;
}}

QLabel#importSubtitle {{
    color: {text_secondary};
    font-size: {font_size_sm};
}}

QLabel#importSectionTitle {{
    color: {text_primary};
    font-size: {font_size};
    font-weight: 600;
}}

QLabel#importHint {{
    color: {text_muted};
    font-size: {font_size_xs};
    line-height: 1.45;
}}

QLabel#importDropIcon {{
    min-width: 54px;
    min-height: 54px;
    max-width: 54px;
    max-height: 54px;
    background: {bg_card};
    color: {text_secondary};
    border: 1px solid {border};
    border-radius: 27px;
    font-size: 30px;
    font-weight: 300;
}}

QLabel#importDropTitle {{
    color: {text_primary};
    font-size: {font_size_lg};
    font-weight: 600;
}}

QLabel#importDropDesc {{
    color: {text_secondary};
    font-size: {font_size_sm};
}}

QLabel#importCountPill,
QLabel#importKindPill,
QLabel#importReadyPill {{
    background: {bg_hover};
    color: {text_secondary};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 3px 9px;
    font-size: {font_size_xs};
    font-weight: 500;
}}

QLabel#importReadyPill {{
    background: #E7EDE3;
    color: {accent_green};
    border-color: #C9D4C5;
}}

QLabel#importEmpty {{
    color: {text_muted};
    font-size: {font_size_sm};
    padding: 28px 8px;
}}

QLabel#importFileName {{
    color: {text_primary};
    font-size: {font_size_sm};
    font-weight: 500;
}}

QLabel#importFilePath {{
    color: {text_muted};
    font-size: {font_size_xs};
}}

QLabel#importFileIcon {{
    background: {bg_hover};
    border: 1px solid {border};
    border-radius: {radius_md};
}}

QLabel#dialogTitle {{
    color: {text_primary};
    font-size: 20px;
    font-weight: 600;
}}

QLabel#dialogSubtitle {{
    color: {text_secondary};
    font-size: {font_size_sm};
}}

QLabel#statValue {{
    color: {text_primary};
    font-size: 18px;
    font-weight: 600;
}}

QLabel#statWarnValue {{
    color: {danger};
    font-size: 18px;
    font-weight: 600;
}}

QLabel#proofStatusStrong {{
    color: {text_primary};
    font-size: {font_size_sm};
    font-weight: 600;
}}

/* ═══ 顶栏菜单入口 ═══ */
QPushButton#topMenuBtn {{
    background: rgba(255, 253, 248, 180);
    border: 1px solid {border};
    border-radius: {radius_lg};
    padding: 5px 8px;
    color: {text_secondary};
    font-size: {font_size_sm};
}}
QPushButton#topMenuBtn:hover {{
    background: {bg_card};
    border-color: {border_input};
    color: {text_primary};
}}
QPushButton#topMenuBtn:pressed {{
    background: {bg_hover};
}}
QPushButton#topMenuBtn::menu-indicator {{
    image: none;
    width: 0;
}}

/* ═══ 横校左侧导航 ═══ */
QFrame#proofNavRail {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: {radius_lg};
}}
QPushButton#proofRailBtn {{
    background: transparent;
    border: none;
    border-radius: {radius_md};
    color: {text_secondary};
    font-size: {font_size_sm};
    font-weight: 600;
}}
QPushButton#proofRailBtn:hover {{
    background: {bg_hover};
    color: {text_primary};
}}
QPushButton#proofRailBtn:checked {{
    background: {bg_hover};
    color: {text_primary};
}}
QFrame#proofLeftPane QListWidget#pageDirectoryList {{
    background: transparent;
}}

QFrame#proofProgressPopup {{
    min-width: 190px;
}}

/* ═══ 中间悬浮工具条按钮 ═══ */
QPushButton#pillToolBtn {{
    background: transparent;
    border: none;
    border-radius: {radius_lg};
    color: {text_secondary};
    padding: 4px 8px;
    font-size: {font_size_sm};
}}
QPushButton#pillToolBtn:hover {{
    color: {text_primary};
    background: {bg_hover};
}}
QPushButton#pillToolBtn:checked {{
    color: {brand};
    background: rgba(44, 44, 44, 10);
    font-weight: 500;
}}
QPushButton#pillToolBtn:pressed {{
    background: {border_input};
    border-radius: {radius_md};
}}

/* ═══ 右侧类型按钮网格 ═══ */
QPushButton#blockTypeButton {{
    background: {bg_panel};
    border: none;
    border-radius: {radius_lg};
    padding: 8px 4px;
    color: {text_primary};
    font-size: {font_size_sm};
}}
QPushButton#blockTypeButton:hover {{
    background: {bg_panel};
}}
QPushButton#blockTypeButton:checked {{
    background: {brand};
    color: {text_on_brand};
    font-weight: 500;
}}
QPushButton#blockTypeButton:checked:enabled {{
    color: {text_on_brand};
}}
QPushButton#blockTypeButton:pressed {{
    background: {brand_pressed};
    color: {text_on_brand};
    border-radius: {radius_lg};
}}

/* ═══ 顶部大流程按钮（胶囊切换） ═══ */
QPushButton#workflowStepBtn {{
    background: transparent;
    border: none;
    border-radius: 9px;
    color: {text_secondary};
    font-size: {font_size_sm};
    padding: 6px 20px;
    font-weight: 500;
}}
QPushButton#workflowStepBtn:hover {{
    color: {text_primary};
    background: rgba(255, 253, 248, 120);
}}
QPushButton#workflowStepBtn:checked {{
    background: {bg_card};
    color: {text_primary};
    font-weight: 700;
    border-radius: 9px;
    border: 1px solid {border};
}}
QPushButton#workflowStepBtn:pressed {{
    background: {bg_hover};
    border-radius: 9px;
}}

/* ═══ 主操作按钮（碳黑） ═══ */
QPushButton#darkBtn {{
    background: {brand};
    color: {text_on_brand};
    border: none;
    border-radius: {radius_xl};
    padding: 7px 22px;
    font-size: {font_size_sm};
    font-weight: 500;
    letter-spacing: 1px;
}}
QPushButton#darkBtn:enabled {{
    color: {text_on_brand};
}}
QPushButton#darkBtn:hover    {{ background: {brand_hover}; }}
QPushButton#darkBtn:pressed  {{ background: {brand_pressed}; }}
QPushButton#darkBtn:disabled {{
    background: #D8D2C8;
    color: {text_secondary};
}}

/* ═══ 次要按钮（圆角边框） ═══ */
QPushButton#secondaryBtn, QPushButton#defaultBtn {{
    border: 1px solid {border_input};
    border-radius: {radius_xl};
    padding: 6px 16px;
    background: transparent;
    color: {text_primary};
    font-size: {font_size};
}}
QPushButton#secondaryBtn:hover, QPushButton#defaultBtn:hover {{
    background: {bg_panel};
    border-color: {brand};
    color: {brand};
}}
QPushButton#secondaryBtn:pressed, QPushButton#defaultBtn:pressed {{
    background: {bg_hover};
}}
QPushButton#secondaryBtn:disabled, QPushButton#defaultBtn:disabled {{
    color: {text_disabled};
    background: {bg_input_dis};
    border-color: {border};
}}

/* ═══ Ghost 按钮 ═══ */
QPushButton#ghostBtn {{
    background: transparent;
    color: {text_secondary};
    border: none;
    padding: 6px 10px;
    font-size: {font_size_sm};
    border-radius: {radius_lg};
}}
QPushButton#ghostBtn:hover {{
    color: {text_primary};
    background: {bg_hover};
}}
QPushButton#ghostBtn:pressed {{
    background: {border_input};
}}
QPushButton#ghostBtn:checked {{
    background: {bg_hover};
    color: {text_primary};
    border: 1px solid {border_input};
}}

QPushButton#iconBtn {{
    background: transparent;
    border: none;
    border-radius: {radius_lg};
    color: {text_secondary};
    font-size: {font_size_lg};
}}
QPushButton#iconBtn:hover {{
    background: {bg_hover};
    color: {text_primary};
}}
QPushButton#iconBtn:pressed {{
    background: {border_input};
}}

QPushButton#navItemBtn {{
    background: transparent;
    border: none;
    border-radius: {radius_lg};
    color: {text_secondary};
    font-size: {font_size};
    font-weight: 500;
    padding: 10px 14px;
    text-align: left;
}}
QPushButton#navItemBtn:hover {{
    background: {bg_hover};
    color: {text_primary};
}}
QPushButton#navItemBtn:checked {{
    background: {bg_hover};
    color: {text_primary};
    border-left: 3px solid {brand};
}}

QPushButton#candidateButton {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: {radius_lg};
    padding: 6px 12px;
    color: {text_primary};
    font-size: {font_size};
}}
QPushButton#candidateButton:hover {{
    border-color: {accent_green};
    background: {bg_hover};
}}

/* ═══ 页面导航 ═══ */
QPushButton#pageNavBtn {{
    background: transparent;
    border: none;
    color: {text_secondary};
    font-size: {font_size_sm};
}}
QPushButton#pageNavBtn:hover {{ color: {text_primary}; }}

QLabel#pageNavLabel {{
    color: {text_primary};
    font-family: {font_family_mono};
    font-weight: 500;
    font-size: {font_size};
}}

/* ═══ 主色按钮变体 ═══ */
QPushButton#primaryBtn {{
    background: {brand};
    color: {text_on_brand};
    border: 1px solid {brand};
    border-radius: {radius_xl};
    padding: 6px 16px;
    font-size: {font_size};
    font-weight: 500;
}}
QPushButton#primaryBtn:enabled {{
    color: {text_on_brand};
}}
QPushButton#primaryBtn:hover    {{ background: {brand_hover}; border-color: {brand_hover}; color: {text_on_brand}; }}
QPushButton#primaryBtn:pressed  {{ background: {brand_pressed}; }}
QPushButton#primaryBtn:disabled {{
    background: #D8D2C8;
    border-color: #D8D2C8;
    color: {text_secondary};
}}

/* ═══ 危险按钮 ═══ */
QPushButton#dangerBtn {{
    background: transparent;
    color: {danger};
    border: 1px solid {danger_border};
    border-radius: {radius_md};
}}
QPushButton#dangerBtn:hover {{ background: {danger_bg}; }}

/* ═══ 运行按钮（sage 绿） ═══ */
QPushButton#runBtn {{
    background: {accent_green};
    color: {text_on_brand};
    border: none;
    border-radius: {radius_xl};
    font-size: {font_size};
    font-weight: 500;
    padding: 0;
}}
QPushButton#runBtn:hover    {{ background: {accent_green_hover}; }}
QPushButton#runBtn:pressed  {{ background: {accent_green_pressed}; }}
QPushButton#runBtn:disabled {{ background: {accent_green_disabled}; color: {text_muted}; }}

/* ═══ 进度条 ═══ */
QProgressBar {{
    background: {progress_track};
    border: none;
    border-radius: 0;
    text-align: center;
    color: {text_primary};
    font-size: {font_size_xs};
    height: 16px;
    max-height: 16px;
}}
QProgressBar::chunk {{ background: {progress_fill}; border-radius: 0; }}

/* ═══ 分组与复选框 ═══ */
QGroupBox {{
    background: {bg_card};
    border: 1px solid {border};
    border-radius: {radius_lg};
    margin-top: 16px;
    padding: 14px 12px 12px 12px;
    color: {text_primary};
    font-weight: 500;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {text_primary};
    background: {bg_root};
}}

QCheckBox {{
    spacing: 8px;
    color: {text_primary};
    font-size: {font_size};
    padding: 5px 2px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {border_input};
    border-radius: 4px;
    background: {bg_panel};
}}
QCheckBox::indicator:hover {{
    border-color: {accent_green};
}}
QCheckBox::indicator:checked {{
    background: {brand};
    border-color: {brand};
}}
QCheckBox::indicator:disabled {{
    background: {bg_input_dis};
    border-color: {border};
}}

/* ═══ 表格 ═══ */
QTableWidget {{
    background: {bg_card};
    border: 1px solid {border};
    border-radius: {radius_lg};
    gridline-color: {border};
    color: {text_primary};
    selection-background-color: {bg_hover};
    selection-color: {text_primary};
}}
QHeaderView::section {{
    background: {bg_panel};
    color: {text_secondary};
    border: none;
    border-bottom: 1px solid {border};
    padding: 6px 8px;
    font-size: {font_size_sm};
    font-weight: 500;
}}

/* ═══ 横校行对 ═══ */
QFrame#linePair {{
    background: {bg_card};
    border: 1px solid transparent;
    border-bottom: 1px solid {border};
    border-radius: 0;
}}
QFrame#linePair[active="true"] {{
    background: {bg_panel};
    border: 1px solid {brand};
    border-radius: {radius_lg};
}}

/* ═══ Proof 空状态 ═══ */
QLabel#proofEmpty {{
    color: {text_disabled};
    font-size: {font_size_lg};
    qproperty-alignment: 'AlignCenter';
}}

/* ═══ 页面分隔条 ═══ */
QLabel#pageSep {{
    color: {text_secondary};
    font-size: {font_size_xs};
    background: transparent;
    border: none;
    padding: 6px 0 2px 0;
}}

/* ═══ Splitter ═══ */
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 8px; }}
QSplitter::handle:vertical   {{ height: 8px; }}

/* ═══ Tab（页面/标题切换） ═══ */
QTabWidget::pane {{ border: none; background: transparent; }}
QTabBar::tab {{
    background: transparent;
    padding: 8px 16px;
    color: {text_secondary};
    border: none;
    border-bottom: 2px solid transparent;
    font-size: {font_size_sm};
    font-weight: 500;
}}
QTabBar::tab:selected {{
    color: {text_primary};
    border-bottom: 1.5px solid {brand};
}}
QTabBar::tab:hover:!selected {{ color: {text_primary}; }}

/* ═══ ToolTip ═══ */
QToolTip {{
    background: {tooltip_bg};
    color: {tooltip_text};
    border: none;
    border-radius: {radius_sm};
    padding: 4px 8px;
    font-size: {font_size_xs};
}}

/* ═══ GraphicsView 背景 ═══ */
QGraphicsView {{ background: {canvas_bg}; border: none; }}

/* ═══ QLineEdit / QTextEdit ═══ */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: {radius_md};
    padding: 4px 8px;
    color: {text_primary};
    font-size: {font_size};
    selection-background-color: {brand};
    selection-color: {text_on_brand};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {border_focus};
}}

/* ═══ QComboBox ═══ */
QComboBox {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: {radius_md};
    padding: 4px 8px;
    color: {text_primary};
    font-size: {font_size};
}}
QComboBox:hover {{ border-color: {border_input}; }}
QComboBox QAbstractItemView {{
    background: {bg_panel};
    border: 1px solid {border};
    selection-background-color: {brand};
    selection-color: {text_on_brand};
}}

/* ═══ QTreeWidget ═══ */
QTreeWidget {{
    background: transparent;
    border: none;
    outline: none;
}}
QTreeWidget::item {{
    padding: 4px 8px;
    border-radius: {radius_sm};
}}
QTreeWidget::item:selected {{
    background: {bg_hover};
    color: {text_primary};
}}
QTreeWidget::item:hover:!selected {{
    background: rgba(232, 230, 225, 120);
}}
"""


# ─────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────

_THEMES: dict[str, dict[str, str]] = {
    "light": LIGHT_TOKENS,
}

_FONTS_LOADED = False


def _load_bundled_fonts() -> None:
    global _FONTS_LOADED
    if _FONTS_LOADED:
        return
    font_dir = Path(__file__).resolve().parents[3] / "resources" / "fonts"
    for name in ("JetBrainsMono-Regular.ttf", "JetBrainsMono-Bold.ttf"):
        path = font_dir / name
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))
    _FONTS_LOADED = True


def _normalize(name: str | None) -> str:
    if not name:
        return "light"
    name = str(name).lower().strip()
    return name if name in _THEMES else "light"


def build_qss(theme: str = "light") -> str:
    """根据主题名生成完整 QSS 文本。"""
    tokens = _THEMES[_normalize(theme)]
    return _QSS_TEMPLATE.format(**tokens)


def apply_theme(app: Any, theme: str | None = None) -> str:
    """在 QApplication 上应用主题，并返回实际生效的主题名。"""
    app.setStyle("Fusion")
    _load_bundled_fonts()
    actual = _normalize(theme)
    app.setStyleSheet(build_qss(actual))
    return actual


def available_themes() -> list[str]:
    return list(_THEMES.keys())


__all__ = [
    "LIGHT_TOKENS",
    "apply_theme",
    "build_qss",
    "available_themes",
]
