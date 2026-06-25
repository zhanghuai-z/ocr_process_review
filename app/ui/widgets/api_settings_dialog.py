"""OCR 引擎设置对话框。"""
from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSpinBox,
    QVBoxLayout, QWidget,
)

from app.core.api_profiles import (
    FIXED_LAYOUT_PROFILE,
    KNOWN_API_ENDPOINT_SUFFIXES,
    get_api_model_profile_options,
    get_api_model_profile_url,
    normalize_api_base_url,
    resolve_api_endpoint_for_role,
)
from app.core.app_config import get_config, update_config
from app.core.paddle_v16_client import (
    PaddleV16LayoutClient,
    build_paddle_v16_optional_payload,
    is_paddle_v16_endpoint,
)
from app.utils.icon_manager import get_icon


# ------------------------------------------------------------------ stylesheet

_STYLE = """
QDialog {
    background: #F5F3EE;
    color: #2C2C2C;
    font-family: "Microsoft YaHei UI", "PingFang SC", "Source Han Sans CN", sans-serif;
}
QLabel { color: #2C2C2C; font-size: 14px; }
QLabel#heroEyebrow {
    color: #6B6B6B;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
QLabel#heroTitle {
    color: #2C2C2C;
    font-size: 20px;
    font-weight: 700;
}
QLabel#heroDesc {
    color: #6B6B6B;
    font-size: 12px;
    line-height: 1.5;
}
QLabel#sectionTitle {
    color: #2C2C2C;
    font-size: 13px;
    font-weight: 700;
    padding: 0 0 4px 0;
}
QLabel#sectionDesc { color: #6B6B6B; font-size: 13px; }
QLabel#fieldLabel { color: #4E4E4E; font-size: 12px; font-weight: 500; }
QLabel#noteLabel { color: #8C8A85; font-size: 12px; }
QLabel#pill {
    background: #ECE8DF;
    color: #2C2C2C;
    border: 1px solid #E7E2D8;
    border-radius: 10px;
    padding: 3px 10px;
    font-size: 13px;
    font-weight: 500;
}
QLabel#statusPill {
    background: #E7EDE3;
    color: #5C6B58;
    border: 1px solid #D8D2C8;
    border-radius: 9px;
    padding: 3px 8px;
    font-size: 13px;
    font-weight: 600;
}
QLabel#bannerInfo {
    background: #FFFDF8;
    color: #5C6B58;
    border: 1px solid #D8D2C8;
    border-radius: 8px;
    padding: 8px 10px;
}
QLabel#summaryTitle {
    color: #2C2C2C;
    font-size: 14px;
    font-weight: 600;
}
QLabel#summaryValue {
    color: #2C2C2C;
    font-size: 16px;
    font-weight: 700;
}
QLabel#summaryDesc {
    color: #6B6B6B;
    font-size: 13px;
    line-height: 1.45;
}
QLabel#footerNote {
    color: #A1A1A1;
    font-size: 13px;
    padding: 2px 0 0 2px;
}

QFrame#heroCard {
    background: transparent;
    border: none;
    border-radius: 0;
}
QFrame#card {
    background: transparent;
    border: none;
    border-radius: 0;
}
QFrame#settingsDetailPane {
    background: #F5F3EE;
    border: none;
    border-radius: 0;
}
QFrame#settingsShell {
    background: #F5F3EE;
    border: none;
    border-radius: 0;
}
QFrame#settingsTopBar {
    background: #F5F3EE;
    border-bottom: 1px solid #D9D4CA;
    border-top-left-radius: 0;
    border-top-right-radius: 0;
}
QFrame#modeCard {
    background: #FFFDF8;
    border: 1px solid #E7E2D8;
    border-radius: 12px;
}
QFrame#modeCard[selected="true"] {
    background: #ECE8DF;
    border: 1px solid #2C2C2C;
}
QFrame#sideInfoCard, QFrame#settingsNavPane {
    background: #F7F2EF;
    border-right: 1px solid #E1DCD3;
    border-radius: 0;
}
QFrame#bottomBar {
    background: #F5F3EE;
    border-top: 1px solid #D9D4CA;
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
}
QFrame#segmentControl {
    background: #F5F3EE;
    border: 1px solid #E3DED5;
    border-radius: 8px;
}

QScrollArea {
    background: #F5F3EE;
    border: none;
}
QWidget#settingsScrollContent {
    background: #F5F3EE;
}

QLineEdit, QComboBox, QSpinBox {
    background: #FFFFFF;
    border: 1px solid #DFDDD8;
    border-radius: 4px;
    padding: 6px 8px;
    min-height: 22px;
    font-size: 12px;
    color: #2C2C2C;
    selection-background-color: #2C2C2C;
    selection-color: #FFFFFF;
}
QSpinBox {
    padding-right: 8px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #5C6B58;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {
    background: #F5F3EE;
    color: #A1A1A1;
}
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #FFFDF8;
    border: 1px solid #D1CFCA;
    selection-background-color: #2C2C2C;
    selection-color: #FFFFFF;
}

QRadioButton {
    spacing: 6px;
    padding: 0;
    font-size: 14px;
    font-weight: 600;
    color: #2C2C2C;
}
QRadioButton::indicator { width: 16px; height: 16px; }

QPushButton {
    background: #FFFFFF;
    border: 1px solid #D1CFCA;
    border-radius: 6px;
    padding: 7px 15px;
    font-size: 12px;
    color: #2C2C2C;
}
QPushButton:hover { border-color: #5C6B58; color: #2C2C2C; background: #ECE8DF; }
QPushButton:pressed { background: #E7E2D8; }
QPushButton:disabled { color: #A1A1A1; border-color: #E7E2D8; }

QPushButton#primaryBtn {
    background: #2C2C2C;
    color: white;
    border: 1px solid #2C2C2C;
    font-weight: 600;
}
QPushButton#primaryBtn:hover {
    background: #1A1A1A;
    border-color: #1A1A1A;
    color: white;
}
QPushButton#primaryBtn:pressed { background: #000000; }

QPushButton#testBtn {
    background: #E7EDE3;
    color: #2C2C2C;
    border: 1px solid #C9D4C5;
}
QPushButton#testBtn:hover { background: #DDE7D8; }

QPushButton#subtleBtn {
    background: #FFFDF8;
    color: #6B6B6B;
    border: 1px solid #E7E2D8;
}
QPushButton#subtleBtn:hover {
    color: #2C2C2C;
    border-color: #5C6B58;
}
QPushButton#navItemBtn {
    background: transparent;
    border: none;
    border-radius: 4px;
    color: #6B6B6B;
    font-size: 12px;
    font-weight: 500;
    padding: 9px 12px;
    text-align: left;
}
QPushButton#navItemBtn:hover {
    background: #ECE8DF;
    color: #2C2C2C;
}
QPushButton#navItemBtn:checked {
    background: #ECE8DF;
    color: #2C2C2C;
    border-right: 3px solid #2C2C2C;
}
QPushButton#segmentBtn {
    background: transparent;
    border: none;
    border-radius: 5px;
    padding: 5px 12px;
    min-width: 54px;
    color: #5F5D58;
}
QPushButton#segmentBtn:hover {
    background: #ECE8DF;
    color: #2C2C2C;
}
QPushButton#segmentBtn:checked {
    background: #FFFFFF;
    border: 1px solid #D9D4CA;
    color: #2C2C2C;
    font-weight: 600;
}
QFrame#spinStepper {
    background: #F8F6F1;
    border: 1px solid #DFDDD8;
    border-radius: 4px;
}
QPushButton#spinStepBtn {
    background: transparent;
    border: none;
    border-radius: 3px;
    color: #5F5D58;
    font-size: 9px;
    font-weight: 700;
    padding: 0;
}
QPushButton#spinStepBtn:hover {
    background: #ECE8DF;
    color: #2C2C2C;
}
QPushButton#spinStepBtn:pressed {
    background: #E1DCD3;
}
"""


# ------------------------------------------------------------------ helpers

def _refresh_widget_style(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def _section_card(title: str, desc: str = "") -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("card")
    outer = QVBoxLayout(card)
    outer.setContentsMargins(0, 8, 0, 0)
    outer.setSpacing(10)

    title_lbl = QLabel(title)
    title_lbl.setObjectName("sectionTitle")
    outer.addWidget(title_lbl)
    if desc:
        desc_lbl = QLabel(desc)
        desc_lbl.setObjectName("sectionDesc")
        desc_lbl.setWordWrap(True)
        outer.addWidget(desc_lbl)
    return card, outer


def _form_row(label_text: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(12)
    lbl = QLabel(label_text)
    lbl.setObjectName("fieldLabel")
    lbl.setFixedWidth(112)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    row.addWidget(lbl)
    row.addWidget(widget, 1)
    return row


def _note_row(label: QLabel) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    row.addSpacing(124)
    row.addWidget(label, 1)
    return row


def _spinbox_with_stepper(spinbox: QSpinBox, *, spin_width: int, total_width: int) -> QWidget:
    spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spinbox.setFixedWidth(spin_width)

    container = QWidget()
    container.setFixedWidth(total_width)
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(5)
    row.addWidget(spinbox)

    stepper = QFrame()
    stepper.setObjectName("spinStepper")
    stepper.setFixedSize(23, 32)
    stepper_layout = QVBoxLayout(stepper)
    stepper_layout.setContentsMargins(1, 1, 1, 1)
    stepper_layout.setSpacing(0)

    up_btn = QPushButton("▲")
    down_btn = QPushButton("▼")
    for btn in (up_btn, down_btn):
        btn.setObjectName("spinStepBtn")
        btn.setFixedSize(19, 14)
        btn.setAutoRepeat(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
    up_btn.clicked.connect(spinbox.stepUp)
    down_btn.clicked.connect(spinbox.stepDown)
    stepper_layout.addWidget(up_btn)
    stepper_layout.addWidget(down_btn)
    row.addWidget(stepper)
    return container


def _pill(text: str, name: str = "pill") -> QLabel:
    label = QLabel(text)
    label.setObjectName(name)
    return label


class _ModeCard(QFrame):
    def __init__(self, radio: QRadioButton, title: str, desc: str, badge_text: str):
        super().__init__()
        self._radio = radio
        self.setObjectName("modeCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.addWidget(self._radio)
        top_row.addStretch()
        top_row.addWidget(_pill(badge_text))
        layout.addLayout(top_row)

        title_label = QLabel(title)
        title_label.setObjectName("summaryTitle")
        layout.addWidget(title_label)

        desc_label = QLabel(desc)
        desc_label.setObjectName("summaryDesc")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._radio.setChecked(True)
        super().mousePressEvent(event)


# ------------------------------------------------------------------ dialog

class ApiSettingsDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("apiSettingsDialog")
        self.setWindowTitle("OCR 引擎设置")
        self.setMinimumWidth(760)
        self.resize(820, 650)
        self.setStyleSheet(_STYLE)
        # 根据屏幕可用区域限制最大高度，确保小屏上也能完整操作
        _avail = QApplication.primaryScreen().availableGeometry()
        self.setMaximumHeight(int(_avail.height() * 0.90))
        self._build_ui()
        self._load_config()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        shell = QFrame()
        shell.setObjectName("settingsShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        outer.addWidget(shell, 1)

        top_bar = QFrame()
        top_bar.setObjectName("settingsTopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(22, 13, 18, 13)
        top_layout.setSpacing(12)
        top_title = QLabel("设置中心")
        top_title.setObjectName("heroTitle")
        top_layout.addWidget(top_title)
        top_layout.addStretch()
        shell_layout.addWidget(top_bar)

        content = QWidget()
        content.setObjectName("settingsContent")
        body = QHBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        shell_layout.addWidget(content, 1)

        nav = QFrame()
        nav.setObjectName("settingsNavPane")
        nav.setFixedWidth(148)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(10, 14, 10, 14)
        nav_layout.setSpacing(2)

        nav_title = QLabel("设置")
        nav_title.setObjectName("sectionTitle")
        nav_title.setContentsMargins(2, 0, 2, 8)
        nav_layout.addWidget(nav_title)
        self._settings_nav_buttons: dict[str, QPushButton] = {}
        nav_icons = {
            "engine": "api",
        }
        for idx, (key, text, enabled) in enumerate((
            ("engine", "识别引擎", True),
        )):
            btn = QPushButton(text)
            btn.setObjectName("navItemBtn")
            btn.setIcon(get_icon(nav_icons[key], color="#6B6B6B"))
            btn.setIconSize(QSize(17, 17))
            btn.setCheckable(True)
            btn.setEnabled(enabled)
            if idx == 0:
                btn.setChecked(True)
            self._settings_nav_buttons[key] = btn
            nav_layout.addWidget(btn)
        nav_layout.addStretch()
        body.addWidget(nav)

        detail = QFrame()
        detail.setObjectName("settingsDetailPane")
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(0)
        body.addWidget(detail, 1)

        # 右侧详情滚动容器
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_content.setObjectName("settingsScrollContent")
        root = QVBoxLayout(scroll_content)
        root.setContentsMargins(34, 20, 34, 18)
        root.setSpacing(12)
        scroll.setWidget(scroll_content)
        detail_layout.addWidget(scroll)

        bottom_bar = QFrame()
        bottom_bar.setObjectName("bottomBar")
        bottom_bar.setFixedHeight(52)
        bottom_bar_layout = QHBoxLayout(bottom_bar)
        bottom_bar_layout.setContentsMargins(18, 0, 18, 0)
        bottom_bar_layout.setSpacing(8)
        shell_layout.addWidget(bottom_bar)

        hero = QFrame()
        hero.setObjectName("heroCard")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(6)

        hero_title = QLabel("识别引擎")
        hero_title.setObjectName("heroTitle")
        hero_layout.addWidget(hero_title)

        hero_desc = QLabel("PaddleOCR-VL 版面分析 + CharOCR 文字识别。")
        hero_desc.setObjectName("heroDesc")
        hero_desc.setWordWrap(True)
        hero_layout.addWidget(hero_desc)

        root.addWidget(hero)

        mode_card, mode_layout = _section_card(
            "当前链路",
            "固定使用 PaddleOCR-VL-1.6/API 版面块 + CharOCR micro-recblock，不再暴露并列模式选择。",
        )
        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        self._radio_local = QRadioButton("")
        self._radio_api = QRadioButton("")
        self._radio_hanwang = QRadioButton("CharOCR 混合")
        self._local_mode_card = _ModeCard(
            self._radio_local,
            "",
            "",
            "",
        )
        self._api_mode_card = _ModeCard(
            self._radio_api,
            "",
            "",
            "",
        )
        self._hanwang_mode_card = _ModeCard(
            self._radio_hanwang,
            "VL1.6 + CharOCR",
            "API 提供 VL1.6 版面块；文字块走 CharOCR micro-recblock，公式/表格/图片保留 VL1.6。",
            "需地址/Token",
        )
        mode_row.addWidget(self._hanwang_mode_card, 1)
        mode_layout.addLayout(mode_row)
        root.addWidget(mode_card)
        self._mode_card = mode_card
        self._mode_card.hide()

        self._api_mode_notice = QLabel()
        self._api_mode_notice.setObjectName("bannerInfo")
        self._api_mode_notice.setWordWrap(True)
        self._api_mode_notice.hide()

        self._api_form_panel = QWidget()
        api_form = QVBoxLayout(self._api_form_panel)
        api_form.setContentsMargins(0, 0, 0, 0)
        api_form.setSpacing(9)

        self._api_model_combo = QComboBox()
        for key, label in get_api_model_profile_options():
            self._api_model_combo.addItem(label, key)
        self._api_model_combo.setCurrentIndex(-1)
        self._api_model_row = QWidget()
        self._api_model_row.setLayout(_form_row("官方模型", self._api_model_combo))
        self._api_model_row.hide()
        api_form.addWidget(self._api_model_row)

        self._model_note = QLabel()
        self._model_note.setObjectName("noteLabel")
        self._model_note.setWordWrap(True)
        self._model_note.hide()

        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("https://xxxxx.aistudio-app.com")
        self._url_edit.setClearButtonEnabled(True)
        api_form.addLayout(_form_row("服务地址", self._url_edit))

        self._url_note = QLabel()
        self._url_note.setObjectName("noteLabel")
        self._url_note.setWordWrap(True)
        api_form.addLayout(_note_row(self._url_note))

        token_row = QWidget()
        token_layout = QHBoxLayout(token_row)
        token_layout.setContentsMargins(0, 0, 0, 0)
        token_layout.setSpacing(6)
        self._token_edit = QLineEdit()
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_edit.setPlaceholderText("AiStudio 个人中心 → 访问令牌")
        token_layout.addWidget(self._token_edit, 1)
        self._btn_show_token = QPushButton("显示")
        self._btn_show_token.setObjectName("subtleBtn")
        self._btn_show_token.setCheckable(True)
        self._btn_show_token.setFixedWidth(58)
        self._btn_show_token.setToolTip("显示 / 隐藏 Token")
        self._btn_show_token.toggled.connect(self._toggle_token_visibility)
        token_layout.addWidget(self._btn_show_token)
        api_form.addLayout(_form_row("Token", token_row))

        test_row = QHBoxLayout()
        test_row.setContentsMargins(0, 2, 0, 0)
        test_row.addSpacing(124)
        self._btn_test = QPushButton("测试连接")
        self._btn_test.setObjectName("testBtn")
        self._btn_test.clicked.connect(self._test_connection)
        test_row.addWidget(self._btn_test)
        test_row.addStretch()
        api_form.addLayout(test_row)

        self._api_card, api_layout = _section_card("连接参数", "")
        api_layout.addWidget(self._api_form_panel)
        root.addWidget(self._api_card)

        performance_form = api_form

        timeout_row = QWidget()
        timeout_layout = QHBoxLayout(timeout_row)
        timeout_layout.setContentsMargins(0, 0, 0, 0)
        timeout_layout.setSpacing(0)
        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(5, 600)
        self._timeout_spin.setSuffix(" 秒")
        timeout_layout.addWidget(
            _spinbox_with_stepper(self._timeout_spin, spin_width=88, total_width=116)
        )
        timeout_layout.addStretch()
        self._timeout_row = timeout_row
        performance_form.addLayout(_form_row("请求超时", self._timeout_row))

        concurrency_row = QWidget()
        concurrency_layout = QHBoxLayout(concurrency_row)
        concurrency_layout.setContentsMargins(0, 0, 0, 0)
        concurrency_layout.setSpacing(0)
        self._layout_concurrency_spin = QSpinBox()
        self._layout_concurrency_spin.setRange(1, 10)
        self._layout_concurrency_spin.setToolTip("多页版面分析时同时提交的 Paddle jobs 数。")
        concurrency_layout.addWidget(
            _spinbox_with_stepper(self._layout_concurrency_spin, spin_width=62, total_width=90)
        )
        concurrency_layout.addStretch()
        performance_form.addLayout(_form_row("版面请求并发", concurrency_row))

        ocr_concurrency_row = QWidget()
        ocr_concurrency_layout = QHBoxLayout(ocr_concurrency_row)
        ocr_concurrency_layout.setContentsMargins(0, 0, 0, 0)
        ocr_concurrency_layout.setSpacing(0)
        self._ocr_page_concurrency_spin = QSpinBox()
        self._ocr_page_concurrency_spin.setRange(1, 20)
        self._ocr_page_concurrency_spin.setToolTip("CharOCR 多页文字识别并发数。")
        ocr_concurrency_layout.addWidget(
            _spinbox_with_stepper(self._ocr_page_concurrency_spin, spin_width=62, total_width=90)
        )
        ocr_concurrency_layout.addStretch()
        performance_form.addLayout(_form_row("CharOCR 页并发", ocr_concurrency_row))

        network_row = QWidget()
        network_layout = QHBoxLayout(network_row)
        network_layout.setContentsMargins(0, 0, 0, 0)
        network_layout.setSpacing(0)
        segment = QFrame()
        segment.setObjectName("segmentControl")
        segment_layout = QHBoxLayout(segment)
        segment_layout.setContentsMargins(3, 3, 3, 3)
        segment_layout.setSpacing(2)
        self._network_mode_buttons: dict[str, QPushButton] = {}
        for mode, text in (
            ("auto", "自动"),
            ("env_proxy", "系统代理"),
            ("direct", "直连"),
        ):
            btn = QPushButton(text)
            btn.setObjectName("segmentBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, value=mode: self._set_network_mode(value))
            self._network_mode_buttons[mode] = btn
            segment_layout.addWidget(btn)
        segment.setToolTip("PaddleOCR-VL jobs API 网络路径。")
        network_layout.addWidget(segment, 0)
        network_layout.addStretch()
        api_form.addLayout(_form_row("Paddle 网络", network_row))

        self._summary_model = QLabel()
        self._summary_desc = QLabel()
        self._summary_endpoint_kind = QLabel()
        self._summary_endpoint = QLabel()

        root.addStretch()

        # 按钮放入底部固定栏（已在上方构建好 bottom_bar_layout）
        bottom_bar_layout.addStretch()
        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_ok = QPushButton("保存")
        self._btn_ok.setObjectName("primaryBtn")
        self._btn_ok.clicked.connect(self._save_and_accept)
        bottom_bar_layout.addWidget(self._btn_cancel)
        bottom_bar_layout.addWidget(self._btn_ok)

        def _activate_nav(active_key: str, target: QWidget) -> None:
            for key, btn in self._settings_nav_buttons.items():
                btn.setChecked(key == active_key)
            scroll.ensureWidgetVisible(target, 0, 16)

        for key, target in (
            ("engine", hero),
        ):
            self._settings_nav_buttons[key].clicked.connect(
                lambda _checked=False, k=key, t=target: _activate_nav(k, t)
            )

        self._radio_local.toggled.connect(self._on_mode_changed)
        self._radio_api.toggled.connect(self._on_mode_changed)
        self._radio_hanwang.toggled.connect(self._on_mode_changed)
        self._api_model_combo.currentIndexChanged.connect(self._on_api_model_changed)
        self._url_edit.editingFinished.connect(self._sync_model_from_url)

    # ------------------------------------------------------------------ logic

    def _load_config(self) -> None:
        cfg = get_config()
        self._radio_hanwang.setChecked(True)
        self._api_model_combo.blockSignals(True)
        self._api_model_combo.setCurrentIndex(-1)
        self._api_model_combo.blockSignals(False)

        api_url = cfg.get("api_url", "")
        self._url_edit.setText(api_url)

        self._token_edit.setText(cfg.get("api_token", ""))
        self._timeout_spin.setValue(cfg.get("api_timeout", 30))
        self._layout_concurrency_spin.setValue(int(cfg.get("layout_concurrency", 2)))
        self._ocr_page_concurrency_spin.setValue(int(cfg.get("ocr_page_concurrency", 2)))
        network_mode = str(cfg.get("paddle_api_network_mode", "auto") or "auto")
        self._set_network_mode(network_mode)
        self._btn_show_token.setChecked(False)
        self._on_mode_changed()

    def _selected_mode(self) -> str:
        return "hanwang"

    def _set_mode_card_selected(self, card: QFrame, selected: bool) -> None:
        card.setProperty("selected", selected)
        _refresh_widget_style(card)

    def _refresh_api_preview(self) -> None:
        selected_mode = self._selected_mode()
        url = self._url_edit.text().strip()
        base_url = normalize_api_base_url(url)
        layout_endpoint = resolve_api_endpoint_for_role(
            url,
            profile=FIXED_LAYOUT_PROFILE,
            role="layout",
        ) if url else ""
        ocr_endpoint = resolve_api_endpoint_for_role(
            url,
            profile="pp-ocrv5",
            role="ocr",
        ) if url else ""

        self._model_note.setText("")
        self._summary_model.setText("汉王混合链路：PaddleOCR-VL-1.6 + micro-recblock")
        self._summary_desc.setText(
            "版面分析依赖 PaddleOCR-VL-1.6/API 的 parsing_res_list；文字类 block 交给 Hanwang micro-recblock，"
            "公式、表格、图片类 block 直接保留 VL1.6 内容。"
        )

        if not url:
            self._url_note.setText("请填写不带端点后缀的服务根地址。")
            self._summary_endpoint_kind.setText("端点待填写")
            self._summary_endpoint.setText("尚未填写 API 地址。")
        elif any(url.rstrip("/").endswith(suffix) for suffix in KNOWN_API_ENDPOINT_SUFFIXES):
            endpoint_type = next(
                suffix for suffix in KNOWN_API_ENDPOINT_SUFFIXES
                if url.rstrip("/").endswith(suffix)
            )
            self._url_note.setText(f"当前地址包含 {endpoint_type}，保存时会自动改为基础地址：{base_url}")
            self._summary_endpoint_kind.setText("保存为基础地址")
            self._summary_endpoint.setText(f"base: {base_url}\nlayout: {layout_endpoint}\nocr: {ocr_endpoint}")
        else:
            self._url_note.setText("当前地址未包含端点后缀，会按固定模型链自动补全。")
            self._summary_endpoint_kind.setText("固定链端点")
            self._summary_endpoint.setText(f"layout: {layout_endpoint}\nocr: {ocr_endpoint}")

        self._api_mode_notice.setText(
            "需要 API 地址与 Token：PaddleOCR-VL 获取版面，CharOCR 识别文字。"
        )

    def _on_mode_changed(self) -> None:
        self._radio_hanwang.setChecked(True)
        self._api_form_panel.setEnabled(True)
        self._btn_test.setEnabled(True)
        self._set_mode_card_selected(self._local_mode_card, self._radio_local.isChecked())
        self._set_mode_card_selected(self._api_mode_card, self._radio_api.isChecked())
        self._set_mode_card_selected(self._hanwang_mode_card, self._radio_hanwang.isChecked())
        self._refresh_api_preview()

    def _on_api_model_changed(self) -> None:
        profile_key = self._api_model_combo.currentData()
        if profile_key:
            self._url_edit.setText(get_api_model_profile_url(profile_key))
        self._refresh_api_preview()

    def _sync_model_from_url(self) -> None:
        self._api_model_combo.blockSignals(True)
        self._api_model_combo.setCurrentIndex(-1)
        self._api_model_combo.blockSignals(False)
        self._refresh_api_preview()

    def _set_network_mode(self, mode: str) -> None:
        if mode not in self._network_mode_buttons:
            mode = "auto"
        for value, button in self._network_mode_buttons.items():
            button.blockSignals(True)
            button.setChecked(value == mode)
            button.blockSignals(False)
            _refresh_widget_style(button)

    def _selected_network_mode(self) -> str:
        for mode, button in self._network_mode_buttons.items():
            if button.isChecked():
                return mode
        return "auto"

    def _toggle_token_visibility(self, checked: bool) -> None:
        if checked:
            self._token_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self._btn_show_token.setText("隐藏")
        else:
            self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self._btn_show_token.setText("显示")

    def _save_and_accept(self) -> None:
        api_url = normalize_api_base_url(self._url_edit.text())
        mode = self._selected_mode()
        update_config(
            mode=mode,
            api_model_profile="",
            api_url=api_url,
            api_token=self._token_edit.text().strip(),
            api_timeout=self._timeout_spin.value(),
            api_layout_model_name="",
            layout_concurrency=self._layout_concurrency_spin.value(),
            ocr_page_concurrency=self._ocr_page_concurrency_spin.value(),
            paddle_api_network_mode=self._selected_network_mode(),
        )
        self.accept()

    # ------------------------------------------------------------------ test

    def _test_connection(self) -> None:
        import requests

        url = resolve_api_endpoint_for_role(
            self._url_edit.text().strip(),
            profile=FIXED_LAYOUT_PROFILE,
            role="layout",
        )
        if not url:
            QMessageBox.warning(self, "提示", "请先填写 API 地址。")
            return
        if not is_paddle_v16_endpoint(url):
            QMessageBox.warning(self, "提示", "版面分析只支持 PaddleOCR-VL-1.6 jobs API。")
            return
        token = self._token_edit.text().strip()
        timeout = self._timeout_spin.value()

        try:
            import cv2
            import numpy as np
            img = np.full((200, 400, 3), 240, dtype=np.uint8)
            cv2.putText(img, "OCR Test", (80, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (30, 30, 30), 2)
        except Exception:
            QMessageBox.critical(self, "错误", "无法生成测试图像，请确认 opencv-python 已安装。")
            return

        self._btn_test.setEnabled(False)
        self._btn_test.setText("测试中…")
        try:
            client = PaddleV16LayoutClient(
                jobs_url=url,
                token=token,
                request_timeout=timeout,
                poll_timeout=timeout,
                network_mode=self._selected_network_mode(),
            )
            body = client.analyze_image(
                img,
                optional_payload=build_paddle_v16_optional_payload(),
            )
            code = 200
            err_code = body.get("errorCode", -1)
            err_msg = body.get("errorMsg", "")

            if code == 200 and err_code == 0:
                QMessageBox.information(
                    self, "测试成功",
                    "✓ 连接正常，PaddleOCR-VL-1.6 jobs API 响应成功！",
                )
            elif code in (401, 403) or err_code in (401, 403):
                QMessageBox.warning(
                    self, "鉴权失败",
                    f"HTTP {code}：Token 无效或已过期，请检查访问 Token。",
                )
            else:
                QMessageBox.warning(
                    self, "连接异常",
                    f"HTTP {code}，errorCode={err_code}\n{err_msg}",
                )
        except requests.exceptions.ConnectionError:
            QMessageBox.critical(self, "连接失败", f"无法连接到：\n{url}\n请检查 URL 或网络。")
        except requests.exceptions.Timeout:
            QMessageBox.critical(self, "超时", f"请求超时（>{timeout}s）。")
        except Exception as exc:
            QMessageBox.critical(self, "错误", str(exc))
        finally:
            self._btn_test.setEnabled(True)
            self._btn_test.setText("测试连接")
