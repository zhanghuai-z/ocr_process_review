"""OCR 引擎设置对话框。

特性：
- 模式切换：本地 PaddleOCR ↔ 云端 API
- 4 个官方模型预设（PP-OCRv5 / PP-StructureV3 / PaddleOCR-VL / PaddleOCR-VL-1.5）
  下拉选中后自动填 URL；URL 修改后自动反向匹配
- 测试连接：发送一张小测试图，根据 errorCode + result 类型判断
- 视觉风格参考 ui.jpg：浅色卡片 + 天蓝主色 + 概览式信息面板
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSpinBox,
    QVBoxLayout, QWidget,
)

from app.core.api_profiles import (
    API_MODEL_PROFILES,
    KNOWN_API_ENDPOINT_SUFFIXES,
    detect_api_result_kind,
    get_api_model_profile_options,
    get_api_model_profile_url,
    match_api_model_profile_from_url,
    resolve_api_endpoint,
)
from app.core.ocr_config import get_config, update_config


# ------------------------------------------------------------------ profiles


def build_api_payload(file_b64: str, file_type: int) -> dict[str, object]:
    return {"file": file_b64, "fileType": file_type}


# ------------------------------------------------------------------ stylesheet

_STYLE = """
QDialog {
    background: #f5f7fb;
    color: #222;
    font-family: "Microsoft YaHei UI", "PingFang SC", "Source Han Sans CN", sans-serif;
}
QLabel { color: #222; font-size: 13px; }
QLabel#heroEyebrow {
    color: #1a73e8;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
QLabel#heroTitle {
    color: #17324d;
    font-size: 20px;
    font-weight: 700;
}
QLabel#heroDesc {
    color: #62748a;
    font-size: 13px;
    line-height: 1.5;
}
QLabel#sectionTitle {
    color: #17324d;
    font-size: 14px;
    font-weight: 700;
    padding: 0 0 4px 0;
}
QLabel#sectionDesc { color: #7b8794; font-size: 12px; }
QLabel#fieldLabel { color: #526071; font-size: 13px; font-weight: 500; }
QLabel#noteLabel { color: #8191a4; font-size: 12px; }
QLabel#pill {
    background: #eef5ff;
    color: #1a73e8;
    border: 1px solid #d6e6ff;
    border-radius: 10px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 500;
}
QLabel#statusPill {
    background: #f5f9ff;
    color: #1a73e8;
    border: 1px solid #d9e9ff;
    border-radius: 9px;
    padding: 3px 8px;
    font-size: 12px;
    font-weight: 600;
}
QLabel#bannerInfo {
    background: #f0f6ff;
    color: #37618f;
    border: 1px solid #d6e6ff;
    border-radius: 8px;
    padding: 8px 10px;
}
QLabel#summaryTitle {
    color: #17324d;
    font-size: 13px;
    font-weight: 600;
}
QLabel#summaryValue {
    color: #17324d;
    font-size: 15px;
    font-weight: 700;
}
QLabel#summaryDesc {
    color: #8191a4;
    font-size: 12px;
    line-height: 1.45;
}
QLabel#footerNote {
    color: #8191a4;
    font-size: 12px;
    padding: 2px 0 0 2px;
}

QFrame#heroCard {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #ffffff, stop:1 #eef5ff
    );
    border: 1px solid #dce8f8;
    border-radius: 16px;
}
QFrame#card {
    background: #ffffff;
    border: 1px solid #e3e8ef;
    border-radius: 14px;
}
QFrame#modeCard {
    background: #ffffff;
    border: 1px solid #dbe4ef;
    border-radius: 12px;
}
QFrame#modeCard[selected="true"] {
    background: #f0f6ff;
    border: 1px solid #1a73e8;
}
QFrame#sideInfoCard {
    background: #f8fbff;
    border: 1px solid #dce8f8;
    border-radius: 12px;
}

QLineEdit, QComboBox, QSpinBox {
    background: #ffffff;
    border: 1px solid #d6dde6;
    border-radius: 8px;
    padding: 8px 10px;
    min-height: 24px;
    font-size: 13px;
    color: #222;
    selection-background-color: #cfe2ff;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #1a73e8;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {
    background: #f2f5f8;
    color: #98a4b3;
}
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #ffffff;
    border: 1px solid #d6dde6;
    selection-background-color: #e3f0ff;
    selection-color: #1a73e8;
}

QRadioButton {
    spacing: 6px;
    padding: 0;
    font-size: 13px;
    font-weight: 600;
    color: #17324d;
}
QRadioButton::indicator { width: 16px; height: 16px; }

QPushButton {
    background: #ffffff;
    border: 1px solid #d6dde6;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    color: #344054;
}
QPushButton:hover { border-color: #1a73e8; color: #1a73e8; }
QPushButton:pressed { background: #f0f6ff; }
QPushButton:disabled { color: #aaa; border-color: #e0e4eb; }

QPushButton#primaryBtn {
    background: #1a73e8;
    color: white;
    border: 1px solid #1a73e8;
    font-weight: 600;
}
QPushButton#primaryBtn:hover {
    background: #1666cf;
    border-color: #1666cf;
    color: white;
}
QPushButton#primaryBtn:pressed { background: #1357a8; }

QPushButton#testBtn {
    background: #f0f6ff;
    color: #1a73e8;
    border: 1px solid #b8d4ff;
}
QPushButton#testBtn:hover { background: #e0eeff; }

QPushButton#subtleBtn {
    background: #ffffff;
    color: #526071;
    border: 1px solid #dce3eb;
}
QPushButton#subtleBtn:hover {
    color: #1a73e8;
    border-color: #1a73e8;
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
    outer.setContentsMargins(18, 18, 18, 18)
    outer.setSpacing(12)

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
    row.setSpacing(10)
    lbl = QLabel(label_text)
    lbl.setObjectName("fieldLabel")
    lbl.setFixedWidth(86)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    row.addWidget(lbl)
    row.addWidget(widget, 1)
    return row


def _note_row(label: QLabel) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    row.addSpacing(96)
    row.addWidget(label, 1)
    return row


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
        self.setWindowTitle("OCR 引擎设置")
        self.setMinimumWidth(820)
        self.setStyleSheet(_STYLE)
        # 根据屏幕可用区域限制最大高度，确保小屏上也能完整操作
        _avail = QApplication.primaryScreen().availableGeometry()
        self.setMaximumHeight(int(_avail.height() * 0.90))
        self._build_ui()
        self._load_config()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        # 外层布局：滚动区 + 固定底部按钮栏
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 滚动容器
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        root = QVBoxLayout(scroll_content)
        root.setContentsMargins(20, 20, 20, 12)
        root.setSpacing(14)
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, 1)

        # 底部固定栏（分隔线 + 按钮）
        bottom_bar = QFrame()
        bottom_bar.setObjectName("bottomBar")
        bottom_bar.setStyleSheet(
            "QFrame#bottomBar { background:#f5f7fb;"
            " border-top: 1px solid #e3e8ef; padding: 0; }"
        )
        bottom_bar.setFixedHeight(56)
        bottom_bar_layout = QHBoxLayout(bottom_bar)
        bottom_bar_layout.setContentsMargins(20, 0, 20, 0)
        bottom_bar_layout.setSpacing(8)
        outer.addWidget(bottom_bar)

        hero = QFrame()
        hero.setObjectName("heroCard")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(20, 18, 20, 18)
        hero_layout.setSpacing(10)

        hero_eyebrow = QLabel("ENGINE SETTINGS")
        hero_eyebrow.setObjectName("heroEyebrow")
        hero_layout.addWidget(hero_eyebrow)

        hero_title = QLabel("OCR 引擎与 API 连接")
        hero_title.setObjectName("heroTitle")
        hero_layout.addWidget(hero_title)

        hero_desc = QLabel(
            "保留 GPT 已接入的官方模型预设、自动填 URL 与反向匹配能力，"
            "同时把配置阅读顺序整理成“模式选择 → 接口配置 → 摘要确认”。"
        )
        hero_desc.setObjectName("heroDesc")
        hero_desc.setWordWrap(True)
        hero_layout.addWidget(hero_desc)

        hero_pills = QHBoxLayout()
        hero_pills.setContentsMargins(0, 0, 0, 0)
        hero_pills.setSpacing(8)
        hero_pills.addWidget(_pill("4 个官方模型预设"))
        hero_pills.addWidget(_pill("URL 自动识别"))
        hero_pills.addWidget(_pill("内置测试连接"))
        hero_pills.addStretch()
        hero_layout.addLayout(hero_pills)
        root.addWidget(hero)

        mode_card, mode_layout = _section_card(
            "工作模式",
            "优先先选运行方式，再决定是否填写 API 参数，避免操作顺序混乱。",
        )
        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        self._radio_local = QRadioButton("本地模型")
        self._radio_api = QRadioButton("云端 API")
        self._local_mode_card = _ModeCard(
            self._radio_local,
            "本地 PaddleOCR",
            "适合离线场景，使用本机模型完成版面分析与 OCR。",
            "无需 Token",
        )
        self._api_mode_card = _ModeCard(
            self._radio_api,
            "AiStudio / 自部署 API",
            "适合快速切换官方模型或连接自有服务端点。",
            "支持预设",
        )
        mode_row.addWidget(self._local_mode_card, 1)
        mode_row.addWidget(self._api_mode_card, 1)
        mode_layout.addLayout(mode_row)
        root.addWidget(mode_card)

        self._api_card, api_layout = _section_card(
            "API 连接",
            "下拉框保留 4 个官方模型预设；切换预设时自动回填完整端点，手动改 URL 时会尝试反向识别。",
        )

        self._api_mode_notice = QLabel()
        self._api_mode_notice.setObjectName("bannerInfo")
        self._api_mode_notice.setWordWrap(True)
        api_layout.addWidget(self._api_mode_notice)

        api_body = QHBoxLayout()
        api_body.setSpacing(14)

        self._api_form_panel = QWidget()
        api_form = QVBoxLayout(self._api_form_panel)
        api_form.setContentsMargins(0, 0, 0, 0)
        api_form.setSpacing(10)

        self._api_model_combo = QComboBox()
        for key, label in get_api_model_profile_options():
            spec = API_MODEL_PROFILES[key]
            self._api_model_combo.addItem(f"{label}  —  {spec['desc']}", key)
        self._api_model_combo.setCurrentIndex(-1)
        api_form.addLayout(_form_row("官方模型", self._api_model_combo))

        self._model_note = QLabel()
        self._model_note.setObjectName("noteLabel")
        self._model_note.setWordWrap(True)
        api_form.addLayout(_note_row(self._model_note))

        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("https://xxxxx.aistudio-app.com/layout-parsing")
        self._url_edit.setClearButtonEnabled(True)
        api_form.addLayout(_form_row("API 地址", self._url_edit))

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
        self._btn_show_token.setFixedWidth(64)
        self._btn_show_token.setToolTip("显示 / 隐藏 Token")
        self._btn_show_token.toggled.connect(self._toggle_token_visibility)
        token_layout.addWidget(self._btn_show_token)
        api_form.addLayout(_form_row("访问 Token", token_row))

        timeout_row = QWidget()
        timeout_layout = QHBoxLayout(timeout_row)
        timeout_layout.setContentsMargins(0, 0, 0, 0)
        timeout_layout.setSpacing(0)
        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(5, 120)
        self._timeout_spin.setSuffix(" 秒")
        self._timeout_spin.setFixedWidth(140)
        timeout_layout.addWidget(self._timeout_spin)
        timeout_layout.addStretch()
        api_form.addLayout(_form_row("请求超时", timeout_row))

        test_row = QHBoxLayout()
        test_row.setContentsMargins(0, 4, 0, 0)
        test_row.addSpacing(96)
        self._btn_test = QPushButton("测试连接")
        self._btn_test.setObjectName("testBtn")
        self._btn_test.clicked.connect(self._test_connection)
        test_row.addWidget(self._btn_test)
        test_row.addStretch()
        api_form.addLayout(test_row)

        api_body.addWidget(self._api_form_panel, 3)

        side_card = QFrame()
        side_card.setObjectName("sideInfoCard")
        side_layout = QVBoxLayout(side_card)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(10)

        summary_title = QLabel("当前配置摘要")
        summary_title.setObjectName("summaryTitle")
        side_layout.addWidget(summary_title)

        self._summary_mode = QLabel()
        self._summary_mode.setObjectName("summaryValue")
        side_layout.addWidget(self._summary_mode)

        self._summary_model = QLabel()
        self._summary_model.setObjectName("summaryTitle")
        side_layout.addWidget(self._summary_model)

        self._summary_desc = QLabel()
        self._summary_desc.setObjectName("summaryDesc")
        self._summary_desc.setWordWrap(True)
        side_layout.addWidget(self._summary_desc)

        self._summary_endpoint_kind = QLabel()
        self._summary_endpoint_kind.setObjectName("statusPill")
        side_layout.addWidget(self._summary_endpoint_kind)

        self._summary_endpoint = QLabel()
        self._summary_endpoint.setObjectName("summaryDesc")
        self._summary_endpoint.setWordWrap(True)
        side_layout.addWidget(self._summary_endpoint)

        hint_title = QLabel("操作提示")
        hint_title.setObjectName("summaryTitle")
        side_layout.addWidget(hint_title)

        for text in (
            "优先选择官方模型，可避免端点路径填错。",
            "若粘贴的是服务根地址，测试连接会自动补全 /layout-parsing。",
            "切回本地模式后，API 配置会保留，方便后续再次启用。",
        ):
            hint = QLabel(f"• {text}")
            hint.setObjectName("summaryDesc")
            hint.setWordWrap(True)
            side_layout.addWidget(hint)

        side_layout.addStretch()
        api_body.addWidget(side_card, 2)
        api_layout.addLayout(api_body)
        root.addWidget(self._api_card)

        root.addStretch()

        self._footer_note = QLabel(
            "保存时会继续写入 api_model_profile，并清空旧的 api_layout_model_name，"
            "避免回退到旧版手填模型名逻辑。"
        )
        self._footer_note.setObjectName("footerNote")
        self._footer_note.setWordWrap(True)
        root.addWidget(self._footer_note)

        # 按钮放入底部固定栏（已在上方构建好 bottom_bar_layout）
        bottom_bar_layout.addStretch()
        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_ok = QPushButton("保存")
        self._btn_ok.setObjectName("primaryBtn")
        self._btn_ok.clicked.connect(self._save_and_accept)
        bottom_bar_layout.addWidget(self._btn_cancel)
        bottom_bar_layout.addWidget(self._btn_ok)

        self._radio_local.toggled.connect(self._on_mode_changed)
        self._radio_api.toggled.connect(self._on_mode_changed)
        self._api_model_combo.currentIndexChanged.connect(self._on_api_model_changed)
        self._url_edit.editingFinished.connect(self._sync_model_from_url)

    # ------------------------------------------------------------------ logic

    def _load_config(self) -> None:
        cfg = get_config()
        if cfg["mode"] == "api":
            self._radio_api.setChecked(True)
        else:
            self._radio_local.setChecked(True)

        profile_key = cfg.get("api_model_profile", "") or ""
        if not profile_key:
            profile_key = match_api_model_profile_from_url(cfg.get("api_url", ""))
        index = self._api_model_combo.findData(profile_key) if profile_key else -1
        self._api_model_combo.blockSignals(True)
        self._api_model_combo.setCurrentIndex(index if index >= 0 else -1)
        self._api_model_combo.blockSignals(False)

        api_url = cfg.get("api_url", "")
        if not api_url and index >= 0:
            api_url = get_api_model_profile_url(profile_key)
        self._url_edit.setText(api_url)

        self._token_edit.setText(cfg.get("api_token", ""))
        self._timeout_spin.setValue(cfg.get("api_timeout", 30))
        self._btn_show_token.setChecked(False)
        self._on_mode_changed()

    def _set_mode_card_selected(self, card: QFrame, selected: bool) -> None:
        card.setProperty("selected", selected)
        _refresh_widget_style(card)

    def _refresh_api_preview(self) -> None:
        profile_key = self._api_model_combo.currentData()
        url = self._url_edit.text().strip()
        resolved = resolve_api_endpoint(
            url,
            default_suffix="/layout-parsing",
            profile=profile_key,
        ) if url else ""

        if profile_key and profile_key in API_MODEL_PROFILES:
            spec = API_MODEL_PROFILES[profile_key]
            self._model_note.setText(
                f"已选官方预设：{spec['label']}，会自动填入对应端点。"
            )
            self._summary_model.setText(spec["label"])
            self._summary_desc.setText(spec["desc"])
        else:
            self._model_note.setText(
                "当前未匹配官方预设。可直接填写自部署 Serving 地址，保存时会按 URL 原样使用。"
            )
            self._summary_model.setText("自定义 API 地址")
            self._summary_desc.setText("未匹配到官方模型预设，将根据当前 URL 直接请求。")

        if not url:
            self._url_note.setText("可直接粘贴完整端点，也可只填服务根地址。")
            self._summary_endpoint_kind.setText("端点待填写")
            self._summary_endpoint.setText("尚未填写 API 地址。")
        elif any(url.rstrip("/").endswith(suffix) for suffix in KNOWN_API_ENDPOINT_SUFFIXES):
            endpoint_type = "/ocr" if url.rstrip("/").endswith("/ocr") else "/layout-parsing"
            self._url_note.setText("当前地址已包含完整端点，测试连接时将直接使用。")
            self._summary_endpoint_kind.setText(f"当前端点：{endpoint_type}")
            self._summary_endpoint.setText(url.rstrip("/"))
        else:
            self._url_note.setText(
                "当前地址未包含端点后缀，测试连接时会自动补全 /layout-parsing。"
            )
            self._summary_endpoint_kind.setText("自动补全：/layout-parsing")
            self._summary_endpoint.setText(resolved)

        if self._radio_api.isChecked():
            self._summary_mode.setText("当前模式：云端 API")
            self._api_mode_notice.setText(
                "当前处于云端 API 模式，请确认地址、Token 与模型预设一致后再开始识别。"
            )
        else:
            self._summary_mode.setText("当前模式：本地 PaddleOCR")
            self._api_mode_notice.setText(
                "当前处于本地模式，以下 API 配置会被保留，但本次识别不会调用远端服务。"
            )

    def _on_mode_changed(self) -> None:
        api_enabled = self._radio_api.isChecked()
        self._api_form_panel.setEnabled(api_enabled)
        self._set_mode_card_selected(self._local_mode_card, self._radio_local.isChecked())
        self._set_mode_card_selected(self._api_mode_card, self._radio_api.isChecked())
        self._refresh_api_preview()

    def _on_api_model_changed(self) -> None:
        profile_key = self._api_model_combo.currentData()
        if profile_key:
            self._url_edit.setText(get_api_model_profile_url(profile_key))
        self._refresh_api_preview()

    def _sync_model_from_url(self) -> None:
        profile_key = match_api_model_profile_from_url(self._url_edit.text().strip())
        index = self._api_model_combo.findData(profile_key) if profile_key else -1
        self._api_model_combo.blockSignals(True)
        self._api_model_combo.setCurrentIndex(index if index >= 0 else -1)
        self._api_model_combo.blockSignals(False)
        self._refresh_api_preview()

    def _toggle_token_visibility(self, checked: bool) -> None:
        if checked:
            self._token_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self._btn_show_token.setText("隐藏")
        else:
            self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self._btn_show_token.setText("显示")

    def _save_and_accept(self) -> None:
        mode = "api" if self._radio_api.isChecked() else "local"
        api_url = self._url_edit.text().strip().rstrip("/")
        api_model_profile = (
            self._api_model_combo.currentData()
            or match_api_model_profile_from_url(api_url)
            or ""
        )
        update_config(
            mode=mode,
            api_model_profile=api_model_profile,
            api_url=api_url,
            api_token=self._token_edit.text().strip(),
            api_timeout=self._timeout_spin.value(),
            api_layout_model_name="",
        )
        self.accept()

    # ------------------------------------------------------------------ test

    def _test_connection(self) -> None:
        import base64
        import requests

        url = resolve_api_endpoint(
            self._url_edit.text().strip(),
            default_suffix="/layout-parsing",
            profile=self._api_model_combo.currentData(),
        )
        if not url:
            QMessageBox.warning(self, "提示", "请先填写 API 地址。")
            return
        token = self._token_edit.text().strip()
        timeout = self._timeout_spin.value()

        try:
            import cv2
            import numpy as np
            img = np.full((200, 400, 3), 240, dtype=np.uint8)
            cv2.putText(img, "OCR Test", (80, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (30, 30, 30), 2)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            file_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
        except Exception:
            file_b64 = ""

        if not file_b64:
            QMessageBox.critical(self, "错误", "无法生成测试图像，请确认 opencv-python 已安装。")
            return

        headers: dict = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"token {token}"

        self._btn_test.setEnabled(False)
        self._btn_test.setText("测试中…")
        try:
            payload = build_api_payload(file_b64, 1)
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            code = resp.status_code
            try:
                body = resp.json()
                err_code = body.get("errorCode", -1)
                err_msg = body.get("errorMsg", "")
            except Exception:
                body = {}
                err_code = -1
                err_msg = resp.text[:300]

            if code == 200 and err_code == 0:
                kind = detect_api_result_kind(body)
                kind_label = {
                    "ocr": "PP-OCRv5 /ocr",
                    "layout": "PP-StructureV3 / VL /layout-parsing",
                    "unknown": "未知结构（请确认端点是否正确）",
                }.get(kind, kind)
                QMessageBox.information(
                    self, "测试成功",
                    f"✓ 连接正常，API 响应成功！\n\n响应类型：{kind_label}",
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
