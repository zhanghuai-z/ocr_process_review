"""OCR 引擎设置对话框（光感 light 主题）。

特性：
- 模式切换：本地 PaddleOCR ↔ 云端 API
- 4 个官方模型预设（PP-OCRv5 / PP-StructureV3 / PaddleOCR-VL / PaddleOCR-VL-1.5）
  下拉选中后自动填 URL；URL 修改后自动反向匹配
- 测试连接：发送一张小测试图，根据 errorCode + result 类型判断
- 视觉风格参考 ui.jpg：浅色卡片 + 天蓝主色 + 圆角 + 区段化布局
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QRadioButton, QSizePolicy,
    QSpinBox, QVBoxLayout, QWidget,
)

from app.core.ocr_config import get_config, update_config


# ------------------------------------------------------------------ profiles

KNOWN_API_ENDPOINT_SUFFIXES = ("/ocr", "/layout-parsing")

API_MODEL_PROFILES: dict[str, dict[str, str]] = {
    "pp-ocrv5": {
        "label": "PP-OCRv5",
        "url": "https://n6z9feddjca4l7b5.aistudio-app.com/ocr",
        "desc": "通用文字识别（/ocr）",
    },
    "pp-structurev3": {
        "label": "PP-StructureV3",
        "url": "https://fbv8f7s7v9u9hbk7.aistudio-app.com/layout-parsing",
        "desc": "版面 + OCR（/layout-parsing）",
    },
    "paddleocr-vl": {
        "label": "PaddleOCR-VL",
        "url": "https://c92fu3s8m4y5i0je.aistudio-app.com/layout-parsing",
        "desc": "VL 大模型版面解析",
    },
    "paddleocr-vl-1.5": {
        "label": "PaddleOCR-VL-1.5",
        "url": "https://15j75bd0964dzbwe.aistudio-app.com/layout-parsing",
        "desc": "VL 1.5 升级版",
    },
}


def get_api_model_profile_options() -> list[tuple[str, str]]:
    return [(k, v["label"]) for k, v in API_MODEL_PROFILES.items()]


def get_api_model_profile_url(profile: str | None) -> str:
    if isinstance(profile, str) and profile in API_MODEL_PROFILES:
        return API_MODEL_PROFILES[profile]["url"]
    return API_MODEL_PROFILES["pp-structurev3"]["url"]


def match_api_model_profile_from_url(api_url: str | None) -> str | None:
    normalized = (api_url or "").strip().rstrip("/")
    for key, spec in API_MODEL_PROFILES.items():
        if spec["url"].rstrip("/") == normalized:
            return key
    return None


def resolve_api_endpoint(api_url: str | None, default_suffix: str = "/layout-parsing") -> str:
    url = (api_url or "").strip().rstrip("/")
    if not url:
        return ""
    if any(url.endswith(suffix) for suffix in KNOWN_API_ENDPOINT_SUFFIXES):
        return url
    return f"{url}{default_suffix}"


def build_api_payload(file_b64: str, file_type: int) -> dict[str, object]:
    return {"file": file_b64, "fileType": file_type}


def detect_api_result_kind(data: dict) -> str:
    if not isinstance(data, dict):
        return "unknown"
    result = data.get("result", {})
    if not isinstance(result, dict):
        return "unknown"
    if isinstance(result.get("ocrResults"), list):
        return "ocr"
    if isinstance(result.get("layoutParsingResults"), list):
        return "layout"
    return "unknown"


# ------------------------------------------------------------------ stylesheet

_STYLE = """
QDialog {
    background: #f5f7fb;
    color: #222;
    font-family: "Microsoft YaHei UI", "PingFang SC", "Source Han Sans CN", sans-serif;
}
QLabel { color: #222; font-size: 13px; }
QLabel#sectionTitle {
    color: #1a73e8;
    font-size: 14px;
    font-weight: bold;
    padding: 0 0 4px 0;
}
QLabel#sectionDesc { color: #888; font-size: 12px; }
QLabel#fieldLabel { color: #555; font-size: 13px; }
QLabel#noteLabel { color: #888; font-size: 12px; }

QFrame#card {
    background: #ffffff;
    border: 1px solid #e3e8ef;
    border-radius: 8px;
}

QLineEdit, QComboBox, QSpinBox {
    background: #ffffff;
    border: 1px solid #d6dde6;
    border-radius: 6px;
    padding: 6px 8px;
    min-height: 22px;
    font-size: 13px;
    color: #222;
    selection-background-color: #cfe2ff;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
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

QRadioButton { spacing: 6px; padding: 4px; font-size: 13px; color: #333; }
QRadioButton::indicator { width: 16px; height: 16px; }

QPushButton {
    background: #ffffff;
    border: 1px solid #d6dde6;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 13px;
    color: #333;
}
QPushButton:hover { border-color: #1a73e8; color: #1a73e8; }
QPushButton:pressed { background: #f0f6ff; }
QPushButton:disabled { color: #aaa; border-color: #e0e4eb; }

QPushButton#primaryBtn {
    background: #1a73e8; color: white; border: 1px solid #1a73e8;
    font-weight: 500;
}
QPushButton#primaryBtn:hover { background: #1666cf; border-color: #1666cf; }
QPushButton#primaryBtn:pressed { background: #1357a8; }

QPushButton#testBtn {
    background: #f0f6ff; color: #1a73e8;
    border: 1px solid #b8d4ff;
}
QPushButton#testBtn:hover { background: #e0eeff; }
"""


# ------------------------------------------------------------------ helpers

def _section_card(title: str, desc: str = "") -> tuple[QFrame, QVBoxLayout]:
    """构造一张白底卡片，返回 (card_widget, content_layout)。"""
    card = QFrame()
    card.setObjectName("card")
    outer = QVBoxLayout(card)
    outer.setContentsMargins(16, 14, 16, 14)
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
    """构造一行 label + widget 表单，使用固定标签宽度对齐。"""
    h = QHBoxLayout()
    h.setSpacing(10)
    lbl = QLabel(label_text)
    lbl.setObjectName("fieldLabel")
    lbl.setFixedWidth(96)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    h.addWidget(lbl)
    h.addWidget(widget, 1)
    return h


# ------------------------------------------------------------------ dialog

class ApiSettingsDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OCR 引擎设置")
        self.setMinimumWidth(560)
        self.setStyleSheet(_STYLE)
        self._build_ui()
        self._load_config()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # === 卡片 1：引擎模式 ===
        card1, c1 = _section_card("引擎模式", "选择本地推理或调用云端 API。")
        mode_row = QHBoxLayout()
        mode_row.setSpacing(20)
        self._radio_local = QRadioButton("本地模型（PaddleOCR）")
        self._radio_api = QRadioButton("云端 API（AiStudio / 自部署）")
        mode_row.addWidget(self._radio_local)
        mode_row.addWidget(self._radio_api)
        mode_row.addStretch()
        c1.addLayout(mode_row)
        root.addWidget(card1)

        # === 卡片 2：API 参数 ===
        self._api_card, c2 = _section_card(
            "API 参数",
            "选择官方模型预设可自动填入端点；也可手动填写自部署 Serving 地址。",
        )

        # 模型预设
        self._api_model_combo = QComboBox()
        for key, label in get_api_model_profile_options():
            spec = API_MODEL_PROFILES[key]
            self._api_model_combo.addItem(f"{label}  —  {spec['desc']}", key)
        self._api_model_combo.setCurrentIndex(-1)
        c2.addLayout(_form_row("官方模型", self._api_model_combo))

        # API URL
        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("https://xxxxx.aistudio-app.com/layout-parsing")
        c2.addLayout(_form_row("API 地址", self._url_edit))

        # Token (with show/hide)
        token_row = QWidget()
        th = QHBoxLayout(token_row)
        th.setContentsMargins(0, 0, 0, 0)
        th.setSpacing(6)
        self._token_edit = QLineEdit()
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_edit.setPlaceholderText("AiStudio 个人中心 → 访问令牌")
        th.addWidget(self._token_edit, 1)
        self._btn_show_token = QPushButton("👁")
        self._btn_show_token.setCheckable(True)
        self._btn_show_token.setFixedWidth(36)
        self._btn_show_token.setToolTip("显示 / 隐藏 Token")
        self._btn_show_token.toggled.connect(self._toggle_token_visibility)
        th.addWidget(self._btn_show_token)
        c2.addLayout(_form_row("访问 Token", token_row))

        # 超时
        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(5, 120)
        self._timeout_spin.setSuffix(" 秒")
        self._timeout_spin.setFixedWidth(120)
        timeout_row = QHBoxLayout()
        timeout_row.setSpacing(0)
        timeout_row.addWidget(self._timeout_spin)
        timeout_row.addStretch()
        timeout_w = QWidget()
        timeout_w.setLayout(timeout_row)
        c2.addLayout(_form_row("请求超时", timeout_w))

        # 测试连接
        test_row = QHBoxLayout()
        test_row.addStretch()
        self._btn_test = QPushButton("🧪 测试连接")
        self._btn_test.setObjectName("testBtn")
        self._btn_test.clicked.connect(self._test_connection)
        test_row.addWidget(self._btn_test)
        test_w = QWidget()
        test_w.setLayout(test_row)
        c2.addLayout(_form_row("", test_w))

        root.addWidget(self._api_card)

        # === 底部按钮 ===
        root.addStretch()
        bottom = QHBoxLayout()
        bottom.addStretch()
        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_ok = QPushButton("保存")
        self._btn_ok.setObjectName("primaryBtn")
        self._btn_ok.clicked.connect(self._save_and_accept)
        bottom.addWidget(self._btn_cancel)
        bottom.addWidget(self._btn_ok)
        root.addLayout(bottom)

        # 信号
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
        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        self._api_card.setEnabled(self._radio_api.isChecked())

    def _on_api_model_changed(self) -> None:
        profile_key = self._api_model_combo.currentData()
        if not profile_key:
            return
        self._url_edit.setText(get_api_model_profile_url(profile_key))

    def _sync_model_from_url(self) -> None:
        profile_key = match_api_model_profile_from_url(self._url_edit.text().strip())
        index = self._api_model_combo.findData(profile_key) if profile_key else -1
        self._api_model_combo.blockSignals(True)
        self._api_model_combo.setCurrentIndex(index if index >= 0 else -1)
        self._api_model_combo.blockSignals(False)

    def _toggle_token_visibility(self, checked: bool) -> None:
        if checked:
            self._token_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)

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
            self._url_edit.text().strip(), default_suffix="/layout-parsing",
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
            self._btn_test.setText("🧪 测试连接")
