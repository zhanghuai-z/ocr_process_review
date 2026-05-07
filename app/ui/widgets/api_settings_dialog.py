"""OCR engine settings dialog.

Supports switching between local PaddleOCR and AiStudio cloud API.
Confirmed AiStudio endpoint format:
  POST {api_url}/layout-parsing
  Authorization: token <api_token>
  Body: {"file": "<base64_JPEG>", "fileType": 1}
  Response: {"errorCode": 0, "result": {"layoutParsingResults": [...]}}
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget,
)

from app.core.ocr_config import get_config, update_config

KNOWN_API_ENDPOINT_SUFFIXES = ("/ocr", "/layout-parsing")
API_MODEL_PROFILES: dict[str, dict[str, str]] = {
    "pp-ocrv5": {
        "label": "PP-OCRv5",
        "url": "https://n6z9feddjca4l7b5.aistudio-app.com/ocr",
    },
    "pp-structurev3": {
        "label": "PP-StructureV3",
        "url": "https://fbv8f7s7v9u9hbk7.aistudio-app.com/layout-parsing",
    },
    "paddleocr-vl": {
        "label": "PaddleOCR-VL",
        "url": "https://c92fu3s8m4y5i0je.aistudio-app.com/layout-parsing",
    },
    "paddleocr-vl-1.5": {
        "label": "PaddleOCR-VL-1.5",
        "url": "https://15j75bd0964dzbwe.aistudio-app.com/layout-parsing",
    },
}


def get_api_model_profile_options() -> list[tuple[str, str]]:
    return [(key, value["label"]) for key, value in API_MODEL_PROFILES.items()]


def get_api_model_profile_url(profile: str | None) -> str:
    if isinstance(profile, str) and profile in API_MODEL_PROFILES:
        return API_MODEL_PROFILES[profile]["url"]
    return API_MODEL_PROFILES["pp-ocrv5"]["url"]


def match_api_model_profile_from_url(api_url: str | None) -> str | None:
    normalized_url = (api_url or "").strip().rstrip("/")
    for key, spec in API_MODEL_PROFILES.items():
        if spec["url"].rstrip("/") == normalized_url:
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
    return {
        "file": file_b64,
        "fileType": file_type,
    }


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


class ApiSettingsDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OCR 引擎设置")
        self.setMinimumWidth(520)
        self._build_ui()
        self._load_config()

    # ── UI ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)

        mode_group = QGroupBox("OCR 引擎模式")
        mode_h = QHBoxLayout(mode_group)
        self._radio_local = QRadioButton("本地模型（PaddleOCR）")
        self._radio_api   = QRadioButton("云端 API（AiStudio / 自部署 Serving）")
        mode_h.addWidget(self._radio_local)
        mode_h.addWidget(self._radio_api)
        root.addWidget(mode_group)

        self._api_group = QGroupBox("API 参数")
        form = QFormLayout(self._api_group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        url_note = QLabel(
            "AiStudio 端点示例：<b>https://xxxxx.aistudio-app.com/ocr</b> 或 "
            "<b>https://xxxxx.aistudio-app.com/layout-parsing</b><br>"
            "自部署 Serving 同样建议填写完整端点。<br>"
            "<small>推荐直接填完整 URL；若只填应用根地址，将默认补成 /layout-parsing。</small>"
        )
        url_note.setWordWrap(True)
        url_note.setTextFormat(Qt.TextFormat.RichText)
        form.addRow(url_note)

        self._api_model_combo = QComboBox()
        for key, label in get_api_model_profile_options():
            self._api_model_combo.addItem(label, key)
        self._api_model_combo.setCurrentIndex(-1)
        form.addRow("官方模型：", self._api_model_combo)

        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("https://xxxxx.aistudio-app.com/layout-parsing")
        form.addRow("API 地址：", self._url_edit)

        token_row = QWidget()
        token_h = QHBoxLayout(token_row)
        token_h.setContentsMargins(0, 0, 0, 0)
        self._token_edit = QLineEdit()
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_edit.setPlaceholderText("AiStudio 访问 Token（个人中心 → 访问令牌）")
        token_h.addWidget(self._token_edit)
        self._btn_show_token = QPushButton("显示")
        self._btn_show_token.setCheckable(True)
        self._btn_show_token.setFixedWidth(50)
        self._btn_show_token.toggled.connect(self._toggle_token_visibility)
        token_h.addWidget(self._btn_show_token)
        form.addRow("访问 Token：", token_row)

        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(5, 120)
        self._timeout_spin.setSuffix(" 秒")
        form.addRow("请求超时：", self._timeout_spin)

        self._btn_test = QPushButton("测试连接")
        self._btn_test.clicked.connect(self._test_connection)
        form.addRow("", self._btn_test)

        root.addWidget(self._api_group)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._save_and_accept)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

        self._radio_local.toggled.connect(self._on_mode_changed)
        self._radio_api.toggled.connect(self._on_mode_changed)
        self._api_model_combo.currentIndexChanged.connect(self._on_api_model_changed)
        self._url_edit.editingFinished.connect(self._sync_model_from_url)

    # ── logic ──────────────────────────────────────────────────

    def _load_config(self) -> None:
        cfg = get_config()
        if cfg["mode"] == "api":
            self._radio_api.setChecked(True)
        else:
            self._radio_local.setChecked(True)
        profile_key = cfg.get("api_model_profile", "")
        if not profile_key:
            profile_key = match_api_model_profile_from_url(cfg.get("api_url", ""))
        index = self._api_model_combo.findData(profile_key)
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
        self._api_group.setEnabled(self._radio_api.isChecked())

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
            self._btn_show_token.setText("隐藏")
        else:
            self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self._btn_show_token.setText("显示")

    def _save_and_accept(self) -> None:
        mode = "api" if self._radio_api.isChecked() else "local"
        api_url = self._url_edit.text().strip().rstrip("/")
        api_model_profile = self._api_model_combo.currentData() or match_api_model_profile_from_url(api_url) or ""
        update_config(
            mode=mode,
            api_model_profile=api_model_profile,
            api_url=api_url,
            api_token=self._token_edit.text().strip(),
            api_timeout=self._timeout_spin.value(),
            api_layout_model_name="",
        )
        self.accept()

    def _test_connection(self) -> None:
        """POST a real-sized JPEG to /layout-parsing to verify connectivity."""
        import base64
        import requests

        url = resolve_api_endpoint(self._url_edit.text().strip(), default_suffix="/layout-parsing")
        token = self._token_edit.text().strip()
        timeout = self._timeout_spin.value()

        # Build a 200x400 gray image with text — small enough to be fast,
        # large enough that the server won't reject it as invalid.
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

        try:
            payload = build_api_payload(file_b64, 1)
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            code = resp.status_code
            try:
                body = resp.json()
                err_code = body.get("errorCode", -1)
                err_msg  = body.get("errorMsg", "")
            except Exception:
                err_code = -1
                err_msg  = resp.text[:300]

            if code == 200 and err_code == 0:
                kind = detect_api_result_kind(body) if isinstance(body, dict) else "unknown"
                kind_label = {
                    "ocr": "PP-OCRv5 /ocr",
                    "layout": "PP-StructureV3 / VL /layout-parsing",
                    "unknown": "未知结构",
                }.get(kind, kind)
                QMessageBox.information(self, "测试成功", f"连接正常，API 响应成功！\n识别到响应类型：{kind_label}")
            elif code in (401, 403) or err_code in (401, 403):
                QMessageBox.warning(
                    self, "鉴权失败",
                    f"HTTP {code}：Token 无效或已过期，请检查访问 Token。"
                )
            else:
                QMessageBox.warning(
                    self, "连接异常",
                    f"HTTP {code}，errorCode={err_code}\n{err_msg}"
                )
        except requests.exceptions.ConnectionError:
            QMessageBox.critical(self, "连接失败", f"无法连接到：\n{url}\n请检查 URL 或网络。")
        except requests.exceptions.Timeout:
            QMessageBox.critical(self, "超时", f"请求超时（>{timeout}s）。")
        except Exception as exc:
            QMessageBox.critical(self, "错误", str(exc))
