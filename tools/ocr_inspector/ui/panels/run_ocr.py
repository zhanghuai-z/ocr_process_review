"""Run OCR panel — lets the user call PaddleOCR directly on the current image.

Supports:
  * PaddleOCR (text OCR, PaddleOCR 3.x)
  * PPStructureV3 (layout + OCR)

Parameters are exposed as editable controls.  Results are parsed through the
existing PaddleAdapter and loaded into AppState.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from tools.ocr_inspector.adapters.paddle import PaddleAdapter
from tools.ocr_inspector.state import AppState


# ---------------------------------------------------------------------------
# Worker signal carrier (must be a QObject to use signals from a bg thread)
# ---------------------------------------------------------------------------

class _Signals(QObject):
    finished = Signal(object, str)   # (raw_result_dict, image_path)
    error    = Signal(str)


# ---------------------------------------------------------------------------
# Result flatteners
# ---------------------------------------------------------------------------

def _append_list_field(target: dict[str, Any], key: str, value: Any) -> None:
    if isinstance(value, list):
        target.setdefault(key, [])
        target[key].extend(value)


_RELEVANT_RESPONSE_FIELDS = (
    "layoutParsingResults",
    "ocrResults",
    "prunedResult",
    "overall_ocr_res",
    "rec_texts",
    "rec_boxes",
    "rec_polys",
    "rec_polygons",
    "dt_polys",
    "text_word",
    "textWord",
    "text_word_region",
    "textWordRegion",
    "text_word_boxes",
    "textWordBoxes",
    "parsing_res_list",
    "layout_det_res",
    "markdown",
)


def _field_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return 1
    if value in (None, ""):
        return 0
    return 1


def _summarize_relevant_response_fields(value: Any) -> dict[str, int]:
    summary: dict[str, int] = {}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key in _RELEVANT_RESPONSE_FIELDS:
                    summary[key] = summary.get(key, 0) + _field_count(child)
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return {key: summary[key] for key in sorted(summary)}


def _word_box_field_count(summary: dict[str, int]) -> int:
    return sum(summary.get(key, 0) for key in ("text_word_region", "textWordRegion", "text_word_boxes", "textWordBoxes"))


def _summarize_runtime_params(params: dict[str, Any]) -> dict[str, Any]:
    pred = params.get("ocr_pred", {}) if isinstance(params, dict) else {}
    init = params.get("ocr_init", {}) if isinstance(params, dict) else {}
    structure = params.get("structure", {}) if isinstance(params, dict) else {}
    summary: dict[str, Any] = {
        "returnWordBox": pred.get("return_word_box"),
        "useDocOrientationClassify": init.get("use_doc_orientation_classify"),
        "useDocUnwarping": init.get("use_doc_unwarping"),
        "useTextlineOrientation": init.get("use_textline_orientation"),
        "textDetThresh": pred.get("text_det_thresh", structure.get("text_det_thresh")),
        "textDetBoxThresh": pred.get("text_det_box_thresh", structure.get("text_det_box_thresh")),
        "textDetUnclipRatio": pred.get("text_det_unclip_ratio", structure.get("text_det_unclip_ratio")),
        "textDetLimitSideLen": pred.get("text_det_limit_side_len", structure.get("text_det_limit_side_len")),
        "textDetLimitType": pred.get("text_det_limit_type", structure.get("text_det_limit_type")),
        "textRecScoreThresh": pred.get("text_rec_score_thresh", structure.get("text_rec_score_thresh")),
    }
    return {key: value for key, value in summary.items() if value is not None}


def _attach_inspector_runtime_meta(
    raw: dict[str, Any],
    *,
    source: str,
    pipeline: str,
    image_path: str,
    params: dict[str, Any] | None = None,
    request_summary: dict[str, Any] | None = None,
    response_raw: Any | None = None,
    api_url: str = "",
    api_model_profile: str = "",
) -> dict[str, Any]:
    meta = raw.setdefault("_inspector_meta", {})
    meta["source"] = source
    meta["pipeline"] = pipeline
    meta["image_path"] = image_path
    if api_url:
        meta["api_url"] = api_url
    if api_model_profile:
        meta["api_model_profile"] = api_model_profile
    if request_summary is None and params is not None:
        request_summary = _summarize_runtime_params(params)
    if request_summary is not None:
        meta["request_summary"] = dict(request_summary)
    response_summary = _summarize_relevant_response_fields(response_raw if response_raw is not None else raw)
    flattened_summary = _summarize_relevant_response_fields(raw)
    meta["response_field_summary"] = response_summary
    meta["flattened_field_summary"] = flattened_summary
    meta["response_word_box_field_count"] = _word_box_field_count(response_summary)
    meta["flattened_word_box_field_count"] = _word_box_field_count(flattened_summary)
    return raw


def _merge_pruned_like_result(merged: dict[str, Any], source: dict[str, Any]) -> None:
    parsing_res = source.get("parsing_res_list")
    if isinstance(parsing_res, list):
        merged.setdefault("parsing_res_list", [])
        merged["parsing_res_list"].extend(v for v in parsing_res if isinstance(v, dict))

    layout_det = source.get("layout_det_res")
    if isinstance(layout_det, dict):
        merged.setdefault("layout_det_res", {"boxes": []})
        boxes = layout_det.get("boxes")
        if isinstance(boxes, list):
            merged["layout_det_res"].setdefault("boxes", [])
            merged["layout_det_res"]["boxes"].extend(v for v in boxes if isinstance(v, dict))
        for key, value in layout_det.items():
            if key != "boxes":
                merged["layout_det_res"][key] = value

    overall = source.get("overall_ocr_res")
    if not isinstance(overall, dict) and any(key in source for key in ("rec_texts", "rec_boxes", "rec_polys", "dt_polys")):
        overall = {
            key: source[key]
            for key in ("rec_texts", "rec_scores", "rec_boxes", "rec_polys", "rec_polygons", "dt_polys")
            if key in source
        }
    if isinstance(overall, dict):
        merged.setdefault("overall_ocr_res", {})
        for key, value in overall.items():
            if isinstance(value, list):
                merged["overall_ocr_res"].setdefault(key, [])
                merged["overall_ocr_res"][key].extend(value)
            else:
                merged["overall_ocr_res"][key] = value

    for canonical, aliases in (
        ("text_word", ("text_word", "textWord")),
        ("text_word_region", ("text_word_region", "textWordRegion", "text_word_boxes", "textWordBoxes")),
    ):
        empty_value: list[Any] | None = None
        for alias in aliases:
            value = source.get(alias)
            if isinstance(value, list) and value:
                _append_list_field(merged, canonical, value)
                break
            if isinstance(value, list) and empty_value is None:
                empty_value = value
        else:
            if empty_value is not None:
                _append_list_field(merged, canonical, empty_value)


def _drop_empty_result_fields(raw: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(raw)
    if not cleaned.get("parsing_res_list"):
        cleaned.pop("parsing_res_list", None)
    layout_det = cleaned.get("layout_det_res")
    if isinstance(layout_det, dict) and not layout_det.get("boxes"):
        cleaned.pop("layout_det_res", None)
    if not cleaned.get("overall_ocr_res"):
        cleaned.pop("overall_ocr_res", None)
    if not cleaned.get("text_word"):
        cleaned.pop("text_word", None)
    if not cleaned.get("text_word_region"):
        cleaned.pop("text_word_region", None)
    return cleaned


def _flatten_paddle_result(result_list: list) -> dict:
    """Convert PaddleOCR 3.x predict() list to the dict shape PaddleAdapter expects."""
    merged: dict[str, Any] = {}
    for item in result_list:
        if hasattr(item, "json"):
            d = item.json()
            if isinstance(d, dict):
                _merge_pruned_like_result(merged, d)
    if not merged and result_list and hasattr(result_list[0], "json"):
        return result_list[0].json()
    return _drop_empty_result_fields(merged)


def _flatten_structure_result(result_list: list) -> dict:
    """Convert PPStructureV3 predict() output to raw dict for PaddleAdapter."""
    out: dict[str, Any] = {}
    for item in result_list:
        if hasattr(item, "json"):
            d = item.json()
            if isinstance(d, dict):
                if any(key in d for key in (
                    "overall_ocr_res",
                    "parsing_res_list",
                    "layout_det_res",
                    "text_word",
                    "textWord",
                    "text_word_region",
                    "textWordRegion",
                    "text_word_boxes",
                    "textWordBoxes",
                )):
                    _merge_pruned_like_result(out, d)
                else:
                    out.setdefault("parsing_res_list", [])
                    out["parsing_res_list"].append(d)
    return _drop_empty_result_fields(out)


def _run_structure_ocr(image_path: str, params: dict[str, Any]) -> dict:
    """Run structure-aware OCR, falling back to a compatible local path when needed.

    PaddleOCR internally uses cv2.imread, which fails on Windows for non-ASCII
    paths. When the path contains non-ASCII chars, pre-load the image as a numpy
    array and pass that to predict() instead.
    """
    # PaddleOCR's predict() accepts numpy arrays (BGR), so we can bypass
    # cv2.imread's non-ASCII path limitation by reading bytes ourselves.
    try:
        input_img = _imread_any_path(image_path)
    except Exception:
        # Fallback to string path if pre-loading fails (e.g. unsupported format)
        input_img = image_path  # type: ignore[assignment]

    try:
        from paddleocr import PPStructureV3

        engine = PPStructureV3(**params["structure"])
        result = list(engine.predict(input_img))
        raw = _flatten_structure_result(result)
        return _attach_inspector_runtime_meta(
            raw,
            source="local",
            pipeline="ppstructure",
            image_path=image_path,
            params=params,
            response_raw=[item.json() for item in result if hasattr(item, "json")],
        )
    except Exception as exc:
        if "pipeline (PP-StructureV3) does not exist" not in str(exc):
            raise

        from paddleocr import PaddleOCR

        init_p = {k: v for k, v in params["ocr_init"].items() if v is not None}
        init_p.update({
            "layout": True,
            "table": False,
            "ocr": True,
            "show_log": False,
        })
        pred_p = {k: v for k, v in params["ocr_pred"].items() if v is not None}
        result = list(PaddleOCR(**init_p).predict(input_img, **pred_p))
        raw = _flatten_paddle_result(result)
        if isinstance(raw, dict):
            meta = raw.setdefault("_inspector_meta", {})
            meta["fallback"] = "paddleocr_layout_compat"
            meta["reason"] = str(exc)
            _attach_inspector_runtime_meta(
                raw,
                source="local",
                pipeline="paddleocr_layout_compat",
                image_path=image_path,
                params=params,
                response_raw=[item.json() for item in result if hasattr(item, "json")],
            )
        return raw


def _flatten_api_result(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten AiStudio /layout-parsing response into PaddleAdapter-friendly shape."""
    result = data.get("result", data) if isinstance(data, dict) else data
    if not isinstance(result, dict):
        raise RuntimeError("API response is not a dict")

    if any(key in result for key in ("overall_ocr_res", "parsing_res_list", "layout_det_res")):
        return dict(result)

    items: list[dict[str, Any]] = []
    for key in ("layoutParsingResults", "ocrResults"):
        value = result.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))

    merged: dict[str, Any] = {}

    for item in items:
        source = item.get("prunedResult")
        if not isinstance(source, dict):
            source = item
        if not isinstance(source, dict):
            continue

        _merge_pruned_like_result(merged, source)

    return _drop_empty_result_fields(merged)


def _build_api_request_body(file_b64: str, params: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    from app.engines.real_ocr_adapter import ApiOcrEngine

    profile = str(cfg.get("api_model_profile", "") or "").strip()
    endpoint_url = str(cfg.get("_resolved_api_url", "") or "").strip()
    body = ApiOcrEngine()._build_request_body(
        file_b64,
        1,
        profile=profile,
        endpoint_url=endpoint_url,
    )
    model_name = str(cfg.get("api_layout_model_name", "") or "").strip()
    if model_name:
        body["model_name"] = model_name

    if "returnWordBox" in body:
        body.update({
            "returnWordBox": params["ocr_pred"]["return_word_box"],
            "useDocOrientationClassify": params["ocr_init"]["use_doc_orientation_classify"],
            "useDocUnwarping": params["ocr_init"]["use_doc_unwarping"],
            "useTextlineOrientation": params["ocr_init"]["use_textline_orientation"],
            "textDetThresh": params["ocr_pred"]["text_det_thresh"],
            "textDetBoxThresh": params["ocr_pred"]["text_det_box_thresh"],
            "textDetUnclipRatio": params["ocr_pred"]["text_det_unclip_ratio"],
            "textDetLimitSideLen": params["ocr_pred"]["text_det_limit_side_len"],
            "textDetLimitType": params["ocr_pred"]["text_det_limit_type"],
            "textRecScoreThresh": params["ocr_pred"]["text_rec_score_thresh"],
        })
    return body


def _imread_any_path(image_path: str):
    """Read image via bytes->imdecode to support non-ASCII (e.g. Chinese) paths.

    cv2.imread silently fails on Windows when the path contains non-ASCII chars.
    Reading raw bytes first (OS-agnostic via pathlib) and decoding in memory
    works for all paths and all formats that cv2 supports (JPEG, PNG, TIFF...).
    """
    import cv2
    import numpy as np

    try:
        raw = Path(image_path).read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Cannot read image file: {image_path}") from exc
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot decode image (unsupported format or corrupt): {image_path}")
    return img


def _run_api_ocr(image_path: str, params: dict[str, Any]) -> dict[str, Any]:
    import base64

    import cv2
    import requests

    from app.core.api_profiles import get_api_model_profile, resolve_api_endpoint
    from app.core.ocr_config import get_config

    cfg = get_config()
    profile = str(cfg.get("api_model_profile", "") or "").strip()
    url = resolve_api_endpoint(
        cfg.get("api_url", ""),
        default_suffix="/layout-parsing",
        profile=profile,
    )
    if not url:
        raise RuntimeError("API URL is not configured. Open OCR 引擎设置 first.")

    timeout = int(cfg.get("api_timeout", 30) or 30)
    token = str(cfg.get("api_token", "") or "").strip()
    img = _imread_any_path(image_path)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {image_path}")
    file_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    cfg = dict(cfg)
    cfg["_resolved_api_url"] = url
    body = _build_api_request_body(file_b64, params, cfg)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"token {token}"

    resp = requests.post(url, json=body, headers=headers, timeout=timeout)
    resp.raise_for_status()

    response_json = resp.json()
    raw = _flatten_api_result(response_json)
    if profile:
        meta = raw.setdefault("_inspector_meta", {})
        meta["api_model_capability"] = dict(get_api_model_profile(profile))
    request_summary = {
        "returnWordBox": body.get("returnWordBox"),
        "useDocOrientationClassify": body.get("useDocOrientationClassify"),
        "useDocUnwarping": body.get("useDocUnwarping"),
        "useTextlineOrientation": body.get("useTextlineOrientation"),
        "textDetThresh": body.get("textDetThresh"),
        "textDetBoxThresh": body.get("textDetBoxThresh"),
        "textDetUnclipRatio": body.get("textDetUnclipRatio"),
        "textDetLimitSideLen": body.get("textDetLimitSideLen"),
        "textDetLimitType": body.get("textDetLimitType"),
        "textRecScoreThresh": body.get("textRecScoreThresh"),
        "model_name": body.get("model_name", ""),
    }
    _attach_inspector_runtime_meta(
        raw,
        source="api",
        pipeline="aistudio",
        image_path=image_path,
        request_summary=request_summary,
        response_raw=response_json,
        api_url=url,
        api_model_profile=profile,
    )
    meta = raw.setdefault("_inspector_meta", {})
    meta["api_request_summary"] = request_summary
    raw["api_model_profile"] = profile
    return raw


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class RunOcrPanel(QWidget):
    """Controls panel for running PaddleOCR live on an image."""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._signals = _Signals()
        self._signals.finished.connect(self._on_ocr_finished)
        self._signals.error.connect(self._on_ocr_error)
        self._last_raw: Optional[dict] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── image path ──────────────────────────────────────────────────────
        img_group = QGroupBox("Image")
        img_layout = QHBoxLayout(img_group)
        self._img_edit = QLineEdit()
        self._img_edit.setPlaceholderText("(uses active page image)")
        self._img_btn = QPushButton("…")
        self._img_btn.setFixedWidth(28)
        self._img_btn.clicked.connect(self._browse_image)
        img_layout.addWidget(self._img_edit)
        img_layout.addWidget(self._img_btn)
        layout.addWidget(img_group)

        # ── model selection ──────────────────────────────────────────────────
        model_group = QGroupBox("Model")
        model_form = QFormLayout(model_group)
        self._model_combo = QComboBox()
        self._model_combo.addItem("PaddleOCR (文字检测+识别)", "paddleocr")
        self._model_combo.addItem("PPStructureV3 (版面分析)", "ppstructure")
        self._model_combo.addItem("AiStudio API (/layout-parsing，共享主程序设置)", "api")
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_form.addRow("Pipeline:", self._model_combo)

        self._lang_combo = QComboBox()
        for lang in ["ch", "en", "japan", "korean", "fr", "de", "ar"]:
            self._lang_combo.addItem(lang)
        model_form.addRow("Language:", self._lang_combo)

        self._ocr_version_combo = QComboBox()
        self._ocr_version_combo.addItem("PP-OCRv5 (default)", "")
        self._ocr_version_combo.addItem("PP-OCRv4", "PP-OCRv4")
        self._ocr_version_combo.addItem("PP-OCRv3", "PP-OCRv3")
        model_form.addRow("OCR version:", self._ocr_version_combo)

        self._api_cfg_btn = QPushButton("OCR 引擎设置…")
        self._api_cfg_btn.clicked.connect(self._show_api_settings)
        self._api_cfg_summary = QLabel()
        self._api_cfg_summary.setWordWrap(True)
        self._api_cfg_summary.setStyleSheet("color:#6b7280;")
        api_cfg_box = QWidget()
        api_cfg_layout = QVBoxLayout(api_cfg_box)
        api_cfg_layout.setContentsMargins(0, 0, 0, 0)
        api_cfg_layout.setSpacing(4)
        api_cfg_layout.addWidget(self._api_cfg_btn)
        api_cfg_layout.addWidget(self._api_cfg_summary)
        model_form.addRow("Shared API:", api_cfg_box)
        layout.addWidget(model_group)

        # ── preprocessing ────────────────────────────────────────────────────
        pre_group = QGroupBox("预处理")
        pre_form = QFormLayout(pre_group)
        self._use_orientation = QCheckBox()
        self._use_orientation.setChecked(False)
        self._use_orientation.setToolTip("自动识别并旋转文档方向 (0°/90°/180°/270°)\n对竖放扫描件有用；关闭节省约 50ms")
        pre_form.addRow("文档方向校正:", self._use_orientation)

        self._use_unwarping = QCheckBox()
        self._use_unwarping.setChecked(False)
        self._use_unwarping.setToolTip("对扫描件做 TPS 去扭曲\n⚠ 开启后坐标仍在处理后图像像素空间")
        pre_form.addRow("图像去扭曲:", self._use_unwarping)

        self._use_textline_orient = QCheckBox()
        self._use_textline_orient.setChecked(False)
        self._use_textline_orient.setToolTip("识别并处理竖排文本行 (日文/中文竖排)\n关闭节省约 5ms/行")
        pre_form.addRow("文本行方向:", self._use_textline_orient)
        layout.addWidget(pre_group)

        # ── det params ───────────────────────────────────────────────────────
        det_group = QGroupBox("检测参数 (det)")
        det_form = QFormLayout(det_group)

        self._det_thresh = QDoubleSpinBox()
        self._det_thresh.setRange(0.01, 1.0)
        self._det_thresh.setSingleStep(0.05)
        self._det_thresh.setDecimals(2)
        self._det_thresh.setValue(0.3)
        self._det_thresh.setToolTip("像素二值化阈值 (0~1)\n值越小：更多像素判为文字边缘 → 召回↑ 噪声↑\n值越大：只保留高置信像素 → 精确度↑ 漏字↑\n默认 0.3")
        det_form.addRow("det_thresh:", self._det_thresh)

        self._det_box_thresh = QDoubleSpinBox()
        self._det_box_thresh.setRange(0.01, 1.0)
        self._det_box_thresh.setSingleStep(0.05)
        self._det_box_thresh.setDecimals(2)
        self._det_box_thresh.setValue(0.6)
        self._det_box_thresh.setToolTip("检测框得分过滤阈值 (0~1)\n值越大：去掉低置信检测框 → 精确度↑ 召回↓\n默认 0.6")
        det_form.addRow("det_box_thresh:", self._det_box_thresh)

        self._det_unclip_ratio = QDoubleSpinBox()
        self._det_unclip_ratio.setRange(0.5, 5.0)
        self._det_unclip_ratio.setSingleStep(0.1)
        self._det_unclip_ratio.setDecimals(2)
        self._det_unclip_ratio.setValue(2.0)
        self._det_unclip_ratio.setToolTip("检测框扩张比例 (Vatti clipping)\n值越大框越宽松，有助包住完整字符\n过大会合并相邻行\n默认 2.0")
        det_form.addRow("det_unclip_ratio:", self._det_unclip_ratio)

        self._det_limit_side_len = QSpinBox()
        self._det_limit_side_len.setRange(64, 4096)
        self._det_limit_side_len.setSingleStep(64)
        self._det_limit_side_len.setValue(1536)
        self._det_limit_side_len.setToolTip("检测前图像最长边缩放上限 (px)\n值越小：速度快，细小文字易丢失\n值越大：细节保留好，内存/速度代价高\n默认 1536")
        det_form.addRow("det_limit_side_len:", self._det_limit_side_len)

        self._det_limit_type_combo = QComboBox()
        self._det_limit_type_combo.addItem("max")
        self._det_limit_type_combo.addItem("min")
        det_form.addRow("det_limit_type:", self._det_limit_type_combo)
        layout.addWidget(det_group)

        # ── rec params ───────────────────────────────────────────────────────
        rec_group = QGroupBox("识别参数 (rec)")
        rec_form = QFormLayout(rec_group)

        self._rec_score_thresh = QDoubleSpinBox()
        self._rec_score_thresh.setRange(0.0, 1.0)
        self._rec_score_thresh.setSingleStep(0.05)
        self._rec_score_thresh.setDecimals(2)
        self._rec_score_thresh.setValue(0.0)
        self._rec_score_thresh.setToolTip("识别置信度过滤阈值 (0~1)\n调高可减少乱码行，但可能漏掉难字\n0.0 = 不过滤（默认）")
        rec_form.addRow("rec_score_thresh:", self._rec_score_thresh)

        self._return_word_box = QCheckBox()
        self._return_word_box.setChecked(True)
        self._return_word_box.setToolTip("开启后返回字/词级 bounding box\n→ overall_ocr_res.text_word_region 有值\n→ canvas 中橙色字框才会出现\n默认 True，避免 Inspector 只能看到 line/fallback 级 bbox")
        rec_form.addRow("return_word_box:", self._return_word_box)
        layout.addWidget(rec_group)

        # ── layout params (PPStructureV3 only) ────────────────────────────────
        self._layout_group = QGroupBox("版面参数 (layout)")
        layout_form = QFormLayout(self._layout_group)

        self._layout_thresh = QDoubleSpinBox()
        self._layout_thresh.setRange(0.01, 1.0)
        self._layout_thresh.setSingleStep(0.05)
        self._layout_thresh.setDecimals(2)
        self._layout_thresh.setValue(0.5)
        layout_form.addRow("layout_threshold:", self._layout_thresh)

        self._layout_nms = QCheckBox()
        self._layout_nms.setChecked(True)
        layout_form.addRow("layout_nms:", self._layout_nms)

        self._layout_unclip = QDoubleSpinBox()
        self._layout_unclip.setRange(0.5, 5.0)
        self._layout_unclip.setSingleStep(0.1)
        self._layout_unclip.setDecimals(2)
        self._layout_unclip.setValue(1.0)
        layout_form.addRow("layout_unclip_ratio:", self._layout_unclip)
        layout.addWidget(self._layout_group)
        self._layout_group.setVisible(False)

        # ── run / save buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run OCR")
        self._run_btn.setStyleSheet("font-weight:bold; padding:4px 8px;")
        self._run_btn.clicked.connect(self._run_ocr)
        btn_row.addWidget(self._run_btn)

        self._save_btn = QPushButton("Save JSON…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_result)
        btn_row.addWidget(self._save_btn)
        layout.addLayout(btn_row)

        # ── log ──────────────────────────────────────────────────────────────
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)
        self._log.setStyleSheet("font-size:11px; font-family:monospace;")
        log_layout.addWidget(self._log)
        layout.addWidget(log_group)

        layout.addStretch()
        self.refresh_api_summary()
        self._on_model_changed(self._model_combo.currentIndex())

    # ── slots ──────────────────────────────────────────────────────────────

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)"
        )
        if path:
            self._img_edit.setText(path)

    def _on_model_changed(self, _index: int) -> None:
        is_structure = self._model_combo.currentData() == "ppstructure"
        is_api = self._model_combo.currentData() == "api"
        self._layout_group.setVisible(is_structure)
        self._lang_combo.setEnabled(not is_structure and not is_api)
        self._ocr_version_combo.setEnabled(not is_structure and not is_api)
        self._api_cfg_btn.setEnabled(True)

    def refresh_api_summary(self) -> None:
        try:
            from app.core.ocr_config import get_config
            from app.ui.widgets.api_settings_dialog import resolve_api_endpoint
        except ImportError:
            self._api_cfg_summary.setText("(API config module unavailable)")
            return
        cfg = get_config()
        url = resolve_api_endpoint(cfg.get("api_url", ""), default_suffix="/layout-parsing")
        if not url:
            self._api_cfg_summary.setText("未配置。点击“OCR 引擎设置…”后即可复用主程序 API。")
            return
        profile = str(cfg.get("api_model_profile", "") or "").strip() or "custom"
        timeout = int(cfg.get("api_timeout", 30) or 30)
        self._api_cfg_summary.setText(f"profile={profile} · timeout={timeout}s\n{url}")

    def _show_api_settings(self) -> None:
        try:
            from app.ui.widgets.api_settings_dialog import ApiSettingsDialog
        except ImportError as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "API Settings", f"Not available:\n{exc}")
            return
        dlg = ApiSettingsDialog(self)
        if dlg.exec():
            self.refresh_api_summary()

    def _run_ocr(self) -> None:
        image_path = self._img_edit.text().strip()
        if not image_path:
            page = self._state.active_page
            if page and page.image_path:
                image_path = page.image_path
        if not image_path or not Path(image_path).is_file():
            QMessageBox.warning(self, "No image",
                                "Please specify a valid image file first.\n"
                                "(Set an image via the toolbar 'Set Image…' button, "
                                "or type a path above.)")
            return

        self._run_btn.setEnabled(False)
        self._log.clear()
        self._log.append(f"Running OCR on: {image_path}")

        params = self._collect_params()
        model_key = self._model_combo.currentData()
        signals = self._signals

        def worker() -> None:
            try:
                if model_key == "ppstructure":
                    raw = _run_structure_ocr(image_path, params)
                elif model_key == "api":
                    raw = _run_api_ocr(image_path, params)
                else:
                    from paddleocr import PaddleOCR
                    init_p = {k: v for k, v in params["ocr_init"].items() if v is not None}
                    pred_p = {k: v for k, v in params["ocr_pred"].items() if v is not None}
                    engine = PaddleOCR(**init_p)
                    # Pre-load image as numpy to bypass cv2.imread non-ASCII path issue
                    try:
                        input_img = _imread_any_path(image_path)
                    except Exception:
                        input_img = image_path  # type: ignore[assignment]
                    result = list(engine.predict(input_img, **pred_p))
                    raw = _flatten_paddle_result(result)
                    if isinstance(raw, dict):
                        _attach_inspector_runtime_meta(
                            raw,
                            source="local",
                            pipeline="paddleocr",
                            image_path=image_path,
                            params=params,
                            response_raw=[item.json() for item in result if hasattr(item, "json")],
                        )
                signals.finished.emit(raw, image_path)
            except Exception as exc:
                signals.error.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _collect_params(self) -> dict:
        lang = self._lang_combo.currentText()
        ver = self._ocr_version_combo.currentData() or None

        ocr_init: dict[str, Any] = {
            "lang": lang,
            "ocr_version": ver,
            "use_doc_orientation_classify": self._use_orientation.isChecked(),
            "use_doc_unwarping": self._use_unwarping.isChecked(),
            "use_textline_orientation": self._use_textline_orient.isChecked(),
        }
        ocr_pred: dict[str, Any] = {
            "text_det_thresh": self._det_thresh.value(),
            "text_det_box_thresh": self._det_box_thresh.value(),
            "text_det_unclip_ratio": self._det_unclip_ratio.value(),
            "text_det_limit_side_len": self._det_limit_side_len.value(),
            "text_det_limit_type": self._det_limit_type_combo.currentText(),
            "text_rec_score_thresh": self._rec_score_thresh.value(),
            "return_word_box": self._return_word_box.isChecked(),
        }
        structure: dict[str, Any] = {
            "text_det_thresh": self._det_thresh.value(),
            "text_det_box_thresh": self._det_box_thresh.value(),
            "text_det_unclip_ratio": self._det_unclip_ratio.value(),
            "text_det_limit_side_len": self._det_limit_side_len.value(),
            "text_det_limit_type": self._det_limit_type_combo.currentText(),
            "text_rec_score_thresh": self._rec_score_thresh.value(),
            "layout_threshold": self._layout_thresh.value(),
            "layout_nms": self._layout_nms.isChecked(),
            "layout_unclip_ratio": self._layout_unclip.value(),
        }
        return {"ocr_init": ocr_init, "ocr_pred": ocr_pred, "structure": structure}

    def _on_ocr_finished(self, raw: dict, image_path: str) -> None:
        self._run_btn.setEnabled(True)
        self._last_raw = raw
        self._save_btn.setEnabled(True)
        self._log.append("OCR finished — parsing result…")
        meta = raw.get("_inspector_meta", {}) if isinstance(raw, dict) else {}
        if meta.get("fallback") == "paddleocr_layout_compat":
            self._log.append("INFO: PPStructureV3 unavailable locally; used PaddleOCR layout compatibility mode.")
        if meta.get("source") == "api":
            self._log.append(f"INFO: AiStudio API → {meta.get('api_url', '')}")
        self._append_runtime_meta(meta)
        try:
            doc = PaddleAdapter().parse(raw, source_path="<live>", image_path=image_path)
        except Exception as exc:
            self._log.append(f"[ERROR] Adapter parse failed: {exc}")
            return
        if meta.get("source") == "api":
            doc.engine = "paddle-api"
        elif meta.get("fallback") == "paddleocr_layout_compat":
            doc.engine = "paddle-local-compat"
        n_lines = sum(len(list(p.all_lines)) for p in doc.pages)
        n_blocks = sum(len(p.blocks) for p in doc.pages)
        n_chars = sum(len(list(p.all_chars)) for p in doc.pages)
        n_ocr_chars = sum(1 for p in doc.pages for ch in p.all_chars if ch.bbox_source == "ocr" and ch.bbox is not None)
        doc.parse_log.append(f"INFO: [IR] parsed lines={n_lines} chars={n_chars} ocr_char_token_nodes={n_ocr_chars}")
        self._log.append(
            f"OK — {len(doc.pages)} page(s), {n_blocks} block(s), {n_lines} line(s), {n_ocr_chars}/{n_chars} OCR char/token box(es)"
        )
        for msg in (doc.parse_log or []):
            if "WARNING" in str(msg) or "ERROR" in str(msg):
                self._log.append(f"  ⚠ {msg}")
        self._state.set_document(doc)
        # _on_doc_loaded in main window handles combo + set_active_page automatically

    def _append_runtime_meta(self, meta: dict[str, Any]) -> None:
        if not meta:
            return
        source = meta.get("source", "")
        pipeline = meta.get("pipeline", "")
        profile = meta.get("api_model_profile", "")
        endpoint = meta.get("api_url", "")
        self._log.append(f"Runtime: source={source or '-'} pipeline={pipeline or '-'} profile={profile or '-'}")
        if endpoint:
            self._log.append(f"Endpoint: {endpoint}")

        request_summary = meta.get("request_summary") or meta.get("api_request_summary") or {}
        if isinstance(request_summary, dict) and request_summary:
            visible = {k: v for k, v in request_summary.items() if k != "file"}
            self._log.append("Request params: " + json.dumps(visible, ensure_ascii=False, sort_keys=True))

        response_fields = meta.get("response_field_summary") or {}
        flattened_fields = meta.get("flattened_field_summary") or {}
        if isinstance(response_fields, dict):
            self._log.append("Response fields: " + json.dumps(response_fields, ensure_ascii=False, sort_keys=True))
        if isinstance(flattened_fields, dict):
            self._log.append("Flattened fields: " + json.dumps(flattened_fields, ensure_ascii=False, sort_keys=True))

    def _on_ocr_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._log.append(f"[ERROR] {msg}")

    def _save_result(self) -> None:
        if self._last_raw is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save OCR result", "ocr_result.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._last_raw, fh, ensure_ascii=False, indent=2)
            self._log.append(f"Saved → {path}")
        except Exception as exc:
            self._log.append(f"[ERROR] save failed: {exc}")
