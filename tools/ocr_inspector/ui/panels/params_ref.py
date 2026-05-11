"""Params Reference panel — explains every PaddleOCR parameter.

Organised into categories matching paddle.py / run_ocr.py:
  - 检测 (det)
  - 识别 (rec)
  - 预处理
  - 版面 (layout)
  - 子模块开关
  - PPStructureV3 专有

Column: param name | type | default | description | 对输出的影响
"""
from __future__ import annotations
import json
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QHeaderView, QLabel, QSplitter, QTextEdit, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

# ---------------------------------------------------------------------------
# Parameter database
# ---------------------------------------------------------------------------
# Each entry: (param_name, type_str, default, description_zh, effect_zh)
_PARAMS: list[tuple] = [
    # ── 检测 Detection ──
    ("__cat__", "检测 (Text Detection)", "", "", ""),
    ("textDetThresh", "float", "0.3",
     "像素级二值化阈值",
     "值越小，越多像素被判定为文字边缘，召回↑精确度↓；太低会引入噪声框"),
    ("textDetBoxThresh", "float", "0.6",
     "检测框得分过滤阈值",
     "过滤低置信度框；值越大，越严格，精确度↑召回↓"),
    ("textDetUnclipRatio", "float", "1.5",
     "检测框扩张比例 (Vatti clipping algorithm)",
     "值越大框越宽松，有助于包住完整字符；过大会合并相邻行"),
    ("textDetLimitSideLen", "int", "960",
     "送入检测网络前的最长边缩放上限 (px)",
     "值越小速度越快但细小文字可能丢失；值越大细节越好但内存/速度代价高"),
    ("textDetLimitType", "str", "max",
     "缩放策略：max=限制最长边，min=限制最短边",
     "max 适合普通图片；min 用于极端长条图"),

    # ── 识别 Recognition ──
    ("__cat__", "识别 (Text Recognition)", "", "", ""),
    ("textRecScoreThresh", "float", "0.0",
     "识别结果置信度过滤阈值",
     "过滤低置信度行文本；调高可减少乱码，但可能漏掉难字"),
    ("returnWordBox", "bool", "False",
     "是否返回字/词级 bounding box (text_word_region)",
     "开启后 overall_ocr_res.text_word_region 有值；canvas 中橙色字框才会出现"),

    # ── 预处理 Preprocessing ──
    ("__cat__", "预处理 (Preprocessing)", "", "", ""),
    ("useDocOrientationClassify", "bool", "True",
     "文档方向分类（0°/90°/180°/270°自动旋转）",
     "开启后会自动旋转送入检测网络的图像；关闭可省时间但竖排文档会漏字"),
    ("useDocUnwarping", "bool", "False",
     "文档去扭曲 (TPS/STN unwarp)",
     "对扫描件有用；开启会改变坐标空间（坐标仍在处理后图上）"),
    ("useTextlineOrientation", "bool", "True",
     "文本行方向分类（横排/竖排）",
     "开启后能处理竖排日文/中文；关闭节省约 5ms/行"),

    # ── 版面 Layout ──
    ("__cat__", "版面检测 (Layout)", "", "", ""),
    ("layoutThreshold", "float", "0.5",
     "版面检测框置信度阈值",
     "与 textDetBoxThresh 类似但作用于 layout_det_res；调高过滤噪声框"),
    ("layoutNms", "bool", "True",
     "版面检测框 NMS 开关",
     "开启消除重叠框；关闭时重叠框全部保留（调试用）"),
    ("layoutUnclipRatio", "float", "1.5",
     "版面检测框扩张比例",
     "与 textDetUnclipRatio 类似，控制布局框大小"),
    ("layoutMergeBboxesMode", "str", "large",
     "布局框合并策略：large / small / union",
     "large=大框优先；small=小框优先；union=全并"),

    # ── 子模块 ──
    ("__cat__", "子模块开关 (Sub-modules)", "", "", ""),
    ("useSealRecognition", "bool", "False",
     "印章识别模块",
     "开启后 parsing_res_list 中会出现 seal 类型 block；关闭节省内存"),
    ("useTableRecognition", "bool", "True",
     "表格识别模块",
     "开启后表格区域有结构化输出（HTML）；关闭退化为普通文字行"),
    ("useFormulaRecognition", "bool", "False",
     "公式识别模块 (LaTeX)",
     "开启后识别 formula block；关闭节省显存约 400MB"),
    ("useChartRecognition", "bool", "False",
     "图表识别模块",
     "开启后对 figure block 做图表解析"),

    # ── PPStructureV3 专有 ──
    ("__cat__", "PPStructureV3 专有 (predict params)", "", "", ""),
    ("usePdfiaDocument", "bool", "True",
     "返回 PdfiaDocument 结构（含 markdown / json）",
     "关闭退化为 dict 列表；大多数调试时应开启"),
]

_HEADER_COL_WIDTHS = [180, 60, 80, 240, 280]
_HEADER_LABELS = ["参数名", "类型", "默认值", "说明", "对输出的影响"]


@dataclass(frozen=True)
class ParamMatrixRow:
    name: str
    current_value: object
    default_value: object
    models: str
    endpoint: str
    sent: bool
    reason: str


_MATRIX_SPECS: dict[str, tuple[object, str, str]] = {
    "returnWordBox": (False, "PP-OCRv5, PP-StructureV3", "请求 text_word/text_word_region；VL family 不支持真实 word/char bbox。"),
    "useDocOrientationClassify": (True, "全部官方 profile", "坐标稳定保护项；Inspector 默认关闭，避免服务端旋转后 bbox 与原图漂移。"),
    "useDocUnwarping": (False, "全部官方 profile", "坐标稳定保护项；开启会改变处理后图坐标空间。"),
    "useTextlineOrientation": (True, "全部官方 profile", "坐标稳定保护项；Inspector 默认关闭，避免行方向分类造成坐标语义变化。"),
    "textDetThresh": (0.3, "PP-OCRv5, PP-StructureV3", "文本检测阈值，仅 OCR detector/recognizer family 接收。"),
    "textDetBoxThresh": (0.6, "PP-OCRv5, PP-StructureV3", "检测框分数阈值，仅 OCR detector/recognizer family 接收。"),
    "textDetUnclipRatio": (1.5, "PP-OCRv5, PP-StructureV3", "检测框扩张比例，仅 OCR detector/recognizer family 接收。"),
    "textDetLimitSideLen": (960, "PP-OCRv5, PP-StructureV3", "检测前最长边限制，仅 OCR detector/recognizer family 接收。"),
    "textDetLimitType": ("max", "PP-OCRv5, PP-StructureV3", "检测缩放策略，仅 OCR detector/recognizer family 接收。"),
    "textRecScoreThresh": (0.0, "PP-OCRv5, PP-StructureV3", "识别分数过滤阈值，仅 OCR detector/recognizer family 接收。"),
    "layoutThreshold": (0.5, "本地 PPStructureV3", "共享 AiStudio body 暂不发送；本地 Run OCR 的 PPStructureV3 控件使用。"),
    "layoutNms": (True, "本地 PPStructureV3", "共享 AiStudio body 暂不发送；避免不同云端 profile 因未知字段拒绝请求。"),
    "layoutUnclipRatio": (1.5, "本地 PPStructureV3", "共享 AiStudio body 暂不发送；本地 layout 调试使用。"),
}


def build_paddle_param_matrix(profile: str = "pp-structurev3", endpoint_url: str | None = None) -> tuple[list[ParamMatrixRow], dict]:
    """Build the visible parameter matrix and final API payload preview."""
    from app.core.api_profiles import API_MODEL_PROFILES, get_api_request_options

    profile_spec = API_MODEL_PROFILES.get(profile) or API_MODEL_PROFILES["pp-structurev3"]
    endpoint = endpoint_url or str(profile_spec.get("url") or "")
    options = get_api_request_options(profile, endpoint)
    payload = {"file": "<base64 omitted>", "fileType": 1}
    payload.update(options)

    family = profile_spec.get("request_family")
    rows: list[ParamMatrixRow] = []
    for name, (default, models, note) in _MATRIX_SPECS.items():
        sent = name in options
        current = options.get(name, default)
        if sent:
            reason = "已发送：当前 profile 支持该参数。" if family == "ocr-word-box" else "已发送：坐标稳定保护项。"
        elif family == "vl-layout" and name in ("returnWordBox", "textDetThresh", "textDetBoxThresh", "textDetUnclipRatio", "textDetLimitSideLen", "textDetLimitType", "textRecScoreThresh"):
            reason = "未发送：PaddleOCR-VL/VL-1.5 返回 line/block/markdown，不承诺真实 token/char bbox，可能拒绝 OCR detector 参数。"
        else:
            reason = f"未发送：{note}"
        rows.append(ParamMatrixRow(
            name=name,
            current_value=current,
            default_value=default,
            models=models,
            endpoint=str(profile_spec.get("endpoint_suffix") or ""),
            sent=sent,
            reason=reason,
        ))
    return rows, payload


class ParamsRefPanel(QWidget):
    """Read-only parameter reference panel (no state subscription)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel(
            "<b>PaddleOCR 参数速查</b> · "
            "<small>坐标空间: 所有 bbox 坐标均在 <b>原始图像像素空间</b> "
            "(dataInfo.width × dataInfo.height)；"
            "若开启 useDocUnwarping 则坐标基于处理后图</small>"
        )
        title.setWordWrap(True)
        layout.addWidget(title)

        tree = QTreeWidget()
        tree.setColumnCount(5)
        tree.setHeaderLabels(_HEADER_LABELS)
        tree.setAlternatingRowColors(True)
        tree.setRootIsDecorated(True)
        tree.setUniformRowHeights(False)
        tree.setWordWrap(True)
        for i, w in enumerate(_HEADER_COL_WIDTHS):
            tree.setColumnWidth(i, w)

        current_cat: QTreeWidgetItem | None = None

        for entry in _PARAMS:
            if entry[0] == "__cat__":
                current_cat = QTreeWidgetItem(tree, [entry[1]])
                current_cat.setFont(0, _bold_font())
                current_cat.setExpanded(True)
                continue
            name, type_, default, desc, effect = entry
            row = QTreeWidgetItem(current_cat or tree)
            row.setText(0, name)
            row.setText(1, type_)
            row.setText(2, default)
            row.setText(3, desc)
            row.setText(4, effect)
            row.setToolTip(3, desc)
            row.setToolTip(4, effect)

        layout.addWidget(tree)

        matrix_title = QLabel("<b>当前请求矩阵</b> · 显示参数是否会进入最终 API payload；VL 不支持真实 char/token bbox 时必须诚实降级。")
        matrix_title.setWordWrap(True)
        layout.addWidget(matrix_title)

        self._profile_combo = QComboBox()
        try:
            from app.core.api_profiles import API_MODEL_PROFILES
            for key, spec in API_MODEL_PROFILES.items():
                self._profile_combo.addItem(str(spec.get("label") or key), key)
        except Exception:
            self._profile_combo.addItem("PP-StructureV3", "pp-structurev3")
        self._profile_combo.currentIndexChanged.connect(self._refresh_matrix)
        layout.addWidget(self._profile_combo)

        self._matrix = QTreeWidget()
        self._matrix.setColumnCount(7)
        self._matrix.setHeaderLabels(["参数", "当前值", "默认值", "适用模型", "端点", "发送", "原因"])
        for i, w in enumerate([170, 90, 90, 170, 120, 60, 420]):
            self._matrix.setColumnWidth(i, w)
        self._matrix.setAlternatingRowColors(True)
        self._matrix.setRootIsDecorated(False)
        self._matrix.setUniformRowHeights(False)
        self._matrix.setWordWrap(True)
        layout.addWidget(self._matrix, 1)

        self._payload = QTextEdit()
        self._payload.setReadOnly(True)
        self._payload.setMaximumHeight(140)
        self._payload.setStyleSheet("font-size:11px; font-family:monospace;")
        layout.addWidget(QLabel("<b>最终 payload 预览</b>"))
        layout.addWidget(self._payload)
        self._refresh_matrix()

    def _refresh_matrix(self) -> None:
        profile = self._profile_combo.currentData() or "pp-structurev3"
        rows, payload = build_paddle_param_matrix(str(profile))
        self._matrix.clear()
        for row in rows:
            item = QTreeWidgetItem(self._matrix, [
                row.name,
                str(row.current_value),
                str(row.default_value),
                row.models,
                row.endpoint,
                "是" if row.sent else "否",
                row.reason,
            ])
            if not row.sent:
                item.setForeground(5, Qt.darkGray)
            item.setToolTip(6, row.reason)
        self._payload.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))


def _bold_font():
    from PySide6.QtGui import QFont
    f = QFont()
    f.setBold(True)
    return f
