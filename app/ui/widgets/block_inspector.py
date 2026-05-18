"""\u53f3\u4fa7 Block Inspector\uff1a\u5c55\u793a\u9009\u4e2d\u7248\u9762\u5757\u7684\u8be6\u7ec6\u5c5e\u6027 + \u5f53\u524d\u9875\u5757\u7c7b\u578b\u7edf\u8ba1\u3002

\u4ea4\u4ed8\uff1a\u7c7b\u578b / \u7f6e\u4fe1\u5ea6 / bbox / \u884c\u5b57\u7edf\u8ba1 / \u6765\u6e90 \uff0b \u300c\u7edf\u8ba1\u300d\u5757\u7c7b\u578b\u8fdb\u5ea6\u6761\u3002
\u9762\u677f\u4f9b LayoutPanel\uff08\u4ee5\u53ca\u5176\u4ed6\u9700\u8981 block \u8be6\u60c5\u7684\u5730\u65b9\uff09\u590d\u7528\u3002
"""
from __future__ import annotations
from typing import Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget,
)

from app.models import Block, Page
from app.models.enums import BlockType
from app.ui.widgets.confidence_badge import ConfidenceBadge


# \u5757\u7c7b\u578b \u2192 \u4e2d\u6587\u77ed\u540d\u79f0\uff08\u7edf\u8ba1\u6761\u663e\u793a\u7528\uff09
_BLOCK_TYPE_LABEL: dict[str, str] = {
    BlockType.TEXT.value:           "\u6b63\u6587",
    BlockType.TITLE.value:          "\u6807\u9898",
    BlockType.FIGURE.value:         "\u56fe",
    BlockType.FIGURE_CAPTION.value: "\u56fe\u6ce8",
    BlockType.TABLE.value:          "\u8868",
    BlockType.TABLE_CAPTION.value:  "\u8868\u6ce8",
    BlockType.REFERENCE.value:      "\u5f15\u6587",
    BlockType.EQUATION.value:       "\u516c\u5f0f",
    BlockType.UNKNOWN.value:        "\u5176\u4ed6",
}


class _StatsRow(QWidget):
    """\u7edf\u8ba1\u533a\u5355\u6761\uff1a\u6807\u7b7e + \u8fdb\u5ea6\u6761 + \u8ba1\u6570\u3002"""

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(18)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._lbl = QLabel(label)
        self._lbl.setObjectName("statsLabel")
        self._lbl.setFixedWidth(36)
        row.addWidget(self._lbl)

        self._bar = QProgressBar()
        self._bar.setObjectName("statsBar")
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setRange(0, 100)
        row.addWidget(self._bar, 1)

        self._count = QLabel("0")
        self._count.setObjectName("statsCount")
        self._count.setFixedWidth(28)
        self._count.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._count)

    def set_value(self, count: int, max_count: int) -> None:
        pct = int(round(count / max_count * 100)) if max_count > 0 else 0
        self._bar.setValue(pct)
        self._count.setText(str(count))


class BlockInspector(QWidget):
    """\u9009\u4e2d\u7248\u9762\u5757\u7684\u53f3\u4fa7\u5c5e\u6027\u9762\u677f\u3002"""

    WIDTH = 320

    # \u7edf\u8ba1\u533a\u9ed8\u8ba4\u5c55\u793a\u7684\u7c7b\u578b\u987a\u5e8f
    _STATS_TYPES = (
        BlockType.TEXT.value,
        BlockType.TITLE.value,
        BlockType.FIGURE.value,
        BlockType.FIGURE_CAPTION.value,
        BlockType.TABLE.value,
        BlockType.UNKNOWN.value,
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebarBar")
        self.setMinimumWidth(260)
        self.setMaximumWidth(380)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── \u9009\u4e2d\u5757\u533a ────────────────────────────────
        title = QLabel("\u9009\u4e2d\u5757")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        root.addWidget(self._field_label("\u7c7b\u578b"))
        self._lbl_type = QLabel("\u2014")
        self._lbl_type.setObjectName("fieldLabel")
        root.addWidget(self._lbl_type)

        root.addWidget(self._field_label("\u7f6e\u4fe1\u5ea6"))
        self._conf_badge = ConfidenceBadge(1.0)
        self._conf_badge.hide()
        self._lbl_conf_na = QLabel("\u65e0")
        self._lbl_conf_na.setObjectName("muted")
        root.addWidget(self._conf_badge, alignment=Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self._lbl_conf_na)

        root.addWidget(self._field_label("\u8fb9\u754c\u6846"))
        self._lbl_bbox = QLabel("\u2014")
        self._lbl_bbox.setObjectName("fieldLabel")
        root.addWidget(self._lbl_bbox)

        root.addWidget(self._field_label("\u884c / \u5b57\u7edf\u8ba1"))
        self._lbl_stats = QLabel("\u2014")
        self._lbl_stats.setObjectName("fieldLabel")
        root.addWidget(self._lbl_stats)

        root.addWidget(self._field_label("\u6765\u6e90"))
        self._lbl_source = QLabel("\u2014")
        self._lbl_source.setObjectName("muted")
        root.addWidget(self._lbl_source)

        # ── \u5f53\u524d\u9875\u7edf\u8ba1\u533a ───────────────────────────
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setObjectName("inspectorSep")
        sep1.setFixedHeight(1)
        root.addSpacing(4)
        root.addWidget(sep1)

        stats_title = QLabel("\u7edf\u8ba1")
        stats_title.setObjectName("sectionTitle")
        root.addWidget(stats_title)

        self._stats_rows: dict[str, _StatsRow] = {}
        for type_val in self._STATS_TYPES:
            row = _StatsRow(_BLOCK_TYPE_LABEL.get(type_val, type_val))
            self._stats_rows[type_val] = row
            root.addWidget(row)

        self._lbl_stats_empty = QLabel("\u672c\u9875\u65e0\u5757")
        self._lbl_stats_empty.setObjectName("muted")
        self._lbl_stats_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_stats_empty.hide()
        root.addWidget(self._lbl_stats_empty)

        root.addStretch(1)

        # \u7a7a\u72b6\u6001\u63d0\u793a
        self._empty_hint = QLabel("\u70b9\u51fb\u56fe\u7247\u4e2d\u7684\u5757\u67e5\u770b\u8be6\u60c5")
        self._empty_hint.setObjectName("muted")
        self._empty_hint.setWordWrap(True)
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._empty_hint)

        self.clear()

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("noteLabel")
        return lbl

    # ── public API ────────────────────────────────────────────

    def set_block(self, block: Optional[Block]) -> None:
        if block is None:
            self._clear_selection_fields()
            return

        self._empty_hint.hide()

        bt = getattr(block, "block_type", None)
        self._lbl_type.setText(getattr(bt, "value", str(bt) if bt else "\u2014"))

        bb = getattr(block, "bbox", None)
        if bb is not None:
            self._lbl_bbox.setText(f"x={bb.x}  y={bb.y}\nw={bb.w}  h={bb.h}")
        else:
            self._lbl_bbox.setText("\u2014")

        conf = getattr(block, "avg_confidence", None)
        if conf is not None:
            self._conf_badge.set_score(float(conf))
            self._conf_badge.show()
            self._lbl_conf_na.hide()
        else:
            self._conf_badge.hide()
            self._lbl_conf_na.show()

        lines = getattr(block, "lines", []) or []
        line_count = len(lines)
        char_count = sum(len(getattr(ln, "chars", []) or []) for ln in lines)
        self._lbl_stats.setText(f"{line_count} \u884c  /  {char_count} \u5b57")

        src = getattr(block, "source", None)
        self._lbl_source.setText(getattr(src, "value", str(src) if src else "\u2014"))

    def set_page_stats(self, page: Optional[Page]) -> None:
        """\u66f4\u65b0\u300c\u7edf\u8ba1\u300d\u533a\u7684\u5757\u7c7b\u578b\u5206\u5e03\u8fdb\u5ea6\u6761\u3002"""
        blocks: Sequence[Block] = getattr(page, "blocks", []) if page else []
        if not blocks:
            for row in self._stats_rows.values():
                row.set_value(0, 0)
                row.setVisible(False)
            self._lbl_stats_empty.setVisible(True)
            return

        # \u6309\u7c7b\u578b\u8ba1\u6570
        counts: dict[str, int] = {t: 0 for t in self._STATS_TYPES}
        for blk in blocks:
            t = getattr(getattr(blk, "block_type", None), "value", None)
            if t in counts:
                counts[t] += 1
            else:
                counts[BlockType.UNKNOWN.value] += 1
        max_count = max(counts.values()) if counts else 0
        for type_val, row in self._stats_rows.items():
            row.set_value(counts[type_val], max_count)
            row.setVisible(True)
        self._lbl_stats_empty.setVisible(False)

    def clear(self) -> None:
        self._clear_selection_fields()
        self.set_page_stats(None)

    def _clear_selection_fields(self) -> None:
        self._lbl_type.setText("\u2014")
        self._lbl_bbox.setText("\u2014")
        self._lbl_stats.setText("\u2014")
        self._lbl_source.setText("\u2014")
        self._conf_badge.hide()
        self._lbl_conf_na.show()
        self._empty_hint.show()


__all__ = ["BlockInspector"]
