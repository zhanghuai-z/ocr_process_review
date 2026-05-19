"""y 轴对应条 —— hproof-yaxis-verdicts 第一任务。

设计意图（用户原话）
====================

"横校图文对应必须明确成 y 轴元素对应" —— 不是 hover linkage、不是"字框能亮一下"，
而是 **图和文在纵向上对齐**：用户看着行图里第 i 个字，**向正下方看**，立刻
落到文本里第 i 个字。

实现
====

本 widget 是一条窄条（约 26 px 高），位于 _img_lbl 的正下方、editor 的正上方，
宽度与图像 pixmap 完全相同。

它按 ``line.chars[i].bbox`` 的 x 坐标（经过 _line_crop_origin 平移 + _render_scale
缩放后，落在 widget 像素坐标）绘制对应的文本字 text[i]：

  ─────────────────────────────────────────────
  │ 图像行                                       │ ← _img_lbl
  ├─────────────────────────────────────────────┤
  │ 文 │ 本 │ 字 │ 符 │ 行 │ ...                │ ← AlignmentRibbon
  │  ↑      ↑   ↑    ↑                          │  每字 x 中心 = 上方图字 x 中心
  ├─────────────────────────────────────────────┤
  │ [可编辑 editor]                              │ ← _editor（编辑入口仍在这里）
  └─────────────────────────────────────────────┘

每字颜色来源于 :class:`CharVerdict`（绿/橙/红/灰），证据链由 verdict.evidence
描述；user_modified 用细下划线提示（不洗白颜色）。

降级
====

满足以下任何一条 → 进入降级模式，**不画**字符 x 对应，只画提示语：

  - line.chars 为空 / 任一 char 缺 bbox
  - len(text) != len(chars)（用户编辑导致长度不一致；本轮 editor 已有 fixed-length
    保护，但仍可能在极端 case 下出现）
  - render_scale 未就绪（图像尚未渲染）

降级时显式写出 "未对齐 / 仅行级对应"，绝对不假装能 y 轴对应到具体某字 —— 这是
本轮 TASK 第一项的硬约束（"宁可诚实降级，也不要造一个'看着像对应实际是错位'
的版本"）。
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .char_verdict import CharVerdict, COLOR_UNVERIFIED


class AlignmentRibbon(QWidget):
    """图字 y 轴对应条。被 _LinePair 拥有；不持有 line 引用。"""

    HEIGHT = 26
    TICK_H = 4

    char_clicked = Signal(int)  # 用户点击到第 i 个字（-1 = 没点到任何字）

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(False)

        self._text: str = ""
        self._x_centers: List[Optional[float]] = []  # widget 像素坐标系
        self._widths: List[float] = []
        self._verdicts: List[Optional[CharVerdict]] = []
        self._cursor_idx: int = -1
        self._hover_idx: int = -1
        self._note: str = ""

        font = QFont()
        font.setFamilies([
            "Microsoft YaHei UI", "Noto Sans CJK SC", "PingFang SC", "SimSun",
        ])
        font.setPixelSize(16)
        self.setFont(font)

    # ── 对外接口 ────────────────────────────────────────────────

    def set_aligned(
        self,
        *,
        text: str,
        x_centers: List[Optional[float]],
        widths: List[float],
        verdicts: List[Optional[CharVerdict]],
        pixmap_width: int,
        cursor_idx: int = -1,
        hover_idx: int = -1,
    ) -> None:
        """正常模式：图字一一对应都已就绪，按 x 坐标绘制每字。"""
        self._text = text
        self._x_centers = list(x_centers)
        self._widths = list(widths)
        self._verdicts = list(verdicts)
        self._cursor_idx = cursor_idx
        self._hover_idx = hover_idx
        self._note = ""
        self.setFixedWidth(max(40, int(pixmap_width)))
        self.update()

    def set_degraded(self, reason: str, pixmap_width: int) -> None:
        """降级模式：写出 reason，不画任何字符；调用方应禁用图字点击逻辑。"""
        self._text = ""
        self._x_centers = []
        self._widths = []
        self._verdicts = []
        self._cursor_idx = -1
        self._hover_idx = -1
        self._note = reason
        self.setFixedWidth(max(40, int(pixmap_width)))
        self.update()

    def set_cursor_idx(self, idx: int) -> None:
        if idx == self._cursor_idx:
            return
        self._cursor_idx = idx
        self.update()

    def set_hover_idx(self, idx: int) -> None:
        if idx == self._hover_idx:
            return
        self._hover_idx = idx
        self.update()

    def is_degraded(self) -> bool:
        return bool(self._note)

    # ── 绘制 ────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), QColor("#fafbfc"))

            if self._note:
                p.setPen(QPen(QColor("#888"), 1))
                p.drawText(
                    self.rect(),
                    int(Qt.AlignmentFlag.AlignCenter),
                    self._note,
                )
                return

            text = self._text
            n = min(len(text), len(self._x_centers))
            for i in range(n):
                xc = self._x_centers[i]
                if xc is None:
                    continue
                w_raw = self._widths[i] if i < len(self._widths) else 12.0
                w = max(8.0, float(w_raw))
                left = int(round(xc - w / 2.0))
                right = int(round(xc + w / 2.0))
                cell = QRect(left, 1, max(1, right - left), self.height() - 2)

                # 当前字 / 悬停字背景
                if i == self._cursor_idx:
                    p.fillRect(cell, QColor("#cfe2ff"))
                elif i == self._hover_idx:
                    p.fillRect(cell, QColor("#e8f0fe"))

                # 顶部刻度（与图像列边对齐的视觉锚点）
                p.setPen(QPen(QColor("#9aa6b2"), 1))
                p.drawLine(left, 0, left, self.TICK_H)
                p.drawLine(max(left, right - 1), 0,
                           max(left, right - 1), self.TICK_H)

                # 文本字（verdict 颜色）
                verdict = self._verdicts[i] if i < len(self._verdicts) else None
                color = verdict.color if verdict is not None else COLOR_UNVERIFIED
                p.setPen(QPen(QColor(color), 1))
                p.drawText(
                    cell,
                    int(Qt.AlignmentFlag.AlignCenter),
                    text[i],
                )

                # user_modified 下划线（不洗白颜色，仅作行为标记）
                if verdict is not None and verdict.user_modified:
                    p.setPen(QPen(QColor("#5d4037"), 1, Qt.PenStyle.SolidLine))
                    p.drawLine(
                        left + 2, self.height() - 2,
                        right - 2, self.height() - 2,
                    )
        finally:
            p.end()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._note:
            self.char_clicked.emit(-1)
            return
        try:
            x = float(event.position().x())
        except AttributeError:
            x = float(event.x())
        best_i = -1
        best_d = float("inf")
        for i, xc in enumerate(self._x_centers):
            if xc is None:
                continue
            d = abs(xc - x)
            if d < best_d:
                best_d = d
                best_i = i
        self.char_clicked.emit(best_i)
