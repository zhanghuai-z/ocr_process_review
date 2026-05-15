"""横校面板（PRD 3.2）：滚动列表，每行显示【图像行 + 识别文本】对。

布局（参照 ui-2.jpg）：
  ┌────────────────────────────────────────────────────────────┐
  │  ↑上一行  ↓下一行  [保存 Ctrl+S]  [标记 F5]  [跳过 F6]     │ ← 工具栏
  ├────────────────────────────────────────────────────────────┤
  │ 图像行  1 │ [扫描图像行] ─────────────────────────────────  │
  │ 识别文本 1│  识别出的文字文本                           ●待确认 │
  │───────────────────────────────────────────────────────────│
  │ 图像行  2 │ [扫描图像行]                                    │
  │ 识别文本 2│  文字文本                                   ●已确认 │
  │  ─ ─ ─ ─ ─ ─ ─ ─ active row: 蓝色左边框 ─ ─ ─ ─ ─ ─ ─ ─ ─│
  │▌图像行  3 │ [扫描图像行]                                    │  ← active
  │▌识别文本 3│▶ [可编辑文本区]                             1处差异│  ← active
  └────────────────────────────────────────────────────────────┘
  状态栏：总字数 12,523 | 差异 128 (1.02%)  ● 与原文一致 ● 疑似 ● 不确认 ● 已标记

快捷键（文本编辑框内有效）：
  Enter       — 确认当前行并跳到下一行
  Ctrl+↑/↓    — 上一行 / 下一行（不确认）
  F5          — 跳到下一疑点 / 标记当前行
  F6          — 跳过本行（状态不变）
  Escape      — 还原当前行到 OCR 原始文本
  Ctrl+S      — 保存全部修改
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor, QImage, QKeySequence, QPixmap, QShortcut,
    QTextCharFormat, QTextCursor,
)
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from app.models import Block, Line, OcrProject, Page, ProofStatus
from app.core.page_image_cache import PageImageCache
from app.ui.proof.char_cell_row import CharCellRow
from app.core.proof_line_utils import iter_unique_page_hproof_lines
from app.core.proof_state_bus import ProofStateBus
from app.core import quality_probe as qp
from app.services.proof_probe_text_service import (
    displayed_text as _displayed_text,
    save_displayed_edit as _save_displayed_edit,
    resolve_block_line_index as _resolve_block_line_index,
)
from app.services.proof_image_service import clamp_line_box_pixels
from app.ui.widgets.confidence_badge import ConfidenceBadge

# ── 样式常量 ──────────────────────────────────────────────────
ROW_PAD_Y    = 4     # 裁图上下各加 4px
IMAGE_ROW_H  = 32    # 行图像显示高度（px）
TEXT_FONT_PX = 24    # 30px 回调为缩小 20%，继续贴近 32px 行图中线
TEXT_EDITOR_MAX_H = 50
LINE_PAIR_H = 54
# Phase 17 blocker：字格模式下需要为 CharCellRow 留够竖向空间。
# CharCellRow 自身固定高 = IMG_H(36) + EDIT_H(26) + 6 内边距 = 68，
# 加上 _LinePair 上下各 ~4 px 自身布局 padding，给 76 px 不裁切。
CELL_PAIR_H = 76
TEXT_FONT_FAMILY = "'Microsoft YaHei UI','Noto Sans CJK SC','PingFang SC','SimSun',sans-serif"
TEXT_DEFAULT_COLOR = "#c5221f"
TEXT_VISITED_COLOR = "#188038"
LABEL_W      = 88    # 左侧行号列宽
STATUS_W     = 80    # 右侧状态列宽
LOW_CONF     = 0.80

_STATUS_COLOR = {
    ProofStatus.OK:           "#4CAF50",
    ProofStatus.MODIFIED:     "#1a73e8",
    ProofStatus.AUTO_FLAGGED: "#FF9800",
    ProofStatus.UNCHECKED:    "#c8d0db",
}
_STATUS_LABEL = {
    ProofStatus.OK:           "已确认",
    ProofStatus.MODIFIED:     "已修改",
    ProofStatus.AUTO_FLAGGED: "⚑ 疑点",
    ProofStatus.UNCHECKED:    "待确认",
}


# ─────────────────────────────────────────────────────────────
# 行内文本编辑器（拦截 Enter/方向键/F 键）
# ─────────────────────────────────────────────────────────────

class _RowEditor(QPlainTextEdit):
    """嵌入行内的单行文本编辑器，拦截专用快捷键。"""

    confirm_requested = Signal()
    prev_requested    = Signal()
    next_requested    = Signal()
    flag_requested    = Signal()
    skip_requested    = Signal()
    revert_requested  = Signal()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        mod = event.modifiers()
        no_mod = mod == Qt.KeyboardModifier.NoModifier
        ctrl   = mod == Qt.KeyboardModifier.ControlModifier

        if key == Qt.Key.Key_Return and no_mod:
            self.confirm_requested.emit(); return
        if key == Qt.Key.Key_Up and ctrl:
            self.prev_requested.emit(); return
        if key == Qt.Key.Key_Down and ctrl:
            self.next_requested.emit(); return
        if key == Qt.Key.Key_F5:
            self.flag_requested.emit(); return
        if key == Qt.Key.Key_F6:
            self.skip_requested.emit(); return
        if key == Qt.Key.Key_Escape:
            self.revert_requested.emit(); return
        super().keyPressEvent(event)


# ─────────────────────────────────────────────────────────────
# 评测位 (quality probe) 显示↔真实 桥接
# 共享实现见 app.services.proof_probe_text_service；本文件保留同名局部别名
# 以保持调用点不变（_displayed_text / _save_displayed_edit / _resolve_block_line_index）。
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# 单行对（图像行 + 识别文本行）控件
# ─────────────────────────────────────────────────────────────

class _LinePair(QFrame):
    """显示一行的【扫描图像行 + OCR 识别文本】对。"""

    clicked       = Signal(int)   # 发出自身 idx
    text_saved    = Signal(int, str)  # (idx, new_text)
    confirmed     = Signal(int)   # Enter 确认
    prev_req      = Signal()
    next_req      = Signal()
    flag_req      = Signal()
    skip_req      = Signal()

    def __init__(
        self,
        idx: int,
        block: Block,
        line: Line,
        page: Page,
        line_in_page: int,   # 在页面内的行序号（1-based，用于显示）
        cache: PageImageCache,
        parent=None,
    ):
        super().__init__(parent)
        self._idx          = idx
        self._block        = block
        self._line         = line
        self._page         = page
        self._line_in_page = line_in_page
        self._cache        = cache
        self._active       = False
        self._image_loaded = False
        self._line_crop = None
        self._line_crop_origin: tuple[int, int] = (0, 0)
        # Phase 11 task 2：字格模式开关 + 懒构建的 CharCellRow
        self._cell_mode_enabled = False
        self._cell_row: CharCellRow | None = None
        # Phase 12：cell_mode 下当前聚焦的 cell idx；用于行图像高亮
        self._cell_focus_idx: int | None = None
        # 最近一次行图像缩放比例，用于把 _img_lbl 上的点击位置反查回原图坐标
        self._render_scale: float = 1.0

        self.setObjectName("linePair")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._build_ui()

    # ── 构建 ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setFixedHeight(LINE_PAIR_H)
        # Phase 17 blocker：先用普通高度，cell mode 开启时由 _apply_pair_height 切到 CELL_PAIR_H。
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QWidget()
        row = QHBoxLayout(content)
        row.setContentsMargins(0, 2, 8, 2)
        row.setSpacing(6)
        root.addWidget(content)

        # 蓝色激活条（左边框）
        self._active_bar = QWidget()
        self._active_bar.setFixedWidth(4)
        self._active_bar.setStyleSheet("background: transparent;")
        row.addWidget(self._active_bar)
        self._active_bar2 = self._active_bar

        self._lbl_img_hdr = QLabel(f"图像行 {self._line_in_page}")
        self._lbl_img_hdr.setFixedWidth(58)
        self._lbl_img_hdr.setObjectName("muted")
        self._lbl_img_hdr.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._lbl_img_hdr.setStyleSheet("font-size:11px; color:#999; padding-right:8px;")
        row.addWidget(self._lbl_img_hdr)

        self._img_lbl = QLabel()
        self._img_lbl.setFixedHeight(IMAGE_ROW_H)
        self._img_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._img_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._img_lbl.setStyleSheet("background:#fafbfc; padding:2px 0;")
        row.addWidget(self._img_lbl, 5, Qt.AlignmentFlag.AlignVCenter)

        self._lbl_txt_hdr = QLabel(f"识别文本 {self._line_in_page}")
        self._lbl_txt_hdr.setFixedWidth(58)
        self._lbl_txt_hdr.setObjectName("muted")
        self._lbl_txt_hdr.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._lbl_txt_hdr.setStyleSheet("font-size:11px; color:#999; padding-right:8px;")
        row.addWidget(self._lbl_txt_hdr)

        # 文本展示（非激活）
        self._text_lbl = QLabel(_displayed_text(self._line, self._page, self._block))
        self._text_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._text_lbl.setStyleSheet(
            f"font-family:{TEXT_FONT_FAMILY}; font-size:{TEXT_FONT_PX}px; "
            "padding:0; color:#222;"
        )
        self._text_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._text_lbl.setWordWrap(False)
        self._text_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        row.addWidget(self._text_lbl, 6, Qt.AlignmentFlag.AlignVCenter)

        # 文本编辑器（激活时可见）
        self._editor = _RowEditor()
        self._editor.setStyleSheet(
            f"font-family:{TEXT_FONT_FAMILY}; font-size:{TEXT_FONT_PX}px; "
            "padding:0 6px;"
        )
        self._editor.setFixedHeight(TEXT_EDITOR_MAX_H)
        self._editor.document().setDocumentMargin(0)
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._editor.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._editor.hide()
        # 信号转发
        self._editor.confirm_requested.connect(lambda: self.confirmed.emit(self._idx))
        self._editor.prev_requested.connect(self.prev_req)
        self._editor.next_requested.connect(self.next_req)
        self._editor.flag_requested.connect(self.flag_req)
        self._editor.skip_requested.connect(self.skip_req)
        self._editor.revert_requested.connect(self._revert)
        self._editor.selectionChanged.connect(self._render_line_image)
        self._editor.cursorPositionChanged.connect(self._render_line_image)
        row.addWidget(self._editor, 6, Qt.AlignmentFlag.AlignVCenter)

        # 状态标签
        self._status_lbl = QLabel()
        self._status_lbl.setFixedWidth(STATUS_W)
        self._status_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._status_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._refresh_status()
        row.addWidget(self._status_lbl)

        # 注册点击区域。_img_lbl 走 cell_mode-aware 的反查（普通模式下仍是行激活）。
        for w in (self, self._text_lbl, self._lbl_img_hdr, self._lbl_txt_hdr):
            w.mousePressEvent = self._on_click  # type: ignore[method-assign]
        self._img_lbl.mousePressEvent = self._img_clicked_lookup  # type: ignore[method-assign]

    # ── 对外接口 ──────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        blue = "#1a73e8"
        if active:
            bar_style = f"background:{blue}; border-radius:2px;"
            bg = "#f0f6ff"
        else:
            # 保存编辑内容
            if not self._editor.isHidden():
                new_text = self._editor.toPlainText()
                # 注意：new_text 是“显示空间”文本，要和显示空间比较
                if new_text != _displayed_text(self._line, self._page, self._block):
                    self.text_saved.emit(self._idx, new_text)
            bar_style = "background:transparent;"
            bg = "transparent"

        self._active_bar.setStyleSheet(bar_style)
        self._active_bar2.setStyleSheet(bar_style)
        self._img_lbl.setStyleSheet(f"background:#fafbfc; padding:2px 0;")
        self.setStyleSheet(
            f"QFrame#linePair {{ background:{bg}; }}"
            if active else ""
        )

        if active:
            self._text_lbl.hide()
            self._refresh_active_widgets()
        else:
            self._editor.hide()
            if self._cell_row is not None:
                self._cell_row.hide()
            self._cell_focus_idx = None
            self._text_lbl.setText(_displayed_text(self._line, self._page, self._block))
            self._text_lbl.show()
            self._render_line_image()

        self._refresh_status()

    def set_cell_mode(self, enabled: bool) -> None:
        """切换字格模式。仅当本行 active 时才会真正切换可见控件，
        非 active 行只记录状态，等下次 set_active(True) 时按状态构建。

        Phase 17 blocker：必须在所有 pair 上同步调整 fixedHeight，
        否则 cell mode 下 CharCellRow(68) 会被 LINE_PAIR_H(54) 裁掉下半截，
        且全局 panel 滚动时不希望行高跳变。

        Phase 19 blocker 1：切换前先把 active 行 editor 里 in-flight
        未保存文本 flush 出去，否则 _refresh_active_widgets 会直接 hide
        editor，新文本就丢了。"""
        if self._cell_mode_enabled == enabled:
            return
        # Phase 19 blocker 1：切换前先 flush editor 未保存文本
        self._flush_editor_if_dirty()
        self._cell_mode_enabled = enabled
        self._apply_pair_height()
        if self._active:
            # 重新走一遍 active=True 路径，让可见控件按新状态切换
            self._refresh_active_widgets()

    def _flush_editor_if_dirty(self) -> None:
        """Phase 19 blocker 1：若 active 行 editor 当前可见且文本与显示空间
        不一致，则发 text_saved，让面板把它通过 _save_displayed_edit 落盘。
        与 set_active(False) 里现有逻辑同口径。"""
        if not self._active:
            return
        if self._editor.isHidden():
            return
        new_text = self._editor.toPlainText()
        if new_text != _displayed_text(self._line, self._page, self._block):
            self.text_saved.emit(self._idx, new_text)

    def _apply_pair_height(self) -> None:
        """Phase 17 blocker：按 cell mode 状态切 pair fixedHeight，避免裁切。"""
        target = CELL_PAIR_H if self._cell_mode_enabled else LINE_PAIR_H
        if self.height() != target:
            self.setFixedHeight(target)

    def _refresh_active_widgets(self) -> None:
        """active 状态下按 _cell_mode_enabled 切换 editor / cell_row 显示。"""
        # 不论何种模式，先统一把两侧隐藏；激活的那侧再 show
        self._editor.hide()
        if self._cell_row is not None:
            self._cell_row.hide()
        if self._cell_mode_enabled:
            if self._cell_row is None:
                # Phase 18 blocker 2：用显示空间文本（含 quality-probe fake_char）
                # 构建字格，与保存路径 _save_displayed_edit 的输入空间一致。
                self._cell_row = CharCellRow(
                    self._line, self._page, self._cache,
                    parent=self,
                    display_text=_displayed_text(self._line, self._page, self._block),
                )
                self._cell_row.text_committed.connect(self._on_cell_text_committed)
                self._cell_row.commit_requested.connect(
                    lambda: self.confirmed.emit(self._idx)
                )
                self._cell_row.focus_changed.connect(self._on_cell_focus_changed)
                # Phase 14b: cell 末尾 → / Tab 越界 → 跨行（复用行级 next_req/prev_req）
                self._cell_row.next_off_end.connect(self.next_req)
                self._cell_row.prev_off_start.connect(self.prev_req)
                # 插入到 root layout 的 editor 同位置
                root = self.layout()
                # editor 在 layout 中是倒数第二个 widget（最后一个是 status_lbl）
                root.insertWidget(root.count() - 1, self._cell_row, 6)
            self._cell_row.show()
            if self._cell_row.has_cells:
                self._cell_row.focus_first()
        else:
            self._editor.setPlainText(_displayed_text(self._line, self._page, self._block))
            self._highlight_low_conf()
            self._editor.show()
            self._editor.setFocus()
        self.load_image()

    def _invalidate_cell_row(self) -> None:
        """Phase 16 blocker: 销毁缓存的 _cell_row。

        使用场景（任何会让底层 line.text / line.chars 变化的入口）：
          1. rebind() —— pair 指向新 Line，旧 cell_row 是旧 Line 上 build 的
          2. HProofPanel._on_external_line_changed() —— VProof 改完文本回流
        销毁后：若本 pair 仍然 active 且 cell_mode 开着，会立刻按新 line 重建 UI；
        否则等下次 _refresh_active_widgets 时按需懒构。
        """
        if self._cell_row is None:
            return
        # 切断信号、移出 layout、释放
        try:
            self._cell_row.text_committed.disconnect()
            self._cell_row.commit_requested.disconnect()
            self._cell_row.focus_changed.disconnect()
            self._cell_row.next_off_end.disconnect()
            self._cell_row.prev_off_start.disconnect()
        except (RuntimeError, TypeError):
            pass
        self._cell_row.hide()
        self._cell_row.setParent(None)
        self._cell_row.deleteLater()
        self._cell_row = None
        self._cell_focus_idx = None
        # 若仍 active 且 cell_mode 开着，立即重建（保持用户视觉不闪 → 等同于
        # editor 模式下 _on_external_line_changed 会立即 setPlainText）。
        if self._active and self._cell_mode_enabled:
            self._refresh_active_widgets()

    def _on_cell_text_committed(self, new_text: str) -> None:
        """字格模式：每次 cell 编辑都同步到 line（不等失焦）。

        发 text_saved 让 HProofPanel._save_pair 走统一保存路径。"""
        if new_text != _displayed_text(self._line, self._page, self._block):
            self.text_saved.emit(self._idx, new_text)

    def _on_cell_focus_changed(self, idx: int) -> None:
        """cell_mode 下某 cell 拿到焦点 → 重渲染行图高亮该字。"""
        if self._cell_focus_idx == idx:
            return
        self._cell_focus_idx = idx
        self._render_line_image()

    def _img_clicked_lookup(self, event) -> None:
        """点击行图区域时：cell_mode 下反查最近的 char.bbox → 把焦点送回对应 cell。
        非 cell_mode 维持原行级激活行为。"""
        if not (self._cell_mode_enabled and self._active and self._cell_row
                and self._cell_row.has_cells and self._line.chars):
            self._on_click(event)
            return
        # 反查：把 _img_lbl 内 px 坐标 → 原图坐标 → 找最近 char
        try:
            pos = event.position()
            click_x = pos.x()
        except AttributeError:
            click_x = float(event.x())
        ox, _oy = self._line_crop_origin
        scale = self._render_scale or 1.0
        # _img_lbl 是 left-aligned 的 pixmap；click_x 对应 crop 内偏移 / scale
        orig_x = ox + click_x / scale
        # Phase 13: 先做精确命中（落在某 char.bbox 区间内）
        for i, ch in enumerate(self._line.chars):
            if ch.bbox is None:
                continue
            if ch.bbox.x <= orig_x <= ch.bbox.x2:
                self._cell_row.focus_cell(i)
                return
        # fallback：中心距离最近
        best_idx = None
        best_dist = float("inf")
        for i, ch in enumerate(self._line.chars):
            if ch.bbox is None:
                continue
            cx = (ch.bbox.x + ch.bbox.x2) / 2.0
            d = abs(cx - orig_x)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx is not None:
            self._cell_row.focus_cell(best_idx)

    def load_image(self) -> None:
        """懒加载行图像。"""
        if self._image_loaded:
            return
        self._image_loaded = True
        bb = self._line.bbox
        if bb.w <= 0 or bb.h <= 0:
            self._img_lbl.setText("—")
            return
        image = self._cache.get_page_image(self._page.display_image_path)
        if image is None:
            self._img_lbl.setText("（无图像）")
            return
        H, W = image.shape[:2]
        clamped = clamp_line_box_pixels(bb, W, H, pad_y=ROW_PAD_Y)
        if clamped is None:
            self._img_lbl.setText("（行框异常）")
            return
        x1, y1, x2, y2 = clamped
        crop = image[y1:y2, x1:x2].copy()
        self._line_crop = crop
        self._line_crop_origin = (x1, y1)
        h, w = crop.shape[:2]
        # 极小 bbox（OCR 出错时高/宽 < 5px）放大后会产生伪影/碎裂，
        # 而非真实行图。直接显示占位避免误导用户。
        if h < 5 or w < 5:
            self._img_lbl.setText("（行框异常）")
            return
        self._render_line_image()

    def _render_line_image(self) -> None:
        if self._line_crop is None:
            return
        crop = self._line_crop.copy()
        highlight_range: tuple[int, int] | None = None
        if (self._active and self._cell_mode_enabled
                and self._cell_focus_idx is not None
                and 0 <= self._cell_focus_idx < len(self._line.chars)):
            # cell_mode：高亮当前聚焦 cell 对应的 char.bbox
            highlight_range = (self._cell_focus_idx, self._cell_focus_idx + 1)
        elif not self._editor.isHidden():
            cursor = self._editor.textCursor()
            start = min(cursor.selectionStart(), cursor.selectionEnd())
            end = max(cursor.selectionStart(), cursor.selectionEnd())
            if end > start:
                highlight_range = (start, end)
            else:
                pos = cursor.position() - 1
                if 0 <= pos < len(self._line.chars):
                    highlight_range = (pos, pos + 1)
        if highlight_range is not None:
            start, end = highlight_range
            ox, oy = self._line_crop_origin
            for idx in range(start, min(end, len(self._line.chars))):
                char = self._line.chars[idx]
                if char.bbox is None:
                    continue
                x1 = max(0, char.bbox.x - ox)
                y1 = max(0, char.bbox.y - oy)
                x2 = min(crop.shape[1] - 1, char.bbox.x2 - ox)
                y2 = min(crop.shape[0] - 1, char.bbox.y2 - oy)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 128, 255), 2)
                    overlay = crop.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 128, 255), -1)
                    crop = cv2.addWeighted(overlay, 0.18, crop, 0.82, 0)
        h, w = crop.shape[:2]
        # 缩放到 IMAGE_ROW_H 高度，同时限制最大宽度（避免超宽行撑开布局）。
        # 严格保持宽高比：先按高度缩放；若超宽再按宽度缩放重算高度。
        scale = IMAGE_ROW_H / h
        new_w = max(1, int(round(w * scale)))
        MAX_LINE_W = 1200
        if new_w > MAX_LINE_W:
            scale = MAX_LINE_W / w
            new_h = max(1, int(round(h * scale)))
            crop = cv2.resize(crop, (MAX_LINE_W, new_h), interpolation=cv2.INTER_AREA)
        else:
            crop = cv2.resize(crop, (new_w, IMAGE_ROW_H), interpolation=cv2.INTER_AREA)
        self._render_scale = scale
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rh, rw = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), rw, rh, rw * 3, QImage.Format.Format_RGB888)
        self._img_lbl.setPixmap(QPixmap.fromImage(qimg))

    def refresh_text(self) -> None:
        """外部更新 line.text 后刷新显示。

        Phase 19 blocker 2：active + cell_mode 时，缓存的 _cell_row 上挂的是
        旧的 display_text（quality-probe 切换前的显示空间），必须作废重建，
        否则普通行标签已经按新 active store 更新了，但字格仍显示旧空间文本。
        _invalidate_cell_row 会在 active+cell_mode 下立刻按当前 line+display
        重建 _cell_row。"""
        if not self._active:
            self._text_lbl.setText(_displayed_text(self._line, self._page, self._block))
        elif self._cell_mode_enabled:
            self._invalidate_cell_row()
        self._refresh_status()

    def rebind(self, block: Block, line: Line, page: Page, line_in_page: int) -> None:
        """Point this UI row at the current project Line without rebuilding it."""
        self._block = block
        self._line = line
        self._page = page
        self._line_in_page = line_in_page
        self._lbl_img_hdr.setText(f"图像 {line_in_page}")
        self._lbl_txt_hdr.setText(f"文本 {line_in_page}")
        self._image_loaded = False
        self._line_crop = None
        if self._editor.isHidden():
            self._text_lbl.setText(_displayed_text(line, page, block))
        # Phase 16 blocker：rebind 后旧 _cell_row 是旧 line 上 build 的，必须作废，
        # 否则 cell mode 编辑会用旧 char 内容覆盖新 line.text。
        self._invalidate_cell_row()
        self._refresh_status()

    @property
    def line(self) -> Line:
        return self._line

    @property
    def page(self) -> Page:
        return self._page

    @property
    def block(self) -> Block:
        return self._block

    # ── 私有 ──────────────────────────────────────────────────

    def _on_click(self, event) -> None:
        self.clicked.emit(self._idx)

    def _revert(self) -> None:
        # 还原到 OCR 原始文本；如果原文本位置上有评测位，还原后仍需
        # 覆盖 fake_char 以保证评测过程不被“一键跳过”在显示上消除。
        original_true = self._line.original_text or self._line.ocr_text or self._line.text or ""
        store = qp.get_active_store()
        original_display = original_true
        if store is not None:
            idx = _resolve_block_line_index(self._page, self._block, self._line)
            if idx is not None:
                bi, li = idx
                probes = store.for_line(self._page.page_number, bi, li)
                if probes:
                    original_display = qp.apply_probes_to_display(original_true, probes)
        self._editor.blockSignals(True)
        self._editor.setPlainText(original_display)
        self._editor.blockSignals(False)

    def _highlight_low_conf(self) -> None:
        cur = QTextCursor(self._editor.document())
        cur.select(QTextCursor.SelectionType.Document)
        cur.setCharFormat(QTextCharFormat())

    def _refresh_status(self) -> None:
        status = self._line.proof_status
        color = _STATUS_COLOR.get(status, "#ccc")
        label = _STATUS_LABEL.get(status, "")
        self._status_lbl.setText(
            f"<span style='color:{color};font-size:11px;'>● {label}</span>"
        )


# ─────────────────────────────────────────────────────────────
# 横校面板主体
# ─────────────────────────────────────────────────────────────

class HProofPanel(QWidget):
    """横校面板：滚动列表 + 工具栏，对照 ui-2.jpg 设计。"""

    proof_saved = Signal()
    page_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pages: List[Page] = []
        self._items: List[Tuple[Block, Line, Page, int]] = []
        self._pairs: List[_LinePair] = []
        self._current_idx: int = 0
        self._filter_updating = False
        self._cache = PageImageCache.instance()
        self._bus = ProofStateBus.instance()
        # Phase 18 blocker 1：面板级 cell_mode 状态，merge_pages 新建 pair 时同步。
        self._cell_mode_enabled: bool = False
        # H/V 校对联动：订阅其他 panel 编辑事件；origin == id(self) 的事件忽略。
        # Phase 18 blocker 3：保留 unsubscribe 句柄，控件销毁时释放，避免长会话死订阅。
        self._bus_unsub = self._bus.subscribe(
            "line.proof_changed", self._on_external_line_changed,
        )
        self.destroyed.connect(lambda *_: self._teardown_bus())
        self._build_ui()

    def _teardown_bus(self) -> None:
        """Phase 18 blocker 3：释放 ProofStateBus 订阅。

        QObject.destroyed 信号在 Python 端仍可调用 unsubscribe；幂等，多次安全。"""
        unsub = getattr(self, "_bus_unsub", None)
        if unsub is not None:
            try:
                unsub()
            except Exception:
                pass
            self._bus_unsub = None

    # ── UI 构建 ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 工具栏 ─────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        toolbar.setFixedHeight(46)
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(12, 0, 12, 0)
        tl.setSpacing(4)

        self._btn_prev = QPushButton("↑ 上一行")
        self._btn_next = QPushButton("↓ 下一行")
        self._btn_save = QPushButton("保存  Ctrl+S")
        self._btn_save.setObjectName("primaryBtn")
        self._btn_flag = QPushButton("⚑ 标记  F5")
        self._btn_skip = QPushButton("跳过  F6")

        for btn in (self._btn_prev, self._btn_next, self._btn_save,
                    self._btn_flag, self._btn_skip):
            btn.setMinimumHeight(30)
            tl.addWidget(btn)

        # Phase 11 task 2：字格模式开关（实验，line.chars 为空的行会回退）
        from PySide6.QtWidgets import QToolButton  # local import：仅本处用
        self._btn_cell_mode = QToolButton()
        self._btn_cell_mode.setText("字格模式")
        self._btn_cell_mode.setCheckable(True)
        self._btn_cell_mode.setMinimumHeight(30)
        self._btn_cell_mode.setToolTip(
            "实验：把当前行拆成『一个字一个 cell』编辑。\n"
            "用 Tab/←/→ 在 cell 间跳；Enter 提交并跳到下一处。\n"
            "（line.chars 为空的行会显示回退提示）"
        )
        self._btn_cell_mode.toggled.connect(self._on_toggle_cell_mode)
        tl.addWidget(self._btn_cell_mode)

        # 评测/正确率统计 入口已迁移到 MainWindow『设置』菜单的「正确率统计…」。
        # 这里不再放工具栏按钮，避免和设置入口重复。
        tl.addStretch()

        self._page_combo = QComboBox()
        self._page_combo.setMinimumWidth(120)
        self._page_combo.currentIndexChanged.connect(self._on_page_filter_changed)
        tl.addWidget(self._page_combo)

        self._progress_lbl = QLabel("0 / 0")
        self._progress_lbl.setObjectName("muted")
        tl.addWidget(self._progress_lbl)

        # 统计徽章
        self._stat_lbl = QLabel("")
        self._stat_lbl.setObjectName("muted")
        self._stat_lbl.setStyleSheet("font-size:11px; color:#666; margin-left:8px;")
        tl.addWidget(self._stat_lbl)

        root.addWidget(toolbar)

        # ── 滚动列表 ───────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(0)
        self._list_layout.addStretch()

        # 空状态提示（无数据时显示）
        self._empty_lbl = QLabel("完成 OCR 识别后，横校数据将在此展示")
        self._empty_lbl.setObjectName("proofEmpty")
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setMinimumHeight(120)
        self._list_layout.insertWidget(0, self._empty_lbl)

        self._scroll.setWidget(self._list_widget)
        root.addWidget(self._scroll, 1)

        # ── 底部状态栏 ─────────────────────────────────────────
        statusbar = QWidget()
        statusbar.setFixedHeight(28)
        statusbar.setStyleSheet("background:#f5f7fb; border-top:1px solid #e3e8ef;")
        sl = QHBoxLayout(statusbar)
        sl.setContentsMargins(12, 0, 12, 0)
        sl.setSpacing(16)

        self._total_lbl = QLabel("总字数 0")
        self._diff_lbl  = QLabel("差异 0 (0%)")
        for lbl in (self._total_lbl, self._diff_lbl):
            lbl.setStyleSheet("font-size:11px; color:#666;")
            sl.addWidget(lbl)

        # 图例
        for color, label in (
            ("#4CAF50", "与原文一致"),
            ("#FF9800", "疑似错误"),
            ("#c8d0db", "待确认"),
            ("#1a73e8", "已修改"),
        ):
            dot = QLabel(
                f"<span style='color:{color}'>●</span>"
                f"<span style='color:#666;font-size:11px;'> {label}</span>"
            )
            dot.setTextFormat(Qt.TextFormat.RichText)
            sl.addWidget(dot)

        sl.addStretch()
        root.addWidget(statusbar)

        # ── 信号 ───────────────────────────────────────────────
        self._btn_prev.clicked.connect(self._prev)
        self._btn_next.clicked.connect(self._next)
        # Phase 21 blocker：工具栏"保存"按钮原本走 _save_current，
        # 在 cell mode 下 editor 隐藏 → _save_current 内部 isHidden 早退 → 全程
        # no-op，连 proof_saved 都不发。Ctrl+S 走的是 _save_all（_save_current
        # + 无条件 emit proof_saved），用户感知"保存还能用"。这里把按钮也指向
        # _save_all，保持与 Ctrl+S 完全同语义。
        # 注意：cell mode 下文本本身是按 cell 实时落盘的（textEdited → text_committed
        # → _on_cell_text_committed → text_saved → _save_displayed_edit），
        # 所以按钮的"保存"语义实际是"广播 proof_saved + 走一次现有保存路径"。
        self._btn_save.clicked.connect(self._save_all)
        self._btn_flag.clicked.connect(self._toggle_flag)
        self._btn_skip.clicked.connect(self._next)

        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_all)
        # 滚动时触发懒加载
        self._scroll.verticalScrollBar().valueChanged.connect(
            lambda _: self._load_visible_images()
        )

    # ── 公共 API ───────────────────────────────────────────────

    def load_pages(self, pages: List[Page]) -> None:
        self._pages = pages
        self._refresh_page_filter()
        self._render_pages(self._filtered_pages())

    def merge_pages(self, pages: List[Page]) -> None:
        """Merge OCR background updates without rebuilding active editors."""
        if not self._pairs:
            self.load_pages(pages)
            return
        self._pages = pages
        self._refresh_page_filter()
        loaded_keys = {
            self._line_key(block, line, page, li): index
            for index, (block, line, page, li) in enumerate(self._items)
        }
        prev_page_number = self._items[-1][2].page_number if self._items else -1
        added = False
        for page in self._filtered_pages():
            page_line_num = 1
            for block, line, li in iter_unique_page_hproof_lines(page):
                key = self._line_key(block, line, page, li)
                existing_index = loaded_keys.get(key)
                if existing_index is not None:
                    self._items[existing_index] = (block, line, page, li)
                    self._pairs[existing_index].rebind(block, line, page, page_line_num)
                    page_line_num += 1
                    continue
                if page.page_number != prev_page_number:
                    sep = QLabel(f"── 第 {page.page_number} 页 ──")
                    sep.setObjectName("pageSep")
                    sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    sep.setMinimumHeight(26)
                    self._list_layout.insertWidget(self._list_layout.count() - 1, sep)
                    prev_page_number = page.page_number
                self._append_pair(block, line, page, li, page_line_num)
                loaded_keys[key] = len(self._items) - 1
                page_line_num += 1
                added = True
        if added:
            self._empty_lbl.setVisible(False)
            self._update_stats()
            self._progress_lbl.setText(f"{self._current_idx + 1} / {len(self._pairs)}")
            QTimer.singleShot(100, self._load_visible_images)

    def _refresh_page_filter(self) -> None:
        from pathlib import Path

        current = self._page_combo.currentData()
        page_options = [
            (page.page_number, Path(page.source_path or page.image_path).name)
            for page in self._pages
            if any(True for _ in iter_unique_page_hproof_lines(page))
        ]
        page_numbers = [page_number for page_number, _name in page_options]
        self._filter_updating = True
        self._page_combo.clear()
        self._page_combo.addItem("全部页面", None)
        for page_number, name in page_options:
            self._page_combo.addItem(f"第 {page_number} 页  {name}", page_number)
        if current in page_numbers:
            index = self._page_combo.findData(current)
            self._page_combo.setCurrentIndex(index)
        self._filter_updating = False

    def _filtered_pages(self) -> List[Page]:
        page_number = self._page_combo.currentData()
        if page_number is None:
            return self._pages
        return [page for page in self._pages if page.page_number == page_number]

    def _on_page_filter_changed(self) -> None:
        if self._filter_updating:
            return
        page_number = self._page_combo.currentData()
        if page_number is not None:
            self.page_selected.emit(int(page_number))
        self._render_pages(self._filtered_pages())

    def set_current_page_number(self, page_number: int) -> None:
        index = self._page_combo.findData(page_number)
        if index >= 0 and self._page_combo.currentIndex() != index:
            self._page_combo.setCurrentIndex(index)

    def _render_pages(self, pages: List[Page]) -> None:
        self._items.clear()
        self._pairs.clear()

        # 清空旧 _LinePair。从后往前递删；skip _empty_lbl 和布局末尾的 stretch。
        # 为什么要 skip ：之前代码会 delete _empty_lbl ，导致二次 load 时
        # `self._empty_lbl.setVisible(...)` 变成访问已销毁的 C++ 对象 →
        # RuntimeError 中断后续清理 → 旧 _LinePair 仍贴在 _list_widget 上 →
        # 产生“重复页面”现象。
        for i in range(self._list_layout.count() - 1, -1, -1):
            item = self._list_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if w is None or w is self._empty_lbl:
                continue
            self._list_layout.takeAt(i)
            w.setParent(None)
            w.deleteLater()

        # 无数据时显示空状态
        has_data = any(True for page in pages for _ in iter_unique_page_hproof_lines(page))
        self._empty_lbl.setVisible(not has_data)

        prev_page_number: int = -1
        for page in pages:
            page_line_num = 1
            for block, line, li in iter_unique_page_hproof_lines(page):
                # 每页第一行前插入页面分隔条，让用户清晰知道当前所处页面
                if page.page_number != prev_page_number:
                    sep = QLabel(f"── 第 {page.page_number} 页 ──")
                    sep.setObjectName("pageSep")
                    sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    sep.setMinimumHeight(26)
                    self._list_layout.insertWidget(
                        self._list_layout.count() - 1, sep
                    )
                    prev_page_number = page.page_number
                self._append_pair(block, line, page, li, page_line_num)
                page_line_num += 1

        self._current_idx = 0
        self._update_stats()
        if self._pairs:
            self._activate(0)
            # 懒加载前 30 行图像
            QTimer.singleShot(100, self._load_visible_images)

    def _append_pair(self, block: Block, line: Line, page: Page, li: int, page_line_num: int) -> None:
        self._items.append((block, line, page, li))
        pair = _LinePair(
            len(self._pairs), block, line, page, page_line_num,
            self._cache,
        )
        pair.clicked.connect(self._on_pair_clicked)
        pair.text_saved.connect(self._on_text_saved)
        pair.confirmed.connect(self._on_confirmed)
        pair.prev_req.connect(self._prev)
        pair.next_req.connect(self._next)
        pair.flag_req.connect(self._toggle_flag)
        pair.skip_req.connect(self._next)
        self._pairs.append(pair)
        self._list_layout.insertWidget(self._list_layout.count() - 1, pair)
        # Phase 18 blocker 1：merge_pages 期间新增 pair 必须继承当前面板 cell mode，
        # 否则会出现"已开启字格模式，但新合入的页仍是普通模式"的不一致。
        if self._cell_mode_enabled:
            pair.set_cell_mode(True)

    def _line_key(self, block: Block, line: Line, page: Page, line_idx: int) -> tuple:
        bbox = line.bbox.normalize()
        return (
            page.display_image_path,
            page.source_path,
            int(page.source_page_index),
            int(page.page_number),
            block.block_type.value,
            int(block.order),
            int(line_idx),
            int(bbox.x),
            int(bbox.y),
            int(bbox.w),
            int(bbox.h),
        )

    def reset(self) -> None:
        self.load_pages([])
        self._progress_lbl.setText("0 / 0")
        self._stat_lbl.setText("")
        self._total_lbl.setText("总字数 0")
        self._diff_lbl.setText("差异 0 (0%)")

    # ── 导航 ───────────────────────────────────────────────────

    def _prev(self) -> None:
        if self._current_idx > 0:
            self._save_current(silent=True)
            self._activate(self._current_idx - 1)

    def _next(self) -> None:
        if self._current_idx < len(self._pairs) - 1:
            self._save_current(silent=True)
            self._activate(self._current_idx + 1)

    def _activate(self, idx: int) -> None:
        idx = max(0, min(idx, len(self._pairs) - 1))
        if 0 <= self._current_idx < len(self._pairs) and self._current_idx != idx:
            self._pairs[self._current_idx].set_active(False)
        self._current_idx = idx
        pair = self._pairs[idx]
        pair.set_active(True)
        # 滚动到可见
        QTimer.singleShot(30, lambda: self._scroll.ensureWidgetVisible(pair, 0, 40))
        self._progress_lbl.setText(f"{idx + 1} / {len(self._pairs)}")

    def _on_pair_clicked(self, idx: int) -> None:
        if idx != self._current_idx:
            self._save_current(silent=True)
            self._activate(idx)

    def _on_confirmed(self, idx: int) -> None:
        """Enter 键确认当前行。"""
        if not self._items:
            return
        _, line, page, _ = self._items[idx]
        # 先保存文本
        self._save_current(silent=True)
        line.proof_status = ProofStatus.OK
        self._bus.publish(
            "line.proof_changed",
            page_id=page.id,
            line_id=line.id,
            status=ProofStatus.OK.value,
            origin=id(self),
        )
        self.proof_saved.emit()
        self._pairs[idx].refresh_text()
        self._update_stats()
        self._next()

    def _on_text_saved(self, idx: int, new_text: str) -> None:
        """_LinePair 在 set_active(False) 时保存；cell mode 下每次 cell 编辑
        也走这里（_LinePair._on_cell_text_committed → text_saved）。

        Phase 22 blocker 2：保存成功后，被改动行的 proof_status 通常会从
        UNCHECKED 变 MODIFIED。必须立即刷新该 pair 的状态点 ——
        但**不能**调 pair.refresh_text()（active+cell_mode 下它会作废重建
        _cell_row → 把焦点拉回第 0 格）。直接调 _refresh_status() 只更新
        状态标签，不触碰 _cell_row / editor / 焦点。"""
        if idx >= len(self._items):
            return
        block, line, page, _ = self._items[idx]
        if _save_displayed_edit(line, page, block, new_text):
            self._bus.publish(
                "line.proof_changed",
                page_id=page.id,
                line_id=line.id,
                status=line.proof_status.value,
                origin=id(self),
            )
            self.proof_saved.emit()
            self._update_stats()
            # Phase 22 blocker 2：即时刷新 active pair 状态点
            if 0 <= idx < len(self._pairs):
                self._pairs[idx]._refresh_status()

    def _save_current(self, *, silent: bool = False) -> None:
        """将当前编辑器内容保存到 line 对象。"""
        if not self._pairs or self._current_idx >= len(self._pairs):
            return
        pair = self._pairs[self._current_idx]
        if not pair._editor.isHidden():
            new_text = pair._editor.toPlainText()
            block, line, page, _ = self._items[self._current_idx]
            if _save_displayed_edit(line, page, block, new_text):
                self._bus.publish(
                    "line.proof_changed",
                    page_id=page.id,
                    line_id=line.id,
                    status=line.proof_status.value,
                    origin=id(self),
                )
                self.proof_saved.emit()
                pair.refresh_text()
                self._update_stats()

    def _save_all(self) -> None:
        self._save_current()
        self.proof_saved.emit()

    def _toggle_flag(self) -> None:
        if not self._items or self._current_idx >= len(self._items):
            return
        _, line, page, _ = self._items[self._current_idx]
        new_status = (
            ProofStatus.UNCHECKED
            if line.proof_status == ProofStatus.AUTO_FLAGGED
            else ProofStatus.AUTO_FLAGGED
        )
        line.proof_status = new_status
        self._bus.publish(
            "line.proof_changed",
            page_id=page.id,
            line_id=line.id,
            status=new_status.value,
            origin=id(self),
        )
        self._pairs[self._current_idx].refresh_text()
        self._update_stats()

    # ── 统计 ───────────────────────────────────────────────────

    def _on_external_line_changed(self, **kwargs) -> None:
        """收到外部（纵校）发来的 line.proof_changed → 找到本 panel 中
        line.id 匹配的行，刷新该行显示并重算统计。

        回路保护：origin == id(self) 时直接跳过（自己 publish 的事件）。
        """
        if kwargs.get("origin") == id(self):
            return
        line_id = kwargs.get("line_id")
        if line_id is None:
            return
        touched = False
        for i, (block, line, page, _li) in enumerate(self._items):
            if line.id == line_id:
                pair = self._pairs[i]
                if i == self._current_idx and not pair._editor.isHidden():
                    pair._editor.blockSignals(True)
                    pair._editor.setPlainText(_displayed_text(line, page, block))
                    pair._editor.blockSignals(False)
                # Phase 16 blocker：底层 line.text 已被 VProof 改写，
                # 缓存的 _cell_row 是旧 line.text 上算出的 cell_inits / trailing_overflow，
                # 必须作废，否则 cell mode 下一次编辑会用旧内容覆盖新文本。
                # Phase 20 blocker 2：active+cell_mode 行的作废+重建由
                # refresh_text 内部统一负责（Phase 19 起即如此）；这里如果再显式
                # 调一次 _invalidate_cell_row，就会双重作废+重建，焦点被拉回第 0 格。
                # 只在非 active pair 上显式作废 —— 此时 refresh_text 不会自动作废，
                # 但用户尚未激活该行，缓存的 stale cell_row 必须先丢掉，
                # 等下次 _refresh_active_widgets 才会按新 line.text 重新构建。
                if not pair._active:
                    pair._invalidate_cell_row()
                pair.refresh_text()
                touched = True
        if touched:
            self._update_stats()

    def _update_stats(self) -> None:
        total_chars = sum(len(ln.text or "") for _, ln, _, _ in self._items)
        diff_count  = sum(
            1 for _, ln, _, _ in self._items
            if ln.proof_status in (ProofStatus.MODIFIED, ProofStatus.AUTO_FLAGGED)
        )
        pct = (diff_count / max(1, len(self._items))) * 100
        self._total_lbl.setText(f"总字数 {total_chars:,}")
        self._diff_lbl.setText(f"差异 {diff_count} ({pct:.1f}%)")


    def _on_toggle_cell_mode(self, checked: bool) -> None:
        """字格模式切换：广播到所有 _LinePair；当前行立即 re-activate
        以让可见控件按新状态切换。

        Phase 18 blocker 1：同步面板级状态，让 merge_pages 后新建 pair 自动继承。

        Phase 20 blocker 1：active pair 的 _refresh_active_widgets 由 set_cell_mode
        自身在状态变化分支里负责调用（Phase 17 起即如此），这里再补一次
        会让 active 行可见地闪一下、cell_row 还会被构建两次。删掉手动二次刷新。"""
        self._cell_mode_enabled = checked
        for pair in self._pairs:
            pair.set_cell_mode(checked)

    def refresh_quality_probe_state(self) -> None:
        """重新渲染所有可见行，使显示空间文本与全局 active store 对齐。

        Phase 11：「评测：开/关」工具栏按钮已迁移到设置→正确率统计 对话框，
        所以这里只剩刷新 pair 显示一件事。"""
        for pair in self._pairs:
            pair.refresh_text()

    # ── 懒加载图像 ─────────────────────────────────────────────

    def _load_visible_images(self) -> None:
        """加载当前视口附近行的图像。"""
        if not self._pairs:
            return
        vp = self._scroll.viewport()
        vp_top = self._scroll.verticalScrollBar().value()
        vp_bot = vp_top + vp.height()
        for pair in self._pairs:
            y = pair.y()
            h = pair.height()
            # 加载视口 ±2 屏范围内的图像
            if y + h >= vp_top - vp.height() * 2 and y <= vp_bot + vp.height() * 2:
                pair.load_image()
