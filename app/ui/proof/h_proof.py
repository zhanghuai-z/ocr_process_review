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
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from app.models import Block, Line, Page, ProofStatus
from app.core.page_image_cache import PageImageCache
from app.core.proof_state_bus import ProofStateBus
from app.ui.widgets.confidence_badge import ConfidenceBadge

# ── 样式常量 ──────────────────────────────────────────────────
ROW_PAD_Y    = 4     # 裁图上下各加 4px
IMAGE_ROW_H  = 52    # 行图像显示高度（px）
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

        self.setObjectName("linePair")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._build_ui()

    # ── 构建 ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # === 图像行 ===
        img_row = QWidget()
        img_row.setFixedHeight(IMAGE_ROW_H + 8)
        il = QHBoxLayout(img_row)
        il.setContentsMargins(0, 2, 8, 2)
        il.setSpacing(0)

        # 蓝色激活条（左边框）
        self._active_bar = QWidget()
        self._active_bar.setFixedWidth(4)
        self._active_bar.setStyleSheet("background: transparent;")
        il.addWidget(self._active_bar)

        lbl_img_hdr = QLabel(f"图像行 {self._line_in_page}")
        lbl_img_hdr.setFixedWidth(LABEL_W)
        lbl_img_hdr.setObjectName("muted")
        lbl_img_hdr.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        lbl_img_hdr.setStyleSheet("font-size:11px; color:#999; padding-right:8px;")
        il.addWidget(lbl_img_hdr)

        self._img_lbl = QLabel()
        self._img_lbl.setFixedHeight(IMAGE_ROW_H)
        self._img_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._img_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._img_lbl.setStyleSheet("background:#fafbfc; padding:2px 0;")
        il.addWidget(self._img_lbl, 1)

        root.addWidget(img_row)

        # === 识别文本行 ===
        txt_row = QWidget()
        tr = QHBoxLayout(txt_row)
        tr.setContentsMargins(0, 2, 8, 4)
        tr.setSpacing(0)

        # 占位（左边框相同宽度）
        self._active_bar2 = QWidget()
        self._active_bar2.setFixedWidth(4)
        self._active_bar2.setStyleSheet("background: transparent;")
        tr.addWidget(self._active_bar2)

        lbl_txt_hdr = QLabel(f"识别文本 {self._line_in_page}")
        lbl_txt_hdr.setFixedWidth(LABEL_W)
        lbl_txt_hdr.setObjectName("muted")
        lbl_txt_hdr.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        lbl_txt_hdr.setStyleSheet("font-size:11px; color:#999; padding-right:8px;")
        tr.addWidget(lbl_txt_hdr)

        # 文本展示（非激活）
        self._text_lbl = QLabel(self._line.text or "")
        self._text_lbl.setStyleSheet("font-size:15px; padding:2px 0; color:#222;")
        self._text_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._text_lbl.setWordWrap(False)
        self._text_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        tr.addWidget(self._text_lbl, 1)

        # 文本编辑器（激活时可见）
        self._editor = _RowEditor()
        self._editor.setStyleSheet("font-size:15px; padding:2px 6px;")
        self._editor.setMaximumHeight(46)
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
        tr.addWidget(self._editor, 1)

        # 状态标签
        self._status_lbl = QLabel()
        self._status_lbl.setFixedWidth(STATUS_W)
        self._status_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._status_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._refresh_status()
        tr.addWidget(self._status_lbl)

        root.addWidget(txt_row)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#edf0f5; margin:0; padding:0;")
        root.addWidget(sep)

        # 注册点击区域
        for w in (self, img_row, txt_row, self._img_lbl, self._text_lbl):
            w.mousePressEvent = self._on_click  # type: ignore[method-assign]

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
            if self._editor.isVisible():
                new_text = self._editor.toPlainText()
                if new_text != self._line.text:
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
            self._editor.setPlainText(self._line.text or "")
            self._highlight_low_conf()
            self._editor.show()
            self._editor.setFocus()
            # 确保图像已加载
            self.load_image()
        else:
            self._editor.hide()
            self._text_lbl.setText(self._line.text or "")
            self._text_lbl.show()

        self._refresh_status()

    def load_image(self) -> None:
        """懒加载行图像。"""
        if self._image_loaded:
            return
        self._image_loaded = True
        bb = self._line.bbox
        if bb.w <= 0 or bb.h <= 0:
            self._img_lbl.setText("—")
            return
        # 使用 PageImageCache.get_line_crop 获取行切图（已处理越界情况）
        crop = self._cache.get_line_crop(
            self._page.display_image_path, bb, pad_y=ROW_PAD_Y
        )
        if crop is None:
            self._img_lbl.setText("（无图像）")
            return
        h, w = crop.shape[:2]
        if h <= 0:
            self._img_lbl.setText("—")
            return
        # 缩放到 IMAGE_ROW_H 高度，同时限制最大宽度（避免超宽行撑开布局）
        scale = IMAGE_ROW_H / h
        new_w = max(1, int(w * scale))
        MAX_LINE_W = 1200
        if new_w > MAX_LINE_W:
            scale = MAX_LINE_W / w
            new_h = max(1, int(h * scale))
            crop = cv2.resize(crop, (MAX_LINE_W, new_h), interpolation=cv2.INTER_AREA)
        else:
            crop = cv2.resize(crop, (new_w, IMAGE_ROW_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rh, rw = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), rw, rh, rw * 3, QImage.Format.Format_RGB888)
        self._img_lbl.setPixmap(QPixmap.fromImage(qimg))

    def refresh_text(self) -> None:
        """外部更新 line.text 后刷新显示。"""
        if not self._active:
            self._text_lbl.setText(self._line.text or "")
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
        original = self._line.original_text or self._line.ocr_text or self._line.text or ""
        self._editor.blockSignals(True)
        self._editor.setPlainText(original)
        self._editor.blockSignals(False)

    def _highlight_low_conf(self) -> None:
        if not self._line.chars:
            return
        red = QTextCharFormat()
        red.setBackground(QColor(255, 140, 0, 70))
        normal = QTextCharFormat()
        text = self._line.text or ""
        for i, ch in enumerate(self._line.chars):
            if i >= len(text):
                break
            cur = self._editor.textCursor()
            cur.setPosition(i)
            cur.setPosition(i + 1, QTextCursor.MoveMode.KeepAnchor)
            cur.setCharFormat(red if ch.confidence < LOW_CONF else normal)

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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._items: List[Tuple[Block, Line, Page, int]] = []
        self._pairs: List[_LinePair] = []
        self._current_idx: int = 0
        self._cache = PageImageCache.instance()
        self._bus = ProofStateBus.instance()
        self._build_ui()

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

        tl.addStretch()

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
        self._btn_save.clicked.connect(self._save_current)
        self._btn_flag.clicked.connect(self._toggle_flag)
        self._btn_skip.clicked.connect(self._next)

        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_all)

    # ── 公共 API ───────────────────────────────────────────────

    def load_pages(self, pages: List[Page]) -> None:
        self._items.clear()
        self._pairs.clear()

        # 清空旧控件
        while self._list_layout.count() > 1:  # keep the stretch at end
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        line_num = 1  # 全局行号
        for page in pages:
            for block in page.text_blocks:
                for li, line in enumerate(block.lines):
                    self._items.append((block, line, page, li))
                    pair = _LinePair(
                        len(self._pairs), block, line, page, line_num,
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
                    self._list_layout.insertWidget(
                        self._list_layout.count() - 1, pair
                    )
                    line_num += 1

        self._current_idx = 0
        self._update_stats()
        if self._pairs:
            self._activate(0)
            # 懒加载前 30 行图像
            QTimer.singleShot(100, self._load_visible_images)

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
        )
        self.proof_saved.emit()
        self._pairs[idx].refresh_text()
        self._update_stats()
        self._next()

    def _on_text_saved(self, idx: int, new_text: str) -> None:
        """_LinePair 在 set_active(False) 时保存。"""
        if idx >= len(self._items):
            return
        _, line, page, _ = self._items[idx]
        if new_text != line.text:
            line.update_text(new_text)
            self._bus.publish(
                "line.proof_changed",
                page_id=page.id,
                line_id=line.id,
                status=line.proof_status.value,
            )
            self.proof_saved.emit()
            self._update_stats()

    def _save_current(self, *, silent: bool = False) -> None:
        """将当前编辑器内容保存到 line 对象。"""
        if not self._pairs or self._current_idx >= len(self._pairs):
            return
        pair = self._pairs[self._current_idx]
        if pair._editor.isVisible():
            new_text = pair._editor.toPlainText()
            _, line, page, _ = self._items[self._current_idx]
            if new_text != line.text:
                line.update_text(new_text)
                self._bus.publish(
                    "line.proof_changed",
                    page_id=page.id,
                    line_id=line.id,
                    status=line.proof_status.value,
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
        )
        self._pairs[self._current_idx].refresh_text()
        self._update_stats()

    # ── 统计 ───────────────────────────────────────────────────

    def _update_stats(self) -> None:
        total_chars = sum(len(ln.text or "") for _, ln, _, _ in self._items)
        diff_count  = sum(
            1 for _, ln, _, _ in self._items
            if ln.proof_status in (ProofStatus.MODIFIED, ProofStatus.AUTO_FLAGGED)
        )
        pct = (diff_count / max(1, len(self._items))) * 100
        self._total_lbl.setText(f"总字数 {total_chars:,}")
        self._diff_lbl.setText(f"差异 {diff_count} ({pct:.1f}%)")

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
