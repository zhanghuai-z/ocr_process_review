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
    QScrollArea, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from app.models import Block, Line, OcrProject, Page, ProofStatus
from app.core.page_image_cache import PageImageCache
from app.ui.widgets.page_directory import PageDirectoryList
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
TEXT_FONT_PX = 18    # 30px 缩小 40%，贴近 32px 行图中线
TEXT_EDITOR_MAX_H = 32
# Phase 24：横校改为"上图下字"竖排布局：image_row(32) + text_row(32) +
# 中间 spacing(4) + 上下各 ~4 px = 76 px。
LINE_PAIR_H = 76
# Phase 17 blocker：字格模式下需要为 CharCellRow 留够竖向空间。
# CharCellRow 自身固定高 = IMG_H(36) + EDIT_H(26) + 6 内边距 = 68。
# Phase 24（上图下字后）：image_row(32) + cell_row(68) + spacing(4) +
# 上下 padding(8) = 112 px。
CELL_PAIR_H = 112
TEXT_FONT_FAMILY = "'Microsoft YaHei UI','Noto Sans CJK SC','PingFang SC','SimSun',sans-serif"
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
        # 最近一次行图像缩放比例，用于把 _img_lbl 上的点击位置反查回原图坐标
        self._render_scale: float = 1.0

        self.setObjectName("linePair")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._build_ui()

    # ── 构建 ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setFixedHeight(LINE_PAIR_H)
        # Phase 25：彻底取消字格模式 / 取消"图像/文本"hdr 标签 / 整行文本框
        # 始终可见。布局保留 Phase 24 的"上图下字"骨架：
        #   root QHBoxLayout = [active_bar | content_v(img_row, editor) | status]
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 2, 8, 2)
        root.setSpacing(4)

        # 蓝色激活条（左边框）
        self._active_bar = QWidget()
        self._active_bar.setFixedWidth(4)
        self._active_bar.setStyleSheet("background: transparent;")
        root.addWidget(self._active_bar)
        self._active_bar2 = self._active_bar

        # 中间内容容器：上图下字
        self._content = QWidget()
        content_v = QVBoxLayout(self._content)
        content_v.setContentsMargins(0, 0, 0, 0)
        content_v.setSpacing(2)

        # ── 上：行图像（去掉左侧"图像 N"hdr，直接占满宽度）──────
        self._img_lbl = QLabel()
        self._img_lbl.setFixedHeight(IMAGE_ROW_H)
        self._img_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._img_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._img_lbl.setStyleSheet("background:#fafbfc;")
        content_v.addWidget(self._img_lbl)

        # ── 下：整行文本框（永远可见；弱光标 + 等宽 + 与图像 y 对齐）──
        self._editor = _RowEditor()
        self._editor.setStyleSheet(
            # font-family 与图像下沿对齐：用等宽优先 + 紧凑行高，便于人工
            # 快速逐字确认；padding 0 让首字对齐图像左侧首字。
            f"font-family:{TEXT_FONT_FAMILY}; font-size:{TEXT_FONT_PX}px; "
            "padding:0; background:#ffffff; border:1px solid #e3e8ef;"
        )
        self._editor.setFixedHeight(TEXT_EDITOR_MAX_H)
        self._editor.document().setDocumentMargin(2)
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._editor.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        # Phase 25：弱光标 —— cursor width 0，不显示插入符；逐字定位/高亮
        # 通过 extraSelections + cursor.setPosition 体现。
        self._editor.setCursorWidth(0)
        # 初始填入显示空间文本
        self._editor.setPlainText(_displayed_text(self._line, self._page, self._block))
        # 信号转发
        self._editor.confirm_requested.connect(lambda: self.confirmed.emit(self._idx))
        self._editor.prev_requested.connect(self.prev_req)
        self._editor.next_requested.connect(self.next_req)
        self._editor.flag_requested.connect(self.flag_req)
        self._editor.skip_requested.connect(self.skip_req)
        self._editor.revert_requested.connect(self._revert)
        self._editor.selectionChanged.connect(self._refresh_extra_selections)
        self._editor.selectionChanged.connect(self._render_line_image)
        self._editor.cursorPositionChanged.connect(self._refresh_extra_selections)
        self._editor.cursorPositionChanged.connect(self._render_line_image)
        # 编辑触发置信度高亮重绘（修过的字按 OK 颜色处理）
        self._editor.textChanged.connect(self._refresh_extra_selections)
        content_v.addWidget(self._editor)

        root.addWidget(self._content, 1)

        # 状态标签
        self._status_lbl = QLabel()
        self._status_lbl.setFixedWidth(STATUS_W)
        self._status_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._status_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._refresh_status()
        root.addWidget(self._status_lbl)

        # 行级单击（active_bar / 整体）走 _on_click 激活本行；图像点击走
        # _img_clicked_lookup（仍按字 bbox 反查 + cursor 定位到该字）。
        for w in (self, self._active_bar):
            w.mousePressEvent = self._on_click  # type: ignore[method-assign]
        self._img_lbl.mousePressEvent = self._img_clicked_lookup  # type: ignore[method-assign]

        # 首次绘制置信度底色
        self._refresh_extra_selections()

    # ── 对外接口 ──────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        blue = "#1a73e8"
        if active:
            bar_style = f"background:{blue}; border-radius:2px;"
            bg = "#f0f6ff"
            self._editor.setFocus()
        else:
            # 切走前先把 in-flight 文本保存（编辑器始终可见）
            self._flush_editor_if_dirty()
            bar_style = "background:transparent;"
            bg = "transparent"

        self._active_bar.setStyleSheet(bar_style)
        self._active_bar2.setStyleSheet(bar_style)
        self.setStyleSheet(
            f"QFrame#linePair {{ background:{bg}; }}"
            if active else ""
        )
        self._refresh_status()
        self._refresh_extra_selections()
        self._render_line_image()

    def _flush_editor_if_dirty(self) -> None:
        """若 editor 当前文本与显示空间文本不一致，发 text_saved 让面板落盘。
        Phase 25：editor 始终可见，不再判 isHidden。"""
        new_text = self._editor.toPlainText()
        if new_text != _displayed_text(self._line, self._page, self._block):
            self.text_saved.emit(self._idx, new_text)

    # ── Phase 25：弱光标 + 逐字高亮（取代字格模式）──────────────

    def _refresh_extra_selections(self) -> None:
        """根据 line.chars 的置信度 + 当前光标位置生成 extraSelections：
        - 低置信度字符：浅红/橙底色
        - 当前光标所在/选中字符：蓝色边框（用 outline 风格的 background）

        弱光标即"看不到 caret，但有当前字格高亮"。
        """
        try:
            from PySide6.QtWidgets import QTextEdit
        except Exception:
            return
        editor = self._editor
        doc_text = editor.toPlainText()
        sels: list = []

        # 1) 置信度底色（按 line.chars 一一映射；超出/缺失按 OK 处理）
        chars = self._line.chars or []
        n = min(len(doc_text), len(chars))
        for i in range(n):
            conf = getattr(chars[i], "confidence", 1.0) or 1.0
            if conf >= LOW_CONF:
                continue
            sel = QTextEdit.ExtraSelection()
            cur = QTextCursor(editor.document())
            cur.setPosition(i)
            cur.setPosition(i + 1, QTextCursor.MoveMode.KeepAnchor)
            fmt = QTextCharFormat()
            # 红/橙阶梯
            if conf < 0.5:
                fmt.setBackground(QColor("#fde0e0"))
            else:
                fmt.setBackground(QColor("#fff1d6"))
            sel.format = fmt
            sel.cursor = cur
            sels.append(sel)

        # 2) 当前字（光标所在位置或 selection 范围）边框/底色
        cursor = editor.textCursor()
        if cursor.hasSelection():
            start, end = cursor.selectionStart(), cursor.selectionEnd()
        else:
            pos = cursor.position()
            start, end = pos, min(pos + 1, len(doc_text))
        if end > start and self._active:
            sel = QTextEdit.ExtraSelection()
            cur = QTextCursor(editor.document())
            cur.setPosition(start)
            cur.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            fmt = QTextCharFormat()
            fmt.setBackground(QColor("#cfe2ff"))   # 蓝色高亮 = 当前字
            sel.format = fmt
            sel.cursor = cur
            sels.append(sel)

        editor.setExtraSelections(sels)

    def _img_clicked_lookup(self, event) -> None:
        """点击行图区域：先按 char.bbox 反查最近的字 → 把 editor 光标定到该字。
        若无 chars 信息则退回行级激活。"""
        if not self._line.chars:
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
        orig_x = ox + click_x / scale
        target_idx = None
        for i, ch in enumerate(self._line.chars):
            if ch.bbox is None:
                continue
            if ch.bbox.x <= orig_x <= ch.bbox.x2:
                target_idx = i
                break
        if target_idx is None:
            best_dist = float("inf")
            for i, ch in enumerate(self._line.chars):
                if ch.bbox is None:
                    continue
                cx = (ch.bbox.x + ch.bbox.x2) / 2.0
                d = abs(cx - orig_x)
                if d < best_dist:
                    best_dist = d
                    target_idx = i
        if target_idx is None:
            self._on_click(event)
            return
        # 先激活本行
        if not self._active:
            # 复用 _on_click 的 emit 路径
            self._on_click(event)
        # 把 editor 光标定到该字
        from PySide6.QtGui import QTextCursor
        cur = self._editor.textCursor()
        cur.setPosition(target_idx)
        cur.setPosition(target_idx + 1, QTextCursor.MoveMode.KeepAnchor)
        self._editor.setTextCursor(cur)
        self._editor.setFocus()
        self._refresh_extra_selections()

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
        # Phase 25：editor 始终可见，按光标/选区在行图上高亮对应 char.bbox。
        if self._active:
            cursor = self._editor.textCursor()
            start = min(cursor.selectionStart(), cursor.selectionEnd())
            end = max(cursor.selectionStart(), cursor.selectionEnd())
            if end > start:
                highlight_range = (start, end)
            else:
                pos = cursor.position()
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
        """外部（VProof / probe 切换）更新 line.text 后同步 editor 文本。

        Phase 25：editor 始终可见 → 直接 blockSignals + setPlainText 重写当前
        文本，避免触发 dirty flush。active 行上若用户正在编辑，会被覆盖
        （这是与 V 同步的既有行为；实时 dirty 已通过 _flush_editor_if_dirty
        在 set_active 切走前落盘）。
        """
        new_disp = _displayed_text(self._line, self._page, self._block)
        if self._editor.toPlainText() != new_disp:
            self._editor.blockSignals(True)
            self._editor.setPlainText(new_disp)
            self._editor.blockSignals(False)
        self._refresh_status()
        self._refresh_extra_selections()

    def rebind(self, block: Block, line: Line, page: Page, line_in_page: int) -> None:
        """Point this UI row at the current project Line without rebuilding it."""
        self._block = block
        self._line = line
        self._page = page
        self._line_in_page = line_in_page
        self._image_loaded = False
        self._line_crop = None
        # Phase 25：editor 始终可见。rebind 不触碰 editor 内容，
        # 与 Phase 24 之前 "isHidden() 才同步" 的行为等价 —— 这样用户在原行上
        # 未提交的编辑（dirty 文本）不会被 merge_pages 路径上的 rebind 覆盖。
        # 真正的"显示空间已变"由 refresh_text() 单独负责。
        self._refresh_status()
        self._refresh_extra_selections()

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
        # Phase 25：低置信度高亮已经通过 _refresh_extra_selections 实现
        # （per-char ExtraSelection 背景色），这里保留方法仅为兼容旧调用点。
        self._refresh_extra_selections()

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
        self._selected_page_number: int | None = None  # Phase 25：左侧目录唯一过滤源
        self._cache = PageImageCache.instance()
        self._bus = ProofStateBus.instance()
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
        # Phase 24：横校改为三栏布局：
        #   左：复用版面分析的页面目录（PageDirectoryList）
        #   中：原滚动列表（_LinePair 已改为上图下字）
        #   右：工具栏 + 快捷键说明（用户明确要求"放进界面"）
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("hproofSplitter")

        # ── 左：页面目录 ────────────────────────────────────────
        self._page_dir = PageDirectoryList()
        self._page_dir.page_selected.connect(self._on_page_dir_selected)
        splitter.addWidget(self._page_dir)

        # ── 中：滚动列表 ────────────────────────────────────────
        center = QWidget()
        center_v = QVBoxLayout(center)
        center_v.setContentsMargins(0, 0, 0, 0)
        center_v.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
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
        center_v.addWidget(self._scroll, 1)
        splitter.addWidget(center)

        # ── 右：工具栏（垂直）+ 快捷键说明 ──────────────────────
        right = QWidget()
        right.setObjectName("hproofRightDock")
        right.setMinimumWidth(180)
        right.setMaximumWidth(260)
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(8, 8, 8, 8)
        right_v.setSpacing(8)

        # 操作按钮组（Phase 25：右栏精简，只剩保存 / 标记 / 跳过）
        actions_lbl = QLabel("操作")
        actions_lbl.setStyleSheet("font-weight:600; color:#444; font-size:12px;")
        right_v.addWidget(actions_lbl)

        self._btn_save = QPushButton("保存")
        self._btn_save.setObjectName("primaryBtn")
        self._btn_save.setToolTip("保存所有修改  (Ctrl+S)")
        self._btn_flag = QPushButton("⚑ 标记  F5")
        self._btn_skip = QPushButton("跳过  F6")
        for btn in (self._btn_save, self._btn_flag, self._btn_skip):
            btn.setMinimumHeight(30)
            right_v.addWidget(btn)

        # 统计徽章（保留：用于全局进度小字提示）
        self._stat_lbl = QLabel("")
        self._stat_lbl.setObjectName("muted")
        self._stat_lbl.setStyleSheet("font-size:11px; color:#666;")
        right_v.addWidget(self._stat_lbl)

        right_v.addStretch()
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([180, 9999, 180])
        root.addWidget(splitter, 1)

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
            QTimer.singleShot(100, self._load_visible_images)

    def _refresh_page_filter(self) -> None:
        """Phase 25：仅维护左侧 PageDirectoryList 的页面项；不再有 combo。"""
        from pathlib import Path

        usable_pages = [
            page for page in self._pages
            if any(True for _ in iter_unique_page_hproof_lines(page))
        ]
        page_numbers = [p.page_number for p in usable_pages]
        self._filter_updating = True
        if hasattr(self, "_page_dir"):
            self._page_dir.set_pages(usable_pages)
            if self._selected_page_number in page_numbers:
                self._page_dir.set_current_index(
                    page_numbers.index(self._selected_page_number)
                )
        # 若选中页消失（被移除），回退到"全部页面"
        if self._selected_page_number is not None and                 self._selected_page_number not in page_numbers:
            self._selected_page_number = None
        self._filter_updating = False

    def _filtered_pages(self) -> List[Page]:
        if self._selected_page_number is None:
            return self._pages
        return [p for p in self._pages if p.page_number == self._selected_page_number]

    def set_current_page_number(self, page_number: int) -> None:
        """外部联动调用：把过滤器切换到指定页。"""
        self._selected_page_number = int(page_number)
        # 同步左侧目录视觉
        if hasattr(self, "_page_dir"):
            usable_numbers = [
                p.page_number for p in self._pages
                if any(True for _ in iter_unique_page_hproof_lines(p))
            ]
            if page_number in usable_numbers:
                self._page_dir.set_current_index(usable_numbers.index(page_number))
        self._render_pages(self._filtered_pages())

    def _on_page_dir_selected(self, dir_idx: int) -> None:
        """Phase 25：左栏页面目录是页面过滤的唯一入口。点击 → 切换过滤 + 重渲染。"""
        if self._filter_updating:
            return
        usable_pages = [
            page for page in self._pages
            if any(True for _ in iter_unique_page_hproof_lines(page))
        ]
        if not (0 <= dir_idx < len(usable_pages)):
            return
        target = usable_pages[dir_idx]
        if self._selected_page_number == target.page_number:
            return
        self._selected_page_number = target.page_number
        self.page_selected.emit(int(target.page_number))
        self._render_pages(self._filtered_pages())

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
        # Phase 25：editor 始终可见，直接读其当前文本与 line 比较保存
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
                # Phase 25：editor 始终可见 → 直接走 refresh_text 同步显示文本。
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
