"""纵校面板：四象限。

布局：
  ┌─────────────┬─────────────────┐
  │ TL: 当前页   │ TR: 同字图像集合 │
  │   字符列表    │   （横跨多页）    │
  ├─────────────┼─────────────────┤
  │ BL: 当前页   │ BR: 当前页原图   │
  │   可编辑文本  │   含版面框       │
  └─────────────┴─────────────────┘

交互：
- TL 单击字 → BL 中高亮该字 + BR 中绘制 bbox 高亮 + TR 重建同字图集
- BR 点击 block → BL 滚动到该 block 起始
- BL 编辑文字 → 自动保存到对应 line（按行匹配）
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import (
    QColor, QImage, QPixmap, QTextCharFormat, QTextCursor, QIcon,
)
from PySide6.QtWidgets import (
    QGraphicsRectItem, QHBoxLayout, QLabel, QListView, QListWidget,
    QListWidgetItem, QPlainTextEdit, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

from app.models import BBox, Block, Char, Line, Page, ProofStatus
from app.ui.widgets.image_viewer import ImageViewer
from app.ui.widgets.confidence_badge import ConfidenceBadge

CHAR_THUMB_SIZE = 64
GALLERY_THUMB_SIZE = 56
LOW_CONF_THRESHOLD = 0.80


def _bbox_for_char(line: Line, idx: int) -> Optional[BBox]:
    """获取字符在行中的 bbox。若有 line.chars 则用 char.bbox，否则按行宽均分。"""
    if 0 <= idx < len(line.chars) and line.chars[idx].bbox is not None:
        return line.chars[idx].bbox
    # 回退：等分行宽
    n = max(len(line.text), 1)
    if n == 0 or idx < 0 or idx >= n:
        return None
    bb = line.bbox
    cw = max(bb.w / n, 1)
    return BBox(int(bb.x + idx * cw), bb.y, int(cw), bb.h)


def _crop_bgr(img: np.ndarray, bbox: BBox, pad: int = 2) -> np.ndarray:
    H, W = img.shape[:2]
    x1 = max(0, bbox.x - pad)
    y1 = max(0, bbox.y - pad)
    x2 = min(W, bbox.x + bbox.w + pad)
    y2 = min(H, bbox.y + bbox.h + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return img[y1:y2, x1:x2].copy()


def _bgr_to_qpixmap(bgr: np.ndarray, size: int) -> QPixmap:
    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        return QPixmap(size, size)
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(
        size, size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class VProofPanel(QWidget):
    """纵校面板：四象限协同。"""

    proof_saved = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: List[Page] = []
        self._current_page_idx: int = 0
        # 当前页拍平的字符列表：[(char_str, line_obj, idx_in_line, abs_text_offset)]
        self._page_chars: List[Tuple[str, Line, int, int]] = []
        # 当前页 cv2 image 缓存
        self._page_img: Optional[np.ndarray] = None
        # BR 上当前的高亮 item
        self._highlight_item: Optional[QGraphicsRectItem] = None
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # 标题行
        top = QHBoxLayout()
        top.setContentsMargins(12, 8, 12, 4)
        title = QLabel("⑤ 纵校")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        top.addWidget(title)

        self._page_label = QLabel("页 0 / 0")
        self._page_label.setStyleSheet("color:#aaa; margin-left:12px;")
        top.addWidget(self._page_label)

        self._btn_prev_page = QPushButton("← 上一页")
        self._btn_prev_page.clicked.connect(self._prev_page)
        self._btn_next_page = QPushButton("下一页 →")
        self._btn_next_page.clicked.connect(self._next_page)
        top.addWidget(self._btn_prev_page)
        top.addWidget(self._btn_next_page)

        top.addStretch()
        self._conf_badge = ConfidenceBadge(1.0)
        top.addWidget(self._conf_badge)
        root.addLayout(top)

        # 主体 4 象限
        outer = QSplitter(Qt.Orientation.Horizontal)
        left_split = QSplitter(Qt.Orientation.Vertical)
        right_split = QSplitter(Qt.Orientation.Vertical)

        # --- TL: 当前页字符列表 ---
        tl_box = QWidget()
        tl_layout = QVBoxLayout(tl_box)
        tl_layout.setContentsMargins(4, 4, 4, 4)
        tl_layout.addWidget(QLabel("当前页字符（点击定位）："))
        self._char_list = QListWidget()
        self._char_list.setViewMode(QListView.ViewMode.IconMode)
        self._char_list.setIconSize(QSize(CHAR_THUMB_SIZE, CHAR_THUMB_SIZE))
        self._char_list.setResizeMode(QListView.ResizeMode.Adjust)
        self._char_list.setMovement(QListView.Movement.Static)
        self._char_list.setSpacing(4)
        self._char_list.setUniformItemSizes(True)
        self._char_list.itemClicked.connect(self._on_char_clicked)
        tl_layout.addWidget(self._char_list)
        left_split.addWidget(tl_box)

        # --- BL: 当前页可编辑文本 ---
        bl_box = QWidget()
        bl_layout = QVBoxLayout(bl_box)
        bl_layout.setContentsMargins(4, 4, 4, 4)
        bl_header = QHBoxLayout()
        bl_header.addWidget(QLabel("当前页 OCR 文本（直接编辑）："))
        bl_header.addStretch()
        self._btn_save = QPushButton("✎ 保存修改")
        self._btn_save.clicked.connect(self._save_current_page_text)
        self._btn_ok = QPushButton("✓ 确认本页")
        self._btn_ok.clicked.connect(self._mark_page_ok)
        bl_header.addWidget(self._btn_save)
        bl_header.addWidget(self._btn_ok)
        bl_layout.addLayout(bl_header)
        self._text_edit = QPlainTextEdit()
        self._text_edit.setStyleSheet("font-size:16px;")
        bl_layout.addWidget(self._text_edit)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#888; font-size:12px;")
        bl_layout.addWidget(self._status_lbl)
        left_split.addWidget(bl_box)
        left_split.setStretchFactor(0, 1)
        left_split.setStretchFactor(1, 1)

        # --- TR: 同字图像集合 ---
        tr_box = QWidget()
        tr_layout = QVBoxLayout(tr_box)
        tr_layout.setContentsMargins(4, 4, 4, 4)
        self._gallery_label = QLabel("同字图像（请先在左上选择一个字）")
        self._gallery_label.setStyleSheet("color:#aaa;")
        tr_layout.addWidget(self._gallery_label)
        self._gallery_list = QListWidget()
        self._gallery_list.setViewMode(QListView.ViewMode.IconMode)
        self._gallery_list.setIconSize(QSize(GALLERY_THUMB_SIZE, GALLERY_THUMB_SIZE))
        self._gallery_list.setResizeMode(QListView.ResizeMode.Adjust)
        self._gallery_list.setMovement(QListView.Movement.Static)
        self._gallery_list.setSpacing(4)
        self._gallery_list.itemClicked.connect(self._on_gallery_clicked)
        tr_layout.addWidget(self._gallery_list)
        right_split.addWidget(tr_box)

        # --- BR: 原图 + 版面框 ---
        br_box = QWidget()
        br_layout = QVBoxLayout(br_box)
        br_layout.setContentsMargins(4, 4, 4, 4)
        br_layout.addWidget(QLabel("原图 + 版面框："))
        self._viewer = ImageViewer()
        self._viewer.block_clicked.connect(self._on_block_clicked)
        br_layout.addWidget(self._viewer)
        right_split.addWidget(br_box)
        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 2)

        outer.addWidget(left_split)
        outer.addWidget(right_split)
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 1)
        root.addWidget(outer)

    # ------------------------------------------------------------------ public

    def load_pages(self, pages: List[Page]) -> None:
        self._pages = pages
        self._current_page_idx = 0
        self._show_page(0)

    # ------------------------------------------------------------------ helpers

    def _show_page(self, idx: int) -> None:
        if not self._pages:
            self._char_list.clear()
            self._gallery_list.clear()
            self._text_edit.clear()
            self._page_label.setText("页 0 / 0")
            return
        idx = max(0, min(idx, len(self._pages) - 1))
        self._current_page_idx = idx
        page = self._pages[idx]
        self._page_label.setText(f"页 {idx+1} / {len(self._pages)}")

        # 加载图像
        self._page_img = cv2.imread(page.display_image_path)

        # BR：图像 + 版面框
        self._viewer.set_image(page.display_image_path)
        self._viewer.show_blocks(page.blocks)
        self._highlight_item = None

        # BL：拼接所有 block 的全文
        self._rebuild_text_edit(page)

        # TL：拍平字符
        self._rebuild_char_list(page)

        # 整页平均置信度
        confs = [
            ln.confidence
            for blk in page.text_blocks
            for ln in blk.lines
            if ln.text
        ]
        if confs:
            self._conf_badge.set_score(sum(confs) / len(confs))
        else:
            self._conf_badge.set_score(1.0)

    def _rebuild_text_edit(self, page: Page) -> None:
        """把全页所有可识别 block 的 line.text 用换行拼起来，
        并记录每个 line 在 plain text 中的起始 offset。"""
        self._text_line_offsets: List[Tuple[Line, int, int]] = []  # (line, start, end)
        parts: List[str] = []
        offset = 0
        for blk in page.text_blocks:
            for ln in blk.lines:
                txt = ln.text
                self._text_line_offsets.append((ln, offset, offset + len(txt)))
                parts.append(txt)
                offset += len(txt) + 1  # +1 换行
        self._text_edit.blockSignals(True)
        self._text_edit.setPlainText("\n".join(parts))
        self._text_edit.blockSignals(False)

    def _rebuild_char_list(self, page: Page) -> None:
        self._char_list.clear()
        self._page_chars = []
        if self._page_img is None:
            return
        for blk in page.text_blocks:
            for ln in blk.lines:
                # 行起始 offset
                start_off = next(
                    (s for (l, s, e) in self._text_line_offsets if l is ln), 0,
                )
                for ci, ch_str in enumerate(ln.text):
                    bbox = _bbox_for_char(ln, ci)
                    if bbox is None or bbox.area <= 0:
                        continue
                    crop = _crop_bgr(self._page_img, bbox, pad=3)
                    pix = _bgr_to_qpixmap(crop, CHAR_THUMB_SIZE)
                    item = QListWidgetItem(QIcon(pix), ch_str)
                    # 低置信度字符标红
                    conf = (
                        ln.chars[ci].confidence
                        if ci < len(ln.chars) else ln.confidence
                    )
                    if conf < LOW_CONF_THRESHOLD:
                        item.setBackground(QColor(255, 87, 34, 80))
                    item.setData(Qt.ItemDataRole.UserRole, len(self._page_chars))
                    self._char_list.addItem(item)
                    self._page_chars.append(
                        (ch_str, ln, ci, start_off + ci),
                    )

    # ---- 交互 ----

    def _on_char_clicked(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None or idx >= len(self._page_chars):
            return
        ch_str, line, ci, abs_off = self._page_chars[idx]
        # BL：选中该字
        cursor = self._text_edit.textCursor()
        cursor.setPosition(abs_off)
        cursor.setPosition(abs_off + 1, QTextCursor.MoveMode.KeepAnchor)
        self._text_edit.setTextCursor(cursor)
        self._text_edit.setFocus()
        # BR：高亮 bbox
        bbox = _bbox_for_char(line, ci)
        if bbox is not None and bbox.area > 0:
            if self._highlight_item is not None:
                try:
                    self._viewer.scene().removeItem(self._highlight_item)
                except RuntimeError:
                    pass
            self._highlight_item = self._viewer.show_line_highlight(bbox, flagged=True)
            self._viewer.centerOn(bbox.x + bbox.w / 2, bbox.y + bbox.h / 2)
        # TR：重建同字图集
        self._rebuild_gallery(ch_str)

    def _rebuild_gallery(self, ch_str: str) -> None:
        self._gallery_list.clear()
        self._gallery_label.setText(f"同字图像：「{ch_str}」")
        count = 0
        for p_idx, page in enumerate(self._pages):
            img = cv2.imread(page.display_image_path)
            if img is None:
                continue
            for blk in page.text_blocks:
                for ln in blk.lines:
                    for ci, c in enumerate(ln.text):
                        if c != ch_str:
                            continue
                        bbox = _bbox_for_char(ln, ci)
                        if bbox is None or bbox.area <= 0:
                            continue
                        crop = _crop_bgr(img, bbox, pad=3)
                        pix = _bgr_to_qpixmap(crop, GALLERY_THUMB_SIZE)
                        item = QListWidgetItem(QIcon(pix), f"P{p_idx+1}")
                        item.setData(
                            Qt.ItemDataRole.UserRole, (p_idx, ln, ci),
                        )
                        self._gallery_list.addItem(item)
                        count += 1
                        if count >= 200:
                            self._gallery_label.setText(
                                f"同字图像：「{ch_str}」（已限制为前 200 个）"
                            )
                            return

    def _on_gallery_clicked(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        p_idx, line, ci = data
        if p_idx != self._current_page_idx:
            self._current_page_idx = p_idx
            self._show_page(p_idx)
        # 跳到该字
        for k, (_ch, ln, _ci, _off) in enumerate(self._page_chars):
            if ln is line and _ci == ci:
                # 模拟点击
                it = self._char_list.item(k)
                if it:
                    self._char_list.setCurrentItem(it)
                    self._on_char_clicked(it)
                return

    def _on_block_clicked(self, block: Block) -> None:
        # BL 滚动到该 block 第一行
        if not block.lines:
            return
        first_line = block.lines[0]
        for ln, start, end in self._text_line_offsets:
            if ln is first_line:
                cursor = self._text_edit.textCursor()
                cursor.setPosition(start)
                self._text_edit.setTextCursor(cursor)
                self._text_edit.ensureCursorVisible()
                break

    # ---- 保存 / 标记 ----

    def _save_current_page_text(self) -> None:
        """把 BL 编辑后的文本按行写回 line.text。"""
        if not self._pages:
            return
        new_lines = self._text_edit.toPlainText().split("\n")
        page = self._pages[self._current_page_idx]
        i = 0
        changed = False
        for blk in page.text_blocks:
            for ln in blk.lines:
                new_text = new_lines[i] if i < len(new_lines) else ""
                if new_text != ln.text:
                    ln.update_text(new_text)
                    changed = True
                i += 1
        # 更新左上字符列表
        self._rebuild_char_list(page)
        if changed:
            self.proof_saved.emit()
            self._status_lbl.setText("✎ 已保存")
        else:
            self._status_lbl.setText("（无改动）")

    def _mark_page_ok(self) -> None:
        if not self._pages:
            return
        # 先保存
        self._save_current_page_text()
        page = self._pages[self._current_page_idx]
        for blk in page.text_blocks:
            for ln in blk.lines:
                ln.proof_status = ProofStatus.OK
        self.proof_saved.emit()
        self._status_lbl.setText("✓ 本页已确认")

    def _prev_page(self) -> None:
        self._show_page(self._current_page_idx - 1)

    def _next_page(self) -> None:
        self._show_page(self._current_page_idx + 1)
