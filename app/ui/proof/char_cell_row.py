"""Phase 11 task 2 — minimal CharCell / fixed-cell mode for HProof.

设计目标（最小可感知版）：
- 每个字符 = 一个固定尺寸的 cell：上半区是该字符的图像 crop，下半区是
  长度限制为 1 的 QLineEdit；
- Tab / Shift+Tab 在 cell 间移动光标（Qt 默认 focus chain 已经满足，
  无需额外代码）；
- Enter 在任意 cell 上提交本行（emit commit_requested），让 panel 跳到
  下一处 AUTO_FLAGGED；
- 任意 cell 编辑后：收集所有 cell 文本拼成新 line.text，emit text_saved；
- line.chars 为空（OCR 没产生 per-char bbox）时，回退展示一条提示，避免
  让用户以为字格模式坏了。

刻意不做（写在 handoff section 3）：
- candidate float（候选悬浮）/ 图文双向高亮 / 序列对齐 / 基线对齐
- 长行虚拟化 / cell 拖拽 / 多字符 cell 合并

最重要的 invariant：本控件的所有交互最终都是通过 line.update_text(...) 改
project 状态；上层 _LinePair / HProofPanel 不需要知道它内部用了多少 cell。
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Optional, Tuple

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.core.page_image_cache import PageImageCache
from app.models import Line, Page
from app.ui.proof.confidence_utils import char_confidence, normalize_confidence

CELL_W = 32
IMG_H  = 36
EDIT_H = 26

# Phase 14b：字符种类分类 → cell 视觉/策略可分流
KIND_CJK     = "cjk"      # 汉字、CJK 标点（默认主体）
KIND_LETTER  = "letter"   # 拉丁字母
KIND_DIGIT   = "digit"    # 阿拉伯数字
KIND_PUNCT   = "punct"    # ASCII 半角标点 + 常见全角标点
KIND_FORMULA = "formula"  # LaTeX/公式碎片（\、{、}、$）
KIND_OTHER   = "other"

_PUNCT_SET = set(""",.;:!?\"'()[]{}<>-_/\\|~`@#%&*+=、，。；：！？「」『』（）【】《》—…·""")

def classify_char_kind(ch: str) -> str:
    """轻量字符种类分类 —— Phase 14b 第一版策略边界。"""
    if not ch:
        return KIND_OTHER
    c = ch[0]
    if c in "\${}^_":
        return KIND_FORMULA
    if c.isdigit():
        return KIND_DIGIT
    if c.isascii() and c.isalpha():
        return KIND_LETTER
    if c in _PUNCT_SET:
        return KIND_PUNCT
    if "一" <= c <= "鿿" or "　" <= c <= "〿":
        return KIND_CJK
    return KIND_OTHER


def line_looks_like_formula(text: str) -> bool:
    """Phase 14b：粗略判断本行是否公式行 → 字格模式 fallback 不构建 cell。

    边界：line.text 含 \begin{ 或 包裹 $$ 或 整行 ≥40% 字符是公式 kind 时视为公式。
    """
    if not text:
        return False
    if ("\\" + "begin{") in text or text.strip().startswith("$$") or text.strip().endswith("$$"):
        return True
    formula_chars = sum(1 for c in text if classify_char_kind(c) == KIND_FORMULA)
    return formula_chars / max(1, len(text)) >= 0.4


class _CharCellLineEdit(QLineEdit):
    """单字符输入，禁用 Qt 的 Enter『提交焦点』默认行为，改为发自定义信号。"""

    enter_pressed = Signal()
    next_cell_requested = Signal()
    prev_cell_requested = Signal()
    focus_in = Signal()
    find_low_conf_requested = Signal()  # Ctrl+J

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Phase 13: 去掉 maxLength=1，允许 1 个 bbox 对应多字符 (N→M)
        # 含空字符（用户可删空表示该 bbox 无字）。
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(CELL_W, EDIT_H)
        self._base_border = "#ddd"
        self._apply_style()

    def set_confidence_hint(self, conf: float | None) -> None:
        """根据 OCR confidence 着色边框：低 → 红，中 → 黄，高 → 灰。"""
        conf = normalize_confidence(conf)
        if conf is None:
            self._base_border = "#ddd"
            self._apply_style()
            return
        if conf < 0.6:
            self._base_border = "#d93025"   # 红
        elif conf < 0.85:
            self._base_border = "#f9a825"   # 黄
        else:
            self._base_border = "#ddd"      # 灰
        self._apply_style()

    def set_align_hint(self, kind: str) -> None:
        """Phase 14a：sequence-aware reseat 的 align kind 着色。
        replace → 红实线（"text 改了"）；insert → 灰虚线（"OCR 多出，text 没填"）。"""
        if kind == "replace":
            self._base_border = "#d93025"
            self._apply_style()
        elif kind == "insert":
            self.setStyleSheet(
                "QLineEdit { font-size:14px; padding:0; "
                "border:1px dashed #bbb; background:#fafafa; }"
                "QLineEdit:focus { background:#f0f6ff; border:1px solid #1a73e8; }"
            )

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"QLineEdit {{ font-size:14px; padding:0; "
            f"border:1px solid {self._base_border}; }}"
            "QLineEdit:focus { background:#f0f6ff; border:1px solid #1a73e8; }"
        )

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        super().focusInEvent(event)
        self.focus_in.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        key = event.key()
        # Phase 14b：Ctrl+J 跳下一低置信度 cell
        if key == Qt.Key.Key_J and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.find_low_conf_requested.emit()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.enter_pressed.emit()
            return
        # Phase 15 blocker 2：Tab / Shift+Tab 走 cell 链（无条件，不看光标位置），
        # 越界时由 CharCellRow._focus_neighbor 发 next_off_end / prev_off_start。
        # Key_Backtab = Shift+Tab；Qt 在 Shift+Tab 时给 Tab 时不一定带 modifier。
        if key == Qt.Key.Key_Tab:
            self.next_cell_requested.emit()
            return
        if key == Qt.Key.Key_Backtab:
            self.prev_cell_requested.emit()
            return
        if key == Qt.Key.Key_Right and self.cursorPosition() == len(self.text()):
            self.next_cell_requested.emit()
            return
        if key == Qt.Key.Key_Left and self.cursorPosition() == 0:
            self.prev_cell_requested.emit()
            return
        super().keyPressEvent(event)


class CharCellRow(QFrame):
    """一行的字格视图。

    Signals
    -------
    text_committed(str): 用户编辑了任何 cell 后，发出当前拼接文本。
    commit_requested(): 用户在 cell 上按了 Enter（请求保存并跳到下一处）。
    """

    text_committed   = Signal(str)
    commit_requested = Signal()
    focus_changed    = Signal(int)   # cell idx 拿到焦点
    next_off_end     = Signal()      # 在最后一个 cell 上仍按 → / Tab → 跨行
    prev_off_start   = Signal()      # 在第一个 cell 上仍按 ← / Shift+Tab → 跨行

    def __init__(
        self,
        line: Line,
        page: Page,
        cache: PageImageCache,
        parent: Optional[QWidget] = None,
        display_text: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._line  = line
        self._page  = page
        self._cache = cache
        self._cells: List[_CharCellLineEdit] = []
        # Phase 15 blocker 1: trailing_overflow 必须在 _on_cell_changed 拼接时带回，
        # 否则用户在字格模式编辑会静默丢掉行末未对上图的字。默认空串。
        self._trailing_overflow: str = ""
        # Phase 18 blocker 2：显式接收"显示空间"文本（quality-probe 开启时含 fake_char）。
        # 默认 fall back 到 line.text，保持纯单元测试 / 无 probe 场景行为不变。
        self._display_text: str = display_text if display_text is not None else (line.text or "")
        self.setObjectName("charCellRow")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(IMG_H + EDIT_H + 6)
        self._build()

    # ── 构建 ──────────────────────────────────────────────────

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)

        if not self._line.chars:
            # 没有 per-char bbox：回退提示，不阻断主流程
            tip = QLabel("（本行 OCR 未产生 per-char bbox，字格模式不可用，请回退普通模式）")
            tip.setStyleSheet("color:#999; font-size:11px;")
            root.addWidget(tip)
            root.addStretch()
            return
        # Phase 14b：公式行 fallback —— 字格模式对长公式无意义，明确边界
        # Phase 18 blocker 2：基于显示空间文本判断（quality-probe 开启时一致）
        if line_looks_like_formula(self._display_text or ""):
            tip = QLabel("（本行疑似公式，字格模式 fallback；请用普通模式编辑 LaTeX）")
            tip.setStyleSheet("color:#0b66c3; font-size:11px;")
            root.addWidget(tip)
            root.addStretch()
            return

        image = self._cache.get_page_image(self._page.display_image_path)
        # Phase 14a-2：行级统一 scale —— 取所有 char.bbox 高度的中位数作 ref_h，
        # 把 IMG_H 上限映射到 ref_h，避免不同字大小不一引起阅读节奏抖动。
        bb_heights = [c.bbox.h for c in self._line.chars
                      if c.bbox is not None and c.bbox.h > 0]
        if bb_heights:
            sorted_h = sorted(bb_heights)
            ref_h = sorted_h[len(sorted_h) // 2]
        else:
            ref_h = IMG_H
        self._row_scale = (IMG_H - 4) / max(1, ref_h)  # 留 2px 上下空隙
        # Phase 14a：sequence-aware reseat
        # 用 SequenceMatcher 把 line.text 字符序列对齐到 chars 字符序列，
        # 得到每个 cell 的 (init_text, kind) 元组：
        #   kind="equal"   text 与 OCR 字符一致 → 普通边框
        #   kind="replace" text 与 OCR 不同     → 红边警示
        #   kind="insert"  OCR 没有字而 text 有 → cell 留空（不强塞）
        # text 末尾多出来的字符进 trailing_overflow，挂到最后一格做提示。
        # Phase 18 blocker 2：用显示空间文本对齐字格 → 落盘走 _save_displayed_edit 一致
        cell_inits, trailing_overflow = self._align_text_to_chars(
            list(self._display_text or ""),
            [c.char for c in self._line.chars],
        )
        self._trailing_overflow = trailing_overflow
        for idx, char in enumerate(self._line.chars):
            init_text, kind = cell_inits[idx]
            conf = char_confidence(self._line, idx)
            cell = self._build_cell(idx, init_text, char.bbox, image,
                                    conf, align_kind=kind)
            root.addLayout(cell)
        if trailing_overflow:
            tip = QLabel(f"+{len(trailing_overflow)} 字未对上图（{trailing_overflow!r}）")
            tip.setStyleSheet("color:#a8651b; font-size:11px; padding-left:6px;")
            root.addWidget(tip)
        root.addStretch()

    @staticmethod
    def _align_text_to_chars(text_chars: List[str],
                             ocr_chars: List[str]) -> Tuple[List[Tuple[str, str]], str]:
        """把用户文本 text_chars 对齐到 OCR 字符序列 ocr_chars。

        返回 (cell_inits, trailing_overflow):
          cell_inits 与 ocr_chars 同长；每项 = (init_text, kind)。
          trailing_overflow = text 末尾未能塞入任何 cell 的剩余字符串。

        kind 分类：
          - "equal":   text 与 OCR 一致 → 直接放
          - "replace": 1↔1 替换 → 放并标 replace
          - "insert":  OCR 多出（text 没对应）→ 留空字符串
          - "drop":    text 多出但落在中间 → 合并到下一个 equal/replace cell 的前缀
        末尾 text 多余的不放 cell，作为 trailing_overflow 返回。
        """
        n = len(ocr_chars)
        cell_inits: List[Tuple[str, str]] = [("", "insert")] * n
        if not text_chars:
            return cell_inits, ""
        sm = SequenceMatcher(a=ocr_chars, b=text_chars, autojunk=False)
        # 用 opcodes 遍历，把 text 的字符按 ocr 索引塞回去
        # 同时记录被"跳过"的 text 字符 → 合并到下一个 ocr 位置（pending_prefix）
        pending_prefix = ""
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    cell_inits[i1 + k] = (pending_prefix + text_chars[j1 + k], "equal")
                    pending_prefix = ""
            elif tag == "replace":
                # 1↔1 重叠位置 replace；text 多出来部分进 pending_prefix；ocr 多出来部分留空
                overlap = min(i2 - i1, j2 - j1)
                for k in range(overlap):
                    cell_inits[i1 + k] = (pending_prefix + text_chars[j1 + k], "replace")
                    pending_prefix = ""
                # text 多出（j2-j1 > overlap）→ 进 pending_prefix
                if (j2 - j1) > overlap:
                    pending_prefix += "".join(text_chars[j1 + overlap: j2])
                # ocr 多出（i2-i1 > overlap）→ 那些 cell 保留默认 ("", "insert")
            elif tag == "delete":
                # OCR 有，text 没 → cell 保留 ("", "insert")
                pass
            elif tag == "insert":
                # text 多出 → 累积到 pending_prefix，等下一个 ocr 位置吸收
                pending_prefix += "".join(text_chars[j1: j2])
        # 收尾：剩余 pending_prefix 没人接，作为 trailing_overflow
        return cell_inits, pending_prefix

    def _build_cell(self, idx: int, ch: str, bbox, image, conf: float | None = None,
                    align_kind: str = "equal") -> QVBoxLayout:
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        # ── 图像 crop ──
        img_lbl = QLabel()
        img_lbl.setFixedSize(CELL_W, IMG_H)
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl.setStyleSheet("background:#fafbfc; border:1px solid #eee;")
        if image is not None and bbox is not None and bbox.w > 0 and bbox.h > 0:
            H, W = image.shape[:2]
            x1 = max(0, bbox.x); y1 = max(0, bbox.y)
            x2 = min(W, bbox.x2); y2 = min(H, bbox.y2)
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                ch_h, ch_w = crop.shape[:2]
                # Phase 14a-2：用行级统一 scale；超出 CELL_W 时按宽度收缩。
                scale = self._row_scale
                if ch_w * scale > CELL_W:
                    scale = CELL_W / max(1, ch_w)
                new_w = max(1, int(round(ch_w * scale)))
                new_h = max(1, int(round(ch_h * scale)))
                resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.tobytes(), new_w, new_h, new_w * 3,
                              QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
                # baseline 底部对齐：固定 IMG_H 的 label 内，pixmap 居底
                img_lbl.setPixmap(pix)
                img_lbl.setAlignment(
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom
                )
        col.addWidget(img_lbl)

        # ── 编辑 cell ──
        edit = _CharCellLineEdit()
        # align_kind 优先级高于 confidence：replace=红、insert=灰虚线、equal=按 conf
        if align_kind == "replace":
            edit.set_align_hint("replace")
        elif align_kind == "insert":
            edit.set_align_hint("insert")
        else:
            edit.set_confidence_hint(conf)
        edit.setText(ch)
        edit.textEdited.connect(lambda _t, _i=idx: self._on_cell_changed())
        edit.enter_pressed.connect(self.commit_requested.emit)
        edit.next_cell_requested.connect(lambda _i=idx: self._focus_neighbor(_i, +1))
        edit.prev_cell_requested.connect(lambda _i=idx: self._focus_neighbor(_i, -1))
        edit.focus_in.connect(lambda _i=idx: self.focus_changed.emit(_i))
        edit.find_low_conf_requested.connect(
            lambda _i=idx: self.focus_next_low_conf(_i)
        )
        col.addWidget(edit)
        self._cells.append(edit)
        return col

    # ── 行为 ──────────────────────────────────────────────────

    def _focus_neighbor(self, idx: int, delta: int) -> None:
        nxt = idx + delta
        if 0 <= nxt < len(self._cells):
            self._cells[nxt].setFocus()
            self._cells[nxt].setCursorPosition(len(self._cells[nxt].text()))
        elif delta > 0:
            self.next_off_end.emit()
        else:
            self.prev_off_start.emit()

    def _on_cell_changed(self) -> None:
        # Phase 15 blocker 1：把行末 trailing_overflow 一起带回，杜绝静默丢字。
        # 若用户希望删掉行末多余文本，可在普通模式编辑；字格模式不主动改 overflow。
        new_text = "".join(c.text() for c in self._cells) + self._trailing_overflow
        self.text_committed.emit(new_text)

    def focus_first(self) -> None:
        if self._cells:
            self._cells[0].setFocus()
            self._cells[0].selectAll()

    def focus_cell(self, idx: int) -> None:
        """外部（图→文）请求把焦点放到第 idx 个 cell。"""
        if 0 <= idx < len(self._cells):
            self._cells[idx].setFocus()
            self._cells[idx].selectAll()

    def focus_next_low_conf(self, from_idx: int = -1, threshold: float = 0.85) -> bool:
        """跳到下一个 OCR confidence < threshold 的 cell；返回是否找到。"""
        if not self._line.chars:
            return False
        n = len(self._line.chars)
        for k in range(1, n + 1):
            i = (from_idx + k) % n
            ch = self._line.chars[i]
            conf = char_confidence(self._line, i)
            if conf is not None and conf < threshold and i < len(self._cells):
                self.focus_cell(i)
                return True
        return False

    def current_focus_idx(self) -> int:
        """返回当前持有焦点的 cell idx；无则 -1。"""
        for i, c in enumerate(self._cells):
            if c.hasFocus():
                return i
        return -1

    @property
    def has_cells(self) -> bool:
        return bool(self._cells)
