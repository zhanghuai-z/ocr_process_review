"""Vertical proof and same-text index view over an immutable workspace.

The panel owns only display state: it reads :class:`ProofWorkspaceView`
snapshots and emits ``ProofEditCommand`` values for every mutation request.

Interaction model restored from the mature vertical proof view (d4c6dfe):

- direct typing on the gallery (printable keys or IME commit strings)
  overwrites the selected occurrences; Backspace/Delete fills blanks so the
  proof text length never changes implicitly;
- a candidate panel ranked from the current glyph, the OCR source glyph at
  the same position and the built-in confusable tables;
- an :class:`ImageViewer` original-image pane that highlights the selected
  character box and its line, and reverse-locates the text context when the
  user clicks inside a line box;
- conservative multi-select batch edits: only occurrences on the currently
  displayed page are replaced, cross-page occurrences are skipped and the
  skip count is reported in the status bar;
- selection survives workspace snapshot replacements: occurrences are
  re-located by their stable occurrence keys.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from PySide6.QtCore import QEvent, QItemSelectionModel, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.application.contracts import ProofEditCommand
from app.application.proof_workspace import (
    ProofLineView,
    ProofPageView,
    ProofStateView,
    ProofTextUnitView,
    ProofWorkspaceView,
)
from app.ui.proof.char_verdict import COLOR_ERROR, COLOR_LIKELY_OK
from app.ui.proof.confidence_view import ProofCharView, build_char_views
from app.ui.widgets.effects import apply_soft_shadow
from app.ui.widgets.image_viewer import ImageViewer


IMAGE_SIZE = QSize(620, 420)
# Gallery cells mirror the mature delegate: a 56px thumbnail with breathing
# room inside a 70px grid, six items per row and at most three visible rows.
GALLERY_THUMB = 56
GALLERY_CELL = GALLERY_THUMB + 14
GALLERY_ITEMS_PER_ROW = 6
LOW_CONF = 0.80

DEFAULT_CONFUSABLE_CANDIDATES = {
    # 田/由/甲/申 系
    "田": ["由", "甲", "申", "曲"],
    "由": ["田", "甲", "申"],
    "甲": ["田", "由", "申"],
    "申": ["田", "由", "甲"],
    # 日/曰/目/口 系
    "日": ["曰", "目", "口"],
    "曰": ["日", "目", "口"],
    "目": ["日", "曰", "自", "首"],
    "口": ["日", "曰", "囗", "回"],
    "囗": ["口", "回", "国"],
    # 己/已/巳 系
    "己": ["已", "巳"],
    "已": ["己", "巳"],
    "巳": ["己", "已"],
    # 末/未 系
    "未": ["末", "朱"],
    "末": ["未", "朱"],
    "朱": ["未", "末"],
    # 人/入/八 系
    "人": ["入", "八", "个"],
    "入": ["人", "八"],
    "八": ["人", "入"],
    # 大/太/犬 系
    "大": ["太", "犬", "夫"],
    "太": ["大", "犬", "夫"],
    "犬": ["大", "太"],
    "夫": ["大", "天", "夭"],
    "天": ["夫", "夭", "无"],
    "夭": ["天", "夫"],
    # 干/千/午/壬 系
    "干": ["千", "午", "于"],
    "千": ["干", "午"],
    "午": ["干", "千", "牛"],
    "牛": ["午", "牟"],
    # 土/士/王/玉/主/王 系
    "土": ["士", "工"],
    "士": ["土", "仕"],
    "王": ["玉", "主", "壬"],
    "玉": ["王", "主"],
    "主": ["王", "玉", "住"],
    "壬": ["王", "工", "土"],
    # 工/匚 系
    "工": ["土", "士", "壬"],
    # 又/叉/义 系
    "又": ["叉", "义"],
    "叉": ["又"],
    "义": ["又", "乂"],
    # 力/刀/办 系
    "力": ["刀", "办"],
    "刀": ["力", "刃"],
    "刃": ["刀"],
    # 木/术/本/朩
    "木": ["术", "本", "朩"],
    "术": ["木", "朮"],
    "本": ["木", "未", "末"],
    # 水/氺/永/冰
    "水": ["氺", "永", "冰"],
    "永": ["水", "求"],
    "冰": ["水", "永"],
    # 火/灬
    "火": ["灬", "炎"],
    # 心/必
    "心": ["必", "忄"],
    "必": ["心"],
    # 巾/币/市
    "巾": ["币", "市", "布"],
    "币": ["巾", "市"],
    "市": ["巾", "币", "布"],
    "布": ["巾", "市"],
    # 戊/戌/戍/戎/成
    "戊": ["戌", "戍", "戎", "成"],
    "戌": ["戊", "戍", "戎"],
    "戍": ["戊", "戌", "戎"],
    "戎": ["戊", "戌", "戍"],
    "成": ["戊", "戌"],
    # 凡/几/丸
    "凡": ["几", "丸"],
    "几": ["凡", "九"],
    "丸": ["凡", "九"],
    "九": ["几", "丸"],
    # 北/比/此
    "北": ["比", "兆"],
    "比": ["北", "此"],
    "此": ["比"],
    # 卜/上/下/不
    "卜": ["上", "下", "不"],
    "上": ["卜", "下"],
    "下": ["卜", "上"],
    "不": ["卜", "丕"],
    # 千/午/牛/年
    "年": ["午", "牛"],
    # 自/白/百
    "自": ["白", "百", "目"],
    "白": ["自", "百"],
    "百": ["白", "自"],
    # 干/于/亏
    "于": ["干", "亏", "乎"],
    "亏": ["于", "夸"],
    # 风/凤/凡
    "风": ["凤", "凡"],
    "凤": ["风", "凡"],
    # 鸟/乌/马
    "鸟": ["乌"],
    "乌": ["鸟"],
    # 兔/免
    "兔": ["免", "兑"],
    "免": ["兔"],
    # 衣/农/表
    "衣": ["农", "表"],
    "农": ["衣"],
    "表": ["衣"],
    # 万/方
    "万": ["方"],
    "方": ["万"],
    # 历/厉
    "历": ["厉"],
    "厉": ["历"],
    # 京/亨/享
    "京": ["亨", "享", "亰"],
    "亨": ["京", "享"],
    "享": ["京", "亨"],
    # 长
    "长": ["镸"],
    # 兵/丘
    "兵": ["丘"],
    "丘": ["兵"],
    # 体/休
    "休": ["体"],
    "体": ["休"],
    # 化/华
    "化": ["华"],
    "华": ["化"],
    # 今/令/兮
    "今": ["令", "兮"],
    "令": ["今"],
    "兮": ["今"],
    # 半/羊/丰
    "半": ["羊", "丰"],
    "丰": ["半", "羊"],
    "羊": ["半", "丰"],
    # 兄/见/贝/页
    "兄": ["见", "克"],
    "见": ["兄", "贝"],
    "贝": ["见", "页"],
    "页": ["贝", "顶"],
    # 圆/园/团
    "圆": ["园", "团"],
    "园": ["圆"],
    "团": ["圆", "园"],
    # 句/旬/勺
    "句": ["旬", "勺"],
    "旬": ["句"],
    "勺": ["句"],
}


@dataclass(frozen=True, slots=True)
class _SceneBBox:
    """Duck-typed ``x/y/w/h`` box for :class:`ImageViewer` overlay calls."""

    x: int
    y: int
    w: int
    h: int


def _entry_key(entry: ProofCharView) -> tuple[str, str, int, str | None]:
    return (entry.proof_uid, entry.text_unit_uid, entry.char_index, entry.atom_uid)


def _gallery_crop_pad(entry: ProofCharView) -> int:
    """Crop padding (source pixels) for one gallery thumbnail.

    CJK glyphs keep a small adaptive margin so strokes never touch the cell
    edge; latin/digit boxes are tight by construction and only get a 1px
    safety margin; half-width punctuation needs 2px of context to stay
    legible.
    """

    char = entry.text or ""
    if entry.bbox is not None:
        adaptive = max(2, round(min(entry.bbox[2] - entry.bbox[0], entry.bbox[3] - entry.bbox[1]) * 0.12))
    else:
        adaptive = 2
    if len(char) == 1 and char.isascii() and not char.isspace():
        return 1 if char.isalnum() else 2
    return adaptive


def _page_pixmap(
    page: ProofPageView,
    bbox: tuple[int, int, int, int] | None,
    target_size: QSize | None = None,
    *,
    cache: dict[str, QPixmap] | None = None,
    pad: int = 0,
) -> QPixmap:
    """Return a character crop, never a full-page thumbnail.

    ``cache`` maps ``page_uid`` to a decoded full-page pixmap so repeated
    crops of one page decode the source image at most once.  ``pad`` expands
    the crop rect in source pixels before scaling.
    """

    if bbox is None:
        return QPixmap()
    source = cache.get(page.page_uid) if cache is not None else None
    if source is None:
        source = QPixmap(page.image_path)
        if cache is not None and not source.isNull():
            cache[page.page_uid] = source
    if source.isNull() or page.width <= 0 or page.height <= 0:
        return QPixmap()
    left, top, right, bottom = bbox
    x_scale = source.width() / page.width
    y_scale = source.height() / page.height
    crop_rect = QRect(
        round(left * x_scale),
        round(top * y_scale),
        max(1, round((right - left) * x_scale)),
        max(1, round((bottom - top) * y_scale)),
    )
    if pad > 0:
        crop_rect = crop_rect.adjusted(-pad, -pad, pad, pad)
    crop_rect = crop_rect.intersected(source.rect())
    if crop_rect.isEmpty():
        return QPixmap()
    crop = source.copy(crop_rect)
    target = target_size if target_size is not None and not target_size.isEmpty() else IMAGE_SIZE
    return crop.scaled(
        max(1, target.width()),
        max(1, target.height()),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class _GalleryListWidget(QListWidget):
    """Same-text gallery with direct typing and IME commit overwrite."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._gallery_owner: VProofPanel | None = None
        # QListWidget does not accept IME input by default; opting in here
        # routes committed CJK strings through inputMethodEvent below.
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)

    def set_gallery_owner(self, owner: "VProofPanel") -> None:
        self._gallery_owner = owner

    def inputMethodQuery(self, query):  # type: ignore[override]
        if query == Qt.InputMethodQuery.ImEnabled:
            return True
        return super().inputMethodQuery(query)

    def inputMethodEvent(self, event) -> None:  # type: ignore[override]
        """Overwrite the selection with committed IME text.

        Preedit strings are swallowed (never rendered into the model); only
        the commit string becomes a replacement, and it may contain several
        characters.
        """

        owner = self._gallery_owner
        commit = event.commitString() if event is not None else ""
        if owner is None or not commit:
            super().inputMethodEvent(event)
            return
        if owner._gallery_direct_overwrite(commit):
            event.accept()
            return
        super().inputMethodEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        owner = self._gallery_owner
        if owner is None:
            super().keyPressEvent(event)
            return
        mods = event.modifiers()
        # Ctrl/Alt/Meta chords (navigation, select-all, undo) stay with the
        # shortcuts and the native selection behaviour.
        disallowed = (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        )
        if mods & disallowed:
            super().keyPressEvent(event)
            return
        key = event.key()
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if owner._gallery_direct_blank():
                event.accept()
                return
            super().keyPressEvent(event)
            return
        text = event.text()
        if len(text) == 1 and text.isprintable() and not text.isspace():
            if owner._gallery_direct_overwrite(text):
                event.accept()
                return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        owner = self._gallery_owner
        if owner is not None and event.button() == Qt.MouseButton.RightButton:
            position = event.position().toPoint()
            item = self.itemAt(position)
            if item is not None and not item.isSelected():
                self.setCurrentItem(item, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            owner._show_edit_bubble_at(position)
            event.accept()
            return
        super().mousePressEvent(event)


class VProofPanel(QWidget):
    """Vertical proof view that emits application proof edit commands."""

    proof_edit_requested = Signal(object)

    def __init__(
        self,
        workspace: ProofWorkspaceView | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._workspace: ProofWorkspaceView | None = None
        self._pages: dict[str, ProofPageView] = {}
        self._states: dict[str, ProofStateView] = {}
        self._lines: dict[tuple[str, str], ProofLineView] = {}
        self._units: dict[tuple[str, str], ProofTextUnitView] = {}
        self._entries: tuple[ProofCharView, ...] = ()
        self._entries_by_text: dict[str, tuple[ProofCharView, ...]] = {}
        self._selected_char = ""
        self._selected_tokens: list[str] = []
        self._selected_entry: ProofCharView | None = None
        self._page_pixmaps: dict[str, QPixmap] = {}
        self._page_pixmap_keys: dict[str, tuple[str, int]] = {}
        self._viewer_page_uid: str | None = None
        self._candidate_buttons: list[QPushButton] = []
        self._build_ui()
        self._build_shortcuts()
        if workspace is not None:
            self.set_workspace(workspace)

    @property
    def workspace(self) -> ProofWorkspaceView | None:
        return self._workspace

    # ───────────────────────── UI ─────────────────────────

    def _build_ui(self) -> None:
        self.setObjectName("proofRoot")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setObjectName("proofSplitter")
        main_splitter.setHandleWidth(10)
        main_splitter.setChildrenCollapsible(False)
        left = QFrame()
        left.setObjectName("proofLeftPane")
        left.setMinimumWidth(180)
        left.setMaximumWidth(240)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_header = QHBoxLayout()
        left_title = QLabel("字符索引")
        left_title.setObjectName("sectionTitle")
        left_header.addWidget(left_title)
        left_header.addStretch(1)
        self._char_count = QLabel("0 项")
        self._char_count.setObjectName("muted")
        left_header.addWidget(self._char_count)
        left_layout.addLayout(left_header)
        self._char_search = QLineEdit()
        self._char_search.setPlaceholderText("搜索字符…")
        self._char_search.textChanged.connect(self._rebuild_char_list)
        left_layout.addWidget(self._char_search)
        self._char_list = QListWidget()
        self._char_list.setObjectName("charIndexList")
        # Multi-select merges the occurrences of several glyph buckets into
        # one gallery; single selection stays the common path.
        self._char_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._char_list.itemSelectionChanged.connect(self._on_char_selection_changed)
        left_layout.addWidget(self._char_list, 1)
        main_splitter.addWidget(left)

        content = QWidget()
        content.setObjectName("proofContentPane")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)
        self._content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._content_splitter.setObjectName("proofContentSplitter")
        self._content_splitter.setHandleWidth(10)

        center = QFrame()
        center.setObjectName("proofCenterPane")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(10, 10, 10, 10)
        center_layout.setSpacing(10)
        gallery_card = QFrame()
        gallery_card.setObjectName("proofCard")
        gallery_layout = QVBoxLayout(gallery_card)
        gallery_layout.setContentsMargins(10, 8, 10, 10)
        gallery_header = QHBoxLayout()
        self._gallery_header = QLabel("相同字索引")
        self._gallery_header.setObjectName("sectionTitle")
        gallery_header.addWidget(self._gallery_header)
        gallery_header.addStretch(1)
        self._btn_refresh = QPushButton("刷新")
        self._btn_refresh.setObjectName("ghostBtn")
        self._btn_refresh.clicked.connect(self.refresh_view)
        gallery_header.addWidget(self._btn_refresh)
        gallery_layout.addLayout(gallery_header)
        self._gallery = _GalleryListWidget()
        self._gallery.set_gallery_owner(self)
        self._gallery.setObjectName("proofGallery")
        self._gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self._gallery.setFlow(QListWidget.Flow.LeftToRight)
        self._gallery.setWrapping(True)
        self._gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._gallery.setMovement(QListWidget.Movement.Static)
        self._gallery.setUniformItemSizes(True)
        self._gallery.setGridSize(QSize(GALLERY_CELL, GALLERY_CELL))
        self._gallery.setIconSize(QSize(GALLERY_THUMB, GALLERY_THUMB))
        self._gallery.setSpacing(4)
        self._gallery.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._gallery.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._gallery.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        self._gallery.currentItemChanged.connect(self._on_gallery_changed)
        gallery_layout.addWidget(self._gallery, 1)
        center_layout.addWidget(gallery_card, 3)

        self._candidate_panel = self._build_candidate_panel()
        center_layout.addWidget(self._candidate_panel)

        context_card = QFrame()
        context_card.setObjectName("proofCard")
        context_layout = QVBoxLayout(context_card)
        context_layout.setContentsMargins(10, 8, 10, 10)
        context_title = QLabel("OCR 文本上下文")
        context_title.setObjectName("sectionTitle")
        context_layout.addWidget(context_title)
        self._ocr_context = QPlainTextEdit()
        self._ocr_context.setObjectName("vproofOcrContext")
        self._ocr_context.setReadOnly(True)
        self._ocr_context.setPlaceholderText("OCR 文本上下文")
        context_layout.addWidget(self._ocr_context, 1)
        center_layout.addWidget(context_card, 2)

        self._status = QLabel("暂无可校对字符")
        self._status.setObjectName("muted")
        center_layout.addWidget(self._status)

        viewer = QFrame()
        viewer.setObjectName("proofRightPane")
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(12, 12, 12, 12)
        viewer_title = QLabel("原稿上下文")
        viewer_title.setObjectName("sectionTitle")
        viewer_layout.addWidget(viewer_title)
        self._image = ImageViewer()
        # Read-only proofing surface: panning stays available, block editing
        # is never enabled here.
        self._image.set_edit_mode(False)
        self._image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image.viewport().installEventFilter(self)
        viewer_layout.addWidget(self._image, 1)

        # The center column must fit one row of six 56px thumbnails:
        # 6 * (THUMB + 8) + paddings ~= 424, plus candidate/context chrome.
        center.setMinimumWidth(max(520, GALLERY_ITEMS_PER_ROW * (GALLERY_THUMB + 8) + 40))
        viewer.setMinimumWidth(300)
        self._content_splitter.addWidget(center)
        self._content_splitter.addWidget(viewer)
        self._content_splitter.setStretchFactor(0, 1)
        self._content_splitter.setStretchFactor(1, 1)
        self._content_splitter.setSizes([740, 560])
        content_layout.addWidget(self._content_splitter)
        main_splitter.addWidget(content)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([210, 1100])
        root.addWidget(main_splitter, 1)
        self._build_edit_bubble()

    def _build_candidate_panel(self) -> QFrame:
        # Only actionable candidates are visible; diagnostics live in the
        # panel tooltip so they never crowd the OCR context.
        box = QFrame()
        box.setObjectName("candidatePanel")
        box.setFixedHeight(72)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(4)
        self._candidate_title = QLabel("候选字")
        self._candidate_title.setObjectName("sectionTitle")
        self._candidate_buttons_row = QHBoxLayout()
        self._candidate_buttons_row.setSpacing(6)
        layout.addWidget(self._candidate_title)
        layout.addLayout(self._candidate_buttons_row)
        return box

    def _build_edit_bubble(self) -> None:
        self._edit_bubble = QFrame(self)
        self._edit_bubble.setObjectName("vproofEditBubble")
        apply_soft_shadow(self._edit_bubble, blur_radius=24, y_offset=4, alpha=24)
        layout = QHBoxLayout(self._edit_bubble)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(0)
        self._edit_bubble_input = QLineEdit()
        self._edit_bubble_input.setObjectName("vproofEditBubbleInput")
        self._edit_bubble_input.setFrame(False)
        self._edit_bubble_input.setMaxLength(32)
        self._edit_bubble_input.returnPressed.connect(self._apply_edit_bubble)
        self._edit_bubble_input.installEventFilter(self)
        layout.addWidget(self._edit_bubble_input)
        self._edit_bubble.resize(156, 42)
        self._edit_bubble.hide()

    def _build_shortcuts(self) -> None:
        def scoped(key: str, parent: QWidget, handler) -> None:
            shortcut = QShortcut(QKeySequence(key), parent)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(handler)

        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._redo_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self._redo_y_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        for shortcut in (self._undo_shortcut, self._redo_shortcut, self._redo_y_shortcut):
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._undo_shortcut.activated.connect(lambda: self._apply_history(-1))
        self._redo_shortcut.activated.connect(lambda: self._apply_history(1))
        self._redo_y_shortcut.activated.connect(lambda: self._apply_history(1))
        # Global char-bucket stepping works regardless of the focus location.
        scoped("Ctrl+.", self, lambda: self._step_char_list(1))
        scoped("Ctrl+,", self, lambda: self._step_char_list(-1))
        # Gallery-local navigation: Alt+Left/Right steps one occurrence,
        # Alt+Up/Down steps one wrapped row, Ctrl+A selects every visible
        # occurrence, Esc collapses the selection back to the current item.
        scoped("Alt+Right", self._gallery, lambda: self._step_gallery(1))
        scoped("Alt+Left", self._gallery, lambda: self._step_gallery(-1))
        scoped("Alt+Down", self._gallery, lambda: self._step_gallery_row(1))
        scoped("Alt+Up", self._gallery, lambda: self._step_gallery_row(-1))
        scoped("Ctrl+A", self._gallery, self._select_all_gallery)
        scoped("Escape", self._gallery, self._clear_gallery_selection)
        scoped("Ctrl+A", self._char_list, self._select_all_visible_chars)
        scoped("Escape", self._char_list, self._clear_char_list_multi)

    # ─────────────────── workspace ───────────────────

    def set_workspace(self, workspace: ProofWorkspaceView | None) -> None:
        """Replace the displayed immutable snapshot."""

        if workspace is not None and not isinstance(workspace, ProofWorkspaceView):
            raise TypeError("VProofPanel requires ProofWorkspaceView or None")
        restore_tokens = self._selected_tokens_for_restore()
        restore_keys = tuple(_entry_key(entry) for entry in self._selected_entries())
        self._workspace = workspace
        self._pages = {page.page_uid: page for page in workspace.pages} if workspace else {}
        self._states = {state.proof_uid: state for state in workspace.proof_states} if workspace else {}
        self._lines.clear()
        self._units.clear()
        self._prune_page_pixmap_cache()
        entries: list[ProofCharView] = []
        if workspace is not None:
            for state in workspace.proof_states:
                units = {unit.text_unit_uid: unit for unit in state.text_units}
                page = self._pages.get(state.page_uid)
                if page is None:
                    raise ValueError(f"proof state {state.proof_uid!r} has no page view")
                for line in state.lines:
                    if line.render_kind != "text":
                        # 公式/表格不进纵校（同旧 CharIndexService：
                        # 全书同字校对主线只收纯文本行）
                        continue
                    unit = units.get(line.text_unit_uid)
                    if unit is None:
                        raise ValueError(f"proof line {line.text_unit_uid!r} has no text unit view")
                    self._lines[(state.proof_uid, unit.text_unit_uid)] = line
                    self._units[(state.proof_uid, unit.text_unit_uid)] = unit
                    entries.extend(build_char_views(line, page))
        self._entries = tuple(
            sorted(entries, key=lambda item: (item.page_number, item.proof_uid, item.text_unit_uid, item.char_index))
        )
        grouped: dict[str, list[ProofCharView]] = defaultdict(list)
        for entry in self._entries:
            grouped[entry.text].append(entry)
        self._entries_by_text = {text: tuple(values) for text, values in grouped.items()}
        self._selected_char = ""
        self._selected_tokens = []
        self._selected_entry = None
        # Re-locate the previous selection through its stable occurrence
        # keys first: after an edit the glyph itself may have changed
        # buckets, and the occurrence's *current* token is the truthful
        # context.  Fall back to the previously selected tokens when the
        # occurrences vanished entirely, so a refresh never drops the
        # user's working context.
        tokens: list[str] = []
        if restore_keys:
            wanted = set(restore_keys)
            for entry in self._entries:
                if _entry_key(entry) in wanted and entry.text not in tokens:
                    tokens.append(entry.text)
        if not tokens:
            tokens = [token for token in restore_tokens if token in self._entries_by_text]
        self._rebuild_char_list(restore_tokens=tokens, restore_keys=restore_keys)
        if not self._entries:
            self._status.setText("暂无可校对字符")
            self._status.setStyleSheet("")
        else:
            self._status.setText(f"{len(self._entries)} 个字符")
            self._status.setStyleSheet("")

    def _prune_page_pixmap_cache(self) -> None:
        """Drop cached page pixmaps whose page vanished or whose image changed."""

        valid = {
            page.page_uid: (page.image_path, page.image_revision)
            for page in self._pages.values()
        }
        for page_uid in list(self._page_pixmaps):
            if valid.get(page_uid) != self._page_pixmap_keys.get(page_uid):
                self._page_pixmaps.pop(page_uid, None)
                self._page_pixmap_keys.pop(page_uid, None)

    def clear_workspace(self) -> None:
        self.set_workspace(None)

    def refresh_view(self) -> None:
        """Rebuild local index structures from the existing immutable view."""

        self.set_workspace(self._workspace)

    # ─────────────────── page selector / char index ───────────────────

    def _filtered_entries(self) -> tuple[ProofCharView, ...]:
        # 纵校是全书同字索引，不存在页过滤概念
        return self._entries

    def _rebuild_char_list(
        self,
        restore_tokens: tuple[str, ...] | list[str] = (),
        restore_keys: tuple[tuple[str, str, int, str | None], ...] = (),
    ) -> None:
        query = self._char_search.text()
        available = self._filtered_entries()
        counts = Counter(entry.text for entry in available if not query or query in entry.text)
        self._char_list.blockSignals(True)
        self._char_list.clear()
        for text, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            item = QListWidgetItem(f"{text}  {count}")
            item.setData(Qt.ItemDataRole.UserRole, text)
            self._char_list.addItem(item)
        self._char_count.setText(f"{len(available)} 项")
        rows: list[int] = []
        if restore_tokens:
            wanted = set(restore_tokens)
            rows = [
                row
                for row in range(self._char_list.count())
                if self._char_list.item(row).data(Qt.ItemDataRole.UserRole) in wanted
            ]
        if not rows and self._char_list.count():
            rows = [0]
        for row in rows:
            self._char_list.item(row).setSelected(True)
        if rows:
            self._char_list.setCurrentRow(rows[0])
        self._char_list.blockSignals(False)
        if rows:
            self._apply_char_selection(restore_keys=restore_keys)
        else:
            self._selected_char = ""
            self._selected_tokens = []
            self._set_gallery(())

    def _selected_char_tokens(self) -> list[str]:
        tokens: list[str] = []
        for item in self._char_list.selectedItems():
            token = item.data(Qt.ItemDataRole.UserRole)
            if token and token not in tokens:
                tokens.append(token)
        return tokens

    def _selected_tokens_for_restore(self) -> list[str]:
        tokens = self._selected_char_tokens()
        if tokens:
            return tokens
        return list(self._selected_tokens)

    def _on_char_selection_changed(self) -> None:
        self._apply_char_selection()

    def _apply_char_selection(
        self,
        restore_keys: tuple[tuple[str, str, int, str | None], ...] = (),
    ) -> None:
        tokens = self._selected_char_tokens()
        if not tokens:
            # A bare clearSelection keeps the previous context, matching the
            # mature panel's behaviour for transient selection resets.
            return
        self._selected_tokens = list(tokens)
        if len(tokens) > 1:
            merged: list[ProofCharView] = []
            for token in tokens:
                merged.extend(self._entries_for_text(token))
            merged.sort(key=lambda item: (item.page_number, item.proof_uid, item.text_unit_uid, item.char_index))
            self._selected_char = "".join(tokens)
            joined = " ".join(f'"{token}"' for token in tokens)
            header = f"跨字索引（{len(tokens)} 字 / 共 {len(merged)} 处）：{joined}"
            self._set_gallery(tuple(merged), header=header, restore_keys=restore_keys)
            return
        self._selected_char = tokens[0]
        entries = self._entries_for_text(tokens[0])
        header = f"相同字索引 · {tokens[0]}（共 {len(entries)} 处）"
        self._set_gallery(entries, header=header, restore_keys=restore_keys)

    def _entries_for_text(self, text: str) -> tuple[ProofCharView, ...]:
        return self._entries_by_text.get(text, ())

    def _step_char_list(self, delta: int) -> None:
        count = self._char_list.count()
        if count <= 0:
            return
        row = self._char_list.currentRow()
        row = 0 if row < 0 else (row + delta) % count
        self._char_list.setCurrentRow(row, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def _select_all_visible_chars(self) -> None:
        """Ctrl+A selects every char bucket currently listed."""

        self._char_list.selectAll()

    def _clear_char_list_multi(self) -> None:
        """Esc collapses the char-list multi-select back to the current row."""

        current = self._char_list.currentItem()
        self._char_list.blockSignals(True)
        self._char_list.clearSelection()
        if current is not None:
            current.setSelected(True)
        self._char_list.blockSignals(False)
        if current is not None:
            self._apply_char_selection()

    # ─────────────────── gallery ───────────────────

    def _set_gallery(
        self,
        entries: tuple[ProofCharView, ...],
        *,
        header: str | None = None,
        restore_keys: tuple[tuple[str, str, int, str | None], ...] = (),
    ) -> None:
        self._gallery.blockSignals(True)
        self._gallery.clear()
        if header is not None:
            self._gallery_header.setText(header)
        else:
            self._gallery_header.setText(
                "相同字索引" if not self._selected_char else f"相同字索引 · {self._selected_char}"
            )
        for entry in entries:
            item = QListWidgetItem(self._entry_icon(entry), "")
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setToolTip(
                f"第 {entry.page_number} 页 · {entry.proof_uid}/{entry.text_unit_uid} · 位 #{entry.char_index + 1}"
            )
            self._gallery.addItem(item)
        if self._gallery.count():
            if not self._restore_gallery_selection_by_occurrence_keys(restore_keys):
                self._gallery.setCurrentRow(0)
                first = self._gallery.item(0)
                entry = first.data(Qt.ItemDataRole.UserRole)
                self._selected_entry = entry if isinstance(entry, ProofCharView) else None
        else:
            self._selected_entry = None
        self._gallery.blockSignals(False)
        self._render_entry(self._selected_entry)
        self._update_candidate_panel(self._selected_entry)
        self._focus_gallery_unless_side_input()

    def _restore_gallery_selection_by_occurrence_keys(
        self,
        occurrence_keys: tuple[tuple[str, str, int, str | None], ...],
    ) -> bool:
        """Re-select gallery rows matching stable occurrence keys.

        Returns ``True`` when at least one occurrence survived the workspace
        rebuild; the first match becomes the current row.
        """

        if not occurrence_keys:
            return False
        wanted = set(occurrence_keys)
        matched_rows: list[int] = []
        for row in range(self._gallery.count()):
            entry = self._gallery.item(row).data(Qt.ItemDataRole.UserRole)
            if isinstance(entry, ProofCharView) and _entry_key(entry) in wanted:
                matched_rows.append(row)
        if not matched_rows:
            return False
        selection = self._gallery.selectionModel()
        model = self._gallery.model()
        for row in matched_rows:
            selection.select(model.index(row, 0), QItemSelectionModel.SelectionFlag.Select)
        selection.setCurrentIndex(
            model.index(matched_rows[0], 0),
            QItemSelectionModel.SelectionFlag.NoUpdate,
        )
        entry = self._gallery.item(matched_rows[0]).data(Qt.ItemDataRole.UserRole)
        self._selected_entry = entry if isinstance(entry, ProofCharView) else None
        return True

    def _entry_icon(self, entry: ProofCharView) -> QIcon:
        page = self._pages.get(entry.page_uid)
        if page is None:
            return QIcon()
        # 2x source density: Qt only ever down-scales the thumbnail.
        cell = GALLERY_CELL * 2
        canvas = QPixmap(cell, cell)
        canvas.fill(QColor("#FFFDF8"))
        painter = QPainter(canvas)
        self._page_source_pixmap(page)  # decodes once per page, keys the cache
        crop = _page_pixmap(
            page,
            entry.bbox,
            QSize(cell - 16, cell - 16),
            cache=self._page_pixmaps,
            pad=_gallery_crop_pad(entry),
        )
        if not crop.isNull():
            painter.drawPixmap((cell - crop.width()) // 2, (cell - crop.height()) // 2, crop)
        else:
            pen_color = QColor("#5C6B58")
            painter.setPen(pen_color)
            font = painter.font()
            font.setPointSize(18)
            painter.setFont(font)
            painter.drawText(
                canvas.rect(),
                Qt.AlignmentFlag.AlignCenter,
                (entry.text or "?")[:4],
            )
        # 旧版心智：相同字索引就是普通的字符切图，不画边框/置信度标记
        painter.end()
        return QIcon(canvas)

    def _remember_page_pixmap(self, page: ProofPageView, pixmap: QPixmap) -> None:
        self._page_pixmaps[page.page_uid] = pixmap
        self._page_pixmap_keys[page.page_uid] = (page.image_path, page.image_revision)

    def _page_source_pixmap(self, page: ProofPageView) -> QPixmap:
        cached = self._page_pixmaps.get(page.page_uid)
        if cached is not None and not cached.isNull():
            return cached
        pixmap = QPixmap(page.image_path)
        if not pixmap.isNull():
            self._remember_page_pixmap(page, pixmap)
        return pixmap

    def _on_gallery_changed(
        self,
        current: QListWidgetItem | None,
        _previous,
    ) -> None:
        entry = current.data(Qt.ItemDataRole.UserRole) if current else None
        self._selected_entry = entry if isinstance(entry, ProofCharView) else None
        self._render_entry(self._selected_entry)
        self._update_candidate_panel(self._selected_entry)
        self._focus_gallery_unless_side_input()

    def _focus_gallery_unless_side_input(self) -> None:
        """Keep the gallery keyboard loop reachable after selection changes."""

        focus = QApplication.focusWidget()
        if focus in (self._char_search, self._edit_bubble_input):
            return
        self._gallery.setFocus()

    def _selected_entries(self) -> tuple[ProofCharView, ...]:
        values: list[ProofCharView] = []
        for item in self._gallery.selectedItems():
            entry = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(entry, ProofCharView):
                values.append(entry)
        if not values and self._selected_entry is not None:
            values.append(self._selected_entry)
        return tuple(values)

    def _gallery_items_per_row(self) -> int:
        grid = self._gallery.gridSize()
        cell = grid.width() if grid.width() > 0 else GALLERY_CELL
        width = self._gallery.viewport().width()
        return max(1, width // max(1, cell))

    def _step_gallery(self, delta: int) -> None:
        count = self._gallery.count()
        if count <= 0:
            return
        row = self._gallery.currentRow()
        row = 0 if row < 0 else (row + delta) % count
        self._gallery.setCurrentRow(row, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def _step_gallery_row(self, delta: int) -> None:
        self._step_gallery(delta * self._gallery_items_per_row())

    def _select_all_gallery(self) -> None:
        self._gallery.selectAll()

    def _clear_gallery_selection(self) -> None:
        """Esc collapses the gallery multi-select back to the current item."""

        current = self._gallery.currentItem()
        self._gallery.clearSelection()
        if current is not None:
            current.setSelected(True)

    # ─────────────────── edit commands ───────────────────

    def _emit(self, command: ProofEditCommand) -> None:
        self.proof_edit_requested.emit(command)

    def _apply_replacement_to_selected(self, text: str) -> int:
        selected = self._selected_entries()
        # Conservative multi-select semantics: a batch only touches entries
        # on the page currently displayed in the viewer; cross-page entries
        # are skipped and the skip count is reported, never silently edited.
        scope_page_uid: str | None = None
        if len(selected) > 1 and self._selected_entry is not None:
            scope_page_uid = self._selected_entry.page_uid
        skipped_offpage = 0
        grouped: dict[str, dict[tuple[str, str], dict[int, str]]] = defaultdict(dict)
        for entry in selected:
            if scope_page_uid is not None and entry.page_uid != scope_page_uid:
                skipped_offpage += 1
                continue
            key = (entry.proof_uid, entry.text_unit_uid)
            unit = self._units.get(key)
            if unit is None or entry.char_index >= len(unit.text):
                continue
            changes = grouped[entry.proof_uid].setdefault(key, {})
            changes[entry.char_index] = text
        emitted = 0
        for proof_uid, unit_changes in grouped.items():
            state = self._states[proof_uid]
            replacements: list[tuple[str, str]] = []
            for key, changes in sorted(unit_changes.items(), key=lambda item: (self._units[item[0]].order, item[0][1])):
                unit = self._units[key]
                chars = list(unit.text)
                for char_index, replacement in changes.items():
                    chars[char_index] = replacement
                updated = "".join(chars)
                if updated != unit.text:
                    replacements.append((unit.text_unit_uid, updated))
            if not replacements:
                continue
            self._emit(
                ProofEditCommand(
                    proof_uid=proof_uid,
                    op="replace_many",
                    expected_revision=state.revision,
                    expected_fingerprint=state.fingerprint,
                    replacements=tuple(replacements),
                )
            )
            emitted += len(replacements)
        if skipped_offpage:
            self._set_status_message(
                f"已应用到 {emitted} 处（跨页 {skipped_offpage} 处未改）",
                "ok" if emitted else "error",
            )
        return emitted

    def _gallery_direct_overwrite(self, text: str) -> bool:
        applied = self._apply_replacement_to_selected(text)
        if applied:
            self._set_status_message(f'✓ 直输替换为 "{text}"（{applied} 处）', "ok")
            return True
        return False

    def _gallery_direct_blank(self) -> bool:
        # One gallery entry is one proof character; replacing it with a
        # single space blanks the slot while keeping the text length locked.
        applied = self._apply_replacement_to_selected(" ")
        if applied:
            self._set_status_message(f"✓ 已清空为空白（{applied} 处）", "ok")
            return True
        return False

    def _apply_history(self, direction: int) -> None:
        if direction not in {-1, 1}:
            raise ValueError("history direction must be -1 or 1")
        entry = self._selected_entry
        if entry is None:
            return
        state = self._states.get(entry.proof_uid)
        if state is None:
            return
        self._emit(
            ProofEditCommand(
                proof_uid=state.proof_uid,
                op="undo" if direction < 0 else "redo",
                expected_revision=state.revision,
                expected_fingerprint=state.fingerprint,
            )
        )

    def _set_status_message(self, text: str, tone: str = "muted") -> None:
        colors = {"ok": COLOR_LIKELY_OK, "error": COLOR_ERROR}
        self._status.setText(text)
        color = colors.get(tone)
        self._status.setStyleSheet(f"color: {color};" if color else "")

    # ─────────────────── edit bubble ───────────────────

    def _show_edit_bubble_at(self, position: QPoint) -> None:
        item = self._gallery.itemAt(position)
        if item is not None and not item.isSelected():
            self._gallery.setCurrentItem(item, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        if item is None and self._selected_entry is None:
            return
        selected_count = max(1, len(self._gallery.selectedItems()))
        self._edit_bubble_input.setText(self._selected_entry.text if self._selected_entry else "")
        self._edit_bubble_input.setPlaceholderText(f"替换 {selected_count} 处")
        global_position = self._gallery.viewport().mapToGlobal(position)
        local_position = self.mapFromGlobal(global_position)
        x = max(8, min(local_position.x() + 8, self.width() - self._edit_bubble.width() - 8))
        y = max(8, min(local_position.y() + 8, self.height() - self._edit_bubble.height() - 8))
        self._edit_bubble.move(x, y)
        self._edit_bubble.show()
        self._edit_bubble.raise_()
        self._edit_bubble_input.setFocus()
        self._edit_bubble_input.selectAll()

    def _apply_edit_bubble(self) -> None:
        text = self._edit_bubble_input.text()
        applied = self._apply_replacement_to_selected(text)
        self._edit_bubble.hide()
        if applied:
            message = f'✓ 已应用 "{text}" 到 {applied} 处' if applied > 1 else f'✓ 已应用 "{text}"'
            self._set_status_message(message, "ok")
        else:
            self._set_status_message("改字：当前页没有可替换的目标", "error")

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is getattr(self, "_edit_bubble_input", None) and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                self._edit_bubble.hide()
                return True
        if (
            watched is self._image.viewport()
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
        ):
            scene_position = self._image.mapToScene(event.pos())
            self._on_image_point_clicked(scene_position.x(), scene_position.y())
        return super().eventFilter(watched, event)

    # ─────────────────── candidates ───────────────────

    def _clear_candidate_buttons(self) -> None:
        while self._candidate_buttons_row.count():
            item = self._candidate_buttons_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._candidate_buttons = []

    def _update_candidate_panel(self, entry: ProofCharView | None) -> None:
        self._clear_candidate_buttons()
        if entry is None:
            self._candidate_panel.setToolTip("无候选")
            return
        candidates = self._ranked_candidates(entry)[:5]
        for index, candidate in enumerate(candidates):
            button = QPushButton(candidate)
            button.setObjectName("primaryBtn" if index == 0 else "candidateButton")
            button.setMinimumHeight(24)
            button.setToolTip(
                f"替换当前选中字为：{candidate}" + ("（最高可信候选）" if index == 0 else "")
            )
            button.clicked.connect(
                lambda _checked=False, value=candidate: self._apply_candidate(value)
            )
            self._candidate_buttons_row.addWidget(button)
            self._candidate_buttons.append(button)
        self._candidate_buttons_row.addStretch()
        if len(candidates) <= 1:
            reason = self._diagnose_single_candidate(entry)
            self._candidate_panel.setToolTip(
                f"仅 1 候选：暂无替代建议，不代表此字正确。{reason}"
            )
        else:
            self._candidate_panel.setToolTip("")

    def _ranked_candidates(self, entry: ProofCharView) -> list[str]:
        """Aggregate candidates from high to low trust, de-duplicated.

        Priority: the current glyph itself, the OCR source glyph at the same
        position, then first- and second-level confusables.
        """

        ranked: list[str] = []

        def push(value: str | None) -> None:
            if value and value not in ranked:
                ranked.append(value)

        push(entry.text)
        push(entry.ocr_char)
        first_level = DEFAULT_CONFUSABLE_CANDIDATES.get(entry.text, [])
        for value in first_level:
            push(value)
        for first in first_level:
            for second in DEFAULT_CONFUSABLE_CANDIDATES.get(first, []):
                push(second)
        return ranked

    def _diagnose_single_candidate(self, entry: ProofCharView) -> str:
        """Explain why the candidate sources contributed no new glyph."""

        reasons: list[str] = []
        if not entry.ocr_char:
            reasons.append("OCR 无对应字")
        elif entry.ocr_char == entry.text:
            reasons.append("OCR 与当前字一致")
        if entry.text not in DEFAULT_CONFUSABLE_CANDIDATES:
            reasons.append(f"易混淆字典无 '{entry.text}' 条目")
        if not reasons:
            return "（来源均无新字）"
        return "原因：" + " / ".join(reasons) + "。"

    def _apply_candidate(self, candidate: str) -> None:
        applied = self._apply_replacement_to_selected(candidate)
        if applied == 0:
            self._set_status_message("当前页没有可替换的目标", "error")
            return
        message = f"✓ 已批量应用候选到 {applied} 处" if applied > 1 else "✓ 已应用候选"
        self._set_status_message(message, "ok")

    # ─────────────────── rendering ───────────────────

    def _paragraph_context(
        self,
        entry: ProofCharView,
        line: ProofLineView,
    ) -> tuple[str, int]:
        """Full paragraph (same layout region) proof text + char offset.

        纵校以字符为单位，但人工判断对错需要段落语境：把同一版面 region
        的各行校对文本拼成整段，返回字符在段落中的偏移。
        """

        own_unit = self._units.get((entry.proof_uid, entry.text_unit_uid))
        if own_unit is None:
            return "-", 0
        region_ids = set(line.region_uids)
        if line.region_uid:
            region_ids.add(line.region_uid)
        if not region_ids:
            return own_unit.text or "-", entry.char_index
        peers: list[tuple[int, str, str]] = []
        for peer_line in self._lines.values():
            if peer_line.proof_uid != line.proof_uid:
                continue
            peer_regions = set(peer_line.region_uids)
            if peer_line.region_uid:
                peer_regions.add(peer_line.region_uid)
            if not (region_ids & peer_regions):
                continue
            unit = self._units.get((peer_line.proof_uid, peer_line.text_unit_uid))
            if unit is not None:
                peers.append((peer_line.order, peer_line.text_unit_uid, unit.text))
        peers.sort(key=lambda item: (item[0], item[1]))
        parts: list[str] = []
        offset = 0
        for _order, text_unit_uid, text in peers:
            if text_unit_uid == entry.text_unit_uid:
                offset = sum(len(part) + 1 for part in parts)
            parts.append(text)
        if not parts:
            return own_unit.text or "-", entry.char_index
        return "\n".join(parts), offset + entry.char_index

    def _render_entry(self, entry: ProofCharView | None) -> None:
        if entry is None:
            self._ocr_context.clear()
            self._ocr_context.setExtraSelections([])
            self._image.clear()
            self._viewer_page_uid = None
            return
        page = self._pages.get(entry.page_uid)
        unit = self._units.get((entry.proof_uid, entry.text_unit_uid))
        line = self._lines.get((entry.proof_uid, entry.text_unit_uid))
        if page is None or unit is None or line is None:
            self._render_entry(None)
            return
        # 上下文为整段校对文本（同版面 region），按段落偏移高亮对应字
        context_text, highlight_offset = self._paragraph_context(entry, line)
        self._ocr_context.setPlainText(context_text)
        self._ocr_context.setExtraSelections([])
        if 0 <= highlight_offset < len(context_text):
            cursor = self._ocr_context.textCursor()
            cursor.setPosition(highlight_offset)
            cursor.movePosition(
                QTextCursor.MoveOperation.Right,
                QTextCursor.MoveMode.KeepAnchor,
                max(1, len(entry.text)),
            )
            selection = QTextEdit.ExtraSelection()
            selection.cursor = cursor
            selection.format.setBackground(QColor("#FFE28A"))
            selection.format.setForeground(QColor("#1F2937"))
            self._ocr_context.setExtraSelections([selection])
            view_cursor = self._ocr_context.textCursor()
            view_cursor.setPosition(highlight_offset)
            self._ocr_context.setTextCursor(view_cursor)
            self._ocr_context.ensureCursorVisible()
        self._update_viewer(entry, page)

    def _update_viewer(
        self,
        entry: ProofCharView,
        page: ProofPageView,
    ) -> None:
        # 纵校以字符为单位：只画字符框（细线轻填充微外扩，不压字、无行高亮）
        pixmap = self._page_source_pixmap(page)
        if pixmap.isNull():
            self._image.clear()
            self._viewer_page_uid = None
            return
        if self._viewer_page_uid != entry.page_uid:
            self._image.set_image_from_qimage(pixmap.toImage())
            self._viewer_page_uid = entry.page_uid
        x_scale = pixmap.width() / page.width if page.width > 0 else 1.0
        y_scale = pixmap.height() / page.height if page.height > 0 else 1.0

        def to_scene(bbox: tuple[int, int, int, int]) -> _SceneBBox:
            left, top, right, bottom = bbox
            return _SceneBBox(
                round(left * x_scale),
                round(top * y_scale),
                max(1, round((right - left) * x_scale)),
                max(1, round((bottom - top) * y_scale)),
            )

        if entry.bbox is not None:
            self._image.highlight_bbox(
                to_scene(entry.bbox),
                zoom=True,
                pen_width=2,
                fill_alpha=24,
                padding=2,
            )

    def _on_image_point_clicked(self, scene_x: float, scene_y: float) -> bool:
        """Reverse-locate the text context from a click inside the page image."""

        page_uid = self._viewer_page_uid
        if page_uid is None:
            return False
        page = self._pages.get(page_uid)
        pixmap = self._page_pixmaps.get(page_uid)
        if page is None or pixmap is None or pixmap.isNull():
            return False
        x_scale = pixmap.width() / page.width if page.width > 0 else 1.0
        y_scale = pixmap.height() / page.height if page.height > 0 else 1.0
        if x_scale <= 0 or y_scale <= 0:
            return False
        point_x = scene_x / x_scale
        point_y = scene_y / y_scale
        best: tuple[int, ProofLineView] | None = None
        for line in self._lines.values():
            if line.page_uid != page_uid or line.bbox is None:
                continue
            left, top, right, bottom = line.bbox
            if left <= point_x <= right and top <= point_y <= bottom:
                area = (right - left) * (bottom - top)
                if best is None or area < best[0]:
                    best = (area, line)
        if best is None:
            return False
        line = best[1]
        for row in range(self._gallery.count()):
            entry = self._gallery.item(row).data(Qt.ItemDataRole.UserRole)
            if (
                isinstance(entry, ProofCharView)
                and (entry.proof_uid, entry.text_unit_uid) == (line.proof_uid, line.text_unit_uid)
            ):
                self._gallery.setCurrentRow(row, QItemSelectionModel.SelectionFlag.ClearAndSelect)
                return True
        for entry in self._entries:
            if (entry.proof_uid, entry.text_unit_uid) == (line.proof_uid, line.text_unit_uid):
                # The clicked line is outside the current gallery filter;
                # still surface its context and image evidence.
                self._render_entry(entry)
                return True
        return False


__all__ = ["VProofPanel"]
