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
- book-wide multi-select edits are emitted as one atomic application command;
- selection survives workspace snapshot replacements: occurrences are
  re-located by their stable occurrence keys.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass

from PySide6.QtCore import QEvent, QItemSelectionModel, QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QIcon,
    QImage,
    QImageReader,
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

from app.application.contracts import ProofBatchEditCommand, ProofEditCommand
from app.application.proof_candidates import has_confusable_entry, ranked_candidates
from app.application.proof_workspace import (
    ProofLineView,
    ProofPageView,
    ProofStateView,
    ProofTextUnitView,
    ProofWorkspacePatch,
    ProofWorkspaceView,
    apply_proof_workspace_patch,
)
from app.ui.proof.char_verdict import COLOR_ERROR, COLOR_LIKELY_OK
from app.ui.proof.confidence_view import ProofCharView, build_char_views
from app.ui.image_orientation import (
    rotate_bbox,
    rotate_image,
    rotate_pixmap,
    rotated_size,
    source_point_from_display,
)
from app.ui.widgets.effects import apply_soft_shadow
from app.ui.widgets.image_viewer import ImageViewer


IMAGE_SIZE = QSize(620, 420)
_ICON_PENDING_ROLE = Qt.ItemDataRole.UserRole + 2
# Gallery cells mirror the mature delegate: a 56px thumbnail with breathing
# room inside a 70px grid, six items per row and at most three visible rows.
GALLERY_THUMB = 56
GALLERY_CELL = GALLERY_THUMB + 14
GALLERY_ITEMS_PER_ROW = 6
LOW_CONF = 0.80

@dataclass(frozen=True, slots=True)
class _SceneBBox:
    """Duck-typed ``x/y/w/h`` box for :class:`ImageViewer` overlay calls."""

    x: int
    y: int
    w: int
    h: int


def _entry_key(entry: ProofCharView) -> tuple[str, str, int, int, str | None]:
    return (
        entry.proof_uid,
        entry.text_unit_uid,
        entry.char_index,
        entry.char_end,
        entry.atom_uid,
    )


def _char_index_group(text: str) -> int:
    """Order character buckets as CJK, letters, digits, then punctuation."""

    if len(text) == 1 and _is_cjk_char(text):
        return 0
    if text.isalpha():
        return 1
    if text.isdigit():
        return 2
    return 3


def _is_cjk_char(char: str) -> bool:
    if len(char) != 1:
        return False
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x323AF
    )


def _gallery_crop_pad(entry: ProofCharView) -> int:
    """Return bounded context padding without pulling in adjacent glyphs."""

    if entry.bbox is None:
        return 0
    left, top, right, bottom = entry.bbox
    text = entry.text or ""
    visible = tuple(char for char in text if not char.isspace())
    if visible and all(_is_cjk_char(char) for char in visible):
        return 0
    if len(text) == 1 and text.isascii() and not text.isspace():
        return 1 if text.isalnum() else 2
    shortest_edge = max(1, min(right - left, bottom - top))
    return max(2, min(6, round(shortest_edge * 0.10)))


def _page_pixmap(
    page: ProofPageView,
    bbox: tuple[int, int, int, int] | None,
    target_size: QSize | None = None,
    *,
    pad: int = 0,
) -> QPixmap:
    """Return a crop through the same scaled-page transform as the viewer."""

    if bbox is None:
        return QPixmap()
    reader = QImageReader(page.image_path)
    source_size = reader.size()
    if (
        source_size.width() <= 0
        or source_size.height() <= 0
        or page.width <= 0
        or page.height <= 0
    ):
        return QPixmap()
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(page.width, right + pad)
    bottom = min(page.height, bottom + pad)
    if right <= left or bottom <= top:
        return QPixmap()
    x_scale = source_size.width() / page.width
    y_scale = source_size.height() / page.height
    source_rect = QRect(
        max(0, round(left * x_scale)),
        max(0, round(top * y_scale)),
        max(1, round((right - left) * x_scale)),
        max(1, round((bottom - top) * y_scale)),
    ).intersected(QRect(0, 0, source_size.width(), source_size.height()))
    if source_rect.isEmpty():
        return QPixmap()
    # Decode the original-resolution bbox before scaling.  Downscaling a full
    # 600-DPI page first collapses punctuation into two or three pixels.
    reader.setClipRect(source_rect)
    image = reader.read()
    if image.isNull():
        return QPixmap()
    image = rotate_image(image, page.display_rotation_quarters_clockwise)
    target = target_size if target_size is not None and not target_size.isEmpty() else IMAGE_SIZE
    return QPixmap.fromImage(image).scaled(
        max(1, target.width()),
        max(1, target.height()),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _crop_page_pixmap(
    page: ProofPageView,
    page_pixmap: QPixmap,
    bbox: tuple[int, int, int, int] | None,
    target_size: QSize | None = None,
    *,
    pad: int = 0,
) -> QPixmap:
    """Crop one bbox from a page pixmap using page-coordinate geometry."""

    if bbox is None or page_pixmap.isNull() or page.width <= 0 or page.height <= 0:
        return QPixmap()
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(page.width, right + pad)
    bottom = min(page.height, bottom + pad)
    x_scale = page_pixmap.width() / page.width
    y_scale = page_pixmap.height() / page.height
    crop_rect = QRect(
        round(left * x_scale),
        round(top * y_scale),
        max(1, round((right - left) * x_scale)),
        max(1, round((bottom - top) * y_scale)),
    ).intersected(page_pixmap.rect())
    if crop_rect.isEmpty():
        return QPixmap()
    target = target_size if target_size is not None and not target_size.isEmpty() else IMAGE_SIZE
    crop = rotate_pixmap(
        page_pixmap.copy(crop_rect),
        page.display_rotation_quarters_clockwise,
    )
    return crop.scaled(
        max(1, target.width()),
        max(1, target.height()),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _crop_page_image(
    page: ProofPageView,
    page_image: QImage,
    bbox: tuple[int, int, int, int] | None,
    target_size: QSize,
    *,
    pad: int = 0,
) -> QPixmap:
    """Crop one original page decode without reopening the image file."""

    if bbox is None or page_image.isNull() or page.width <= 0 or page.height <= 0:
        return QPixmap()
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(page.width, right + pad)
    bottom = min(page.height, bottom + pad)
    source_rect = QRect(
        round(left * page_image.width() / page.width),
        round(top * page_image.height() / page.height),
        max(1, round((right - left) * page_image.width() / page.width)),
        max(1, round((bottom - top) * page_image.height() / page.height)),
    ).intersected(page_image.rect())
    if source_rect.isEmpty():
        return QPixmap()
    crop = rotate_image(
        page_image.copy(source_rect),
        page.display_rotation_quarters_clockwise,
    )
    return QPixmap.fromImage(crop).scaled(
        max(1, target_size.width()),
        max(1, target_size.height()),
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
        self._page_pixmap_keys: dict[str, tuple[str, int, int]] = {}
        self._icon_cache: OrderedDict[tuple[object, ...], QIcon] = OrderedDict()
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
        left_title = QLabel("字符/词索引")
        left_title.setObjectName("sectionTitle")
        left_header.addWidget(left_title)
        left_header.addStretch(1)
        self._char_count = QLabel("0 项")
        self._char_count.setObjectName("muted")
        left_header.addWidget(self._char_count)
        left_layout.addLayout(left_header)
        self._char_search = QLineEdit()
        self._char_search.setPlaceholderText("搜索字符或词…")
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
        self._gallery_header = QLabel("相同字符/词索引")
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
        self._gallery.verticalScrollBar().valueChanged.connect(
            self._load_visible_gallery_icons
        )
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
        scoped("Ctrl+Alt+Right", self._gallery, lambda: self._step_gallery(1))
        scoped("Ctrl+Alt+Left", self._gallery, lambda: self._step_gallery(-1))
        scoped("Shift+Alt+Right", self._gallery, lambda: self._toggle_adjacent_gallery(1))
        scoped("Shift+Alt+Left", self._gallery, lambda: self._toggle_adjacent_gallery(-1))
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
        self._install_entries(entries, restore_tokens=restore_tokens, restore_keys=restore_keys)

    def _install_entries(
        self,
        entries: list[ProofCharView] | tuple[ProofCharView, ...],
        *,
        restore_tokens: tuple[str, ...] | list[str] = (),
        restore_keys: tuple[tuple[str, str, int, int, str | None], ...] = (),
    ) -> None:
        self._entries = tuple(sorted(
            entries,
            key=lambda item: (
                item.page_number,
                item.proof_uid,
                item.text_unit_uid,
                item.char_index,
            ),
        ))
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
            self._status.setText(f"{len(self._entries)} 项")
            self._status.setStyleSheet("")

    def apply_workspace_patch(self, patch: ProofWorkspacePatch) -> None:
        """Re-index only the proof text units named by a committed patch."""

        if self._workspace is None:
            raise RuntimeError("cannot apply a proof patch without a workspace")
        restore_tokens = self._selected_tokens_for_restore()
        restore_keys = tuple(_entry_key(entry) for entry in self._selected_entries())
        self._workspace = apply_proof_workspace_patch(self._workspace, patch)
        self._states = {state.proof_uid: state for state in self._workspace.proof_states}
        changed_keys = {
            (state.proof_uid, unit.text_unit_uid)
            for state in patch.states
            for unit in state.text_units
        }
        entries = [
            entry
            for entry in self._entries
            if (entry.proof_uid, entry.text_unit_uid) not in changed_keys
        ]
        for key in changed_keys:
            self._lines.pop(key, None)
            self._units.pop(key, None)
            state = self._states[key[0]]
            unit = next(item for item in state.text_units if item.text_unit_uid == key[1])
            line = next(item for item in state.lines if item.text_unit_uid == key[1])
            if line.render_kind != "text":
                continue
            page = self._pages.get(state.page_uid)
            if page is None:
                raise ValueError(f"proof state {state.proof_uid!r} has no page view")
            self._lines[key] = line
            self._units[key] = unit
            entries.extend(build_char_views(line, page))
        self._install_entries(entries, restore_tokens=restore_tokens, restore_keys=restore_keys)

    def _prune_page_pixmap_cache(self) -> None:
        """Drop cached page pixmaps whose page vanished or whose image changed."""

        valid = {
            page.page_uid: (
                page.image_path,
                page.image_revision,
                page.display_rotation_quarters_clockwise,
            )
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
        restore_keys: tuple[tuple[str, str, int, int, str | None], ...] = (),
    ) -> None:
        query = self._char_search.text()
        available = self._filtered_entries()
        counts = Counter(entry.text for entry in available if not query or query in entry.text)
        self._char_list.blockSignals(True)
        self._char_list.clear()
        for text, count in sorted(
            counts.items(),
            key=lambda item: (_char_index_group(item[0]), -item[1], item[0].casefold(), item[0]),
        ):
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
        restore_keys: tuple[tuple[str, str, int, int, str | None], ...] = (),
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
        header = f"相同字符/词索引 · {tokens[0]}（共 {len(entries)} 处）"
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
        restore_keys: tuple[tuple[str, str, int, int, str | None], ...] = (),
    ) -> None:
        self._gallery.blockSignals(True)
        self._gallery.clear()
        aspect = 1.0
        word_boxes = [
            entry.bbox
            for entry in entries
            if entry.char_end > entry.char_index + 1 and entry.bbox is not None
        ]
        if word_boxes:
            aspect = min(
                3.5,
                max(
                    1.0,
                    max(
                        (right - left) / max(1, bottom - top)
                        for left, top, right, bottom in word_boxes
                    ),
                ),
            )
        thumb_width = round(GALLERY_THUMB * aspect)
        gallery_size = QSize(thumb_width + 14, GALLERY_CELL)
        self._gallery.setGridSize(gallery_size)
        self._gallery.setIconSize(QSize(thumb_width, GALLERY_THUMB))
        if header is not None:
            self._gallery_header.setText(header)
        else:
            self._gallery_header.setText(
                "相同字符/词索引"
                if not self._selected_char
                else f"相同字符/词索引 · {self._selected_char}"
            )
        for entry in entries:
            # 图标懒加载：先占位，仅可视范围 ±1 屏的条目真正解码切图
            item = QListWidgetItem("")
            item.setSizeHint(gallery_size)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setData(_ICON_PENDING_ROLE, True)
            geometry_note = "" if entry.available else " · 无精确字符框"
            position_note = (
                f"位 #{entry.char_index + 1}"
                if entry.char_end == entry.char_index + 1
                else f"位 #{entry.char_index + 1}-{entry.char_end}"
            )
            item.setToolTip(
                f"第 {entry.page_number} 页 · {entry.proof_uid}/{entry.text_unit_uid}"
                f" · {position_note}{geometry_note}"
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
        QTimer.singleShot(0, self._load_visible_gallery_icons)

    def _load_visible_gallery_icons(self) -> None:
        """Decode gallery icons only for items near the visible viewport."""

        viewport_rect = self._gallery.viewport().rect()
        padded = viewport_rect.adjusted(
            0,
            -viewport_rect.height(),
            0,
            viewport_rect.height(),
        )
        pending_by_page: dict[str, list[tuple[QListWidgetItem, ProofCharView]]] = defaultdict(list)
        for row in range(self._gallery.count()):
            item = self._gallery.item(row)
            if not item.data(_ICON_PENDING_ROLE):
                continue
            if not self._gallery.visualItemRect(item).intersects(padded):
                continue
            entry = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(entry, ProofCharView):
                pending_by_page[entry.page_uid].append((item, entry))
        for page_uid, pending in pending_by_page.items():
            page = self._pages.get(page_uid)
            page_image = QImageReader(page.image_path).read() if page is not None else QImage()
            for item, entry in pending:
                item.setIcon(self._entry_icon(entry, page_image=page_image))
                item.setData(_ICON_PENDING_ROLE, False)

    def _restore_gallery_selection_by_occurrence_keys(
        self,
        occurrence_keys: tuple[tuple[str, str, int, int, str | None], ...],
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

    def _entry_icon(self, entry: ProofCharView, *, page_image: QImage | None = None) -> QIcon:
        page = self._pages.get(entry.page_uid)
        if page is None or entry.bbox is None:
            return QIcon()
        pad = _gallery_crop_pad(entry)
        cache_key = (
            page.page_uid,
            page.image_path,
            page.image_revision,
            page.display_rotation_quarters_clockwise,
            entry.bbox,
            pad,
            self._gallery.iconSize().width(),
            self._gallery.iconSize().height(),
        )
        cached = self._icon_cache.get(cache_key)
        if cached is not None:
            self._icon_cache.move_to_end(cache_key)
            return cached
        # 2x source density: Qt only ever down-scales the thumbnail.
        canvas_width = max(1, self._gallery.iconSize().width() * 2)
        canvas_height = max(1, self._gallery.iconSize().height() * 2)
        canvas = QPixmap(canvas_width, canvas_height)
        canvas.fill(QColor("#FFFDF8"))
        painter = QPainter(canvas)
        target = QSize(max(1, canvas_width - 16), max(1, canvas_height - 16))
        crop = (
            _crop_page_image(page, page_image, entry.bbox, target, pad=pad)
            if page_image is not None and not page_image.isNull()
            else _page_pixmap(page, entry.bbox, target, pad=pad)
        )
        if crop.isNull():
            painter.end()
            return QIcon()
        painter.drawPixmap(
            (canvas_width - crop.width()) // 2,
            (canvas_height - crop.height()) // 2,
            crop,
        )
        # 索引项只展示 observation crop，不叠加边框或置信度标记。
        painter.end()
        icon = QIcon(canvas)
        self._icon_cache[cache_key] = icon
        while len(self._icon_cache) > 512:
            self._icon_cache.popitem(last=False)
        return icon

    def _remember_page_pixmap(self, page: ProofPageView, pixmap: QPixmap) -> None:
        self._page_pixmaps[page.page_uid] = pixmap
        self._page_pixmap_keys[page.page_uid] = (
            page.image_path,
            page.image_revision,
            page.display_rotation_quarters_clockwise,
        )

    def _page_source_pixmap(self, page: ProofPageView) -> QPixmap:
        cached = self._page_pixmaps.get(page.page_uid)
        if cached is not None and not cached.isNull():
            return cached
        # 缩放解码（≤2000px 宽）：查看原稿不需要 600 DPI 全尺寸解码
        reader = QImageReader(page.image_path)
        source_size = reader.size()
        if source_size.width() > 2000:
            reader.setScaledSize(
                QSize(2000, round(source_size.height() * 2000 / source_size.width()))
            )
        image = reader.read()
        image = rotate_image(image, page.display_rotation_quarters_clockwise)
        pixmap = QPixmap.fromImage(image) if not image.isNull() else QPixmap()
        if pixmap.isNull():
            return pixmap
        # 小型 LRU：保留最近 2 页（当前 + 上一页），跨页切换不反复解码，
        # 高分辨率多页项目也不长期持有全页 QPixmap
        self._remember_page_pixmap(page, pixmap)
        while len(self._page_pixmaps) > 2:
            oldest = next(iter(self._page_pixmaps))
            self._page_pixmaps.pop(oldest, None)
            self._page_pixmap_keys.pop(oldest, None)
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

    def _toggle_adjacent_gallery(self, delta: int) -> None:
        """Toggle the adjacent occurrence while preserving the current item."""

        count = self._gallery.count()
        if count <= 0:
            return
        row = self._gallery.currentRow()
        row = 0 if row < 0 else (row + delta) % count
        item = self._gallery.item(row)
        item.setSelected(not item.isSelected())

    def _select_all_gallery(self) -> None:
        self._gallery.selectAll()

    def _clear_gallery_selection(self) -> None:
        """Esc collapses the gallery multi-select back to the current item."""

        current = self._gallery.currentItem()
        self._gallery.clearSelection()
        if current is not None:
            current.setSelected(True)

    # ─────────────────── edit commands ───────────────────

    def _emit(self, command: ProofEditCommand | ProofBatchEditCommand) -> None:
        self.proof_edit_requested.emit(command)

    def _apply_replacement_to_selected(self, text: str, *, pad_to_span: bool = False) -> int:
        # 字符/词索引覆盖全项目：批量修改作用于所有选中项，跨页替换按
        # proof state 分组为各自的 replace_many 命令表达，不做 UI 裁剪
        selected = self._selected_entries()
        grouped: dict[
            str,
            dict[tuple[str, str], dict[tuple[int, int], str]],
        ] = defaultdict(dict)
        for entry in selected:
            key = (entry.proof_uid, entry.text_unit_uid)
            unit = self._units.get(key)
            if unit is None or entry.char_end > len(unit.text):
                continue
            changes = grouped[entry.proof_uid].setdefault(key, {})
            span_width = entry.char_end - entry.char_index
            replacement = text
            if pad_to_span:
                replacement = (text or "")[:span_width].ljust(span_width)
            changes[(entry.char_index, entry.char_end)] = replacement
        commands: list[ProofEditCommand] = []
        emitted = 0
        for proof_uid, unit_changes in grouped.items():
            state = self._states[proof_uid]
            replacements: list[tuple[str, str]] = []
            ordered_changes = sorted(
                unit_changes.items(),
                key=lambda item: (self._units[item[0]].order, item[0][1]),
            )
            for key, changes in ordered_changes:
                unit = self._units[key]
                updated = unit.text
                for (start, end), replacement in sorted(
                    changes.items(),
                    key=lambda item: item[0],
                    reverse=True,
                ):
                    updated = updated[:start] + replacement + updated[end:]
                if updated != unit.text:
                    replacements.append((unit.text_unit_uid, updated))
            if not replacements:
                continue
            commands.append(
                ProofEditCommand(
                    proof_uid=proof_uid,
                    op="replace_many",
                    expected_revision=state.revision,
                    expected_fingerprint=state.fingerprint,
                    replacements=tuple(replacements),
                )
            )
            emitted += len(replacements)
        if len(commands) == 1:
            self._emit(commands[0])
        elif commands:
            self._emit(ProofBatchEditCommand(commands=tuple(commands)))
        return emitted

    def _gallery_direct_overwrite(self, text: str) -> bool:
        applied = self._apply_replacement_to_selected(text)
        if applied:
            self._set_status_message(f'已提交直输替换为 "{text}"（{applied} 处）', "ok")
            return True
        return False

    def _gallery_direct_blank(self) -> bool:
        # Blank the complete occurrence while preserving its current span.
        applied = self._apply_replacement_to_selected(" ", pad_to_span=True)
        if applied:
            self._set_status_message(f"已提交清空为空白（{applied} 处）", "ok")
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
        return list(ranked_candidates(entry.text, entry.ocr_char))

    def _diagnose_single_candidate(self, entry: ProofCharView) -> str:
        """Explain why the candidate sources contributed no new glyph."""

        reasons: list[str] = []
        if not entry.ocr_char:
            reasons.append("OCR 无对应字")
        elif entry.ocr_char == entry.text:
            reasons.append("OCR 与当前字一致")
        if not has_confusable_entry(entry.text):
            reasons.append(f"易混淆字典无 '{entry.text}' 条目")
        if not reasons:
            return "（来源均无新字）"
        return "原因：" + " / ".join(reasons) + "。"

    def _apply_candidate(self, candidate: str) -> None:
        applied = self._apply_replacement_to_selected(candidate)
        if applied == 0:
            self._set_status_message("当前选区没有可替换的目标", "error")
            return
        message = f"已提交批量候选替换（{applied} 处）" if applied > 1 else "已提交候选替换"
        self._set_status_message(message, "ok")

    # ─────────────────── rendering ───────────────────

    def _page_context(
        self,
        entry: ProofCharView,
        line: ProofLineView,
    ) -> tuple[str, int]:
        """Full page proof text grouped by layout region + char offset."""

        own_unit = self._units.get((entry.proof_uid, entry.text_unit_uid))
        if own_unit is None:
            return "-", 0
        peers: list[tuple[int, str, str, tuple[str, ...]]] = []
        for peer_line in self._lines.values():
            if peer_line.proof_uid != line.proof_uid:
                continue
            unit = self._units.get((peer_line.proof_uid, peer_line.text_unit_uid))
            if unit is not None:
                region_uids = peer_line.region_uids or (
                    (peer_line.region_uid,) if peer_line.region_uid else ()
                )
                peers.append(
                    (peer_line.order, peer_line.text_unit_uid, unit.text, region_uids)
                )
        peers.sort(key=lambda item: (item[0], item[1]))
        text = ""
        offset = 0
        previous_regions: tuple[str, ...] | None = None
        for _order, text_unit_uid, unit_text, region_uids in peers:
            if text:
                text += "\n" if region_uids == previous_regions else "\n\n"
            if text_unit_uid == entry.text_unit_uid:
                offset = len(text)
            text += unit_text
            previous_regions = region_uids
        if not text:
            return own_unit.text or "-", entry.char_index
        return text, offset + entry.char_index

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
        # 页面只做弱聚合：全文按版面段落分隔，当前 occurrence 仍由稳定索引定位。
        context_text, highlight_offset = self._page_context(entry, line)
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
        # 只画当前字符或 word observation 框，不扩张成行高亮。
        pixmap = self._page_source_pixmap(page)
        if pixmap.isNull():
            self._image.clear()
            self._viewer_page_uid = None
            return
        if self._viewer_page_uid != entry.page_uid:
            self._image.set_image_from_qimage(pixmap.toImage())
            self._viewer_page_uid = entry.page_uid
        display_width, display_height = rotated_size(
            page.width,
            page.height,
            page.display_rotation_quarters_clockwise,
        )
        x_scale = pixmap.width() / display_width if display_width > 0 else 1.0
        y_scale = pixmap.height() / display_height if display_height > 0 else 1.0

        def to_scene(bbox: tuple[int, int, int, int]) -> _SceneBBox:
            bbox = rotate_bbox(
                bbox,
                page.width,
                page.height,
                page.display_rotation_quarters_clockwise,
            )
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
        display_width, display_height = rotated_size(
            page.width,
            page.height,
            page.display_rotation_quarters_clockwise,
        )
        x_scale = pixmap.width() / display_width if display_width > 0 else 1.0
        y_scale = pixmap.height() / display_height if display_height > 0 else 1.0
        if x_scale <= 0 or y_scale <= 0:
            return False
        display_x = scene_x / x_scale
        display_y = scene_y / y_scale
        point_x, point_y = source_point_from_display(
            display_x,
            display_y,
            page.width,
            page.height,
            page.display_rotation_quarters_clockwise,
        )
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
