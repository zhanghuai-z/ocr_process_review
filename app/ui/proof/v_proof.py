"""Vertical proof and same-text index view over an immutable workspace."""
from __future__ import annotations

from collections import Counter, defaultdict

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPixmap, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
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
from app.ui.proof.char_verdict import classify_char
from app.ui.proof.confidence_view import ProofCharView, build_char_views, char_confidence


IMAGE_SIZE = QSize(620, 420)


def _entry_key(entry: ProofCharView) -> tuple[str, str, int, str | None]:
    return (entry.proof_uid, entry.text_unit_uid, entry.char_index, entry.atom_uid)


def _page_pixmap(
    page: ProofPageView,
    bbox: tuple[int, int, int, int] | None,
    target_size: QSize | None = None,
) -> QPixmap:
    """Return a character crop, never a full-page thumbnail."""

    if bbox is None:
        return QPixmap()
    source = QPixmap(page.image_path)
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
    ).intersected(source.rect())
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


def _occurrence_offset(
    line: ProofLineView,
    page: ProofPageView,
    entry: ProofCharView,
) -> int | None:
    """Find the active OCR offset represented by one immutable char view."""

    offset = 0
    for candidate in build_char_views(line, page):
        if _entry_key(candidate) == _entry_key(entry):
            return offset if offset < len(line.ocr_text) else None
        if candidate.ocr_char is not None:
            offset += len(candidate.ocr_char)
    return None


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
        self._selected_page_uid: str | None = None
        self._selected_char = ""
        self._selected_entry: ProofCharView | None = None
        self._build_ui()
        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._redo_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self._redo_y_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        for shortcut in (self._undo_shortcut, self._redo_shortcut, self._redo_y_shortcut):
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._undo_shortcut.activated.connect(lambda: self._apply_history(-1))
        self._redo_shortcut.activated.connect(lambda: self._apply_history(1))
        self._redo_y_shortcut.activated.connect(lambda: self._apply_history(1))
        if workspace is not None:
            self.set_workspace(workspace)

    @property
    def workspace(self) -> ProofWorkspaceView | None:
        return self._workspace

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
        self._char_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._char_list.currentItemChanged.connect(self._on_char_changed)
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
        self._page_select = QComboBox()
        self._page_select.currentIndexChanged.connect(self._on_page_changed)
        gallery_header.addWidget(self._page_select)
        gallery_layout.addLayout(gallery_header)
        self._gallery = QListWidget()
        self._gallery.setObjectName("proofGallery")
        self._gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self._gallery.setFlow(QListWidget.Flow.LeftToRight)
        self._gallery.setWrapping(True)
        self._gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._gallery.setMovement(QListWidget.Movement.Static)
        self._gallery.setIconSize(QSize(62, 62))
        self._gallery.setSpacing(6)
        self._gallery.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._gallery.currentItemChanged.connect(self._on_gallery_changed)
        gallery_layout.addWidget(self._gallery, 1)
        center_layout.addWidget(gallery_card, 2)

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
        self._evidence = QLabel()
        self._evidence.setWordWrap(True)
        center_layout.addWidget(self._evidence)

        viewer = QFrame()
        viewer.setObjectName("proofRightPane")
        viewer_layout = QVBoxLayout(viewer)
        viewer_layout.setContentsMargins(12, 12, 12, 12)
        viewer_title = QLabel("原稿上下文")
        viewer_title.setObjectName("sectionTitle")
        viewer_layout.addWidget(viewer_title)
        self._image = QLabel("无可用原稿图像")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumHeight(180)
        self._image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        viewer_layout.addWidget(self._image, 1)

        center.setMinimumWidth(480)
        viewer.setMinimumWidth(300)
        self._content_splitter.addWidget(center)
        self._content_splitter.addWidget(viewer)
        self._content_splitter.setStretchFactor(0, 1)
        self._content_splitter.setStretchFactor(1, 1)
        self._content_splitter.setSizes([620, 540])
        content_layout.addWidget(self._content_splitter)
        main_splitter.addWidget(content)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([210, 1100])
        root.addWidget(main_splitter, 1)
        self._gallery.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._gallery.customContextMenuRequested.connect(self._show_edit_bubble_at)
        self._build_edit_bubble()

    def _build_edit_bubble(self) -> None:
        self._edit_bubble = QFrame(self)
        self._edit_bubble.setObjectName("vproofEditBubble")
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

    def set_workspace(self, workspace: ProofWorkspaceView | None) -> None:
        """Replace the displayed immutable snapshot."""

        if workspace is not None and not isinstance(workspace, ProofWorkspaceView):
            raise TypeError("VProofPanel requires ProofWorkspaceView or None")
        selected = self._selected_page_uid
        self._workspace = workspace
        self._pages = {page.page_uid: page for page in workspace.pages} if workspace else {}
        self._states = {state.proof_uid: state for state in workspace.proof_states} if workspace else {}
        self._lines.clear()
        self._units.clear()
        entries: list[ProofCharView] = []
        if workspace is not None:
            for state in workspace.proof_states:
                units = {unit.text_unit_uid: unit for unit in state.text_units}
                page = self._pages.get(state.page_uid)
                if page is None:
                    raise ValueError(f"proof state {state.proof_uid!r} has no page view")
                for line in state.lines:
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
        page_uids = set(self._pages)
        self._selected_page_uid = selected if selected in page_uids else None
        self._selected_char = ""
        self._selected_entry = None
        self._populate_page_selector()
        self._rebuild_char_list()
        self._status.setText(
            "暂无可校对字符" if not self._entries else f"{len(self._entries)} 个字符"
        )

    def clear_workspace(self) -> None:
        self.set_workspace(None)

    def refresh_view(self) -> None:
        """Rebuild local index structures from the existing immutable view."""

        self.set_workspace(self._workspace)

    def _populate_page_selector(self) -> None:
        self._page_select.blockSignals(True)
        self._page_select.clear()
        self._page_select.addItem("全部页面", "")
        selected_index = 0
        for page in sorted(self._pages.values(), key=lambda item: (item.page_number, item.page_uid)):
            self._page_select.addItem(f"第 {page.page_number} 页", page.page_uid)
            if page.page_uid == self._selected_page_uid:
                selected_index = self._page_select.count() - 1
        self._page_select.setCurrentIndex(selected_index)
        self._page_select.blockSignals(False)

    def _on_page_changed(self, index: int) -> None:
        value = self._page_select.itemData(index)
        self._selected_page_uid = value or None
        self._rebuild_char_list()

    def _filtered_entries(self) -> tuple[ProofCharView, ...]:
        if self._selected_page_uid is None:
            return self._entries
        return tuple(entry for entry in self._entries if entry.page_uid == self._selected_page_uid)

    def _rebuild_char_list(self) -> None:
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
        if self._char_list.count():
            self._char_list.setCurrentRow(0)
            self._char_list.blockSignals(False)
            self._on_char_changed(self._char_list.item(0), None)
        else:
            self._char_list.blockSignals(False)
            self._selected_char = ""
            self._set_gallery(())

    def _on_char_changed(self, current: QListWidgetItem | None, _previous) -> None:
        self._selected_char = current.data(Qt.ItemDataRole.UserRole) if current else ""
        self._set_gallery(self._entries_for_text(self._selected_char))

    def _entries_for_text(self, text: str) -> tuple[ProofCharView, ...]:
        entries = self._entries_by_text.get(text, ())
        if self._selected_page_uid is None:
            return entries
        return tuple(entry for entry in entries if entry.page_uid == self._selected_page_uid)

    def _set_gallery(self, entries: tuple[ProofCharView, ...]) -> None:
        self._gallery.blockSignals(True)
        self._gallery.clear()
        self._gallery_header.setText(
            "相同字索引" if not self._selected_char else f"相同字索引 · {self._selected_char}"
        )
        for entry in entries:
            item = QListWidgetItem(self._entry_icon(entry), f"第 {entry.page_number} 页")
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setToolTip(
                f"{entry.proof_uid}/{entry.text_unit_uid} · {entry.char_index}"
            )
            self._gallery.addItem(item)
        if self._gallery.count():
            self._gallery.setCurrentRow(0)
            first = self._gallery.item(0)
            entry = first.data(Qt.ItemDataRole.UserRole)
            self._selected_entry = entry if isinstance(entry, ProofCharView) else None
        else:
            self._selected_entry = None
        self._gallery.blockSignals(False)
        self._render_entry(self._selected_entry)

    def _entry_icon(self, entry: ProofCharView) -> QIcon:
        page = self._pages.get(entry.page_uid)
        if page is None:
            return QIcon()
        pixmap = _page_pixmap(page, entry.bbox, QSize(62, 62))
        return QIcon(pixmap) if not pixmap.isNull() else QIcon()

    def _on_gallery_changed(
        self,
        current: QListWidgetItem | None,
        _previous,
    ) -> None:
        entry = current.data(Qt.ItemDataRole.UserRole) if current else None
        self._selected_entry = entry if isinstance(entry, ProofCharView) else None
        self._render_entry(self._selected_entry)

    def _selected_entries(self) -> tuple[ProofCharView, ...]:
        values: list[ProofCharView] = []
        for item in self._gallery.selectedItems():
            entry = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(entry, ProofCharView):
                values.append(entry)
        if not values and self._selected_entry is not None:
            values.append(self._selected_entry)
        return tuple(values)

    def _emit(self, command: ProofEditCommand) -> None:
        self.proof_edit_requested.emit(command)

    def _apply_replacement_to_selected(self, text: str) -> int:
        selected = self._selected_entries()
        grouped: dict[str, dict[tuple[str, str], dict[int, str]]] = defaultdict(dict)
        for entry in selected:
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
        return emitted

    def _gallery_direct_overwrite(self, text: str) -> bool:
        return self._apply_replacement_to_selected(text) > 0

    def _gallery_direct_blank(self) -> bool:
        return self._apply_replacement_to_selected("") > 0

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

    def _show_edit_bubble_at(self, position: QPoint) -> None:
        item = self._gallery.itemAt(position)
        if item is None:
            return
        self._gallery.setCurrentItem(item)
        self._edit_bubble_input.setText(self._selected_entry.text if self._selected_entry else "")
        global_position = self._gallery.viewport().mapToGlobal(position)
        local_position = self.mapFromGlobal(global_position)
        self._edit_bubble.move(local_position.x(), local_position.y())
        self._edit_bubble.show()
        self._edit_bubble.raise_()
        self._edit_bubble_input.setFocus()
        self._edit_bubble_input.selectAll()

    def _apply_edit_bubble(self) -> None:
        text = self._edit_bubble_input.text()
        self._apply_replacement_to_selected(text)
        self._edit_bubble.hide()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self._edit_bubble_input and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                self._edit_bubble.hide()
                return True
        return super().eventFilter(watched, event)

    def _render_entry(self, entry: ProofCharView | None) -> None:
        if entry is None:
            self._ocr_context.clear()
            self._ocr_context.setExtraSelections([])
            self._image.setText("无可用原稿图像")
            self._image.setPixmap(QPixmap())
            self._evidence.clear()
            return
        page = self._pages.get(entry.page_uid)
        unit = self._units.get((entry.proof_uid, entry.text_unit_uid))
        line = self._lines.get((entry.proof_uid, entry.text_unit_uid))
        if page is None or unit is None or line is None:
            self._render_entry(None)
            return
        verdict = classify_char(
            confidence=char_confidence(entry),
            text_char=entry.text,
            ocr_char=entry.ocr_char,
        )
        ocr_text = line.ocr_text or "-"
        self._ocr_context.setPlainText(ocr_text)
        self._ocr_context.setExtraSelections([])
        occurrence_offset = _occurrence_offset(line, page, entry)
        if occurrence_offset is not None and occurrence_offset < len(line.ocr_text):
            cursor = self._ocr_context.textCursor()
            cursor.setPosition(occurrence_offset)
            cursor.movePosition(
                QTextCursor.MoveOperation.Right,
                QTextCursor.MoveMode.KeepAnchor,
                max(1, len(entry.ocr_char or "")),
            )
            selection = QTextEdit.ExtraSelection()
            selection.cursor = cursor
            selection.format.setBackground(QColor("#FFE28A"))
            selection.format.setForeground(QColor("#1F2937"))
            self._ocr_context.setExtraSelections([selection])
            view_cursor = self._ocr_context.textCursor()
            view_cursor.setPosition(occurrence_offset)
            self._ocr_context.setTextCursor(view_cursor)
            self._ocr_context.ensureCursorVisible()
        self._evidence.setText(
            f"{verdict.severity}: {verdict.evidence} · 校对字符 {entry.text} · "
            f"区域 {entry.region_uid or '-'} · 字符 {entry.atom_uid or '-'}"
        )
        self._evidence.setStyleSheet(f"color: {verdict.color};")
        pixmap = _page_pixmap(page, entry.bbox, self._image.size())
        self._image.setPixmap(pixmap)
        self._image.setText("" if not pixmap.isNull() else "无可用原稿图像")

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._selected_entry is not None:
            page = self._pages.get(self._selected_entry.page_uid)
            if page is not None:
                pixmap = _page_pixmap(page, self._selected_entry.bbox, self._image.size())
                self._image.setPixmap(pixmap)


__all__ = ["VProofPanel"]
