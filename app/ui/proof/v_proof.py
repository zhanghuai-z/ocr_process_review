"""Session-backed vertical proof and same-text index view."""
from __future__ import annotations

from collections import Counter, defaultdict

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
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
    QVBoxLayout,
    QWidget,
    QApplication,
)

from app.core.char_index import CharIndex, CharIndexEntry
from app.core.proof_session import ProofSessionResult
from app.models.ocr_records import OcrAtom, OcrLine
from app.models.proof_records import ProofState, ProofTextUnit
from app.models.project_session import PageRecord, ProjectSession, RevisionConflictError
from app.services.proof_session_service import ProofSessionError, ProofSessionService
from app.ui.proof.char_verdict import classify_char
from app.ui.proof.confidence_utils import (
    ProofContext,
    build_proof_contexts,
    char_confidence,
)


IMAGE_SIZE = (620, 420)


def _entry_key(entry: CharIndexEntry) -> tuple[str, str, int, str | None]:
    return (entry.proof_uid, entry.text_unit_uid, entry.char_index, entry.atom_uid)


def _page_pixmap(
    page: PageRecord,
    bbox: tuple[int, int, int, int] | None,
) -> QPixmap:
    pixmap = QPixmap(page.image_path)
    if pixmap.isNull():
        return QPixmap()
    if bbox is not None and page.width > 0 and page.height > 0:
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#D43D3D"), 3))
        left, top, right, bottom = bbox
        x_scale = pixmap.width() / page.width
        y_scale = pixmap.height() / page.height
        painter.drawRect(
            QRect(
                round(left * x_scale),
                round(top * y_scale),
                max(1, round((right - left) * x_scale)),
                max(1, round((bottom - top) * y_scale)),
            )
        )
        painter.end()
    return pixmap.scaled(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class VProofPanel(QWidget):
    """Vertical proof view over the active OCR observations and proof states."""

    proof_changed = Signal(object)

    def __init__(
        self,
        session: ProjectSession | None = None,
        proof_service: ProofSessionService | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session: ProjectSession | None = None
        self._proof_service: ProofSessionService | None = None
        self._contexts: tuple[ProofContext, ...] = ()
        self._pages: dict[str, PageRecord] = {}
        self._states: dict[str, ProofState] = {}
        self._indexes: dict[str, CharIndex] = {}
        self._entries: tuple[CharIndexEntry, ...] = ()
        self._entries_by_text: dict[str, tuple[CharIndexEntry, ...]] = {}
        self._selected_page_uid: str | None = None
        self._selected_char = ""
        self._selected_entry: CharIndexEntry | None = None
        self._build_ui()
        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._redo_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self._redo_y_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        for shortcut in (self._undo_shortcut, self._redo_shortcut, self._redo_y_shortcut):
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._undo_shortcut.activated.connect(lambda: self._apply_history(-1))
        self._redo_shortcut.activated.connect(lambda: self._apply_history(1))
        self._redo_y_shortcut.activated.connect(lambda: self._apply_history(1))
        if session is not None:
            self.load_session(session, proof_service)

    @property
    def session(self) -> ProjectSession | None:
        return self._session

    @property
    def proof_service(self) -> ProofSessionService | None:
        return self._proof_service

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
        self._btn_refresh.clicked.connect(self.refresh_from_session)
        gallery_header.addWidget(self._btn_refresh)
        self._page_select = QComboBox()
        self._page_select.currentIndexChanged.connect(self._on_page_changed)
        gallery_header.addWidget(self._page_select)
        gallery_layout.addLayout(gallery_header)
        self._gallery = QListWidget()
        self._gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self._gallery.setFlow(QListWidget.Flow.LeftToRight)
        self._gallery.setWrapping(True)
        self._gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._gallery.setMovement(QListWidget.Movement.Static)
        self._gallery.setIconSize(QSize(62, 62))
        self._gallery.setSpacing(6)
        self._gallery.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._gallery.currentItemChanged.connect(self._on_gallery_changed)
        gallery_layout.addWidget(self._gallery, 1)
        center_layout.addWidget(gallery_card, 2)

        context_card = QFrame()
        context_card.setObjectName("proofCard")
        context_layout = QVBoxLayout(context_card)
        context_layout.setContentsMargins(10, 8, 10, 10)
        context_title = QLabel("文本上下文")
        context_title.setObjectName("sectionTitle")
        context_layout.addWidget(context_title)
        self._ocr_context = QPlainTextEdit()
        self._ocr_context.setReadOnly(True)
        self._ocr_context.setPlaceholderText("OCR 文本上下文")
        context_layout.addWidget(self._ocr_context, 1)
        self._proof_context = QPlainTextEdit()
        self._proof_context.setReadOnly(True)
        self._proof_context.setPlaceholderText("校对文本上下文")
        context_layout.addWidget(self._proof_context, 1)
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
        self._image.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
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

    def _set_session(
        self,
        session: ProjectSession,
        proof_service: ProofSessionService | None,
    ) -> None:
        if not isinstance(session, ProjectSession):
            raise TypeError("VProofPanel requires ProjectSession")
        service = proof_service or ProofSessionService(session)
        if service.project_uid != session.project_uid:
            raise ValueError("proof service and project session must share a project UID")
        self._session = session
        self._proof_service = service

    def load_session(
        self,
        session: ProjectSession,
        proof_service: ProofSessionService | None = None,
        *,
        selected_page_uid: str | None = None,
    ) -> None:
        self._set_session(session, proof_service)
        self._selected_page_uid = selected_page_uid
        self._populate_page_selector()
        self.refresh_from_session()

    def set_session(
        self,
        session: ProjectSession,
        proof_service: ProofSessionService | None = None,
    ) -> None:
        """Bind the view to an existing project session and proof service."""

        self.load_session(session, proof_service)

    def load_pages(self, session: ProjectSession) -> None:
        """Load a migrated session; legacy page collections are not accepted."""

        self.load_session(session)

    def merge_pages(self, session: ProjectSession) -> None:
        selected = self._selected_page_uid
        self.load_session(session, selected_page_uid=selected)

    def clear_session(self) -> None:
        self._session = None
        self._proof_service = None
        self._contexts = ()
        self._pages.clear()
        self._states.clear()
        self._indexes.clear()
        self._entries = ()
        self._entries_by_text.clear()
        self._selected_page_uid = None
        self._selected_char = ""
        self._selected_entry = None
        self._populate_page_selector()
        self._rebuild_char_list()
        self._render_entry(None)
        self._status.setText("暂无可校对字符")

    def _populate_page_selector(self) -> None:
        self._page_select.blockSignals(True)
        self._page_select.clear()
        self._page_select.addItem("全部页面", "")
        selected_index = 0
        if self._session is not None:
            pages = sorted(
                self._session.page_repository.all(),
                key=lambda item: (item.page_number, item.uid),
            )
            for page in pages:
                self._page_select.addItem(f"第 {page.page_number} 页", page.uid)
                if page.uid == self._selected_page_uid:
                    selected_index = self._page_select.count() - 1
        self._page_select.setCurrentIndex(selected_index)
        self._page_select.blockSignals(False)

    def _on_page_changed(self, index: int) -> None:
        value = self._page_select.itemData(index)
        self._selected_page_uid = str(value) if value else None
        self._rebuild_char_list()

    def _filtered_entries(self) -> tuple[CharIndexEntry, ...]:
        if self._selected_page_uid is None:
            return self._entries
        return tuple(
            entry for entry in self._entries if entry.page_uid == self._selected_page_uid
        )

    def refresh_from_session(self) -> None:
        if self._session is None or self._proof_service is None:
            self.clear_session()
            return
        try:
            contexts = build_proof_contexts(self._session, self._proof_service)
        except (KeyError, ValueError, RuntimeError) as exc:
            self._contexts = ()
            self._entries = ()
            self._indexes.clear()
            self._rebuild_char_list()
            self._status.setText(f"无法加载校对内容：{exc}")
            return
        self._contexts = contexts
        self._pages = {context.page.uid: context.page for context in contexts}
        self._states = {context.state.uid: context.state for context in contexts}
        self._indexes = {context.state.uid: context.index for context in contexts}
        self._entries = tuple(
            entry
            for context in contexts
            for entry in context.index.entries
        )
        grouped: dict[str, list[CharIndexEntry]] = defaultdict(list)
        for entry in self._filtered_entries():
            grouped[entry.text].append(entry)
        self._entries_by_text = {
            text: tuple(items) for text, items in grouped.items()
        }
        self._rebuild_char_list()
        self._status.setText(
            f"{len(self._entries)} 个索引字符 · "
            f"{len(self._states)} 个校对页"
        )

    def _rebuild_char_list(self) -> None:
        entries = self._filtered_entries()
        counts = Counter(entry.text for entry in entries)
        query = self._char_search.text().strip().casefold()
        if query:
            counts = Counter({
                text: count for text, count in counts.items()
                if query in text.casefold()
            })
        self._char_count.setText(f"{len(counts)} 项")
        selected = self._selected_char
        self._char_list.blockSignals(True)
        self._char_list.clear()
        for text, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            item = QListWidgetItem(f"{text}  ({count})")
            item.setData(Qt.ItemDataRole.UserRole, text)
            self._char_list.addItem(item)
        self._char_list.blockSignals(False)
        target_row = next(
            (
                row
                for row in range(self._char_list.count())
                if self._char_list.item(row).data(Qt.ItemDataRole.UserRole) == selected
            ),
            0,
        )
        if self._char_list.count():
            self._char_list.setCurrentRow(target_row)
        else:
            self._selected_char = ""
            self._gallery.clear()
            self._gallery_header.setText("相同字索引")
            self._render_entry(None)

    def _on_char_changed(self, current: QListWidgetItem | None, _previous) -> None:
        if current is None:
            self._selected_char = ""
            self._gallery.clear()
            self._render_entry(None)
            return
        value = current.data(Qt.ItemDataRole.UserRole)
        self._selected_char = str(value)
        self._set_gallery(self._entries_for_text(self._selected_char))

    def _entries_for_text(self, text: str) -> tuple[CharIndexEntry, ...]:
        return tuple(entry for entry in self._filtered_entries() if entry.text == text)

    def _set_gallery(self, entries: tuple[CharIndexEntry, ...]) -> None:
        previous_key = _entry_key(self._selected_entry) if self._selected_entry else None
        self._gallery.blockSignals(True)
        self._gallery.clear()
        for entry in entries:
            geometry = "有字框" if entry.available else "无字框"
            item = QListWidgetItem(
                f"第 {entry.page_number} 页\n{geometry}"
            )
            icon = self._entry_icon(entry)
            if not icon.isNull():
                item.setIcon(icon)
            item.setSizeHint(QSize(84, 92))
            item.setToolTip(
                f"第 {entry.page_number} 页 · 行 {entry.line_uid or '-'} · "
                f"文本 {entry.text_unit_uid} · {geometry}"
            )
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self._gallery.addItem(item)
        self._gallery.blockSignals(False)
        selected_row = next(
            (
                row
                for row in range(self._gallery.count())
                if _entry_key(self._gallery.item(row).data(Qt.ItemDataRole.UserRole)) == previous_key
            ),
            0,
        )
        if self._gallery.count():
            self._gallery.setCurrentRow(selected_row)
        else:
            self._selected_entry = None
            self._render_entry(None)
        self._gallery_header.setText(
            f"{self._selected_char!r} · {self._gallery.count()} 处"
        )

    def _entry_icon(self, entry: CharIndexEntry) -> QIcon:
        page = self._pages.get(entry.page_uid)
        if page is None or entry.bbox is None:
            return QIcon()
        pixmap = QPixmap(page.image_path)
        if pixmap.isNull():
            return QIcon()
        left, top, right, bottom = entry.bbox
        rect = QRect(left, top, max(1, right - left), max(1, bottom - top))
        rect = rect.intersected(pixmap.rect())
        if rect.isEmpty():
            return QIcon()
        crop = pixmap.copy(rect).scaled(
            62,
            62,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QIcon(crop)

    def _on_gallery_changed(self, current: QListWidgetItem | None, _previous) -> None:
        if current is None:
            self._selected_entry = None
            self._render_entry(None)
            return
        entry = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(entry, CharIndexEntry):
            return
        self._selected_entry = entry
        self._render_entry(entry)

    def _selected_entries(self) -> tuple[CharIndexEntry, ...]:
        entries = tuple(
            item.data(Qt.ItemDataRole.UserRole) for item in self._gallery.selectedItems()
        )
        valid = tuple(entry for entry in entries if isinstance(entry, CharIndexEntry))
        if valid:
            return valid
        return (self._selected_entry,) if self._selected_entry is not None else ()

    def _apply_replacement_to_selected(self, text: str) -> int:
        if self._proof_service is None:
            return 0
        selected = self._selected_entries()
        if not selected:
            return 0
        grouped: dict[str, list[CharIndexEntry]] = defaultdict(list)
        for entry in selected:
            grouped[entry.proof_uid].append(entry)
        changed_count = 0
        for proof_uid, entries in grouped.items():
            try:
                state = self._proof_service.get_state(proof_uid)
                units = {unit.uid: unit for unit in state.text_units}
                replacements = {unit.uid: unit.text for unit in state.text_units}
                by_unit: dict[str, list[CharIndexEntry]] = defaultdict(list)
                for entry in entries:
                    by_unit[entry.text_unit_uid].append(entry)
                for unit_uid, unit_entries in by_unit.items():
                    if unit_uid not in units:
                        raise ValueError(f"text unit is not present: {unit_uid}")
                    updated = replacements[unit_uid]
                    for entry in sorted(
                        unit_entries,
                        key=lambda item: item.char_index,
                        reverse=True,
                    ):
                        if not 0 <= entry.char_index < len(updated):
                            raise ValueError("indexed character is stale")
                        updated = (
                            updated[: entry.char_index]
                            + text
                            + updated[entry.char_index + 1 :]
                        )
                    replacements[unit_uid] = updated
                result = self._proof_service.replace_text_units(
                    proof_uid,
                    {uid: value for uid, value in replacements.items() if value != units[uid].text},
                    expected_revision=state.revision,
                    expected_fingerprint=state.fingerprint,
                    expected_unit_revisions={
                        uid: units[uid].revision for uid in by_unit
                    },
                    expected_unit_fingerprints={
                        uid: units[uid].fingerprint for uid in by_unit
                    },
                )
            except (RevisionConflictError, ProofSessionError, ValueError) as exc:
                self._status.setText(f"编辑冲突：{exc}")
                return changed_count
            if result.changed:
                changed_count += len(entries)
                self.proof_changed.emit(result)
        if changed_count:
            self.refresh_from_session()
        return changed_count

    def _gallery_direct_overwrite(self, text: str) -> bool:
        return self._apply_replacement_to_selected(text) > 0

    def _gallery_direct_blank(self) -> bool:
        return self._apply_replacement_to_selected("") > 0

    def _apply_history(self, direction: int) -> None:
        focus = QApplication.focusWidget()
        if isinstance(focus, QLineEdit):
            available = focus.isUndoAvailable() if direction < 0 else focus.isRedoAvailable()
            if available:
                focus.undo() if direction < 0 else focus.redo()
                return
        if isinstance(focus, QPlainTextEdit):
            available = (
                focus.document().isUndoAvailable()
                if direction < 0
                else focus.document().isRedoAvailable()
            )
            if available and not focus.isReadOnly():
                focus.undo() if direction < 0 else focus.redo()
                return
        service = self._proof_service
        proof_uid = self._selected_entry.proof_uid if self._selected_entry is not None else None
        if service is None or proof_uid is None:
            return
        try:
            state = service.get_state(proof_uid)
            operation = service.undo if direction < 0 else service.redo
            result = operation(
                proof_uid,
                expected_revision=state.revision,
                expected_fingerprint=state.fingerprint,
            )
        except (RevisionConflictError, ProofSessionError, ValueError) as exc:
            self._status.setText(f"撤销冲突：{exc}")
            return
        if not result.changed:
            self._status.setText("没有可撤销的校对操作" if direction < 0 else "没有可重做的校对操作")
            return
        self.proof_changed.emit(result)
        self.refresh_from_session()

    def _show_edit_bubble_at(self, position: QPoint) -> None:
        item = self._gallery.itemAt(position)
        if item is None:
            return
        if not item.isSelected():
            self._gallery.clearSelection()
            item.setSelected(True)
        self._gallery.setCurrentItem(item)
        entry = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(entry, CharIndexEntry):
            return
        selected = self._selected_entries()
        initial_text = entry.text if len(selected) == 1 else ""
        self._edit_bubble_input.setText(initial_text)
        self._edit_bubble_input.selectAll()
        self._edit_bubble_input.setPlaceholderText(f"替换 {len(selected)} 处")
        panel_position = self._gallery.mapTo(self, position)
        x = max(8, min(panel_position.x() + 8, self.width() - self._edit_bubble.width() - 8))
        y = max(8, min(panel_position.y() + 8, self.height() - self._edit_bubble.height() - 8))
        self._edit_bubble.move(x, y)
        self._edit_bubble.show()
        self._edit_bubble.raise_()
        self._edit_bubble_input.setFocus()

    def _apply_edit_bubble(self) -> None:
        text = self._edit_bubble_input.text()
        changed = self._apply_replacement_to_selected(text)
        if not changed:
            self._status.setText("没有可提交的纵校改动")
            return
        self._edit_bubble.hide()
        self._gallery.setFocus()
        self._status.setText(f"已修改 {changed} 处")

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self._edit_bubble_input:
            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                self._edit_bubble.hide()
                return True
            if event.type() == QEvent.Type.FocusOut:
                QTimer.singleShot(0, self._edit_bubble.hide)
        return super().eventFilter(watched, event)

    def _render_entry(self, entry: CharIndexEntry | None) -> None:
        if entry is None or self._session is None:
            self._ocr_context.clear()
            self._proof_context.clear()
            self._image.setText("无可用原稿图像")
            self._image.setPixmap(QPixmap())
            self._evidence.clear()
            return
        page = self._pages.get(entry.page_uid)
        service = self._proof_service
        if page is None or service is None:
            return
        try:
            state = service.get_state(entry.proof_uid)
            unit = next(item for item in state.text_units if item.uid == entry.text_unit_uid)
        except (KeyError, ValueError, ProofSessionError) as exc:
            self._status.setText(f"无法读取选中的校对文本：{exc}")
            return
        observations = self._session.ocr_observation_repository
        observation_line: OcrLine | None = None
        atom: OcrAtom | None = None
        if entry.line_uid:
            try:
                observation_line = observations.get_line(entry.line_uid)
            except KeyError:
                observation_line = None
        if entry.atom_uid:
            try:
                atom = observations.get_atom(entry.atom_uid)
            except KeyError:
                atom = None
        ocr_char = (
            observation_line.text[entry.char_index]
            if observation_line is not None and entry.char_index < len(observation_line.text)
            else None
        )
        verdict = classify_char(
            confidence=char_confidence(atom, observation_line),
            text_char=entry.text,
            ocr_char=ocr_char,
        )
        self._ocr_context.setPlainText(
            f"OCR 行 {entry.line_uid or '-'}：{observation_line.text if observation_line else '-'}"
        )
        self._proof_context.setPlainText(f"校对文本 {unit.uid}：{unit.text}")
        self._evidence.setText(
            f"{verdict.severity}: {verdict.evidence} · "
            f"区域 {atom.region_uid if atom else '-'} · 字符 {entry.atom_uid or '-'}"
        )
        self._evidence.setStyleSheet(f"color: {verdict.color};")
        pixmap = _page_pixmap(page, entry.bbox)
        self._image.setPixmap(pixmap)
        self._image.setText("" if not pixmap.isNull() else "无可用原稿图像")

    def closeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        super().closeEvent(event)


__all__ = ["VProofPanel"]
