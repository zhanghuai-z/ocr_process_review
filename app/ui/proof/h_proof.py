"""Session-backed horizontal proof view.

The panel is a projection over one ``ProjectSession``. OCR observations supply
the image, source text, geometry, and confidence; ``ProofState`` supplies the
editable text and status. Every write is delegated to ``ProofSessionService``
with the editor's revision and fingerprint tokens.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from PySide6.QtCore import QEvent, QRect, QSize, Qt, Signal
from PySide6.QtGui import QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.core.char_index import CharIndexEntry
from app.core.proof_session import ProofEditorSnapshot, ProofSessionResult
from app.models.ocr_records import OcrAtom, OcrLine
from app.models.proof_records import ProofState, ProofTextUnit
from app.models.project_session import PageRecord, ProjectSession, RevisionConflictError
from app.services.proof_session_service import ProofSessionError, ProofSessionService
from app.ui.proof.confidence_utils import (
    ProofContext,
    build_proof_contexts,
    char_confidence,
    line_confidence,
)
from app.ui.widgets.page_directory import PageDirectoryList


ROW_IMAGE_SIZE = QSize(300, 54)
STATUS_COLORS = {
    "checked": "#4E7A63",
    "ok": "#4E7A63",
    "modified": "#5C6B58",
    "flagged": "#C67B22",
    "unchecked": "#8090A0",
}

STATUS_LABELS = {
    "unchecked": "待校对",
    "checked": "已校对",
    "modified": "已修改",
    "conflict": "有冲突",
}


@dataclass(frozen=True, slots=True)
class _ProofRow:
    key: tuple[str, str]
    page: PageRecord
    state: ProofState
    unit: ProofTextUnit
    snapshot: ProofEditorSnapshot
    entries: tuple[CharIndexEntry, ...]
    line: OcrLine | None
    atom: OcrAtom | None
    region_uid: str
    ocr_text: str
    bbox: tuple[int, int, int, int] | None
    confidence: float | None


def _row_for_unit(context: ProofContext, unit: ProofTextUnit, service: ProofSessionService) -> _ProofRow:
    entries = tuple(
        entry for entry in context.index.entries if entry.text_unit_uid == unit.uid
    )
    lines = context.lines_by_uid
    atoms = context.atoms_by_uid
    line = next(
        (lines[entry.line_uid] for entry in entries if entry.line_uid in lines),
        None,
    )
    atom = next(
        (atoms[entry.atom_uid] for entry in entries if entry.atom_uid in atoms),
        None,
    )
    snapshot = service.editor_snapshot(context.state.uid, unit.uid)
    bbox = line.bbox if line is not None else next(
        (entry.bbox for entry in entries if entry.bbox is not None),
        None,
    )
    return _ProofRow(
        key=(context.state.uid, unit.uid),
        page=context.page,
        state=context.state,
        unit=unit,
        snapshot=snapshot,
        entries=entries,
        line=line,
        atom=atom,
        region_uid=atom.region_uid if atom is not None else (line.region_uid if line else ""),
        ocr_text=line.text if line is not None else "",
        bbox=bbox,
        confidence=char_confidence(atom, line) if atom is not None else line_confidence(line, ()) if line else None,
    )


class _CommitTextEdit(QPlainTextEdit):
    """Small editor with proof navigation commands owned by the view."""

    commit_requested = Signal()
    cancel_requested = Signal()
    navigate_requested = Signal(int)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_S:
                self.commit_requested.emit()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Up:
                self.navigate_requested.emit(-1)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Down:
                self.navigate_requested.emit(1)
                event.accept()
                return
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and not (
            modifiers & Qt.KeyboardModifier.ShiftModifier
        ):
            self.commit_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self.cancel_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


def _image_for_row(page: PageRecord, bbox: tuple[int, int, int, int] | None) -> QPixmap:
    pixmap = QPixmap(page.image_path)
    if pixmap.isNull():
        return QPixmap()
    if bbox is not None:
        left, top, right, bottom = bbox
        rect = QRect(left, top, max(1, right - left), max(1, bottom - top))
        rect = rect.intersected(pixmap.rect())
        if not rect.isEmpty():
            pixmap = pixmap.copy(rect)
    return pixmap


class _ProofRowWidget(QFrame):
    commit_requested = Signal()
    cancel_requested = Signal()
    navigate_requested = Signal(int)
    confirm_requested = Signal()

    def __init__(self, row: _ProofRow, parent=None) -> None:
        super().__init__(parent)
        self.row = row
        self.setObjectName("linePair")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(132)
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 4, 8, 4)
        root.setSpacing(6)

        self._active_bar = QWidget()
        self._active_bar.setObjectName("proofRowActiveBar")
        self._active_bar.setFixedWidth(4)
        root.addWidget(self._active_bar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        header = QHBoxLayout()
        self._title = QLabel(f"第 {row.page.page_number} 页 · 第 {row.unit.order + 1} 行")
        self._title.setObjectName("muted")
        header.addWidget(self._title, 1)
        self._status = QLabel()
        header.addWidget(self._status)
        content_layout.addLayout(header)

        self._image = QLabel()
        self._image.setObjectName("proofLineImage")
        self._image.setFixedHeight(58)
        self._image.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._image.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._image.setText("无可用行图像")
        self._line_crop = _image_for_row(row.page, row.bbox)
        content_layout.addWidget(self._image)

        self.editor = _CommitTextEdit()
        self.editor.setObjectName("proofLineEditor")
        self.editor.setPlainText(row.unit.text)
        self.editor.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.editor.setFixedHeight(46)
        self.editor.document().setDocumentMargin(0)
        self.editor.setPlaceholderText("校对文本")
        self.editor.commit_requested.connect(self.commit_requested)
        self.editor.cancel_requested.connect(self.cancel_requested)
        self.editor.navigate_requested.connect(self.navigate_requested)
        content_layout.addWidget(self.editor)
        root.addWidget(content, 1)

        self.confirm_button = QPushButton("确认")
        self.confirm_button.setObjectName("secondaryBtn")
        self.confirm_button.setFixedSize(64, 28)
        self.confirm_button.clicked.connect(self.confirm_requested)
        root.addWidget(self.confirm_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.set_status(row.unit.status)
        self._refresh_image()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_image()

    def _refresh_image(self) -> None:
        if self._line_crop.isNull():
            self._image.setPixmap(QPixmap())
            self._image.setText("无可用行图像")
            return
        target = self._image.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        self._image.setPixmap(self._line_crop.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        self._image.setText("")

    def set_status(self, status: str) -> None:
        color = STATUS_COLORS.get(status, STATUS_COLORS["unchecked"])
        self._status.setText(STATUS_LABELS.get(status, status))
        self._status.setStyleSheet(f"color: {color}; font-weight: 600;")


class HProofPanel(QWidget):
    """Horizontal proof editor over one session-scoped proof aggregate."""

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
        self._rows: tuple[_ProofRow, ...] = ()
        self._visible_rows: tuple[_ProofRow, ...] = ()
        self._row_widgets: dict[tuple[str, str], _ProofRowWidget] = {}
        self._dirty_text: dict[tuple[str, str], str] = {}
        self._selected_page_uid: str | None = None
        self._status_message = ""
        self._build_ui()
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

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("hproofSplitter")
        self._splitter.setChildrenCollapsible(False)

        left = QFrame()
        left.setObjectName("proofLeftPane")
        left.setMinimumWidth(210)
        left.setMaximumWidth(270)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)
        directory_title = QLabel("页面")
        directory_title.setObjectName("sectionTitle")
        left_layout.addWidget(directory_title)
        self._page_directory = PageDirectoryList()
        self._page_directory.page_selected.connect(self._on_page_directory_selected)
        left_layout.addWidget(self._page_directory, 1)
        self._splitter.addWidget(left)

        center = QWidget()
        center.setObjectName("proofCenterPane")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(10, 10, 10, 10)
        center_layout.setSpacing(0)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("proofScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._rows_root = QWidget()
        self._rows_root.setObjectName("proofLineList")
        self._rows_layout = QVBoxLayout(self._rows_root)
        self._rows_layout.setContentsMargins(6, 4, 6, 6)
        self._rows_layout.setSpacing(3)
        self._rows_layout.addStretch(1)
        self._scroll.setWidget(self._rows_root)
        center_layout.addWidget(self._scroll, 1)
        self._splitter.addWidget(center)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([236, 1000])
        root.addWidget(self._splitter, 1)

        self._status_bar = QWidget()
        self._status_bar.setObjectName("proofStatusBar")
        self._status_bar.setFixedHeight(44)
        toolbar = QHBoxLayout(self._status_bar)
        toolbar.setContentsMargins(12, 3, 12, 3)
        toolbar.setSpacing(8)
        self._btn_prev = QPushButton("上一行")
        self._btn_next = QPushButton("下一行")
        self._btn_save = QPushButton("保存")
        self._btn_refresh = QPushButton("刷新")
        self._btn_prev.setObjectName("ghostBtn")
        self._btn_next.setObjectName("ghostBtn")
        self._btn_save.setObjectName("primaryBtn")
        self._btn_refresh.setObjectName("ghostBtn")
        self._btn_prev.clicked.connect(lambda: self._focus_row(-1))
        self._btn_next.clicked.connect(lambda: self._focus_row(1))
        self._btn_save.clicked.connect(self.save)
        self._btn_refresh.clicked.connect(self.refresh_from_session)
        toolbar.addWidget(self._btn_prev)
        toolbar.addWidget(self._btn_next)
        toolbar.addWidget(self._btn_save)
        toolbar.addWidget(self._btn_refresh)
        self._status = QLabel("暂无可校对内容")
        self._status.setObjectName("muted")
        toolbar.addSpacing(8)
        toolbar.addWidget(self._status)
        toolbar.addStretch(1)
        self._scope = QLabel("全部页面")
        self._scope.setObjectName("proofStatusStrong")
        toolbar.addWidget(self._scope)
        root.addWidget(self._status_bar)

    def _set_session(
        self,
        session: ProjectSession,
        proof_service: ProofSessionService | None,
    ) -> None:
        if not isinstance(session, ProjectSession):
            raise TypeError("HProofPanel requires ProjectSession")
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
        selected_page_number: int | None = None,
    ) -> None:
        self._set_session(session, proof_service)
        self._selected_page_uid = selected_page_uid
        if selected_page_number is not None:
            self._selected_page_uid = next(
                (
                    page.uid
                    for page in session.page_repository.all()
                    if page.page_number == selected_page_number
                ),
                selected_page_uid,
            )
        self._populate_page_selector()
        self.refresh_from_session()

    def set_session(
        self,
        session: ProjectSession,
        proof_service: ProofSessionService | None = None,
    ) -> None:
        """Bind the view to an existing project session and proof service."""

        self.load_session(session, proof_service)

    def load_pages(
        self,
        session: ProjectSession,
        *,
        selected_page_number: int | None = None,
    ) -> None:
        """Load a migrated session; legacy page collections are not accepted."""

        self.load_session(
            session,
            selected_page_number=selected_page_number,
        )

    def merge_pages(self, session: ProjectSession) -> None:
        """Refresh the existing session projection without importing UI state."""

        selected = self._selected_page_uid
        self.load_session(session, selected_page_uid=selected)

    def clear_session(self) -> None:
        self._session = None
        self._proof_service = None
        self._contexts = ()
        self._rows = ()
        self._visible_rows = ()
        self._dirty_text.clear()
        self._selected_page_uid = None
        self._populate_page_selector()
        self._render_rows()
        self._status.setText("暂无可校对内容")

    def _populate_page_selector(self) -> None:
        pages = () if self._session is None else tuple(sorted(
            self._session.page_repository.all(),
            key=lambda item: (item.page_number, item.uid),
        ))
        self._page_directory.set_pages(pages)
        if self._selected_page_uid and any(
            page.uid == self._selected_page_uid for page in pages
        ):
            self._page_directory.set_current_uid(self._selected_page_uid)

    def _on_page_directory_selected(self, page_uid: str) -> None:
        self._selected_page_uid = page_uid
        page = next(
            (item for item in self._session.page_repository.all() if item.uid == page_uid),
            None,
        ) if self._session is not None else None
        self._scope.setText(f"第 {page.page_number} 页" if page is not None else "当前页面")
        self._render_rows()

    def refresh_from_session(self) -> None:
        if self._session is None or self._proof_service is None:
            self._contexts = ()
            self._rows = ()
            self._visible_rows = ()
            self._render_rows()
            self._status.setText("暂无可校对内容")
            return
        try:
            contexts = build_proof_contexts(self._session, self._proof_service)
        except (KeyError, ValueError, RuntimeError) as exc:
            self._contexts = ()
            self._rows = ()
            self._visible_rows = ()
            self._render_rows()
            self._status.setText(f"无法加载校对内容：{exc}")
            return
        rows = [
            _row_for_unit(context, unit, self._proof_service)
            for context in contexts
            for unit in context.state.text_units
        ]
        self._contexts = contexts
        self._rows = tuple(
            sorted(rows, key=lambda row: (row.page.page_number, row.state.uid, row.unit.order, row.unit.uid))
        )
        self._dirty_text.clear()
        self._render_rows()
        self._status.setText(
            f"{len(self._rows)} 行 · "
            f"{sum(len(row.unit.text) for row in self._rows)} 字符"
        )

    def _render_rows(self) -> None:
        self._visible_rows = tuple(
            row
            for row in self._rows
            if self._selected_page_uid is None or row.page.uid == self._selected_page_uid
        )
        self._row_widgets.clear()
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for row in self._visible_rows:
            widget = _ProofRowWidget(row, self._rows_root)
            widget.editor.textChanged.connect(
                lambda row=row, widget=widget: self._on_text_changed(row, widget)
            )
            widget.commit_requested.connect(lambda row=row, widget=widget: self._commit_row(row, widget))
            widget.cancel_requested.connect(lambda row=row, widget=widget: self._cancel_row(row, widget))
            widget.navigate_requested.connect(lambda delta, row=row: self._navigate_from(row, delta))
            widget.confirm_requested.connect(lambda row=row, widget=widget: self._confirm_row(row, widget))
            self._rows_layout.addWidget(widget)
            self._row_widgets[row.key] = widget
        self._rows_layout.addStretch(1)

    def _on_text_changed(self, row: _ProofRow, widget: _ProofRowWidget) -> None:
        text = widget.editor.toPlainText()
        if text == row.unit.text:
            self._dirty_text.pop(row.key, None)
        else:
            self._dirty_text[row.key] = text

    def _commit_row(self, row: _ProofRow, widget: _ProofRowWidget) -> bool:
        text = widget.editor.toPlainText()
        if text == row.unit.text:
            self._dirty_text.pop(row.key, None)
            return False
        service = self._proof_service
        if service is None:
            return False
        try:
            result = service.replace_text(
                row.state.uid,
                row.unit.uid,
                text,
                expected_revision=row.snapshot.state_revision,
                expected_fingerprint=row.snapshot.state_fingerprint,
                expected_unit_revision=row.snapshot.text_unit_revision,
                expected_unit_fingerprint=row.snapshot.text_unit_fingerprint,
            )
        except (RevisionConflictError, ProofSessionError, ValueError) as exc:
            self._status.setText(f"编辑冲突：{exc}")
            return False
        self._dirty_text.pop(row.key, None)
        self._publish_result(result)
        self.refresh_from_session()
        return result.changed

    def _save_all(self) -> bool:
        if not self._dirty_text or self._proof_service is None:
            return True
        rows_by_key = {row.key: row for row in self._visible_rows}
        grouped: dict[str, list[_ProofRow]] = defaultdict(list)
        for key in self._dirty_text:
            row = rows_by_key.get(key)
            if row is not None:
                grouped[row.state.uid].append(row)
        changed = False
        for state_uid, rows in grouped.items():
            state = rows[0].state
            replacements = {
                row.unit.uid: self._dirty_text[row.key] for row in rows
            }
            try:
                result = self._proof_service.replace_text_units(
                    state_uid,
                    replacements,
                    expected_revision=state.revision,
                    expected_fingerprint=state.fingerprint,
                    expected_unit_revisions={
                        row.unit.uid: row.snapshot.text_unit_revision for row in rows
                    },
                    expected_unit_fingerprints={
                        row.unit.uid: row.snapshot.text_unit_fingerprint for row in rows
                    },
                )
            except (RevisionConflictError, ProofSessionError, ValueError) as exc:
                self._status.setText(f"保存冲突：{exc}")
                return False
            changed = changed or result.changed
            self._publish_result(result)
        self._dirty_text.clear()
        if changed:
            self.refresh_from_session()
        return True

    def save(self) -> bool:
        return self._save_all()

    def _publish_result(self, result: ProofSessionResult) -> None:
        if result.changed:
            self.proof_changed.emit(result)

    def _find_row(self, key: tuple[str, str]) -> _ProofRow | None:
        return next((row for row in self._rows if row.key == key), None)

    def _confirm_row(self, row: _ProofRow, widget: _ProofRowWidget) -> None:
        self._commit_row(row, widget)
        current = self._find_row(row.key)
        if current is None or self._proof_service is None:
            return
        try:
            result = self._proof_service.set_status(
                current.state.uid,
                current.unit.uid,
                "checked",
                expected_revision=current.snapshot.state_revision,
                expected_fingerprint=current.snapshot.state_fingerprint,
                expected_unit_revision=current.snapshot.text_unit_revision,
                expected_unit_fingerprint=current.snapshot.text_unit_fingerprint,
            )
        except (RevisionConflictError, ProofSessionError, ValueError) as exc:
            self._status.setText(f"状态冲突：{exc}")
            return
        self._publish_result(result)
        self.refresh_from_session()

    def _cancel_row(self, row: _ProofRow, widget: _ProofRowWidget) -> None:
        self._dirty_text.pop(row.key, None)
        widget.editor.blockSignals(True)
        widget.editor.setPlainText(row.unit.text)
        widget.editor.blockSignals(False)

    def _navigate_from(self, row: _ProofRow, delta: int) -> None:
        self._commit_row(row, self._row_widgets[row.key])
        self._focus_row(delta, current_key=row.key)

    def _focus_row(self, delta: int, *, current_key: tuple[str, str] | None = None) -> None:
        if not self._visible_rows:
            return
        key = current_key
        if key is None:
            focused = next(
                (item.row.key for item in self._row_widgets.values() if item.editor.hasFocus()),
                self._visible_rows[0].key,
            )
            key = focused
        current_index = next(
            (index for index, row in enumerate(self._visible_rows) if row.key == key),
            0,
        )
        target = max(0, min(len(self._visible_rows) - 1, current_index + delta))
        widget = self._row_widgets.get(self._visible_rows[target].key)
        if widget is not None:
            widget.editor.setFocus(Qt.FocusReason.OtherFocusReason)
            widget.editor.selectAll()

    def closeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        self._save_all()
        super().closeEvent(event)


__all__ = ["HProofPanel"]
