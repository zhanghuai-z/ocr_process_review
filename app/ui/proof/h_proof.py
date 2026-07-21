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
    return pixmap.scaled(
        ROW_IMAGE_SIZE,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


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
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(4)

        header = QHBoxLayout()
        self._title = QLabel(f"第 {row.page.page_number} 页 · 第 {row.unit.order + 1} 行")
        self._title.setObjectName("sectionTitle")
        header.addWidget(self._title, 1)
        self._status = QLabel()
        header.addWidget(self._status)
        self.confirm_button = QPushButton("确认")
        self.confirm_button.setObjectName("secondaryBtn")
        self.confirm_button.setFixedHeight(24)
        self.confirm_button.clicked.connect(self.confirm_requested)
        header.addWidget(self.confirm_button)
        root.addLayout(header)

        body = QHBoxLayout()
        self._image = QLabel()
        self._image.setFixedSize(ROW_IMAGE_SIZE)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setText("无可用行图像")
        pixmap = _image_for_row(row.page, row.bbox)
        if not pixmap.isNull():
            self._image.setPixmap(pixmap)
            self._image.setText("")
        body.addWidget(self._image)

        text_column = QVBoxLayout()
        self._observation = QLabel(f"OCR 原文：{row.ocr_text or '无有效识别行'}")
        self._observation.setObjectName("muted")
        self._observation.setWordWrap(True)
        text_column.addWidget(self._observation)
        self.editor = _CommitTextEdit()
        self.editor.setPlainText(row.unit.text)
        self.editor.setMinimumHeight(42)
        self.editor.setMaximumHeight(74)
        self.editor.setPlaceholderText("校对文本")
        self.editor.commit_requested.connect(self.commit_requested)
        self.editor.cancel_requested.connect(self.cancel_requested)
        self.editor.navigate_requested.connect(self.navigate_requested)
        text_column.addWidget(self.editor, 1)
        body.addLayout(text_column, 1)
        root.addLayout(body)
        self.set_status(row.unit.status)

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
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)

        self._toolbar = QFrame()
        self._toolbar.setObjectName("proofToolbar")
        toolbar = QHBoxLayout(self._toolbar)
        toolbar.setContentsMargins(8, 4, 8, 4)
        toolbar.setSpacing(6)
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
        self._page_select = QComboBox()
        self._page_select.currentIndexChanged.connect(self._on_page_changed)
        toolbar.addWidget(self._page_select, 1)
        root.addWidget(self._toolbar)

        self._status = QLabel("暂无可校对内容")
        self._status.setObjectName("proofStatusBar")
        root.addWidget(self._status)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("proofScroll")
        self._scroll.setWidgetResizable(True)
        self._rows_root = QWidget()
        self._rows_root.setObjectName("proofLineList")
        self._rows_layout = QVBoxLayout(self._rows_root)
        self._rows_layout.setContentsMargins(6, 4, 6, 6)
        self._rows_layout.setSpacing(6)
        self._rows_layout.addStretch(1)
        self._scroll.setWidget(self._rows_root)
        root.addWidget(self._scroll, 1)

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
        self._page_select.blockSignals(True)
        self._page_select.clear()
        self._page_select.addItem("全部页面", "")
        selected_index = 0
        if self._session is not None:
            for page in sorted(
                self._session.page_repository.all(),
                key=lambda item: (item.page_number, item.uid),
            ):
                self._page_select.addItem(f"第 {page.page_number} 页", page.uid)
                if page.uid == self._selected_page_uid:
                    selected_index = self._page_select.count() - 1
        self._page_select.setCurrentIndex(selected_index)
        self._page_select.blockSignals(False)

    def _on_page_changed(self, index: int) -> None:
        value = self._page_select.itemData(index)
        self._selected_page_uid = str(value) if value else None
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
