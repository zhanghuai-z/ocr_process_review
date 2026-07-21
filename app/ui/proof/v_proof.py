"""Session-backed vertical proof and same-text index view."""
from __future__ import annotations

from collections import Counter, defaultdict

from PySide6.QtCore import QEvent, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
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
        root.setSpacing(6)

        toolbar = QHBoxLayout()
        self._btn_refresh = QPushButton("刷新索引")
        self._btn_refresh.clicked.connect(self.refresh_from_session)
        toolbar.addWidget(self._btn_refresh)
        self._page_select = QComboBox()
        self._page_select.currentIndexChanged.connect(self._on_page_changed)
        toolbar.addWidget(self._page_select, 1)
        root.addLayout(toolbar)

        self._status = QLabel("暂无可校对字符")
        self._status.setObjectName("muted")
        root.addWidget(self._status)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(QLabel("字符索引"))
        self._char_list = QListWidget()
        self._char_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._char_list.currentItemChanged.connect(self._on_char_changed)
        left_layout.addWidget(self._char_list, 1)
        main_splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        self._gallery_header = QLabel("相同字索引")
        right_layout.addWidget(self._gallery_header)
        self._gallery = QListWidget()
        self._gallery.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._gallery.currentItemChanged.connect(self._on_gallery_changed)
        right_layout.addWidget(self._gallery, 1)

        self._ocr_context = QPlainTextEdit()
        self._ocr_context.setReadOnly(True)
        self._ocr_context.setMaximumHeight(58)
        self._ocr_context.setPlaceholderText("OCR 文本上下文")
        right_layout.addWidget(self._ocr_context)
        self._proof_context = QPlainTextEdit()
        self._proof_context.setReadOnly(True)
        self._proof_context.setMaximumHeight(58)
        self._proof_context.setPlaceholderText("校对文本上下文")
        right_layout.addWidget(self._proof_context)

        self._image = QLabel("无可用原稿图像")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumHeight(180)
        self._image.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        right_layout.addWidget(self._image, 2)

        edit_row = QHBoxLayout()
        self._edit_input = QLineEdit()
        self._edit_input.setPlaceholderText("替换选中的字符")
        self._edit_input.returnPressed.connect(self._on_apply)
        self._btn_apply = QPushButton("应用")
        self._btn_apply.clicked.connect(self._on_apply)
        edit_row.addWidget(self._edit_input, 1)
        edit_row.addWidget(self._btn_apply)
        right_layout.addLayout(edit_row)
        self._evidence = QLabel()
        self._evidence.setWordWrap(True)
        right_layout.addWidget(self._evidence)
        main_splitter.addWidget(right)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        root.addWidget(main_splitter, 1)

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

    def _on_gallery_changed(self, current: QListWidgetItem | None, _previous) -> None:
        if current is None:
            self._selected_entry = None
            self._render_entry(None)
            return
        entry = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(entry, CharIndexEntry):
            return
        self._selected_entry = entry
        self._edit_input.setText(entry.text)
        self._render_entry(entry)

    def _selected_entries(self) -> tuple[CharIndexEntry, ...]:
        entries = tuple(
            item.data(Qt.ItemDataRole.UserRole) for item in self._gallery.selectedItems()
        )
        valid = tuple(entry for entry in entries if isinstance(entry, CharIndexEntry))
        if valid:
            return valid
        return (self._selected_entry,) if self._selected_entry is not None else ()

    def _on_apply(self) -> None:
        text = self._edit_input.text()
        changed = self._apply_replacement_to_selected(text)
        if changed:
            self._status.setText(f"已修改 {changed} 处")

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
