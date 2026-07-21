"""Horizontal proof view over an immutable :class:`ProofWorkspaceView`.

The widget owns only display state and an editor's transient text buffer.  It
never reads a repository or applies a proof edit.  Every mutation request is
represented by a ``ProofEditCommand`` emitted to the application boundary.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.application.contracts import ProofEditCommand
from app.application.proof_workspace import (
    ProofAtomView,
    ProofLineView,
    ProofPageView,
    ProofStateView,
    ProofTextUnitView,
    ProofWorkspaceView,
)
from app.ui.proof.confidence_view import ProofCharView, normalize_confidence
from app.ui.proof.formula_renderer import (
    FormulaRenderResult,
    render_formula_pixmap,
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
class _AtomPlacement:
    atom: ProofAtomView
    char_indices: tuple[int, ...]


def _atom_kind(atom: ProofAtomView) -> str | None:
    return atom.render_kind if atom.render_kind in {"formula", "table"} else None


def _explicit_char_span(atom: ProofAtomView) -> tuple[int, int] | None:
    raw_span = atom.char_span
    if raw_span is None:
        return None
    if not isinstance(raw_span, (tuple, list)) or len(raw_span) != 2:
        raise TypeError("ProofAtomView.char_span must contain two integers")
    start, end = raw_span
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError("ProofAtomView.char_span must contain two integers")
    if start < 0 or end <= start:
        raise ValueError("ProofAtomView.char_span must be a non-empty half-open span")
    return start, end


def _atom_placements(line: ProofLineView) -> tuple[_AtomPlacement, ...]:
    placements: list[_AtomPlacement] = []
    text_length = len(line.proof_text)
    for atom in sorted(line.atoms, key=lambda item: (item.atom_index, item.atom_uid)):
        span = _explicit_char_span(atom)
        char_indices = () if span is None else tuple(range(*span))
        if any(char_index >= text_length for char_index in char_indices):
            raise ValueError(
                f"atom {atom.atom_uid!r} char span exceeds proof text length"
            )
        placements.append(_AtomPlacement(atom=atom, char_indices=char_indices))
    return tuple(placements)


def _char_views(
    line: ProofLineView,
    page: ProofPageView,
) -> tuple[ProofCharView, ...]:
    """Map each proof character to its immutable atom geometry when available."""

    if line.page_uid != page.page_uid:
        raise ValueError("line and page views must reference the same page UID")
    placements = _atom_placements(line)
    by_char_index: dict[int, tuple[_AtomPlacement, int]] = {}
    for placement in placements:
        for offset, char_index in enumerate(placement.char_indices):
            if char_index in by_char_index:
                raise ValueError(f"proof character index {char_index} maps to multiple atoms")
            by_char_index[char_index] = (placement, offset)
    result: list[ProofCharView] = []
    for char_index, text_char in enumerate(line.proof_text):
        mapped = by_char_index.get(char_index)
        placement, offset = mapped if mapped is not None else (None, -1)
        atom = placement.atom if placement is not None else None
        ocr_char = (
            atom.text[offset]
            if atom is not None and 0 <= offset < len(atom.text)
            else None
        )
        result.append(
            ProofCharView(
                proof_uid=line.proof_uid,
                text_unit_uid=line.text_unit_uid,
                char_index=char_index,
                text=text_char,
                page_uid=page.page_uid,
                page_number=page.page_number,
                image_path=page.image_path,
                line_uid=atom.line_uid if atom is not None else line.line_uid,
                region_uid=atom.region_uid if atom is not None else line.region_uid,
                atom_uid=atom.atom_uid if atom is not None else None,
                atom_index=atom.atom_index if atom is not None else None,
                bbox=atom.bbox if atom is not None else None,
                confidence=normalize_confidence(atom.confidence) if atom is not None else None,
                ocr_char=ocr_char,
                available=atom is not None,
            )
        )
    return tuple(result)


def _row_kind(line: ProofLineView, placements: tuple[_AtomPlacement, ...]) -> str:
    kinds = {
        kind
        for kind in (line.render_kind, *(_atom_kind(item.atom) for item in placements))
        if kind in {"formula", "table"}
    }
    if len(kinds) > 1:
        raise ValueError(f"proof line has conflicting explicit structure kinds: {sorted(kinds)}")
    return next(iter(kinds), "text")


def _row_preview_source(
    line: ProofLineView,
    placements: tuple[_AtomPlacement, ...],
    kind: str,
) -> str:
    if kind == "formula":
        formula_text = "".join(
            item.atom.text
            for item in placements
            if _atom_kind(item.atom) == "formula" and item.atom.text
        )
        return formula_text or line.proof_text
    return line.proof_text


def _row_image_bbox(
    line: ProofLineView,
    placements: tuple[_AtomPlacement, ...],
) -> tuple[int, int, int, int] | None:
    boxes = [item.atom.bbox for item in placements]
    if line.bbox is not None:
        boxes.insert(0, line.bbox)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


@dataclass(frozen=True, slots=True)
class _ProofRow:
    key: tuple[str, str]
    page: ProofPageView
    state: ProofStateView
    unit: ProofTextUnitView
    line: ProofLineView
    entries: tuple[ProofCharView, ...]
    atom_placements: tuple[_AtomPlacement, ...]
    kind: str
    preview_source: str


class _CommitTextEdit(QPlainTextEdit):
    """Editor whose proof shortcuts are routed to the parent view."""

    commit_requested = Signal()
    cancel_requested = Signal()
    navigate_requested = Signal(int)
    history_requested = Signal(int)
    focused = Signal()

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        super().focusInEvent(event)
        self.focused.emit()

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
            if event.key() == Qt.Key.Key_Z:
                self.history_requested.emit(1 if modifiers & Qt.KeyboardModifier.ShiftModifier else -1)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Y:
                self.history_requested.emit(1)
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


class _ProofLineImage(QLabel):
    """Clickable image projection with no OCR ownership."""

    clicked = Signal(QPoint)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.position().toPoint())
        super().mousePressEvent(event)


def _image_for_row(
    page: ProofPageView,
    bbox: tuple[int, int, int, int] | None,
) -> QPixmap:
    pixmap = QPixmap(page.image_path)
    if pixmap.isNull():
        return QPixmap()
    if bbox is not None:
        left, top, right, bottom = bbox
        x_scale = pixmap.width() / page.width if page.width > 0 else 1.0
        y_scale = pixmap.height() / page.height if page.height > 0 else 1.0
        rect = QRect(
            round(left * x_scale),
            round(top * y_scale),
            max(1, round((right - left) * x_scale)),
            max(1, round((bottom - top) * y_scale)),
        ).intersected(pixmap.rect())
        if not rect.isEmpty():
            pixmap = pixmap.copy(rect)
    return pixmap


def _page_filename(page: ProofPageView) -> str:
    source = page.source_path or page.image_path
    if not source:
        return f"第 {page.page_number} 页"
    return Path(source).name or source


class _ProofPageCard(QFrame):
    """Thumbnail card backed only by one immutable proof page view."""

    def __init__(self, page: ProofPageView, parent=None) -> None:
        if not isinstance(page, ProofPageView):
            raise TypeError("proof page card requires ProofPageView")
        super().__init__(parent)
        self.page = page
        self.setObjectName("proofPageCard")
        self.setProperty("selected", False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(164)
        self.setMaximumHeight(180)
        self.setStyleSheet(
            "QFrame#proofPageCard {"
            " background: #FBFAF7; border: 1px solid #D9D2C7; border-radius: 6px;"
            " padding: 5px;"
            "}"
            "QFrame#proofPageCard[selected=\"true\"] {"
            " background: #F2F7F3; border: 2px solid #547A68;"
            "}"
            "QLabel#proofPageNumber {"
            " background: #2C2C2C; color: #FFFFFF; padding: 3px 7px;"
            " border-radius: 4px; font-weight: 600;"
            "}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(5)

        self._image_frame = QWidget()
        self._image_frame.setObjectName("proofPageImageFrame")
        self._image_frame.setMinimumHeight(116)
        self._image_frame.setMaximumHeight(126)
        image_layout = QVBoxLayout(self._image_frame)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self._thumbnail = QLabel()
        self._thumbnail.setObjectName("proofPageThumbnail")
        self._thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumbnail.setText("无页面缩略图")
        image_layout.addWidget(self._thumbnail)
        self._page_number = QLabel(f"第 {page.page_number} 页", self._image_frame)
        self._page_number.setObjectName("proofPageNumber")
        self._page_number.adjustSize()
        root.addWidget(self._image_frame)

        self._filename = QLabel(_page_filename(page))
        self._filename.setObjectName("proofPageFilename")
        self._filename.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._filename.setWordWrap(False)
        self._filename.setToolTip(page.source_path or page.image_path)
        root.addWidget(self._filename)

        source = page.thumbnail_path or page.image_path
        self._source_pixmap = QPixmap(source) if source else QPixmap()
        self._refresh_thumbnail()
        self._position_page_number()

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", bool(selected))
        self.style().unpolish(self)
        self.style().polish(self)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_thumbnail()
        self._position_page_number()

    def _position_page_number(self) -> None:
        self._page_number.adjustSize()
        self._page_number.move(
            max(0, self._image_frame.width() - self._page_number.width() - 8),
            8,
        )
        self._page_number.raise_()

    def _refresh_thumbnail(self) -> None:
        if self._source_pixmap.isNull():
            self._thumbnail.setPixmap(QPixmap())
            self._thumbnail.setText("无页面缩略图")
            return
        target = self._thumbnail.size()
        if target.width() <= 0 or target.height() <= 0:
            self._thumbnail.setPixmap(self._source_pixmap)
            self._thumbnail.setText("")
            return
        self._thumbnail.setPixmap(
            self._source_pixmap.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._thumbnail.setText("")


class _ProofRowWidget(QFrame):
    commit_requested = Signal()
    cancel_requested = Signal()
    navigate_requested = Signal(int)
    history_requested = Signal(int)
    confirm_requested = Signal()
    activated = Signal()

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
        self._preview_button = QPushButton()
        self._preview_button.setObjectName("proofPreviewButton")
        self._preview_button.setVisible(row.kind == "formula")
        self._preview_button.clicked.connect(self._toggle_preview)
        header.addWidget(self._preview_button)
        content_layout.addLayout(header)

        self._image = _ProofLineImage()
        self._image.setObjectName("proofLineImage")
        self._image.setFixedHeight(58)
        self._image.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._image.setText("无可用行图像")
        self._line_bbox = _row_image_bbox(row.line, row.atom_placements)
        self._line_crop = _image_for_row(row.page, self._line_bbox)
        self._selected_char_index: int | None = None
        self._focus_depth = "far"
        self._displayed_pixmap_size = QSize()
        self._image.clicked.connect(self._on_image_clicked)
        content_layout.addWidget(self._image)

        self._preview_label = QLabel()
        self._preview_label.setObjectName("proofSpecialPreview")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._preview_label.setMinimumHeight(28)
        self._preview_label.setMaximumHeight(72)
        self._preview_label.setText("")
        self._preview_visible = row.kind == "formula"
        self._preview_label.setVisible(self._preview_visible)
        content_layout.addWidget(self._preview_label)
        self._formula_preview: FormulaRenderResult | None = None

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
        self.editor.history_requested.connect(self.history_requested)
        self.editor.focused.connect(self.activated)
        self.editor.cursorPositionChanged.connect(self._on_cursor_position_changed)
        self.editor.textChanged.connect(self._refresh_editor_geometry)
        content_layout.addWidget(self.editor)
        root.addWidget(content, 1)

        self.confirm_button = QPushButton("确认")
        self.confirm_button.setObjectName("secondaryBtn")
        self.confirm_button.setFixedSize(64, 28)
        self.confirm_button.clicked.connect(self.confirm_requested)
        root.addWidget(self.confirm_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.set_status(row.unit.status)
        self.set_active(False)
        self._refresh_preview()
        self._refresh_image()

    def set_active(self, active: bool) -> None:
        self.setProperty("active", active)
        self._active_bar.setVisible(active)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_focus_depth(self, depth: str) -> None:
        if depth not in {"active", "near", "far"}:
            raise ValueError(f"unsupported proof row focus depth: {depth!r}")
        self._focus_depth = depth
        active = depth == "active"
        self.set_active(active)
        self.editor.setVisible(active)
        self.confirm_button.setVisible(active)
        extra = 48 if self.row.kind == "formula" else 0
        self.setMinimumHeight((132 if active else 82) + extra)
        self.setMaximumHeight((148 if active else 92) + extra)
        self.setProperty("focusDepth", depth)
        self.style().unpolish(self)
        self.style().polish(self)
        self._refresh_image()

    def _refresh_preview(self) -> None:
        if self.row.kind == "formula":
            self._formula_preview = render_formula_pixmap(
                self.row.preview_source,
                target_height=34,
            )
            preview = self._formula_preview
            label = "公式"
        else:
            preview = None
            label = ""
        if preview is None:
            self._preview_label.setPixmap(QPixmap())
            self._preview_label.setText("无可用预览" if label else "")
        else:
            self._preview_label.setPixmap(preview.pixmap)
            self._preview_label.setText("")
            self._preview_label.setToolTip(self.row.preview_source)
        self._preview_button.setText(
            f"隐藏{label}预览" if self._preview_visible else f"{label}预览"
        )

    def _toggle_preview(self) -> None:
        if self.row.kind != "formula":
            return
        self._preview_visible = not self._preview_visible
        self._preview_label.setVisible(self._preview_visible)
        self._refresh_preview()

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
        source = QPixmap(self._line_crop)
        if self._selected_char_index is not None and self._line_bbox is not None:
            entry = next(
                (
                    item
                    for item in self.row.entries
                    if item.char_index == self._selected_char_index and item.bbox is not None
                ),
                None,
            )
            if entry is not None and entry.bbox is not None:
                line_left, line_top, _line_right, _line_bottom = self._line_bbox
                left, top, right, bottom = entry.bbox
                painter = QPainter(source)
                painter.setPen(QPen(QColor("#D45555"), 2))
                painter.setBrush(QColor(212, 85, 85, 32))
                line_width = max(1, self._line_bbox[2] - line_left)
                line_height = max(1, self._line_bbox[3] - line_top)
                painter.drawRect(
                    QRect(
                        round((left - line_left) * source.width() / line_width),
                        round((top - line_top) * source.height() / line_height),
                        max(1, round((right - left) * source.width() / line_width)),
                        max(1, round((bottom - top) * source.height() / line_height)),
                    )
                )
                painter.end()
        opacity = {"active": 1.0, "near": 0.68, "far": 0.38}[self._focus_depth]
        if opacity < 1.0:
            faded = QPixmap(source.size())
            faded.fill(Qt.GlobalColor.transparent)
            painter = QPainter(faded)
            painter.setOpacity(opacity)
            painter.drawPixmap(0, 0, source)
            painter.end()
            source = faded
        scaled = source.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._displayed_pixmap_size = scaled.size()
        self._image.setPixmap(scaled)
        self._image.setText("")
        self._refresh_editor_geometry()

    def _proof_char_x_positions(self) -> dict[int, float]:
        """Return immutable atom-derived character starts in page coordinates."""

        positions: dict[int, float] = {}
        for placement in self.row.atom_placements:
            indices = placement.char_indices
            if not indices:
                continue
            left, _top, right, _bottom = placement.atom.bbox
            slot_width = (right - left) / len(indices)
            for offset, char_index in enumerate(indices):
                positions[char_index] = left + offset * slot_width
        return positions

    def _refresh_editor_geometry(self) -> None:
        """Project immutable atom geometry into the active text editor.

        The projection is deliberately visual-only. When an edit changes the
        number of characters, the OCR geometry no longer describes that text,
        so the editor uses normal typography until a new workspace snapshot is
        supplied by the application layer.
        """

        text = self.editor.toPlainText()
        displayed_width = self._displayed_pixmap_size.width()
        displayed_height = self._displayed_pixmap_size.height()
        font = QFont(self.editor.font())
        font.setPixelSize(max(17, round(displayed_height * 0.62)) if displayed_height else 19)
        self.editor.setFont(font)

        document = self.editor.document()
        selection = self.editor.textCursor()
        saved_anchor = selection.anchor()
        saved_position = selection.position()
        all_text = QTextCursor(document)
        all_text.select(QTextCursor.SelectionType.Document)
        base_format = QTextCharFormat()
        base_format.setFont(font)
        base_format.setFontLetterSpacingType(QFont.SpacingType.AbsoluteSpacing)
        base_format.setFontLetterSpacing(0.0)

        self.editor.blockSignals(True)
        all_text.setCharFormat(base_format)
        block_cursor = QTextCursor(document)
        block_format = QTextBlockFormat(block_cursor.blockFormat())
        block_format.setLeftMargin(0.0)
        block_cursor.setBlockFormat(block_format)

        positions = self._proof_char_x_positions()
        geometry_valid = (
            bool(text)
            and len(text) == len(self.row.line.proof_text)
            and self._line_bbox is not None
            and displayed_width > 0
            and bool(positions)
        )
        if geometry_valid and self._line_bbox is not None:
            line_left, _top, line_right, _bottom = self._line_bbox
            line_width = max(1, line_right - line_left)
            desired = {
                index: (page_x - line_left) * displayed_width / line_width
                for index, page_x in positions.items()
            }
            first_index = min(desired)
            block_format.setLeftMargin(max(0.0, desired[first_index]))
            block_cursor.setBlockFormat(block_format)
            metrics = QFontMetricsF(font)
            for index in range(len(text) - 1):
                current_x = desired.get(index)
                next_x = desired.get(index + 1)
                if current_x is None or next_x is None:
                    continue
                spacing = next_x - current_x - metrics.horizontalAdvance(text[index])
                char_cursor = QTextCursor(document)
                char_cursor.setPosition(index)
                char_cursor.setPosition(index + 1, QTextCursor.MoveMode.KeepAnchor)
                char_format = QTextCharFormat(base_format)
                char_format.setFontLetterSpacing(spacing)
                char_cursor.setCharFormat(char_format)

        selection.setPosition(min(saved_anchor, len(text)))
        selection.setPosition(
            min(saved_position, len(text)),
            QTextCursor.MoveMode.KeepAnchor,
        )
        self.editor.setTextCursor(selection)
        self.editor.blockSignals(False)

    def _on_cursor_position_changed(self) -> None:
        if not self.row.entries:
            return
        cursor = self.editor.textCursor()
        position = cursor.selectionStart() if cursor.hasSelection() else cursor.position()
        if position >= len(self.editor.toPlainText()) and position > 0:
            position -= 1
        available = {entry.char_index for entry in self.row.entries if entry.bbox is not None}
        self._selected_char_index = position if position in available else None
        self._refresh_image()

    def _on_image_clicked(self, point: QPoint) -> None:
        self.activated.emit()
        if self._line_bbox is None or self._displayed_pixmap_size.isEmpty():
            self.editor.setFocus()
            return
        y_offset = max(0, (self._image.height() - self._displayed_pixmap_size.height()) // 2)
        if not (
            0 <= point.x() < self._displayed_pixmap_size.width()
            and y_offset <= point.y() < y_offset + self._displayed_pixmap_size.height()
        ):
            return
        source_x = point.x() * self._line_crop.width() / self._displayed_pixmap_size.width()
        source_y = (point.y() - y_offset) * self._line_crop.height() / self._displayed_pixmap_size.height()
        line_left, line_top, line_right, line_bottom = self._line_bbox
        page_x = line_left + source_x * (line_right - line_left) / self._line_crop.width()
        page_y = line_top + source_y * (line_bottom - line_top) / self._line_crop.height()
        entries = tuple(entry for entry in self.row.entries if entry.bbox is not None)
        if not entries:
            self.editor.setFocus()
            return
        entry = min(
            entries,
            key=lambda item: (
                0
                if item.bbox is not None
                and item.bbox[0] <= page_x <= item.bbox[2]
                and item.bbox[1] <= page_y <= item.bbox[3]
                else 1,
                abs(((item.bbox[0] + item.bbox[2]) / 2) - page_x) if item.bbox else float("inf"),
            ),
        )
        cursor = self.editor.textCursor()
        cursor.setPosition(min(entry.char_index, len(self.editor.toPlainText())))
        cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, 1)
        self.editor.setTextCursor(cursor)
        self.editor.setFocus()

    def set_status(self, status: str) -> None:
        color = STATUS_COLORS.get(status, STATUS_COLORS["unchecked"])
        self._status.setText(STATUS_LABELS.get(status, status))
        self._status.setStyleSheet(f"color: {color}; font-weight: 600;")


class HProofPanel(QWidget):
    """Horizontal proof editor that consumes one immutable workspace snapshot."""

    proof_edit_requested = Signal(object)

    def __init__(
        self,
        workspace: ProofWorkspaceView | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._workspace: ProofWorkspaceView | None = None
        self._rows: tuple[_ProofRow, ...] = ()
        self._visible_rows: tuple[_ProofRow, ...] = ()
        self._row_widgets: dict[tuple[str, str], _ProofRowWidget] = {}
        self._page_cards: dict[str, _ProofPageCard] = {}
        self._dirty_text: dict[tuple[str, str], str] = {}
        self._selected_page_uid: str | None = None
        self._active_key: tuple[str, str] | None = None
        self._build_ui()
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
        self._page_directory = QListWidget()
        self._page_directory.setObjectName("pageDirectoryList")
        self._page_directory.setSpacing(8)
        self._page_directory.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._page_directory.currentItemChanged.connect(self._on_page_directory_selected)
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
        self._btn_refresh.clicked.connect(self.refresh_view)
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

    def set_workspace(self, workspace: ProofWorkspaceView | None) -> None:
        """Replace the displayed immutable snapshot."""

        if workspace is not None and not isinstance(workspace, ProofWorkspaceView):
            raise TypeError("HProofPanel requires ProofWorkspaceView or None")
        selected = self._selected_page_uid
        self._workspace = workspace
        self._dirty_text.clear()
        pages = self._pages()
        page_uids = {page.page_uid for page in pages}
        self._selected_page_uid = selected if selected in page_uids else (pages[0].page_uid if pages else None)
        self._populate_page_selector()
        self._build_rows()
        self._render_rows()

    def clear_workspace(self) -> None:
        self.set_workspace(None)

    def _pages(self) -> tuple[ProofPageView, ...]:
        if self._workspace is None:
            return ()
        return tuple(sorted(self._workspace.pages, key=lambda item: (item.page_number, item.page_uid)))

    def _populate_page_selector(self) -> None:
        self._page_directory.blockSignals(True)
        self._page_directory.clear()
        self._page_cards.clear()
        pages = self._pages()
        for page in pages:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, page.page_uid)
            item.setSizeHint(QSize(0, 174))
            self._page_directory.addItem(item)
            card = _ProofPageCard(page)
            self._page_directory.setItemWidget(item, card)
            self._page_cards[page.page_uid] = card
        current = next(
            (index for index, page in enumerate(pages) if page.page_uid == self._selected_page_uid),
            -1,
        )
        if current >= 0:
            self._page_directory.setCurrentRow(current)
        self._page_directory.blockSignals(False)
        for page_uid, card in self._page_cards.items():
            card.set_selected(page_uid == self._selected_page_uid)
        self._scope.setText(
            f"第 {pages[current].page_number} 页" if current >= 0 else "全部页面"
        )

    def _on_page_directory_selected(self, current: QListWidgetItem | None, _previous) -> None:
        self._selected_page_uid = (
            current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        )
        page = next(
            (item for item in self._pages() if item.page_uid == self._selected_page_uid),
            None,
        )
        for page_uid, card in self._page_cards.items():
            card.set_selected(page_uid == self._selected_page_uid)
        self._scope.setText(f"第 {page.page_number} 页" if page is not None else "全部页面")
        self._render_rows()

    def _build_rows(self) -> None:
        if self._workspace is None:
            self._rows = ()
            return
        pages = {page.page_uid: page for page in self._workspace.pages}
        rows: list[_ProofRow] = []
        for state in self._workspace.proof_states:
            page = pages.get(state.page_uid)
            if page is None:
                raise ValueError(f"proof state {state.proof_uid!r} has no page view")
            units = {unit.text_unit_uid: unit for unit in state.text_units}
            for line in state.lines:
                unit = units.get(line.text_unit_uid)
                if unit is None:
                    raise ValueError(f"proof line {line.text_unit_uid!r} has no text unit view")
                atom_placements = _atom_placements(line)
                kind = _row_kind(line, atom_placements)
                rows.append(
                    _ProofRow(
                        key=(state.proof_uid, unit.text_unit_uid),
                        page=page,
                        state=state,
                        unit=unit,
                        line=line,
                        entries=_char_views(line, page),
                        atom_placements=atom_placements,
                        kind=kind,
                        preview_source=_row_preview_source(line, atom_placements, kind),
                    )
                )
        self._rows = tuple(
            sorted(rows, key=lambda row: (row.page.page_number, row.state.proof_uid, row.unit.order, row.unit.text_unit_uid))
        )

    def refresh_view(self) -> None:
        """Repaint the current snapshot without querying or mutating state."""

        self._build_rows()
        self._render_rows()

    def _render_rows(self) -> None:
        self._visible_rows = tuple(
            row
            for row in self._rows
            if self._selected_page_uid is None or row.page.page_uid == self._selected_page_uid
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
            widget.commit_requested.connect(
                lambda row=row, widget=widget: self._commit_row(row, widget)
            )
            widget.cancel_requested.connect(
                lambda row=row, widget=widget: self._cancel_row(row, widget)
            )
            widget.navigate_requested.connect(
                lambda direction, row=row: self._navigate_from(row, direction)
            )
            widget.history_requested.connect(
                lambda direction, row=row: self._apply_history(row, direction)
            )
            widget.confirm_requested.connect(
                lambda row=row, widget=widget: self._confirm_row(row, widget)
            )
            widget.activated.connect(lambda row=row: self._activate_row(row.key))
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, widget)
            self._row_widgets[row.key] = widget
        if self._visible_rows:
            active_key = self._active_key if self._active_key in self._row_widgets else self._visible_rows[0].key
            self._activate_row(active_key)
        else:
            self._active_key = None
        self._status.setText(
            "暂无可校对内容"
            if not self._visible_rows
            else f"{len(self._visible_rows)} 行 · {sum(len(row.unit.text) for row in self._visible_rows)} 字符"
        )

    def _activate_row(self, key: tuple[str, str] | None) -> None:
        self._active_key = key
        index = next(
            (index for index, row in enumerate(self._visible_rows) if row.key == key),
            -1,
        )
        for row_index, row in enumerate(self._visible_rows):
            widget = self._row_widgets.get(row.key)
            if widget is None:
                continue
            distance = abs(row_index - index) if index >= 0 else 99
            widget.set_focus_depth("active" if distance == 0 else "near" if distance == 1 else "far")

    def _on_text_changed(self, row: _ProofRow, widget: _ProofRowWidget) -> None:
        text = widget.editor.toPlainText()
        if text == row.unit.text:
            self._dirty_text.pop(row.key, None)
        else:
            self._dirty_text[row.key] = text
        self._activate_row(row.key)

    def _emit(self, command: ProofEditCommand) -> None:
        self.proof_edit_requested.emit(command)

    def _replace_command(
        self,
        row: _ProofRow,
        text: str,
        *,
        status: str = "modified",
    ) -> ProofEditCommand:
        return ProofEditCommand(
            proof_uid=row.state.proof_uid,
            op="replace_text",
            expected_revision=row.state.revision,
            expected_fingerprint=row.state.fingerprint,
            text_unit_uid=row.unit.text_unit_uid,
            text=text,
            status=status,
            expected_unit_revision=row.unit.revision,
            expected_unit_fingerprint=row.unit.fingerprint,
        )

    def _apply_history(self, row: _ProofRow, direction: int) -> None:
        if direction not in {-1, 1}:
            raise ValueError("history direction must be -1 or 1")
        self._emit(
            ProofEditCommand(
                proof_uid=row.state.proof_uid,
                op="undo" if direction < 0 else "redo",
                expected_revision=row.state.revision,
                expected_fingerprint=row.state.fingerprint,
            )
        )

    def _commit_row(self, row: _ProofRow, widget: _ProofRowWidget) -> bool:
        text = widget.editor.toPlainText()
        if text == row.unit.text:
            self._dirty_text.pop(row.key, None)
            return False
        self._emit(self._replace_command(row, text))
        self._dirty_text.pop(row.key, None)
        widget.set_status("modified")
        return True

    def _save_all(self) -> bool:
        grouped: dict[str, list[tuple[_ProofRow, str]]] = defaultdict(list)
        rows = {row.key: row for row in self._rows}
        for key, text in tuple(self._dirty_text.items()):
            row = rows.get(key)
            if row is not None and text != row.unit.text:
                grouped[row.state.proof_uid].append((row, text))
        emitted = False
        for proof_uid, values in grouped.items():
            state = values[0][0].state
            replacements = tuple(
                (row.unit.text_unit_uid, text)
                for row, text in sorted(values, key=lambda item: (item[0].unit.order, item[0].unit.text_unit_uid))
            )
            self._emit(
                ProofEditCommand(
                    proof_uid=proof_uid,
                    op="replace_many",
                    expected_revision=state.revision,
                    expected_fingerprint=state.fingerprint,
                    replacements=replacements,
                )
            )
            emitted = True
        if emitted:
            self._dirty_text.clear()
        return emitted

    def save(self) -> bool:
        return self._save_all()

    def _find_row(self, key: tuple[str, str]) -> _ProofRow | None:
        return next((row for row in self._rows if row.key == key), None)

    def _confirm_row(self, row: _ProofRow, widget: _ProofRowWidget) -> None:
        text = widget.editor.toPlainText()
        if text != row.unit.text:
            self._emit(self._replace_command(row, text, status="checked"))
            self._dirty_text.pop(row.key, None)
        else:
            self._emit(
                ProofEditCommand(
                    proof_uid=row.state.proof_uid,
                    op="set_status",
                    expected_revision=row.state.revision,
                    expected_fingerprint=row.state.fingerprint,
                    text_unit_uid=row.unit.text_unit_uid,
                    status="checked",
                    expected_unit_revision=row.unit.revision,
                    expected_unit_fingerprint=row.unit.fingerprint,
                )
            )
        widget.set_status("checked")

    def _cancel_row(self, row: _ProofRow, widget: _ProofRowWidget) -> None:
        self._dirty_text.pop(row.key, None)
        widget.editor.blockSignals(True)
        widget.editor.setPlainText(row.unit.text)
        widget.editor.blockSignals(False)
        widget.set_status(row.unit.status)

    def _navigate_from(self, row: _ProofRow, delta: int) -> None:
        widget = self._row_widgets.get(row.key)
        if widget is not None:
            self._commit_row(row, widget)
        self._focus_row(delta, current_key=row.key)

    def _focus_row(
        self,
        delta: int,
        *,
        current_key: tuple[str, str] | None = None,
    ) -> None:
        if not self._visible_rows:
            return
        key = current_key or self._active_key or self._visible_rows[0].key
        current = next(
            (index for index, row in enumerate(self._visible_rows) if row.key == key),
            0,
        )
        target = max(0, min(len(self._visible_rows) - 1, current + delta))
        target_key = self._visible_rows[target].key
        self._activate_row(target_key)
        widget = self._row_widgets.get(target_key)
        if widget is not None:
            widget.editor.setFocus()

    def closeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        self._save_all()
        super().closeEvent(event)


__all__ = ["HProofPanel"]
