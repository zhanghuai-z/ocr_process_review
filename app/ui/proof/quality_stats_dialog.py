"""Read-only proof quality statistics over an immutable workspace view."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.application.proof_workspace import ProofWorkspaceView


@dataclass(frozen=True, slots=True)
class _QualityStats:
    state_count: int
    unit_count: int
    character_count: int
    checked_character_count: int
    modified_unit_count: int
    rebind_count: int

    @property
    def checked_ratio(self) -> float | None:
        if self.character_count == 0:
            return None
        return self.checked_character_count / self.character_count


def _stats_for(workspace: ProofWorkspaceView | None) -> _QualityStats:
    if workspace is None:
        return _QualityStats(0, 0, 0, 0, 0, 0)
    states = tuple(workspace.proof_states)
    units = tuple(unit for state in states for unit in state.text_units)
    checked = sum(
        len(unit.text)
        for unit in units
        if unit.status in {"checked", "ok"}
    )
    modified = sum(1 for unit in units if unit.status == "modified")
    rebound = sum(1 for state in states if state.rebind_required)
    return _QualityStats(
        state_count=len(states),
        unit_count=len(units),
        character_count=sum(len(unit.text) for unit in units),
        checked_character_count=checked,
        modified_unit_count=modified,
        rebind_count=rebound,
    )


class _RatioRing(QWidget):
    """Compact checked-character ratio indicator."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(140, 140)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        self._ratio: float | None = None
        self._main_text = "-"
        self._sub_text = ""

    def set_ratio(
        self,
        ratio: float | None,
        sub_text: str = "",
        main_text: str = "",
    ) -> None:
        self._ratio = ratio
        self._sub_text = sub_text
        self._main_text = main_text or "-"
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        size = min(self.width(), self.height())
        margin = 10
        rect = QRectF(
            (self.width() - size) / 2 + margin,
            (self.height() - size) / 2 + margin,
            size - 2 * margin,
            size - 2 * margin,
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(
            QPen(QColor("#E7E2D8"), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        )
        painter.drawArc(rect, 0, 360 * 16)
        if self._ratio is not None:
            ratio = max(0.0, min(1.0, self._ratio))
            color = QColor("#4E7A63" if ratio >= 0.8 else "#C67B22")
            painter.setPen(
                QPen(color, 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            )
            painter.drawArc(rect, 90 * 16, int(-ratio * 360 * 16))
        painter.setPen(QColor("#2C2C2C"))
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            rect.adjusted(0, 0, 0, -int(rect.height() * 0.25)),
            Qt.AlignmentFlag.AlignCenter,
            self._main_text,
        )
        if self._sub_text:
            font.setPointSize(9)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QColor("#6B6B6B"))
            painter.drawText(
                rect.adjusted(0, int(rect.height() * 0.30), 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                self._sub_text,
            )
        painter.end()


class QualityStatsDetailDialog(QDialog):
    """Detailed rows derived from proof state DTOs."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("qualityStatsDetailDialog")
        self.setWindowTitle("Proof state details")
        self.resize(760, 460)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        self._summary = QLabel()
        root.addWidget(self._summary)
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Scope", "Proof state", "Text unit", "Order", "Status", "Text"]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self._table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

    def populate(self, workspace: ProofWorkspaceView | None) -> None:
        if workspace is None:
            self._table.setRowCount(0)
            self._summary.setText("No proof workspace")
            return
        page_numbers = {page.page_uid: page.page_number for page in workspace.pages}
        rows = [
            (page_numbers.get(state.page_uid, "-"), state, unit)
            for state in sorted(workspace.proof_states, key=lambda item: (item.page_uid, item.proof_uid))
            for unit in state.text_units
        ]
        self._summary.setText(f"{len(rows)} text units")
        self._table.setRowCount(len(rows))
        for row, (page_number, state, unit) in enumerate(rows):
            values = (
                str(page_number),
                state.proof_uid,
                unit.text_unit_uid,
                str(unit.order),
                unit.status,
                unit.text,
            )
            for column, value in enumerate(values):
                self._table.setItem(row, column, QTableWidgetItem(value))

    def table(self) -> QTableWidget:
        return self._table


class QualityStatsDialog(QDialog):
    """Read-only proof quality summary backed by one workspace snapshot."""

    def __init__(
        self,
        workspace: ProofWorkspaceView | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if workspace is not None and not isinstance(workspace, ProofWorkspaceView):
            raise TypeError("QualityStatsDialog requires ProofWorkspaceView or None")
        self.setObjectName("qualityStatsDialog")
        self.setWindowTitle("Proof quality statistics")
        self.setModal(True)
        self.resize(420, 360)
        self._workspace = workspace
        self._detail_dialog: QualityStatsDetailDialog | None = None
        self._stats = _QualityStats(0, 0, 0, 0, 0, 0)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)
        top = QHBoxLayout()
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self._on_manual_refresh)
        top.addWidget(self._btn_refresh)
        self._switch_status = QLabel()
        self._switch_status.setObjectName("muted")
        top.addWidget(self._switch_status, 1)
        root.addLayout(top)

        self._ring = _RatioRing()
        root.addWidget(self._ring, 1, Qt.AlignmentFlag.AlignCenter)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)
        self._btn_detail = QPushButton("Details")
        self._btn_detail.clicked.connect(self._show_details)
        root.addWidget(self._btn_detail)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)
        self._refresh_view()

    def set_workspace(self, workspace: ProofWorkspaceView | None) -> None:
        if workspace is not None and not isinstance(workspace, ProofWorkspaceView):
            raise TypeError("QualityStatsDialog requires ProofWorkspaceView or None")
        self._workspace = workspace
        self._refresh_view()

    def _refresh_view(self) -> None:
        self._stats = _stats_for(self._workspace)
        ratio = self._stats.checked_ratio
        main = "-" if ratio is None else f"{ratio * 100:.0f}%"
        sub = (
            "No proof text"
            if ratio is None
            else f"{self._stats.checked_character_count}/{self._stats.character_count} chars"
        )
        self._ring.set_ratio(ratio, sub, main)
        self._switch_status.setText(
            f"{self._stats.state_count} proof states / {self._stats.unit_count} units"
        )
        self._summary.setText(
            f"Modified units: {self._stats.modified_unit_count}\n"
            f"Rebind required: {self._stats.rebind_count}"
        )

    def _on_manual_refresh(self) -> None:
        self._refresh_view()

    def _show_details(self) -> None:
        if self._detail_dialog is None:
            self._detail_dialog = QualityStatsDetailDialog(self)
        self._detail_dialog.populate(self._workspace)
        self._detail_dialog.show()
        self._detail_dialog.raise_()
        self._detail_dialog.activateWindow()

    def detail_table(self) -> QTableWidget:
        if self._detail_dialog is None:
            self._detail_dialog = QualityStatsDetailDialog(self)
        self._detail_dialog.populate(self._workspace)
        return self._detail_dialog.table()

    @property
    def stats(self) -> _QualityStats:
        return self._stats


__all__ = ["QualityStatsDetailDialog", "QualityStatsDialog"]
