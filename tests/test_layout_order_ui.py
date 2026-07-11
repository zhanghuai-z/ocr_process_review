from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QKeyEvent
from PySide6.QtWidgets import QApplication, QGraphicsItem

from app.models import BBox, Block, BlockType, Line, OcrProject, Page, PageStatus
from app.models.page_state import page_needs_ocr_rerun
from app.models.layout_snapshot_projection import sync_page_layout_snapshot_from_projection
from app.models.layout_snapshot_store import layout_snapshot_for_page
from app.ui.widgets.image_viewer import ImageViewer


def _qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _page_with_layout() -> tuple[Page, list[Block]]:
    blocks = [
        Block(BlockType.TEXT, BBox(10 + index * 40, 10, 30, 20), order=index)
        for index in range(3)
    ]
    page = Page(image_path="", width=140, height=60, blocks=blocks)
    sync_page_layout_snapshot_from_projection(page, source_engine="test_layout_order_ui")
    return page, blocks


def test_order_path_uses_crossed_segments_instead_of_mouse_sample_points():
    _qapp()
    viewer = ImageViewer()
    viewer.set_image_from_qimage(QImage(140, 60, QImage.Format.Format_RGB888))
    page, blocks = _page_with_layout()
    viewer.show_blocks(page.blocks)
    viewer.set_order_mode("path")

    viewer._append_order_path_segment(QPointF(0, 20), QPointF(139, 20))

    assert viewer._order_path_uids == [block.uid for block in blocks]
    assert all(
        not bool(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        for item, _block in viewer._block_items
    )
    viewer.close()


def test_order_path_uses_first_frame_crossing_for_nested_blocks():
    _qapp()
    viewer = ImageViewer()
    viewer.set_image_from_qimage(QImage(140, 80, QImage.Format.Format_RGB888))
    outer = Block(BlockType.TEXT, BBox(10, 10, 110, 50), order=0)
    inner = Block(BlockType.EQUATION, BBox(40, 20, 30, 20), order=1)
    viewer.show_blocks([outer, inner])
    viewer.set_order_mode("path")

    viewer._append_order_path_segment(QPointF(0, 30), QPointF(139, 30))

    assert viewer._order_path_uids == [outer.uid, inner.uid]
    viewer.close()


def test_order_path_does_not_claim_parent_when_trajectory_stays_inside_its_frame():
    _qapp()
    viewer = ImageViewer()
    viewer.set_image_from_qimage(QImage(140, 80, QImage.Format.Format_RGB888))
    outer = Block(BlockType.TEXT, BBox(10, 10, 110, 50), order=0)
    inner = Block(BlockType.EQUATION, BBox(40, 20, 30, 20), order=1)
    viewer.show_blocks([outer, inner])
    viewer.set_order_mode("path")

    viewer._append_order_path_segment(QPointF(30, 30), QPointF(80, 30))

    assert viewer._order_path_uids == [inner.uid]
    viewer.close()


def test_order_mode_keeps_space_pan_and_restores_order_cursor():
    _qapp()
    viewer = ImageViewer()
    viewer.set_image_from_qimage(QImage(140, 80, QImage.Format.Format_RGB888))
    viewer.set_order_mode("path")

    viewer.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier))
    assert viewer.is_pan_mode_active() is True
    viewer.keyReleaseEvent(QKeyEvent(QKeyEvent.Type.KeyRelease, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier))

    assert viewer.is_pan_mode_active() is False
    assert viewer.viewport().cursor().shape() == Qt.CursorShape.CrossCursor
    viewer.close()


def test_layout_panel_click_order_commits_complete_uid_sequence(monkeypatch):
    _qapp()
    import app.ui.recognize.layout_panel as layout_panel_module

    monkeypatch.setattr(
        layout_panel_module,
        "get_config",
        lambda: {"layout_order_tools_enabled": True},
    )
    page, blocks = _page_with_layout()
    panel = layout_panel_module.LayoutPanel()
    contract_changes: list[tuple[int, str]] = []
    order_changes: list[int] = []
    panel.block_contract_changed.connect(
        lambda page_number, kind: contract_changes.append((page_number, kind))
    )
    panel.block_order_changed.connect(order_changes.append)
    panel.set_pages([page])

    panel._set_order_mode("click")
    for block in (blocks[2], blocks[0], blocks[1]):
        panel._on_order_block_activated(block.uid)

    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert [block.uid for block in snapshot.blocks] == [
        blocks[2].uid,
        blocks[0].uid,
        blocks[1].uid,
    ]
    assert [block.uid for block in page.blocks] == [
        blocks[2].uid,
        blocks[0].uid,
        blocks[1].uid,
    ]
    assert contract_changes == []
    assert order_changes == [page.page_number]
    assert panel._order_mode == ""
    panel.close()


def test_layout_panel_rejects_incomplete_path_without_mutating_order(monkeypatch):
    _qapp()
    import app.ui.recognize.layout_panel as layout_panel_module

    monkeypatch.setattr(
        layout_panel_module,
        "get_config",
        lambda: {"layout_order_tools_enabled": True},
    )
    page, blocks = _page_with_layout()
    panel = layout_panel_module.LayoutPanel()
    panel.set_pages([page])
    panel._set_order_mode("path")

    panel._on_order_path_finished((blocks[1].uid, blocks[0].uid))

    snapshot = layout_snapshot_for_page(page)
    assert snapshot is not None
    assert [block.uid for block in snapshot.blocks] == [block.uid for block in blocks]
    assert panel._order_mode == "path"
    assert "2 / 3" in panel._status_lbl.text()
    panel.close()


def test_layout_panel_refuses_order_mode_when_snapshot_projection_is_incomplete(monkeypatch):
    _qapp()
    import app.ui.recognize.layout_panel as layout_panel_module

    monkeypatch.setattr(
        layout_panel_module,
        "get_config",
        lambda: {"layout_order_tools_enabled": True},
    )
    page, _blocks = _page_with_layout()
    page.blocks.pop()
    panel = layout_panel_module.LayoutPanel()
    panel.set_pages([page])

    panel._set_order_mode("path")

    assert panel._order_mode == ""
    assert "投影不完整" in panel._status_lbl.text()
    panel.close()


def test_controller_persists_order_change_without_invalidating_ocr(monkeypatch):
    from app.controllers.workflow_controller import WorkflowController

    page, blocks = _page_with_layout()
    blocks[0].lines = [Line(text="正文", bbox=blocks[0].bbox, confidence=0.9)]
    page.status = PageStatus.OCR_DONE
    controller = WorkflowController()
    controller._project = OcrProject(name="order", pages=[page])
    saved: list[bool] = []
    synced: list[bool] = []
    monkeypatch.setattr(controller, "_save_if_bound", lambda: saved.append(True))
    monkeypatch.setattr(controller, "sync_proof_panels", lambda: synced.append(True))

    controller.handle_block_order_changed(page.page_number)

    assert page.status == PageStatus.OCR_DONE
    assert page_needs_ocr_rerun(page) is False
    assert blocks[0].lines[0].text == "正文"
    assert saved == [True]
    assert synced == [True]
    controller.close()


def test_layout_order_controls_are_hidden_when_experiment_is_disabled(monkeypatch):
    _qapp()
    import app.ui.recognize.layout_panel as layout_panel_module

    monkeypatch.setattr(
        layout_panel_module,
        "get_config",
        lambda: {"layout_order_tools_enabled": False},
    )
    panel = layout_panel_module.LayoutPanel()

    assert panel._btn_order_numbers.isHidden()
    assert panel._btn_order_click.isHidden()
    assert panel._btn_order_path.isHidden()
    assert panel._viewer._order_mode == ""
    panel.close()
