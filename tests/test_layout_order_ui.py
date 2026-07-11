from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QGraphicsItem

from app.models import BBox, Block, BlockType, Page
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
    changes: list[tuple[int, str]] = []
    panel.block_contract_changed.connect(lambda page_number, kind: changes.append((page_number, kind)))
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
    assert changes == [(page.page_number, "block_order_changed")]
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
