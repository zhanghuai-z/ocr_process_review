"""Regression coverage for VProof gallery height budgeting."""

from __future__ import annotations

import os

import pytest
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _new_panel():
    from app.ui.proof.v_proof import VProofPanel

    return VProofPanel()


def _force_layout(panel):
    panel.resize(1500, 900)
    panel.show()
    QApplication.processEvents()
    panel._gallery_box.adjustSize()
    QApplication.processEvents()


def test_gallery_view_height_fits_one_row():
    from app.ui.proof.v_proof import GALLERY_THUMB

    panel = _new_panel()
    panel._resize_gallery_for_entries(1)
    _force_layout(panel)
    item_h = GALLERY_THUMB + 18
    assert panel._gallery_view.height() >= item_h
    panel.close()


def test_gallery_view_height_fits_three_rows():
    from app.ui.proof.v_proof import GALLERY_THUMB, GALLERY_MAX_ROWS

    panel = _new_panel()
    panel._resize_gallery_for_entries(GALLERY_MAX_ROWS * 6)
    _force_layout(panel)
    item_h = GALLERY_THUMB + 18
    assert panel._gallery_view.height() >= GALLERY_MAX_ROWS * item_h
    panel.close()


def test_gallery_box_fixed_height_accounts_for_chrome():
    from app.ui.proof.v_proof import GALLERY_THUMB

    panel = _new_panel()
    panel._resize_gallery_for_entries(1)
    item_h = GALLERY_THUMB + 18
    assert panel._gallery_box.height() == item_h + 8 + 48
    panel.close()
