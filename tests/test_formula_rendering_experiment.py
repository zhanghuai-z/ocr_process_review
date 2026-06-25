from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication


def test_formula_rendering_experiment_renders_latin_math(monkeypatch):
    pytest.importorskip("matplotlib")
    from app.experimental.formula_rendering import render_formula_pixmap

    monkeypatch.delenv("OCR_EXPERIMENTAL_FORMULA_RENDER", raising=False)
    app = QApplication.instance() or QApplication([])
    assert app is not None

    result = render_formula_pixmap(r"$Y_t = \alpha + \beta Incentive_c \times Post_t$", target_height=34)

    assert result is not None
    assert result.pixmap.width() > 20
    assert result.logical_width > 20
    assert result.logical_height == 34
    assert result.device_pixel_ratio >= 1.0
    assert result.pixmap.devicePixelRatio() == pytest.approx(result.device_pixel_ratio)


def test_formula_rendering_experiment_skips_cjk_mixed_lines(monkeypatch):
    pytest.importorskip("matplotlib")
    from app.experimental.formula_rendering import normalize_formula_latex, render_formula_pixmap

    monkeypatch.delenv("OCR_EXPERIMENTAL_FORMULA_RENDER", raising=False)
    app = QApplication.instance() or QApplication([])
    assert app is not None

    text = "变量。核心解释变量 Incentive_c × Post_t 为强度变量"

    assert normalize_formula_latex(text) == ""
    assert render_formula_pixmap(text, target_height=34) is None


def test_formula_rendering_experiment_can_be_disabled(monkeypatch):
    pytest.importorskip("matplotlib")
    from app.experimental.formula_rendering import render_formula_pixmap

    monkeypatch.setenv("OCR_EXPERIMENTAL_FORMULA_RENDER", "0")
    app = QApplication.instance() or QApplication([])
    assert app is not None

    assert render_formula_pixmap("Incentive_c × Post_t", target_height=34) is None


def test_formula_rendering_experiment_normalizes_target_logical_size(monkeypatch):
    pytest.importorskip("matplotlib")
    from app.experimental.formula_rendering import render_formula_pixmap

    monkeypatch.delenv("OCR_EXPERIMENTAL_FORMULA_RENDER", raising=False)
    app = QApplication.instance() or QApplication([])
    assert app is not None

    small = render_formula_pixmap(r"$x^2$", target_height=18, oversample=1.0)
    large = render_formula_pixmap(r"$x^2$", target_height=42, oversample=2.0)

    assert small is not None
    assert large is not None
    assert small.logical_height == 18
    assert large.logical_height == 42
    assert large.logical_width > small.logical_width
