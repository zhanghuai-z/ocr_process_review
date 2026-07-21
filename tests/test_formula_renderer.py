from __future__ import annotations

from pathlib import Path
import shutil

import pytest
from PySide6.QtWidgets import QApplication


_TINY_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="20" '
    b'viewBox="0 0 100 20"><rect width="100" height="20" fill="#222"/></svg>'
)


def test_formula_renderer_renders_latin_math(monkeypatch):
    from app.ui.proof import formula_renderer

    monkeypatch.delenv("OCR_FORMULA_RENDER", raising=False)
    monkeypatch.delenv("OCR_FORMULA_ENGINE", raising=False)
    monkeypatch.setattr(formula_renderer, "_render_mathjax_svg", lambda _latex, _color: _TINY_SVG)
    app = QApplication.instance() or QApplication([])
    assert app is not None

    result = formula_renderer.render_formula_pixmap(
        r"$Y_t = \alpha + \beta Incentive_c \times Post_t$",
        target_height=34,
    )

    assert result is not None
    assert result.pixmap.width() > 20
    assert result.logical_width > 20
    assert result.logical_height == 34
    assert result.device_pixel_ratio >= 1.0
    assert result.pixmap.devicePixelRatio() == pytest.approx(result.device_pixel_ratio)
    assert result.backend == "mathjax_svg"


def test_formula_renderer_skips_cjk_mixed_lines(monkeypatch):
    from app.ui.proof.formula_renderer import normalize_formula_latex, render_formula_pixmap

    monkeypatch.delenv("OCR_FORMULA_RENDER", raising=False)
    app = QApplication.instance() or QApplication([])
    assert app is not None

    text = "变量。核心解释变量 Incentive_c × Post_t 为强度变量"

    assert normalize_formula_latex(text) == ""
    assert render_formula_pixmap(text, target_height=34) is None


def test_formula_renderer_can_be_disabled(monkeypatch):
    from app.ui.proof.formula_renderer import render_formula_pixmap

    monkeypatch.setenv("OCR_FORMULA_RENDER", "0")
    app = QApplication.instance() or QApplication([])
    assert app is not None

    assert render_formula_pixmap("Incentive_c × Post_t", target_height=34) is None


def test_formula_renderer_can_force_latex_svg(monkeypatch):
    from app.ui.proof import formula_renderer

    monkeypatch.delenv("OCR_FORMULA_RENDER", raising=False)
    monkeypatch.setenv("OCR_FORMULA_ENGINE", "latex_svg")
    monkeypatch.setattr(formula_renderer, "_render_latex_svg", lambda _latex, _color_hex: _TINY_SVG)
    app = QApplication.instance() or QApplication([])
    assert app is not None

    result = formula_renderer.render_formula_pixmap(r"$E=mc^2$", target_height=28)

    assert result is not None
    assert result.backend == "latex_svg"
    assert result.logical_height == 28


def test_formula_renderer_normalizes_target_logical_size(monkeypatch):
    from app.ui.proof import formula_renderer

    monkeypatch.delenv("OCR_FORMULA_RENDER", raising=False)
    monkeypatch.delenv("OCR_FORMULA_ENGINE", raising=False)
    monkeypatch.setattr(formula_renderer, "_render_mathjax_svg", lambda _latex, _color: _TINY_SVG)
    app = QApplication.instance() or QApplication([])
    assert app is not None

    small = formula_renderer.render_formula_pixmap(r"$x^2$", target_height=18, oversample=1.0)
    large = formula_renderer.render_formula_pixmap(r"$x^2$", target_height=42, oversample=2.0)

    assert small is not None
    assert large is not None
    assert small.logical_height == 18
    assert large.logical_height == 42
    assert large.logical_width > small.logical_width


def test_formula_renderer_prefers_mathjax_by_default(monkeypatch):
    from app.ui.proof import formula_renderer

    monkeypatch.delenv("OCR_FORMULA_RENDER", raising=False)
    monkeypatch.delenv("OCR_FORMULA_ENGINE", raising=False)
    monkeypatch.setattr(formula_renderer, "_render_mathjax_svg", lambda _latex, _color: _TINY_SVG)
    formula_renderer.clear_formula_render_cache()
    app = QApplication.instance() or QApplication([])
    assert app is not None

    result = formula_renderer.render_formula_pixmap(
        r"$\frac{a+b}{c+d} = \sum_{i=1}^{n} x_i$",
        target_height=30,
    )

    assert result is not None
    assert result.backend == "mathjax_svg"
    assert result.logical_height == 30


def test_formula_rendering_node_executable_can_be_overridden(monkeypatch, tmp_path):
    from app.ui.proof import formula_renderer

    node = tmp_path / ("node.exe")
    node.write_text("", encoding="utf-8")

    monkeypatch.setenv("OCR_MATHJAX_NODE_BIN", str(node))

    assert formula_renderer._mathjax_node_executable() == str(node)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not available")
def test_formula_rendering_mathjax_supports_aligned_environment(monkeypatch):
    from app.ui.proof import formula_renderer

    node_modules = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "formula"
        / "mathjax"
        / "node_modules"
    )
    if not node_modules.exists():
        pytest.skip("bundled mathjax resources are not available")

    monkeypatch.setenv("OCR_MATHJAX_NODE_MODULES", str(node_modules))
    formula_renderer.clear_formula_render_cache()

    svg = formula_renderer._render_mathjax_svg(
        r"\begin{aligned}y_{it}=\beta_0+\beta_k k_{it}\\+X_{it}\boldsymbol{\gamma}^{\prime}\end{aligned}",
        "#2C2C2C",
    )

    assert b"data-mjx-error" not in svg
    assert b"Unknown environment" not in svg
    assert b"data-background" not in svg
