from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services import ImportService
from app.utils.image_io import read_cv_image


def test_read_cv_image_decodes_non_ascii_path(tmp_path):
    image_path = tmp_path / "中文目录" / "page.png"
    image_path.parent.mkdir()
    expected = np.full((6, 8, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", expected)

    assert ok
    encoded.tofile(str(image_path))

    actual = read_cv_image(image_path)

    assert actual is not None
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)


def test_read_cv_image_returns_none_for_missing_file(tmp_path):
    assert read_cv_image(tmp_path / "missing.png") is None


def test_import_materializes_non_ascii_source_to_ascii_cache_path(tmp_path):
    from PIL import Image

    source = tmp_path / "用户导入目录" / "中文样例.png"
    source.parent.mkdir()
    Image.new("RGB", (12, 8), color="white").save(source)

    result = ImportService(cache_dir=tmp_path / "work_cache").import_paths([str(source)])

    assert result.success_count == 1
    page = result.pages[0]
    assert page.source_path == str(source)
    assert page.display_image_path.endswith(".png")
    assert page.display_image_path.rsplit("/", 1)[-1].isascii()
    assert Path(page.display_image_path).name.startswith("image_")
