from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services import ImportJobRequest, ImportService
from app.models.project_session import DuplicateUidError, ProjectRecord, ProjectSession
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

    session = ProjectSession(ProjectRecord("project-1", "Image import"))
    service = ImportService(cache_dir=tmp_path / "work_cache")
    result = service.execute(ImportJobRequest("project-1", (str(source),)))
    service.commit(session, result)

    assert result.success_count == 1
    page = result.pages[0]
    assert page.source_path == str(source)
    assert page.image_path.endswith(".png")
    assert page.image_path.rsplit("/", 1)[-1].isascii()
    assert Path(page.image_path).name.startswith("page_")


def test_import_page_batch_rejects_duplicates_without_partial_adoption(tmp_path):
    from PIL import Image

    source = tmp_path / "page.png"
    Image.new("RGB", (12, 8), color="white").save(source)
    session = ProjectSession(ProjectRecord("project-1", "Atomic import"))
    service = ImportService(cache_dir=tmp_path / "work_cache")
    result = service.execute(ImportJobRequest("project-1", (str(source),)))

    with pytest.raises(DuplicateUidError):
        session.adopt_import_pages((result.pages[0], result.pages[0]))

    assert session.page_repository.all() == ()


def test_import_commit_rejects_result_prepared_against_stale_page_collection(tmp_path):
    from PIL import Image

    source = tmp_path / "page.png"
    Image.new("RGB", (12, 8), color="white").save(source)
    session = ProjectSession(ProjectRecord("project-1", "Stale import"))
    service = ImportService(cache_dir=tmp_path / "work_cache")
    result = service.execute(
        ImportJobRequest(
            "project-1",
            (str(source),),
            expected_page_uids=(),
        )
    )
    session.page_repository.put(result.pages[0], expected_revision=0)

    with pytest.raises(RuntimeError, match="stale page collection"):
        service.commit(session, result)

    assert session.page_repository.all() == (result.pages[0],)
