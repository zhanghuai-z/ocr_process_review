"""Import source files into project-scoped immutable page records."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from app.models.entity_id import new_entity_uid
from app.models.project_session import PageRecord, ProjectSession


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
PDF_EXTENSION = ".pdf"


@dataclass(frozen=True, slots=True)
class ImportFailure:
    source_path: str
    error: str


@dataclass(frozen=True, slots=True)
class ImportResult:
    project_uid: str
    expected_page_uids: tuple[str, ...]
    pages: tuple[PageRecord, ...]
    failures: tuple[ImportFailure, ...]

    @property
    def success_count(self) -> int:
        return len(self.pages)

    @property
    def failure_count(self) -> int:
        return len(self.failures)


@dataclass(frozen=True, slots=True)
class ImportJobRequest:
    project_uid: str
    paths: tuple[str, ...]
    first_page_number: int = 1
    expected_page_uids: tuple[str, ...] = ()


class ImportService:
    """Decode immutable import requests and commit validated page batches."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

    def execute(self, request: ImportJobRequest) -> ImportResult:
        """Decode import sources without mutating a project session."""
        if not isinstance(request, ImportJobRequest):
            raise TypeError("import execution requires ImportJobRequest")
        pages: list[PageRecord] = []
        failures: list[ImportFailure] = []
        for raw_path in request.paths:
            path = Path(raw_path)
            if not path.is_file():
                failures.append(ImportFailure(str(raw_path), "source file does not exist"))
                continue
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                try:
                    page = self._import_image(
                        request.project_uid,
                        path,
                        page_number=request.first_page_number + len(pages),
                        source_page_index=0,
                    )
                except Exception as exc:
                    failures.append(ImportFailure(str(path), str(exc)))
                else:
                    pages.append(page)
                continue
            if suffix == PDF_EXTENSION:
                try:
                    imported = self._import_pdf(
                        request.project_uid,
                        path,
                        first_page_number=request.first_page_number + len(pages),
                    )
                except Exception as exc:
                    failures.append(ImportFailure(str(path), str(exc)))
                else:
                    pages.extend(imported.pages)
                    failures.extend(imported.failures)
                continue
            failures.append(ImportFailure(str(path), f"unsupported source format: {suffix}"))

        return ImportResult(
            project_uid=request.project_uid,
            expected_page_uids=request.expected_page_uids,
            pages=tuple(pages),
            failures=tuple(failures),
        )

    @staticmethod
    def commit(session: ProjectSession, result: ImportResult) -> ImportResult:
        """Adopt decoded page records through the page repository boundary."""
        if not isinstance(result, ImportResult):
            raise TypeError("import commit requires ImportResult")
        if result.project_uid != session.project_uid:
            raise ValueError("import result belongs to another project")
        current_page_uids = tuple(page.uid for page in session.page_repository.all())
        if current_page_uids != result.expected_page_uids:
            raise RuntimeError("import result belongs to a stale page collection")
        for page in result.pages:
            if page.project_uid != session.project_uid:
                raise ValueError("import result belongs to another project")
        session.adopt_import_pages(result.pages)
        return result

    def _import_image(
        self,
        project_uid: str,
        source_path: Path,
        *,
        page_number: int,
        source_page_index: int,
    ) -> PageRecord:
        page_uid = new_entity_uid("page")
        image_path = self._materialize_image(source_path, page_uid)
        width, height = self._image_size(image_path)
        thumbnail_path = self._materialize_thumbnail(image_path, page_uid)
        page = PageRecord(
            project_uid=project_uid,
            uid=page_uid,
            image_path=str(image_path),
            source_path=str(source_path),
            cache_image_path=str(image_path) if self._cache_dir is not None else "",
            thumbnail_path=thumbnail_path,
            width=width,
            height=height,
            page_number=page_number,
            source_page_index=source_page_index,
            status="imported",
            error="",
            image_hash=_sha256_file(image_path),
            image_revision=1,
        )
        return page

    def _import_pdf(
        self,
        project_uid: str,
        source_path: Path,
        *,
        first_page_number: int,
    ) -> ImportResult:
        import fitz

        if self._cache_dir is None:
            self._cache_dir = source_path.parent / f".{source_path.stem}.ocr-cache"
        pages: list[PageRecord] = []
        failures: list[ImportFailure] = []
        document = fitz.open(str(source_path))
        try:
            for source_page_index in range(len(document)):
                page_number = first_page_number + len(pages)
                page_uid = new_entity_uid("page")
                try:
                    rendered = document.load_page(source_page_index).get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        alpha=False,
                    )
                    image_path = self._pdf_image_path(page_uid)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    rendered.save(str(image_path))
                    width, height = self._image_size(image_path)
                    thumbnail_path = self._materialize_thumbnail(image_path, page_uid)
                    page = PageRecord(
                        project_uid=project_uid,
                        uid=page_uid,
                        image_path=str(image_path),
                        source_path=str(source_path),
                        cache_image_path=str(image_path),
                        thumbnail_path=thumbnail_path,
                        width=width,
                        height=height,
                        page_number=page_number,
                        source_page_index=source_page_index,
                        status="imported",
                        error="",
                        image_hash=_sha256_file(image_path),
                        image_revision=1,
                    )
                except Exception as exc:
                    failures.append(
                        ImportFailure(
                            str(source_path),
                            f"PDF page {source_page_index + 1} import failed: {exc}",
                        )
                    )
                else:
                    pages.append(page)
        finally:
            document.close()
        return ImportResult(
            project_uid=project_uid,
            expected_page_uids=(),
            pages=tuple(pages),
            failures=tuple(failures),
        )

    def _materialize_image(self, source_path: Path, page_uid: str) -> Path:
        if self._cache_dir is None:
            return source_path.resolve()
        destination = self._cache_dir / "images" / f"{page_uid}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image, ImageOps

        with Image.open(source_path) as image:
            normalized = ImageOps.exif_transpose(image)
            if normalized.mode not in {"RGB", "L"}:
                normalized = normalized.convert("RGB")
            normalized.save(destination, format="PNG")
        return destination

    @staticmethod
    def _image_size(path: Path) -> tuple[int, int]:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("source image has invalid dimensions")
        return width, height

    def _materialize_thumbnail(self, image_path: Path, page_uid: str) -> str:
        if self._cache_dir is None:
            return ""
        from PIL import Image

        destination = self._cache_dir / "thumbnails" / f"{page_uid}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            thumbnail = image.copy()
            thumbnail.thumbnail((200, 200))
            if thumbnail.mode not in {"RGB", "L"}:
                thumbnail = thumbnail.convert("RGB")
            thumbnail.save(destination, format="PNG")
        return str(destination)

    def _pdf_image_path(self, page_uid: str) -> Path:
        if self._cache_dir is None:
            raise ValueError("PDF import requires a cache_dir")
        return self._cache_dir / "images" / f"{page_uid}.png"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["ImportFailure", "ImportResult", "ImportService"]
