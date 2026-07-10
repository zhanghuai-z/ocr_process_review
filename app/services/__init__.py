"""导入服务：支持图片和 PDF 导入。

处理流程：
1. 校验路径存在
2. 按扩展名分类
3. 图片直接读取尺寸
4. PDF 用 PyMuPDF 渲染为 PNG 到缓存目录
5. 对每一页生成 Page 对象
6. 返回成功页和失败列表；不因一个文件失败中断全部导入
"""
from __future__ import annotations
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

from app.core.logging import get_logger
from app.models import BBox, Page, PageStatus

logger = get_logger(__name__)

# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
PDF_EXTENSION = ".pdf"

# 缓存子目录
CACHE_IMAGE_DIR = "images"
CACHE_THUMBNAIL_DIR = "thumbnails"


class ImportResult:
    """导入结果。"""
    def __init__(self) -> None:
        self.pages: List[Page] = []
        self.failed: List[Tuple[str, str]] = []  # (path, reason)

    def add_page(self, page: Page) -> None:
        self.pages.append(page)

    def add_failure(self, path: str, reason: str) -> None:
        self.failed.append((path, reason))

    @property
    def success_count(self) -> int:
        return len(self.pages)

    @property
    def failed_count(self) -> int:
        return len(self.failed)

    def extend(self, other: "ImportResult") -> None:
        self.pages.extend(other.pages)
        self.failed.extend(other.failed)


class ImportService:
    """导入服务。"""

    def __init__(self, cache_dir: str | Path | None = None):
        self._cache_dir: Optional[Path] = None
        if cache_dir:
            self._cache_dir = Path(cache_dir)

    def set_cache_dir(self, cache_dir: str | Path) -> None:
        self._cache_dir = Path(cache_dir)

    def import_paths(self, paths: List[str]) -> ImportResult:
        """导入多个路径（图片和/或 PDF）。

        Args:
            paths: 文件路径列表。

        Returns:
            ImportResult: 包含成功页和失败列表。
        """
        result = ImportResult()

        for path in paths:
            p = Path(path)
            if not p.exists():
                result.add_failure(path, f"文件不存在：{path}")
                continue

            suffix = p.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                try:
                    page = self._import_image(path, p)
                    result.add_page(page)
                except Exception as e:
                    logger.error("Image import failed: %s: %s", path, e)
                    result.add_failure(path, str(e))
            elif suffix == PDF_EXTENSION:
                try:
                    pdf_result = self._import_pdf(path, p)
                    result.extend(pdf_result)
                except Exception as e:
                    logger.error("PDF import failed: %s: %s", path, e)
                    result.add_failure(path, str(e))
            else:
                result.add_failure(path, f"不支持的文件格式：{suffix}")

        if result.failed:
            logger.warning(
                "Import completed with %d failures", result.failed_count
            )

        # 统一按最终导入顺序重排页码，避免多张图片都显示为第 1 页
        for page_number, page in enumerate(result.pages, start=1):
            page.page_number = page_number

        return result

    def _import_image(self, path: str, p: Path) -> Page:
        """处理单张图片。"""
        work_path = self._materialize_image(path, p)
        from PIL import Image
        with Image.open(work_path) as img:
            w, h = img.size
        thumb_path = self._generate_thumbnail(work_path, Path(work_path))

        return Page(
            image_path=work_path,
            width=w,
            height=h,
            page_number=1,
            source_path=path,
            source_type="image",
            source_page_index=1,
            cache_image_path=work_path,
            thumbnail_path=thumb_path or "",
            status=PageStatus.IMPORTED,
        )

    def _import_pdf(self, pdf_path: str, p: Path) -> ImportResult:
        """处理 PDF：每页渲染为图片。

        使用 PyMuPDF 将每页渲染为 PNG，输出到缓存目录。
        """
        result = ImportResult()
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)

        for index in range(len(doc)):
            try:
                page = doc[index]
                # Matrix(2, 2) ≈ 144 DPI 清晰度可调
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

                out_name = self._pdf_page_cache_name(p, index)
                out_path = self._get_cache_path() / out_name

                pix.save(str(out_path))

                # 获取图片尺寸
                from PIL import Image
                with Image.open(str(out_path)) as img:
                    w, h = img.size

                result.add_page(Page(
                    image_path=str(out_path),
                    width=w,
                    height=h,
                    page_number=index + 1,
                    source_path=pdf_path,
                    source_type="pdf",
                    source_page_index=index + 1,
                    cache_image_path=str(out_path),
                    thumbnail_path=self._generate_thumbnail(str(out_path), Path(out_name)) or "",
                    status=PageStatus.IMPORTED,
                ))
            except Exception as e:
                logger.error("PDF page render failed: page %d: %s", index + 1, e)
                result.add_failure(pdf_path, f"第 {index + 1} 页渲染失败：{e}")

        doc.close()
        return result

    @classmethod
    def _pdf_page_cache_name(cls, src: Path, page_index: int) -> str:
        digest = cls._source_digest(src)
        return f"page_{digest}_p{page_index + 1:04d}.png"

    def _materialize_image(self, src_path: str, src: Path) -> str:
        """生成统一工作图，消除 EXIF/格式差异带来的坐标漂移。"""
        if not self._cache_dir:
            return src_path

        from PIL import Image, ImageOps

        digest = self._source_digest(src)
        dst = self._get_cache_path() / f"image_{digest}.png"
        if not dst.exists():
            with Image.open(src_path) as img:
                normalized = ImageOps.exif_transpose(img)
                if normalized.mode not in ("RGB", "RGBA", "L"):
                    normalized = normalized.convert("RGB")
                elif normalized.mode == "RGBA":
                    normalized = normalized.convert("RGB")
                normalized.save(str(dst), "PNG")
        return str(dst)

    @staticmethod
    def _source_digest(src: Path) -> str:
        digest = hashlib.sha1()
        try:
            resolved = src.resolve()
        except OSError:
            resolved = src.absolute()
        digest.update(str(resolved).encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        with src.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:10]

    def _generate_thumbnail(self, src_path: str, src: Path) -> Optional[str]:
        """生成缩略图。"""
        if not self._cache_dir:
            return None
        try:
            from PIL import Image, ImageOps
            thumb_dir = self._get_thumbnail_path()
            thumb_path = thumb_dir / f"{src.stem}.png"

            if not thumb_path.exists():
                with Image.open(src_path) as img:
                    normalized = ImageOps.exif_transpose(img)
                    normalized.thumbnail((200, 200))
                    if normalized.mode not in ("RGB", "RGBA", "L"):
                        normalized = normalized.convert("RGB")
                    elif normalized.mode == "RGBA":
                        normalized = normalized.convert("RGB")
                    normalized.save(str(thumb_path), "PNG")

            return str(thumb_path)
        except Exception as e:
            logger.warning("Thumbnail generation failed: %s: %s", src_path, e)
            return None

    def _ensure_cache_dir(self) -> Path:
        """确保缓存目录存在。"""
        if not self._cache_dir:
            self._cache_dir = Path.cwd() / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    def _get_cache_path(self) -> Path:
        base = self._ensure_cache_dir()
        img_dir = base / CACHE_IMAGE_DIR
        img_dir.mkdir(parents=True, exist_ok=True)
        return img_dir

    def _get_thumbnail_path(self) -> Path:
        base = self._ensure_cache_dir()
        thumb_dir = base / CACHE_THUMBNAIL_DIR
        thumb_dir.mkdir(parents=True, exist_ok=True)
        return thumb_dir
