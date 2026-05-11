"""OCR Inspector — bbox crop utility.

Public API
----------
crop_bbox(image_path, bbox, *, padding=0) -> PIL.Image.Image
    Crop the given BBox region from the image at image_path.
    Uses Path.read_bytes() → PIL to bypass cv2/fopen non-ASCII path issues.

crop_polygon(image_path, polygon, *, padding=0) -> PIL.Image.Image
    Crop the tight bounding box of the polygon.

save_crop(image_path, bbox, output_path, *, padding=0) -> None
    Crop + save.

find_text_matches(doc, query, *, case_sensitive=False) -> list[TextMatch]
    Find all chars / lines in doc whose text contains query.
    Returns list of TextMatch(node, page, image_path, bbox, match_text, kind).
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from tools.ocr_inspector.models.ir import (
    BBox, CharNode, DocumentNode, LineNode, PageNode, Polygon,
)
from tools.ocr_inspector.core import query_text_search_index

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Image loading (non-ASCII path safe)
# ---------------------------------------------------------------------------

def _load_pil(image_path: str):
    """Load image via bytes so non-ASCII / Chinese paths work on Windows."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for crop operations. pip install Pillow") from exc
    raw = Path(image_path).read_bytes()
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ---------------------------------------------------------------------------
# Core crop functions
# ---------------------------------------------------------------------------

def crop_bbox(
    image_path: str,
    bbox: BBox,
    *,
    padding: int = 0,
) -> "PIL.Image.Image":  # type: ignore[name-defined]
    """Crop a BBox region from image_path.

    Args:
        image_path: Absolute path to image. Non-ASCII (Chinese) paths work.
        bbox: BBox in image pixel space (x, y, w, h).
        padding: Extra pixels to expand the crop on all sides.

    Returns:
        PIL Image of the cropped region (RGB).
    """
    img = _load_pil(image_path)
    W, H = img.size
    x1 = max(0, int(bbox.x) - padding)
    y1 = max(0, int(bbox.y) - padding)
    x2 = min(W, int(bbox.x2) + padding)
    y2 = min(H, int(bbox.y2) + padding)
    if x2 <= x1 or y2 <= y1:
        from PIL import Image
        return Image.new("RGB", (1, 1), (200, 200, 200))
    return img.crop((x1, y1, x2, y2))


def crop_polygon(
    image_path: str,
    polygon: Polygon,
    *,
    padding: int = 0,
) -> "PIL.Image.Image":  # type: ignore[name-defined]
    """Crop the tight bounding box of a polygon."""
    bbox = polygon.to_bbox()
    if bbox is None:
        raise ValueError("Polygon is empty or invalid")
    return crop_bbox(image_path, bbox, padding=padding)


def crop_node(
    image_path: str,
    node: Union[CharNode, LineNode],
    *,
    padding: int = 2,
) -> Optional["PIL.Image.Image"]:  # type: ignore[name-defined]
    """Crop the bbox of any IR node. Returns None if node has no bbox."""
    if node.polygon is not None:
        try:
            return crop_polygon(image_path, node.polygon, padding=padding)
        except ValueError:
            pass
    if node.bbox is not None:
        return crop_bbox(image_path, node.bbox, padding=padding)
    return None


def save_crop(
    image_path: str,
    bbox: BBox,
    output_path: str,
    *,
    padding: int = 0,
) -> None:
    """Crop bbox from image_path and save to output_path."""
    crop = crop_bbox(image_path, bbox, padding=padding)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)


# ---------------------------------------------------------------------------
# Text search
# ---------------------------------------------------------------------------

@dataclass
class TextMatch:
    """A single text match found in an OCR document."""
    node: Union[CharNode, LineNode]    # the IR node containing the match
    page: PageNode
    image_path: str
    bbox: Optional[BBox]
    match_text: str                    # the text of the node
    query: str                         # the search query
    kind: str                          # "char" or "line"
    identity: str                      # stable search-index identity

    def crop(self, *, padding: int = 4) -> Optional["PIL.Image.Image"]:  # type: ignore[name-defined]
        """Convenience: crop this match's bbox from the page image."""
        if not self.image_path:
            return None
        return crop_node(self.image_path, self.node, padding=padding)


def find_text_matches(
    doc: DocumentNode,
    query: str,
    *,
    case_sensitive: bool = False,
    search_chars: bool = True,
    search_lines: bool = True,
) -> List[TextMatch]:
    """Search all chars and lines in doc for query.

    Searches character nodes first (precise level), then line nodes.
    For char nodes, matches by char.char or char.token_text.
    For line nodes, matches by line.text.

    Args:
        doc: DocumentNode to search.
        query: Text to search for (substring match).
        case_sensitive: If False (default), case-insensitive matching.
        search_chars: Include char-level matches (from real word boxes).
        search_lines: Include line-level matches.

    Returns:
        List of TextMatch ordered by page → line → position.
    """
    if not query:
        return []

    return [
        TextMatch(
            node=entry.node,
            page=entry.page,
            image_path=entry.page.image_path or "",
            bbox=entry.bbox,
            match_text=entry.text,
            query=query,
            kind=entry.kind,
            identity=entry.identity,
        )
        for entry in query_text_search_index(
            doc,
            query,
            case_sensitive=case_sensitive,
            include_chars=search_chars,
            include_lines=search_lines,
        )
    ]


# ---------------------------------------------------------------------------
# PIL → QPixmap helper (for UI use)
# ---------------------------------------------------------------------------

def pil_to_qpixmap(pil_img):
    """Convert a PIL Image to QPixmap (for display in Qt widgets)."""
    from PIL.ImageQt import ImageQt
    from PySide6.QtGui import QPixmap
    qt_img = ImageQt(pil_img)
    return QPixmap.fromImage(qt_img)
