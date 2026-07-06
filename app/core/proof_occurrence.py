"""Runtime identity for proof occurrences.

The proof views need to point at the same mutable OCR facts without treating a
page text editor as the source of truth.  These DTOs are runtime-only: they are
compiled from Page/Block/Line/Char state and are not persisted directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Iterable

from app.core.proof_line_facts import proof_line_facts
from app.models import BBox, Block, Line, Page
from app.models.layout_block_view import LayoutBlockView, iter_page_layout_block_views
from app.models.ocr_character_observation import line_ocr_chars
from app.models.ocr_observation import block_ocr_line_observations_by_uid, line_ocr_bbox


@dataclass(frozen=True)
class ProofOccurrence:
    """One selectable proof occurrence in VProof/HProof views."""

    page_uid: str = ""
    page_id: int | None = None
    page_number: int = 0
    page_path: str = ""
    block_uid: str = ""
    block_id: int | None = None
    block_order: int = 0
    line_uid: str = ""
    line_id: int | None = None
    line_idx: int = 0
    line_signature: str = ""
    span_start: int = 0
    span_end: int = 0
    char_indices: tuple[int, ...] = tuple()
    index_key: str = ""
    text: str = ""
    bbox: BBox | None = None
    bbox_source: str = ""
    bbox_granularity: str = ""
    collection_kind: str = "char"
    confidence: float = 0.0

    @property
    def span_len(self) -> int:
        return max(0, self.span_end - self.span_start)


def line_signature(line: Line) -> str:
    """Return a stable digest of the line facts proof edits depend on."""

    facts = proof_line_facts(line)
    char_parts: list[tuple[Any, ...]] = []
    for idx, char in enumerate(line_ocr_chars(line)):
        bbox = char.bbox.to_dict() if char.bbox is not None else None
        char_parts.append((
            idx,
            char.uid,
            char.id,
            char.char,
            char.token_text,
            char.bbox_source,
            char.bbox_granularity,
            bbox,
            round(float(char.confidence or 0.0), 6),
        ))
    payload = repr((
        line.uid,
        line.id,
        facts.text,
        facts.status_value,
        facts.review_flags,
        tuple(char_parts),
    ))
    return sha1(payload.encode("utf-8")).hexdigest()


def proof_line_identity_key(
    page: Page,
    block: Block,
    line: Line,
    line_idx: int,
) -> tuple[object, ...]:
    """Return the runtime identity key used to match a proof line after refresh.

    Stable model UIDs are authoritative. Geometry is only a last-resort
    fallback for synthetic or not-yet-persisted runtime rows; it must not be
    treated as the normal identity path.
    """

    page_uid = str(getattr(page, "uid", "") or "")
    block_uid = str(getattr(block, "uid", "") or "")
    line_uid = str(getattr(line, "uid", "") or "")
    if line_idx >= 0 and page_uid and block_uid and line_uid:
        return ("uid", page_uid, block_uid, line_uid)
    bbox = line_ocr_bbox(line).normalize()
    return (
        "geometry",
        str(getattr(page, "display_image_path", "") or ""),
        str(getattr(page, "source_path", "") or ""),
        int(getattr(page, "source_page_index", 0) or 0),
        int(getattr(page, "page_number", 0) or 0),
        block.block_type.value,
        int(getattr(block, "order", 0) or 0),
        int(line_idx),
        int(bbox.x),
        int(bbox.y),
        int(bbox.w),
        int(bbox.h),
    )


def proof_page_identity_key(page: Page) -> tuple[object, ...]:
    """Return the proof runtime identity key for a page."""

    if page.uid:
        return ("uid", page.uid)
    if page.id is not None:
        return ("id", page.id)
    return ("path", page.display_image_path, page.page_number)


def proof_entry_page_identity_key(entry: Any) -> tuple[object, ...]:
    """Return the page identity key carried by a proof index entry."""

    page_uid = str(getattr(entry, "page_uid", "") or "")
    if page_uid:
        return ("uid", page_uid)
    page_id = getattr(entry, "page_id", None)
    if page_id is not None:
        return ("id", page_id)
    return (
        "path",
        str(getattr(entry, "page_path", "") or ""),
        int(getattr(entry, "page_number", 0) or 0),
    )


def occurrence_from_entry(entry: Any, page: Page, block: Block) -> ProofOccurrence:
    token = str(getattr(entry, "token_text", "") or getattr(entry, "char", "") or "")
    span_start = int(getattr(entry, "char_idx", 0) or 0)
    span_end = span_start + max(1, len(token))
    line = getattr(entry, "line", None)
    return ProofOccurrence(
        page_uid=str(getattr(page, "uid", "") or ""),
        page_id=getattr(page, "id", None),
        page_number=int(getattr(page, "page_number", 0) or 0),
        page_path=str(getattr(page, "display_image_path", "") or ""),
        block_uid=str(getattr(block, "uid", "") or ""),
        block_id=getattr(block, "id", None),
        block_order=int(getattr(block, "order", 0) or 0),
        line_uid=str(getattr(line, "uid", "") or ""),
        line_id=getattr(line, "id", None),
        line_idx=int(getattr(entry, "line_idx", 0) or 0),
        line_signature=str(getattr(entry, "line_signature", "") or ""),
        span_start=span_start,
        span_end=span_end,
        char_indices=(span_start,),
        index_key=str(getattr(entry, "char", "") or ""),
        text=token,
        bbox=getattr(entry, "bbox", None),
        bbox_source=str(getattr(entry, "bbox_source", "") or ""),
        bbox_granularity=str(getattr(entry, "bbox_granularity", "") or ""),
        collection_kind=str(getattr(entry, "collection_kind", "") or "char"),
        confidence=float(getattr(entry, "confidence", 0.0) or 0.0),
    )


def proof_occurrence_key(occurrence: ProofOccurrence) -> tuple[object, ...]:
    """Return a stable runtime key for one selectable occurrence.

    The key intentionally excludes ``line_signature`` and text.  Those values
    are mutable facts used by callers as conflict guards; occurrence identity
    should survive a legitimate edit or a line move inside the same page.
    """

    return (
        "occ",
        _identity_key(
            occurrence.page_uid,
            occurrence.page_id,
            (occurrence.page_path, occurrence.page_number),
        ),
        _identity_key(occurrence.block_uid, occurrence.block_id, occurrence.block_order),
        _identity_key(occurrence.line_uid, occurrence.line_id, occurrence.line_idx),
        max(0, int(occurrence.span_start)),
        max(0, int(occurrence.span_end)),
        tuple(int(idx) for idx in occurrence.char_indices),
    )


def resolve_occurrence_key_in_pages(
    pages: Iterable[Page],
    occurrence_key: tuple[object, ...],
) -> tuple[Page, Block, Line, int] | None:
    """Resolve an occurrence key back to its current owning line.

    Stable line identity is authoritative.  The block identity is used to prefer
    the original owner, but a line may legitimately move between blocks during
    layout/proof synchronization, so the resolver falls back to a page-wide
    line search before giving up.
    """

    parsed = _parse_occurrence_key(occurrence_key)
    if parsed is None:
        return None
    page_identity, block_identity, line_identity, _span_start, _span_end, _char_indices = parsed
    for page in pages:
        if not _matches_page_identity(page, page_identity):
            continue
        preferred = _resolve_line_in_matching_blocks(page, block_identity, line_identity)
        if preferred is not None:
            return preferred
        fallback = _resolve_line_any_block(page, line_identity)
        if fallback is not None:
            return fallback
    return None


def resolve_entry_owner(
    pages: Iterable[Page],
    entry: Any,
) -> tuple[Page, Block] | None:
    """Resolve a runtime char/token entry back to its owning page/block.

    Stable UIDs are authoritative, row IDs are the next fallback, and
    path/page-number/order matching exists only for synthetic or not-yet-
    persisted runtime entries.
    Keeping this here prevents VProof and future proof views from each carrying
    a slightly different owner-resolution policy.
    """

    for page in pages:
        if not _entry_matches_page(entry, page):
            continue
        for view in iter_page_layout_block_views(page):
            block = view.runtime_block
            if block is None:
                continue
            if not _entry_matches_block(entry, view):
                continue
            if _entry_line_in_block_uid(entry, view.uid):
                return page, block
        for view in iter_page_layout_block_views(page):
            block = view.runtime_block
            if block is None:
                continue
            if _entry_matches_block_order(entry, view) and _entry_line_in_block_uid(entry, view.uid):
                return page, block
    return None


def _entry_matches_page(entry: Any, page: Page) -> bool:
    entry_page_uid = str(getattr(entry, "page_uid", "") or "")
    if entry_page_uid:
        return page.uid == entry_page_uid
    entry_page_id = getattr(entry, "page_id", None)
    if entry_page_id is not None:
        return page.id == entry_page_id
    return (
        page.display_image_path == str(getattr(entry, "page_path", "") or "")
        and page.page_number == int(getattr(entry, "page_number", 0) or 0)
    )


def _entry_matches_block(entry: Any, view: LayoutBlockView) -> bool:
    block_uid = str(getattr(entry, "block_uid", "") or "")
    if block_uid:
        return view.uid == block_uid
    return _entry_matches_block_order(entry, view)


def _entry_matches_block_order(entry: Any, view: LayoutBlockView) -> bool:
    return view.order == int(getattr(entry, "block_order", 0) or 0)


def _entry_line_in_block_uid(entry: Any, block_uid: str) -> bool:
    line = getattr(entry, "line", None)
    return line is not None and any(
        candidate is line for candidate in block_ocr_line_observations_by_uid(block_uid)
    )


def _identity_key(uid: str, row_id: int | None, fallback: object) -> tuple[object, ...]:
    if uid:
        return ("uid", uid)
    if row_id is not None:
        return ("id", row_id)
    return ("fallback", fallback)


def _parse_occurrence_key(
    key: tuple[object, ...],
) -> tuple[
    tuple[object, ...],
    tuple[object, ...],
    tuple[object, ...],
    int,
    int,
    tuple[int, ...],
] | None:
    if len(key) != 7 or key[0] != "occ":
        return None
    page_identity, block_identity, line_identity = key[1], key[2], key[3]
    if not (
        isinstance(page_identity, tuple)
        and isinstance(block_identity, tuple)
        and isinstance(line_identity, tuple)
    ):
        return None
    try:
        span_start = int(key[4])
        span_end = int(key[5])
        char_indices = tuple(int(idx) for idx in key[6])
    except (TypeError, ValueError):
        return None
    return page_identity, block_identity, line_identity, span_start, span_end, char_indices


def _matches_page_identity(page: Page, identity: tuple[object, ...]) -> bool:
    kind = identity[0] if identity else ""
    if kind == "uid":
        return page.uid == identity[1]
    if kind == "id":
        return page.id == identity[1]
    if kind == "fallback":
        path, page_number = identity[1]
        return page.display_image_path == path and page.page_number == page_number
    return False


def _matches_block_identity(view: LayoutBlockView, identity: tuple[object, ...]) -> bool:
    kind = identity[0] if identity else ""
    if kind == "uid":
        return view.uid == identity[1]
    if kind == "id":
        block = view.runtime_block
        return block is not None and block.id == identity[1]
    if kind == "fallback":
        return view.order == identity[1]
    return False


def _matches_line_identity(line: Line, line_idx: int, identity: tuple[object, ...]) -> bool:
    kind = identity[0] if identity else ""
    if kind == "uid":
        return line.uid == identity[1]
    if kind == "id":
        return line.id == identity[1]
    if kind == "fallback":
        return line_idx == identity[1]
    return False


def _resolve_line_in_matching_blocks(
    page: Page,
    block_identity: tuple[object, ...],
    line_identity: tuple[object, ...],
) -> tuple[Page, Block, Line, int] | None:
    for view in iter_page_layout_block_views(page):
        block = view.runtime_block
        if block is None:
            continue
        if not _matches_block_identity(view, block_identity):
            continue
        for line_idx, line in enumerate(block_ocr_line_observations_by_uid(view.uid)):
            if _matches_line_identity(line, line_idx, line_identity):
                return page, block, line, line_idx
    return None


def _resolve_line_any_block(
    page: Page,
    line_identity: tuple[object, ...],
) -> tuple[Page, Block, Line, int] | None:
    kind = line_identity[0] if line_identity else ""
    if kind not in {"uid", "id"}:
        return None
    for view in iter_page_layout_block_views(page):
        block = view.runtime_block
        if block is None:
            continue
        for line_idx, line in enumerate(block_ocr_line_observations_by_uid(view.uid)):
            if _matches_line_identity(line, line_idx, line_identity):
                return page, block, line, line_idx
    return None
