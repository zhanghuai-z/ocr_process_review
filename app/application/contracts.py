"""Immutable application contracts for the layout workspace.

The application layer exposes layout facts as value objects.  Repository records
are accepted only by the projection factories; the resulting values contain no
runtime owner or persistence object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.project_session import PageRecord


@dataclass(frozen=True, slots=True)
class ImportFailureView:
    source_path: str
    error: str


@dataclass(frozen=True, slots=True)
class ImportCompletionView:
    page_uids: tuple[str, ...]
    failures: tuple[ImportFailureView, ...]

    @property
    def success_count(self) -> int:
        return len(self.page_uids)

    @property
    def failure_count(self) -> int:
        return len(self.failures)


def _stable_uid(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty stable UID")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: object, field_name: str) -> int:
    result = _non_negative_int(value, field_name)
    if result == 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _bbox(value: object, field_name: str = "bbox") -> BBox:
    if not isinstance(value, BBox):
        raise TypeError(f"{field_name} must be BBox")
    if value.w <= 0 or value.h <= 0:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _uids(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    result = tuple(values)
    for value in result:
        _stable_uid(value, field_name)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicate UIDs")
    return result


@dataclass(frozen=True, slots=True)
class BlockView:
    """A read-only application projection of one layout block."""

    block_uid: str
    page_uid: str
    block_type: BlockType
    bbox: BBox
    order: int
    source_label: str
    origin: BlockOrigin
    ocr_policy: OcrPolicy
    authorship: BlockSource

    def __post_init__(self) -> None:
        _stable_uid(self.block_uid, "block_uid")
        _stable_uid(self.page_uid, "page_uid")
        if not isinstance(self.block_type, BlockType):
            raise TypeError("block_type must be BlockType")
        _bbox(self.bbox)
        _non_negative_int(self.order, "order")
        _text(self.source_label, "source_label")
        if not isinstance(self.origin, BlockOrigin):
            raise TypeError("origin must be BlockOrigin")
        if not isinstance(self.ocr_policy, OcrPolicy):
            raise TypeError("ocr_policy must be OcrPolicy")
        if not isinstance(self.authorship, BlockSource):
            raise TypeError("authorship must be BlockSource")

    @property
    def uid(self) -> str:
        return self.block_uid

    @classmethod
    def from_snapshot(cls, page_uid: str, block: LayoutBlockSnapshot) -> "BlockView":
        if not isinstance(block, LayoutBlockSnapshot):
            raise TypeError("block must be LayoutBlockSnapshot")
        return cls(
            block_uid=block.uid,
            page_uid=_stable_uid(page_uid, "page_uid"),
            block_type=block.block_type,
            bbox=block.bbox,
            order=block.order,
            source_label=block.source_label,
            origin=block.origin,
            ocr_policy=block.ocr_policy,
            authorship=block.authorship,
        )


@dataclass(frozen=True, slots=True)
class PageView:
    """A read-only page projection containing its current layout snapshot."""

    page_uid: str
    project_uid: str
    image_path: str
    source_path: str
    cache_image_path: str
    thumbnail_path: str
    width: int
    height: int
    page_number: int
    source_page_index: int
    status: str
    error: str
    image_hash: str
    image_revision: int
    layout_revision: int | None
    artifact_uid: str
    source_engine: str
    source_run_id: str
    blocks: tuple[BlockView, ...]

    def __post_init__(self) -> None:
        _stable_uid(self.page_uid, "page_uid")
        _stable_uid(self.project_uid, "project_uid")
        for field_name in (
            "image_path",
            "source_path",
            "cache_image_path",
            "thumbnail_path",
            "status",
            "error",
            "image_hash",
            "artifact_uid",
            "source_engine",
            "source_run_id",
        ):
            _text(getattr(self, field_name), field_name)
        _positive_int(self.width, "width")
        _positive_int(self.height, "height")
        _positive_int(self.page_number, "page_number")
        _non_negative_int(self.source_page_index, "source_page_index")
        _non_negative_int(self.image_revision, "image_revision")
        if self.layout_revision is not None:
            _positive_int(self.layout_revision, "layout_revision")

        blocks = tuple(self.blocks)
        if any(not isinstance(block, BlockView) for block in blocks):
            raise TypeError("blocks must contain only BlockView values")
        if any(block.page_uid != self.page_uid for block in blocks):
            raise ValueError("page blocks must belong to the page view")
        if len({block.uid for block in blocks}) != len(blocks):
            raise ValueError("page view contains duplicate block UIDs")
        if tuple(block.order for block in blocks) != tuple(range(len(blocks))):
            raise ValueError("page view block order must match sequence")
        object.__setattr__(self, "blocks", blocks)

    @property
    def uid(self) -> str:
        return self.page_uid

    @property
    def block_uids(self) -> tuple[str, ...]:
        return tuple(block.uid for block in self.blocks)

    @classmethod
    def from_records(
        cls,
        page: PageRecord,
        layout: LayoutSnapshot | None,
    ) -> "PageView":
        if not isinstance(page, PageRecord):
            raise TypeError("page must be PageRecord")
        if layout is not None and not isinstance(layout, LayoutSnapshot):
            raise TypeError("layout must be LayoutSnapshot or None")
        if layout is not None and layout.page_uid != page.uid:
            raise ValueError("layout snapshot does not belong to page")

        return cls(
            page_uid=page.uid,
            project_uid=page.project_uid,
            image_path=page.image_path,
            source_path=page.source_path,
            cache_image_path=page.cache_image_path,
            thumbnail_path=page.thumbnail_path,
            width=page.width,
            height=page.height,
            page_number=page.page_number,
            source_page_index=page.source_page_index,
            status=page.status,
            error=page.error,
            image_hash=page.image_hash,
            image_revision=page.image_revision,
            layout_revision=layout.revision if layout is not None else None,
            artifact_uid=layout.artifact_uid if layout is not None else "",
            source_engine=layout.source_engine if layout is not None else "",
            source_run_id=layout.source_run_id if layout is not None else "",
            blocks=(
                tuple(
                    BlockView.from_snapshot(page.uid, block)
                    for block in layout.blocks
                )
                if layout is not None
                else ()
            ),
        )


@dataclass(frozen=True, slots=True)
class LayoutWorkspaceView:
    """The immutable read model for the layout workspace."""

    project_uid: str
    project_name: str
    pages: tuple[PageView, ...]

    def __post_init__(self) -> None:
        _stable_uid(self.project_uid, "project_uid")
        _text(self.project_name, "project_name")
        pages = tuple(self.pages)
        if any(not isinstance(page, PageView) for page in pages):
            raise TypeError("pages must contain only PageView values")
        if any(page.project_uid != self.project_uid for page in pages):
            raise ValueError("workspace pages must belong to the workspace project")
        if len({page.uid for page in pages}) != len(pages):
            raise ValueError("workspace contains duplicate page UIDs")
        object.__setattr__(self, "pages", pages)

    @property
    def page_uids(self) -> tuple[str, ...]:
        return tuple(page.uid for page in self.pages)


_EDIT_OPERATIONS = frozenset(
    {"draw", "delete", "move", "resize", "change_type", "change_types", "merge"}
)


@dataclass(frozen=True, slots=True)
class LayoutEditCommand:
    """An application-level layout edit intent addressed by stable IDs."""

    page_uid: str
    expected_revision: int
    op: str
    block_uid: str = ""
    block_uids: tuple[str, ...] = ()
    bbox: BBox | None = None
    block_type: BlockType | None = None
    source_label: str = ""
    new_block_uid: str = ""
    primary_block_uid: str = ""

    def __post_init__(self) -> None:
        _stable_uid(self.page_uid, "page_uid")
        _non_negative_int(self.expected_revision, "expected_revision")
        if self.op not in _EDIT_OPERATIONS:
            raise ValueError(f"unsupported layout edit operation: {self.op!r}")
        for field_name in ("block_uid", "new_block_uid", "primary_block_uid"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be str")
            if value and value != value.strip():
                raise ValueError(f"{field_name} must be a stable UID")
        _text(self.source_label, "source_label")

        block_uids = _uids(self.block_uids, "block_uids")
        object.__setattr__(self, "block_uids", block_uids)
        if self.bbox is not None:
            _bbox(self.bbox)
        if self.block_type is not None and not isinstance(self.block_type, BlockType):
            raise TypeError("block_type must be BlockType or None")

        if self.op == "draw":
            self._require_bbox()
            self._require_block_type()
        elif self.op in {"move", "resize"}:
            self._require_block_uid()
            self._require_bbox()
        elif self.op == "delete":
            self._require_block_uid()
        elif self.op == "change_type":
            self._require_block_uid()
            self._require_block_type()
        elif self.op == "change_types":
            if not self.block_uids:
                raise ValueError("change_types requires at least one block UID")
            self._require_block_type()
        elif self.op == "merge":
            if not self.block_uids:
                raise ValueError("merge requires at least one block UID")
            self._require_bbox()
            self._require_block_type()

    @property
    def operation(self) -> str:
        return self.op

    def _require_block_uid(self) -> None:
        if not self.block_uid:
            raise ValueError(f"{self.op} requires block_uid")

    def _require_bbox(self) -> None:
        if self.bbox is None:
            raise ValueError(f"{self.op} requires bbox")

    def _require_block_type(self) -> None:
        if self.block_type is None:
            raise ValueError(f"{self.op} requires block_type")

    @classmethod
    def draw(
        cls,
        page_uid: str,
        expected_revision: int,
        bbox: BBox,
        block_type: BlockType,
        source_label: str = "",
        *,
        new_block_uid: str = "",
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="draw",
            bbox=bbox,
            block_type=block_type,
            source_label=source_label,
            new_block_uid=new_block_uid,
        )

    @classmethod
    def delete(
        cls,
        page_uid: str,
        expected_revision: int,
        block_uid: str,
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="delete",
            block_uid=block_uid,
        )

    @classmethod
    def move(
        cls,
        page_uid: str,
        expected_revision: int,
        block_uid: str,
        bbox: BBox,
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="move",
            block_uid=block_uid,
            bbox=bbox,
        )

    @classmethod
    def resize(
        cls,
        page_uid: str,
        expected_revision: int,
        block_uid: str,
        bbox: BBox,
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="resize",
            block_uid=block_uid,
            bbox=bbox,
        )

    @classmethod
    def change_type(
        cls,
        page_uid: str,
        expected_revision: int,
        block_uid: str,
        block_type: BlockType,
        source_label: str = "",
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="change_type",
            block_uid=block_uid,
            block_type=block_type,
            source_label=source_label,
        )

    @classmethod
    def change_types(
        cls,
        page_uid: str,
        expected_revision: int,
        block_uids: Iterable[str],
        block_type: BlockType,
        source_label: str = "",
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="change_types",
            block_uids=tuple(block_uids),
            block_type=block_type,
            source_label=source_label,
        )

    @classmethod
    def merge(
        cls,
        page_uid: str,
        expected_revision: int,
        block_uids: Iterable[str],
        bbox: BBox,
        block_type: BlockType,
        source_label: str = "",
        *,
        primary_block_uid: str = "",
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="merge",
            block_uids=tuple(block_uids),
            bbox=bbox,
            block_type=block_type,
            source_label=source_label,
            primary_block_uid=primary_block_uid,
        )


@dataclass(frozen=True, slots=True)
class LayoutEditResult:
    """An immutable application result after a layout edit and CAS write."""

    command: LayoutEditCommand
    page_uid: str
    revision_before: int
    revision_after: int
    affected_block_uids: tuple[str, ...] = ()
    page_view: PageView | None = None
    accepted: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.command, LayoutEditCommand):
            raise TypeError("command must be LayoutEditCommand")
        _stable_uid(self.page_uid, "page_uid")
        if self.page_uid != self.command.page_uid:
            raise ValueError("result page_uid must match command page_uid")
        _non_negative_int(self.revision_before, "revision_before")
        _non_negative_int(self.revision_after, "revision_after")
        if self.revision_after < self.revision_before:
            raise ValueError("revision_after cannot precede revision_before")
        object.__setattr__(
            self,
            "affected_block_uids",
            _uids(self.affected_block_uids, "affected_block_uids"),
        )
        if self.page_view is not None:
            if not isinstance(self.page_view, PageView):
                raise TypeError("page_view must be PageView or None")
            if self.page_view.uid != self.page_uid:
                raise ValueError("result page_view must belong to page_uid")
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be bool")
        _text(self.reason, "reason")

    @property
    def op(self) -> str:
        return self.command.op

    @property
    def operation(self) -> str:
        return self.command.op

    @property
    def revision(self) -> int:
        return self.revision_after


_PROOF_OPERATIONS = frozenset({"replace_text", "replace_many", "set_status", "undo", "redo"})


@dataclass(frozen=True, slots=True)
class ProofEditCommand:
    """One proof mutation addressed only by stable business IDs and CAS facts."""

    proof_uid: str
    op: str
    expected_revision: int
    expected_fingerprint: str
    text_unit_uid: str = ""
    text: str = ""
    status: str = "modified"
    replacements: tuple[tuple[str, str], ...] = ()
    expected_unit_revision: int | None = None
    expected_unit_fingerprint: str | None = None

    def __post_init__(self) -> None:
        _stable_uid(self.proof_uid, "proof_uid")
        if self.op not in _PROOF_OPERATIONS:
            raise ValueError(f"unsupported proof edit operation: {self.op!r}")
        _non_negative_int(self.expected_revision, "expected_revision")
        _text(self.expected_fingerprint, "expected_fingerprint")
        _text(self.text, "text")
        _text(self.status, "status")
        if self.op in {"replace_text", "set_status"}:
            _stable_uid(self.text_unit_uid, "text_unit_uid")
        values = tuple(self.replacements)
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[1], str)
            for item in values
        ):
            raise TypeError("replacements must contain (text_unit_uid, text) pairs")
        for uid, _text_value in values:
            _stable_uid(uid, "replacement text_unit_uid")
        if len({uid for uid, _ in values}) != len(values):
            raise ValueError("replacements must not contain duplicate text unit UIDs")
        if self.op == "replace_many" and not values:
            raise ValueError("replace_many requires at least one replacement")
        object.__setattr__(self, "replacements", values)
        if self.expected_unit_revision is not None:
            _non_negative_int(self.expected_unit_revision, "expected_unit_revision")
        if self.expected_unit_fingerprint is not None:
            _text(self.expected_unit_fingerprint, "expected_unit_fingerprint")


@dataclass(frozen=True, slots=True)
class ProofBatchEditCommand:
    """One atomic application intent spanning several proof states."""

    commands: tuple[ProofEditCommand, ...]

    def __post_init__(self) -> None:
        values = tuple(self.commands)
        if not values:
            raise ValueError("proof batch edit requires at least one command")
        if any(not isinstance(command, ProofEditCommand) for command in values):
            raise TypeError("proof batch edit requires ProofEditCommand values")
        if any(command.op != "replace_many" for command in values):
            raise ValueError("proof batch edit supports replace_many commands only")
        if len({command.proof_uid for command in values}) != len(values):
            raise ValueError("proof batch edit contains duplicate proof UIDs")
        object.__setattr__(self, "commands", values)


@dataclass(frozen=True, slots=True)
class ProofEditResult:
    command: ProofEditCommand
    changed: bool
    proof_uid: str
    revision: int
    fingerprint: str
    changed_text_unit_uids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.command, ProofEditCommand):
            raise TypeError("command must be ProofEditCommand")
        _stable_uid(self.proof_uid, "proof_uid")
        _non_negative_int(self.revision, "revision")
        _text(self.fingerprint, "fingerprint")
        object.__setattr__(
            self,
            "changed_text_unit_uids",
            _uids(self.changed_text_unit_uids, "changed_text_unit_uids"),
        )


@dataclass(frozen=True, slots=True)
class ProofBatchEditResult:
    command: ProofBatchEditCommand
    results: tuple[ProofEditResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.command, ProofBatchEditCommand):
            raise TypeError("command must be ProofBatchEditCommand")
        values = tuple(self.results)
        if len(values) != len(self.command.commands):
            raise ValueError("proof batch result count must match command count")
        if any(not isinstance(result, ProofEditResult) for result in values):
            raise TypeError("results must contain ProofEditResult values")
        object.__setattr__(self, "results", values)

    @property
    def changed(self) -> bool:
        return any(result.changed for result in self.results)


__all__ = [
    "BlockView",
    "LayoutEditCommand",
    "LayoutEditResult",
    "LayoutWorkspaceView",
    "PageView",
    "ProofBatchEditCommand",
    "ProofBatchEditResult",
    "ProofEditCommand",
    "ProofEditResult",
]
