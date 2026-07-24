"""Pure layout edits over immutable page layout snapshots."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from app.models.entity_id import ensure_entity_uid
from app.models.enums import BlockSource, BlockType, OcrPolicy
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.geometry import BBox
from app.models.layout_origin import BlockOrigin
from app.models.project_session import RevisionConflictError


_DEFAULT_OCR_POLICY: dict[BlockType, OcrPolicy] = {
    BlockType.TEXT: OcrPolicy.TEXT_OCR,
    BlockType.TITLE: OcrPolicy.TEXT_OCR,
    BlockType.FIGURE_CAPTION: OcrPolicy.TEXT_OCR,
    BlockType.TABLE_CAPTION: OcrPolicy.TEXT_OCR,
    BlockType.REFERENCE: OcrPolicy.TEXT_OCR,
    BlockType.EQUATION: OcrPolicy.PRESERVE_AS_FORMULA,
    BlockType.TABLE: OcrPolicy.PRESERVE_AS_TABLE,
    BlockType.FIGURE: OcrPolicy.SKIP,
    BlockType.UNKNOWN: OcrPolicy.SKIP,
}


def _copy_bbox(bbox: BBox | None) -> BBox | None:
    if bbox is None:
        return None
    return BBox(x=bbox.x, y=bbox.y, w=bbox.w, h=bbox.h)


def _copy_origin(origin: BlockOrigin) -> BlockOrigin:
    return replace(origin, original_bbox=_copy_bbox(origin.original_bbox))


def _copy_block(
    block: LayoutBlockSnapshot,
    **changes: Any,
) -> LayoutBlockSnapshot:
    values = {
        "block_type": block.block_type,
        "bbox": _copy_bbox(block.bbox),
        "order": block.order,
        "source_label": block.source_label,
        "origin": _copy_origin(block.origin),
        "ocr_policy": block.ocr_policy,
        "authorship": block.authorship,
        "uid": block.uid,
    }
    values.update(changes)
    if values["bbox"] is None:
        raise ValueError("layout block bbox cannot be None")
    if values["origin"] is None:
        raise ValueError("layout block origin cannot be None")
    return LayoutBlockSnapshot(**values)


@dataclass(frozen=True, slots=True)
class LayoutEditCommand:
    """An edit request addressed to one page-layout revision."""

    page_uid: str
    expected_revision: int
    op: str
    block_uid: str = ""
    block_uids: tuple[str, ...] = ()
    snapshot_blocks: tuple[LayoutBlockSnapshot, ...] = ()
    bbox: BBox | None = None
    block_type: BlockType | None = None
    source_label: str = ""
    new_block_uid: str = ""
    primary_block_uid: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.page_uid, str) or not self.page_uid.strip():
            raise ValueError("page_uid must be a non-empty stable UID")
        if (
            isinstance(self.expected_revision, bool)
            or not isinstance(self.expected_revision, int)
            or self.expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        if not isinstance(self.op, str) or not self.op.strip():
            raise ValueError("op must be non-empty text")
        if not isinstance(self.source_label, str):
            raise TypeError("source_label must be str")
        if not isinstance(self.block_uid, str):
            raise TypeError("block_uid must be str")
        if not isinstance(self.new_block_uid, str):
            raise TypeError("new_block_uid must be str")
        if not isinstance(self.primary_block_uid, str):
            raise TypeError("primary_block_uid must be str")
        if self.op == "create_block" and not self.new_block_uid:
            object.__setattr__(self, "new_block_uid", ensure_entity_uid("", "block"))

        block_uids = tuple(self.block_uids)
        if any(not isinstance(uid, str) for uid in block_uids):
            raise TypeError("block_uids must contain only strings")
        object.__setattr__(self, "block_uids", block_uids)

        snapshot_blocks = tuple(self.snapshot_blocks)
        if any(not isinstance(block, LayoutBlockSnapshot) for block in snapshot_blocks):
            raise TypeError("snapshot_blocks must contain only LayoutBlockSnapshot values")
        object.__setattr__(
            self,
            "snapshot_blocks",
            tuple(_copy_block(block) for block in snapshot_blocks),
        )
        if self.bbox is not None:
            if not isinstance(self.bbox, BBox):
                raise TypeError("bbox must be BBox or None")
            object.__setattr__(self, "bbox", _copy_bbox(self.bbox))

    @classmethod
    def create_block(
        cls,
        page_uid: str,
        expected_revision: int,
        bbox: BBox,
        block_type: BlockType,
        source_label: str,
        *,
        new_block_uid: str = "",
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="create_block",
            bbox=bbox,
            block_type=block_type,
            source_label=source_label,
            new_block_uid=new_block_uid,
        )

    @classmethod
    def delete_block(
        cls,
        page_uid: str,
        expected_revision: int,
        block_uid: str,
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="delete_block",
            block_uid=block_uid,
        )

    @classmethod
    def change_kind(
        cls,
        page_uid: str,
        expected_revision: int,
        block_uid: str,
        *,
        block_type: BlockType,
        source_label: str,
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="change_kind",
            block_uid=block_uid,
            block_type=block_type,
            source_label=source_label,
        )

    @classmethod
    def change_kinds(
        cls,
        page_uid: str,
        expected_revision: int,
        block_uids: Iterable[str],
        *,
        block_type: BlockType,
        source_label: str,
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="change_kinds",
            block_uids=tuple(block_uids),
            block_type=block_type,
            source_label=source_label,
        )

    @classmethod
    def resize_block(
        cls,
        page_uid: str,
        expected_revision: int,
        block_uid: str,
        *,
        bbox: BBox,
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="resize_block",
            block_uid=block_uid,
            bbox=bbox,
        )

    @classmethod
    def merge_blocks(
        cls,
        page_uid: str,
        expected_revision: int,
        block_uids: Iterable[str],
        bbox: BBox,
        *,
        block_type: BlockType,
        source_label: str,
        primary_block_uid: str = "",
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="merge_blocks",
            block_uids=tuple(block_uids),
            bbox=bbox,
            block_type=block_type,
            source_label=source_label,
            primary_block_uid=primary_block_uid,
        )

    @classmethod
    def restore_blocks(
        cls,
        page_uid: str,
        expected_revision: int,
        blocks: Iterable[LayoutBlockSnapshot],
    ) -> "LayoutEditCommand":
        return cls(
            page_uid=page_uid,
            expected_revision=expected_revision,
            op="restore_blocks",
            snapshot_blocks=tuple(blocks),
        )


@dataclass(frozen=True, slots=True)
class OcrInvalidationDescriptor:
    """The OCR work that a caller must discard or recompute after an edit."""

    page_uid: str
    block_uids: tuple[str, ...]
    reason: str

    @property
    def affected_block_uids(self) -> tuple[str, ...]:
        return self.block_uids

    @property
    def invalidated_block_uids(self) -> tuple[str, ...]:
        return self.block_uids


@dataclass(frozen=True, slots=True)
class LayoutEditResult:
    """The complete value result of applying one layout command."""

    op: str
    snapshot: LayoutSnapshot
    affected_block_uids: tuple[str, ...]
    before: dict[str, Any]
    after: dict[str, Any]
    ocr_invalidation: OcrInvalidationDescriptor

    @property
    def new_snapshot(self) -> LayoutSnapshot:
        return self.snapshot

    @property
    def before_values(self) -> dict[str, Any]:
        return self.before

    @property
    def after_values(self) -> dict[str, Any]:
        return self.after


class LayoutEditService:
    """Apply layout commands without touching runtime state or stores."""

    def apply(
        self,
        current: LayoutSnapshot,
        command: LayoutEditCommand,
    ) -> LayoutEditResult:
        if not isinstance(current, LayoutSnapshot):
            raise TypeError("current must be LayoutSnapshot")
        if not isinstance(command, LayoutEditCommand):
            raise TypeError("command must be LayoutEditCommand")
        if current.page_uid != command.page_uid:
            raise ValueError(
                f"layout command page {command.page_uid!r} does not match "
                f"snapshot page {current.page_uid!r}"
            )
        if current.revision != command.expected_revision:
            raise RevisionConflictError(
                f"layout revision mismatch: expected {command.expected_revision}, "
                f"current {current.revision}"
            )

        self._validate_snapshot_blocks(current.blocks)
        if command.op == "create_block":
            return self._create_block(current, command)
        if command.op == "delete_block":
            return self._delete_block(current, command)
        if command.op == "change_kind":
            return self._change_kind(current, command)
        if command.op == "change_kinds":
            return self._change_kinds(current, command)
        if command.op == "resize_block":
            return self._resize_block(current, command)
        if command.op == "merge_blocks":
            return self._merge_blocks(current, command)
        if command.op == "restore_blocks":
            return self._restore_blocks(current, command)
        raise ValueError(f"Unsupported layout edit command: {command.op}")

    def _create_block(
        self,
        current: LayoutSnapshot,
        command: LayoutEditCommand,
    ) -> LayoutEditResult:
        bbox = self._require_bbox(command)
        block_type = self._require_block_type(command)
        uid = command.new_block_uid
        if uid in {block.uid for block in current.blocks}:
            raise ValueError(f"layout block uid {uid!r} already exists")
        new_block = LayoutBlockSnapshot(
            block_type=block_type,
            bbox=_copy_bbox(bbox),
            order=len(current.blocks),
            source_label=command.source_label,
            origin=BlockOrigin(
                created_by=BlockSource.MANUAL_DRAW.value,
                source_engine="layout_edit",
                vendor_label=command.source_label,
                original_bbox=_copy_bbox(bbox),
                original_kind=block_type,
            ),
            ocr_policy=self._default_ocr_policy(block_type),
            authorship=BlockSource.MANUAL_DRAW,
            uid=uid,
        )
        next_blocks = self._normalize_blocks((*current.blocks, new_block))
        return self._result(
            current,
            command,
            next_blocks,
            affected_block_uids=(uid,),
            before={},
            after={"block": self.snapshot_block_state(self._block_by_uid(next_blocks, uid))},
            reason="layout_block_created",
        )

    def _delete_block(
        self,
        current: LayoutSnapshot,
        command: LayoutEditCommand,
    ) -> LayoutEditResult:
        block_uid = self._require_block_uid(command)
        block = self._block_by_uid(current.blocks, block_uid)
        next_blocks = self._normalize_blocks(
            candidate for candidate in current.blocks if candidate.uid != block_uid
        )
        return self._result(
            current,
            command,
            next_blocks,
            affected_block_uids=(block_uid,),
            before={"block": self.snapshot_block_state(block)},
            after={},
            reason="layout_block_deleted",
        )

    def _change_kind(
        self,
        current: LayoutSnapshot,
        command: LayoutEditCommand,
    ) -> LayoutEditResult:
        block_uid = self._require_block_uid(command)
        block_type = self._require_block_type(command)
        before_block = self._block_by_uid(current.blocks, block_uid)
        changed = _copy_block(
            before_block,
            block_type=block_type,
            source_label=command.source_label,
            ocr_policy=self._default_ocr_policy(block_type),
            authorship=BlockSource.USER_EDITED,
        )
        next_blocks = self._normalize_blocks(
            changed if candidate.uid == block_uid else candidate
            for candidate in current.blocks
        )
        after_block = self._block_by_uid(next_blocks, block_uid)
        return self._result(
            current,
            command,
            next_blocks,
            affected_block_uids=(block_uid,),
            before={"block": self.snapshot_block_state(before_block)},
            after={"block": self.snapshot_block_state(after_block)},
            reason="layout_block_kind_changed",
        )

    def _change_kinds(
        self,
        current: LayoutSnapshot,
        command: LayoutEditCommand,
    ) -> LayoutEditResult:
        block_uids = self._require_block_uids(command, "change_kinds")
        block_type = self._require_block_type(command)
        before_blocks = tuple(self._block_by_uid(current.blocks, uid) for uid in block_uids)
        selected = set(block_uids)
        next_blocks = self._normalize_blocks(
            _copy_block(
                candidate,
                block_type=block_type,
                source_label=command.source_label,
                ocr_policy=self._default_ocr_policy(block_type),
                authorship=BlockSource.USER_EDITED,
            )
            if candidate.uid in selected
            else candidate
            for candidate in current.blocks
        )
        after_blocks = tuple(self._block_by_uid(next_blocks, uid) for uid in block_uids)
        return self._result(
            current,
            command,
            next_blocks,
            affected_block_uids=block_uids,
            before={"blocks": [self.snapshot_block_state(block) for block in before_blocks]},
            after={"blocks": [self.snapshot_block_state(block) for block in after_blocks]},
            reason="layout_block_kinds_changed",
        )

    def _resize_block(
        self,
        current: LayoutSnapshot,
        command: LayoutEditCommand,
    ) -> LayoutEditResult:
        block_uid = self._require_block_uid(command)
        bbox = self._require_bbox(command)
        before_block = self._block_by_uid(current.blocks, block_uid)
        resized = _copy_block(
            before_block,
            bbox=_copy_bbox(bbox),
            authorship=BlockSource.USER_EDITED,
        )
        next_blocks = self._normalize_blocks(
            resized if candidate.uid == block_uid else candidate
            for candidate in current.blocks
        )
        after_block = self._block_by_uid(next_blocks, block_uid)
        return self._result(
            current,
            command,
            next_blocks,
            affected_block_uids=(block_uid,),
            before={"block": self.snapshot_block_state(before_block)},
            after={"block": self.snapshot_block_state(after_block)},
            reason="layout_block_resized",
        )

    def _merge_blocks(
        self,
        current: LayoutSnapshot,
        command: LayoutEditCommand,
    ) -> LayoutEditResult:
        block_uids = self._require_merge_uids(command)
        block_by_uid = {block.uid: block for block in current.blocks}
        selected = [block_by_uid[uid] for uid in block_uids]
        selected.sort(key=lambda block: (block.order, current.blocks.index(block)))
        primary_uid = command.primary_block_uid or selected[0].uid
        if primary_uid not in {block.uid for block in selected}:
            raise ValueError("primary_block_uid must identify one merged block")
        if selected[0].uid != primary_uid:
            selected = [block_by_uid[primary_uid], *[block for block in selected if block.uid != primary_uid]]
        selected_uids = tuple(block.uid for block in selected)
        primary = block_by_uid[primary_uid]
        block_type = self._require_block_type(command)
        bbox = self._require_bbox(command)
        merged_bbox = self._merge_bbox(bbox, (block.bbox for block in selected))
        merged = _copy_block(
            primary,
            block_type=block_type,
            bbox=merged_bbox,
            source_label=command.source_label,
            ocr_policy=self._default_ocr_policy(block_type),
            authorship=BlockSource.USER_EDITED,
        )
        remove_uids = set(selected_uids) - {primary_uid}
        next_blocks = self._normalize_blocks(
            merged if candidate.uid == primary_uid else candidate
            for candidate in current.blocks
            if candidate.uid not in remove_uids
        )
        after_block = self._block_by_uid(next_blocks, primary_uid)
        return self._result(
            current,
            command,
            next_blocks,
            affected_block_uids=selected_uids,
            before={
                "blocks": [self.snapshot_block_state(block) for block in selected],
            },
            after={"block": self.snapshot_block_state(after_block)},
            reason="layout_blocks_merged",
        )

    def _restore_blocks(
        self,
        current: LayoutSnapshot,
        command: LayoutEditCommand,
    ) -> LayoutEditResult:
        self._validate_snapshot_blocks(command.snapshot_blocks)
        next_blocks = self._normalize_blocks(command.snapshot_blocks)
        affected = self._unique_uids(
            (*[block.uid for block in current.blocks], *[block.uid for block in next_blocks])
        )
        return self._result(
            current,
            command,
            next_blocks,
            affected_block_uids=affected,
            before={
                "blocks": [self.snapshot_block_state(block) for block in current.blocks],
            },
            after={
                "blocks": [self.snapshot_block_state(block) for block in next_blocks],
            },
            reason="layout_snapshot_restored",
        )

    def _result(
        self,
        current: LayoutSnapshot,
        command: LayoutEditCommand,
        blocks: tuple[LayoutBlockSnapshot, ...],
        *,
        affected_block_uids: tuple[str, ...],
        before: dict[str, Any],
        after: dict[str, Any],
        reason: str,
    ) -> LayoutEditResult:
        next_snapshot = LayoutSnapshot(
            page_uid=current.page_uid,
            revision=current.revision + 1,
            artifact_uid=current.artifact_uid,
            source_engine="layout_edit",
            source_run_id=f"layout-edit:{current.page_uid}:{current.revision + 1}",
            blocks=tuple(_copy_block(block) for block in blocks),
        )
        affected = tuple(affected_block_uids)
        return LayoutEditResult(
            op=command.op,
            snapshot=next_snapshot,
            affected_block_uids=affected,
            before=before,
            after=after,
            ocr_invalidation=OcrInvalidationDescriptor(
                page_uid=current.page_uid,
                block_uids=affected,
                reason=reason,
            ),
        )

    @staticmethod
    def _default_ocr_policy(block_type: BlockType) -> OcrPolicy:
        try:
            return _DEFAULT_OCR_POLICY[block_type]
        except KeyError as exc:
            raise ValueError(f"unsupported block type: {block_type!r}") from exc

    @staticmethod
    def _require_block_uid(command: LayoutEditCommand) -> str:
        if not command.block_uid:
            raise ValueError(f"{command.op} requires a block uid")
        return command.block_uid

    @staticmethod
    def _require_bbox(command: LayoutEditCommand) -> BBox:
        if command.bbox is None:
            raise ValueError(f"{command.op} requires a bbox")
        return command.bbox

    @staticmethod
    def _require_block_type(command: LayoutEditCommand) -> BlockType:
        if command.block_type is None:
            raise ValueError(f"{command.op} requires a block type")
        return command.block_type

    @staticmethod
    def _require_merge_uids(command: LayoutEditCommand) -> tuple[str, ...]:
        if not command.block_uids:
            raise ValueError("merge_blocks requires at least one block uid")
        if any(not uid for uid in command.block_uids):
            raise ValueError("merge_blocks requires non-empty block uids")
        if len(set(command.block_uids)) != len(command.block_uids):
            raise ValueError("merge_blocks does not accept duplicate block uids")
        return command.block_uids

    @staticmethod
    def _require_block_uids(command: LayoutEditCommand, operation: str) -> tuple[str, ...]:
        if not command.block_uids:
            raise ValueError(f"{operation} requires at least one block uid")
        if any(not uid for uid in command.block_uids):
            raise ValueError(f"{operation} requires non-empty block uids")
        if len(set(command.block_uids)) != len(command.block_uids):
            raise ValueError(f"{operation} does not accept duplicate block uids")
        return command.block_uids

    @staticmethod
    def _validate_snapshot_blocks(blocks: Iterable[LayoutBlockSnapshot]) -> None:
        seen: set[str] = set()
        for block in blocks:
            if not isinstance(block, LayoutBlockSnapshot):
                raise TypeError("layout snapshots may contain only LayoutBlockSnapshot values")
            if not block.uid:
                raise ValueError("layout block uid must be non-empty")
            if block.uid in seen:
                raise ValueError(f"duplicate layout block uid: {block.uid!r}")
            seen.add(block.uid)

    @staticmethod
    def _block_by_uid(
        blocks: Iterable[LayoutBlockSnapshot],
        block_uid: str,
    ) -> LayoutBlockSnapshot:
        for block in blocks:
            if block.uid == block_uid:
                return block
        raise ValueError(f"layout block {block_uid!r} is not present in the layout snapshot")

    @staticmethod
    def _normalize_blocks(
        blocks: Iterable[LayoutBlockSnapshot],
    ) -> tuple[LayoutBlockSnapshot, ...]:
        normalized: list[LayoutBlockSnapshot] = []
        seen: set[str] = set()
        for order, block in enumerate(blocks):
            if not isinstance(block, LayoutBlockSnapshot):
                raise TypeError("layout snapshots may contain only LayoutBlockSnapshot values")
            if block.uid in seen:
                raise ValueError(f"duplicate layout block uid: {block.uid!r}")
            seen.add(block.uid)
            normalized.append(_copy_block(block, order=order))
        return tuple(normalized)

    @staticmethod
    def _unique_uids(uids: Iterable[str]) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for uid in uids:
            if uid not in seen:
                seen.add(uid)
                result.append(uid)
        return tuple(result)

    @staticmethod
    def _merge_bbox(requested: BBox, selected: Iterable[BBox]) -> BBox:
        boxes = tuple(selected)
        if not boxes:
            return _copy_bbox(requested)  # pragma: no cover - merge validates selection
        x1 = min(requested.x1, *(bbox.x1 for bbox in boxes))
        y1 = min(requested.y1, *(bbox.y1 for bbox in boxes))
        x2 = max(requested.x2, *(bbox.x2 for bbox in boxes))
        y2 = max(requested.y2, *(bbox.y2 for bbox in boxes))
        return BBox.from_xyxy(x1, y1, x2, y2)

    @staticmethod
    def snapshot_block_state(block: LayoutBlockSnapshot) -> dict[str, Any]:
        return {
            "uid": block.uid,
            "block_type": block.block_type.value,
            "bbox": list(block.bbox.to_xyxy()),
            "order": block.order,
            "source_label": block.source_label,
            "ocr_policy": block.ocr_policy.value,
        }


__all__ = [
    "LayoutEditCommand",
    "LayoutEditResult",
    "LayoutEditService",
    "OcrInvalidationDescriptor",
]
