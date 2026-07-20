"""Business entry point for mutating editable layout blocks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.core.block_attributes import normalize_source_label
from app.core.inline_formula_edit_state import inline_formula_origin_bbox, mark_inline_formula_origin_handled
from app.core.ocr_dispatch_policy import default_ocr_policy_for_block
from app.core.paddle_artifact_index import (
    BINDING_AMBIGUOUS,
    BINDING_EMPTY_REVIEW,
    PaddleArtifactIndex,
    apply_paddle_binding_to_block,
)
from app.models import BBox, Block, BlockOrigin, BlockType, LayoutEditEvent, OcrPolicy, Page
from app.models.layout_block_view import current_layout_snapshot, iter_page_layout_block_views
from app.models.block_state import mark_ocr_text_invalidated, paddle_binding_dict, set_paddle_binding
from app.models.layout_block_state import (
    mark_layout_block_manual_draw,
    mark_layout_block_user_edited,
    set_layout_block_note,
    set_layout_block_ocr_policy,
    set_layout_block_order,
    set_layout_block_source_label,
    set_layout_block_type,
)
from app.models.layout_snapshot import LayoutBlockSnapshot, LayoutSnapshot
from app.models.layout_snapshot_projection import (
    apply_layout_snapshot_block_to_projection,
    replace_page_layout_projection_from_snapshot,
)
from app.models.layout_snapshot_store import set_layout_snapshot_for_page
from app.models.ocr_observation import (
    block_ocr_line_observations_by_uid,
    clear_block_ocr_line_observations,
    discard_block_ocr_line_projection,
    set_ocr_line_bbox,
)


STRUCTURAL_BINDING_BLOCK_TYPES = {BlockType.EQUATION, BlockType.TABLE, BlockType.FIGURE}
_PRESERVE_EXPLICIT_SOURCE_LABELS = {
    "abstract",
    "chart",
    "display_formula",
    "doc_title",
    "figure",
    "figure_caption",
    "figure_title",
    "footer",
    "footnote",
    "formula",
    "formula_number",
    "header",
    "heading_1",
    "heading_2",
    "heading_3",
    "heading_4",
    "heading_5",
    "heading_6",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference_content",
    "table",
    "table_caption",
    "table_title",
    "text",
    "title",
}


@dataclass(frozen=True)
class LayoutEditCommand:
    op: str
    page: Page
    block_uid: str = ""
    block_uids: tuple[str, ...] = ()
    snapshot_blocks: tuple[LayoutBlockSnapshot, ...] = ()
    bbox: BBox | None = None
    block_type: BlockType | None = None
    source_label: str = ""
    before: dict | None = None

    @classmethod
    def create_block(
        cls,
        page: Page,
        bbox: BBox,
        block_type: BlockType,
        source_label: str,
    ) -> "LayoutEditCommand":
        return cls("create_block", page, bbox=bbox, block_type=block_type, source_label=source_label)

    @classmethod
    def delete_block(cls, page: Page, block_uid: str) -> "LayoutEditCommand":
        return cls("delete_block", page, block_uid=block_uid)

    @classmethod
    def change_kind(
        cls,
        page: Page,
        block_uid: str,
        *,
        block_type: BlockType,
        source_label: str,
    ) -> "LayoutEditCommand":
        return cls("change_kind", page, block_uid=block_uid, block_type=block_type, source_label=source_label)

    @classmethod
    def merge_blocks(
        cls,
        page: Page,
        block_uids: Iterable[str],
        bbox: BBox,
        *,
        block_type: BlockType,
        source_label: str,
    ) -> "LayoutEditCommand":
        return cls(
            "merge_blocks",
            page,
            block_uids=tuple(block_uids),
            bbox=bbox,
            block_type=block_type,
            source_label=source_label,
        )

    @classmethod
    def update_geometry(
        cls,
        page: Page,
        block_uid: str,
        *,
        bbox: BBox,
        before: dict | None = None,
    ) -> "LayoutEditCommand":
        return cls("resize_block", page, block_uid=block_uid, bbox=bbox, before=before)

    @classmethod
    def restore_blocks(
        cls,
        page: Page,
        blocks: Iterable[LayoutBlockSnapshot],
        *,
        before: dict | None = None,
    ) -> "LayoutEditCommand":
        return cls("restore_blocks", page, snapshot_blocks=tuple(blocks), before=before)


@dataclass(frozen=True)
class LayoutEditResult:
    op: str
    block: Block | None = None
    before: dict | None = None
    after: dict | None = None
    binding_status: str = ""

    @property
    def binding_empty_review(self) -> bool:
        return self.binding_status == BINDING_EMPTY_REVIEW

    @property
    def binding_ambiguous(self) -> bool:
        return self.binding_status == BINDING_AMBIGUOUS


class LayoutEditService:
    """Apply layout edit commands and record their audit events."""

    def apply(self, command: LayoutEditCommand) -> LayoutEditResult:
        if command.op == "create_block":
            return self._create_block(
                command.page,
                self._require_bbox(command),
                self._require_block_type(command),
                command.source_label,
            )
        if command.op == "delete_block":
            return self._delete_block(
                command.page,
                self._require_block_uid(command),
            )
        if command.op == "change_kind":
            return self._change_block_kind(
                command.page,
                self._require_block_uid(command),
                block_type=self._require_block_type(command),
                source_label=command.source_label,
            )
        if command.op == "merge_blocks":
            return self._merge_blocks_into_bbox(
                command.page,
                command.block_uids,
                self._require_bbox(command),
                block_type=self._require_block_type(command),
                source_label=command.source_label,
            )
        if command.op == "resize_block":
            return self._update_block_geometry(
                command.page,
                self._require_block_uid(command),
                bbox=self._require_bbox(command),
                before=command.before,
            )
        if command.op == "restore_blocks":
            return self._restore_blocks(command.page, command.snapshot_blocks, before=command.before)
        raise ValueError(f"Unsupported layout edit command: {command.op}")

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
    def _runtime_block_by_uid(page: Page, block_uid: str) -> Block:
        for view in iter_page_layout_block_views(page):
            if view.uid == block_uid and view.runtime_block is not None:
                return view.runtime_block
        raise ValueError(f"layout block {block_uid!r} is not present in the active runtime projection")

    @staticmethod
    def _runtime_block_map(page: Page) -> dict[str, Block]:
        return {
            view.uid: view.runtime_block
            for view in iter_page_layout_block_views(page)
            if view.runtime_block is not None
        }

    @staticmethod
    def block_state(block: Block) -> dict:
        return {
            "uid": block.uid,
            "block_type": getattr(block.block_type, "value", str(block.block_type)),
            "bbox": list(block.bbox.to_xyxy()),
            "order": block.order,
            "source_label": block.source_label,
            "ocr_policy": getattr(block.ocr_policy, "value", str(block.ocr_policy)),
        }

    def record_snapshot_edit(
        self,
        page: Page,
        op: str,
        block_uid: str,
        *,
        before: dict,
        after: dict,
    ) -> LayoutEditEvent:
        event = LayoutEditEvent(
            page_uid=page.uid,
            target_uid=block_uid,
            op=op,
            before=before,
            after=after,
            actor="user",
        )
        page.layout_edit_events.append(event)
        return event

    def _persist_user_block_geometry(self, page: Page, block: Block) -> dict:
        mark_layout_block_user_edited(block)
        mark_ocr_text_invalidated(block, "block_geometry_changed")
        clear_block_ocr_line_observations(block.uid)
        discard_block_ocr_line_projection(block)
        if block.block_type not in STRUCTURAL_BINDING_BLOCK_TYPES:
            return {}
        if not self._update_existing_manual_binding_bbox(block):
            binding = self.bind_manual_block_to_paddle(page, block)
        else:
            binding = paddle_binding_dict(block)
        if self.is_generated_inline_formula_block(block):
            self.mark_generated_inline_formula_handled(page, block)
        return binding

    def _update_block_geometry(
        self,
        page: Page,
        block_uid: str,
        *,
        bbox: BBox,
        before: dict | None,
    ) -> LayoutEditResult:
        snapshot = current_layout_snapshot(page)
        snapshot_index, snapshot_block = self._snapshot_block_for_uid(snapshot, block_uid)
        before = before or self.snapshot_block_state(snapshot_block)
        block = self._runtime_block_by_uid(page, block_uid)
        previous_ocr_policy = block.ocr_policy
        previous_ocr_policy = block.ocr_policy
        provisional = self._replace_snapshot_block(snapshot, snapshot_index, bbox=bbox)
        apply_layout_snapshot_block_to_projection(block, provisional)
        binding = self._persist_user_block_geometry(page, block)
        final_snapshot_block = self._replace_snapshot_block(
            snapshot,
            snapshot_index,
            bbox=block.bbox,
            origin=block.origin or provisional.origin,
            ocr_policy=block.ocr_policy,
            note=block.note,
        )
        after = self.snapshot_block_state(final_snapshot_block)
        event = self.record_snapshot_edit(
            page,
            "resize_block",
            block_uid,
            before=before,
            after=after,
        )
        next_snapshot = self._snapshot_with_replaced_block(
            snapshot,
            snapshot_index,
            final_snapshot_block,
            source_run_id=event.uid,
        )
        set_layout_snapshot_for_page(page, next_snapshot)
        apply_layout_snapshot_block_to_projection(block, final_snapshot_block)
        return LayoutEditResult(
            op="resize_block",
            block=block,
            before=before,
            after=after,
            binding_status=str(binding.get("status") or ""),
        )

    def _create_block(
        self,
        page: Page,
        bbox: BBox,
        block_type: BlockType,
        source_label: str,
    ) -> LayoutEditResult:
        new_block = Block(
            block_type=block_type,
            bbox=bbox,
            source_label=source_label,
        )
        mark_layout_block_manual_draw(new_block)
        set_layout_block_order(new_block, len(current_layout_snapshot(page).blocks))
        set_layout_block_ocr_policy(new_block, default_ocr_policy_for_block(new_block))
        binding = self.bind_manual_block_to_paddle(page, new_block)
        snapshot = current_layout_snapshot(page)
        new_snapshot_block = self._snapshot_block_from_edit_block(new_block, order=len(snapshot.blocks))
        after = {"block": self.snapshot_block_state(new_snapshot_block)}
        event = self.record_snapshot_edit(
            page,
            "create_block",
            new_block.uid,
            before={},
            after=after,
        )
        next_snapshot = self._snapshot_with_blocks(
            snapshot,
            (*snapshot.blocks, new_snapshot_block),
            source_run_id=event.uid,
        )
        set_layout_snapshot_for_page(page, next_snapshot)
        replace_page_layout_projection_from_snapshot(
            page,
            next_snapshot,
            candidate_blocks=[new_block],
        )
        return LayoutEditResult(
            op="create_block",
            block=new_block,
            before={},
            after=after,
            binding_status=str(binding.get("status") or ""),
        )

    def _delete_block(self, page: Page, block_uid: str) -> LayoutEditResult:
        snapshot = current_layout_snapshot(page)
        snapshot_index, snapshot_block = self._snapshot_block_for_uid(snapshot, block_uid)
        block = self._runtime_block_by_uid(page, block_uid)
        previous_ocr_policy = block.ocr_policy
        previous_ocr_policy = block.ocr_policy
        before = {"block": self.snapshot_block_state(snapshot_block)}
        clear_block_ocr_line_observations(block_uid)
        discard_block_ocr_line_projection(block)
        self.mark_generated_inline_formula_handled(page, block, op="delete_inline_formula")
        event = self.record_snapshot_edit(
            page,
            "delete_block",
            block_uid,
            before=before,
            after={},
        )
        next_snapshot = self._snapshot_with_blocks(
            snapshot,
            tuple(
                candidate
                for index, candidate in enumerate(snapshot.blocks)
                if index != snapshot_index
            ),
            source_run_id=event.uid,
        )
        set_layout_snapshot_for_page(page, next_snapshot)
        replace_page_layout_projection_from_snapshot(page, next_snapshot)
        return LayoutEditResult(op="delete_block", block=block, before=before, after={})

    def _restore_blocks(
        self,
        page: Page,
        blocks: Iterable[LayoutBlockSnapshot],
        *,
        before: dict | None,
    ) -> LayoutEditResult:
        restore_blocks = tuple(blocks)
        snapshot = current_layout_snapshot(page)
        before = before or {"blocks": [self.snapshot_block_state(block) for block in snapshot.blocks]}
        next_snapshot_blocks = tuple(
            self._snapshot_block_with_order(block, order)
            for order, block in enumerate(restore_blocks)
        )
        after = {"blocks": [self.snapshot_block_state(block) for block in next_snapshot_blocks]}
        event = self.record_snapshot_edit(
            page,
            "restore_blocks",
            "",
            before=before,
            after=after,
        )
        next_snapshot = self._snapshot_with_blocks(
            snapshot,
            next_snapshot_blocks,
            source_run_id=event.uid,
        )
        set_layout_snapshot_for_page(page, next_snapshot)
        replace_page_layout_projection_from_snapshot(page, next_snapshot)
        return LayoutEditResult(op="restore_blocks", before=before, after=after)

    def _change_block_kind(
        self,
        page: Page,
        block_uid: str,
        *,
        block_type: BlockType,
        source_label: str,
    ) -> LayoutEditResult:
        snapshot = current_layout_snapshot(page)
        snapshot_index, snapshot_block = self._snapshot_block_for_uid(snapshot, block_uid)
        block = self._runtime_block_by_uid(page, block_uid)
        previous_ocr_policy = block.ocr_policy
        before = {"block": self.snapshot_block_state(snapshot_block)}
        provisional = self._replace_snapshot_block(
            snapshot,
            snapshot_index,
            block_type=block_type,
            source_label=source_label,
        )
        apply_layout_snapshot_block_to_projection(block, provisional)
        set_layout_block_type(block, block_type)
        set_layout_block_source_label(block, source_label)
        mark_layout_block_user_edited(block)
        mark_ocr_text_invalidated(block, "block_kind_changed")
        next_ocr_policy = default_ocr_policy_for_block(block)
        set_layout_block_ocr_policy(block, next_ocr_policy)
        if next_ocr_policy != previous_ocr_policy:
            clear_block_ocr_line_observations(block.uid)
            discard_block_ocr_line_projection(block)
        binding = self.bind_manual_block_to_paddle(page, block)
        final_snapshot_block = self._replace_snapshot_block(
            snapshot,
            snapshot_index,
            block_type=block.block_type,
            source_label=block.source_label,
            origin=block.origin or provisional.origin,
            ocr_policy=block.ocr_policy,
            note=block.note,
        )
        after = {"block": self.snapshot_block_state(final_snapshot_block)}
        event = self.record_snapshot_edit(
            page,
            "change_kind",
            block_uid,
            before=before,
            after=after,
        )
        next_snapshot = self._snapshot_with_replaced_block(
            snapshot,
            snapshot_index,
            final_snapshot_block,
            source_run_id=event.uid,
        )
        set_layout_snapshot_for_page(page, next_snapshot)
        apply_layout_snapshot_block_to_projection(block, final_snapshot_block)
        return LayoutEditResult(
            op="change_kind",
            block=block,
            before=before,
            after=after,
            binding_status=str(binding.get("status") or ""),
        )

    @staticmethod
    def snapshot_block_state(block: LayoutBlockSnapshot) -> dict:
        return {
            "uid": block.uid,
            "block_type": getattr(block.block_type, "value", str(block.block_type)),
            "bbox": list(block.bbox.to_xyxy()),
            "order": block.order,
            "source_label": block.source_label,
            "ocr_policy": getattr(block.ocr_policy, "value", str(block.ocr_policy)),
        }

    @staticmethod
    def _snapshot_block_for_uid(
        snapshot: LayoutSnapshot,
        block_uid: str,
    ) -> tuple[int, LayoutBlockSnapshot]:
        for index, snapshot_block in enumerate(snapshot.blocks):
            if snapshot_block.uid == block_uid:
                return index, snapshot_block
        raise ValueError(f"layout block {block_uid!r} is not present in the active layout snapshot")

    @staticmethod
    def _replace_snapshot_block(
        snapshot: LayoutSnapshot,
        index: int,
        *,
        block_type: BlockType | None = None,
        bbox: BBox | None = None,
        order: int | None = None,
        source_label: str | None = None,
        origin: BlockOrigin | None = None,
        ocr_policy: OcrPolicy | None = None,
        note: str | None = None,
    ) -> LayoutBlockSnapshot:
        current = snapshot.blocks[index]
        return LayoutBlockSnapshot(
            block_type=block_type if block_type is not None else current.block_type,
            bbox=bbox if bbox is not None else current.bbox,
            order=order if order is not None else current.order,
            source_label=source_label if source_label is not None else current.source_label,
            origin=origin if origin is not None else current.origin,
            ocr_policy=ocr_policy if ocr_policy is not None else current.ocr_policy,
            note=note if note is not None else current.note,
            uid=current.uid,
        )

    @staticmethod
    def _snapshot_with_replaced_block(
        snapshot: LayoutSnapshot,
        index: int,
        block: LayoutBlockSnapshot,
        *,
        source_run_id: str,
    ) -> LayoutSnapshot:
        blocks = list(snapshot.blocks)
        blocks[index] = block
        return LayoutEditService._snapshot_with_blocks(
            snapshot,
            tuple(blocks),
            source_run_id=source_run_id,
        )

    @staticmethod
    def _snapshot_with_blocks(
        snapshot: LayoutSnapshot,
        blocks: tuple[LayoutBlockSnapshot, ...],
        *,
        source_run_id: str,
    ) -> LayoutSnapshot:
        return LayoutSnapshot(
            page_uid=snapshot.page_uid,
            artifact_uid=snapshot.artifact_uid,
            source_engine="layout_edit",
            source_run_id=source_run_id,
            blocks=blocks,
        )

    @staticmethod
    def _snapshot_block_from_edit_block(
        block: Block,
        *,
        order: int,
    ) -> LayoutBlockSnapshot:
        return LayoutBlockSnapshot(
            block_type=block.block_type,
            bbox=block.bbox,
            order=order,
            source_label=block.source_label,
            origin=block.origin or BlockOrigin(
                created_by=getattr(block.source, "value", str(block.source)),
                source_label=block.source_label,
                original_bbox=block.bbox,
                original_kind=block.block_type,
            ),
            ocr_policy=block.ocr_policy,
            note=block.note,
            uid=block.uid,
        )

    @staticmethod
    def _snapshot_block_with_order(
        block: LayoutBlockSnapshot,
        order: int,
    ) -> LayoutBlockSnapshot:
        return LayoutBlockSnapshot(
            block_type=block.block_type,
            bbox=block.bbox,
            order=order,
            source_label=block.source_label,
            origin=block.origin,
            ocr_policy=block.ocr_policy,
            note=block.note,
            uid=block.uid,
        )

    def _merge_blocks_into_bbox(
        self,
        page: Page,
        block_uids: Iterable[str],
        bbox: BBox,
        *,
        block_type: BlockType,
        source_label: str,
    ) -> LayoutEditResult:
        snapshot = current_layout_snapshot(page)
        runtime_by_uid = self._runtime_block_map(page)
        snapshot_by_uid = {
            snapshot_block.uid: (index, snapshot_block)
            for index, snapshot_block in enumerate(snapshot.blocks)
        }
        ordered_uids = tuple(uid for uid in block_uids if uid)
        for block_uid in ordered_uids:
            if block_uid not in snapshot_by_uid:
                raise ValueError(f"layout block {block_uid!r} is not present in the active layout snapshot")
            if block_uid not in runtime_by_uid:
                raise ValueError(f"layout block {block_uid!r} is not present in the active runtime projection")
        ordered = sorted(
            (
                (snapshot_by_uid[block_uid][0], snapshot_by_uid[block_uid][1], runtime_by_uid[block_uid])
                for block_uid in ordered_uids
            ),
            key=lambda item: (item[1].order, item[1].bbox.y, item[1].bbox.x),
        )
        if not ordered:
            raise ValueError("merge_blocks_into_bbox requires at least one block")
        before = {"blocks": [self.snapshot_block_state(snapshot_block) for _index, snapshot_block, _block in ordered]}
        primary_index, primary_snapshot, primary = ordered[0]
        x1 = min([bbox.x1, *(snapshot_block.bbox.x1 for _index, snapshot_block, _block in ordered)])
        y1 = min([bbox.y1, *(snapshot_block.bbox.y1 for _index, snapshot_block, _block in ordered)])
        x2 = max([bbox.x2, *(snapshot_block.bbox.x2 for _index, snapshot_block, _block in ordered)])
        y2 = max([bbox.y2, *(snapshot_block.bbox.y2 for _index, snapshot_block, _block in ordered)])
        merged_bbox = BBox.from_xyxy(x1, y1, x2, y2).clamp(page.width, page.height)
        provisional = self._replace_snapshot_block(
            snapshot,
            primary_index,
            block_type=block_type,
            bbox=merged_bbox,
            source_label=source_label,
        )
        apply_layout_snapshot_block_to_projection(primary, provisional)
        clear_block_ocr_line_observations(primary.uid)
        discard_block_ocr_line_projection(primary)
        mark_layout_block_user_edited(primary)
        set_layout_block_ocr_policy(primary, default_ocr_policy_for_block(primary))
        set_layout_block_note(primary, "manual_draw_merge_requires_ocr_rerun")
        mark_ocr_text_invalidated(primary, "manual_draw_merge")
        binding = self.bind_manual_block_to_paddle(page, primary)
        for _index, _snapshot_block, block in ordered[1:]:
            self.mark_generated_inline_formula_handled(page, block, op="merge_inline_formula")
        final_primary_snapshot = self._replace_snapshot_block(
            snapshot,
            primary_index,
            block_type=primary.block_type,
            bbox=primary.bbox,
            source_label=primary.source_label,
            origin=primary.origin or primary_snapshot.origin,
            ocr_policy=primary.ocr_policy,
            note=primary.note,
        )
        remove_uids = {snapshot_block.uid for _index, snapshot_block, _block in ordered[1:]}
        next_snapshot_blocks: list[LayoutBlockSnapshot] = []
        for snapshot_block in snapshot.blocks:
            if snapshot_block.uid in remove_uids:
                continue
            if snapshot_block.uid == final_primary_snapshot.uid:
                next_snapshot_blocks.append(final_primary_snapshot)
            else:
                next_snapshot_blocks.append(snapshot_block)
        next_snapshot_blocks = [
            self._snapshot_block_with_order(snapshot_block, order)
            for order, snapshot_block in enumerate(next_snapshot_blocks)
        ]
        after_block = next(
            snapshot_block
            for snapshot_block in next_snapshot_blocks
            if snapshot_block.uid == final_primary_snapshot.uid
        )
        after = {"block": self.snapshot_block_state(after_block)}
        event = self.record_snapshot_edit(
            page,
            "merge_blocks",
            primary.uid,
            before=before,
            after=after,
        )
        next_snapshot = self._snapshot_with_blocks(
            snapshot,
            tuple(next_snapshot_blocks),
            source_run_id=event.uid,
        )
        set_layout_snapshot_for_page(page, next_snapshot)
        replace_page_layout_projection_from_snapshot(
            page,
            next_snapshot,
            candidate_blocks=[primary],
        )
        return LayoutEditResult(
            op="merge_blocks",
            block=primary,
            before=before,
            after=after,
            binding_status=str(binding.get("status") or ""),
        )

    def bind_manual_block_to_paddle(self, page: Page, block: Block) -> dict:
        if block.block_type not in STRUCTURAL_BINDING_BLOCK_TYPES:
            return {}
        explicit_source_label = block.source_label
        binding = PaddleArtifactIndex.from_page(page).bind_manual_bbox(block.bbox, block.block_type)
        apply_paddle_binding_to_block(block, binding)
        self._preserve_inline_formula_origin_binding(block)
        if normalize_source_label(explicit_source_label) in _PRESERVE_EXPLICIT_SOURCE_LABELS:
            set_layout_block_source_label(block, explicit_source_label)
        return paddle_binding_dict(block)

    @staticmethod
    def is_generated_inline_formula_block(block: Block) -> bool:
        return (
            normalize_source_label(block.source_label) == "inline_formula"
            and inline_formula_origin_bbox(block) is not None
        )

    @staticmethod
    def mark_generated_inline_formula_handled(
        page: Page,
        block: Block,
        *,
        op: str = "claim_inline_formula",
    ) -> None:
        mark_inline_formula_origin_handled(page, block, op=op)

    @staticmethod
    def _update_existing_manual_binding_bbox(block: Block) -> bool:
        binding = paddle_binding_dict(block)
        if not binding:
            return False
        status = str(binding.get("status") or "")
        if status in {BINDING_EMPTY_REVIEW, BINDING_AMBIGUOUS}:
            return False
        try:
            if int(binding.get("parent_index", -1)) < 0:
                return False
        except (TypeError, ValueError):
            return False

        manual_bbox = list(block.bbox.to_xyxy())
        next_binding = dict(binding)
        next_binding["manual_bbox"] = manual_bbox
        origin_bbox = inline_formula_origin_bbox(block)
        if not next_binding.get("candidate_bbox") and origin_bbox is not None:
            next_binding["candidate_bbox"] = [int(value) for value in origin_bbox]
        source_label = str(next_binding.get("source_label") or block.source_label or block.block_type.value)
        set_layout_block_source_label(block, source_label)
        set_layout_block_ocr_policy(block, OcrPolicy.PRESERVE_AS_FORMULA)
        for line in block_ocr_line_observations_by_uid(block.uid):
            set_ocr_line_bbox(line, block.bbox)
        set_paddle_binding(block, next_binding)
        return True

    @staticmethod
    def _preserve_inline_formula_origin_binding(block: Block) -> None:
        if normalize_source_label(block.source_label) != "inline_formula":
            return
        origin_bbox = inline_formula_origin_bbox(block)
        if origin_bbox is None:
            return
        binding = paddle_binding_dict(block)
        if not binding or binding.get("candidate_bbox"):
            return
        next_binding = dict(binding)
        next_binding["candidate_bbox"] = [int(value) for value in origin_bbox]
        set_paddle_binding(block, next_binding)
