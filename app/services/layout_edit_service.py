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
from app.models import BBox, Block, BlockType, LayoutEditEvent, OcrPolicy, Page
from app.models.block_state import mark_ocr_text_invalidated, paddle_binding_dict, set_paddle_binding
from app.models.layout_block_state import (
    mark_layout_block_manual_draw,
    mark_layout_block_user_edited,
    set_layout_block_bbox,
    set_layout_block_note,
    set_layout_block_ocr_policy,
    set_layout_block_order,
    set_layout_block_source_label,
    set_layout_block_type,
)
from app.models.layout_projection import (
    append_page_layout_block,
    page_layout_blocks,
    replace_page_layout_blocks,
)
from app.models.ocr_observation import block_ocr_lines, clear_block_ocr_lines, set_ocr_line_bbox


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
    block: Block | None = None
    blocks: tuple[Block, ...] = ()
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
    def delete_block(cls, page: Page, block: Block) -> "LayoutEditCommand":
        return cls("delete_block", page, block=block)

    @classmethod
    def change_kind(
        cls,
        page: Page,
        block: Block,
        *,
        block_type: BlockType,
        source_label: str,
    ) -> "LayoutEditCommand":
        return cls("change_kind", page, block=block, block_type=block_type, source_label=source_label)

    @classmethod
    def merge_blocks(
        cls,
        page: Page,
        blocks: Iterable[Block],
        bbox: BBox,
        *,
        block_type: BlockType,
        source_label: str,
    ) -> "LayoutEditCommand":
        return cls(
            "merge_blocks",
            page,
            blocks=tuple(blocks),
            bbox=bbox,
            block_type=block_type,
            source_label=source_label,
        )

    @classmethod
    def update_geometry(cls, page: Page, block: Block, *, before: dict | None = None) -> "LayoutEditCommand":
        return cls("resize_block", page, block=block, before=before)

    @classmethod
    def restore_blocks(
        cls,
        page: Page,
        blocks: Iterable[Block],
        *,
        before: dict | None = None,
    ) -> "LayoutEditCommand":
        return cls("restore_blocks", page, blocks=tuple(blocks), before=before)


@dataclass(frozen=True)
class LayoutEditResult:
    op: str
    block: Block | None = None
    before: dict | None = None
    after: dict | None = None
    binding_status: str = ""
    binding_text: str = ""

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
            return self._delete_block(command.page, self._require_block(command))
        if command.op == "change_kind":
            return self._change_block_kind(
                command.page,
                self._require_block(command),
                block_type=self._require_block_type(command),
                source_label=command.source_label,
            )
        if command.op == "merge_blocks":
            return self._merge_blocks_into_bbox(
                command.page,
                command.blocks,
                self._require_bbox(command),
                block_type=self._require_block_type(command),
                source_label=command.source_label,
            )
        if command.op == "resize_block":
            return self._update_block_geometry(
                command.page,
                self._require_block(command),
                before=command.before,
            )
        if command.op == "restore_blocks":
            return self._restore_blocks(command.page, command.blocks, before=command.before)
        raise ValueError(f"Unsupported layout edit command: {command.op}")

    @staticmethod
    def _require_block(command: LayoutEditCommand) -> Block:
        if command.block is None:
            raise ValueError(f"{command.op} requires a block")
        return command.block

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
    def block_state(block: Block) -> dict:
        return {
            "uid": block.uid,
            "block_type": getattr(block.block_type, "value", str(block.block_type)),
            "bbox": list(block.bbox.to_xyxy()),
            "order": block.order,
            "source_label": block.source_label,
            "ocr_policy": getattr(block.ocr_policy, "value", str(block.ocr_policy)),
        }

    def record_edit(
        self,
        page: Page,
        op: str,
        block: Block | None,
        *,
        before: dict,
        after: dict,
    ) -> None:
        page.layout_edit_events.append(
            LayoutEditEvent(
                page_uid=page.uid,
                target_uid=block.uid if block is not None else "",
                op=op,
                before=before,
                after=after,
                actor="user",
            )
        )

    def _persist_user_block_geometry(self, page: Page, block: Block) -> dict:
        if block.block_type not in STRUCTURAL_BINDING_BLOCK_TYPES:
            return {}
        mark_layout_block_user_edited(block)
        mark_ocr_text_invalidated(block, "block_geometry_changed")
        if not self._update_existing_manual_binding_bbox(block):
            clear_block_ocr_lines(block)
            binding = self.bind_manual_block_to_paddle(page, block)
        else:
            binding = paddle_binding_dict(block)
        if self.is_generated_inline_formula_block(block):
            self.mark_generated_inline_formula_handled(page, block)
        return binding

    def _update_block_geometry(
        self,
        page: Page,
        block: Block,
        *,
        before: dict | None,
    ) -> LayoutEditResult:
        before = before or self.block_state(block)
        binding = self._persist_user_block_geometry(page, block)
        after = self.block_state(block)
        self.record_edit(page, "resize_block", block, before=before, after=after)
        return LayoutEditResult(
            op="resize_block",
            block=block,
            before=before,
            after=after,
            binding_status=str(binding.get("status") or ""),
            binding_text=str(binding.get("text") or ""),
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
        set_layout_block_ocr_policy(new_block, default_ocr_policy_for_block(new_block))
        binding = self.bind_manual_block_to_paddle(page, new_block)
        append_page_layout_block(page, new_block)
        after = {"block": self.block_state(new_block)}
        self.record_edit(page, "create_block", new_block, before={}, after=after)
        return LayoutEditResult(
            op="create_block",
            block=new_block,
            before={},
            after=after,
            binding_status=str(binding.get("status") or ""),
            binding_text=str(binding.get("text") or ""),
        )

    def _delete_block(self, page: Page, block: Block) -> LayoutEditResult:
        before = {"block": self.block_state(block)}
        self.mark_generated_inline_formula_handled(page, block, op="delete_inline_formula")
        replace_page_layout_blocks(
            page,
            [candidate for candidate in page_layout_blocks(page) if candidate is not block],
        )
        self.record_edit(page, "delete_block", block, before=before, after={})
        return LayoutEditResult(op="delete_block", block=block, before=before, after={})

    def _restore_blocks(
        self,
        page: Page,
        blocks: Iterable[Block],
        *,
        before: dict | None,
    ) -> LayoutEditResult:
        next_blocks = list(blocks)
        before = before or {"blocks": [self.block_state(block) for block in page_layout_blocks(page)]}
        replace_page_layout_blocks(page, next_blocks)
        for order, block in enumerate(page_layout_blocks(page)):
            set_layout_block_order(block, order)
        after = {"blocks": [self.block_state(block) for block in page_layout_blocks(page)]}
        self.record_edit(page, "restore_blocks", None, before=before, after=after)
        return LayoutEditResult(op="restore_blocks", before=before, after=after)

    def _change_block_kind(
        self,
        page: Page,
        block: Block,
        *,
        block_type: BlockType,
        source_label: str,
    ) -> LayoutEditResult:
        before = {"block": self.block_state(block)}
        set_layout_block_type(block, block_type)
        set_layout_block_source_label(block, source_label)
        mark_layout_block_user_edited(block)
        set_layout_block_ocr_policy(block, default_ocr_policy_for_block(block))
        binding = self.bind_manual_block_to_paddle(page, block)
        after = {"block": self.block_state(block)}
        self.record_edit(page, "change_kind", block, before=before, after=after)
        return LayoutEditResult(
            op="change_kind",
            block=block,
            before=before,
            after=after,
            binding_status=str(binding.get("status") or ""),
            binding_text=str(binding.get("text") or ""),
        )

    def _merge_blocks_into_bbox(
        self,
        page: Page,
        blocks: Iterable[Block],
        bbox: BBox,
        *,
        block_type: BlockType,
        source_label: str,
    ) -> LayoutEditResult:
        ordered = sorted(blocks, key=lambda block: (block.order, block.bbox.y, block.bbox.x))
        if not ordered:
            raise ValueError("merge_blocks_into_bbox requires at least one block")
        before = {"blocks": [self.block_state(block) for block in ordered]}
        primary = ordered[0]
        x1 = min([bbox.x1, *(block.bbox.x1 for block in ordered)])
        y1 = min([bbox.y1, *(block.bbox.y1 for block in ordered)])
        x2 = max([bbox.x2, *(block.bbox.x2 for block in ordered)])
        y2 = max([bbox.y2, *(block.bbox.y2 for block in ordered)])
        set_layout_block_bbox(primary, BBox.from_xyxy(x1, y1, x2, y2).clamp(page.width, page.height))
        set_layout_block_type(primary, block_type)
        set_layout_block_source_label(primary, source_label)
        clear_block_ocr_lines(primary)
        mark_layout_block_user_edited(primary)
        set_layout_block_ocr_policy(primary, default_ocr_policy_for_block(primary))
        set_layout_block_note(primary, "manual_draw_merge_requires_ocr_rerun")
        mark_ocr_text_invalidated(primary, "manual_draw_merge")
        binding = self.bind_manual_block_to_paddle(page, primary)
        for block in ordered[1:]:
            self.mark_generated_inline_formula_handled(page, block, op="merge_inline_formula")
        remove_ids = {id(block) for block in ordered[1:]}
        replace_page_layout_blocks(
            page,
            [block for block in page_layout_blocks(page) if id(block) not in remove_ids],
        )
        for order, block in enumerate(page_layout_blocks(page)):
            set_layout_block_order(block, order)
        after = {"block": self.block_state(primary)}
        self.record_edit(page, "merge_blocks", primary, before=before, after=after)
        return LayoutEditResult(
            op="merge_blocks",
            block=primary,
            before=before,
            after=after,
            binding_status=str(binding.get("status") or ""),
            binding_text=str(binding.get("text") or ""),
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
        for line in block_ocr_lines(block):
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
