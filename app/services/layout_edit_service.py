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
from app.models import BBox, Block, BlockSource, BlockType, LayoutEditEvent, OcrPolicy, Page
from app.models.block_state import mark_ocr_text_invalidated, paddle_binding_dict, set_paddle_binding
from app.models.ocr_observation import block_ocr_lines, clear_block_ocr_lines


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

    def persist_user_block_geometry(self, page: Page, block: Block) -> dict:
        if block.block_type not in STRUCTURAL_BINDING_BLOCK_TYPES:
            return {}
        block.source = BlockSource.USER_EDITED
        mark_ocr_text_invalidated(block, "block_geometry_changed")
        if not self._update_existing_manual_binding_bbox(block):
            clear_block_ocr_lines(block)
            binding = self.bind_manual_block_to_paddle(page, block)
        else:
            binding = paddle_binding_dict(block)
        if self.is_generated_inline_formula_block(block):
            self.mark_generated_inline_formula_handled(page, block)
        return binding

    def create_block(
        self,
        page: Page,
        bbox: BBox,
        block_type: BlockType,
        source_label: str,
    ) -> LayoutEditResult:
        new_block = Block(
            block_type=block_type,
            bbox=bbox,
            source=BlockSource.MANUAL_DRAW,
            source_label=source_label,
        )
        new_block.ocr_policy = default_ocr_policy_for_block(new_block)
        binding = self.bind_manual_block_to_paddle(page, new_block)
        page.blocks.append(new_block)
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

    def delete_block(self, page: Page, block: Block) -> LayoutEditResult:
        before = {"block": self.block_state(block)}
        self.mark_generated_inline_formula_handled(page, block, op="delete_inline_formula")
        page.blocks = [candidate for candidate in page.blocks if candidate is not block]
        self.record_edit(page, "delete_block", block, before=before, after={})
        return LayoutEditResult(op="delete_block", block=block, before=before, after={})

    def change_block_kind(
        self,
        page: Page,
        block: Block,
        *,
        block_type: BlockType,
        source_label: str,
    ) -> LayoutEditResult:
        before = {"block": self.block_state(block)}
        block.block_type = block_type
        block.source_label = source_label
        block.source = BlockSource.USER_EDITED
        block.ocr_policy = default_ocr_policy_for_block(block)
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

    def merge_blocks_into_bbox(
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
        primary.bbox = BBox.from_xyxy(x1, y1, x2, y2).clamp(page.width, page.height)
        primary.block_type = block_type
        primary.source_label = source_label
        clear_block_ocr_lines(primary)
        primary.source = BlockSource.USER_EDITED
        primary.ocr_policy = default_ocr_policy_for_block(primary)
        primary.note = "manual_draw_merge_requires_ocr_rerun"
        mark_ocr_text_invalidated(primary, "manual_draw_merge")
        binding = self.bind_manual_block_to_paddle(page, primary)
        for block in ordered[1:]:
            self.mark_generated_inline_formula_handled(page, block, op="merge_inline_formula")
        remove_ids = {id(block) for block in ordered[1:]}
        page.blocks = [block for block in page.blocks if id(block) not in remove_ids]
        for order, block in enumerate(page.blocks):
            block.order = order
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
            block.source_label = explicit_source_label
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
        block.source_label = source_label
        block.ocr_policy = OcrPolicy.PRESERVE_AS_FORMULA
        for line in block_ocr_lines(block):
            line.bbox = block.bbox
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
