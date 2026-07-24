from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.models import (
    BBox,
    BlockOrigin,
    BlockSource,
    BlockType,
    LayoutBlockSnapshot,
    LayoutSnapshot,
    OcrPolicy,
)
from app.models.project_session import RevisionConflictError
from app.services.layout_edit_service import (
    LayoutEditCommand,
    LayoutEditService,
)


def _block(
    uid: str,
    block_type: BlockType,
    bbox: BBox,
    order: int,
    source_label: str,
    *,
    ocr_policy: OcrPolicy | None = None,
    authorship: BlockSource = BlockSource.AUTO_LAYOUT,
) -> LayoutBlockSnapshot:
    return LayoutBlockSnapshot(
        block_type=block_type,
        bbox=bbox,
        order=order,
        source_label=source_label,
        origin=BlockOrigin(
            source_engine="fixture",
            vendor_label=source_label,
            original_bbox=bbox,
            original_kind=block_type,
        ),
        ocr_policy=ocr_policy or OcrPolicy.TEXT_OCR,
        authorship=authorship,
        uid=uid,
    )


def _snapshot() -> LayoutSnapshot:
    return LayoutSnapshot(
        page_uid="page-uid",
        revision=7,
        artifact_uid="artifact-uid",
        source_engine="fixture",
        source_run_id="fixture-run",
        blocks=(
            _block(
                "block-primary",
                BlockType.TEXT,
                BBox.from_xyxy(10, 10, 40, 30),
                0,
                "text",
            ),
            _block(
                "block-secondary",
                BlockType.TEXT,
                BBox.from_xyxy(50, 12, 80, 32),
                1,
                "text",
            ),
            _block(
                "block-tail",
                BlockType.TABLE,
                BBox.from_xyxy(5, 50, 35, 75),
                2,
                "table",
                ocr_policy=OcrPolicy.PRESERVE_AS_TABLE,
            ),
        ),
    )


def test_create_returns_snapshot_with_new_uid_and_normalized_order():
    current = _snapshot()
    command = LayoutEditCommand.create_block(
        current.page_uid,
        current.revision,
        BBox.from_xyxy(90, 10, 120, 30),
        BlockType.EQUATION,
        "formula",
    )

    result = LayoutEditService().apply(current, command)

    assert result.op == "create_block"
    assert result.snapshot.revision == current.revision + 1
    assert command.new_block_uid.startswith("block_")
    assert result.snapshot.blocks[-1].uid == command.new_block_uid
    assert result.snapshot.blocks[-1].authorship is BlockSource.MANUAL_DRAW
    assert result.snapshot.blocks[-1].ocr_policy is OcrPolicy.PRESERVE_AS_FORMULA
    assert [block.order for block in result.snapshot.blocks] == [0, 1, 2, 3]
    assert result.affected_block_uids == (command.new_block_uid,)
    assert result.ocr_invalidation.block_uids == (command.new_block_uid,)
    assert result.ocr_invalidation.reason == "layout_block_created"
    assert not hasattr(result, "block")


def test_delete_normalizes_remaining_order_and_audits_removed_block():
    current = _snapshot()

    result = LayoutEditService().apply(
        current,
        LayoutEditCommand.delete_block(
            current.page_uid,
            current.revision,
            "block-secondary",
        ),
    )

    assert [block.uid for block in result.snapshot.blocks] == [
        "block-primary",
        "block-tail",
    ]
    assert [block.order for block in result.snapshot.blocks] == [0, 1]
    assert result.before["block"]["uid"] == "block-secondary"
    assert result.after == {}
    assert result.affected_block_uids == ("block-secondary",)
    assert result.ocr_invalidation.reason == "layout_block_deleted"


def test_change_kind_and_resize_preserve_uid_and_return_new_values():
    current = _snapshot()
    service = LayoutEditService()

    changed = service.apply(
        current,
        LayoutEditCommand.change_kind(
            current.page_uid,
            current.revision,
            "block-primary",
            block_type=BlockType.TITLE,
            source_label="heading_1",
        ),
    )
    resized = service.apply(
        changed.snapshot,
        LayoutEditCommand.resize_block(
            current.page_uid,
            changed.snapshot.revision,
            "block-primary",
            bbox=BBox.from_xyxy(12, 14, 44, 36),
        ),
    )

    changed_block = changed.snapshot.blocks[0]
    resized_block = resized.snapshot.blocks[0]
    assert changed_block.uid == "block-primary"
    assert changed_block.block_type is BlockType.TITLE
    assert changed_block.ocr_policy is OcrPolicy.TEXT_OCR
    assert changed_block.authorship is BlockSource.USER_EDITED
    assert resized_block.uid == changed_block.uid
    assert resized_block.bbox == BBox.from_xyxy(12, 14, 44, 36)
    assert resized.before["block"]["bbox"] == [10, 10, 40, 30]
    assert resized.after["block"]["bbox"] == [12, 14, 44, 36]
    assert resized.ocr_invalidation.reason == "layout_block_resized"


def test_change_kinds_updates_selected_blocks_in_one_revision() -> None:
    current = _snapshot()

    result = LayoutEditService().apply(
        current,
        LayoutEditCommand.change_kinds(
            current.page_uid,
            current.revision,
            ("block-primary", "block-secondary"),
            block_type=BlockType.TITLE,
            source_label="paragraph_title",
        ),
    )

    changed = result.snapshot.blocks[:2]
    assert result.snapshot.revision == current.revision + 1
    assert result.affected_block_uids == ("block-primary", "block-secondary")
    assert all(block.block_type is BlockType.TITLE for block in changed)
    assert all(block.source_label == "paragraph_title" for block in changed)
    assert all(block.authorship is BlockSource.USER_EDITED for block in changed)
    assert result.snapshot.blocks[2] == current.blocks[2]
    assert result.ocr_invalidation.reason == "layout_block_kinds_changed"


def test_merge_preserves_explicit_primary_uid_removes_secondaries_and_normalizes():
    current = _snapshot()
    command = LayoutEditCommand.merge_blocks(
        current.page_uid,
        current.revision,
        ("block-secondary", "block-primary"),
        BBox.from_xyxy(8, 8, 84, 36),
        block_type=BlockType.TEXT,
        source_label="text",
        primary_block_uid="block-primary",
    )

    result = LayoutEditService().apply(current, command)

    assert [block.uid for block in result.snapshot.blocks] == [
        "block-primary",
        "block-tail",
    ]
    assert result.snapshot.blocks[0].uid == "block-primary"
    assert result.snapshot.blocks[0].bbox == BBox.from_xyxy(8, 8, 84, 36)
    assert [block.order for block in result.snapshot.blocks] == [0, 1]
    assert [item["uid"] for item in result.before["blocks"]] == [
        "block-primary",
        "block-secondary",
    ]
    assert result.affected_block_uids == ("block-primary", "block-secondary")
    assert result.ocr_invalidation.reason == "layout_blocks_merged"


def test_restore_uses_snapshot_values_and_reports_replaced_uids():
    current = _snapshot()
    restored = (
        _block(
            "block-restored",
            BlockType.FIGURE,
            BBox.from_xyxy(1, 2, 30, 40),
            0,
            "figure",
            ocr_policy=OcrPolicy.SKIP,
        ),
        _block(
            "block-primary",
            BlockType.TABLE,
            BBox.from_xyxy(40, 2, 70, 40),
            1,
            "table",
            ocr_policy=OcrPolicy.PRESERVE_AS_TABLE,
            authorship=BlockSource.USER_EDITED,
        ),
    )

    result = LayoutEditService().apply(
        current,
        LayoutEditCommand.restore_blocks(
            current.page_uid,
            current.revision,
            restored,
        ),
    )

    assert [block.uid for block in result.snapshot.blocks] == [
        "block-restored",
        "block-primary",
    ]
    assert [block.order for block in result.snapshot.blocks] == [0, 1]
    assert result.snapshot.blocks[1].authorship is BlockSource.USER_EDITED
    assert result.affected_block_uids == (
        "block-primary",
        "block-secondary",
        "block-tail",
        "block-restored",
    )
    assert result.ocr_invalidation.reason == "layout_snapshot_restored"


def test_stale_revision_is_rejected_before_editing():
    current = _snapshot()
    command = LayoutEditCommand.delete_block(
        current.page_uid,
        current.revision - 1,
        "block-primary",
    )

    with pytest.raises(RevisionConflictError, match="revision mismatch"):
        LayoutEditService().apply(current, command)

    assert current.revision == 7
    assert [block.uid for block in current.blocks] == [
        "block-primary",
        "block-secondary",
        "block-tail",
    ]


def test_apply_does_not_mutate_or_share_mutable_values_with_prior_snapshot():
    current = _snapshot()
    original_blocks = current.blocks
    result = LayoutEditService().apply(
        current,
        LayoutEditCommand.resize_block(
            current.page_uid,
            current.revision,
            "block-primary",
            bbox=BBox.from_xyxy(11, 11, 41, 31),
        ),
    )

    assert current.blocks is original_blocks
    assert current.blocks[0].bbox == BBox.from_xyxy(10, 10, 40, 30)
    assert result.snapshot.blocks[0] is not current.blocks[0]
    assert result.snapshot.blocks[1] is not current.blocks[1]
    assert result.snapshot.blocks[1].bbox is not current.blocks[1].bbox
    with pytest.raises(FrozenInstanceError):
        result.snapshot.blocks[1].bbox.x = 999  # type: ignore[misc]
    assert current.blocks[1].bbox.x == 50
    with pytest.raises(FrozenInstanceError):
        result.snapshot.blocks = ()  # type: ignore[misc]


def test_command_is_snapshot_addressed_and_has_no_legacy_page_or_block_field():
    command = LayoutEditCommand.delete_block("page-uid", 7, "block-primary")

    assert command.page_uid == "page-uid"
    assert command.expected_revision == 7
    assert not hasattr(command, "page")
    assert not hasattr(command, "block")
