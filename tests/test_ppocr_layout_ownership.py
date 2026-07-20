from __future__ import annotations

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint
from app.core.ppocr_layout_ownership import build_layout_ownership
from app.models import BBox, BlockOrigin, BlockType, LayoutBlockSnapshot, LayoutSnapshot, OcrPolicy


def _block(
    uid: str,
    block_type: BlockType,
    bbox: tuple[int, int, int, int],
    *,
    order: int,
    policy: OcrPolicy,
    label: str,
) -> LayoutBlockSnapshot:
    box = BBox.from_xyxy(*bbox)
    return LayoutBlockSnapshot(
        uid=uid,
        block_type=block_type,
        bbox=box,
        order=order,
        source_label=label,
        origin=BlockOrigin(
            vendor_label=label,
            original_bbox=box,
            original_kind=block_type,
        ),
        ocr_policy=policy,
    )


def test_layout_ownership_uses_snapshot_blocks_and_structural_exclusion():
    snapshot = LayoutSnapshot(
        page_uid="page-1",
        revision=1,
        artifact_uid="layout-1",
        source_engine="test",
        source_run_id="layout-run-1",
        blocks=(
            _block("text", BlockType.TEXT, (0, 0, 200, 60), order=0, policy=OcrPolicy.TEXT_OCR, label="text"),
            _block("table", BlockType.TABLE, (80, 0, 140, 60), order=1, policy=OcrPolicy.PRESERVE_AS_TABLE, label="table"),
        ),
    )
    ownership = build_layout_ownership(
        snapshot,
        (
            PpOcrV6LineHint(index=0, text="正文", bbox=(0, 0, 70, 40), words=()),
            PpOcrV6LineHint(index=1, text="表格", bbox=(80, 0, 140, 60), words=()),
        ),
        page_width=200,
        page_height=60,
    )

    assert [decision.text_block.block.uid for decision in ownership.decisions[:1]] == ["text"]
    assert ownership.decisions[0].is_structural_exclusion is False
    assert ownership.decisions[1].structural_block.block.uid == "table"
    assert ownership.decisions[1].is_structural_exclusion is True
    assert [group.block_uid for group in ownership.line_geometry_contexts()] == ["text"]


def test_layout_ownership_does_not_create_a_physical_group_for_vertical_text():
    snapshot = LayoutSnapshot(
        page_uid="page-1",
        revision=1,
        artifact_uid="layout-1",
        source_engine="test",
        source_run_id="layout-run-1",
        blocks=(
            _block(
                "vertical",
                BlockType.TEXT,
                (20, 0, 80, 100),
                order=0,
                policy=OcrPolicy.TEXT_OCR,
                label="vertical_text",
            ),
        ),
    )
    ownership = build_layout_ownership(
        snapshot,
        (PpOcrV6LineHint(index=4, text="宁", bbox=(20, 10, 80, 50), words=()),),
        page_width=100,
        page_height=100,
    )

    assert ownership.decisions[0].is_vertical_text is True
    assert ownership.line_geometry_contexts() == ()
