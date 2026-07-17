"""Exact, block-local alignment between VL text and PP physical rows."""
from __future__ import annotations

import unicodedata

from app.adapters.paddle.ppocr_v6_prepass import PpOcrV6LineHint
from app.core.ppocr_layout_ownership import LayoutOwnership
from app.models.ocr_routing_observation import (
    BlockAlignmentStatus,
    BlockObservationAlignment,
    BlockVlObservation,
    BlockVlObservationStatus,
    LineCutOwnershipDirective,
)
from app.models.physical_line_geometry import PhysicalLineResolution, ResolvedPhysicalLine


def align_block_observations(
    observations: tuple[BlockVlObservation, ...],
    physical_lines: PhysicalLineResolution,
    ownership: LayoutOwnership,
) -> tuple[BlockObservationAlignment, ...]:
    source_lines = {
        decision.line.index: decision.line
        for decision in ownership.decisions
    }
    rows_by_block: dict[str, list[ResolvedPhysicalLine]] = {}
    for row in physical_lines.rows:
        rows_by_block.setdefault(row.owner_block_uid, []).append(row)
    return tuple(
        _align_one_block(
            observation,
            tuple(rows_by_block.get(observation.block_uid, ())),
            source_lines,
        )
        for observation in observations
    )


def _align_one_block(
    observation: BlockVlObservation,
    rows: tuple[ResolvedPhysicalLine, ...],
    source_lines: dict[int, PpOcrV6LineHint],
) -> BlockObservationAlignment:
    if observation.status == BlockVlObservationStatus.EMPTY:
        return BlockObservationAlignment(
            block_uid=observation.block_uid,
            status=BlockAlignmentStatus.EMPTY,
            vl_observation_uid=observation.uid,
            detail="VL observation completed without text; PP routing remains authoritative for geometry",
        )
    marker, body = _leading_semantic_number_marker(observation.text)
    if not marker:
        vl_text = _normalize_alignment_text(observation.text)
        pp_text, source_indices = _normalized_block_pp_text(rows, source_lines)
        if vl_text == pp_text:
            return BlockObservationAlignment(
                block_uid=observation.block_uid,
                status=BlockAlignmentStatus.EXACT,
                vl_observation_uid=observation.uid,
                pp_source_indices=source_indices,
                detail="VL block text exactly matches the monotonic PP row stream",
            )
        return BlockObservationAlignment(
            block_uid=observation.block_uid,
            status=BlockAlignmentStatus.AMBIGUOUS,
            vl_observation_uid=observation.uid,
            pp_source_indices=source_indices,
            detail=(
                "VL block text does not exactly match the monotonic PP row stream: "
                f"vl={vl_text!r} pp={pp_text!r}"
            ),
        )

    full_normalized = _normalize_alignment_text(observation.text)
    body_normalized = _normalize_alignment_text(body)
    ordered_rows = sorted(rows, key=lambda row: (row.bbox[1], row.bbox[0], row.representative_index))
    for row in ordered_rows:
        row_sources = _row_sources(row, source_lines)
        if any(
            _normalize_alignment_text(source.text).startswith(_normalize_alignment_text(marker))
            and full_normalized.startswith(_normalize_alignment_text(source.text))
            for source in row_sources
            if _normalize_alignment_text(source.text)
        ):
            return BlockObservationAlignment(
                block_uid=observation.block_uid,
                status=BlockAlignmentStatus.EXACT,
                vl_observation_uid=observation.uid,
                pp_source_indices=row.source_indices,
                detail="PP row already preserves the explicit VL marker",
            )

    candidates: list[tuple[ResolvedPhysicalLine, PpOcrV6LineHint]] = []
    if body_normalized:
        for row in ordered_rows:
            for source in _row_sources(row, source_lines):
                source_text = _normalize_alignment_text(source.text)
                if source_text and body_normalized.startswith(source_text):
                    candidates.append((row, source))
                    break
    if len(candidates) != 1:
        return BlockObservationAlignment(
            block_uid=observation.block_uid,
            status=BlockAlignmentStatus.AMBIGUOUS,
            vl_observation_uid=observation.uid,
            detail=(
                "explicit VL marker cannot be aligned to one unique PP body source: "
                f"marker={marker!r} candidates={len(candidates)}"
            ),
        )

    row, body_source = candidates[0]
    ordered_sources = tuple(
        source
        for candidate_row in ordered_rows
        for source in _row_sources(candidate_row, source_lines)
    )
    body_position = ordered_sources.index(body_source)
    pp_body_text = "".join(
        _normalize_alignment_text(source.text)
        for source in ordered_sources[body_position:]
    )
    if pp_body_text != body_normalized:
        return BlockObservationAlignment(
            block_uid=observation.block_uid,
            status=BlockAlignmentStatus.AMBIGUOUS,
            vl_observation_uid=observation.uid,
            pp_source_indices=row.source_indices,
            detail=(
                "explicit VL marker body does not exactly match the monotonic PP stream: "
                f"vl={body_normalized!r} pp={pp_body_text!r}"
            ),
        )
    boundary_x = int(body_source.bbox[0])
    if boundary_x <= row.bbox[0] or boundary_x >= row.bbox[2]:
        return BlockObservationAlignment(
            block_uid=observation.block_uid,
            status=BlockAlignmentStatus.AMBIGUOUS,
            vl_observation_uid=observation.uid,
            pp_source_indices=row.source_indices,
            detail="aligned PP body has no distinct leading geometry for the explicit VL marker",
        )
    directive = LineCutOwnershipDirective(
        block_uid=observation.block_uid,
        line_index=row.representative_index,
        bbox=(row.bbox[0], row.bbox[1], boundary_x, row.bbox[3]),
        reason="vl_explicit_leading_number_marker",
        vl_text=marker,
    )
    return BlockObservationAlignment(
        block_uid=observation.block_uid,
        status=BlockAlignmentStatus.EXACT,
        vl_observation_uid=observation.uid,
        pp_source_indices=row.source_indices,
        directives=(directive,),
        detail="explicit VL marker aligned to the unique PP body prefix",
    )


def _row_sources(
    row: ResolvedPhysicalLine,
    source_lines: dict[int, PpOcrV6LineHint],
) -> tuple[PpOcrV6LineHint, ...]:
    return tuple(sorted(
        (source_lines[index] for index in row.source_indices if index in source_lines),
        key=lambda source: (source.bbox[0], source.index),
    ))


def _normalized_block_pp_text(
    rows: tuple[ResolvedPhysicalLine, ...],
    source_lines: dict[int, PpOcrV6LineHint],
) -> tuple[str, tuple[int, ...]]:
    ordered_rows = sorted(rows, key=lambda row: (row.bbox[1], row.bbox[0], row.representative_index))
    sources = tuple(
        source
        for row in ordered_rows
        for source in _row_sources(row, source_lines)
    )
    return (
        "".join(_normalize_alignment_text(source.text) for source in sources),
        tuple(source.index for source in sources),
    )


def _leading_semantic_number_marker(text: str) -> tuple[str, str]:
    value = str(text or "")
    start = 0
    while start < len(value) and value[start].isspace():
        start += 1
    end = start
    while end < len(value) and _is_semantic_number_marker(value[end]):
        end += 1
    if end == start:
        return "", value
    return value[start:end], value[end:].lstrip()


def _is_semantic_number_marker(char: str) -> bool:
    name = unicodedata.name(char, "")
    return (
        ("CIRCLED" in name or "PARENTHESIZED" in name)
        and ("DIGIT" in name or "NUMBER" in name)
    )


def _normalize_alignment_text(text: str) -> str:
    # Preserve Unicode semantic identity: NFKC would turn ⑪ into plain digits.
    return "".join(char for char in str(text or "") if not char.isspace())


__all__ = ["align_block_observations"]
