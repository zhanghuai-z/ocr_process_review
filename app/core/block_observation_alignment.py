"""Block-local alignment between VL text and PP physical rows."""
from __future__ import annotations

from difflib import SequenceMatcher
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


_MARKER_BODY_MATCH_THRESHOLD = 0.60


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
        return BlockObservationAlignment(
            block_uid=observation.block_uid,
            status=BlockAlignmentStatus.NOT_REQUIRED,
            vl_observation_uid=observation.uid,
            detail="ordinary VL block text does not require cross-engine text alignment",
        )

    body_normalized = _normalize_alignment_text(body)
    ordered_rows = sorted(rows, key=lambda row: (row.bbox[1], row.bbox[0], row.representative_index))
    ordered_source_rows = tuple(
        (candidate_row, source)
        for candidate_row in ordered_rows
        for source in _row_sources(candidate_row, source_lines)
    )
    ordered_sources = tuple(source for _row, source in ordered_source_rows)
    preserved_candidates: list[tuple[ResolvedPhysicalLine, float]] = []
    for source_position, (row, source) in enumerate(ordered_source_rows):
        source_marker, source_body = _leading_semantic_number_marker(source.text)
        if source_marker != marker:
            continue
        pp_body_text = _normalize_alignment_text(source_body) + "".join(
            _normalize_alignment_text(item.text)
            for item in ordered_sources[source_position + 1:]
        )
        body_match_ratio = (
            _text_match_ratio(body_normalized, pp_body_text)
            if body_normalized
            else 1.0
        )
        if body_match_ratio >= _MARKER_BODY_MATCH_THRESHOLD:
            preserved_candidates.append((row, body_match_ratio))
    if len(preserved_candidates) == 1:
        row, body_match_ratio = preserved_candidates[0]
        return BlockObservationAlignment(
            block_uid=observation.block_uid,
            status=(
                BlockAlignmentStatus.EXACT
                if body_match_ratio == 1.0
                else BlockAlignmentStatus.MATCHED
            ),
            vl_observation_uid=observation.uid,
            pp_source_indices=row.source_indices,
            detail=(
                "PP row preserves the explicit VL marker and its local body matches: "
                f"body_match_ratio={body_match_ratio:.3f}"
            ),
        )

    marker_number = _semantic_marker_number(marker)
    candidates: list[
        tuple[ResolvedPhysicalLine, PpOcrV6LineHint, float]
    ] = []
    if body_normalized:
        for row in ordered_rows:
            row_sources = _row_sources(row, source_lines)
            if not row_sources:
                continue
            body_source_index = 0
            if (
                marker_number
                and _normalize_alignment_text(row_sources[0].text) == marker_number
            ):
                body_source_index = 1
            if body_source_index >= len(row_sources):
                continue
            body_source = row_sources[body_source_index]
            boundary_x = int(body_source.bbox[0])
            if boundary_x <= row.bbox[0] or boundary_x >= row.bbox[2]:
                continue
            body_position = ordered_sources.index(body_source)
            pp_body_text = "".join(
                _normalize_alignment_text(source.text)
                for source in ordered_sources[body_position:]
            )
            body_match_ratio = _text_match_ratio(body_normalized, pp_body_text)
            if body_match_ratio >= _MARKER_BODY_MATCH_THRESHOLD:
                candidates.append((row, body_source, body_match_ratio))
    if len(candidates) != 1:
        return BlockObservationAlignment(
            block_uid=observation.block_uid,
            status=BlockAlignmentStatus.AMBIGUOUS,
            vl_observation_uid=observation.uid,
            detail=(
                "explicit VL marker cannot be aligned to one unique PP body source: "
                f"marker={marker!r} threshold={_MARKER_BODY_MATCH_THRESHOLD:.3f} "
                f"candidates={len(candidates)}"
            ),
        )

    row, body_source, body_match_ratio = candidates[0]
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
        status=(
            BlockAlignmentStatus.EXACT
            if body_match_ratio == 1.0
            else BlockAlignmentStatus.MATCHED
        ),
        vl_observation_uid=observation.uid,
        pp_source_indices=row.source_indices,
        directives=(directive,),
        detail=(
            "explicit VL marker aligned to the unique PP body prefix: "
            f"body_match_ratio={body_match_ratio:.3f}"
        ),
    )


def _row_sources(
    row: ResolvedPhysicalLine,
    source_lines: dict[int, PpOcrV6LineHint],
) -> tuple[PpOcrV6LineHint, ...]:
    return tuple(sorted(
        (source_lines[index] for index in row.source_indices if index in source_lines),
        key=lambda source: (source.bbox[0], source.index),
    ))


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


def _semantic_marker_number(marker: str) -> str:
    values: list[str] = []
    for char in marker:
        try:
            number = unicodedata.numeric(char)
        except (TypeError, ValueError):
            return ""
        if not float(number).is_integer():
            return ""
        values.append(str(int(number)))
    return "".join(values)


def _normalize_alignment_text(text: str) -> str:
    # Preserve Unicode semantic identity: NFKC would turn ⑪ into plain digits.
    return "".join(char for char in str(text or "") if not char.isspace())


def _text_match_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


__all__ = ["align_block_observations"]
