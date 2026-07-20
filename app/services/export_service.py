"""Pure export-boundary helpers over immutable project snapshots.

The export service does not discover or project runtime application objects.
The caller supplies one coherent ``ExportProjectSnapshot`` and this module
only indexes the immutable records contained by that snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from app.models.export_snapshot import ExportPageSnapshot, ExportProjectSnapshot
from app.models.layout_snapshot import LayoutBlockSnapshot
from app.models.ocr_records import OcrLine
from app.models.proof_records import ProofState, ProofTextUnit
from app.models.project_session import ProjectSession, RecordNotFoundError


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass(frozen=True, slots=True)
class ExportLineFacts:
    """The read-only text/proof facts used by every export format."""

    text: str
    ocr_text: str
    status: str = "unchecked"
    flags: tuple[str, ...] = ()

    @property
    def corrected(self) -> bool:
        return bool(self.ocr_text) and self.text != self.ocr_text


def sanitize_export_filename(name: str, fallback: str = "ocr_export") -> str:
    """Generate a cross-platform safe export filename stem."""
    cleaned = _INVALID_FILENAME_CHARS.sub("_", (name or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")
    if not cleaned:
        cleaned = fallback
    if cleaned.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_"
    return cleaned[:120]


def build_export_path(out_dir: str | Path, project_name: str, fmt: str) -> Path:
    """Build the output path for a snapshot export and create its directory."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    normalized = fmt.lower().strip().lstrip(".")
    base = sanitize_export_filename(project_name)
    if normalized == "txt":
        return directory / f"{base}.utf8.txt"
    if normalized in {"pdf-single", "pdf-dual"}:
        return directory / f"{base}.{normalized}.pdf"
    suffix = {
        "markdown": "md",
        "json": "json",
        "pdf": "pdf",
    }.get(normalized, normalized)
    return directory / f"{base}.{suffix}"


def iter_export_pages(snapshot: ExportProjectSnapshot) -> Iterable[ExportPageSnapshot]:
    """Yield pages in the order adopted by the immutable project snapshot."""
    _require_project_snapshot(snapshot)
    return snapshot.pages


def capture_export_snapshot(session: ProjectSession) -> ExportProjectSnapshot:
    """Capture one coherent export transaction from project repositories."""
    if not isinstance(session, ProjectSession):
        raise TypeError("export capture requires ProjectSession")
    ocr = session.ocr_observation_repository
    all_regions = ocr.all_regions()
    all_lines = ocr.all_lines()
    all_atoms = ocr.all_atoms()
    all_candidates = ocr.all_candidates()
    all_proof = session.proof_repository.all_states()
    all_bindings = session.binding_repository.all()
    all_table_texts = session.table_text_repository.all()
    pages: list[ExportPageSnapshot] = []
    for page in sorted(session.page_repository.all(), key=lambda item: item.page_number):
        layout = session.layout_repository.get(page.uid)
        try:
            pointer = ocr.get_active_pointer(page.uid)
        except RecordNotFoundError:
            batch = None
        else:
            batch = ocr.get_batch(pointer.batch_uid, fingerprint=pointer.batch_fingerprint)
        region_uids = set(batch.region_uids) if batch is not None else set()
        line_uids = set(batch.line_uids) if batch is not None else set()
        atom_uids = set(batch.atom_uids) if batch is not None else set()
        candidate_uids = set(batch.candidate_uids) if batch is not None else set()
        block_uids = {block.uid for block in layout.blocks}
        pages.append(ExportPageSnapshot(
            page=page,
            layout=layout,
            active_ocr_batch=batch,
            ocr_regions=tuple(item for item in all_regions if item.uid in region_uids),
            ocr_lines=tuple(item for item in all_lines if item.uid in line_uids),
            ocr_atoms=tuple(item for item in all_atoms if item.uid in atom_uids),
            ocr_candidates=tuple(item for item in all_candidates if item.uid in candidate_uids),
            proof_states=tuple(
                item for item in all_proof
                if item.anchor_snapshot.scope_uid == page.uid
            ),
            bindings=tuple(
                item for item in all_bindings
                if item.source_uid in block_uids or item.target_uid in region_uids
            ),
            table_texts=tuple(item for item in all_table_texts if item.page_uid == page.uid),
        ))
    return ExportProjectSnapshot(project=session.project_record, pages=tuple(pages))


def iter_export_blocks(
    page_snapshot: ExportPageSnapshot,
    *,
    include_empty: bool = False,
) -> Iterable[LayoutBlockSnapshot]:
    """Yield layout facts in snapshot order, optionally retaining empty blocks."""
    _require_page_snapshot(page_snapshot)
    for block in page_snapshot.layout.blocks:
        if include_empty or tuple(iter_export_lines(page_snapshot, block)):
            yield block


def iter_export_lines(
    page_snapshot: ExportPageSnapshot,
    block: LayoutBlockSnapshot,
) -> Iterable[OcrLine]:
    """Yield active OCR lines bound to one immutable layout block."""
    _require_page_snapshot(page_snapshot)
    if not isinstance(block, LayoutBlockSnapshot):
        raise TypeError("export block requires LayoutBlockSnapshot")

    region_uids = _region_uids_for_block(page_snapshot, block)
    active_line_uids = _active_line_uids(page_snapshot)
    lines = [
        line
        for line in page_snapshot.ocr_lines
        if line.page_uid == page_snapshot.page.uid
        and line.region_uid in region_uids
        and (active_line_uids is None or line.uid in active_line_uids)
    ]
    return tuple(sorted(lines, key=_line_sort_key))


def get_export_text(page_snapshot: ExportPageSnapshot, line: OcrLine) -> str:
    """Return proof text when it is adopted, otherwise the OCR observation."""
    return export_line_facts(page_snapshot, line).text


def export_line_facts(page_snapshot: ExportPageSnapshot, line: OcrLine) -> ExportLineFacts:
    """Resolve one OCR line against the page's current proof value records."""
    _require_page_snapshot(page_snapshot)
    if not isinstance(line, OcrLine):
        raise TypeError("export line requires OcrLine")
    if line.page_uid != page_snapshot.page.uid:
        raise ValueError("export line belongs to another page")
    return _proof_facts_by_line(page_snapshot).get(
        line.uid,
        ExportLineFacts(text=line.text, ocr_text=line.text),
    )


def build_export_summary(snapshot: ExportProjectSnapshot) -> dict[str, int]:
    """Return export readiness counters from snapshot records only."""
    _require_project_snapshot(snapshot)
    total_lines = 0
    confirmed_lines = 0
    modified_lines = 0
    flagged_lines = 0
    unrecognized_blocks = 0

    for page_snapshot in snapshot.pages:
        facts_by_line = _proof_facts_by_line(page_snapshot)
        for block in page_snapshot.layout.blocks:
            lines = tuple(iter_export_lines(page_snapshot, block))
            if _enum_value(block.ocr_policy) == "text_ocr" and not lines:
                unrecognized_blocks += 1
            for line in lines:
                total_lines += 1
                facts = facts_by_line.get(line.uid, ExportLineFacts(line.text, line.text))
                if facts.status == "ok":
                    confirmed_lines += 1
                elif facts.status == "modified" or facts.corrected:
                    modified_lines += 1
                elif facts.status == "auto_flagged" or facts.flags:
                    flagged_lines += 1

    return {
        "total_pages": len(snapshot.pages),
        "total_lines": total_lines,
        "unproofed_lines": max(0, total_lines - confirmed_lines - modified_lines),
        "flagged_lines": flagged_lines,
        "unrecognized_blocks": unrecognized_blocks,
    }


def check_export_readiness(snapshot: ExportProjectSnapshot) -> list[str]:
    """Return user-facing readiness warnings for one export transaction."""
    _require_project_snapshot(snapshot)
    summary = build_export_summary(snapshot)
    warnings: list[str] = []
    if not snapshot.pages:
        warnings.append("项目中没有页面")
        return warnings
    if summary["unrecognized_blocks"] > 0:
        warnings.append(f"{summary['unrecognized_blocks']} 个块尚未 OCR 识别")
    if summary["unproofed_lines"] > 0:
        warnings.append(f"{summary['unproofed_lines']} 行尚未校对确认")
    if summary["flagged_lines"] > 0:
        warnings.append(f"{summary['flagged_lines']} 行存在低置信度标记")
    return warnings


def _region_uids_for_block(
    page_snapshot: ExportPageSnapshot,
    block: LayoutBlockSnapshot,
) -> set[str]:
    """Resolve the explicit layout-block -> OCR-region binding."""
    region_uids = {
        binding.target_uid
        for binding in page_snapshot.bindings
        if binding.source_uid == block.uid and binding.relation == "observed_by"
    }
    known_region_uids = {region.uid for region in page_snapshot.ocr_regions}
    region_uids &= known_region_uids
    if not region_uids and block.uid in known_region_uids:
        # A direct shared stable UID is an explicit value-level association.
        region_uids.add(block.uid)
    return region_uids


def _active_line_uids(page_snapshot: ExportPageSnapshot) -> set[str] | None:
    batch = page_snapshot.active_ocr_batch
    return None if batch is None else set(batch.line_uids)


def _line_sort_key(line: OcrLine) -> tuple[int, int, int, str]:
    left, top, _right, _bottom = line.bbox
    return line.order, top, left, line.uid


def _proof_facts_by_line(page_snapshot: ExportPageSnapshot) -> dict[str, ExportLineFacts]:
    lines = tuple(
        sorted(
            (
                line
                for line in page_snapshot.ocr_lines
                if line.page_uid == page_snapshot.page.uid
                and (_active_line_uids(page_snapshot) is None
                     or line.uid in _active_line_uids(page_snapshot))
            ),
            key=_line_sort_key,
        )
    )
    facts = {line.uid: ExportLineFacts(text=line.text, ocr_text=line.text) for line in lines}
    states = [
        state
        for state in page_snapshot.proof_states
        if state.anchor_snapshot.scope_uid == page_snapshot.page.uid
        and not state.rebind_required
    ]
    if not states or not lines:
        return facts
    state = max(states, key=lambda item: (item.revision, item.uid))
    units = tuple(state.text_units)
    if not units:
        return facts

    direct_texts = _direct_proof_texts(units, len(lines))
    if direct_texts is not None:
        for line, (text, status) in zip(lines, direct_texts):
            facts[line.uid] = ExportLineFacts(
                text=text,
                ocr_text=line.text,
                status=status,
            )
        return facts

    aligned = _aligned_proof_texts(lines, state)
    for line_uid, value in aligned.items():
        facts[line_uid] = value
    return facts


def _direct_proof_texts(
    units: tuple[ProofTextUnit, ...],
    line_count: int,
) -> tuple[tuple[str, str], ...] | None:
    if len(units) == line_count:
        return tuple((unit.text, unit.status) for unit in units)
    if len(units) == 1 and line_count > 1:
        split = tuple(units[0].text.splitlines())
        if len(split) == line_count:
            return tuple((text, units[0].status) for text in split)
    return ((units[0].text, units[0].status),) if line_count == 1 else None


def _aligned_proof_texts(
    lines: tuple[OcrLine, ...],
    state: ProofState,
) -> dict[str, ExportLineFacts]:
    if not state.alignment_slices:
        return {}
    offsets: list[tuple[OcrLine, int, int]] = []
    cursor = 0
    for line in lines:
        end = cursor + len(line.text)
        offsets.append((line, cursor, end))
        cursor = end + 1

    result: dict[str, ExportLineFacts] = {}
    for line, line_start, line_end in offsets:
        slices = [
            item
            for item in state.alignment_slices
            if item.source_end > line_start and item.source_start < line_end
        ]
        if not slices:
            continue
        text = line.text
        for item in sorted(slices, key=lambda value: value.source_start, reverse=True):
            start = max(0, item.source_start - line_start)
            end = min(len(text), item.source_end - line_start)
            if start > end:
                continue
            text = text[:start] + item.proof_text + text[end:]
        status = "modified" if text != line.text else "unchecked"
        result[line.uid] = ExportLineFacts(text=text, ocr_text=line.text, status=status)
    return result


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _require_project_snapshot(snapshot: ExportProjectSnapshot) -> None:
    if not isinstance(snapshot, ExportProjectSnapshot):
        raise TypeError("export requires ExportProjectSnapshot")


def _require_page_snapshot(page_snapshot: ExportPageSnapshot) -> None:
    if not isinstance(page_snapshot, ExportPageSnapshot):
        raise TypeError("export requires ExportPageSnapshot")


__all__ = [
    "ExportLineFacts",
    "build_export_path",
    "build_export_summary",
    "check_export_readiness",
    "export_line_facts",
    "get_export_text",
    "iter_export_blocks",
    "iter_export_lines",
    "iter_export_pages",
    "sanitize_export_filename",
]
