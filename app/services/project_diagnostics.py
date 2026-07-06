"""Project diagnostics for persisted proof/OCR truth drift.

The diagnostics are read-only.  They report facts that need human review or a
dedicated repair tool; they must not mutate project data while scanning.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.core.proof_char_text import char_display_text, chars_display_text
from app.core.proof_line_facts import proof_display_text
from app.models import Block, Char, Line, OcrProject, Page
from app.models.layout_projection import page_layout_blocks
from app.models.layout_snapshot_store import layout_snapshot_for_page
from app.models.ocr_character_observation import iter_line_ocr_char_occurrences, line_ocr_chars
from app.models.ocr_observation import block_ocr_lines


@dataclass(frozen=True)
class ProjectDiagnosticIssue:
    severity: str
    code: str
    message: str
    page_number: int | None = None
    page_uid: str = ""
    block_uid: str = ""
    line_uid: str = ""
    char_uid: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "page_number": self.page_number,
            "page_uid": self.page_uid,
            "block_uid": self.block_uid,
            "line_uid": self.line_uid,
            "char_uid": self.char_uid,
            "details": self.details,
        }


@dataclass(frozen=True)
class ProjectDiagnosticReport:
    project_name: str
    issues: tuple[ProjectDiagnosticIssue, ...]

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def counts_by_code(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "issue_count": self.issue_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "counts_by_code": self.counts_by_code(),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Project Diagnostics: {self.project_name}",
            "",
            f"- issues: {self.issue_count}",
            f"- errors: {self.error_count}",
            f"- warnings: {self.warning_count}",
            "",
        ]
        if not self.issues:
            lines.append("No issues found.")
            return "\n".join(lines)
        lines.append("| severity | code | page | message |")
        lines.append("|---|---|---:|---|")
        for issue in self.issues:
            page = "" if issue.page_number is None else str(issue.page_number)
            lines.append(
                f"| {issue.severity} | {issue.code} | {page} | "
                f"{_escape_markdown_cell(issue.message)} |"
            )
        return "\n".join(lines)


def diagnose_project(
    project: OcrProject,
    *,
    probe_store: Any | None = None,
) -> ProjectDiagnosticReport:
    issues: list[ProjectDiagnosticIssue] = []
    issues.extend(_diagnose_duplicate_uids(project))
    issues.extend(_diagnose_layout_projection_drift(project))
    issues.extend(_diagnose_line_char_contract(project))
    if probe_store is not None:
        issues.extend(_diagnose_probe_anchors(project, probe_store))
    return ProjectDiagnosticReport(project_name=project.name, issues=tuple(issues))


def _diagnose_duplicate_uids(project: OcrProject) -> list[ProjectDiagnosticIssue]:
    issues: list[ProjectDiagnosticIssue] = []
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = {
        "page": defaultdict(list),
        "block": defaultdict(list),
        "line": defaultdict(list),
        "char": defaultdict(list),
    }
    for page_idx, page in enumerate(project.pages):
        _record_uid(buckets["page"], page.uid, page, page_idx)
        for block_idx, block in enumerate(page_layout_blocks(page)):
            _record_uid(buckets["block"], block.uid, page, page_idx, block, block_idx)
            for line_idx, line in enumerate(block_ocr_lines(block)):
                _record_uid(
                    buckets["line"],
                    line.uid,
                    page,
                    page_idx,
                    block,
                    block_idx,
                    line,
                    line_idx,
                )
                for occurrence in iter_line_ocr_char_occurrences(line):
                    _record_uid(
                        buckets["char"],
                        occurrence.char.uid,
                        page,
                        page_idx,
                        block,
                        block_idx,
                        line,
                        line_idx,
                        occurrence.char,
                        occurrence.char_index,
                    )
    for kind, values in buckets.items():
        for uid, locations in values.items():
            if uid and len(locations) > 1:
                first = locations[0]
                issues.append(
                    ProjectDiagnosticIssue(
                        severity="error",
                        code=f"duplicate_{kind}_uid",
                        message=f"duplicate {kind} uid {uid}",
                        page_number=first.get("page_number"),
                        page_uid=first.get("page_uid", ""),
                        block_uid=first.get("block_uid", ""),
                        line_uid=first.get("line_uid", ""),
                        char_uid=uid if kind == "char" else "",
                        details={"uid": uid, "locations": locations},
                    )
                )
    return issues


def _diagnose_layout_projection_drift(project: OcrProject) -> list[ProjectDiagnosticIssue]:
    issues: list[ProjectDiagnosticIssue] = []
    for page in project.pages:
        snapshot = layout_snapshot_for_page(page)
        if snapshot is None:
            continue
        runtime_blocks = page_layout_blocks(page)
        runtime_by_uid = {
            block.uid: (block_idx, block)
            for block_idx, block in enumerate(runtime_blocks)
            if block.uid
        }
        snapshot_uids = {block.uid for block in snapshot.blocks if block.uid}
        for snapshot_idx, snapshot_block in enumerate(snapshot.blocks):
            pair = runtime_by_uid.get(snapshot_block.uid)
            if pair is None:
                issues.append(ProjectDiagnosticIssue(
                    severity="error",
                    code="layout_snapshot_missing_runtime_block",
                    message="layout snapshot block has no runtime projection",
                    page_number=page.page_number,
                    page_uid=page.uid,
                    block_uid=snapshot_block.uid,
                    details={"snapshot_index": snapshot_idx},
                ))
                continue
            runtime_idx, runtime_block = pair
            drift: dict[str, Any] = {}
            if runtime_block.block_type != snapshot_block.block_type:
                drift["block_type"] = {
                    "snapshot": snapshot_block.block_type.value,
                    "runtime": runtime_block.block_type.value,
                }
            if runtime_block.bbox != snapshot_block.bbox:
                drift["bbox"] = {
                    "snapshot": snapshot_block.bbox.to_dict(),
                    "runtime": runtime_block.bbox.to_dict(),
                }
            if runtime_block.order != snapshot_block.order:
                drift["order"] = {
                    "snapshot": snapshot_block.order,
                    "runtime": runtime_block.order,
                }
            if runtime_block.source_label != snapshot_block.source_label:
                drift["source_label"] = {
                    "snapshot": snapshot_block.source_label,
                    "runtime": runtime_block.source_label,
                }
            if drift:
                issues.append(ProjectDiagnosticIssue(
                    severity="error",
                    code="layout_projection_drift",
                    message="runtime block projection diverges from layout snapshot",
                    page_number=page.page_number,
                    page_uid=page.uid,
                    block_uid=snapshot_block.uid,
                    details={
                        "snapshot_index": snapshot_idx,
                        "runtime_block_index": runtime_idx,
                        "drift": drift,
                    },
                ))
        for runtime_idx, runtime_block in enumerate(runtime_blocks):
            if runtime_block.uid and runtime_block.uid not in snapshot_uids:
                issues.append(ProjectDiagnosticIssue(
                    severity="warning",
                    code="layout_runtime_orphan_block",
                    message="runtime block projection is not present in layout snapshot",
                    page_number=page.page_number,
                    page_uid=page.uid,
                    block_uid=runtime_block.uid,
                    details={"runtime_block_index": runtime_idx},
                ))
    return issues


def _record_uid(
    bucket: dict[str, list[dict[str, Any]]],
    uid: str,
    page: Page,
    page_idx: int,
    block: Block | None = None,
    block_idx: int | None = None,
    line: Line | None = None,
    line_idx: int | None = None,
    char: Char | None = None,
    char_idx: int | None = None,
) -> None:
    if not uid:
        return
    bucket[uid].append({
        "page_index": page_idx,
        "page_number": page.page_number,
        "page_uid": page.uid,
        "block_index": block_idx,
        "block_uid": getattr(block, "uid", "") if block is not None else "",
        "line_index": line_idx,
        "line_uid": getattr(line, "uid", "") if line is not None else "",
        "char_index": char_idx,
        "char_uid": getattr(char, "uid", "") if char is not None else "",
    })


def _diagnose_line_char_contract(project: OcrProject) -> list[ProjectDiagnosticIssue]:
    issues: list[ProjectDiagnosticIssue] = []
    for page, block, line, _block_idx, _line_idx in _iter_lines(project):
        display_text = proof_display_text(line)
        chars = list(line_ocr_chars(line))
        if display_text and not chars:
            issues.append(_line_issue(
                "warning",
                "line_without_chars",
                "line has proof text but no OCR char carriers",
                page,
                block,
                line,
                details={"proof_text": _excerpt(display_text)},
            ))
            continue
        if chars:
            carrier_text = chars_display_text(chars)
            if carrier_text != display_text:
                issues.append(_line_issue(
                    "error",
                    "line_chars_text_mismatch",
                    "proof display text does not match OCR char carriers",
                    page,
                    block,
                    line,
                    details={
                        "proof_text": _excerpt(display_text),
                        "chars_text": _excerpt(carrier_text),
                        "proof_len": len(display_text),
                        "chars_len": len(carrier_text),
                    },
                ))
        for char_idx, char in enumerate(chars):
            text = char_display_text(char)
            if text and char.bbox is None:
                issues.append(_char_issue(
                    "warning",
                    "char_missing_bbox",
                    "OCR char carrier has text but no bbox",
                    page,
                    block,
                    line,
                    char,
                    details={"char_index": char_idx, "text": _excerpt(text)},
                ))
            elif char.bbox is not None and (char.bbox.w <= 0 or char.bbox.h <= 0):
                issues.append(_char_issue(
                    "error",
                    "char_invalid_bbox",
                    "OCR char carrier bbox has non-positive size",
                    page,
                    block,
                    line,
                    char,
                    details={"char_index": char_idx, "bbox": char.bbox.to_dict()},
                ))
    return issues


def _diagnose_probe_anchors(
    project: OcrProject,
    probe_store: Any,
) -> list[ProjectDiagnosticIssue]:
    issues: list[ProjectDiagnosticIssue] = []
    pages_by_no = {page.page_number: page for page in project.pages}
    for probe in _probe_iter(probe_store):
        key = probe.key
        page = pages_by_no.get(key.page_number)
        if page is None:
            issues.append(_probe_issue(
                "warning",
                "probe_anchor_missing_page",
                "quality probe points to a missing page",
                details={"probe": _probe_details(probe)},
            ))
            continue
        blocks = page_layout_blocks(page)
        if key.block_index < 0 or key.block_index >= len(blocks):
            issues.append(_probe_issue(
                "warning",
                "probe_anchor_missing_block",
                "quality probe points to a missing block",
                page=page,
                details={"probe": _probe_details(probe)},
            ))
            continue
        block = blocks[key.block_index]
        lines = block_ocr_lines(block)
        if key.line_index < 0 or key.line_index >= len(lines):
            issues.append(_probe_issue(
                "warning",
                "probe_anchor_missing_line",
                "quality probe points to a missing OCR line",
                page=page,
                block=block,
                details={"probe": _probe_details(probe)},
            ))
            continue
        line = lines[key.line_index]
        text = proof_display_text(line)
        if key.char_index < 0 or key.char_index >= len(text):
            issues.append(_probe_issue(
                "warning",
                "probe_anchor_missing_char",
                "quality probe points outside the current proof text",
                page=page,
                block=block,
                line=line,
                details={"probe": _probe_details(probe), "proof_len": len(text)},
            ))
            continue
        if probe.observation == "pending" and text[key.char_index] != probe.true_char:
            issues.append(_probe_issue(
                "warning",
                "stale_pending_probe_anchor",
                "pending quality probe anchor no longer matches true_char",
                page=page,
                block=block,
                line=line,
                details={
                    "probe": _probe_details(probe),
                    "current_char": text[key.char_index],
                },
            ))
    return issues


def _iter_lines(project: OcrProject) -> Iterable[tuple[Page, Block, Line, int, int]]:
    for page in project.pages:
        for block_idx, block in enumerate(page_layout_blocks(page)):
            for line_idx, line in enumerate(block_ocr_lines(block)):
                yield page, block, line, block_idx, line_idx


def _line_issue(
    severity: str,
    code: str,
    message: str,
    page: Page,
    block: Block,
    line: Line,
    *,
    details: dict[str, Any],
) -> ProjectDiagnosticIssue:
    return ProjectDiagnosticIssue(
        severity=severity,
        code=code,
        message=message,
        page_number=page.page_number,
        page_uid=page.uid,
        block_uid=block.uid,
        line_uid=line.uid,
        details=details,
    )


def _char_issue(
    severity: str,
    code: str,
    message: str,
    page: Page,
    block: Block,
    line: Line,
    char: Char,
    *,
    details: dict[str, Any],
) -> ProjectDiagnosticIssue:
    return ProjectDiagnosticIssue(
        severity=severity,
        code=code,
        message=message,
        page_number=page.page_number,
        page_uid=page.uid,
        block_uid=block.uid,
        line_uid=line.uid,
        char_uid=char.uid,
        details=details,
    )


def _probe_issue(
    severity: str,
    code: str,
    message: str,
    *,
    page: Page | None = None,
    block: Block | None = None,
    line: Line | None = None,
    details: dict[str, Any],
) -> ProjectDiagnosticIssue:
    return ProjectDiagnosticIssue(
        severity=severity,
        code=code,
        message=message,
        page_number=page.page_number if page is not None else None,
        page_uid=page.uid if page is not None else "",
        block_uid=block.uid if block is not None else "",
        line_uid=line.uid if line is not None else "",
        details=details,
    )


def _probe_iter(probe_store: Any) -> Iterable[Any]:
    all_fn = getattr(probe_store, "all", None)
    if callable(all_fn):
        return list(all_fn())
    return []


def _probe_details(probe: Any) -> dict[str, Any]:
    key = getattr(probe, "key", None)
    return {
        "page_number": getattr(key, "page_number", None),
        "block_index": getattr(key, "block_index", None),
        "line_index": getattr(key, "line_index", None),
        "char_index": getattr(key, "char_index", None),
        "true_char": getattr(probe, "true_char", ""),
        "fake_char": getattr(probe, "fake_char", ""),
        "observation": getattr(probe, "observation", ""),
    }


def _excerpt(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _escape_markdown_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
