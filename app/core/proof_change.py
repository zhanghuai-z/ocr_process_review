"""Shared proof-edit change contract.

This module is intentionally UI-free.  Proof panels, probe services, and
persistence code use the same object so text/status/probe mutations cannot be
lost behind a bare boolean return value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProofLineRef:
    page_uid: str = ""
    block_uid: str = ""
    line_uid: str = ""
    page_id: int | None = None
    block_id: int | None = None
    line_id: int | None = None
    write_chars: bool = False
    line: Any = None

    @property
    def key(self) -> tuple[object, ...]:
        if self.line_uid:
            return ("uid", self.line_uid)
        if self.line_id is not None:
            return ("id", self.line_id)
        return ("object", id(self.line))

    @classmethod
    def from_entities(
        cls,
        page: object,
        block: object,
        line: object,
        *,
        write_chars: bool = False,
    ) -> "ProofLineRef":
        return cls(
            page_uid=str(getattr(page, "uid", "") or ""),
            block_uid=str(getattr(block, "uid", "") or ""),
            line_uid=str(getattr(line, "uid", "") or ""),
            page_id=getattr(page, "id", None),
            block_id=getattr(block, "id", None),
            line_id=getattr(line, "id", None),
            write_chars=write_chars,
            line=line,
        )


@dataclass(frozen=True)
class ProofChangeSet:
    text_changed: bool = False
    status_changed: bool = False
    probe_changed: bool = False
    index_changed: bool = False
    conflict: bool = False
    cancelled: bool = False
    readonly: bool = False
    line_refs: tuple[ProofLineRef, ...] = tuple()

    @property
    def needs_persist(self) -> bool:
        return self.text_changed or self.status_changed or self.probe_changed

    @property
    def requires_line_scope(self) -> bool:
        return self.text_changed or self.status_changed

    @property
    def has_required_scope(self) -> bool:
        return not self.requires_line_scope or bool(self.line_refs)

    @property
    def changed(self) -> bool:
        return self.needs_persist or self.index_changed

    @property
    def blocked(self) -> bool:
        return self.conflict or self.cancelled or self.readonly

    def __bool__(self) -> bool:
        return self.changed or self.blocked

    def merge(self, other: "ProofChangeSet") -> "ProofChangeSet":
        return ProofChangeSet(
            text_changed=self.text_changed or other.text_changed,
            status_changed=self.status_changed or other.status_changed,
            probe_changed=self.probe_changed or other.probe_changed,
            index_changed=self.index_changed or other.index_changed,
            conflict=self.conflict or other.conflict,
            cancelled=self.cancelled or other.cancelled,
            readonly=self.readonly or other.readonly,
            line_refs=_merge_line_refs(self.line_refs, other.line_refs),
        )

    def scoped_to_line(
        self,
        page: object,
        block: object,
        line: object,
        *,
        write_chars: bool | None = None,
    ) -> "ProofChangeSet":
        ref = ProofLineRef.from_entities(
            page,
            block,
            line,
            write_chars=self.text_changed if write_chars is None else write_chars,
        )
        return ProofChangeSet(
            text_changed=self.text_changed,
            status_changed=self.status_changed,
            probe_changed=self.probe_changed,
            index_changed=self.index_changed,
            conflict=self.conflict,
            cancelled=self.cancelled,
            readonly=self.readonly,
            line_refs=_merge_line_refs(self.line_refs, (ref,)),
        )

    @classmethod
    def combine(cls, changes: list["ProofChangeSet"]) -> "ProofChangeSet":
        result = cls()
        for change in changes:
            result = result.merge(change)
        return result


def _merge_line_refs(
    left: tuple[ProofLineRef, ...],
    right: tuple[ProofLineRef, ...],
) -> tuple[ProofLineRef, ...]:
    merged: dict[tuple[object, ...], ProofLineRef] = {}
    for ref in (*left, *right):
        key = ref.key
        existing = merged.get(key)
        if existing is None:
            merged[key] = ref
            continue
        merged[key] = ProofLineRef(
            page_uid=existing.page_uid or ref.page_uid,
            block_uid=existing.block_uid or ref.block_uid,
            line_uid=existing.line_uid or ref.line_uid,
            page_id=existing.page_id if existing.page_id is not None else ref.page_id,
            block_id=existing.block_id if existing.block_id is not None else ref.block_id,
            line_id=existing.line_id if existing.line_id is not None else ref.line_id,
            write_chars=existing.write_chars or ref.write_chars,
            line=existing.line if existing.line is not None else ref.line,
        )
    return tuple(merged.values())
