"""Runtime store for adopted page layout snapshots.

The active ``Page`` dataclass still exposes ``blocks`` as the current runtime
projection.  This store keeps the adopted ``LayoutSnapshot`` outside that object
tree so layout truth can move without adding another field to ``Page``.
"""
from __future__ import annotations

import weakref

from .layout_snapshot import LayoutSnapshot


_SnapshotEntry = tuple[weakref.ReferenceType[object], LayoutSnapshot]
_SNAPSHOTS_BY_PAGE_OBJECT: dict[int, _SnapshotEntry] = {}


def layout_snapshot_for_page(page: object) -> LayoutSnapshot | None:
    entry = _SNAPSHOTS_BY_PAGE_OBJECT.get(id(page))
    if entry is None:
        return None
    page_ref, snapshot = entry
    if page_ref() is not page:
        _SNAPSHOTS_BY_PAGE_OBJECT.pop(id(page), None)
        return None
    return snapshot


def set_layout_snapshot_for_page(page: object, snapshot: LayoutSnapshot) -> None:
    page_id = id(page)

    def _cleanup(_ref: weakref.ReferenceType[object]) -> None:
        entry = _SNAPSHOTS_BY_PAGE_OBJECT.get(page_id)
        if entry is not None and entry[0] is _ref:
            _SNAPSHOTS_BY_PAGE_OBJECT.pop(page_id, None)

    _SNAPSHOTS_BY_PAGE_OBJECT[page_id] = (weakref.ref(page, _cleanup), snapshot)


def clear_layout_snapshot_for_page(page: object) -> None:
    _SNAPSHOTS_BY_PAGE_OBJECT.pop(id(page), None)


__all__ = [
    "clear_layout_snapshot_for_page",
    "layout_snapshot_for_page",
    "set_layout_snapshot_for_page",
]
