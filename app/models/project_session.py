"""Independent project-scoped session and repository contracts.

The session owns every repository instance.  No repository uses process-wide
state, and the OCR repository keeps machine observations append-only.  Layout
snapshots and proof state are separate value boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace as dc_replace
from enum import Enum
import hashlib
import json
from typing import cast

from .layout_snapshot import LayoutSnapshot
from .paddle_artifact import PaddleArtifact
from .ocr_records import (
    OcrActivePointer,
    OcrAtom,
    OcrBatch,
    OcrCandidate,
    OcrLine,
    OcrRegion,
    OcrRun,
)
from .proof_records import (
    ProofAlignmentSegment,
    ProofAlignmentSlice,
    ProofAnchorSnapshot,
    ProofState,
    ProofTextUnit,
)


class RepositoryError(Exception):
    """Base class for repository contract failures."""


class ProjectScopeError(RepositoryError, ValueError):
    """A record does not belong to the repository project."""


class RecordNotFoundError(RepositoryError, KeyError):
    """A stable UID is not present in the scoped repository."""


class DuplicateUidError(RepositoryError, ValueError):
    """A create or append operation would reuse a UID."""


class InvalidRecordError(RepositoryError, ValueError):
    """A record violates a repository-level invariant."""


class RepositoryConflictError(RepositoryError, RuntimeError):
    """Base class for optimistic-concurrency conflicts."""


class RevisionConflictError(RepositoryConflictError):
    """The caller did not write from the current revision."""


class FingerprintConflictError(RevisionConflictError):
    """The caller did not write from the current content fingerprint."""


def _required_uid(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty stable UID")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    return value


def _required_text(value: object, field_name: str) -> str:
    result = _text(value, field_name)
    if not result or result != result.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return result


def _optional_path(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    result = _text(value, field_name)
    if not result or result != result.strip():
        raise ValueError(f"{field_name} must be a non-empty path or None")
    return result


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _bbox(value: object) -> tuple[int, int, int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise TypeError("bbox must contain four integer coordinates")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError("bbox coordinates must be integers")
    result = tuple(value)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("bbox must be non-empty")
    return result  # type: ignore[return-value]


def _metadata(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("attributes must be a sequence of string pairs")
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("attributes must contain string pairs")
        key, item_value = item
        if not isinstance(key, str) or not key:
            raise ValueError("attribute keys must be non-empty strings")
        if not isinstance(item_value, str):
            raise TypeError("attribute values must be strings")
        result.append((key, item_value))
    if len({key for key, _value in result}) != len(result):
        raise ValueError("attributes must not contain duplicate keys")
    return tuple(result)


def _canonical(value: object, *, omit_version: bool = False) -> object:
    if isinstance(value, Enum):
        return _canonical(value.value, omit_version=omit_version)
    if is_dataclass(value):
        return {
            item.name: _canonical(getattr(value, item.name), omit_version=omit_version)
            for item in fields(value)
            if item.name != "fingerprint"
            and not (omit_version and item.name == "revision")
        }
    if isinstance(value, tuple):
        return [_canonical(item, omit_version=omit_version) for item in value]
    if isinstance(value, list):
        return [_canonical(item, omit_version=omit_version) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical(item, omit_version=omit_version)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot fingerprint value of type {type(value).__name__}")


def _fingerprint(value: object, *, omit_version: bool = False) -> str:
    payload = _canonical(value, omit_version=omit_version)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _set_record_fingerprint(record: object) -> None:
    object.__setattr__(record, "fingerprint", _fingerprint(record, omit_version=True))


def _check_snapshot(
    record: object,
    *,
    revision: int | None = None,
    fingerprint: str | None = None,
) -> None:
    current_revision = getattr(record, "revision", None)
    current_fingerprint = getattr(record, "fingerprint", None)
    if revision is not None:
        _non_negative_int(revision, "expected revision")
        if current_revision != revision:
            raise RevisionConflictError(
                f"revision mismatch: expected {revision}, current {current_revision}"
            )
    if fingerprint is not None:
        expected = _required_text(fingerprint, "expected fingerprint")
        if current_fingerprint != expected:
            raise FingerprintConflictError("fingerprint mismatch for immutable record")


class LayoutRepository:
    """Current page layout truth keyed uniquely by ``page_uid``."""

    __slots__ = ("_project_uid", "_snapshots")

    def __init__(self, project_uid: str) -> None:
        self._project_uid = _required_uid(project_uid, "project_uid")
        self._snapshots: dict[str, LayoutSnapshot] = {}

    @property
    def project_uid(self) -> str:
        return self._project_uid

    def put(self, next_snapshot: LayoutSnapshot, expected_revision: int) -> LayoutSnapshot:
        if not isinstance(next_snapshot, LayoutSnapshot):
            raise TypeError("layout repository requires LayoutSnapshot")
        page_uid = _required_uid(next_snapshot.page_uid, "page_uid")
        expected = _non_negative_int(expected_revision, "expected revision")
        next_revision = _non_negative_int(next_snapshot.revision, "next snapshot revision")
        current = self._snapshots.get(page_uid)
        if current is None:
            if expected != 0:
                raise RevisionConflictError(
                    "first layout snapshot requires expected revision 0"
                )
        elif current.revision != expected:
            raise RevisionConflictError(
                f"layout revision mismatch: expected {expected}, "
                f"current {current.revision}"
            )
        if next_revision != expected + 1:
            raise RevisionConflictError(
                "next layout snapshot revision must equal expected revision plus one"
            )
        self._snapshots[page_uid] = next_snapshot
        return next_snapshot

    def get(
        self,
        page_uid: str,
        *,
        revision: int | None = None,
        fingerprint: str | None = None,
    ) -> LayoutSnapshot:
        stable_page_uid = _required_uid(page_uid, "page_uid")
        try:
            snapshot = self._snapshots[stable_page_uid]
        except KeyError as exc:
            raise RecordNotFoundError(
                f"layout snapshot is not present for page {stable_page_uid!r}"
            ) from exc
        if revision is not None:
            _non_negative_int(revision, "expected revision")
            if snapshot.revision != revision:
                raise RevisionConflictError(
                    f"revision mismatch: expected {revision}, "
                    f"current {snapshot.revision}"
                )
        if fingerprint is not None:
            expected = _required_text(fingerprint, "expected fingerprint")
            if self.fingerprint_for(snapshot) != expected:
                raise FingerprintConflictError("fingerprint mismatch for layout snapshot")
        return snapshot

    @staticmethod
    def fingerprint_for(snapshot: LayoutSnapshot) -> str:
        if not isinstance(snapshot, LayoutSnapshot):
            raise TypeError("layout repository requires LayoutSnapshot")
        return _fingerprint(snapshot)

    def all(self) -> tuple[LayoutSnapshot, ...]:
        return tuple(self._snapshots[uid] for uid in sorted(self._snapshots))

    def _restore(self, snapshot: LayoutSnapshot) -> None:
        """Hydrate one persisted current snapshot; only ProjectSession may call this."""
        if not isinstance(snapshot, LayoutSnapshot):
            raise TypeError("layout repository requires LayoutSnapshot")
        page_uid = _required_uid(snapshot.page_uid, "page_uid")
        if page_uid in self._snapshots:
            raise DuplicateUidError(f"layout page UID already exists: {page_uid!r}")
        if snapshot.revision < 1:
            raise InvalidRecordError("persisted layout revision must be positive")
        self._snapshots[page_uid] = snapshot

    def __len__(self) -> int:
        return len(self._snapshots)


class PaddleArtifactRepository:
    """Project-scoped append-only Paddle facts."""

    __slots__ = ("_project_uid", "_records")

    def __init__(self, project_uid: str) -> None:
        self._project_uid = _required_uid(project_uid, "project_uid")
        self._records: dict[str, PaddleArtifact] = {}

    @property
    def project_uid(self) -> str:
        return self._project_uid

    def append(self, record: PaddleArtifact) -> PaddleArtifact:
        if not isinstance(record, PaddleArtifact):
            raise TypeError("artifact repository requires PaddleArtifact")
        if record.project_uid != self.project_uid:
            raise ProjectScopeError("Paddle artifact belongs to another project")
        if record.uid in self._records:
            raise DuplicateUidError(f"Paddle artifact UID already exists: {record.uid!r}")
        self._records[record.uid] = record
        return record

    def get(self, uid: str, *, fingerprint: str | None = None) -> PaddleArtifact:
        stable_uid = _required_uid(uid, "uid")
        try:
            record = self._records[stable_uid]
        except KeyError as exc:
            raise RecordNotFoundError(f"Paddle artifact UID is not present: {stable_uid!r}") from exc
        if fingerprint is not None and record.fingerprint != _required_text(fingerprint, "fingerprint"):
            raise FingerprintConflictError("Paddle artifact fingerprint mismatch")
        return record

    def all(self) -> tuple[PaddleArtifact, ...]:
        return tuple(self._records[uid] for uid in sorted(self._records))


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    """Immutable project identity and user-visible metadata."""

    project_uid: str
    name: str = ""
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _text(self.name, "name")
        _set_record_fingerprint(self)


@dataclass(frozen=True, slots=True)
class PageRecord:
    """Immutable page metadata without layout blocks or runtime objects."""

    project_uid: str
    uid: str
    image_path: str
    source_path: str
    cache_image_path: str
    thumbnail_path: str
    width: int
    height: int
    page_number: int
    source_page_index: int
    status: str
    error: str
    image_hash: str
    image_revision: int
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _text(self.image_path, "image_path")
        _text(self.source_path, "source_path")
        _text(self.cache_image_path, "cache_image_path")
        _text(self.thumbnail_path, "thumbnail_path")
        if _non_negative_int(self.width, "width") == 0:
            raise ValueError("width must be positive")
        if _non_negative_int(self.height, "height") == 0:
            raise ValueError("height must be positive")
        if _non_negative_int(self.page_number, "page_number") == 0:
            raise ValueError("page_number must be positive")
        _non_negative_int(self.source_page_index, "source_page_index")
        _required_text(self.status, "status")
        _text(self.error, "error")
        _text(self.image_hash, "image_hash")
        _non_negative_int(self.image_revision, "image_revision")
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint(self, omit_version=False),
        )

    @property
    def page_uid(self) -> str:
        return self.uid


class PageRepository:
    """Current project-scoped page metadata keyed by stable page UID."""

    __slots__ = ("_project_uid", "_pages")

    def __init__(self, project_uid: str) -> None:
        self._project_uid = _required_uid(project_uid, "project_uid")
        self._pages: dict[str, PageRecord] = {}

    @property
    def project_uid(self) -> str:
        return self._project_uid

    def put(self, next_page: PageRecord, expected_revision: int) -> PageRecord:
        if not isinstance(next_page, PageRecord):
            raise TypeError("page repository requires PageRecord")
        if next_page.project_uid != self.project_uid:
            raise ProjectScopeError(
                f"record project {next_page.project_uid!r} does not match "
                f"repository project {self.project_uid!r}"
            )
        expected = _non_negative_int(expected_revision, "expected revision")
        current = self._pages.get(next_page.uid)
        if current is None:
            if expected != 0:
                raise RevisionConflictError(
                    "first page record requires expected revision 0"
                )
        elif current.image_revision != expected:
            raise RevisionConflictError(
                f"page image revision mismatch: expected {expected}, "
                f"current {current.image_revision}"
            )
        if next_page.image_revision != expected + 1:
            raise RevisionConflictError(
                "next page image_revision must equal expected revision plus one"
            )
        self._pages[next_page.uid] = next_page
        return next_page

    def get(
        self,
        uid: str,
        *,
        image_revision: int | None = None,
        fingerprint: str | None = None,
    ) -> PageRecord:
        stable_uid = _required_uid(uid, "uid")
        try:
            page = self._pages[stable_uid]
        except KeyError as exc:
            raise RecordNotFoundError(f"page UID is not present: {stable_uid!r}") from exc
        if image_revision is not None:
            _non_negative_int(image_revision, "expected image revision")
            if page.image_revision != image_revision:
                raise RevisionConflictError(
                    f"image revision mismatch: expected {image_revision}, "
                    f"current {page.image_revision}"
                )
        if fingerprint is not None:
            expected = _required_text(fingerprint, "expected fingerprint")
            if page.fingerprint != expected:
                raise FingerprintConflictError("fingerprint mismatch for page record")
        return page

    def all(self) -> tuple[PageRecord, ...]:
        return tuple(self._pages[uid] for uid in sorted(self._pages))

    def _restore(self, page: PageRecord) -> None:
        """Hydrate one persisted current page; only ProjectSession may call this."""
        if not isinstance(page, PageRecord):
            raise TypeError("page repository requires PageRecord")
        if page.project_uid != self.project_uid:
            raise ProjectScopeError("persisted page belongs to another project")
        if page.uid in self._pages:
            raise DuplicateUidError(f"page UID already exists: {page.uid!r}")
        if page.image_revision < 1:
            raise InvalidRecordError("persisted page image_revision must be positive")
        self._pages[page.uid] = page

    def __len__(self) -> int:
        return len(self._pages)


class _RevisionedStore:
    """Private revisioned storage for non-OCR records."""

    __slots__ = ("_project_uid", "_record_type", "_records")

    def __init__(self, project_uid: str, record_type: type[object]) -> None:
        self._project_uid = _required_uid(project_uid, "project_uid")
        self._record_type = record_type
        self._records: dict[str, object] = {}

    def create(self, record: object) -> object:
        self._validate(record)
        uid = cast(str, getattr(record, "uid"))
        if uid in self._records:
            raise DuplicateUidError(f"UID already exists: {uid!r}")
        if getattr(record, "revision") != 0:
            raise InvalidRecordError("new records must have revision 0")
        stored = dc_replace(record, revision=1)
        self._records[uid] = stored
        return stored

    def get(
        self,
        uid: str,
        *,
        revision: int | None = None,
        fingerprint: str | None = None,
    ) -> object:
        stable_uid = _required_uid(uid, "uid")
        try:
            record = self._records[stable_uid]
        except KeyError as exc:
            raise RecordNotFoundError(f"record UID is not present: {stable_uid!r}") from exc
        _check_snapshot(record, revision=revision, fingerprint=fingerprint)
        return record

    def replace(
        self,
        record: object,
        *,
        expected_revision: int,
        expected_fingerprint: str | None = None,
    ) -> object:
        self._validate(record)
        uid = cast(str, getattr(record, "uid"))
        current = self.get(uid)
        _check_snapshot(
            current,
            revision=expected_revision,
            fingerprint=expected_fingerprint,
        )
        if getattr(record, "revision") != expected_revision:
            raise RevisionConflictError(
                "replacement record revision does not match expected_revision"
            )
        updated = dc_replace(record, revision=getattr(current, "revision") + 1)
        self._records[uid] = updated
        return updated

    def all(self) -> tuple[object, ...]:
        return tuple(self._records[uid] for uid in sorted(self._records))

    def restore(self, record: object) -> object:
        """Hydrate one persisted current record without replaying its history."""
        self._validate(record)
        uid = cast(str, getattr(record, "uid"))
        if uid in self._records:
            raise DuplicateUidError(f"UID already exists: {uid!r}")
        if getattr(record, "revision") < 1:
            raise InvalidRecordError("persisted record revision must be positive")
        self._records[uid] = record
        return record

    def _validate(self, record: object) -> None:
        if not isinstance(record, self._record_type):
            raise TypeError(
                f"repository requires {self._record_type.__name__}, "
                f"got {type(record).__name__}"
            )
        if getattr(record, "project_uid") != self._project_uid:
            raise ProjectScopeError(
                f"record project {getattr(record, 'project_uid')!r} does not match "
                f"repository project {self._project_uid!r}"
            )


@dataclass(frozen=True, slots=True)
class BindingRecord:
    """An explicit relationship between two stable project UIDs."""

    project_uid: str
    uid: str
    source_uid: str
    target_uid: str
    relation: str
    source_fingerprint: str
    target_fingerprint: str
    attributes: tuple[tuple[str, str], ...] = ()
    revision: int = 0
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _required_uid(self.source_uid, "source_uid")
        _required_uid(self.target_uid, "target_uid")
        _required_text(self.relation, "relation")
        _required_text(self.source_fingerprint, "source_fingerprint")
        _required_text(self.target_fingerprint, "target_fingerprint")
        object.__setattr__(self, "attributes", _metadata(self.attributes))
        object.__setattr__(self, "revision", _non_negative_int(self.revision, "revision"))
        _set_record_fingerprint(self)


@dataclass(frozen=True, slots=True)
class TableTextRecord:
    """One text cell in the project table-text layer."""

    project_uid: str
    uid: str
    page_uid: str
    table_uid: str
    row_index: int
    column_index: int
    text: str
    source_fingerprint: str
    revision: int = 0
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _required_uid(self.project_uid, "project_uid")
        _required_uid(self.uid, "uid")
        _required_uid(self.page_uid, "page_uid")
        _required_uid(self.table_uid, "table_uid")
        object.__setattr__(self, "row_index", _non_negative_int(self.row_index, "row_index"))
        object.__setattr__(self, "column_index", _non_negative_int(self.column_index, "column_index"))
        _text(self.text, "text")
        _required_text(self.source_fingerprint, "source_fingerprint")
        object.__setattr__(self, "revision", _non_negative_int(self.revision, "revision"))
        _set_record_fingerprint(self)


class BindingRepository:
    def __init__(self, project_uid: str) -> None:
        self._store = _RevisionedStore(project_uid, BindingRecord)

    @property
    def project_uid(self) -> str:
        return self._store._project_uid

    def create(self, record: BindingRecord) -> BindingRecord:
        return cast(BindingRecord, self._store.create(record))

    def get(
        self,
        uid: str,
        *,
        revision: int | None = None,
        fingerprint: str | None = None,
    ) -> BindingRecord:
        return cast(BindingRecord, self._store.get(uid, revision=revision, fingerprint=fingerprint))

    def replace(
        self,
        record: BindingRecord,
        *,
        expected_revision: int,
        expected_fingerprint: str | None = None,
    ) -> BindingRecord:
        return cast(
            BindingRecord,
            self._store.replace(
                record,
                expected_revision=expected_revision,
                expected_fingerprint=expected_fingerprint,
            ),
        )

    def all(self) -> tuple[BindingRecord, ...]:
        return tuple(cast(BindingRecord, item) for item in self._store.all())


class TableTextRepository:
    def __init__(self, project_uid: str) -> None:
        self._store = _RevisionedStore(project_uid, TableTextRecord)

    @property
    def project_uid(self) -> str:
        return self._store._project_uid

    def create(self, record: TableTextRecord) -> TableTextRecord:
        return cast(TableTextRecord, self._store.create(record))

    def get(
        self,
        uid: str,
        *,
        revision: int | None = None,
        fingerprint: str | None = None,
    ) -> TableTextRecord:
        return cast(TableTextRecord, self._store.get(uid, revision=revision, fingerprint=fingerprint))

    def replace(
        self,
        record: TableTextRecord,
        *,
        expected_revision: int,
        expected_fingerprint: str | None = None,
    ) -> TableTextRecord:
        return cast(
            TableTextRecord,
            self._store.replace(
                record,
                expected_revision=expected_revision,
                expected_fingerprint=expected_fingerprint,
            ),
        )

    def all(self) -> tuple[TableTextRecord, ...]:
        return tuple(cast(TableTextRecord, item) for item in self._store.all())


class OcrObservationRepository:
    """Append-only OCR facts plus one CAS-managed active pointer per scope."""

    __slots__ = (
        "_project_uid",
        "_runs",
        "_regions",
        "_lines",
        "_atoms",
        "_candidates",
        "_batches",
        "_active",
        "_active_uid_by_scope",
    )

    def __init__(self, project_uid: str) -> None:
        self._project_uid = _required_uid(project_uid, "project_uid")
        self._runs: dict[str, OcrRun] = {}
        self._regions: dict[str, OcrRegion] = {}
        self._lines: dict[str, OcrLine] = {}
        self._atoms: dict[str, OcrAtom] = {}
        self._candidates: dict[str, OcrCandidate] = {}
        self._batches: dict[str, OcrBatch] = {}
        self._active: dict[str, OcrActivePointer] = {}
        self._active_uid_by_scope: dict[str, str] = {}

    @property
    def project_uid(self) -> str:
        return self._project_uid

    def append(self, record: object) -> object:
        if isinstance(record, OcrRun):
            return self.append_run(record)
        if isinstance(record, OcrRegion):
            return self.append_region(record)
        if isinstance(record, OcrLine):
            return self.append_line(record)
        if isinstance(record, OcrAtom):
            return self.append_atom(record)
        if isinstance(record, OcrCandidate):
            return self.append_candidate(record)
        if isinstance(record, OcrBatch):
            return self.append_batch(record)
        raise TypeError(
            "OCR append accepts OcrRun, OcrRegion, OcrLine, OcrAtom, "
            "OcrCandidate, or OcrBatch"
        )

    def append_run(self, record: OcrRun) -> OcrRun:
        return self._append(record, self._runs, OcrRun)

    def append_region(self, record: OcrRegion) -> OcrRegion:
        return self._append(record, self._regions, OcrRegion)

    def append_line(self, record: OcrLine) -> OcrLine:
        return self._append(record, self._lines, OcrLine)

    def append_atom(self, record: OcrAtom) -> OcrAtom:
        return self._append(record, self._atoms, OcrAtom)

    def append_candidate(self, record: OcrCandidate) -> OcrCandidate:
        return self._append(record, self._candidates, OcrCandidate)

    def append_batch(self, record: OcrBatch) -> OcrBatch:
        return self._append(record, self._batches, OcrBatch)

    def append_observation_batch(
        self,
        *,
        run: OcrRun,
        regions: tuple[OcrRegion, ...],
        lines: tuple[OcrLine, ...],
        atoms: tuple[OcrAtom, ...],
        candidates: tuple[OcrCandidate, ...],
        batch: OcrBatch,
    ) -> OcrBatch:
        """Validate and append one complete observation unit atomically in memory."""
        typed_groups = (
            (run, OcrRun, self._runs),
            *tuple((item, OcrRegion, self._regions) for item in regions),
            *tuple((item, OcrLine, self._lines) for item in lines),
            *tuple((item, OcrAtom, self._atoms) for item in atoms),
            *tuple((item, OcrCandidate, self._candidates) for item in candidates),
            (batch, OcrBatch, self._batches),
        )
        all_uids: list[str] = []
        for record, record_type, existing in typed_groups:
            if not isinstance(record, record_type):
                raise TypeError(f"OCR batch contains a non-{record_type.__name__} record")
            self._validate_project(record)
            uid = cast(str, getattr(record, "uid"))
            all_uids.append(uid)
            if uid in existing:
                raise DuplicateUidError(f"UID already exists: {uid!r}")
        if len(set(all_uids)) != len(all_uids):
            raise DuplicateUidError("OCR observation batch contains duplicate UIDs")
        if batch.run_uid != run.uid:
            raise InvalidRecordError("OCR batch references another run")
        region_uids = {item.uid for item in regions}
        line_uids = {item.uid for item in lines}
        atom_uids = {item.uid for item in atoms}
        candidate_uids = {item.uid for item in candidates}
        if set(batch.region_uids) != region_uids or set(batch.line_uids) != line_uids:
            raise InvalidRecordError("OCR batch region/line membership is incomplete")
        if set(batch.atom_uids) != atom_uids or set(batch.candidate_uids) != candidate_uids:
            raise InvalidRecordError("OCR batch atom/candidate membership is incomplete")
        if any(item.run_uid != run.uid or item.page_uid != batch.scope_uid for item in regions):
            raise InvalidRecordError("OCR region scope does not match its batch")
        if any(
            item.run_uid != run.uid
            or item.page_uid != batch.scope_uid
            or item.region_uid not in region_uids
            for item in lines
        ):
            raise InvalidRecordError("OCR line references an unknown region or scope")
        if any(
            item.run_uid != run.uid
            or item.region_uid not in region_uids
            or item.line_uid not in line_uids
            for item in atoms
        ):
            raise InvalidRecordError("OCR atom references an unknown line or region")
        if any(
            item.run_uid != run.uid
            or item.batch_uid != batch.uid
            or item.region_uid not in region_uids
            or item.line_uid not in line_uids
            or item.atom_uid not in atom_uids
            for item in candidates
        ):
            raise InvalidRecordError("OCR candidate references an unknown observation")
        if any(set(item.atom_uids) - atom_uids for item in lines):
            raise InvalidRecordError("OCR line lists an unknown atom")
        if any(set(item.candidate_uids) - candidate_uids for item in atoms):
            raise InvalidRecordError("OCR atom lists an unknown candidate")

        self._runs[run.uid] = run
        self._regions.update({item.uid: item for item in regions})
        self._lines.update({item.uid: item for item in lines})
        self._atoms.update({item.uid: item for item in atoms})
        self._candidates.update({item.uid: item for item in candidates})
        self._batches[batch.uid] = batch
        return batch

    def get_run(self, uid: str, *, fingerprint: str | None = None) -> OcrRun:
        return self._get(self._runs, uid, fingerprint=fingerprint)

    def get_region(self, uid: str, *, fingerprint: str | None = None) -> OcrRegion:
        return self._get(self._regions, uid, fingerprint=fingerprint)

    def get_line(self, uid: str, *, fingerprint: str | None = None) -> OcrLine:
        return self._get(self._lines, uid, fingerprint=fingerprint)

    def get_atom(self, uid: str, *, fingerprint: str | None = None) -> OcrAtom:
        return self._get(self._atoms, uid, fingerprint=fingerprint)

    def get_candidate(self, uid: str, *, fingerprint: str | None = None) -> OcrCandidate:
        return self._get(self._candidates, uid, fingerprint=fingerprint)

    def get_batch(self, uid: str, *, fingerprint: str | None = None) -> OcrBatch:
        return self._get(self._batches, uid, fingerprint=fingerprint)

    def switch_active_pointer(
        self,
        next_pointer: OcrActivePointer,
        *,
        expected_revision: int,
        expected_fingerprint: str | None,
    ) -> OcrActivePointer:
        self._validate_active_pointer(next_pointer)
        expected = _non_negative_int(expected_revision, "expected revision")
        current_uid = self._active_uid_by_scope.get(next_pointer.scope_uid)
        if current_uid is None:
            if expected != 0:
                raise RevisionConflictError(
                    "first active pointer requires expected revision 0"
                )
            if expected_fingerprint is not None:
                raise FingerprintConflictError(
                    "first active pointer has no expected fingerprint"
                )
            if next_pointer.revision != 1:
                raise RevisionConflictError(
                    "first active pointer revision must be 1"
                )
            if next_pointer.uid in self._active:
                raise DuplicateUidError(f"UID already exists: {next_pointer.uid!r}")
        else:
            current = self._active[current_uid]
            if current.uid != next_pointer.uid:
                raise InvalidRecordError(
                    "active pointer UID cannot change during a scope switch"
                )
            _check_snapshot(
                current,
                revision=expected,
                fingerprint=expected_fingerprint,
            )
            if next_pointer.revision != current.revision + 1:
                raise RevisionConflictError(
                    "next active pointer revision must equal expected revision plus one"
                )
        self._active[next_pointer.uid] = next_pointer
        self._active_uid_by_scope[next_pointer.scope_uid] = next_pointer.uid
        return next_pointer

    def get_active_pointer(
        self,
        scope_uid: str,
        *,
        revision: int | None = None,
        fingerprint: str | None = None,
    ) -> OcrActivePointer:
        stable_scope_uid = _required_uid(scope_uid, "scope_uid")
        try:
            pointer_uid = self._active_uid_by_scope[stable_scope_uid]
        except KeyError as exc:
            raise RecordNotFoundError(
                f"active pointer is not present for scope {stable_scope_uid!r}"
            ) from exc
        return self._get(self._active, pointer_uid, revision=revision, fingerprint=fingerprint)

    def get_active_pointer_by_uid(
        self,
        uid: str,
        *,
        revision: int | None = None,
        fingerprint: str | None = None,
    ) -> OcrActivePointer:
        return self._get(self._active, uid, revision=revision, fingerprint=fingerprint)

    def all_runs(self) -> tuple[OcrRun, ...]:
        return tuple(self._runs[uid] for uid in sorted(self._runs))

    def all_regions(self) -> tuple[OcrRegion, ...]:
        return tuple(self._regions[uid] for uid in sorted(self._regions))

    def all_lines(self) -> tuple[OcrLine, ...]:
        return tuple(self._lines[uid] for uid in sorted(self._lines))

    def all_atoms(self) -> tuple[OcrAtom, ...]:
        return tuple(self._atoms[uid] for uid in sorted(self._atoms))

    def all_candidates(self) -> tuple[OcrCandidate, ...]:
        return tuple(self._candidates[uid] for uid in sorted(self._candidates))

    def all_batches(self) -> tuple[OcrBatch, ...]:
        return tuple(self._batches[uid] for uid in sorted(self._batches))

    def all_active_pointers(self) -> tuple[OcrActivePointer, ...]:
        return tuple(
            self._active[self._active_uid_by_scope[scope_uid]]
            for scope_uid in sorted(self._active_uid_by_scope)
        )

    def _restore_active_pointer(self, pointer: OcrActivePointer) -> None:
        """Hydrate a persisted current pointer; only ProjectSession may call this."""
        self._validate_active_pointer(pointer)
        if pointer.uid in self._active or pointer.scope_uid in self._active_uid_by_scope:
            raise DuplicateUidError("persisted active OCR pointer is duplicated")
        if pointer.revision < 1:
            raise InvalidRecordError("persisted active pointer revision must be positive")
        self._active[pointer.uid] = pointer
        self._active_uid_by_scope[pointer.scope_uid] = pointer.uid

    def _append(self, record: object, records: dict[str, object], record_type: type[object]) -> object:
        if not isinstance(record, record_type):
            raise TypeError(
                f"OCR append requires {record_type.__name__}, got {type(record).__name__}"
            )
        self._validate_project(record)
        uid = cast(str, getattr(record, "uid"))
        if uid in records:
            raise DuplicateUidError(f"UID already exists: {uid!r}")
        records[uid] = record
        return record

    def _get(
        self,
        records: dict[str, object],
        uid: str,
        *,
        revision: int | None = None,
        fingerprint: str | None = None,
    ) -> object:
        stable_uid = _required_uid(uid, "uid")
        try:
            record = records[stable_uid]
        except KeyError as exc:
            raise RecordNotFoundError(f"record UID is not present: {stable_uid!r}") from exc
        _check_snapshot(record, revision=revision, fingerprint=fingerprint)
        return record

    def _validate_project(self, record: object) -> None:
        if getattr(record, "project_uid") != self.project_uid:
            raise ProjectScopeError(
                f"record project {getattr(record, 'project_uid')!r} does not match "
                f"repository project {self.project_uid!r}"
            )

    def _validate_active_pointer(self, pointer: OcrActivePointer) -> None:
        if not isinstance(pointer, OcrActivePointer):
            raise TypeError("active pointer requires OcrActivePointer")
        self._validate_project(pointer)
        batch = self.get_batch(pointer.batch_uid)
        run = self.get_run(pointer.run_uid)
        if batch.run_uid != pointer.run_uid:
            raise InvalidRecordError("active pointer run does not own its batch")
        if batch.scope_uid != pointer.scope_uid:
            raise InvalidRecordError("active pointer scope does not match its batch")
        if batch.layout_fingerprint != run.layout_fingerprint:
            raise InvalidRecordError("active batch layout fingerprint does not match its run")
        if batch.input_fingerprint != run.input_fingerprint:
            raise InvalidRecordError("active batch input fingerprint does not match its run")
        if pointer.batch_fingerprint != batch.fingerprint:
            raise FingerprintConflictError("active pointer batch fingerprint is stale")


class ProofRepository:
    """Project-scoped proof state with an explicit rebind transition."""

    __slots__ = ("_project_uid", "_states")

    def __init__(self, project_uid: str) -> None:
        self._project_uid = _required_uid(project_uid, "project_uid")
        self._states = _RevisionedStore(self.project_uid, ProofState)

    @property
    def project_uid(self) -> str:
        return self._project_uid

    def create(self, record: ProofState) -> ProofState:
        return self.create_state(record)

    def replace(
        self,
        record: ProofState,
        *,
        expected_revision: int,
        expected_fingerprint: str | None = None,
    ) -> ProofState:
        return self.replace_state(
            record,
            expected_revision=expected_revision,
            expected_fingerprint=expected_fingerprint,
        )

    def create_state(self, record: ProofState) -> ProofState:
        self._validate_state(record)
        return cast(ProofState, self._states.create(record))

    def get_state(
        self,
        uid: str,
        *,
        revision: int | None = None,
        fingerprint: str | None = None,
    ) -> ProofState:
        return cast(ProofState, self._states.get(uid, revision=revision, fingerprint=fingerprint))

    def replace_state(
        self,
        record: ProofState,
        *,
        expected_revision: int,
        expected_fingerprint: str | None = None,
    ) -> ProofState:
        self._validate_state(record)
        return cast(
            ProofState,
            self._states.replace(
                record,
                expected_revision=expected_revision,
                expected_fingerprint=expected_fingerprint,
            ),
        )

    def mark_rebind_required(
        self,
        uid: str,
        anchor_snapshot: ProofAnchorSnapshot,
        *,
        expected_revision: int,
        expected_fingerprint: str | None = None,
    ) -> ProofState:
        current = self.get_state(uid)
        updated = dc_replace(
            current,
            anchor_snapshot=anchor_snapshot,
            alignment_segments=(),
            alignment_slices=(),
            rebind_required=True,
        )
        return self.replace_state(
            updated,
            expected_revision=expected_revision,
            expected_fingerprint=expected_fingerprint,
        )

    def rebind(
        self,
        uid: str,
        anchor_snapshot: ProofAnchorSnapshot,
        alignment_segments: tuple[ProofAlignmentSegment, ...],
        alignment_slices: tuple[ProofAlignmentSlice, ...],
        *,
        expected_revision: int,
        expected_fingerprint: str | None = None,
    ) -> ProofState:
        current = self.get_state(uid)
        if not current.rebind_required:
            raise InvalidRecordError("proof state does not require rebind")
        updated = dc_replace(
            current,
            anchor_snapshot=anchor_snapshot,
            alignment_segments=tuple(alignment_segments),
            alignment_slices=tuple(alignment_slices),
            rebind_required=False,
        )
        return self.replace_state(
            updated,
            expected_revision=expected_revision,
            expected_fingerprint=expected_fingerprint,
        )

    def all_states(self) -> tuple[ProofState, ...]:
        return tuple(cast(ProofState, item) for item in self._states.all())

    def _validate_state(self, record: ProofState) -> None:
        if not isinstance(record, ProofState):
            raise TypeError("proof repository requires ProofState")
        if record.project_uid != self.project_uid:
            raise ProjectScopeError(
                f"record project {record.project_uid!r} does not match "
                f"repository project {self.project_uid!r}"
            )
        if record.anchor_snapshot.project_uid != self.project_uid:
            raise ProjectScopeError("proof anchor belongs to another project")
        for segment in record.alignment_segments:
            if segment.project_uid != self.project_uid:
                raise ProjectScopeError("proof alignment segment belongs to another project")
        for item in record.alignment_slices:
            if item.project_uid != self.project_uid:
                raise ProjectScopeError("proof alignment slice belongs to another project")
        for item in record.text_units:
            if item.project_uid != self.project_uid:
                raise ProjectScopeError("proof text unit belongs to another project")


class ProjectSession:
    """Own all repositories for exactly one stable project UID."""

    __slots__ = (
        "_project_uid",
        "_project_record",
        "_save_path",
        "_layout_repository",
        "_paddle_artifact_repository",
        "_page_repository",
        "_ocr_observation_repository",
        "_proof_repository",
        "_binding_repository",
        "_table_text_repository",
    )

    def __init__(
        self,
        project_uid: str | ProjectRecord,
        *,
        project_record: ProjectRecord | None = None,
        save_path: str | None = None,
    ) -> None:
        if isinstance(project_uid, ProjectRecord):
            if project_record is not None:
                raise TypeError("project_record was supplied twice")
            record = project_uid
        else:
            stable_project_uid = _required_uid(project_uid, "project_uid")
            if project_record is None:
                record = ProjectRecord(project_uid=stable_project_uid)
            else:
                if not isinstance(project_record, ProjectRecord):
                    raise TypeError("project_record requires ProjectRecord")
                if project_record.project_uid != stable_project_uid:
                    raise ProjectScopeError(
                        f"record project {project_record.project_uid!r} does not match "
                        f"session project {stable_project_uid!r}"
                    )
                record = project_record
        self._project_record = record
        self._project_uid = record.project_uid
        self._save_path = _optional_path(save_path, "save_path")
        self._layout_repository = LayoutRepository(self.project_uid)
        self._paddle_artifact_repository = PaddleArtifactRepository(self.project_uid)
        self._page_repository = PageRepository(self.project_uid)
        self._ocr_observation_repository = OcrObservationRepository(self.project_uid)
        self._proof_repository = ProofRepository(self.project_uid)
        self._binding_repository = BindingRepository(self.project_uid)
        self._table_text_repository = TableTextRepository(self.project_uid)

    @property
    def project_uid(self) -> str:
        return self._project_uid

    @property
    def project_record(self) -> ProjectRecord:
        return self._project_record

    @property
    def save_path(self) -> str | None:
        return self._save_path

    @save_path.setter
    def save_path(self, value: str | None) -> None:
        self._save_path = _optional_path(value, "save_path")

    @property
    def layout_repository(self) -> LayoutRepository:
        return self._layout_repository

    @property
    def page_repository(self) -> PageRepository:
        return self._page_repository

    @property
    def paddle_artifact_repository(self) -> PaddleArtifactRepository:
        return self._paddle_artifact_repository

    @property
    def ocr_observation_repository(self) -> OcrObservationRepository:
        return self._ocr_observation_repository

    @property
    def proof_repository(self) -> ProofRepository:
        return self._proof_repository

    @property
    def binding_repository(self) -> BindingRepository:
        return self._binding_repository

    @property
    def table_text_repository(self) -> TableTextRepository:
        return self._table_text_repository

    def adopt_ocr_page_observation(
        self,
        *,
        run: OcrRun,
        regions: tuple[OcrRegion, ...],
        lines: tuple[OcrLine, ...],
        atoms: tuple[OcrAtom, ...],
        candidates: tuple[OcrCandidate, ...],
        batch: OcrBatch,
        pointer: OcrActivePointer,
        expected_pointer_revision: int,
        expected_pointer_fingerprint: str | None,
        bindings: tuple[BindingRecord, ...],
        proof_states: tuple[ProofState, ...],
    ) -> None:
        """Adopt one page OCR result as one rollback-safe aggregate mutation."""
        ocr = self.ocr_observation_repository
        binding_store = self.binding_repository._store
        proof_store = self.proof_repository._states
        new_ocr_uids = {
            "run": (run.uid,),
            "region": tuple(item.uid for item in regions),
            "line": tuple(item.uid for item in lines),
            "atom": tuple(item.uid for item in atoms),
            "candidate": tuple(item.uid for item in candidates),
            "batch": (batch.uid,),
        }
        previous_pointer_uid = ocr._active_uid_by_scope.get(pointer.scope_uid)
        previous_pointer = (
            ocr._active.get(previous_pointer_uid) if previous_pointer_uid is not None else None
        )
        previous_bindings = {
            item.uid: binding_store._records.get(item.uid) for item in bindings
        }
        previous_proof = {
            item.uid: proof_store._records.get(item.uid) for item in proof_states
        }
        try:
            ocr.append_observation_batch(
                run=run,
                regions=regions,
                lines=lines,
                atoms=atoms,
                candidates=candidates,
                batch=batch,
            )
            ocr.switch_active_pointer(
                pointer,
                expected_revision=expected_pointer_revision,
                expected_fingerprint=expected_pointer_fingerprint,
            )
            for binding in bindings:
                current = previous_bindings[binding.uid]
                if current is None:
                    self.binding_repository.create(binding)
                else:
                    current_binding = cast(BindingRecord, current)
                    self.binding_repository.replace(
                        binding,
                        expected_revision=current_binding.revision,
                        expected_fingerprint=current_binding.fingerprint,
                    )
            for proof_state in proof_states:
                current = previous_proof[proof_state.uid]
                if current is None:
                    raise RecordNotFoundError(
                        f"proof state UID is not present: {proof_state.uid!r}"
                    )
                current_state = cast(ProofState, current)
                self.proof_repository.replace_state(
                    proof_state,
                    expected_revision=current_state.revision,
                    expected_fingerprint=current_state.fingerprint,
                )
        except Exception:
            for uid in new_ocr_uids["run"]:
                ocr._runs.pop(uid, None)
            for uid in new_ocr_uids["region"]:
                ocr._regions.pop(uid, None)
            for uid in new_ocr_uids["line"]:
                ocr._lines.pop(uid, None)
            for uid in new_ocr_uids["atom"]:
                ocr._atoms.pop(uid, None)
            for uid in new_ocr_uids["candidate"]:
                ocr._candidates.pop(uid, None)
            for uid in new_ocr_uids["batch"]:
                ocr._batches.pop(uid, None)
            if previous_pointer_uid is None:
                ocr._active.pop(pointer.uid, None)
                ocr._active_uid_by_scope.pop(pointer.scope_uid, None)
            else:
                ocr._active_uid_by_scope[pointer.scope_uid] = previous_pointer_uid
                if previous_pointer is not None:
                    ocr._active[previous_pointer_uid] = previous_pointer
            for uid, previous in previous_bindings.items():
                if previous is None:
                    binding_store._records.pop(uid, None)
                else:
                    binding_store._records[uid] = previous
            for uid, previous in previous_proof.items():
                if previous is not None:
                    proof_store._records[uid] = previous
            raise

    @classmethod
    def from_records(
        cls,
        project_record: ProjectRecord,
        *,
        pages: tuple[PageRecord, ...] = (),
        paddle_artifacts: tuple[PaddleArtifact, ...] = (),
        layouts: tuple[LayoutSnapshot, ...] = (),
        ocr_runs: tuple[OcrRun, ...] = (),
        ocr_regions: tuple[OcrRegion, ...] = (),
        ocr_lines: tuple[OcrLine, ...] = (),
        ocr_atoms: tuple[OcrAtom, ...] = (),
        ocr_candidates: tuple[OcrCandidate, ...] = (),
        ocr_batches: tuple[OcrBatch, ...] = (),
        ocr_active_pointers: tuple[OcrActivePointer, ...] = (),
        proof_states: tuple[ProofState, ...] = (),
        bindings: tuple[BindingRecord, ...] = (),
        table_texts: tuple[TableTextRecord, ...] = (),
        save_path: str | None = None,
    ) -> "ProjectSession":
        """Hydrate a validated session from one persistence snapshot.

        This is the only path that may restore current revisions without
        replaying business mutations. It deliberately accepts no legacy model.
        """
        session = cls(project_record, save_path=save_path)
        for page in pages:
            session.page_repository._restore(page)
        page_uids = {page.uid for page in session.page_repository.all()}
        for artifact in paddle_artifacts:
            if artifact.page_uid not in page_uids:
                raise InvalidRecordError("Paddle artifact references an unknown persisted page")
            session.paddle_artifact_repository.append(artifact)
        artifact_uids = {artifact.uid for artifact in session.paddle_artifact_repository.all()}
        for snapshot in layouts:
            if snapshot.page_uid not in page_uids:
                raise InvalidRecordError("layout references an unknown persisted page")
            if snapshot.artifact_uid and snapshot.artifact_uid not in artifact_uids:
                raise InvalidRecordError("layout references an unknown Paddle artifact")
            session.layout_repository._restore(snapshot)

        ocr = session.ocr_observation_repository
        for record in ocr_runs:
            ocr.append_run(record)
        for record in ocr_regions:
            ocr.append_region(record)
        for record in ocr_lines:
            ocr.append_line(record)
        for record in ocr_atoms:
            ocr.append_atom(record)
        for record in ocr_batches:
            ocr.append_batch(record)
        for record in ocr_candidates:
            ocr.append_candidate(record)
        for pointer in ocr_active_pointers:
            ocr._restore_active_pointer(pointer)

        proof = session.proof_repository
        for record in proof_states:
            proof._validate_state(record)
            proof._states.restore(record)
        for record in bindings:
            session.binding_repository._store.restore(record)
        for record in table_texts:
            session.table_text_repository._store.restore(record)
        return session


__all__ = [
    "BindingRecord",
    "BindingRepository",
    "DuplicateUidError",
    "FingerprintConflictError",
    "InvalidRecordError",
    "LayoutRepository",
    "LayoutSnapshot",
    "OcrObservationRepository",
    "PageRecord",
    "PageRepository",
    "PaddleArtifactRepository",
    "ProjectScopeError",
    "ProjectRecord",
    "ProjectSession",
    "ProofRepository",
    "RecordNotFoundError",
    "RepositoryConflictError",
    "RepositoryError",
    "RevisionConflictError",
    "TableTextRecord",
    "TableTextRepository",
]
