"""Project-scoped Paddle layout use cases."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Protocol

from app.core.layout_analyzer import LayoutAnalyzer
from app.models.entity_id import new_ulid
from app.models.layout_snapshot import LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.project_session import (
    ProjectSession,
    RecordNotFoundError,
    RevisionConflictError,
)


class PaddleLayoutClient(Protocol):
    """The narrow vendor boundary required by the layout use case."""

    def analyze_image_bytes(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        page_uid: str,
        source_run_id: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class LayoutAnalysisCommit:
    page_uid: str
    artifact: PaddleArtifact
    snapshot: LayoutSnapshot

    @property
    def artifact_uid(self) -> str:
        return self.artifact.uid

    @property
    def layout_revision(self) -> int:
        return self.snapshot.revision


class LayoutAnalysisService:
    """Acquire Paddle facts and adopt them through the session repositories."""

    def __init__(
        self,
        client: PaddleLayoutClient,
        *,
        analyzer: LayoutAnalyzer | None = None,
        source_engine: str = "paddleocr-vl-1.6",
    ) -> None:
        self._client = client
        self._analyzer = analyzer if analyzer is not None else LayoutAnalyzer()
        if not isinstance(source_engine, str) or not source_engine.strip():
            raise ValueError("source_engine must be non-empty")
        self._source_engine = source_engine

    def analyze_page(
        self,
        session: ProjectSession,
        page_uid: str,
        *,
        expected_revision: int = 0,
        expected_fingerprint: str | None = None,
        source_run_id: str | None = None,
    ) -> LayoutAnalysisCommit:
        """Run Paddle for one page and adopt the next layout revision via CAS."""
        page = session.page_repository.get(page_uid)
        self._check_layout_cas(
            session,
            page_uid,
            expected_revision=expected_revision,
            expected_fingerprint=expected_fingerprint,
        )
        image_path = Path(page.cache_image_path or page.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"page image is missing: {image_path}")
        image_bytes = image_path.read_bytes()
        if not image_bytes:
            raise ValueError(f"page image is empty: {image_path}")

        run_id = source_run_id or f"layout_{new_ulid()}"
        response = self._client.analyze_image_bytes(
            image_bytes,
            filename=image_path.name or "page.png",
            page_uid=page.uid,
            source_run_id=run_id,
        )
        if not isinstance(response, Mapping):
            raise TypeError("Paddle layout client must return a response object")

        artifact = self._append_artifact(
            session,
            page_uid=page.uid,
            image_hash=page.image_hash,
            source_run_id=run_id,
            source_engine=self._source_engine,
            response=response,
        )
        snapshot = self._analyzer.analyze(
            response,
            page_uid=page.uid,
            page_width=page.width,
            page_height=page.height,
            artifact_uid=artifact.uid,
            source_run_id=artifact.source_run_id,
            source_engine=artifact.source_engine,
            revision=expected_revision + 1,
        )
        adopted = self.adopt_snapshot(
            session,
            snapshot,
            expected_revision=expected_revision,
            expected_fingerprint=expected_fingerprint,
        )
        return LayoutAnalysisCommit(page_uid=page.uid, artifact=artifact, snapshot=adopted)

    def adopt_snapshot(
        self,
        session: ProjectSession,
        snapshot: LayoutSnapshot,
        *,
        expected_revision: int,
        expected_fingerprint: str | None = None,
    ) -> LayoutSnapshot:
        """CAS-adopt an edited or otherwise prepared snapshot."""
        if not isinstance(snapshot, LayoutSnapshot):
            raise TypeError("layout adoption requires a LayoutSnapshot")
        page = session.page_repository.get(snapshot.page_uid)
        self._check_layout_cas(
            session,
            page.uid,
            expected_revision=expected_revision,
            expected_fingerprint=expected_fingerprint,
        )
        if snapshot.revision != expected_revision + 1:
            raise RevisionConflictError(
                "adopted layout revision must equal expected revision plus one"
            )
        if not snapshot.artifact_uid:
            raise ValueError("adopted layout must retain a Paddle artifact reference")
        artifact = session.paddle_artifact_repository.get(snapshot.artifact_uid)
        if artifact.page_uid != page.uid:
            raise ValueError("adopted Paddle artifact belongs to another page")
        if artifact.image_hash != page.image_hash:
            raise ValueError("adopted Paddle artifact belongs to another page image revision")
        return session.layout_repository.put(snapshot, expected_revision=expected_revision)

    def adopt_artifact(
        self,
        session: ProjectSession,
        page_uid: str,
        artifact_uid: str,
        *,
        expected_revision: int,
        expected_fingerprint: str | None = None,
    ) -> LayoutSnapshot:
        """Adopt an existing immutable artifact using the layout CAS contract."""
        page = session.page_repository.get(page_uid)
        self._check_layout_cas(
            session,
            page_uid,
            expected_revision=expected_revision,
            expected_fingerprint=expected_fingerprint,
        )
        artifact = session.paddle_artifact_repository.get(artifact_uid)
        if artifact.page_uid != page.uid:
            raise ValueError("Paddle artifact belongs to another page")
        if artifact.image_hash != page.image_hash:
            raise ValueError("Paddle artifact belongs to another page image revision")
        try:
            response = json.loads(artifact.payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Paddle artifact payload is not valid JSON") from exc
        if not isinstance(response, Mapping):
            raise ValueError("Paddle layout artifact payload must be an object")
        snapshot = self._analyzer.analyze(
            response,
            page_uid=page.uid,
            page_width=page.width,
            page_height=page.height,
            artifact_uid=artifact.uid,
            source_run_id=artifact.source_run_id,
            source_engine=artifact.source_engine,
            revision=expected_revision + 1,
        )
        return self.adopt_snapshot(
            session,
            snapshot,
            expected_revision=expected_revision,
            expected_fingerprint=expected_fingerprint,
        )

    def analyze_pages(
        self,
        session: ProjectSession,
        page_uids: Iterable[str],
        *,
        expected_revisions: Mapping[str, int] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> tuple[LayoutAnalysisCommit, ...]:
        """Analyze a page sequence without introducing a UI worker in core."""
        ordered_page_uids = tuple(page_uids)
        revisions = expected_revisions or {}
        commits: list[LayoutAnalysisCommit] = []
        total = len(ordered_page_uids)
        for index, page_uid in enumerate(ordered_page_uids):
            commit = self.analyze_page(
                session,
                page_uid,
                expected_revision=revisions.get(page_uid, 0),
            )
            commits.append(commit)
            if progress_callback is not None:
                progress_callback(index + 1, total, page_uid)
        return tuple(commits)

    @staticmethod
    def _check_layout_cas(
        session: ProjectSession,
        page_uid: str,
        *,
        expected_revision: int,
        expected_fingerprint: str | None,
    ) -> None:
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        try:
            current = session.layout_repository.get(
                page_uid,
                revision=expected_revision,
                fingerprint=expected_fingerprint,
            )
        except RecordNotFoundError:
            if expected_revision != 0 or expected_fingerprint is not None:
                raise RevisionConflictError(
                    "layout CAS expected an existing snapshot at the supplied revision"
                )
            return
        if current.revision != expected_revision:
            raise RevisionConflictError(
                f"layout revision mismatch: expected {expected_revision}, current {current.revision}"
            )

    @staticmethod
    def _append_artifact(
        session: ProjectSession,
        *,
        page_uid: str,
        image_hash: str,
        source_run_id: str,
        source_engine: str,
        response: Mapping[str, Any],
    ) -> PaddleArtifact:
        try:
            payload_json = json.dumps(
                dict(response),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Paddle response cannot be persisted as JSON") from exc
        artifact = PaddleArtifact(
            project_uid=session.project_uid,
            uid=f"paddle_artifact_{new_ulid()}",
            page_uid=page_uid,
            source_engine=source_engine,
            source_run_id=source_run_id,
            image_hash=image_hash,
            payload_json=payload_json,
        )
        return session.paddle_artifact_repository.append(artifact)


__all__ = [
    "LayoutAnalysisCommit",
    "LayoutAnalysisService",
    "PaddleLayoutClient",
]
