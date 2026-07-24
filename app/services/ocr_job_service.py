"""Project-scoped OCR use case over immutable requests and observations."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from app.core.layout_scope import layout_snapshot_fingerprint
from app.core.ocr_display_orientation import observe_page_display_orientation
from app.core.ppocr_route_compiler import compile_page_routing_plan
from app.models.charocr_execution import CharOcrPageRequest, CharOcrPageResult
from app.models.entity_id import new_ulid
from app.models.layout_snapshot import LayoutSnapshot
from app.models.paddle_artifact import PaddleArtifact
from app.models.ocr_records import (
    OcrActivePointer,
    OcrAtom,
    OcrBatch,
    OcrCandidate,
    OcrLine,
    OcrRegion,
    OcrRun,
)
from app.models.proof_records import (
    ProofAlignmentSegment,
    ProofAlignmentSlice,
    ProofAnchorSnapshot,
    ProofState,
    ProofTextUnit,
)
from app.models.project_session import BindingRecord, PageRecord, ProjectSession, RecordNotFoundError
from app.services.charocr_input_service import compile_charocr_page_request
from app.services.inline_formula_layout_guard import (
    require_current_automatic_inline_formula_layout,
)
from app.services.ocr_routing_observation_service import acquire_routing_observation_bundle


class CharOcrEngine(Protocol):
    engine_id: str

    def recognize_page(
        self,
        image_bgr: np.ndarray,
        request: CharOcrPageRequest,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> CharOcrPageResult: ...


@dataclass(frozen=True, slots=True)
class OcrPageCommit:
    page_uid: str
    run_uid: str
    batch_uid: str
    pointer_revision: int
    proof_states_created: int
    proof_states_marked_for_rebind: int


@dataclass(frozen=True, slots=True)
class OcrPageFailureCommit:
    page_uid: str
    run_uid: str
    batch_uid: str
    pointer_revision: int


@dataclass(frozen=True, slots=True)
class OcrPageJobFailure:
    request: "OcrPageJobRequest"
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, OcrPageJobRequest):
            raise TypeError("OCR page failure requires its immutable request")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("OCR page failure requires a non-empty message")


@dataclass(frozen=True, slots=True)
class OcrPageJobRequest:
    """Immutable OCR input captured before a worker starts."""

    project_uid: str
    page: PageRecord
    layout: LayoutSnapshot
    artifact: PaddleArtifact
    image_bytes: bytes
    expected_pointer_revision: int
    expected_pointer_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class OcrObservationUnit:
    run: OcrRun
    regions: tuple[OcrRegion, ...]
    lines: tuple[OcrLine, ...]
    atoms: tuple[OcrAtom, ...]
    candidates: tuple[OcrCandidate, ...]
    batch: OcrBatch
    block_region_uids: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class OcrPageJobResult:
    """Uncommitted OCR observations produced by a worker."""

    request: OcrPageJobRequest
    records: OcrObservationUnit


def _uid(prefix: str) -> str:
    return f"{prefix}_{new_ulid()}"


class OcrJobService:
    """Compile, execute, and atomically adopt one page OCR observation batch."""

    def __init__(
        self,
        *,
        prepass_client=None,
        vl_client=None,
        prepass_client_factory: Callable[[], object] | None = None,
        vl_client_factory: Callable[[], object] | None = None,
        engine: CharOcrEngine,
    ) -> None:
        if prepass_client is not None and prepass_client_factory is not None:
            raise ValueError("provide prepass_client or prepass_client_factory, not both")
        if vl_client is not None and vl_client_factory is not None:
            raise ValueError("provide vl_client or vl_client_factory, not both")
        if prepass_client is None and prepass_client_factory is None:
            raise ValueError("OCR requires a PP-OCRv6 prepass client")
        if vl_client is None and vl_client_factory is None:
            raise ValueError("OCR requires a Paddle VL observation client")
        if prepass_client_factory is not None and not callable(prepass_client_factory):
            raise TypeError("prepass_client_factory must be callable")
        if vl_client_factory is not None and not callable(vl_client_factory):
            raise TypeError("vl_client_factory must be callable")
        self._prepass_client = prepass_client
        self._vl_client = vl_client
        self._prepass_client_factory = prepass_client_factory
        self._vl_client_factory = vl_client_factory
        self._engine = engine

    @property
    def supports_parallel_pages(self) -> bool:
        """Whether each page receives isolated external clients and a safe engine."""

        return bool(
            self._prepass_client_factory is not None
            and self._vl_client_factory is not None
            and getattr(self._engine, "supports_parallel_pages", False)
        )

    def _page_clients(self) -> tuple[object, object]:
        prepass = (
            self._prepass_client_factory()
            if self._prepass_client_factory is not None
            else self._prepass_client
        )
        vl_client = (
            self._vl_client_factory()
            if self._vl_client_factory is not None
            else self._vl_client
        )
        if prepass is None or vl_client is None:
            raise RuntimeError("OCR page clients are unavailable")
        return prepass, vl_client

    def run_page(
        self,
        session: ProjectSession,
        page_uid: str,
        image_bgr: np.ndarray,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> OcrPageCommit:
        request = self.prepare_page(session, page_uid, image_bgr=image_bgr)
        result = self.execute_page(request, progress_callback=progress_callback)
        return self.commit_page(session, result)

    def prepare_page(
        self,
        session: ProjectSession,
        page_uid: str,
        *,
        image_bgr: np.ndarray | None = None,
    ) -> OcrPageJobRequest:
        """Capture all authoritative OCR inputs and pointer CAS state."""
        import cv2

        page = session.page_repository.get(page_uid)
        layout = session.layout_repository.get(page_uid)
        if not layout.artifact_uid:
            raise RuntimeError("OCR requires an adopted Paddle layout artifact")
        artifact = session.paddle_artifact_repository.get(layout.artifact_uid)
        require_current_automatic_inline_formula_layout(
            layout,
            artifact,
            page_width=page.width,
            page_height=page.height,
        )
        if image_bgr is None:
            image_path = Path(page.cache_image_path or page.image_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"page image is missing: {image_path}")
            image_bytes = image_path.read_bytes()
        else:
            ok, encoded = cv2.imencode(".png", image_bgr)
            if not ok:
                raise ValueError("page image cannot be encoded for immutable OCR request")
            image_bytes = encoded.tobytes()
        if not image_bytes:
            raise ValueError("page image is empty")
        try:
            current_pointer = session.ocr_observation_repository.get_active_pointer(page.uid)
        except RecordNotFoundError:
            expected_pointer_revision = 0
            expected_pointer_fingerprint = None
        else:
            expected_pointer_revision = current_pointer.revision
            expected_pointer_fingerprint = current_pointer.fingerprint
        return OcrPageJobRequest(
            project_uid=session.project_uid,
            page=page,
            layout=layout,
            artifact=artifact,
            image_bytes=image_bytes,
            expected_pointer_revision=expected_pointer_revision,
            expected_pointer_fingerprint=expected_pointer_fingerprint,
        )

    def execute_page(
        self,
        job: OcrPageJobRequest,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> OcrPageJobResult:
        """Run routing and CharOCR without mutating ProjectSession."""
        import cv2

        if not isinstance(job, OcrPageJobRequest):
            raise TypeError("OCR execution requires OcrPageJobRequest")
        page = job.page
        layout = job.layout
        artifact = job.artifact
        prepass_client, vl_client = self._page_clients()
        image_bgr = cv2.imdecode(
            np.frombuffer(job.image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("immutable OCR request image cannot be decoded")
        if progress_callback is not None:
            progress_callback(0, 0, "PP-OCRv6 行框定位")
        prepass = prepass_client.analyze_page(image_bgr, page_uid=page.uid)
        if progress_callback is not None:
            progress_callback(0, 0, "PP-OCRv6 行框完成")
        observations = acquire_routing_observation_bundle(
            page=page,
            snapshot=layout,
            artifact=artifact,
            image_bgr=image_bgr,
            prepass=prepass,
            vl_client=vl_client,
        )
        if progress_callback is not None:
            progress_callback(0, 0, "OCR 路由编译")
        routing_plan = compile_page_routing_plan(
            observations,
            page_width=page.width,
            page_height=page.height,
            page_image_bgr=image_bgr,
        )
        if not routing_plan.is_dispatchable:
            details = "; ".join(
                f"{item.code}@line={item.line_index} bbox={item.bbox}"
                for item in routing_plan.validation_issues[:5]
            )
            raise RuntimeError(f"page OCR routing is not dispatchable: {details}")
        charocr_request = compile_charocr_page_request(
            project_uid=job.project_uid,
            page=page,
            layout=layout,
            artifact=artifact,
            routing_plan=routing_plan,
        )
        result = self._engine.recognize_page(
            image_bgr,
            charocr_request,
            progress_callback=progress_callback,
        )
        if (
            result.page_uid != page.uid
            or result.input_fingerprint != charocr_request.input_fingerprint
        ):
            raise RuntimeError("CharOCR result does not match its immutable request")
        orientation = observe_page_display_orientation(routing_plan)
        orientation_metadata = orientation.metadata()
        orientation_keys = {key for key, _value in orientation_metadata}
        result = replace(
            result,
            metrics=(
                *(item for item in result.metrics if item[0] not in orientation_keys),
                *orientation_metadata,
            ),
        )
        batch_records = _observation_records(
            project_uid=job.project_uid,
            page_uid=page.uid,
            page_fingerprint=page.fingerprint,
            engine_id=self._engine.engine_id,
            layout_fingerprint=layout_snapshot_fingerprint(layout),
            result=result,
        )
        return OcrPageJobResult(request=job, records=batch_records)

    def commit_page(
        self,
        session: ProjectSession,
        result: OcrPageJobResult,
    ) -> OcrPageCommit:
        """CAS-adopt a completed OCR result on the application thread."""
        if not isinstance(result, OcrPageJobResult):
            raise TypeError("OCR commit requires OcrPageJobResult")
        job = result.request
        if session.project_uid != job.project_uid:
            raise RuntimeError("OCR result belongs to another project session")
        page = session.page_repository.get(job.page.uid, fingerprint=job.page.fingerprint)
        layout = session.layout_repository.get(page.uid, revision=job.layout.revision)
        if layout_snapshot_fingerprint(layout) != layout_snapshot_fingerprint(job.layout):
            raise RuntimeError("OCR result belongs to a stale layout snapshot")
        session.paddle_artifact_repository.get(
            job.artifact.uid,
            fingerprint=job.artifact.fingerprint,
        )
        batch_records = result.records
        ocr = session.ocr_observation_repository
        batch = batch_records.batch
        run = batch_records.run
        if job.expected_pointer_revision == 0:
            pointer_uid = f"ocrptr_{page.uid}"
            next_revision = 1
        else:
            current_pointer = ocr.get_active_pointer(
                page.uid,
                revision=job.expected_pointer_revision,
                fingerprint=job.expected_pointer_fingerprint,
            )
            pointer_uid = current_pointer.uid
            next_revision = current_pointer.revision + 1
        pointer = OcrActivePointer(
            project_uid=session.project_uid,
            uid=pointer_uid,
            scope_uid=page.uid,
            batch_uid=batch.uid,
            run_uid=run.uid,
            batch_fingerprint=batch.fingerprint,
            revision=next_revision,
        )
        bindings = self._observation_bindings(
            session,
            block_region_uids=batch_records.block_region_uids,
            regions=batch_records.regions,
            layout=layout,
            layout_fingerprint=run.layout_fingerprint,
        )
        proof_states = self._proof_rebind_states(
            session, page_uid=page.uid, layout_fingerprint=run.layout_fingerprint,
            source_fingerprint=batch.fingerprint,
        )
        new_proof_states = (
            ()
            if proof_states
            else (_initial_proof_state(
                project_uid=session.project_uid,
                page_uid=page.uid,
                layout_fingerprint=run.layout_fingerprint,
                source_fingerprint=batch.fingerprint,
                lines=batch_records.lines,
            ),)
        )
        session.adopt_ocr_page_observation(
            run=run, regions=batch_records.regions, lines=batch_records.lines,
            atoms=batch_records.atoms, candidates=batch_records.candidates,
            batch=batch, pointer=pointer,
            expected_pointer_revision=job.expected_pointer_revision,
            expected_pointer_fingerprint=job.expected_pointer_fingerprint,
            bindings=bindings,
            proof_states=proof_states,
            new_proof_states=new_proof_states,
        )
        return OcrPageCommit(
            page_uid=page.uid,
            run_uid=run.uid,
            batch_uid=batch.uid,
            pointer_revision=pointer.revision,
            proof_states_created=len(new_proof_states),
            proof_states_marked_for_rebind=len(proof_states),
        )

    def commit_page_failure(
        self,
        session: ProjectSession,
        failure: OcrPageJobFailure,
    ) -> OcrPageFailureCommit:
        """CAS-adopt one failed OCR attempt without changing proof state."""
        if not isinstance(failure, OcrPageJobFailure):
            raise TypeError("OCR failure commit requires OcrPageJobFailure")
        job = failure.request
        if session.project_uid != job.project_uid:
            raise RuntimeError("OCR failure belongs to another project session")
        page = session.page_repository.get(job.page.uid, fingerprint=job.page.fingerprint)
        layout = session.layout_repository.get(page.uid, revision=job.layout.revision)
        layout_fingerprint = layout_snapshot_fingerprint(layout)
        if layout_fingerprint != layout_snapshot_fingerprint(job.layout):
            raise RuntimeError("OCR failure belongs to a stale layout snapshot")
        session.paddle_artifact_repository.get(
            job.artifact.uid,
            fingerprint=job.artifact.fingerprint,
        )
        input_fingerprint = hashlib.sha256(
            "\0".join((
                page.fingerprint,
                layout_fingerprint,
                job.artifact.fingerprint,
                self._engine.engine_id,
            )).encode("utf-8")
        ).hexdigest()
        run = OcrRun(
            project_uid=session.project_uid,
            uid=_uid("ocrrun"),
            engine=self._engine.engine_id,
            layout_fingerprint=layout_fingerprint,
            input_fingerprint=input_fingerprint,
            status="failed",
            metadata=(
                ("page_uid", page.uid),
                ("page_fingerprint", page.fingerprint),
                ("image_hash", page.image_hash),
                ("error", failure.message.strip()),
            ),
        )
        batch = OcrBatch(
            project_uid=session.project_uid,
            uid=_uid("ocrbatch"),
            run_uid=run.uid,
            scope_uid=page.uid,
            input_fingerprint=input_fingerprint,
            layout_fingerprint=layout_fingerprint,
            status="failed",
        )
        if job.expected_pointer_revision == 0:
            pointer_uid = f"ocrptr_{page.uid}"
        else:
            current_pointer = session.ocr_observation_repository.get_active_pointer(
                page.uid,
                revision=job.expected_pointer_revision,
                fingerprint=job.expected_pointer_fingerprint,
            )
            pointer_uid = current_pointer.uid
        pointer = OcrActivePointer(
            project_uid=session.project_uid,
            uid=pointer_uid,
            scope_uid=page.uid,
            batch_uid=batch.uid,
            run_uid=run.uid,
            batch_fingerprint=batch.fingerprint,
            revision=job.expected_pointer_revision + 1,
        )
        session.adopt_ocr_page_failure(
            run=run,
            batch=batch,
            pointer=pointer,
            expected_pointer_revision=job.expected_pointer_revision,
            expected_pointer_fingerprint=job.expected_pointer_fingerprint,
        )
        return OcrPageFailureCommit(
            page_uid=page.uid,
            run_uid=run.uid,
            batch_uid=batch.uid,
            pointer_revision=pointer.revision,
        )

    @staticmethod
    def _proof_rebind_states(
        session: ProjectSession,
        *,
        page_uid: str,
        layout_fingerprint: str,
        source_fingerprint: str,
    ) -> tuple[ProofState, ...]:
        repository = session.proof_repository
        updates = []
        for state in repository.all_states():
            if state.anchor_snapshot.scope_uid != page_uid:
                continue
            current_anchor = state.anchor_snapshot
            anchor = ProofAnchorSnapshot(
                project_uid=session.project_uid,
                uid=current_anchor.uid,
                scope_uid=page_uid,
                layout_fingerprint=layout_fingerprint,
                source_fingerprint=source_fingerprint,
                anchor_revision=current_anchor.anchor_revision + 1,
                geometry_fingerprint="",
                revision=current_anchor.revision + 1,
            )
            updates.append(replace(
                state, anchor_snapshot=anchor, alignment_segments=(),
                alignment_slices=(), rebind_required=True,
            ))
        return tuple(updates)

    @staticmethod
    def _observation_bindings(
        session: ProjectSession,
        *,
        block_region_uids: tuple[tuple[str, str], ...],
        regions: tuple[OcrRegion, ...],
        layout: LayoutSnapshot,
        layout_fingerprint: str,
    ) -> tuple[BindingRecord, ...]:
        block_uids = tuple(item[0] for item in block_region_uids)
        if len(set(block_uids)) != len(block_uids):
            raise RuntimeError("CharOCR returned more than one region for a layout block")
        layout_block_uids = {item.uid for item in layout.blocks}
        unknown = set(block_uids) - layout_block_uids
        if unknown:
            raise RuntimeError(
                f"CharOCR returned regions for unknown layout block UIDs: {sorted(unknown)!r}"
            )
        region_by_uid = {item.uid: item for item in regions}
        current_by_uid = {item.uid: item for item in session.binding_repository.all()}
        bindings = []
        for block_uid, region_uid in block_region_uids:
            uid = f"ocrbind_{block_uid}"
            current = current_by_uid.get(uid)
            bindings.append(BindingRecord(
                project_uid=session.project_uid,
                uid=uid,
                source_uid=block_uid,
                target_uid=region_uid,
                relation="observed_by",
                source_fingerprint=layout_fingerprint,
                target_fingerprint=region_by_uid[region_uid].fingerprint,
                revision=current.revision if current is not None else 0,
            ))
        return tuple(bindings)


def _observation_records(
    *,
    project_uid: str,
    page_uid: str,
    page_fingerprint: str,
    engine_id: str,
    layout_fingerprint: str,
    result: CharOcrPageResult,
) -> OcrObservationUnit:
    run_uid = _uid("ocrrun")
    batch_uid = _uid("ocrbatch")
    regions: list[OcrRegion] = []
    lines: list[OcrLine] = []
    atoms: list[OcrAtom] = []
    candidates: list[OcrCandidate] = []
    block_region_uids: list[tuple[str, str]] = []
    for region_index, observed_region in enumerate(result.regions):
        region_uid = _uid("ocrregion")
        block_region_uids.append((observed_region.block_uid, region_uid))
        line_records: list[OcrLine] = []
        regions.append(OcrRegion(
            project_uid=project_uid,
            uid=region_uid,
            run_uid=run_uid,
            page_uid=page_uid,
            bbox=observed_region.bbox,
            kind=observed_region.label or "text",
            order=region_index,
            label=observed_region.label,
        ))
        for observed_line in observed_region.lines:
            line_uid = _uid("ocrline")
            atom_records: list[OcrAtom] = []
            for atom_index, observed_atom in enumerate(observed_line.atoms):
                atom_uid = _uid("ocratom")
                candidate_records = tuple(
                    OcrCandidate(
                        project_uid=project_uid,
                        uid=_uid("ocrcandidate"),
                        run_uid=run_uid,
                        region_uid=region_uid,
                        line_uid=line_uid,
                        atom_uid=atom_uid,
                        batch_uid=batch_uid,
                        text=item.text,
                        confidence=item.confidence,
                        rank=rank,
                        source=item.source or observed_atom.source,
                        bbox=(
                            item.bbox
                            if item.bbox is not None
                            else observed_atom.bbox
                        ),
                    )
                    for rank, item in enumerate(observed_atom.candidates)
                )
                candidates.extend(candidate_records)
                atom_record = OcrAtom(
                    project_uid=project_uid,
                    uid=atom_uid,
                    run_uid=run_uid,
                    region_uid=region_uid,
                    line_uid=line_uid,
                    index=atom_index,
                    text=observed_atom.text,
                    bbox=observed_atom.bbox,
                    confidence=observed_atom.confidence,
                    candidate_uids=tuple(item.uid for item in candidate_records),
                    source=observed_atom.source,
                    granularity=observed_atom.granularity,
                    token_text=observed_atom.token_text,
                    source_fingerprint=result.input_fingerprint,
                )
                atom_records.append(atom_record)
                atoms.append(atom_record)
            line_record = OcrLine(
                project_uid=project_uid,
                uid=line_uid,
                run_uid=run_uid,
                region_uid=region_uid,
                page_uid=page_uid,
                text=observed_line.text,
                bbox=observed_line.bbox,
                confidence=observed_line.confidence,
                order=len(lines),
                atom_uids=tuple(item.uid for item in atom_records),
                source_fingerprint=result.input_fingerprint,
            )
            line_records.append(line_record)
            lines.append(line_record)
    run = OcrRun(
        project_uid=project_uid,
        uid=run_uid,
        engine=engine_id,
        layout_fingerprint=layout_fingerprint,
        input_fingerprint=result.input_fingerprint,
        metadata=(
            *tuple(item for item in result.metrics if item[0] != "page_fingerprint"),
            ("page_fingerprint", page_fingerprint),
        ),
    )
    batch = OcrBatch(
        project_uid=project_uid,
        uid=batch_uid,
        run_uid=run_uid,
        scope_uid=page_uid,
        input_fingerprint=result.input_fingerprint,
        layout_fingerprint=layout_fingerprint,
        region_uids=tuple(item.uid for item in regions),
        line_uids=tuple(item.uid for item in lines),
        atom_uids=tuple(item.uid for item in atoms),
        candidate_uids=tuple(item.uid for item in candidates),
    )
    return OcrObservationUnit(
        run=run, regions=tuple(regions), lines=tuple(lines), atoms=tuple(atoms),
        candidates=tuple(candidates), batch=batch,
        block_region_uids=tuple(block_region_uids),
    )


def _initial_proof_state(
    *,
    project_uid: str,
    page_uid: str,
    layout_fingerprint: str,
    source_fingerprint: str,
    lines: tuple[OcrLine, ...],
) -> ProofState:
    """Create the first human-editable aggregate from one adopted OCR batch."""

    ordered_lines = tuple(sorted(lines, key=lambda item: (item.order, item.uid)))
    anchor_uid = f"proofanchor_{page_uid}"
    text_units: list[ProofTextUnit] = []
    segments: list[ProofAlignmentSegment] = []
    slices: list[ProofAlignmentSlice] = []
    source_cursor = 0
    proof_cursor = 0
    for order, line in enumerate(ordered_lines):
        unit_uid = f"proofunit_{line.uid}"
        segment_uid = f"proofsegment_{line.uid}"
        text_units.append(ProofTextUnit(
            project_uid=project_uid,
            uid=unit_uid,
            order=order,
            text=line.text,
        ))
        source_end = source_cursor + len(line.text)
        proof_end = proof_cursor + len(line.text)
        segments.append(ProofAlignmentSegment(
            project_uid=project_uid,
            uid=segment_uid,
            anchor_uid=anchor_uid,
            text_unit_uid=unit_uid,
            source_line_uids=(line.uid,),
            source_start=source_cursor,
            source_end=source_end,
            proof_start=proof_cursor,
            proof_end=proof_end,
        ))
        slices.append(ProofAlignmentSlice(
            project_uid=project_uid,
            uid=f"proofslice_{line.uid}",
            segment_uid=segment_uid,
            source_start=source_cursor,
            source_end=source_end,
            proof_start=proof_cursor,
            proof_end=proof_end,
            source_text=line.text,
            proof_text=line.text,
        ))
        source_cursor = source_end
        proof_cursor = proof_end
    return ProofState(
        project_uid=project_uid,
        uid=f"proof_{page_uid}",
        anchor_snapshot=ProofAnchorSnapshot(
            project_uid=project_uid,
            uid=anchor_uid,
            scope_uid=page_uid,
            layout_fingerprint=layout_fingerprint,
            source_fingerprint=source_fingerprint,
            anchor_revision=1,
        ),
        text_units=tuple(text_units),
        alignment_segments=tuple(segments),
        alignment_slices=tuple(slices),
    )


__all__ = [
    "CharOcrEngine",
    "OcrJobService",
    "OcrObservationUnit",
    "OcrPageCommit",
    "OcrPageFailureCommit",
    "OcrPageJobFailure",
    "OcrPageJobRequest",
    "OcrPageJobResult",
]
