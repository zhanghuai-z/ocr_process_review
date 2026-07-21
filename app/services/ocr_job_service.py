"""Project-scoped OCR use case over immutable requests and observations."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np

from app.core.layout_scope import layout_snapshot_fingerprint
from app.core.ppocr_route_compiler import compile_page_routing_plan
from app.models.charocr_execution import CharOcrPageRequest, CharOcrPageResult
from app.models.entity_id import new_ulid
from app.models.layout_snapshot import LayoutSnapshot
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
from app.models.project_session import BindingRecord, ProjectSession, RecordNotFoundError
from app.services.charocr_input_service import compile_charocr_page_request
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
class _ObservationUnit:
    run: OcrRun
    regions: tuple[OcrRegion, ...]
    lines: tuple[OcrLine, ...]
    atoms: tuple[OcrAtom, ...]
    candidates: tuple[OcrCandidate, ...]
    batch: OcrBatch
    block_region_uids: tuple[tuple[str, str], ...]


def _uid(prefix: str) -> str:
    return f"{prefix}_{new_ulid()}"


class OcrJobService:
    """Compile, execute, and atomically adopt one page OCR observation batch."""

    def __init__(self, *, prepass_client, vl_client, engine: CharOcrEngine) -> None:
        self._prepass_client = prepass_client
        self._vl_client = vl_client
        self._engine = engine

    def run_page(
        self,
        session: ProjectSession,
        page_uid: str,
        image_bgr: np.ndarray,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> OcrPageCommit:
        page = session.page_repository.get(page_uid)
        layout = session.layout_repository.get(page_uid)
        if not layout.artifact_uid:
            raise RuntimeError("OCR requires an adopted Paddle layout artifact")
        artifact = session.paddle_artifact_repository.get(layout.artifact_uid)
        if progress_callback is not None:
            progress_callback(0, 0, "PP-OCRv6 行框定位")
        prepass = self._prepass_client.analyze_page(image_bgr, page_uid=page.uid)
        if progress_callback is not None:
            progress_callback(0, 0, "PP-OCRv6 行框完成")
        observations = acquire_routing_observation_bundle(
            page=page,
            snapshot=layout,
            artifact=artifact,
            image_bgr=image_bgr,
            prepass=prepass,
            vl_client=self._vl_client,
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
        request = compile_charocr_page_request(
            project_uid=session.project_uid,
            page=page,
            layout=layout,
            artifact=artifact,
            routing_plan=routing_plan,
        )
        result = self._engine.recognize_page(
            image_bgr,
            request,
            progress_callback=progress_callback,
        )
        if result.page_uid != page.uid or result.input_fingerprint != request.input_fingerprint:
            raise RuntimeError("CharOCR result does not match its immutable request")
        batch_records = _observation_records(
            project_uid=session.project_uid,
            page_uid=page.uid,
            engine_id=self._engine.engine_id,
            layout_fingerprint=layout_snapshot_fingerprint(layout),
            result=result,
        )
        ocr = session.ocr_observation_repository
        batch = batch_records.batch
        run = batch_records.run
        try:
            current_pointer = ocr.get_active_pointer(page.uid)
        except RecordNotFoundError:
            pointer_uid = f"ocrptr_{page.uid}"
            expected_revision = 0
            expected_fingerprint = None
            next_revision = 1
        else:
            pointer_uid = current_pointer.uid
            expected_revision = current_pointer.revision
            expected_fingerprint = current_pointer.fingerprint
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
            expected_pointer_revision=expected_revision,
            expected_pointer_fingerprint=expected_fingerprint,
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
    engine_id: str,
    layout_fingerprint: str,
    result: CharOcrPageResult,
) -> _ObservationUnit:
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
                        source=observed_atom.source,
                        bbox=observed_atom.bbox,
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
        metadata=result.metrics,
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
    return _ObservationUnit(
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


__all__ = ["CharOcrEngine", "OcrJobService", "OcrPageCommit"]
