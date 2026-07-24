"""Read-only proof workspace query over one ``ProjectSession``.

The query joins the active OCR observation pointer with the current proof
state. It never refreshes, rebinds, or otherwise mutates either repository.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from app.core.char_index import CharIndexEntry
from app.core.ocr_currentness import CurrentOcrObservation, current_ocr_observation
from app.core.ocr_display_orientation import display_rotation_from_metadata
from app.core.paddle_labels import normalize_paddle_label
from app.models.ocr_records import OcrAtom, OcrBatch, OcrLine, OcrRegion
from app.models.proof_records import ProofState, ProofTextUnit
from app.models.project_session import PageRecord, ProjectSession
from app.services.char_index_service import CharIndexService


BBox = tuple[int, int, int, int]


def _uid(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty stable UID")
    return value


def _bbox(value: BBox, field_name: str = "bbox") -> BBox:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise TypeError(f"{field_name} must contain four integer coordinates")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{field_name} coordinates must be integers")
    result = tuple(value)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{field_name} must be non-empty")
    return result  # type: ignore[return-value]


def _char_span(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError("char_span must contain two integer offsets or be None")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError("char_span offsets must be integers")
    start, end = tuple(value)
    if start < 0 or end <= start:
        raise ValueError("char_span must be a non-empty half-open range")
    return start, end


@dataclass(frozen=True, slots=True)
class ProofPageView:
    """Image and page context only; a page view owns no proof text."""

    project_uid: str
    page_uid: str
    page_number: int
    source_page_index: int
    image_path: str
    source_path: str
    cache_image_path: str
    thumbnail_path: str
    width: int
    height: int
    status: str
    error: str
    image_hash: str
    image_revision: int
    page_fingerprint: str
    display_rotation_quarters_clockwise: int = 0


@dataclass(frozen=True, slots=True)
class ProofTextUnitView:
    """One human-editable proof unit together with its CAS facts."""

    proof_uid: str
    text_unit_uid: str
    order: int
    text: str
    status: str
    revision: int
    fingerprint: str

    @property
    def proof_state_uid(self) -> str:
        return self.proof_uid


@dataclass(frozen=True, slots=True)
class ProofAtomView:
    """One active OCR atom associated with a proof state."""

    proof_uid: str
    batch_uid: str
    page_uid: str
    region_uid: str
    line_uid: str
    atom_uid: str
    atom_index: int
    text: str
    confidence: float
    bbox: BBox
    source: str
    granularity: str
    token_text: str
    render_kind: str
    char_span: tuple[int, int] | None
    geometry_available: bool

    def __post_init__(self) -> None:
        _uid(self.proof_uid, "proof_uid")
        _uid(self.batch_uid, "batch_uid")
        _uid(self.page_uid, "page_uid")
        _uid(self.region_uid, "region_uid")
        _uid(self.line_uid, "line_uid")
        _uid(self.atom_uid, "atom_uid")
        if isinstance(self.atom_index, bool) or not isinstance(self.atom_index, int) or self.atom_index < 0:
            raise ValueError("atom_index must be a non-negative integer")
        if not isinstance(self.text, str):
            raise TypeError("text must be str")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be a number")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        if not isinstance(self.source, str):
            raise TypeError("source must be str")
        if not isinstance(self.granularity, str) or not self.granularity:
            raise ValueError("granularity must be a non-empty str")
        if not isinstance(self.token_text, str):
            raise TypeError("token_text must be str")
        if self.render_kind not in {"text", "formula", "table"}:
            raise ValueError("render_kind must be text, formula, or table")
        object.__setattr__(self, "char_span", _char_span(self.char_span))
        if not isinstance(self.geometry_available, bool):
            raise TypeError("geometry_available must be bool")
        if self.geometry_available != (self.char_span is not None):
            raise ValueError("geometry_available must agree with char_span availability")

    @property
    def proof_state_uid(self) -> str:
        return self.proof_uid

    @property
    def ocr_text(self) -> str:
        return self.text

    @property
    def char_start(self) -> int | None:
        return None if self.char_span is None else self.char_span[0]

    @property
    def char_end(self) -> int | None:
        return None if self.char_span is None else self.char_span[1]


@dataclass(frozen=True, slots=True)
class ProofLineView:
    """A proof text unit projected onto active OCR line and atom facts."""

    proof_uid: str
    batch_uid: str
    page_uid: str
    line_uid: str | None
    region_uid: str | None
    line_uids: tuple[str, ...]
    region_uids: tuple[str, ...]
    text_unit_uid: str
    order: int
    ocr_text: str
    proof_text: str
    status: str
    confidence: float | None
    bbox: BBox | None
    render_kind: str
    atoms: tuple[ProofAtomView, ...]
    state_revision: int
    state_fingerprint: str
    text_unit_revision: int
    text_unit_fingerprint: str
    region_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _uid(self.proof_uid, "proof_uid")
        _uid(self.batch_uid, "batch_uid")
        _uid(self.page_uid, "page_uid")
        if self.line_uid is not None:
            _uid(self.line_uid, "line_uid")
        if self.region_uid is not None:
            _uid(self.region_uid, "region_uid")
        line_uids = tuple(_uid(item, "line_uid") for item in self.line_uids)
        region_uids = tuple(_uid(item, "region_uid") for item in self.region_uids)
        atoms = tuple(self.atoms)
        if any(not isinstance(item, ProofAtomView) for item in atoms):
            raise TypeError("atoms must contain ProofAtomView values")
        if any(item.proof_uid != self.proof_uid for item in atoms):
            raise ValueError("atom belongs to another proof state")
        if any(item.batch_uid != self.batch_uid for item in atoms):
            raise ValueError("atom belongs to another OCR batch")
        if self.line_uid is not None and self.line_uid not in line_uids:
            raise ValueError("line_uid must be included in line_uids")
        if self.region_uid is not None and self.region_uid not in region_uids:
            raise ValueError("region_uid must be included in region_uids")
        object.__setattr__(self, "line_uids", line_uids)
        object.__setattr__(self, "region_uids", region_uids)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(
            self,
            "region_kinds",
            tuple(normalize_paddle_label(item) for item in self.region_kinds),
        )
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise TypeError("confidence must be a number or None")
            object.__setattr__(self, "confidence", float(self.confidence))
        if self.bbox is not None:
            object.__setattr__(self, "bbox", _bbox(self.bbox))
        if self.render_kind not in {"text", "formula", "table", "mixed"}:
            raise ValueError("invalid proof line render_kind")

    @property
    def proof_state_uid(self) -> str:
        return self.proof_uid

    @property
    def text(self) -> str:
        return self.proof_text

    @property
    def ocr_line_uid(self) -> str | None:
        return self.line_uid

    @property
    def atom_uids(self) -> tuple[str, ...]:
        return tuple(item.atom_uid for item in self.atoms)


@dataclass(frozen=True, slots=True)
class ProofStateView:
    """Immutable proof-state metadata and its projected text units."""

    proof_uid: str
    page_uid: str
    scope_uid: str
    anchor_uid: str
    anchor_revision: int
    layout_fingerprint: str
    source_fingerprint: str
    active_pointer_uid: str
    active_pointer_revision: int
    active_batch_uid: str
    revision: int
    fingerprint: str
    rebind_required: bool
    text_units: tuple[ProofTextUnitView, ...]
    lines: tuple[ProofLineView, ...]

    def __post_init__(self) -> None:
        _uid(self.proof_uid, "proof_uid")
        _uid(self.page_uid, "page_uid")
        _uid(self.scope_uid, "scope_uid")
        _uid(self.anchor_uid, "anchor_uid")
        _uid(self.active_pointer_uid, "active_pointer_uid")
        _uid(self.active_batch_uid, "active_batch_uid")
        text_units = tuple(self.text_units)
        lines = tuple(self.lines)
        if any(not isinstance(item, ProofTextUnitView) for item in text_units):
            raise TypeError("text_units must contain ProofTextUnitView values")
        if any(not isinstance(item, ProofLineView) for item in lines):
            raise TypeError("lines must contain ProofLineView values")
        if any(item.proof_uid != self.proof_uid for item in (*text_units, *lines)):
            raise ValueError("proof view item belongs to another proof state")
        object.__setattr__(self, "text_units", text_units)
        object.__setattr__(self, "lines", lines)

    @property
    def proof_state_uid(self) -> str:
        return self.proof_uid

    @property
    def state_uid(self) -> str:
        return self.proof_uid

    @property
    def text_unit_uids(self) -> tuple[str, ...]:
        return tuple(item.text_unit_uid for item in self.text_units)


@dataclass(frozen=True, slots=True)
class FormulaNumberLinkView:
    """Discardable presentation link between a display formula and its number.

    Both text units remain independent OCR/proof facts.  Consumers may use the
    link to present them together, or ignore it without losing either fact.
    """

    proof_uid: str
    page_uid: str
    formula_text_unit_uid: str
    number_text_unit_uid: str

    def __post_init__(self) -> None:
        _uid(self.proof_uid, "proof_uid")
        _uid(self.page_uid, "page_uid")
        _uid(self.formula_text_unit_uid, "formula_text_unit_uid")
        _uid(self.number_text_unit_uid, "number_text_unit_uid")
        if self.formula_text_unit_uid == self.number_text_unit_uid:
            raise ValueError("formula number link requires two distinct text units")


@dataclass(frozen=True, slots=True)
class ProofWorkspaceView:
    """Complete read-only proof query result for one project session."""

    project_uid: str
    pages: tuple[ProofPageView, ...]
    proof_states: tuple[ProofStateView, ...]
    lines: tuple[ProofLineView, ...]
    formula_number_links: tuple[FormulaNumberLinkView, ...] = ()

    def __post_init__(self) -> None:
        _uid(self.project_uid, "project_uid")
        pages = tuple(self.pages)
        proof_states = tuple(self.proof_states)
        lines = tuple(self.lines)
        formula_number_links = tuple(self.formula_number_links)
        if any(not isinstance(item, ProofPageView) for item in pages):
            raise TypeError("pages must contain ProofPageView values")
        if any(not isinstance(item, ProofStateView) for item in proof_states):
            raise TypeError("proof_states must contain ProofStateView values")
        if any(not isinstance(item, ProofLineView) for item in lines):
            raise TypeError("lines must contain ProofLineView values")
        if any(not isinstance(item, FormulaNumberLinkView) for item in formula_number_links):
            raise TypeError("formula_number_links must contain FormulaNumberLinkView values")
        states_by_uid = {state.proof_uid: state for state in proof_states}
        for link in formula_number_links:
            state = states_by_uid.get(link.proof_uid)
            if state is None:
                raise ValueError("formula number link belongs to an unknown proof state")
            if link.page_uid != state.page_uid:
                raise ValueError("formula number link page does not match its proof state")
            unit_uids = set(state.text_unit_uids)
            if {
                link.formula_text_unit_uid,
                link.number_text_unit_uid,
            } - unit_uids:
                raise ValueError("formula number link references an unknown text unit")
        object.__setattr__(self, "pages", pages)
        object.__setattr__(self, "proof_states", proof_states)
        object.__setattr__(self, "lines", lines)
        object.__setattr__(self, "formula_number_links", formula_number_links)

    @property
    def states(self) -> tuple[ProofStateView, ...]:
        return self.proof_states

    @property
    def proof_lines(self) -> tuple[ProofLineView, ...]:
        return self.lines

    @property
    def page_uids(self) -> tuple[str, ...]:
        return tuple(item.page_uid for item in self.pages)

    @property
    def proof_state_uids(self) -> tuple[str, ...]:
        return tuple(item.proof_uid for item in self.proof_states)

    @property
    def atoms(self) -> tuple[ProofAtomView, ...]:
        return tuple(atom for line in self.lines for atom in line.atoms)


@dataclass(frozen=True, slots=True)
class ProofStatePatch:
    """Fresh projections for the text units changed by one proof edit."""

    proof_uid: str
    revision: int
    fingerprint: str
    rebind_required: bool
    text_units: tuple[ProofTextUnitView, ...]
    lines: tuple[ProofLineView, ...]

    def __post_init__(self) -> None:
        _uid(self.proof_uid, "proof_uid")
        text_units = tuple(self.text_units)
        lines = tuple(self.lines)
        if not text_units:
            raise ValueError("proof state patch requires at least one changed text unit")
        if any(item.proof_uid != self.proof_uid for item in (*text_units, *lines)):
            raise ValueError("proof state patch item belongs to another proof state")
        unit_uids = {item.text_unit_uid for item in text_units}
        if unit_uids != {item.text_unit_uid for item in lines}:
            raise ValueError("proof state patch lines must match changed text units")
        object.__setattr__(self, "text_units", text_units)
        object.__setattr__(self, "lines", lines)


@dataclass(frozen=True, slots=True)
class ProofWorkspacePatch:
    """Incremental proof projection produced from committed repository facts."""

    project_uid: str
    states: tuple[ProofStatePatch, ...]

    def __post_init__(self) -> None:
        _uid(self.project_uid, "project_uid")
        states = tuple(self.states)
        if not states:
            raise ValueError("proof workspace patch requires at least one state")
        if len({item.proof_uid for item in states}) != len(states):
            raise ValueError("proof workspace patch contains duplicate proof states")
        object.__setattr__(self, "states", states)


def _require_session(session: ProjectSession) -> ProjectSession:
    if not isinstance(session, ProjectSession):
        raise TypeError("proof workspace query requires ProjectSession")
    return session


def _page_view(page: PageRecord, *, display_rotation: int = 0) -> ProofPageView:
    return ProofPageView(
        project_uid=page.project_uid,
        page_uid=page.uid,
        page_number=page.page_number,
        source_page_index=page.source_page_index,
        image_path=page.image_path,
        source_path=page.source_path,
        cache_image_path=page.cache_image_path,
        thumbnail_path=page.thumbnail_path,
        width=page.width,
        height=page.height,
        status=page.status,
        error=page.error,
        image_hash=page.image_hash,
        image_revision=page.image_revision,
        page_fingerprint=page.fingerprint,
        display_rotation_quarters_clockwise=display_rotation,
    )


def _ordered_lines(ocr, batch: OcrBatch) -> tuple[OcrLine, ...]:
    lines = tuple(ocr.get_line(uid) for uid in batch.line_uids)
    return tuple(sorted(lines, key=lambda item: (item.order, item.uid)))


def _ordered_regions(ocr, batch: OcrBatch) -> tuple[OcrRegion, ...]:
    regions = tuple(ocr.get_region(uid) for uid in batch.region_uids)
    return tuple(sorted(regions, key=lambda item: (item.order, item.uid)))


def _ordered_atoms(ocr, batch: OcrBatch) -> tuple[OcrAtom, ...]:
    atoms = tuple(ocr.get_atom(uid) for uid in batch.atom_uids)
    return tuple(sorted(atoms, key=lambda item: (item.line_uid, item.index, item.uid)))


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _entries_by_unit(
    entries: tuple[CharIndexEntry, ...],
    unit: ProofTextUnit,
) -> tuple[CharIndexEntry, ...]:
    return tuple(item for item in entries if item.text_unit_uid == unit.uid)


def _atom_char_span(
    atom: OcrAtom,
    entries: tuple[CharIndexEntry, ...],
) -> tuple[int, int] | None:
    """Return only a contiguous span backed by current index entries."""

    indices = sorted({
        item.char_index
        for item in entries
        if item.atom_uid == atom.uid
        and item.available
        and item.atom_fingerprint == atom.fingerprint
    })
    if not indices:
        return None
    start, end = indices[0], indices[-1] + 1
    if indices != list(range(start, end)):
        return None
    return start, end


_FORMULA_REGION_KINDS = frozenset({
    "display_equation",
    "display_formula",
    "equation",
    "equation_block",
    "formula",
    "inline_formula",
    "isolated_formula",
})
_DISPLAY_FORMULA_REGION_KINDS = frozenset({
    "display_equation",
    "display_formula",
    "equation",
    "equation_block",
    "isolated_formula",
})
_FORMULA_NUMBER_REGION_KINDS = frozenset({"equation_number", "formula_number"})
_TABLE_REGION_KINDS = frozenset({
    "table",
    "table_block",
    "table_body",
    "table_region",
})


def _render_kind(*, region_kind: str, atom_source: str = "") -> str:
    normalized = normalize_paddle_label(region_kind)
    if normalized in _FORMULA_REGION_KINDS:
        return "formula"
    if normalized in _TABLE_REGION_KINDS:
        return "table"
    if atom_source == "paddle_inline_formula":
        return "formula"
    return "text"


def _atom_view(
    proof_uid: str,
    batch_uid: str,
    page_uid: str,
    atom: OcrAtom,
    region_kind: str,
    entries: tuple[CharIndexEntry, ...],
) -> ProofAtomView:
    char_span = _atom_char_span(atom, entries)
    return ProofAtomView(
        proof_uid=proof_uid,
        batch_uid=batch_uid,
        page_uid=page_uid,
        region_uid=atom.region_uid,
        line_uid=atom.line_uid,
        atom_uid=atom.uid,
        atom_index=atom.index,
        text=atom.text,
        confidence=atom.confidence,
        bbox=atom.bbox,
        source=atom.source,
        granularity=atom.granularity,
        token_text=atom.token_text,
        render_kind=_render_kind(region_kind=region_kind, atom_source=atom.source),
        char_span=char_span,
        geometry_available=char_span is not None,
    )


def _bbox_union(boxes: Iterable[BBox]) -> BBox:
    values = tuple(boxes)
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )  # type: ignore[return-value]


def _line_view(
    *,
    state: ProofState,
    unit: ProofTextUnit,
    batch: OcrBatch,
    page: PageRecord,
    lines_by_uid: dict[str, OcrLine],
    atoms_by_uid: dict[str, OcrAtom],
    regions_by_uid: dict[str, OcrRegion],
    entries: tuple[CharIndexEntry, ...],
) -> ProofLineView:
    segments = tuple(
        item for item in state.alignment_segments if item.text_unit_uid == unit.uid
    )
    if len(segments) > 1:
        raise ValueError(f"proof unit {unit.uid!r} has multiple alignment segments")
    mapped_line_uids = segments[0].source_line_uids if segments else ()
    unknown_line_uids = set(mapped_line_uids) - set(lines_by_uid)
    if unknown_line_uids:
        if state.anchor_snapshot.source_fingerprint != batch.fingerprint:
            mapped_line_uids = ()
        else:
            raise ValueError(
                f"proof unit {unit.uid!r} references OCR lines outside the active batch: "
                f"{sorted(unknown_line_uids)!r}"
            )
    mapped_lines = tuple(
        sorted(
            (lines_by_uid[uid] for uid in mapped_line_uids),
            key=lambda item: (item.order, item.uid),
        )
    )
    line_position = {line.uid: index for index, line in enumerate(mapped_lines)}
    mapped_atoms = tuple(
        sorted(
            (
                atoms_by_uid[atom_uid]
                for line in mapped_lines
                for atom_uid in line.atom_uids
            ),
            key=lambda item: (line_position[item.line_uid], item.index, item.uid),
        )
    )
    region_uids = _unique(item.region_uid for item in mapped_lines)
    region_kinds = _unique(regions_by_uid[uid].kind for uid in region_uids)
    line_render_kinds = _unique(
        _render_kind(region_kind=kind) for kind in region_kinds
    )
    render_kind = (
        line_render_kinds[0]
        if len(line_render_kinds) == 1
        else "mixed" if line_render_kinds else "text"
    )
    line_uid = mapped_lines[0].uid if len(mapped_lines) == 1 else None
    region_uid = region_uids[0] if len(region_uids) == 1 else None
    confidence: float | None
    if mapped_atoms:
        confidence = sum(item.confidence for item in mapped_atoms) / len(mapped_atoms)
    elif mapped_lines:
        confidence = sum(item.confidence for item in mapped_lines) / len(mapped_lines)
    else:
        confidence = None
    bbox: BBox | None
    line_boxes = [item.bbox for item in mapped_lines]
    # 映射多条 OCR 行时取并集（公式/表格行常见）。无行映射时
    # 几何明确不可用，不从字符索引或版面投影反推新的事实。
    bbox = _bbox_union(line_boxes) if line_boxes else None
    return ProofLineView(
        proof_uid=state.uid,
        batch_uid=batch.uid,
        page_uid=page.uid,
        line_uid=line_uid,
        region_uid=region_uid,
        line_uids=mapped_line_uids,
        region_uids=region_uids,
        text_unit_uid=unit.uid,
        order=unit.order,
        ocr_text="".join(item.text for item in mapped_lines),
        proof_text=unit.text,
        status=unit.status,
        confidence=confidence,
        bbox=bbox,
        render_kind=render_kind,
        atoms=tuple(
            _atom_view(
                state.uid,
                batch.uid,
                page.uid,
                atom,
                regions_by_uid[atom.region_uid].kind,
                entries,
            )
            for atom in mapped_atoms
        ),
        state_revision=state.revision,
        state_fingerprint=state.fingerprint,
        text_unit_revision=unit.revision,
        text_unit_fingerprint=unit.fingerprint,
        region_kinds=region_kinds,
    )


def _formula_number_links(state: ProofStateView) -> tuple[FormulaNumberLinkView, ...]:
    """Associate only unambiguous, same-row formula-number observations."""

    formulas = tuple(
        line
        for line in state.lines
        if set(line.region_kinds) & _DISPLAY_FORMULA_REGION_KINDS
        and line.bbox is not None
    )
    numbers = tuple(
        line
        for line in state.lines
        if set(line.region_kinds) & _FORMULA_NUMBER_REGION_KINDS
        and line.bbox is not None
    )
    links: list[FormulaNumberLinkView] = []
    claimed_formula_uids: set[str] = set()
    for number in numbers:
        assert number.bbox is not None
        number_center_x = (number.bbox[0] + number.bbox[2]) / 2.0
        candidates = []
        for formula in formulas:
            assert formula.bbox is not None
            formula_center_x = (formula.bbox[0] + formula.bbox[2]) / 2.0
            vertical_overlap = min(formula.bbox[3], number.bbox[3]) - max(
                formula.bbox[1], number.bbox[1]
            )
            if vertical_overlap > 0 and number_center_x > formula_center_x:
                candidates.append(formula)
        if len(candidates) != 1:
            continue
        formula = candidates[0]
        if formula.text_unit_uid in claimed_formula_uids:
            continue
        claimed_formula_uids.add(formula.text_unit_uid)
        links.append(FormulaNumberLinkView(
            proof_uid=state.proof_uid,
            page_uid=state.page_uid,
            formula_text_unit_uid=formula.text_unit_uid,
            number_text_unit_uid=number.text_unit_uid,
        ))
    return tuple(links)


def _state_view(
    session: ProjectSession,
    state: ProofState,
    pages_by_uid: dict[str, PageRecord],
    observation: CurrentOcrObservation,
) -> ProofStateView:
    scope_uid = state.anchor_snapshot.scope_uid
    page = pages_by_uid.get(scope_uid)
    if page is None:
        raise ValueError(f"proof state {state.uid!r} references an unknown page")
    if state.project_uid != session.project_uid:
        raise ValueError(f"proof state {state.uid!r} belongs to another project")

    ocr = session.ocr_observation_repository
    pointer = observation.pointer
    batch = observation.batch
    if pointer.project_uid != session.project_uid:
        raise ValueError("active OCR pointer belongs to another project")
    if batch.scope_uid != scope_uid:
        raise ValueError("active OCR batch scope does not match proof state")

    regions = _ordered_regions(ocr, batch)
    lines = _ordered_lines(ocr, batch)
    atoms = _ordered_atoms(ocr, batch)
    index = CharIndexService().build(
        page=page,
        batch=batch,
        lines=lines,
        atoms=atoms,
        state=state,
    )
    lines_by_uid = {item.uid: item for item in lines}
    atoms_by_uid = {item.uid: item for item in atoms}
    regions_by_uid = {item.uid: item for item in regions}
    line_views = tuple(
        _line_view(
            state=state,
            unit=unit,
            batch=batch,
            page=page,
            lines_by_uid=lines_by_uid,
            atoms_by_uid=atoms_by_uid,
            regions_by_uid=regions_by_uid,
            entries=_entries_by_unit(index.entries, unit),
        )
        for unit in state.text_units
    )
    text_units = tuple(
        ProofTextUnitView(
            proof_uid=state.uid,
            text_unit_uid=unit.uid,
            order=unit.order,
            text=unit.text,
            status=unit.status,
            revision=unit.revision,
            fingerprint=unit.fingerprint,
        )
        for unit in state.text_units
    )
    return ProofStateView(
        proof_uid=state.uid,
        page_uid=page.uid,
        scope_uid=scope_uid,
        anchor_uid=state.anchor_snapshot.uid,
        anchor_revision=state.anchor_snapshot.anchor_revision,
        layout_fingerprint=state.anchor_snapshot.layout_fingerprint,
        source_fingerprint=state.anchor_snapshot.source_fingerprint,
        active_pointer_uid=pointer.uid,
        active_pointer_revision=pointer.revision,
        active_batch_uid=batch.uid,
        revision=state.revision,
        fingerprint=state.fingerprint,
        rebind_required=state.rebind_required,
        text_units=text_units,
        lines=line_views,
    )


def _state_patch(
    session: ProjectSession,
    state: ProofState,
    changed_text_unit_uids: frozenset[str],
    pages_by_uid: dict[str, PageRecord],
    observation: CurrentOcrObservation,
) -> ProofStatePatch:
    scope_uid = state.anchor_snapshot.scope_uid
    page = pages_by_uid.get(scope_uid)
    if page is None:
        raise ValueError(f"proof state {state.uid!r} references an unknown page")
    if state.project_uid != session.project_uid:
        raise ValueError(f"proof state {state.uid!r} belongs to another project")

    units_by_uid = {unit.uid: unit for unit in state.text_units}
    unknown = changed_text_unit_uids - set(units_by_uid)
    if unknown:
        raise ValueError(f"proof patch references unknown text units: {sorted(unknown)!r}")
    changed_units = tuple(
        unit for unit in state.text_units if unit.uid in changed_text_unit_uids
    )
    if not changed_units:
        raise ValueError("proof patch requires at least one changed text unit")

    ocr = session.ocr_observation_repository
    pointer = observation.pointer
    batch = observation.batch
    if pointer.project_uid != session.project_uid:
        raise ValueError("active OCR pointer belongs to another project")
    if batch.scope_uid != scope_uid:
        raise ValueError("active OCR batch scope does not match proof state")

    regions = _ordered_regions(ocr, batch)
    lines = _ordered_lines(ocr, batch)
    atoms = _ordered_atoms(ocr, batch)
    index = CharIndexService().build(
        page=page,
        batch=batch,
        lines=lines,
        atoms=atoms,
        state=state,
        text_unit_uids=changed_text_unit_uids,
    )
    lines_by_uid = {item.uid: item for item in lines}
    atoms_by_uid = {item.uid: item for item in atoms}
    regions_by_uid = {item.uid: item for item in regions}
    return ProofStatePatch(
        proof_uid=state.uid,
        revision=state.revision,
        fingerprint=state.fingerprint,
        rebind_required=state.rebind_required,
        text_units=tuple(
            ProofTextUnitView(
                proof_uid=state.uid,
                text_unit_uid=unit.uid,
                order=unit.order,
                text=unit.text,
                status=unit.status,
                revision=unit.revision,
                fingerprint=unit.fingerprint,
            )
            for unit in changed_units
        ),
        lines=tuple(
            _line_view(
                state=state,
                unit=unit,
                batch=batch,
                page=page,
                lines_by_uid=lines_by_uid,
                atoms_by_uid=atoms_by_uid,
                regions_by_uid=regions_by_uid,
                entries=_entries_by_unit(index.entries, unit),
            )
            for unit in changed_units
        ),
    )


def build_proof_workspace_patch(
    session: ProjectSession,
    changes: Mapping[str, Iterable[str]],
) -> ProofWorkspacePatch:
    """Project only text units named by committed proof edit results."""

    session = _require_session(session)
    if not isinstance(changes, Mapping) or not changes:
        raise ValueError("proof workspace patch requires committed changes")
    pages = tuple(session.page_repository.all())
    pages_by_uid = {item.uid: item for item in pages}
    patches: list[ProofStatePatch] = []
    for proof_uid, text_unit_uids in changes.items():
        state = session.proof_repository.get_state(_uid(proof_uid, "proof_uid"))
        changed = frozenset(_uid(item, "text_unit_uid") for item in text_unit_uids)
        if not changed:
            raise ValueError("proof workspace patch change set must not be empty")
        observation = current_ocr_observation(session, state.anchor_snapshot.scope_uid)
        if observation is None:
            raise ValueError(f"proof state {proof_uid!r} has no active OCR observation")
        patches.append(_state_patch(session, state, changed, pages_by_uid, observation))
    return ProofWorkspacePatch(project_uid=session.project_uid, states=tuple(patches))


def apply_proof_workspace_patch(
    workspace: ProofWorkspaceView,
    patch: ProofWorkspacePatch,
) -> ProofWorkspaceView:
    """Merge an incremental projection without changing authoritative facts."""

    if workspace.project_uid != patch.project_uid:
        raise ValueError("proof workspace patch belongs to another project")
    patches_by_uid = {item.proof_uid: item for item in patch.states}
    unknown = set(patches_by_uid) - set(workspace.proof_state_uids)
    if unknown:
        raise ValueError(f"proof workspace patch references unknown states: {sorted(unknown)!r}")
    states: list[ProofStateView] = []
    for state in workspace.proof_states:
        state_patch = patches_by_uid.get(state.proof_uid)
        if state_patch is None:
            states.append(state)
            continue
        units_by_uid = {item.text_unit_uid: item for item in state_patch.text_units}
        lines_by_uid = {item.text_unit_uid: item for item in state_patch.lines}
        text_units = tuple(
            units_by_uid.get(item.text_unit_uid, item) for item in state.text_units
        )
        lines = tuple(
            lines_by_uid[item.text_unit_uid]
            if item.text_unit_uid in lines_by_uid
            else replace(
                item,
                state_revision=state_patch.revision,
                state_fingerprint=state_patch.fingerprint,
            )
            for item in state.lines
        )
        states.append(replace(
            state,
            revision=state_patch.revision,
            fingerprint=state_patch.fingerprint,
            rebind_required=state_patch.rebind_required,
            text_units=text_units,
            lines=lines,
        ))
    state_views = tuple(states)
    return ProofWorkspaceView(
        project_uid=workspace.project_uid,
        pages=workspace.pages,
        proof_states=state_views,
        lines=tuple(line for state in state_views for line in state.lines),
        formula_number_links=tuple(
            link for state in state_views for link in _formula_number_links(state)
        ),
    )


def build_proof_workspace_view(
    session: ProjectSession,
    *,
    proof_uid: str | None = None,
) -> ProofWorkspaceView:
    """Build an immutable proof view from current session facts only.

    ``proof_uid`` narrows the proof-state portion of the result while keeping
    the page directory intact. Missing active observations or proof records
    are surfaced by their repository contracts; no legacy data path is used.
    """

    session = _require_session(session)
    pages = tuple(sorted(session.page_repository.all(), key=lambda item: (item.page_number, item.uid)))
    pages_by_uid = {item.uid: item for item in pages}
    if proof_uid is None:
        states = session.proof_repository.all_states()
    else:
        states = (session.proof_repository.get_state(proof_uid),)
    page_order = {page.uid: index for index, page in enumerate(pages)}
    ordered_states = tuple(
        sorted(
            states,
            key=lambda item: (page_order.get(item.anchor_snapshot.scope_uid, len(page_order)), item.uid),
        )
    )
    state_views = tuple(
        _state_view(session, state, pages_by_uid, observation)
        for state in ordered_states
        if (observation := current_ocr_observation(
            session, state.anchor_snapshot.scope_uid
        )) is not None
    )
    lines = tuple(line for state in state_views for line in state.lines)
    formula_number_links = tuple(
        link for state in state_views for link in _formula_number_links(state)
    )
    return ProofWorkspaceView(
        project_uid=session.project_uid,
        pages=tuple(
            _page_view(
                page,
                display_rotation=(
                    display_rotation_from_metadata(observation.run.metadata)
                    if (observation := current_ocr_observation(session, page.uid)) is not None
                    else 0
                ),
            )
            for page in pages
        ),
        proof_states=state_views,
        lines=lines,
        formula_number_links=formula_number_links,
    )


def query_proof_workspace(
    session: ProjectSession,
    *,
    proof_uid: str | None = None,
) -> ProofWorkspaceView:
    """Named query entry point for callers that prefer query terminology."""

    return build_proof_workspace_view(session, proof_uid=proof_uid)


class ProofWorkspaceQuery:
    """Stateless callable facade for the proof workspace query."""

    __slots__ = ()

    def __call__(
        self,
        session: ProjectSession,
        *,
        proof_uid: str | None = None,
    ) -> ProofWorkspaceView:
        return build_proof_workspace_view(session, proof_uid=proof_uid)

    def execute(
        self,
        session: ProjectSession,
        *,
        proof_uid: str | None = None,
    ) -> ProofWorkspaceView:
        return self(session, proof_uid=proof_uid)

    def build(
        self,
        session: ProjectSession,
        *,
        proof_uid: str | None = None,
    ) -> ProofWorkspaceView:
        return self(session, proof_uid=proof_uid)


__all__ = [
    "FormulaNumberLinkView",
    "ProofAtomView",
    "ProofLineView",
    "ProofPageView",
    "ProofStateView",
    "ProofStatePatch",
    "ProofTextUnitView",
    "ProofWorkspaceQuery",
    "ProofWorkspacePatch",
    "ProofWorkspaceView",
    "apply_proof_workspace_patch",
    "build_proof_workspace_patch",
    "build_proof_workspace_view",
    "query_proof_workspace",
]
