# CharOCR Routing Architecture Review

## Scope

This stage reviewed the production path from adopted layout through PP-OCRv6
observations, physical-line geometry, typed routing, and native CharOCR input.
Code and focused regressions were treated as evidence; debug renderers were not.

## Authoritative Flow

`LayoutSnapshot` owns block type and geometry. `RoutingObservationBundle` binds
that immutable snapshot to the current image hash, PP-OCRv6 prepass, and scoped
VL observations. `compile_page_routing_plan()` derives a page-local immutable
`PageRoutingPlan`. The native adapter consumes that plan and writes OCR
observations; it cannot write layout truth or infer a replacement route.

Physical-line geometry and Latin/LineCut partitioning are compiler stages. PP
line and word boxes remain external proposals. Foreground components provide
measured geometry. Formula, table, figure, and decoration ownership is removed
before text dispatch. Invalid or ambiguous ownership rejects only the affected
page.

## Findings

- **P0:** none.
- **P1:** production documentation had drifted from word-level Latin routing
  and LineCut-owned external punctuation. The routing contract was updated in
  this stage.
- **P2:** the native adapter still builds transient PP-VL rows as block-UID
  carriers. This is not a routing authority: missing or extra text rows fail,
  and all crop decisions come from `PageRoutingPlan`. Remove this carrier when
  the native input DTO no longer depends on PP-VL row shape.
- **P2:** the Hanwang adapter remains a large integration module. Further work
  may extract native execution without moving route compilation into it.

No compatibility path, silent fallback, or new package-level import violation
was introduced. Experiment scripts continue to call the production compiler
and remain outside application imports.

## Production Decision

The physical-line resolver, word-level EngCut routing, LineCut remainder,
typed symbol observations, and glyph-integrity ownership rules are accepted as
the production baseline. Future routing changes must enter before the native
boundary by extending observations or the compiler, not by post-hoc engine
guessing.
