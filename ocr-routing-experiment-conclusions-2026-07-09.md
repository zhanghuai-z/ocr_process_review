# OCR Routing Experiment Conclusions 2026-07-09

## Scope

This note records current routing conclusions before the next production change.
It is not a permanent architecture map.

## Confirmed Facts

- VL1.6 layout facts and PP-OCR line facts both come from Paddle, but they have
  different authority. VL1.6 provides adopted layout blocks and inline-formula
  geometry. PP-OCR provides line geometry hints before CharOCR/Hanwang.
- Inline formula routing is a layout-stage fact from VL1.6 or a user-edited
  formula box. It should be locked before Hanwang crops are produced.
- PP-OCR line boxes are useful as physical row geometry. They should guide crop
  routing, not become layout truth.
- PP-OCR word boxes and mask-derived English regions are not stable enough to
  become production text or character truth. They are useful diagnostic inputs
  only. Common failures include punctuation ownership, emails/URLs, PE/VC-like
  tokens, missing narrow glyphs such as `i`, and occasional token/box count
  mismatch.
- EngCut exact output is reliable when binding succeeds. LineCut remains better
  for Chinese character geometry. Mixed lines need routing before Hanwang, not a
  late fallback after bad crops have already been sent.
- Mixed line is a physical-line condition, not a route segment kind. A mixed
  line may contain `formula`, `text_zh`, `text_latin`, and `skip` segments, but
  `text_mixed` and `unknown` are not valid crop destinations.

## Routing Direction

Routing should be one stage that classifies all crop destinations together:

- inline formula: preserve or crop-rebind through Paddle formula recognition;
- Chinese text: send qualified crop to LineCut;
- Latin/digit text: send qualified crop to EngCut;
- mixed line: split into qualified crops before either native component runs.

Hanwang/CharOCR should only receive qualified crops. If a crop cannot be routed
unambiguously, the current page should fail closed with a diagnostic error.
Do not synthesize layout blocks from unmatched OCR output.

## Current Boundary

LayoutSnapshot is layout truth. OCR routing, masks, EngCut/LineCut output, and
debug overlays are observations or plans. They may read layout truth but must not
rewrite it.
