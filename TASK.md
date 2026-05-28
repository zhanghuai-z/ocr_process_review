# Coord Daily Task Board

## Current baseline

- branch: `coord/phase1-stabilization`
- head: `a74497d`

## Retained baseline

- module: `hanwang`
- line: `paddle-line-routing-rebuild`
- coord commit:
  - `a74497d` — `merge: rebuild Paddle line routing`

## Current state

- active dispatch line:
  - module: `hanwang`
  - line: `pre-hanwang-line-splitting`
- coordinator role on this line:
  - coord only dispatches / reviews / merges
  - coord does not implement locally
- line status:
  - GPT follow-up `20e2b84` reviewed
  - accepted
  - syncing into coord completed

## User-fixed architecture

1. standalone formula blocks do not need char splitting
2. standalone formula blocks remain Paddle block-level formula blocks
3. HProof receives one standalone formula block as one line
4. formulas do not enter vertical proof
5. inline formulas stay on their surrounding text line semantically
6. before Hanwang, an original text line must be split by inline-formula/table geometry into
   multiple text sub-lines
7. Hanwang only OCRs those text sub-lines

## Hard dispatch requirements

1. line authority must come from PP-OCRv5 page OCR lines, not Hanwang segmentation groups
2. parent block authority remains Paddle `parsing_res_list`
3. inline-formula/table geometry remains sourced from Paddle `layout_det_res`
4. pre-Hanwang route stage must:
   - assign PP-OCRv5 page lines to parent text blocks by spatial containment
   - split a single text line into ordered text sub-lines when inline formula/table boxes intersect it
   - exclude formula/table spans from Hanwang input
5. standalone formula blocks must bypass Hanwang as single-line outputs
6. do not use `_fallback_line(...chars=[CharResult(...) for ch in text])` style synthetic char
   splitting for standalone formula blocks
7. HProof/VProof behavior must match the user rule above

## Why current baseline is still insufficient

1. current baseline reconstructs line routes mainly from block text + geometry heuristics
2. current standalone skip path still wraps formula blocks through generic fallback-line logic
3. the user has now fixed the stricter target: line must be cut before Hanwang, and standalone
   formula blocks should stay block-level rather than char-synthesized

## Real-file validation target

- sample: `/mnt/d/project/ocr_process/file/244771纵校/120166.tif`
- required outcomes:
  - row 7 `display_formula` stays standalone formula output, one line, no char splitting
  - row 8 `formula_number` stays standalone formula output, one line, no char splitting
  - row 9 is driven by pre-Hanwang line splitting into text sub-lines around inline formulas
  - row 14 stays off the Hanwang text path as a standalone formula-style block

## Latest review result

- GPT delivered:
  - branch: `gpt/paddle-line-routing-rebuild`
  - commit: `20e2b84` — `Wire PP-OCR prepass into Hanwang hybrid flow`
- status:
  - reviewed
  - **accepted**
- accepted closure:
  1. live hybrid path now runs a PP-OCRv5 page-line prepass before Hanwang
  2. `recognize_page_blocks()` consumes real page lines from `page.blocks[*].lines`
  3. standalone formula outputs stay single-line and keep `chars=[]`
- validation:
  1. `tests/test_core.py`: `209 passed`
  2. real `120166.tif` replay:
     - row 7 `display_formula`: one line, no char split
     - row 8 `formula_number`: one line, no char split
     - row 9 `text`: pre-Hanwang split text sub-lines
     - row 14 `formula`: one line, no char split

## Post-merge residual fix

- issue:
  - equation/formula blocks could regain char boxes after proof-index build
- cause:
  - `CharIndexService` consumed `iter_unique_page_text_lines(page)`
  - that iterator still admitted some equation blocks through semantic labels such as `footer`
  - `ensure_line_char_bboxes()` then rebuilt per-char boxes on the formula line
- fix:
  1. proof line iterators now hard-exclude `block.block_type == EQUATION`
  2. char index no longer re-populates standalone formula chars
  3. real `120166.tif` check confirms row 7/8/14 stay `chars=[]` even after `CharIndexService().build([page])`
