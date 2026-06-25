# 路由真值契约

本文档定义 Paddle/VL、PP-OCRv5、Hanwang 在混合 OCR 链路中的职责边界。后续排查和改代码时，以这里的真值归属为准。

## 核心边界

### PaddleOCR-VL 1.6

VL 是版面结构真值来源。

- 提供页面级结构块：正文、标题、表格、图片、公式、页码、脚注等。
- 提供父级 `block_content`，其中公式以 LaTeX 文本出现。
- 提供公式/表格/图片等非正文块的几何框。
- 对 `inline_formula`，几何框用于从正文行中扣除公式区域。
- 对 `footnote`、`vision_footnote`、表格、图片、display formula，默认不送 Hanwang 正文 OCR。

VL 不负责正文纵校 OCR 的最终文字。

### PP-OCRv5

PP-OCRv5 在 Hanwang 混合链路中只作为行几何提示。

- 允许使用 `rec_boxes` / `rec_polys` 作为正文物理行框。
- 不允许把 PP-OCRv5 的正文识别文本作为纵校正文真值。
- 在公式几何完整时，不允许 PP-OCRv5 文本覆盖 VL 父级公式文本。
- 只有当公式几何缺失导致父级公式序列和公式框数量不一致时，PP-OCRv5 行文本可以作为弱对齐提示，用于避免后续公式整体错位；该文本仍不得进入最终正文。

### Hanwang

Hanwang 是正文 OCR 真值来源。

- 输入为 PP-OCRv5 行框扣除 VL 公式框后的 text slice。
- Hanwang `linecut_segimg` 会对 text slice 再切 group。
- Hanwang `linecut_recog` 实际识别的是 group crop，而不是最初的整条 text slice。
- Hanwang 字符框是横校/纵校的主要字框来源。

公式框扣除后，公式文本由 VL 父级 `block_content` 回填，不能由 Hanwang 识别公式，也不能由 PP-OCRv5 回填公式文本。

## Route 字段语义

### `_route_subblocks`

来源：VL 几何记录挂到父级正文块后的结构子框。

用途：

- 表示父块内部需要特殊处理的结构区域。
- `inline_formula` 用于扣除公式区域并回填 VL 公式文本。
- `formula_number`、table、image 等按 skip/非正文逻辑处理。

### `_layout_line_routes`

来源：

- 优先由 PP-OCRv5 行框 + `_route_subblocks` 重建。
- 如果没有 PP-OCRv5 行框，可由 VL 父框和子框推导，作为降级路径。

用途：

- 描述正文父块内每一物理行的 route。
- `text` segment 是送 Hanwang SegImg 的候选区域。
- `formula` segment 是不送 Hanwang、由 VL 公式文本回填的区域。
- marker-only 公式序号如 `$ ^{②} $` 默认不扣洞、不回填，只保留在 Hanwang 文本路径中。

生命周期：

- 有 PP-OCRv5 page-line prepass 时，应重新生成。
- 某个父块没有匹配到 PP-OCRv5 行框时，必须清理旧缓存，避免旧 route 继续参与切图。
- 人工改框、合并、改类型后，必须使旧 OCR 文本和旧 route 失效。

## 调试视图

至少区分三层框：

- `layout_bbox`：VL 父级版面框。
- `route_text_slice_bbox`：扣掉公式后送入 Hanwang SegImg 的区域。
- `recog_group_bbox`：Hanwang SegImg 二次切分后，实际送入 Hanwang Recog 的区域。

如果只看 `route_text_slice_bbox`，可能漏掉 Hanwang 二次切分导致的问题。

当前代码在 Hanwang micro-recblock 输出中写入 `_hanwang_bbox_audit`：

- `layout_block_bbox`：原始 VL/Paddle 父级版面框。
- `effective_block_bbox`：最终 `BlockResult.block_bbox` 使用的框；有 `_layout_line_routes` 时为物理行 route 的 union，否则为父级版面框。
- `effective_block_bbox_source`：`layout_line_routes_union` 或 `layout_block_bbox`。
- `layout_line_route_bboxes`：每条物理行 route 的框。
- `route_text_slice_bboxes`：实际传给 `linecut_segimg` 的 text slice 框。
- `hanwang_recog_group_bboxes`：SegImg 返回并与 text slice 相交后的实际 Recog crop 框。
- `hanwang_segimg_groups`：每个 SegImg group 的审计记录，包含 route slice、SegImg 原始 group、最终 Recog group、是否被裁剪、是否被丢弃。
- `hanwang_segimg_group_clipped_count`：SegImg group 超出 route slice 后被裁剪的数量。
- `hanwang_segimg_group_dropped_count`：SegImg group 与 route slice 不相交而被丢弃的数量。

`LineResult.bbox_source` 也需要保留：

- `hanwang_recog_group`：来自 Hanwang Recog/SegImg group。
- `layout_route_assembled`：由 route text slice + VL inline formula 合成后的行。
- `ppvl_route_skip_segment`：表格、公式等 skip segment 直接由 PPVL route 回填。

普通文字路径不再使用 PPVL 文本 fallback。Hanwang 识别为空时保持空结果，通过审计字段定位问题。

## 120169 当前结论

真实 PP-OCRv5 行框正常。

- `Time_t` 和 `beta_t` 都被正确扣除并由 VL 回填。
- marker-only `②` 不再扣洞，保留给 Hanwang 识别。
- `② -> 1g` 未复现。
- 当前仍存在的 `t -> ，`、`0 -> o` 是 Hanwang 在正常输入框内的局部识别错误，不是切框错误。

## 后续收口方向

1. 给 route/cache 增加明确来源标记，例如 `line_geometry_source=ppocrv5`、`formula_text_source=vl_parent`。
2. UI 调试层读取 `_hanwang_bbox_audit`，同时显示 route text slice 和 Hanwang recog group。
3. 保持正文不采信 PP-OCRv5 文本，只在公式几何缺失时保留弱对齐能力。
4. 建立 Hanwang 局部 token 纠错层，优先处理 `t/c`、`0/o`、`1/l/I`、公式序号等低风险模式。
