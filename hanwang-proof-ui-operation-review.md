# Hanwang HProof/VProof 校对交互 Review

更新时间：2026-06-12

## 当前定位

HProof 是行级校对主入口：按页面/正文行展示“扫描行图 + 可编辑文本”，适合逐行确认、改字、标记疑点。

VProof 是全书同字/同 token 校对入口：左侧建立字符索引，点击某个字符或 token 后，在 gallery 中查看全书出现位置，并支持候选替换、直接键入、批量替换和填空白。

本轮已把 VProof 的字符索引从“只收 CJK”改为“收 CJK + 英文 + 数字 + 标点”。公式/表格仍由横校右侧调试按钮作为辅助视图，不进入普通正文校对主线。

## HProof 数据与操作链

1. 数据入口

`HProofPanel.load_pages()` 接收页面列表，`_iter_page_lines()` 默认走 `iter_unique_page_hproof_lines(page)`，只进入正文类文本行。调试模式下，右侧“公式 / 表格”按钮切到 `iter_unique_page_hproof_debug_lines()`，只用于检查路由命中结果。

2. 行展示

每一行由 `_LinePair` 承载。上方行图来自 `line.bbox + ROW_PAD_Y` 的裁图；下方文本来自 `_displayed_text(line, page, block)`，通过 `quality_probe` 显示空间桥接后展示。

3. 图字对应

当 `len(editor_text) == len(line.chars)` 且每个 `Char.char` 都是单字符时，HProof 进入逐字对齐模式。它用 `line.chars[i].bbox` 的中心点推导 editor 中第 i 个字符的 x 坐标，并在 hover/click 时把图像框和文本槽位互相联动。

如果字符数不一致，或存在多字符 token，HProof 会降级为普通文本行，不再假装第 i 个文本字一定对应第 i 个图像框。

4. 编辑规则

固定长度模式下：

- 普通输入会覆盖当前槽位，不追加新字符。
- Backspace/Delete 不删除槽位，而是把槽位填成 ASCII 空格。
- 粘贴会按选区长度截断或补空格。

这套规则的核心目的是保持“图像字符槽位数量”和“文本长度”一致。

5. 保存与同步

`_save_current()` 和 `_on_text_saved()` 统一调用 `_save_displayed_edit()`，把显示文本写回 `line.final_text`，然后通过 `ProofStateBus` 发布 `line.proof_changed`。VProof 收到事件后会延迟合并刷新，避免 HProof 大批保存时重复重建全书 CharIndex。

## VProof 数据与操作链

1. 字符索引

`VProofPanel` 现在使用 `CharIndexService(include_non_cjk=True)` 构建索引。索引源来自 `iter_unique_page_text_lines(page)`，会收录正文和图注等文字校对线，但排除公式块、表格路由等非普通文本线。

2. 左侧列表

左侧展示 `CharIndexService.sorted_chars()` 的结果。数字连续串会按 token 归并，例如 `2026`；部分公式样式 token 仍会作为 token 展示。英文、数字和标点不再被默认过滤。

3. Gallery

点击左侧字符/token 后，VProof 通过 `CharIndexService.query(token)` 得到所有出现位置。Gallery item 保存 `CharEntry`，包含页号、line、char_idx、bbox、bbox_source、bbox_granularity、token_text 等字段。

4. 单点与批量修改

VProof 的替换都汇入 `_apply_replacement_to_selected()`：

- 单选时替换当前 entry。
- 多选时只替换当前页内的 entry，跨页 entry 会跳过并在状态栏提示。
- 替换时根据每个 entry 自己的 token 长度选区，避免用一个固定长度误吃后续字符。
- Backspace/Delete 会填空白，不删除槽位。

5. 保存与重建

直接键入或清空槽位后，VProof 启动 120ms debounce，调用 `_save_page_text()` 写回 `line.final_text`，再重建 `CharIndexService` 和左侧字符列表。保存后 `_loaded_text` 会同步到当前文本，避免后续 HProof 事件刷新时用旧文本覆盖新改动。

## 本轮和 120186 相关的证据

阶段框回显见：

- `debug/120186_stage_echo/120186_stage_echo.md`
- `debug/120186_stage_echo/120186_stage_overlay_preview.png`
- `debug/120186_stage_echo/120186_b003_l1_pevc_misread.png`
- `debug/120186_stage_echo/120186_b004_l1_lemer.png`
- `debug/120186_stage_echo/120186_b005_l4_pevc_exact.png`

120186 的焦点样例显示：

- `b003/l1` 的物理行框是完整的，问题是 Hanwang/EngCut 都把视觉 `/` 输出成 `!`，反选 fallback 只能找回 `PE` 和 `VC` 两段。
- `b004/l1` 的 `Lerner` 被输出为 `Lemer`，属于原稿 `rn` 黏连导致的高风险人工校对场景。
- `b005/l4` 的 `PE/VC` 可以被 EngCut exact path 正常定位，说明当前“Paddle token + EngCut exact”路径在识别文本一致时可用。

## 当前风险和 UI 优化方向

1. 英文/数字/标点已进入 VProof，但英文词可能以 token 形式出现，后续要决定“英文按字符校”还是“英文按词校”。这会影响同字列表密度和批量替换语义。

2. HProof 的图字对应依赖 `line.chars` 与文本等长。凡是 token-granularity 或人工增删导致长度变化的行，都应明确进入降级态，不应继续显示逐字框。

3. 当前 HProof 的 fixed-length 操作是安全的，但不一定最符合人工编辑习惯。后续可以考虑加“拆分/合并字符槽位”的显式工具，用于处理 `rn -> m`、`PE!VC -> PE/VC` 这类结构性错误。

4. VProof 的批量替换目前限制在当前页，避免跨页误改。这是保守策略。未来如果要做全书批改，应增加确认摘要和撤销事务。

5. 裁图层本轮只做显示 padding 和文本槽宽余量，不改真实 bbox。后续如果要做开发者模式，建议把 VL block、Hanwang line、Hanwang char、EngCut char、Paddle binding、人工框作为可切换图层，而不是继续散落在 debug 脚本里。
