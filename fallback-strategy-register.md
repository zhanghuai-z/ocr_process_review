# Fallback 策略登记

生成时间：2026-07-10

本文登记当前允许存在的 fallback / 降级策略。它的目的不是鼓励兜底逻辑继续扩散，而是规定：

1. fallback 是否会改变项目真值。
2. fallback 必须留下什么来源标记。
3. fallback 是否需要让用户或开发者可见。
4. fallback 后续如何退出。

## 总原则

1. 只处理通信或展示可用性的 fallback，可以不改变项目数据，但必须记录 telemetry、diagnostic 或日志。
2. 任何会生成或替换 bbox、char、line、table cell、export payload 的 fallback，都不能伪装成一手事实，必须写入 `bbox_source`、`bbox_granularity`、`review_flags`、`ExportFallback` 或等价标记。
3. fallback 不得覆盖 Paddle 原始版面事实、Hanwang/CharOCR 一手字符事实、人工版面编辑事实和人工校对文本。
4. fallback 数据进入 proof 时必须能被识别。估算或不可用几何默认不进入相同字索引，除非调用方显式 `include_fallback=True`。
5. 新 fallback 必须先登记到本文，再进入主线代码。

## 当前允许策略

| 策略 | 入口 | 触发条件 | 写入内容 | 可见性 | 真值等级 |
| --- | --- | --- | --- | --- | --- |
| Paddle 网络降级 | `PaddleV16Client._request_with_network_fallback` | proxy/direct 其中一路发生 `ProxyError`、`ConnectionError`、`SSLError` | 只写 `telemetry["*_fallback"]` 和 `telemetry["*_network_mode"]` | 调试 telemetry；不改 OCR 或 layout 数据 | 通信策略，不是项目真值 |
| Paddle 行框缺失降级 | `build_ir_lines_from_item()` + `MISSING_LINE_BBOX_FLAG` | Paddle OCR line 文本存在，但 line bbox 缺失或无效 | 使用调用方提供的 `fallback_bbox`，并写 `missing_line_bbox` review flag | 后续 proof crop / char index 可识别 | 低可信 OCR line 几何 |
| proof 几何补框 | `ProofCropService.normalize_pages()` / `ensure_line_char_bboxes()` | line 有文本但没有可用 char bbox，或 char 数量无法表达显示文本 | 生成 `bbox_source="fallback"` / `bbox_granularity="fallback"`；若 line bbox 不可信则写 `unavailable` | `proof_fallback_warning()` 给出非阻塞警告；`is_char_index_hidden_geometry()` 默认隐藏 | 估算几何，不是一手字符真值 |
| Hanwang 字符 fallback | `hanwang:CharRcg:char_fallback` | Hanwang native 返回单字 fallback 字符框 | 保留 `bbox_source` 中的 `hanwang:` 前缀和 `bbox_granularity="char"` | CharIndex 可索引；不按普通估算 fallback 隐藏 | Hanwang 一手字符几何 |
| PP 拉丁 token 几何降级 | `micro_recblock._engcut_route_line_text_and_chars()` | 一个 EngCut group 与一个 typed PP token 通过中心几何唯一一对一绑定，且 native 字符框重叠或文本不一致 | 写一个 word carrier；文本取 PP token，bbox 取 PP mask 与 native group 并集；`source="ppocrv6:latin_token_geometry_fallback"`，`review_flags=["latin_token_geometry_fallback"]` | OCR audit、proof/debug flag 和 `RunStats.latin_token_geometry_fallbacks` 可见 | 可复核 PP OCR observation，不是 Hanwang 字符真值 |
| 表格 cell bbox 网格降级 | `table_text_layer.build_table_text_layer_cells()` / `equal_grid_fallback` | 图像聚类无法推断 cell bbox，但已有 row/col 结构 | 按表格 bbox 等分生成 cell bbox，标记 `bbox_source="equal_grid_fallback"` | Export/Text layer 可见 | 表格文本层辅助几何，不是字符真值 |
| 导出资产降级 | `ExportFallback` / `image_fallback` / `text_fallback` / `plain_paragraph` | 结构化 table/equation/unknown block 无法完整表达 | Export IR 中写 `fallback`，并在 diagnostics 追加 warning | Markdown/XML/archive/PDF 导出诊断可见 | 导出表达策略，不回写项目真值 |

## 特别边界

### `fallback` 字样不等于低可信

`hanwang:CharRcg:char_fallback` 虽然包含 fallback 字样，但它来自 Hanwang native 的字符级输出，当前按一手字符几何处理。判断几何质量必须走 `is_estimated_or_unavailable_geometry()` 和 `is_char_index_hidden_geometry()`，不得用字符串包含 `fallback` 这种粗判断。

### PPVL 不再作为正文 OCR fallback

正文 OCR 不允许在 Hanwang/CharOCR 失败时静默退回 PPVL 文本并继续当作已识别正文。失败应进入 audit、error 或 review 状态。保留的 PPVL 文本只能作为外部事实或诊断参考，不能覆盖 CharOCR 的正文输出。

PP-OCRv6 prepass 与 PPVL 版面响应不是同一事实。当前只允许两种显式 PP-OCRv6 observation：完整组件归属的独立标点，以及上表登记的唯一几何绑定拉丁 word 降级。两者都必须保留来源；禁止扩大为整行/整块 PP 文本回填。

PP 拉丁 token 降级的退出条件是 EngCut 对该 group 同时给出一致文本和不重叠字符几何，此时自动恢复字符级输出。若未来 native 提供可验证置信度和更可靠斜体几何，可在同一 contract 下淘汰该降级，不保留双路径。

### 公式、表格、图片框不进入普通文字 OCR

Paddle 或人工确认的公式、表格、图片框是结构区域。它们可以有自己的 OCR / 导出策略，但不得因为正文切片 fallback 再被送入 Hanwang 普通文字识别。人工调框后，旧绑定文本必须失效或重新识别，不能沿用旧大框文本。

### proof fallback 必须可见

当 proof 阶段出现估算或不可用几何时，状态栏必须出现非阻塞 warning。开发调试时应能通过 `bbox_source` / `bbox_granularity` / review flag 判断问题来自 fallback，而不是误以为是当前主逻辑产生的一手 bbox。

## 新增 fallback 检查清单

新增任何 fallback 前，必须回答：

1. 它是通信、展示、几何、文本、导出，还是存储 fallback？
2. 它是否会写入项目模型或导出文件？
3. 写入时的 source / granularity / flag / diagnostic 是什么？
4. 用户或开发者在哪里能看到它触发了？
5. 它的退出条件是什么？如果无法退出，为什么它应成为正式策略？

如果答不出来，就不应进入主线代码。
