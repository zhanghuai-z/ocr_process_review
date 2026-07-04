# Fallback 策略登记

生成时间：2026-07-04

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
| 拉丁 EngCut exact | `bind_latin_tokens_to_engcut_chars()` / `LATIN_ENGCUT_BBOX_SOURCE` | Paddle token 或行文本中的拉丁 token 能和 EngCut char 精确对齐 | 写入单字符 bbox，source 为 `hanwang:EngCut:latin_exact` | stats 记录 exact token 数 | 拉丁字符几何真值候选 |
| 拉丁 EngCut word fallback | `_word_fallback_binding()` / `LATIN_ENGCUT_WORD_FALLBACK_STATUS` | exact 找不到，但 EngCut 的词组边界和 token 安全 fuzzy 匹配 | 写入词级 bbox，source 为 `hanwang:EngCut:latin_word_fallback`，status 为 `latin_token_engcut_word_fallback` | stats 记录 word fallback；不成功则标 review | 词级几何，不能冒充逐字符真值 |
| 拉丁低置信目标行探测 | `_engcut_target_line_orders()` | Paddle token 未覆盖，Hanwang 行内出现低置信拉丁候选或疑似 CJK 混淆 | 只决定哪些行送 EngCut probe；不直接改文本 | stats 记录 probe 调用和 review token | 探测策略，不是项目真值 |
| 表格 cell bbox 网格降级 | `table_text_layer.build_table_text_layer_cells()` / `equal_grid_fallback` | 图像聚类无法推断 cell bbox，但已有 row/col 结构 | 按表格 bbox 等分生成 cell bbox，标记 `bbox_source="equal_grid_fallback"` | Export/Text layer 可见 | 表格文本层辅助几何，不是字符真值 |
| 导出资产降级 | `ExportFallback` / `image_fallback` / `text_fallback` / `plain_paragraph` | 结构化 table/equation/unknown block 无法完整表达 | Export IR 中写 `fallback`，并在 diagnostics 追加 warning | Markdown/XML/archive/PDF 导出诊断可见 | 导出表达策略，不回写项目真值 |

## 特别边界

### `fallback` 字样不等于低可信

`hanwang:CharRcg:char_fallback` 虽然包含 fallback 字样，但它来自 Hanwang native 的字符级输出，当前按一手字符几何处理。判断几何质量必须走 `is_estimated_or_unavailable_geometry()` 和 `is_char_index_hidden_geometry()`，不得用字符串包含 `fallback` 这种粗判断。

### PPVL 不再作为正文 OCR fallback

正文 OCR 不允许在 Hanwang/CharOCR 失败时静默退回 PPVL 文本并继续当作已识别正文。失败应进入 audit、error 或 review 状态。保留的 PPVL 文本只能作为外部事实或诊断参考，不能覆盖 CharOCR 的正文输出。

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
