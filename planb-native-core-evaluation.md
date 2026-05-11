# Plan B Native Core Evaluation

## 1. 当前问题为什么反复发生

Inspector 的失败点不在单个 UI 控件，而在 parser / geometry / OCR_IR / canvas 的交界反复重复推导同一组语义：

1. Paddle raw JSON 有多层包装：`result.ocrResults[]`、`layoutParsingResults[]`、`prunedResult`、flatten 后顶层、`overall_ocr_res`，同一字段可能出现在不同位置。
2. 几何字段有多种形态：xyxy、xywh、四点 polygon、flat 8 数字、dict alias。若每层自己解析 bbox，很容易把 polygon 当 xyxy、或把 page-space 再缩放一次。
3. `overall_ocr_res.rec_texts` 是行级主链，`text_word/text_word_region` 是 token/word 级，`parsing_res_list.block_content` 是块级。旧链路容易把上层文本/框误抬成下层语义。
4. UI 需要“是否可见”，但 parser 只给“有没有数据”；如果没有统一 availability diagnostics，用户看到的就是“chars 不显示”，coord 无法直接判断是服务端没返回、parser 丢了，还是 canvas 没消费。
5. 文本找图需要稳定 identity，而不是临时扫描 UI 树；否则搜索命中、canvas 选中、crop 导出之间无法形成可复现链路。

因此 Plan B 不应继续围绕某个 panel 补丁打转，而应先把上述语义放到一个可替换 core seam。

## 2. 本轮实际抽出的 core seam

本轮新增 `tools/ocr_inspector/core/`，当前仍是 Python 实现，但边界按后续可替换 native core 设计：

| 文件 | 职责 |
| --- | --- |
| `tools/ocr_inspector/core/paddle_seam.py` | Paddle raw normalization、bbox/polygon canonicalization、OCR_IR construction、availability diagnostics |
| `tools/ocr_inspector/core/search_index.py` | text-to-node search index、稳定 identity、bbox availability metadata |
| `tools/ocr_inspector/core/__init__.py` | 对 UI/adapter 暴露的 seam API |
| `tools/ocr_inspector/adapters/paddle.py` | 变成 thin wrapper，只负责 detect 与调用 core seam |
| `tools/ocr_inspector/crop.py` | 文本搜索改为消费 core search index，再生成 `TextMatch` |

当前 seam 输出：

- `DocumentNode/PageNode/LineNode/CharNode` IR；
- `CoreDiagnostic(code, severity, layer, message, row_index)`；
- `SearchIndexEntry(identity, node, page, text, kind, bbox, bbox_source, bbox_granularity)`。

这让 UI 壳层只关心 IR、selection、overlay、crop，不再各自理解 Paddle raw schema。

## 3. diagnostics 能区分的失败层

本轮新增 diagnostics 覆盖以下关键判断：

| 诊断 code | 含义 | 对应层 |
| --- | --- | --- |
| `server_missing_text_word_region` | raw response 没有 `text_word_region`，char/token bbox 不可用 | 服务端/请求/模型输出层 |
| `word_regions_preserved` | raw response 有 `text_word_region`，并成功进入 OCR char/token node | parser + OCR_IR 已保住 |
| `word_regions_not_consumed` | raw response 有 `text_word_region`，但没有形成 OCR char/token node | parser / geometry / token matching 失效 |
| `invalid_region_format` | token region 形态不可解析 | geometry canonicalization |
| `unsupported_model_granularity` | 例如 PaddleOCR-VL 只承诺 line/block，不承诺 word/char | 模型能力层 |
| `missing_rec_texts` | 行级主链缺失 | raw normalization |

这解决 coord 要求的三类判定：

1. 服务端没返回：`server_missing_text_word_region`。
2. 返回了但解析时丢了/坏了：`word_regions_not_consumed` + `invalid_region_format`。
3. 保住了但画布没消费：core 会给出 `word_regions_preserved`，若 canvas 仍没有 item，就可把问题定位到 canvas consumer。

## 4. 新增 fixtures / seam tests

本轮在 `tests/test_core.py` 增加 seam 回归：

1. `test_planb_core_seam_diagnoses_missing_word_region`
   - 构造只有 `rec_texts/rec_boxes`、没有 `text_word_region` 的样例；
   - 验证 char bbox 为 `unavailable`，diagnostic 为 `server_missing_text_word_region`。
2. `test_planb_core_seam_preserves_word_region_and_search_identity`
   - 构造 `ocrResults[].prunedResult.text_word/text_word_region` 样例；
   - 验证 word region 进入 IR，search index 生成稳定 identity。
3. `test_planb_core_seam_flags_invalid_word_region`
   - 构造 raw 中存在 `text_word_region` 但 region 格式不可解析的样例；
   - 验证 diagnostic 区分为 `invalid_region_format` 与 `word_regions_not_consumed`。

这些样例覆盖“未返回 / 返回但坏了 / 保住后可被搜索定位”的最小闭环。

## 5. 哪些模块应继续留在 Python

建议继续保留 Python / PySide6：

1. `tools/ocr_inspector/ui/*`：窗口、tab、toolbar、参数面板、日志面板。
2. `tools/ocr_inspector/ui/canvas/*` 的 view shell：QGraphicsScene/QGraphicsItem、selection wiring、hover、zoom/pan。
3. `CropPanel` 的交互层：输入框、列表、预览、保存对话框。
4. settings、project persistence、API profile UI。
5. 调试报告、snapshot/export UX。

理由：这些层依赖 Qt 对象生命周期和用户交互，Python/PySide6 迭代更快，且性能瓶颈不在这里。

## 6. 哪些模块适合下一步改成 Rust

最适合 native/Rust 化的是当前 `tools/ocr_inspector/core/` seam 背后的纯语义核心：

1. Paddle raw normalization：
   - 多包装层展平；
   - snake/camel field alias；
   - model family capability 归一。
2. bbox/polygon canonicalization：
   - xyxy / xywh / polygon / dict alias；
   - 坐标有效性检查；
   - 后续可加入 coordinate conflict detection。
3. OCR_IR construction：
   - line/token/block 分层；
   - token row 与 line row 匹配；
   - char/token availability。
4. diagnostics：
   - 机器可读 code；
   - severity/layer；
   - 支持 snapshot replay。
5. search index：
   - 稳定 identity；
   - 文本归一化；
   - bbox availability 元数据。

这些逻辑不依赖 Qt，输入输出可以是 JSON/dict-like data，适合 PyO3 封装。

## 7. 推荐 Rust 接口边界

建议 Rust core 暴露少量稳定函数，而不是把 UI 细节带进去：

```text
normalize_paddle(raw_json: str, options: NormalizeOptions) -> CoreResultJson
build_search_index(core_result_json: str) -> SearchIndexJson
query_search(index_json: str, query: str, options: SearchOptions) -> SearchResultJson
canonicalize_bbox(raw_region_json: str) -> BBoxResultJson
```

其中 `CoreResultJson` 应包含：

- pages / blocks / lines / chars；
- bbox/polygon；
- source field path；
- bbox_source / bbox_granularity / collection_kind；
- diagnostics；
- raw reference path 或 compact raw fragment。

Python 层只负责把 `CoreResultJson` 转成现有 `DocumentNode` dataclass 或直接适配 UI view model。

## 8. PyO3 / maturin 集成点、收益与风险

### 集成点

1. 新建 `ocr_inspector_core` Rust crate。
2. 用 `maturin` 产出 Python extension。
3. Python 的 `tools/ocr_inspector/core/paddle_seam.py` 保留为 facade：
   - 优先调用 Rust；
   - Rust 不可用时可在开发期 fallback 到 Python seam。
4. CI/打包后续再决定是否强制 native dependency。

### 收益

1. schema / geometry / diagnostics 可以用强类型收口，减少 Python dict 任意访问导致的漂移。
2. token/line 匹配、搜索索引、bbox 校验可更容易做 property tests。
3. native core 不依赖 Qt，可独立 fuzz / snapshot replay。
4. Python UI 仍保持快速迭代。

### 风险

1. 打包复杂度上升：Windows/Linux wheel、PyInstaller/briefcase 兼容要单独验证。
2. 调试门槛上升：团队需要 Rust/PyO3 基础。
3. 如果边界没先稳定，过早 Rust 化只会把不清晰的语义固化到 native 层。
4. Python dataclass 与 Rust JSON schema 需要版本化，否则会出现二次 schema 迁移问题。

## 9. 为什么当前不推荐直接上 C/C++

1. C/C++ 的内存安全和异常边界成本更高，不适合当前仍在快速变化的 OCR schema 解析层。
2. PySide6/Qt 已经在 Python 层，C++ core 与 Python UI 的绑定维护成本会比 PyO3 更高。
3. 当前目标是 schema/geometry/search 的纯数据核心，不需要 C++ 生态特有能力。
4. Rust 更适合做可测试、可替换、内存安全的解析核心；PyO3/maturin 的 Python 集成路径也更清晰。

## 10. 本轮完成范围与仍未完成

### 已完成

1. 建立 `tools/ocr_inspector/core/` seam。
2. PaddleAdapter 改成 thin wrapper，parser 语义进入 core。
3. bbox/polygon canonicalization 从 adapter 私有函数提升到 core。
4. char/token availability diagnostics 机器可读化。
5. 文本搜索改为消费 core search index，并带稳定 identity。
6. 新增三类 seam regression fixture。
7. 提交本评估报告。

### 尚未完成 / 后续建议

1. 主程序 `app/engines/real_ocr_adapter.py` 尚未切到同一个 Inspector core seam；本轮先在 Inspector 内成形边界，避免扩大范围。
2. coordinate conflict diagnostics 目前只预留 code 方向，尚未完整接入 dataInfo / input shape 冲突检测。
3. search index 目前是内存构建，尚未做 snapshot 持久化。
4. Rust 还未落地；当前完成的是 Python seam + Rust 边界评估，不是 native core 实现。

## 11. 决策建议

后续是否转 Rust 的建议：**可以准备，但不要立刻全量替换**。

推荐下一步：

1. 先让 coord 验收当前 Python seam 是否足够清晰。
2. 再选 `canonical_bbox + diagnostics + search index` 三个最纯的模块做 Rust spike。
3. spike 通过后，再把 Paddle normalization / OCR_IR construction 迁入 Rust。
4. UI、canvas、crop preview、settings 继续保留 Python。
