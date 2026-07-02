# OCR Process 当前真值地图

生成时间：2026-07-02

本文用于回答三个问题：

1. 一张图片进入程序后，各阶段数据变成了什么。
2. 近期修补前后，关键逻辑发生了什么变化。
3. 当前哪些字段/模块是“真值”，哪些只是索引、缓存、兼容层或 UI 投影。

这不是最终架构设计。它是当前基线的事实说明，用于后续重构和 review 对齐。

## 一、当前线性主链路

```text
导入文件
  -> Page
     - display_image_path 决定后续渲染、裁剪、OCR 使用的同一张工作图
     - source_path/source_page_index 记录原始来源

版面分析 Paddle VL1.6
  -> Page.raw_layout_artifact
     - Paddle 原始版面事实列表
     - 主要含 block_label、block_bbox、block_content 等
     - 通过 raw_ocr_artifact.raw_layout_records(page) 读取
  -> page.blocks
     - 程序理解后的 Block
     - block_type 是程序大类
     - source_label 是 Paddle 细标签
     - raw_payload 保存外部 vendor fact
     - app_payload 已退役为空；程序派生状态必须使用 typed 字段或事件

路由构建
  -> _route_subblocks / _layout_line_routes
     - 由 Paddle 的公式、表格、图片等结构框推导出文本切片路线
     - 作用是让正文块进入 Hanwang/CharOCR 前扣除或插入公式等结构段

OCR Hanwang/CharOCR
  -> block.lines
     - Line 是可校对文本行
     - Line.bbox 是行几何
     - Line.chars 是字符/词/公式 carrier 的几何与文本观察

校对 HProof/VProof
  -> Line.proof_state / ProofLineState
     - 人工校对后的文本事实
     - proof_display_text(line) 是统一显示文本入口
  -> Line.chars
     - 等长或可对齐编辑时同步字符事实
     - 不能同步时，下游 ProofAtom/CharIndex 必须降级或跳过
  -> ProofChangeSet
     - 描述本次 proof 改动范围与类型

持久化 ProjectStore
  -> SQLite
     - id 是 rowid，仅是存储实现
     - uid 是业务身份
     - proof 增量保存必须按 scoped line 更新，不能全项目覆盖

导出
  -> Export IR / Markdown / PDF
     - 应读取最终文本 proof_display_text(line)
     - 几何来自 Block/Line/Char bbox
     - 不应泄露本机绝对路径
```

## 二、字段真值表

| 领域 | 当前真值 | 不应作为真值 | 说明 |
|---|---|---|---|
| 页面工作图 | `Page.display_image_path` | `image_path` 单独使用 | 渲染、裁剪、OCR 必须同图，否则 bbox 漂移。 |
| 页面身份 | `Page.uid` | `Page.id` | `id` 是 SQLite rowid，可变化；业务引用应走 uid。 |
| 块身份 | `Block.uid` | `Block.id`、order、bbox | order/bbox 可随编辑变化，不是身份。 |
| 行身份 | `Line.uid` | `Line.id`、line index、bbox | HProof/VProof merge 必须 uid 优先，几何只能 fallback。 |
| 字符身份 | `Char.uid` | `Char.id` | 全量保存允许跨父级 move；proof 增量保存不允许跨行认领。 |
| Paddle 原始事实 | `Page.raw_layout_artifact`、`block.raw_payload` | `block.note`、`block.app_payload` | raw_payload 应只放外部返回事实；page 级原始列表不再挂 `ppvl_parsing_res_list`。 |
| 程序派生状态 | typed 字段：`origin`、`paddle_binding`、`ocr_invalidated_reason`、`ocr_audit`、`table_text_layer_cells`、`layout_edit_events` | `block.raw_payload`、`block.app_payload` | `app_payload` 已退役为空，非空会被存储校验拒绝。 |
| 块大类 | `Block.block_type` | Paddle 原始 label 直接判断 | UI 和导出看大类。 |
| Paddle 细标签 | `Block.source_label` / raw label | `Block.block_type` 反推 | 页眉、脚注、公式序号等细分来自 source_label。 |
| 是否进文本 OCR | `should_dispatch_to_text_ocr(block)` / `Block.ocr_policy` | `block_type/source_label/raw_payload` 的组合猜测 | 公式、表格、图片通过明确 policy 阻断。 |
| OCR 原文 | `Line.ocr_text` | `Line.text` 单独判断 | `text` 仍有兼容属性；新逻辑应优先明确 ocr_text。 |
| 校对终稿 | `proof_display_text(line)` / `Line.proof_state` | `final_text` 是否为空、`Line.text` 单独判断 | `final_text_set=True` 时空串也是有效终稿。 |
| 字符可视文本 | `proof_char_text.char_display_text()` | 无条件用 `token_text` | EngCut char bbox 中 token_text 可能是整词元信息，不等于单字显示文本。 |
| Proof 渲染单元 | `ProofAtom` | 原始 `Line.chars` 直接渲染 | ProofAtom 会标记 reliable/unreliable，是 UI 渲染输入，不是源事实。 |
| 字符索引 | `CharIndexService` 查询结果 | CharIndex 当作数据源 | 它是派生索引；错配行会被跳过，不能修复坏数据。 |
| 质量探针 | active probe sidecar | UI 显示假字 | pending fake_char 只在锚点仍匹配 true_char 时注入。 |
| proof 持久化 | `ProofChangeSet.line_refs` + `ProofPersistenceService` | 裸 `proof_saved` 名称 | `proof_saved` 是遗留信号名；真正语义是 proof changed, please persist。 |

## 三、近期关键修补前后逻辑

### 1. 保存身份：rowid -> uid

修改前：

- 保存页面时大量依赖 SQLite rowid。
- 删除再插入会导致 block/line/char rowid 抖动。
- stale rowid 或跨 project/跨 line 重复 uid 可能把别的对象搬过来。

修改后：

- Page/Block/Line/Char 都有稳定 `uid`。
- 全量保存使用 uid 恢复业务对象，rowid 只是存储定位。
- proof 增量保存增加 scoped 规则：`update_proof_lines(..., write_chars=True)` 只同步当前 line 的 char，不能按重复 uid 认领其他 line 的 char row。

当前边界：

- 全量保存仍允许业务对象跨父级移动。
- proof 增量保存不允许跨行移动 char，这是防止校对保存偷字框的硬边界。

### 2. OCR dispatch：裸 recognizable -> 统一 policy

修改前：

- 很多入口只看 `block.recognizable`。
- 公式、表格、图片等结构块可能被统计为待 OCR 或误进入文本 OCR 链。

修改后：

- 运行时统一走 `should_dispatch_to_text_ocr(block)`。
- page OCR 行分配时先检查 blocker，再分配到 text container。
- 结构块作为 blocker，不应进入正文 proof line。

当前边界：

- `recognizable` 已退役，OCR 入口不得恢复裸 bool 判断。

### 3. Paddle 路由：父文本块 + 子结构块

修改前：

- Paddle 子公式/表格框和父文本块关系不稳定。
- 行内公式画框后，容易被父文本块或 Hanwang 文本 OCR 吃掉。

修改后：

- Paddle geometry_records 中公式、表格、图片等结构框会绑定到父 parsing record。
- 生成 `_route_subblocks` 和 `_layout_line_routes`。
- Hanwang/CharOCR 前按 route 切文本片段，并把公式 carrier 插回 line。

当前边界：

- `_route_subblocks`、`_layout_line_routes` 仍是运行时 dict 计划，但不再写入 `Block.raw_payload/app_payload` 持久化模型。
- 这是当前最需要抽象成 `RoutingPlan/DispatchPlan` 的地方。

### 4. Line 文本：空终稿与 fake probe

修改前：

- `final_text=""` 被当成“未设置”，用户删空一行会在保存时恢复 OCR 文本。
- quality probe 的 fake_char 可能被当成真实文本写回。

修改后：

- `final_text_set` 区分“未设置”和“用户明确设为空串”。
- probe 显示空间编辑通过 `save_displayed_edit_result()` 映射回真实文本空间。
- probe observation 改变必须进入 `ProofChangeSet.probe_changed`。

当前边界：

- probe 是 sidecar 真值，必须随 proof change 持久化。
- UI 不能先改 probe 再等待文本提交，提交边界必须唯一。

### 5. Char/token_text：元信息与显示 carrier 分离

修改前：

- `token_text` 经常被当成每个 char 的显示文本。
- EngCut exact 里每个字符有单字母 bbox，但 token_text 保留整词，导致 `Guariglia` 被重建成多次重复整词。
- word/formula carrier 改字后，Line 文本和 char/token_text 容易分裂。

修改后：

- `proof_char_text` 统一判断 display carrier。
- `bbox_granularity == "char"` 且 `char.char` 是单字符时，显示/同步使用 `char.char`。
- `bbox_granularity in {"word", "formula"}` 或 `char.char` 多字符时，才把 token_text/char 当 carrier 显示 span。
- `ProofAtom` 在 `chars_display_text != line.display_text` 时降级为 unreliable token。

当前边界：

- `Line.chars` 不能完整表达 `proof_display_text(line)` 时，不允许 UI 假装字符事实可靠。

### 6. VProof/HProof 保存：位置映射 -> 编辑会话/冲突判断

修改前：

- VProof 保存按第 N 行位置写回模型。
- merge_pages、undo/redo、external refresh、debug filter、page switch 都可能绕过保存校验。
- HProof 用几何 key 判断同一行，bbox/order 轻微变化会丢 dirty editor。

修改后：

- VProof 使用 `_edit_session` 绑定 page key、line ids、slot map、loaded baseline。
- 保存前校验 editor 是否仍属于当前 page。
- block 分隔空行作为 separator slot，结构性换行会被拒绝。
- undo/redo 先完整校验，再写模型。
- HProof `_line_key` uid 优先，几何仅 fallback。
- 切页、debug filter 走 flush gate，冲突时阻断重建。

当前边界：

- HProof 和 VProof 仍是两套局部状态机。
- 下一阶段应抽出统一 `ProofEditSession`，避免继续在两个面板内重复补洞。

### 7. CharIndex：索引防污染，不是数据修复

修改前：

- 如果 Line 文本和 char 几何错配，错字可能进入“相同字索引”集合。
- 两字短行因为阈值问题可能 100% 错配仍入索引。

修改后：

- 长度不一致、大范围错配、两字全错配跳过。
- image cache 在 build 时刷新，避免同路径图片内容变化后使用旧图。

当前边界：

- CharIndex 只能防止错误继续展示。
- 已保存进项目文件的坏 `final_text/chars` 不会自动修复，需要单独诊断工具。

### 8. Worker 错误与进度状态

修改前：

- OCR 后台失败时，如果用户已切到横校/纵校，错误入口不会 finish 底部 OCR placeholder。
- placeholder active 后会吞掉后续 status bar 消息。

修改后：

- `_on_worker_error()` 入口先无条件 `finish()` OCR placeholder。

当前边界：

- 错误展示仍分 UI step：版面/OCR 页走状态栏，proof 页走弹框。

### 9. Percent fallback

修改前：

- 低置信、bbox 重叠的 `01` 可能被 overlap percent fallback 强制改成 `%`。
- recrop 失败或 recrop 返回原样时仍可能强改。

修改后：

- 强制 percent fallback 移除。
- 只有 recrop 明确返回 `%/％` 且原碎片形态符合 percent，才允许替换。
- `01` 保持为 `01`。

## 四、当前“真值/派生/兼容”分层

### 真值层

- `Page.display_image_path`：几何坐标对应的工作图。
- `Page.raw_layout_artifact`：Paddle VL1.6 原始版面事实。
- `Block.uid` / `Line.uid` / `Char.uid`：业务身份。
- `Block.block_type`：程序大类。
- `Block.source_label`：Paddle 或人工绑定的细标签。
- `Line.ocr_text`：OCR 原始文本。
- `proof_display_text(line)`：当前校对文本事实。
- `Line.chars`：字符/词/公式 carrier 几何事实，前提是与 display_text 可对齐。
- `quality_probe` sidecar：质量探针事实。

### 派生层

- `ProofAtom`：proof UI 渲染单元。
- `CharIndexService`：相同字/字符 gallery 索引。
- `_route_subblocks` / `_layout_line_routes`：当前路线计划的运行时 dict 形式。
- `ProofLineViewModel` / UI 状态标签：展示投影。
- `block.note`：提示/调试说明，不应参与关键判断。

### 兼容/过渡层

- `Line.text`：仍作为 OCR 行文本字段；新逻辑不应把它当唯一 proof 真值。
- `Block.app_payload`：退役空字段，仅保留 schema 边界；非空会被拒绝，不再迁移旧项目。
- `proof_saved` 信号名：名称像“已保存”，实际仍是“proof changed，需要持久化”的遗留信号。

## 五、当前仍不健康的职责边界

1. `Page/Block/Line/Char` 是大一统模型。
   - 同时承担 OCR 观察、人工终稿、UI 展示、存储 rowid、导出来源。
   - 后续应拆成 Observation / EditState / ViewModel / Persistence DTO。

2. `raw_payload` 仍是外部证据容器，`app_payload` 已退役。
   - vendor fact 和 app state 已分开，但 route plan 仍是运行时 dict。
   - 后续应抽 `PaddleArtifact`、`LayoutSnapshot`、`RoutingPlan`、`DispatchPlan`、`OcrRunResult`。

3. HProof/VProof 状态机重复。
   - VProof 有 edit session。
   - HProof 有 line key、flush gate、dirty/conflict 判断。
   - 两者应该共享 `ProofEditSession` 和 `SaveResult`，否则下次还会出现入口绕过。

4. UI 仍直接改领域对象。
   - LayoutPanel/HProof/VProof 仍会直接改 Page/Block/Line。
   - 后续应通过 `LayoutEditService`、`ProofEditService` 收口。

5. 已坏项目数据不会自动修复。
   - CharIndex 只会过滤错配，不会改项目文件。
   - 需要独立诊断工具扫描 `proof_display_text(line)`、`chars_display_text`、bbox、uid 重复、probe sidecar。

## 六、后续重构优先级

1. 保留当前补丁成果，不继续扩大局部补丁。
2. 先做 architecture ratchet，只防新增依赖恶化。
3. 引入 `LayoutEditCommand/LayoutEditService`，收口版面编辑入口。
4. 抽 `PaddleArtifact/LayoutSnapshot/RoutingPlan/DispatchPlan/OcrRunResult`，把 route dict 从 payload 中移出。
5. 抽统一 `ProofEditSession/ProofSaveResult`，让 HProof/VProof 共用保存、冲突、重建 gate。
6. 做项目诊断工具，专门检查并报告已持久化错配数据。

## 七、审查时的判断口诀

- 要判断“这个框是谁说的”：先看 Paddle raw，再看人工 binding，再看 route 派生。
- 要判断“这行该不该 OCR”：只看 `should_dispatch_to_text_ocr()`。
- 要判断“这行最终文本是什么”：只看 `proof_display_text(line)`。
- 要判断“这个字符框能不能信”：先比较 `chars_display_text(line.chars)` 与 `line.display_text`。
- 要判断“这个对象是谁”：uid 优先，rowid 只辅助存储。
- 要判断“是否需要保存”：看 `ProofChangeSet.needs_persist` 和 scoped `line_refs`。
- 要判断“索引为什么没有某个字”：CharIndex 可能因为文本/几何不一致主动跳过，这不是源数据消失。
