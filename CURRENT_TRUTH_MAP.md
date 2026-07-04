# OCR Process 当前真值地图

生成时间：2026-07-03

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
     - Paddle 版面事实列表，bbox 已归一到当前 Page 工作图坐标
     - 主要含 block_label、block_bbox、block_content 等
     - 通过 raw_ocr_artifact.raw_layout_records(page) 读取
     - route 子结构不再写入 records；`route_attachments_json` 单独保存由程序绑定出的公式/表格/图片子框
  -> NormalizedLayoutArtifact
     - 对外部版面事实的统一读模型
     - Paddle、未来矢量 PDF 等输入源都应先转成 LayoutRegion/LayoutSubregion
     - 读取时会把 raw records 与 route attachments 合并为 LayoutSubregion
  -> LayoutSnapshot
     - 当前程序采用的版面真值
     - 从 NormalizedLayoutArtifact 编译而来
     - LayoutEditService 的人工编辑会同步到外部 layout_snapshot_store
     - ProjectStore 加载项目后会从持久化 block 投影重建 layout_snapshot_store
  -> page.blocks
     - 当前仍供 UI/OCR/导出读取的运行投影
     - block_type 是程序大类
     - source_label 是 Paddle 细标签
     - 新导入/OCR 运行时块通过 raw_layout_artifact + origin.raw_index 回溯外部事实
     - app_payload 已从 active Block 模型删除；旧 SQLite 列只用于读取边界校验
  -> layout_projection 边界
     - Page.blocks 仍是当前物理存储，但业务层不再直接把它当领域模型入口
     - 非 UI 的 OCR observation、dispatch、ProjectStore、Hanwang、导出、诊断等模块通过 page_layout_blocks/replace_page_layout_blocks 等 helper 访问当前投影
     - LayoutSnapshot 已由 API 编译、LayoutEditService 编辑和 ProjectStore 加载同步维护；后续替换点集中在 projection 边界和持久化

人工版面编辑
  -> LayoutEditCommand / LayoutEditService
     - UI 提交 create/delete/change_kind/merge/geometry_update 命令
     - create/delete/change_kind/merge/geometry_update/restore_blocks 是当前用户编辑写入口
     - 负责写 Block 当前 bbox/type/source_label/ocr_policy
     - Block.bbox/order 的运行时改写通过 layout_block_state helper 写入
     - Block.block_type 的运行时改写通过 layout_block_state helper 写入
     - Block.source 的人工来源标记通过 layout_block_state helper 写入
     - Block.source_label 的运行时改写通过 layout_block_state helper 写入
     - Block.ocr_policy 的策略落点通过 layout_block_state helper 写入
     - 负责写 PaddleBinding、OCR invalidation、layout_edit_events
     - LayoutPanel 只保留用户意图采集、选区、撤销快照和 overlay 展示
  -> LayoutOverlayService
     - 负责读取 Page.raw_layout_artifact 经归一化后的只读 overlay
     - 负责把 Paddle inline_formula overlay 提升为可编辑公式 Block
     - LayoutPanel 不直接解析 raw_layout_records 或 Paddle route dict

路由构建
  -> route attachments / transient route records / RoutingPlan
     - 由 Paddle 的公式、表格、图片等结构框推导出文本切片路线
     - route attachments 按 raw record index 单独持久化，不再污染 raw layout records
     - `layout_records_with_route_attachments(page)` 只在喂给 route compiler 时合成临时 dict
     - 作用是让正文块进入 Hanwang/CharOCR 前扣除或插入公式等结构段

OCR Hanwang/CharOCR
  -> OcrRunResult / OcrProgress
     - 本轮 OCR 的页结果、失败块和进度事件
     - OcrPipeline 只产出该 typed contract，不再在 pipeline 文件内定义结果模型
  -> ocr_observation 边界 + ocr_observation_store（当前仍投影到 Block.lines）
     - Line 是可校对文本行
     - Line.bbox 是行几何
     - Line.bbox 的运行时写入必须通过 `set_ocr_line_bbox()`，不能由 pipeline/Hanwang/校对补框链路裸写
     - OCR 文本行必须通过 `create_ocr_text_line()` 构造，不能由各 OCR producer 手写 `Line(text=..., ocr_text=...)`
     - 业务代码必须通过 app.models.ocr_observation 读写，不能继续裸读写 block.lines
     - ocr_observation_store 是运行时 OCR 行事实边界，Block.lines 只作为当前 UI/存储运行投影
     - 生产代码不得用 `Block(lines=...)` 构造 OCR 行观察；必须先建 Block，再通过 `replace_block_ocr_lines()` 写入
     - 行的 page/block/line index 查找通过 OcrLineOccurrence helper 表达，不再由 proof 服务私有推导
     - ProjectStore 也通过该边界读写，`Block.lines` 只剩运行投影和 observation helper 内部同步目标
  -> ocr_character_observation 边界 + ocr_character_observation_store（当前仍投影到 Line.chars）
     - Char 是字符、词、公式 carrier 的 OCR 几何观察
     - 非 UI、非 OCR producer 的 core/service/export/controller 代码通过 line_ocr_chars/replace_line_ocr_chars 等 helper 访问
     - 生产代码不得用 `Line(chars=...)` 构造 OCR 字符观察；OCR IR 和 Hanwang 投影都先建 Line，再通过 `replace_line_ocr_chars()` 写入
     - Char.bbox 和局部 span 替换必须通过 `set_ocr_char_bbox()` / `replace_line_ocr_char_span()` 写入
     - ProofAtom、CharIndex、ProjectStore、Export IR、ProofCrop、QualityProbe 等不再直接依赖 `Line.chars` 物理字段
     - ocr_character_observation_store 是运行时字符/词/公式 carrier 事实边界，Line.chars 只作为当前 UI/存储运行投影

校对 HProof/VProof
  -> external ProofLineState store
     - 人工校对后的文本事实
     - proof_display_text(line) 是统一显示文本入口
  -> ocr_character_observation_store / Line.chars 运行投影
     - 等长或可对齐编辑时同步字符事实
     - 不能同步时，下游 ProofAtom/CharIndex 必须降级或跳过
  -> ProofChangeSet
     - 描述本次 proof 改动范围与类型

持久化 ProjectStore
  -> SQLite
     - id 是 rowid，仅是存储实现
     - uid 是业务身份
     - proof 增量保存必须按 scoped line 更新，不能全项目覆盖
     - 保存路径只写当前模型事实；不得清空 `Block.lines`、不得清理 route 时顺手改 OCR invalidation
     - 旧 `raw_payload_json/app_payload_json` 已从当前 schema 退出；schema 23 迁移只删除空旧列，非空旧 payload 会拒绝打开

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
| 外部版面证据 | `Page.raw_layout_artifact` + `Block.origin.raw_index` | `block.note`、旧 `block.app_payload`、旧 `block.raw_payload` | active `Block` 不再携带 vendor JSON；artifact 里的 bbox 已归一到工作图坐标；page 级原始列表不再挂 `ppvl_parsing_res_list`。 |
| 外部版面归一化视图 | `NormalizedLayoutArtifact` / `LayoutRegion` / `LayoutSubregion` | 业务服务直接读 Paddle raw dict | Paddle、未来矢量 PDF 等输入源应先归一化，再进入 overlay、manual binding、routing。 |
| 当前版面真值 | `app.models.layout_snapshot.LayoutSnapshot` + `layout_snapshot_store` | `Page.blocks` 直接当导入真值、service 内部临时 DTO | API 版面分析从归一化 artifact 编译 snapshot；人工编辑由 `LayoutEditService` 同步 snapshot store；项目加载从持久化 block 投影重建 snapshot store；`Page.blocks` 仍是当前运行投影。 |
| 当前版面投影访问 | `app.models.layout_projection` | 非 UI 业务层直接 `page.blocks` | `Page.blocks` 暂时还是物理运行对象；非 UI 读取/替换当前投影必须走 projection helper，避免把物理对象树继续扩散成领域模型。 |
| 当前运行版面投影 | `Page.blocks` | 未来新输入源直接写 `Page.blocks` | `Page.blocks` 暂时还是 UI/OCR/导出运行对象，但不应作为矢量 PDF 等新入口的适配目标。 |
| 块几何/顺序 | `Block.bbox` / `Block.order` / `set_layout_block_bbox()` / `set_layout_block_order()` | 各模块直接写 `block.bbox/order`、把 bbox/order 当身份 | bbox/order 是当前投影状态，可随编辑、缩放和 OCR clamp 改变；运行时写入必须经 layout_block_state helper。 |
| 程序派生状态 | typed 字段：`origin`、`paddle_binding`、`ocr_invalidated_reason`、`ocr_audit`、`table_text_layer_cells`、`layout_edit_events` | 旧 `block.raw_payload`、旧 `block.app_payload` | `raw_payload/app_payload` 已从 active model 删除；旧 SQLite 列非空会被存储校验拒绝。 |
| 块提示/调试说明 | `Block.note` / `set_layout_block_note()` / `append_layout_block_note_once()` | 各模块直接写 `block.note`、用 note 做关键判断 | `block.note` 只保留为提示/调试说明，不参与关键路由或导出真值；运行时写入必须经 layout_block_state helper。 |
| 块大类 | `Block.block_type` / `set_layout_block_type()` | Paddle 原始 label 直接判断、各模块直接写 `block.block_type` | UI 和导出看大类；运行时改写必须经 layout_block_state helper。 |
| Paddle 细标签 | `Block.source_label` / raw label / `set_layout_block_source_label()` | `Block.block_type` 反推、各模块直接写 `block.source_label` | 页眉、脚注、公式序号等细分来自 source_label；运行时改写必须经 layout_block_state helper。 |
| 块来源/编辑态 | `app.models.layout_block_state` helper | 各模块直接比较或写入 `Block.source` | `Block.source` 仍是过渡字段，但人工编辑来源、导出 origin 等语义的解释和用户编辑写入都集中到 helper。 |
| 是否进文本 OCR | `DispatchPlan` / `should_dispatch_to_text_ocr(block)` / `Block.ocr_policy` / `set_layout_block_ocr_policy()` | `block_type/source_label/raw_payload` 的组合猜测、`Block` 构造副作用、页级流程自己遍历 `page.blocks`、各模块直接写 `block.ocr_policy` | 单块策略由 policy 决定；页级 OCR 入口统一先构建 `DispatchPlan`，公式、表格、图片作为 blocker 不进入正文 proof line；策略计算仍在规则层，写回 Block 必须走 layout_block_state helper。 |
| 版面编辑写入口 | `LayoutEditCommand` + `LayoutEditService.apply()` | `LayoutPanel` 内部私有 helper、撤销直接替换 `page.blocks`、或直调散参 mutation 方法 | 用户新增、删除、改类型、合并、调框、撤销恢复统一表达为命令，服务内写当前 Block 和审计事件。 |
| raw overlay 展示/提升 | `LayoutOverlayService` + `NormalizedLayoutArtifact` | `LayoutPanel` 直接读 `raw_layout_records`、`_route_subblocks` | UI 只消费服务输出的 overlay 或新建的 inline formula Block；overlay 服务读取归一化 region，不直接解析 Paddle route dict。 |
| 页面流程状态 | `app.models.page_state` helper 写入 `Page.status/error_message/ocr_invalidated_reason` | Controller/UI/Worker/Pipeline 直接写 `Page.status/error_message/ocr_invalidated_reason` | 当前仍是单字段 `Page.status`，但状态流转、后台错误消息和 OCR invalidation 入口已收口，后续拆状态机从该 helper 切入。 |
| OCR 文本观察 | `app.models.ocr_text_observation` helpers + `ocr_text_observation_store`：`create_ocr_text_line()`、`line_ocr_text_observation()`、`line_ocr_text()`、`line_ocr_review_flags()`、`line_has_ocr_review_flag()` | 各 OCR producer 直接 `Line(text=..., ocr_text=...)` 或业务层直接判断 `Line.text` / `Line.ocr_text` / `Line.review_flags` | `text/ocr_text/confidence/review_flags` 是 OCR 文本观察；运行时事实已进入 text observation store，同时投影到 `Line` 供当前 UI/存储读取。 |
| OCR 行观察汇总 | `app.models.ocr_observation` helpers + `ocr_observation_store`：`block_avg_confidence`、`page_ocr_line_count`、`project_ocr_line_count`、`page_has_ocr_result`、`replace_block_ocr_lines`、`set_ocr_line_bbox` 等 | `Block.avg_confidence`、`Page.total_lines`、`Page.has_ocr_result`、`OcrProject.total_lines`、`OcrProject.has_any_ocr_result`、`OcrProject.all_pages_ocr_done`、生产代码 `Block(lines=...)`、各模块直接 `line.bbox = ...` | OCR 行事实已进入运行时 observation store，同时投影到 `Block.lines` 供当前 UI/存储读取；Block/Page/Project 模型不再负责解释置信度、行数、是否有 OCR 或项目 OCR 完成状态。 |
| OCR 字符观察 | `app.models.ocr_character_observation` helpers + `ocr_character_observation_store`：`line_ocr_chars`、`replace_line_ocr_chars`、`replace_line_ocr_char_span`、`set_ocr_char_bbox`、`iter_line_ocr_char_occurrences` 等 | 非 UI 业务层直接 `Line.chars`、生产代码 `Line(chars=...)`、各模块直接 `line.chars[...] = ...` 或 `char.bbox = ...` | 字符/词/公式 carrier 已进入运行时 observation store，同时投影到 `Line.chars` 供当前 UI/存储读取；ProofAtom、CharIndex、ProjectStore、Export IR、ProofCrop、QualityProbe、OCR IR/Hanwang 投影等不再直接读取、构造或改写物理字段。 |
| 校对终稿 | `proof_display_text(line)` / external `ProofLineState` store | `final_text` 是否为空、`Line.text` 单独判断、`Line.proof_state` | `final_text_set=True` 时空串也是有效终稿；active `Line` 不再携带 proof_state 字段。 |
| 字符可视文本 | `proof_char_text.char_display_text()` | 无条件用 `token_text` | EngCut char bbox 中 token_text 可能是整词元信息，不等于单字显示文本。 |
| Proof 渲染单元 | `ProofAtom` | 原始 `Line.chars` 直接渲染 | ProofAtom 会标记 reliable/unreliable，是 UI 渲染输入，不是源事实。 |
| 字符索引 | `CharIndexService` 查询结果 | CharIndex 当作数据源 | 它是派生索引；错配行会被跳过，不能修复坏数据。 |
| 质量探针 | active probe sidecar | UI 显示假字 | pending fake_char 只在锚点仍匹配 true_char 时注入。 |
| 质量探针密度设置 | `quality_probe_sand_count` + `quality_probe_sand_unit_chars` | 旧 `quality_probe_target_ratio` | UI/AppConfig 只保留“每 N 字投放几个沙子”一个口径。 |
| proof 持久化 | `proof_changed(ProofChangeSet)` + `ProofChangeSet.line_refs` + `ProofPersistenceService` | 裸 bool 保存信号、无 scope 自动保存 | proof 信号语义是“校对事实已变更，需要持久化”，具体持久化范围由 `ProofChangeSet` 描述。 |
| proof 写入口 | `ProofEditService` / `proof_line_mutation` | `Line.set_proof_text()` / `Line.set_proof_status()` | `Line` 模型不再持有 proof 写方法，后续写状态必须走显式 helper/service。 |
| proof 重建门禁 | `proof_rebuild_gate` | HProof/VProof 各自解释保存状态 | 视图销毁/重建前是否允许继续，由共享 gate 根据 editor state 与 `ProofEditStatus` 判断。 |

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
- 页级 OCR 入口先构建 `DispatchPlan`，统一暴露 text blocks 和 blockers。
- page OCR 行分配时先检查 blocker，再分配到 text container。
- 结构块作为 blocker，不应进入正文 proof line。
- `Block` 模型不再在 `__post_init__` 中根据 `source_label/block_type` 自动改写 policy；默认策略必须由 importer/UI/service 显式调用 dispatch policy 推导。

当前边界：

- `recognizable` 已退役，OCR 入口不得恢复裸 bool 判断。
- `Block` 构造不得恢复隐式 OCR policy 推导。
- `Block.ocr_policy` 的生产流程写入不得恢复散落赋值，必须经 `app.models.layout_block_state.set_layout_block_ocr_policy()`。
- 页级 OCR 统计、图像读取失败记录、PP-OCRv5 行归属和 Hanwang prepass hint 复用都应读取同一个 `DispatchPlan`。
- Page/Project 模型不得恢复 `text_blocks`、`text_ocr_blocks`、`has_unrecognized_blocks` 这类策略 property；统计和导出状态由服务读取 `DispatchPlan`。
- Block/Page/Project 模型不得恢复 `avg_confidence`、`total_lines`、`has_ocr_result`、`has_any_ocr_result`、`all_pages_ocr_done` 这类 OCR observation summary property；这些读取统一走 `ocr_observation` helper。
- `Block.source` 的人工编辑/导出来源语义不得在 Hanwang、Export、BlockAttributes 内分散解释；LayoutEditService 也不得直接写 `BlockSource.USER_EDITED/MANUAL_DRAW`，必须经 `app.models.layout_block_state` helper。

### 3. Paddle 路由：父文本块 + 子结构块

修改前：

- Paddle 子公式/表格框和父文本块关系不稳定。
- 行内公式画框后，容易被父文本块或 Hanwang 文本 OCR 吃掉。

修改后：

- Paddle geometry_records 中公式、表格、图片等结构框会绑定到父 parsing record。
- route 子结构绑定进入 `RawOcrArtifact.route_attachments` / `route_attachments_json`，raw `records` 保持纯外部事实。
- 生成 typed `RoutingPlan`；`_route_subblocks` 只作为临时 route input record 的字段，`_layout_line_routes` 只作为运行时 cache 形态。
- Hanwang/CharOCR 前按 route 切文本片段，并把公式 carrier 插回 line。

当前边界：

- `RoutingPlan` / `RoutingLine` / `RoutingSegment` / `TextSliceRoute` 已作为生产和读取侧 typed contract；overlay 和 Hanwang 文本切片不应直接消费 route dict。
- raw artifact records 不得持久化 `_route_subblocks`；子结构绑定必须走 route attachments。
- `_route_subblocks` 只允许在 `layout_records_with_route_attachments()` / `layout_region_route_record()` 合成的临时 dict 中出现。
- `_layout_line_routes` 只允许作为 OCR 运行时 cache，不应成为业务读取入口或持久化事实。
- `DispatchPlan` 已收口页级文字 OCR 调度；`OcrRunResult` 已收口 OCR 运行结果和进度 contract；后续还需要加入列模型/阅读顺序模型承接双栏。

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
- 已保存进项目文件的坏 `final_text/chars` 不会自动修复。
- 只读诊断入口已落地：`app.services.project_diagnostics.diagnose_project()` 和 `scripts/diagnose_project.py` 会报告 proof 文本/char carrier 错配、重复 uid、缺失/非法 bbox、quality-probe stale anchor；它只输出报告，不做修复。

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
- `Page.raw_layout_artifact`：Paddle VL1.6 版面证据包，bbox 已归一到当前工作图坐标。
- `RawOcrArtifact.route_attachments` / `route_attachments_json`：程序从 Paddle geometry records 绑定出的子结构 route 事实，按父 raw record index 存储。
- `NormalizedLayoutArtifact`：外部版面事实的统一读模型。
- `LayoutSnapshot`：当前采用的版面真值 contract，定义在 `app.models.layout_snapshot`；API 版面分析、LayoutEditService 人工编辑和 ProjectStore 加载已同步到 `layout_snapshot_store`，再投影到当前 `Page.blocks` 运行投影。
- `Block.uid` / `Line.uid` / `Char.uid`：业务身份。
- `Block.bbox/order`：当前版面投影状态，不是身份；运行时写入必须经 `app.models.layout_block_state`。
- `Block.block_type`：程序大类。
- `Block.source_label`：Paddle 或人工绑定的细标签。
- `Page.status/error_message/ocr_invalidated_reason`：当前页面流程状态字段；Controller、Worker、Pipeline 写入必须经 `app.models.page_state`。
- `ocr_observation`：OCR 行观察访问边界；当前内部使用 `ocr_observation_store` 作为运行时事实并同步投影到 `Block.lines`，ProjectStore、业务代码和 proof 行 occurrence 查找都从这里读写。
- `line_text_contract(line).ocr_text` / `proof_ocr_text(line)`：OCR 原始文本读取口径。
- `proof_display_text(line)`：当前校对文本事实。
- `Line.text` / `Line.ocr_text` / `Line.review_flags`：OCR 文本观察旧投影；运行时事实由 `ocr_text_observation_store` 持有。
- `Line.chars`：字符/词/公式 carrier 旧投影，前提是与 display_text 可对齐；运行时事实由 `ocr_character_observation_store` 持有。
- `quality_probe` sidecar：质量探针事实。

### 派生层

- `ProofAtom`：proof UI 渲染单元。
- `CharIndexService`：相同字/字符 gallery 索引。
- `RoutingPlan` / `RoutingLine` / `RoutingSegment` / `TextSliceRoute`：当前路线计划的 typed 生产/读取口径。
- `layout_records_with_route_attachments()` / `layout_region_route_record()`：为现有 route compiler 合成的临时输入 dict。
- `_route_subblocks` / `_layout_line_routes`：当前路线计划的临时输入/cache 字段；不得进入 raw layout records。
- `ProofLineViewModel` / UI 状态标签：展示投影。
- `block.note`：提示/调试说明，不应参与关键判断；写入必须经 `app.models.layout_block_state`。

### 受控过渡投影/退出边界

- `Page.blocks`：当前采用版面运行投影，不是新输入源的事实入口；API 版面分析、人工编辑和加载路径先同步 `LayoutSnapshot`，再通过 `layout_projection` 投影到当前 block tree。
- `block.raw_payload_json` / `block.app_payload_json` SQLite 旧列：已从当前 schema 和保存/加载 SQL 退出；schema 23 迁移只删除空旧列，非空旧 payload 会拒绝打开。
- `Line.text`：仍作为底层 OCR 行文本字段；读取口径已收口到 `line_text_contract()`，新逻辑不应直接解释它。
- 旧 `line.proof_state`：已退出兼容路径；runtime store 不再消费该 attr，active Line 出现它会被视为模型污染。
- `Block.lines`：仍是当前 UI/存储运行投影；业务代码、ProjectStore 和 proof 行定位已改为通过 `app.models.ocr_observation` 访问，运行时事实在 `ocr_observation_store`。

## 五、当前仍不健康的职责边界

1. `Page/Block/Line/Char` 是大一统模型。
   - 同时承担 OCR 观察、人工终稿、UI 展示、存储 rowid、导出来源。
   - 后续应拆成 Observation / EditState / ViewModel / Persistence DTO。
   - `Block.lines` 尚未物理外置，但直接访问已经被架构测试约束到模型字段和 `ocr_observation` 边界。

2. `raw_payload` 已从 active `Block` 模型退出，仅保留旧 SQLite 列拒绝边界。
- Paddle vendor evidence 和 app state 已分开；route plan 已有 typed producer，运行时 dict 仍保留在子结构输入和 cache serialization。
- `ProjectStore._save_block()` 固定写空 payload，并有架构守卫防止恢复保存时清 route、清 lines、写 invalidation 的旧副作用。
- `NormalizedLayoutArtifact` 已作为读取侧归一化 contract；`LayoutOverlayService` 和 `PaddleArtifactIndex.from_page()` 不再直接遍历 `raw_layout_records`。
- `RoutingPlan` 已作为 overlay 与 Hanwang route 消费端的读取边界；`paddle_line_routing.build_layout_routing_plan()` 是 typed 生产入口，route dict 只作为临时序列化桥。
- `DispatchPlan` 已作为页级文字 OCR 调度边界；`OcrRunResult` 已作为 OCR 运行结果 contract；`LayoutSnapshot` model contract 已作为 API 版面分析和人工编辑的当前版面真值边界，`Page.blocks` 仍是当前运行投影。

3. HProof/VProof 状态机重复。
   - proof 写入和保存状态已经统一到 `ProofEditService` / `ProofEditStatus`、scoped `ProofChangeSet` 和 `ProofPersistenceService`。
   - HProof 有 `HProofRuntimeSession` / `HProofLineEditSession`。
   - VProof 有 `VProofOccurrenceSession` / `ProofReferenceContext` / named text slots。
   - `proof_rebuild_gate` 已统一“editor state + 保存状态是否允许视图重建”的第一层决策；HProof 重建前 dirty/conflict 判断已接入该 gate。
   - HProof/VProof 外部刷新 pending 身份合并已共用 `ProofExternalRefreshQueue`。
   - HProof/VProof 外部刷新已共用 `ProofExternalRefreshPlan`；HProof 使用 touched projection indexes，VProof 使用 affected page keys / current-page reload flag。
   - VProof 当前页 reference context reload 已走 `proof_rebuild_gate_for_reference_context()`；该文本区是只读参考投影，直接文本差异不会被当成可持久化 proof 编辑。
   - 未收口的是 HProof/VProof external refresh 的 UI 执行动作仍按视图形态分开；后续应继续抽共享执行 adapter，而不是再补 UI 单点判断。

4. UI 仍有局部视图状态，但版面对象写入已收口。
   - LayoutPanel 的用户版面编辑入口已迁移到 `LayoutEditCommand` + `LayoutEditService.apply()`。
   - LayoutPanel 的 Paddle raw overlay 解析已迁移到 `LayoutOverlayService`。
   - LayoutPanel 仍维护 undo 快照，但恢复 `page.blocks` 已通过 `LayoutEditCommand.restore_blocks` 进入服务层。
   - LayoutPanel 仍会通过服务生成临时可编辑 inline formula Block。
   - HProof/VProof 的文本写入已走 `ProofEditService` / `ProofChangeSet`；HProof 重建 gate 已收口，HProof/VProof 外部刷新 pending 队列和 plan 已共享，VProof reference reload 已接入 shared gate。
   - 后续重点不是恢复旧 helper，而是让人工编辑直接作用于 `LayoutSnapshot`，并抽出共享 `ProofEditSession`。

5. 已坏项目数据不会自动修复。
   - CharIndex 只会过滤错配，不会改项目文件。
   - `project_diagnostics` 已提供只读扫描：`proof_display_text(line)`、`chars_display_text`、bbox、uid 重复、probe sidecar；修复仍需单独工具或人工确认。

## 六、后续重构优先级

1. 保留当前补丁成果，不继续扩大局部补丁。
2. architecture ratchet 已落地：`architecture_baseline.json` + `tests/test_architecture_import_ratchet.py` 只阻止新增包级违规依赖，不要求一次清空历史债。
3. `DispatchPlan` / `OcrRunResult` 已落地；继续把 OCR observation 物理存储、失败审计和 route dict 从 `Page/Block/Line` 子结构/cache 层压缩出去。
4. `LayoutEditCommand/LayoutEditResult` 已落地；命令执行后已同步 `LayoutSnapshot` store。下一步是让持久化和新输入源读取 snapshot，而不是把 `Page.blocks` 当最终事实。
5. 扩大 `proof_rebuild_gate` 到 HProof/VProof external refresh 的完整共享采集层。
6. `project_diagnostics` 已能只读报告持久化错配数据；后续若要自动修复，应新增独立 repair 工具，不应塞回 CharIndex/HProof/VProof。

## 七、审查时的判断口诀

- 要判断“这个框是谁说的”：先看 Paddle raw，再看人工 binding，再看 route 派生。
- 要判断“这一页哪些块该 OCR”：看 `DispatchPlan`；要判断“单块策略是什么”：看 `should_dispatch_to_text_ocr()`。
- 要判断“这行最终文本是什么”：只看 `proof_display_text(line)`。
- 要判断“这个字符框能不能信”：先通过 `line_ocr_chars(line)` 读取字符观察，再比较 carrier 文本与 `proof_display_text(line)`。
- 要判断“这个对象是谁”：uid 优先，rowid 只辅助存储。
- 要判断“是否需要保存”：看 `ProofChangeSet.needs_persist` 和 scoped `line_refs`。
- 要判断“索引为什么没有某个字”：CharIndex 可能因为文本/几何不一致主动跳过，这不是源数据消失。
