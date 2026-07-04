# 兼容代码退出计划

本文件用于约束过渡期兼容层，避免兼容代码长期变成第二套业务逻辑。

## 原则

1. 兼容层必须有明确替代模型、调用方迁移路径和删除条件。
2. 新业务逻辑只能依赖新入口；兼容入口只能委托新入口，不能再生长判断分支。
3. 每次新增兼容入口，都要在本文件登记，否则视为技术债未受控。
4. 删除兼容层前必须有测试证明：旧入口无人调用，或旧入口只剩迁移测试覆盖。

## 当前兼容点

当前没有登记中的兼容入口。新兼容入口必须先补充到本节，再进入实现。

## 已清理

| 已清理入口 | 替代入口 | 清理说明 |
| --- | --- | --- |
| `OcrProject.ocr_completed` | `has_any_ocr_result` / `all_pages_ocr_done` | 删除兼容 property；controller/UI/tests 改用显式语义。 |
| `WorkflowController.ocr_completed` | `WorkflowController.has_any_ocr_result` | 删除旧 accessor，避免打开项目时继续传播模糊命名。 |
| `Page.recognizable_blocks` | `DispatchPlan` / `count_text_ocr_blocks()` | 删除兼容 property；OCR 入口统计改用统一 dispatch 结果。 |
| `Page.text_blocks` / `Page.text_ocr_blocks` / `OcrProject.has_unrecognized_blocks` | `DispatchPlan` / `iter_text_ocr_blocks()` / export summary service | 删除模型层策略 property；Page/Project 不再解释 OCR 调度策略或未识别块。 |
| `Block.avg_confidence` / `Page.total_lines` / `Page.has_ocr_result` / `OcrProject.total_lines` / `OcrProject.has_any_ocr_result` / `OcrProject.all_pages_ocr_done` | `app.models.ocr_observation` summary helpers | 删除模型层 OCR observation summary property；置信度、行数和是否已有 OCR 结果由 observation 边界解释。 |
| `LayoutAnalyzer._extract_*_from_record()` | `paddle_layout_schema.py` | 删除旧私有 shim；字段解析直接走 schema adapter。 |
| `WorkflowController._page_gate_info()` | `workflow_state.page_gate_info()` | 删除 controller 转发，调用点直接依赖 typed gate helper。 |
| `app.core.ocr_config` | `app.core.app_config.get_config/update_config` | 删除旧配置桥模块，生产和测试导入统一到 AppConfig 入口。 |
| `Line.text <-> final_text` 双向镜像 | external `ProofLineState` store / `proof_display_text(line)` / `Line.ocr_text` | 删除 `__setattr__` 镜像；校对和导出读取 ProofLineState/display helper，`text` 保留为 OCR 行文本字段；active `Line` 不再携带 proof_state。 |
| 旧 `line.proof_state` runtime attr 兼容消费 | external `ProofLineState` store | `proof_line_state_store` 不再消费或清理旧 attr；active Line 出现 `proof_state` 会被视为模型污染并报错。 |
| `quality_probe_target_ratio` AppConfig 旧比例键 | `quality_probe_sand_count` + `quality_probe_sand_unit_chars` | 设置路径只保留“每 N 字投放几个沙子”的密度模型；程序化 `SamplerConfig(target_ratio=...)` 仍可用于测试，但不再从 AppConfig 读取。 |
| `Block.raw_payload/app_payload` app-owned keys | typed 字段：`paddle_binding` / `ocr_invalidated_reason` / `origin` / `ocr_audit` / `table_text_layer_cells` / `layout_edit_events` | `Block.raw_payload/app_payload` 已从 active model 删除；旧 `raw_payload_json/app_payload_json` 不再迁移，非空会被当前 schema 校验拒绝。 |
| 裸 `Page.status` 写入 | `app.models.page_state` | Controller/Store 不再直接写 `Page.status`；当前仍保留单字段，后续状态机拆分从 helper 入口替换。 |
| UI/test hidden compatibility fields/signals | explicit state/table accessors | 删除 `QualityStatsDialog._rate_lbl`、兼容 `_table` property、`NavRail.account_clicked` 空信号；测试改读 typed state / `detail_table()`。 |
| `LayoutPanel` 版面编辑私有业务 helper / 散参 mutation 调用 / 撤销直写 `page.blocks` | `LayoutEditCommand` + `LayoutEditService.apply()` | 删除 `_apply_subtype_to_block`、`_merge_blocks_into_bbox`、`_bind_manual_block_to_paddle`、`_update_existing_manual_binding_bbox` 等 UI 内业务写入口；新增、删除、改类型、合并、调框、撤销恢复统一经命令入口写 typed state 和 layout edit event。 |
| `LayoutPanel` 直接解析 Paddle raw overlay | `LayoutOverlayService` | 删除 UI 内 `raw_layout_records`、`ROUTE_SUBBLOCKS_FIELD`、`bbox_from_variant` 等 raw artifact 解析；只读 overlay 和 inline formula 提升由服务统一产出。 |
| Hanwang/Export/BlockAttributes/LayoutEditService 分散解释或写入 `Block.source` | `app.models.layout_block_state` | 人工编辑来源、导出 origin、路由手工结构判断、用户编辑来源标记集中到 helper；`Block.source` 字段仍保留为后续迁移入口。 |
| 非 UI 业务层直接读写 `Page.blocks` | `app.models.layout_projection` | 当前 `Page.blocks` 仍是物理运行投影；OCR observation、dispatch、ProjectStore、Hanwang、导出、诊断等模块经 projection helper 访问，避免对象树继续扩散为领域模型。 |
| 非 UI/非 OCR producer 业务层直接读写 `Line.chars` | `app.models.ocr_character_observation` | 当前 `Line.chars` 是旧 UI/存储投影；运行时字符事实在 `ocr_character_observation_store`；ProofAtom、CharIndex、ProjectStore、Export IR、ProofCrop、QualityProbe 等经 character observation helper 访问；运行时 span 替换和 `Char.bbox` 写入也经该边界。 |

## 删除顺序

### Phase 1: 冻结兼容层

- 所有新代码禁止直接新增裸兼容字段。
- app-owned 状态必须使用 typed 字段；禁止写入 `raw_payload` 或旧 `app_payload_json`。
- Paddle 返回字段必须先经过 `paddle_layout_schema.py`。
- OCR dispatch 单块策略必须先经过 `ocr_dispatch_policy.py`；页级 OCR 调度必须先经过 `DispatchPlan`。
- `Block` 模型不得在构造时根据外部 label 自动推导 OCR policy。

完成状态：已完成；已登记兼容入口均已删除或收口为正式模型字段。

### Phase 2: 调用方迁移

- `recognizable_blocks` / `text_blocks` / `text_ocr_blocks` 调用点迁移到 `DispatchPlan`。已完成。
- `Page/OcrProject` OCR 行数与 OCR 结果状态调用点迁移到 `ocr_observation` summary helpers。已完成。
- `ocr_completed` 调用点迁移到 `has_any_ocr_result` 或 `all_pages_ocr_done`。已完成。
- `Line.text` 写入点收口到 OCR 源字段；proof 改动通过 `ProofEditService` / `proof_line_mutation` 写入外部 `ProofLineState` store；OCR 文本读取统一经 `line_text_contract()` / `proof_ocr_text()`。已完成。
- controller 中业务 gate 直接调用 `workflow_state.page_gate_info()`。已完成。

完成状态：已完成。

### Phase 3: 数据模型替换

- `paddle_binding` 已升为 `Block.paddle_binding` typed state。
- `_route_subblocks` / `_layout_line_routes` 已限制为运行时 route dict，不进入 `Block.raw_payload` 或旧 `app_payload_json` 持久化模型。
- `ocr_text_invalidated` 已升为 `Block.ocr_invalidated_reason`。
- active `Block` 不再携带 `raw_payload`；新导入和 Hanwang OCR 后重建的 block 通过 `Page.raw_layout_artifact` + `Block.origin.raw_index` 读取原始事实。
- `Block.app_payload` 已从 active model 删除。
- `Block.lines` 的业务访问、ProjectStore 读写、proof 行定位和 OCR 生产投影已迁移到 `app.models.ocr_observation`；生产代码不再用 `Block(lines=...)` 构造 OCR 行观察；`Line.bbox` 运行时写入也经 `set_ocr_line_bbox()`；运行时事实已进入 `ocr_observation_store`，当前字段只保留旧 UI/存储投影。
- `Line.chars` 的业务访问、ProjectStore 读写、ProofAtom 构建、CharIndex 构建、Export IR、ProofCrop 补框、OCR IR 投影和 Hanwang 投影已迁移到 `app.models.ocr_character_observation`；生产代码不再用 `Line(chars=...)` 构造 OCR 字符观察；`Char.bbox` 和 span 替换也经该边界；运行时事实已进入 `ocr_character_observation_store`，当前字段只保留旧 UI/存储投影。
- `Line.text/ocr_text` 的 OCR 生产写入口已收口到 `app.models.ocr_text_observation.create_ocr_text_line()`；OCR IR、Hanwang、fake/local OCR、Paddle manual binding 不再手写 `Line(text=..., ocr_text=...)`。
- UI 字符显示已收口到 `proof_char_text.char_display_text()`，避免直接把 `Char.token_text` 当单字符显示文本。
- `Page.status/error_message/ocr_invalidated_reason` 的写入口已收口到 `app.models.page_state`；LayoutWorker/OcrPipeline 不再直接写后台错误消息。
- `LayoutPanel` 用户版面编辑入口已收口到 `LayoutEditCommand` + `LayoutEditService.apply()`；UI 只负责采集用户动作、维护撤销/选择态和刷新画布。
- `LayoutPanel` raw overlay 解析已收口到 `LayoutOverlayService`；UI 不再直接读取 Paddle raw dict 或 route dict。
- `Block.bbox/order` 的生产运行时写入已收口到 `app.models.layout_block_state`；bbox/order 不得作为业务身份。
- `Block.block_type` 的生产运行时写入已收口到 `app.models.layout_block_state.set_layout_block_type()`；导入/投影构造可传初始值，运行时不得直接赋值。
- `Block.note` 的生产写入已收口到 `app.models.layout_block_state`；它只作为提示/调试说明，不得恢复为关键判断字段。
- `Block.source` 的判断和用户编辑写入已收口到 `app.models.layout_block_state`；生产模块不得各自比较或直接写 `BlockSource.USER_EDITED/MANUAL_DRAW`。
- `Block.source_label` 的生产写入已收口到 `app.models.layout_block_state.set_layout_block_source_label()`；导入/投影构造可传初始值，运行时不得直接赋值。
- `Block.ocr_policy` 的生产写入已收口到 `app.models.layout_block_state.set_layout_block_ocr_policy()`；策略计算仍由 `ocr_dispatch_policy` 负责，业务模块不得直接赋值。
- `ProjectStore` 保存路径已收口为纯持久化写入；旧 `raw_payload_json/app_payload_json` 固定写空对象，运行时 route/旧 payload 只在加载校验处拒绝，不再通过清 `Block.lines` 或写 OCR invalidation 修复业务状态。
- `proof_line_state_store` 不再兼容读取旧 `line.proof_state` attr；proof runtime state 只存在于 external store。
- quality probe 设置不再读取旧 `quality_probe_target_ratio`；AppConfig 只接受 `quality_probe_sand_count` 和 `quality_probe_sand_unit_chars` 作为用户密度真值。
- quality probe bus topic 不再从 `app.core.quality_probe` 兼容导出；订阅方统一从 `app.core.proof_state` 读取。
- 外部版面事实新增 `NormalizedLayoutArtifact` 读取视图；overlay 展示和人工 Paddle 绑定索引先消费归一化 `LayoutRegion/LayoutSubregion`，为矢量 PDF 输入复用同一入口。
- 当前采用版面新增 `app.models.layout_snapshot.LayoutSnapshot` contract；Paddle API 主链从 `NormalizedLayoutArtifact` 编译 snapshot，再投影为 `Page.blocks` 供旧 UI/OCR/导出链路读取。新输入源不得直接适配旧 `Page.blocks`。
- 当前版面 snapshot 新增外部 `layout_snapshot_store`；`LayoutEditService` 人工编辑后同步 snapshot，`ProjectStore` 加载后也从持久化 block 投影重建 snapshot，再保留 `Page.blocks` 作为旧 UI/OCR/导出投影。
- 当前版面投影新增 `app.models.layout_projection` 边界；非 UI 业务层不再直接读写 `Page.blocks`，而是通过 `page_layout_blocks`、`replace_page_layout_blocks` 等 helper 消费或替换当前运行投影。
- layout route 新增 `RoutingPlan` 生产/读取 contract；overlay 公式文本读取、Hanwang 文本切片入口和 Hanwang route band 修正入口不再直接消费 `_layout_line_routes`/`_route_subblocks` dict，旧 dict API 仅作为运行时 cache 序列化边界。
- OCR 页级调度新增 `DispatchPlan`；OCR 管线的页级统计、图像失败记录、PP-OCRv5 行归属和 Hanwang prepass hint 复用不再直接遍历 `page.blocks` 推导文字块。
- OCR 运行结果新增 `OcrRunResult` / `OcrProgress` contract；Pipeline 文件不再定义结果模型，worker/controller/tests 改为依赖服务边界。
- HProof/VProof 外部刷新新增共享 `ProofExternalRefreshPlan`；pending 队列和 plan contract 不再按横校/纵校各自定义。

完成状态：第一阶段完成；`LayoutSnapshot` model contract 和 runtime store 已成为 API 版面分析、人工编辑到旧 `Page.blocks` 的投影边界。物理 OCR observation store 仍是后续增强，不再作为当前兼容入口。

### Phase 4: 删除兼容入口

- 删除已无生产调用的兼容 property / private shim。已完成。
- 删除只服务旧 UI 结构的隐藏字段和信号。已完成。
- 测试从“旧入口仍可用”改为“旧入口已不被主链依赖”。已完成。

完成状态：已完成。

## 后续执行规则

每个主线提交如果新增兼容代码，需要同时回答三件事：

1. 兼容哪个旧调用或旧数据。
2. 新入口是什么。
3. 删除它需要满足什么测试或迁移条件。

如果答不出来，就不应新增兼容层。
