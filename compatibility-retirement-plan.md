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
| `Page.recognizable_blocks` | `Page.text_ocr_blocks` | 删除兼容 property；OCR 入口统计改用统一 dispatch 结果。 |
| `LayoutAnalyzer._extract_*_from_record()` | `paddle_layout_schema.py` | 删除旧私有 shim；字段解析直接走 schema adapter。 |
| `WorkflowController._page_gate_info()` | `workflow_state.page_gate_info()` | 删除 controller 转发，调用点直接依赖 typed gate helper。 |
| `app.core.ocr_config` | `app.core.app_config.get_config/update_config` | 删除旧配置桥模块，生产和测试导入统一到 AppConfig 入口。 |
| `Line.text <-> final_text` 双向镜像 | `Line.proof_state` / `proof_display_text(line)` / `Line.ocr_text` | 删除 `__setattr__` 镜像；校对和导出读取 ProofLineState/display helper，`text` 保留为 OCR 行文本字段。 |
| `Block.raw_payload/app_payload` app-owned keys | typed 字段：`paddle_binding` / `ocr_invalidated_reason` / `origin` / `ocr_audit` / `table_text_layer_cells` / `layout_edit_events` | `app_payload` 已退役为空；旧 app-owned payload 不再迁移，非空会被当前 schema 校验拒绝。 |
| UI/test hidden compatibility fields/signals | explicit state/table accessors | 删除 `QualityStatsDialog._rate_lbl`、兼容 `_table` property、`NavRail.account_clicked` 空信号；测试改读 typed state / `detail_table()`。 |

## 删除顺序

### Phase 1: 冻结兼容层

- 所有新代码禁止直接新增裸兼容字段。
- app-owned 状态必须使用 typed 字段；禁止写入 `raw_payload/app_payload`。
- Paddle 返回字段必须先经过 `paddle_layout_schema.py`。
- OCR dispatch 必须先经过 `ocr_dispatch_policy.py`。

完成状态：已完成；已登记兼容入口均已删除或收口为正式模型字段。

### Phase 2: 调用方迁移

- `recognizable_blocks` 调用点迁移到 `text_ocr_blocks`。已完成。
- `ocr_completed` 调用点迁移到 `has_any_ocr_result` 或 `all_pages_ocr_done`。已完成。
- `Line.text` 写入点收口到 OCR/legacy 源字段，普通 UI 写入 `final_text`。已完成。
- controller 中业务 gate 直接调用 `workflow_state.page_gate_info()`。已完成。

完成状态：已完成。

### Phase 3: 数据模型替换

- `paddle_binding` 已升为 `Block.paddle_binding` typed state。
- `_route_subblocks` / `_layout_line_routes` 已限制为运行时 route dict，不进入 `Block.raw_payload/app_payload` 持久化模型。
- `ocr_text_invalidated` 已升为 `Block.ocr_invalidated_reason`。
- `raw_payload` 已降级为 vendor 原始 payload；`app_payload` 已退役为空。

完成状态：第一阶段完成；page 级 `LayoutSnapshot` 是后续增强，不再作为当前兼容入口。

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
