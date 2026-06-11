# 兼容代码退出计划

本文件用于约束过渡期兼容层，避免兼容代码长期变成第二套业务逻辑。

## 原则

1. 兼容层必须有明确替代模型、调用方迁移路径和删除条件。
2. 新业务逻辑只能依赖新入口；兼容入口只能委托新入口，不能再生长判断分支。
3. 每次新增兼容入口，都要在本文件登记，否则视为技术债未受控。
4. 删除兼容层前必须有测试证明：旧入口无人调用，或旧入口只剩迁移测试覆盖。

## 当前兼容点

| 兼容点 | 当前作用 | 目标新入口 | 退出条件 | 风险 |
| --- | --- | --- | --- | --- |
| `Line.text <-> final_text` 镜像 | 兼容旧 UI / 存储 / 导出调用 `line.text` | proof 层统一读写 `line.final_text`，OCR 原文进入 `line.ocr_text` | UI、导出、proof、存储全部不再直接把 `text` 当最终真值；只保留加载旧库迁移测试 | 文本真值漂移，OCR 原文和人工终稿混淆 |
| `Block.raw_payload` app-owned keys | 过渡保存 Paddle 原始数据、binding、UI flags、OCR invalidation | `block_payload.py` 常量/helper，后续拆 `PaddleArtifact / AnnotationBinding / LayoutSnapshot` | 所有 app-owned key 都只通过 helper；再迁入正式模型表或 dataclass | Paddle 原始真值和应用状态混在同一 dict |
| `WorkflowController._page_gate_info()` | 保持 controller 内部入口 | `workflow_state.page_gate_info()` | UI / controller 逻辑直接依赖 typed `PageGateInfo` 后可降级为私有转发或删除 | 页面 gate 状态再次散落在 controller |
| `app.core.ocr_config` | 兼容旧 OCR 配置访问方式 | `AppConfig` typed/default config | API 设置、引擎创建、测试全部不再通过旧模块读写 | 配置源分裂，UI 显示和真实调用不一致 |
| UI/test compatibility fields, e.g. hidden labels/signals | 保旧测试或旧 UI 调用不崩 | 显式 view model / signal contract | 对应旧测试改为测试新 contract；旧 UI 调用点删除 | UI 文件继续膨胀，测试锁死旧结构 |

## 已清理

| 已清理入口 | 替代入口 | 清理说明 |
| --- | --- | --- |
| `OcrProject.ocr_completed` | `has_any_ocr_result` / `all_pages_ocr_done` | 删除兼容 property；controller/UI/tests 改用显式语义。 |
| `WorkflowController.ocr_completed` | `WorkflowController.has_any_ocr_result` | 删除旧 accessor，避免打开项目时继续传播模糊命名。 |
| `Page.recognizable_blocks` | `Page.text_ocr_blocks` | 删除兼容 property；OCR 入口统计改用统一 dispatch 结果。 |
| `LayoutAnalyzer._extract_*_from_record()` | `paddle_layout_schema.py` | 删除旧私有 shim；字段解析直接走 schema adapter。 |

## 删除顺序

### Phase 1: 冻结兼容层

- 所有新代码禁止直接新增裸兼容字段。
- `raw_payload` app-owned key 必须先经过 `block_payload.py`。
- Paddle 返回字段必须先经过 `paddle_layout_schema.py`。
- OCR dispatch 必须先经过 `ocr_dispatch_policy.py`。

完成状态：部分完成；`recognizable_blocks` / `ocr_completed` / layout record shim 已删除。

### Phase 2: 调用方迁移

- `recognizable_blocks` 调用点迁移到 `text_ocr_blocks`。已完成。
- `ocr_completed` 调用点迁移到 `has_any_ocr_result` 或 `all_pages_ocr_done`。已完成。
- `Line.text` 写入点收口到 proof/OCR 组装层，普通 UI 不直接写。
- controller 中业务 gate 继续下沉到 `workflow_state.py` 或后续 `WorkflowStateMachine`。

完成状态：未完成。

### Phase 3: 数据模型替换

- 把 `paddle_binding` 升为 `AnnotationBinding`。
- 把 `_route_subblocks` / `_layout_line_routes` 升为 `RoutingPlan` 或 `LayoutSnapshot` 字段。
- 把 `ocr_text_invalidated` 升为 block/page revision 或正式 invalidation 字段。
- 把 `raw_payload` 降级为只保存 vendor 原始 payload，不再保存 app 状态。

完成状态：未开始。

### Phase 4: 删除兼容入口

- 删除已无生产调用的兼容 property / private shim。
- 删除只服务旧 UI 结构的隐藏字段和信号。
- 测试从“旧入口仍可用”改为“旧入口已不被主链依赖”。

完成状态：未开始。

## 后续执行规则

每个主线提交如果新增兼容代码，需要同时回答三件事：

1. 兼容哪个旧调用或旧数据。
2. 新入口是什么。
3. 删除它需要满足什么测试或迁移条件。

如果答不出来，就不应新增兼容层。
