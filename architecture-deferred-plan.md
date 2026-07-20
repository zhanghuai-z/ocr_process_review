# 架构治理延期任务记录

> 历史记录，已于 2026-07-20 被当前代码基线取代，不再作为实施指南。
> 当前事实以 `ProjectSession` 、不可变记录、`LayoutSnapshot`、
> append-only OCR observation、`ProofState` 和 `ExportProjectSnapshot` 为准。
> 旧 `Page/Block/Line/Char` 对象树、`Page.blocks` 回写和 runtime layout projection
> 已从生产代码移除。

记录日期：2026-06-17

## 当前决定

架构治理任务后延，不在当前功能线继续展开。

本任务不是取消，而是冻结为后续阶段性工程。当前优先级切回其他业务功能、导出、版面、校对或性能问题。

## 延期原因

- 当前主线仍有更直接影响使用体验的功能和稳定性问题。
- 架构治理会牵涉 UI、Controller、Layout、Paddle、CharOCR、Proof、Export 多个模块，不适合夹在功能调试中穿插推进。
- 直接大规模重构风险高，容易影响当前可用基线。
- 需要先把任务边界记录清楚，避免后续上下文压缩或多 agent 协作时目标漂移。

## 回来时的第一阶段目标

第一阶段只做架构止血和入口收口，不做全量重构。

1. 加 ratchet 架构测试
   - 生成 `architecture_baseline.json`。
   - 记录当前已有违规依赖边。
   - 测试只阻止新增违规，不要求一次清空历史债。
   - 初始重点关注：
     - `app.models -> app.core`
     - `app.core -> app.engines`
     - `app.ui -> app.engines`
     - `app.core` 内新增 Qt `QThread` / `Signal`

2. 新建版面编辑统一入口
   - `LayoutEditCommand`
   - `LayoutEditResult`
   - `LayoutEditService.apply(page, command)`

3. 优先迁移 LayoutPanel 的高风险修改入口
   - 新增框
   - 删除框
   - 移动框
   - 改类型
   - 合并框

4. 保持短期兼容
   - Service 内部短期可以继续回写旧的 `Page.blocks`。
   - UI 不再散落直接修改领域对象。
   - 不在第一阶段引入完整 annotation layer。

5. 补 focused regression
   - 版面编辑前后块数量、类型、bbox、source label 保持预期。
   - 编辑后 OCR invalidation 行为不退化。
   - 合并、删除、新增框不会破坏 proof/export 可见数据。

## 第二阶段目标

在第一阶段可合并后，再开始建立 Paddle/Hanwang 当前链路的 contract。

后续应用层统一使用 `charocr` 命名；底层厂商事实、native 二进制、旧项目迁移边界可以保留 `hanwang` 字样。

建议 DTO：

- `PaddleArtifact`
  - Paddle vendor 原始事实的规范化只读视图。
  - 不再把 app 派生字段直接塞进 raw dict。

- `LayoutSnapshot`
  - 当前 app 理解的页面版面状态。
  - 来源可以是 Paddle、人工编辑、旧项目加载后的回填。

- `RoutingPlan`
  - 哪些区域进入 OCR。
  - 哪些区域被阻断、跳过或作为结构元素保留。

- `DispatchPlan`
  - 实际送入 native/API 的 crop、bbox、类型、上下文。

- `OcrRunResult`
  - OCR 输出、字符框、文本、audit、fallback 信息。

## 第三阶段目标

逐步拆分 Controller 和目录结构。

优先顺序：

1. OCR job/session
   - worker 生命周期
   - target pages
   - CharOCR gate
   - auto-start OCR
   - parallel proof OCR

2. Project/session service
   - 当前项目加载、保存、关闭、dirty 状态。

3. Proof sync
   - 等 `h_proof` / `v_proof` 稳定后再拆。
   - 不先碰 proof 大 UI。

4. 目录迁移
   - `app/core`：纯规则、算法、无副作用函数。
   - `app/integrations/paddle`：Paddle API client、schema、normalize。
   - `app/integrations/charocr`：native bridge、cache、probe adapter。
   - `app/infrastructure`：ProjectStore、QSettings、cache。
   - `app/application` 或 `app/services`：Import、OCR、Layout、Proof use case。

## 明确不在延期任务第一阶段做的事

- 不做全量目录大搬迁。
- 不一次性清空所有历史依赖环。
- 不重写 `h_proof.py` / `v_proof.py`。
- 不一次性替换全链路 `PaddleArtifact/LayoutSnapshot/RoutingPlan`。
- 不把 `hanwang` 字样从 native 文件、厂商资料、旧项目迁移边界强行删除。
- 不让架构测试因为历史债永久红灯。

## 恢复任务时的切入点

恢复时从以下动作开始：

1. 生成当前依赖边清单。
2. 写入 `architecture_baseline.json`。
3. 增加 ratchet architecture test。
4. 新建 `LayoutEditCommand/LayoutEditResult/LayoutEditService`。
5. 只迁移 `LayoutPanel` 的一个最小编辑动作作为样板，再扩展到新增、删除、移动、改类型、合并。
