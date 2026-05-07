# Multi-Agent Collaboration Protocol

本文件定义当前项目的 **双 agent + 协调者** 协作规则。目标不是“多人同时改代码”本身，而是让并行开发时：

- 任务边界清晰
- 热点文件冲突可控
- `plan.md` 成为单一事实源
- 任一 agent 都能低成本接手他人的半成品

## 1. 角色划分

### Coordinator（协调者）

负责：

- 维护 `plan.md`、`AGENT.md`、`README.md`
- 创建/回收 `git worktree`
- 分配任务与文件所有权
- 合并 agent 分支、处理冲突、执行集成回归

不负责：

- 长时间占用某一条独立功能线

### Agent A（引擎/数据流）

优先负责：

- `app/core/`
- `app/services/`
- `app/engines/`
- `app/models/`
- OCR / 版面 / bbox / 存储 / 导出链路

### Agent B（UI/交互）

优先负责：

- `app/ui/`
- 交互流程、可编辑画布、校对工作台、视觉优化

## 2. 单一事实源

并行开发时只允许以下三个“真相源”：

1. `plan.md`：共享任务板、负责人、依赖、当前集成状态
2. `AGENT.md`：协作协议、分支与交接规范
3. `docs/technical-implementation.md`：架构与实现约束

其中：

- **`plan.md` 是唯一共享执行板**
- **`AGENT.md` 是唯一协作规则**
- 代码事实以当前集成分支为准，不以聊天记录为准

## 3. `plan.md` 使用规则

`plan.md` 采用 **单写多读**：

- 默认只有 **Coordinator** 直接改 `plan.md`
- Agent A / Agent B 不直接大改 `plan.md`
- agent 的进展通过 commit、handoff 文本、PR 描述或给 Coordinator 的交接消息同步
- Coordinator 统一回填 `plan.md`

这样做的原因很简单：`plan.md` 是共享文件，双 agent 同时编辑最容易制造无意义冲突。

`plan.md` 至少维护以下信息：

- 当前阶段目标
- 任务分配表
- 热点文件所有权
- 依赖顺序
- 集成阻塞项
- 最近一次 handoff 摘要

## 4. 分支与 worktree 规范

## 4.1 分支模型

推荐固定三类分支：

- 集成分支：`coord/<milestone>`
- Agent A 分支：`agent-a/<task>`
- Agent B 分支：`agent-b/<task>`

示例：

```bash
git switch -c coord/phase1-stabilization
git switch -c agent-a/ocr-pipeline
git switch -c agent-b/editable-canvas
```

原则：

- `main` / 主分支不作为日常并行开发分支
- 所有 agent 分支都从当前 `coord/<milestone>` 拉出
- 合并目标也先回到 `coord/<milestone>`，不是直接进主分支

## 4.2 worktree 目录

建议在仓库同级目录创建 worktree：

```bash
cd /mnt/d/project/ocr_process
mkdir -p ../ocr_process_wt

git worktree add ../ocr_process_wt/coord    -b coord/phase1-stabilization
git worktree add ../ocr_process_wt/agent-a  -b agent-a/ocr-pipeline
git worktree add ../ocr_process_wt/agent-b  -b agent-b/editable-canvas
```

推荐目录：

- `../ocr_process_wt/coord`
- `../ocr_process_wt/agent-a`
- `../ocr_process_wt/agent-b`

这样可以保证：

- 每个 agent 的依赖、生成物、临时修改互不污染
- 可以并行开程序、跑测试、看日志
- 不会因为来回切分支导致工作区脏状态互相覆盖

## 5. 文件所有权与冲突控制

并行时不追求“完全不冲突”，但要尽量避免高频热点同时改。

### Coordinator 默认所有

- `plan.md`
- `AGENT.md`
- `README.md`
- `pyproject.toml`
- `build.bat`
- `build.sh`
- `tests/test_core.py` 的最终集成版本

### Agent A 默认优先

- `app/core/layout_analyzer.py`
- `app/services/ocr_pipeline.py`
- `app/engines/real_ocr_adapter.py`
- `app/models/`
- 存储、bbox、API/模型调用链

### Agent B 默认优先

- `app/ui/`
- 版面面板、OCR 面板、横校/纵校、可编辑画布
- 属性面板、交互快捷键、视觉层

### 热点文件处理规则

以下文件不允许两个 agent 同时长期持有：

- `app/ui/main_window.py`
- `app/controllers/workflow_controller.py`
- `tests/test_core.py`

处理方式：

1. 先在 `plan.md` 标出当前 owner
2. 另一个 agent 若必须修改，先等 owner 交接或由 Coordinator 接管
3. 尽量把共享文件改动压缩成一次短事务

## 6. 思维共享方式

不共享冗长推理过程，统一共享 **可执行信息**。每次 handoff 只写下面 6 项：

1. 做了什么
2. 为什么这样改
3. 改了哪些文件
4. 当前结果/剩余问题
5. 复现或验证命令
6. 下一位 agent 可以直接接什么

标准模板：

```text
[Handoff]
Task: ocr-pipeline
Done:
- 补了 API bbox 四点坐标兼容
- 增加 raw/app overlay 调试图

Files:
- app/core/layout_analyzer.py
- app/core/app_config.py

Decisions:
- API model_name 仅在配置有值时透传，避免破坏旧服务端

Validate:
- python tests/test_core.py

Next:
- 把 overlay 图入口接进 UI 按钮，方便直接查看
```

## 7. 集成节奏

推荐按“小批次、短周期”集成，而不是两个 agent 各自跑很久再硬合。

### 一个标准循环

1. Coordinator 更新 `plan.md`，分配任务
2. Agent A / Agent B 在各自 worktree 开发
3. 每人改完一个可合并单元就同步一次
4. Coordinator 先合低风险分支，再合高风险分支
5. 在 `coord/<milestone>` 上统一跑回归
6. 通过后再推进下一轮

### 合并优先级

1. 数据结构 / 配置 / 公共工具
2. 引擎链路 / bbox / 存储
3. UI 消费层
4. 最后合大范围样式与交互

原因：上游结构先稳定，UI 改动才不容易返工。

## 8. 推荐的首轮并行拆分

结合当前项目状态，建议第一轮直接这样拆：

| 角色 | 主任务 | 次任务 | 主要文件面 |
| --- | --- | --- | --- |
| Coordinator | 集成、计划、冲突处理、回归门禁 | 文档与配置规范 | `plan.md`, `AGENT.md`, `README.md`, 热点文件协调 |
| Agent A | `ocr-pipeline` | `quality-tests` | `app/core/`, `app/services/`, `app/engines/`, `app/models/` |
| Agent B | `editable-canvas` | `proof-workbench`, `ui-redesign` 前置基础 | `app/ui/` |

第二轮再拆：

| 角色 | 主任务 |
| --- | --- |
| Agent A | `bbox-tightener` + OCR/版面/自动收紧联调 |
| Agent B | 校对工作台、疑点队列、视觉与交互优化 |
| Coordinator | `export-delivery`、测试整合、打包与发布链 |

## 9. 合并与提交要求

- agent 分支先 rebase / merge 最新 `coord/<milestone>`
- 每次提交只做一个清晰主题
- 不允许把未验证的临时调试改动直接并到集成分支
- 涉及共享文件时，优先让 Coordinator 执行最终冲突解决

推荐命令：

```bash
git fetch origin
git switch agent-a/ocr-pipeline
git merge coord/phase1-stabilization
python tests/test_core.py
git status
```

## 10. 退出与清理

任务合并后，清理对应 worktree：

```bash
git worktree remove ../ocr_process_wt/agent-a
git branch -d agent-a/ocr-pipeline
```

长期保留：

- `coord/<milestone>` worktree
- 当前活跃 agent worktree

不保留：

- 已合并且不再继续的临时功能分支 worktree

## 11. 当前建议

如果你要立刻启动双 agent 模式，建议按这个顺序：

1. 由 Coordinator 建立 `coord/phase1-stabilization`
2. 创建两个 worktree：`agent-a`、`agent-b`
3. Agent A 先接 `ocr-pipeline`
4. Agent B 先接 `editable-canvas`
5. Coordinator 每完成一轮就回填 `plan.md`

这套框架先解决 **并行不撞车**，再解决 **功能快不快**。
