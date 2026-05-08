请在 **一次** gh copilot ask 中完成以下全部内容。

你的分支：`claude/editable-canvas`
工作目录：`D:\project\ocr_process\worktrees\claude`

## 任务目标

做一轮 **横校 UI 根因排查**，重点解决“横校内容重复了一遍”的问题；不要打补丁式隐藏重复内容。

用户最后确认的问题是：

1. 横校内容 **重复显示了一遍**
2. 用户明确不接受“补丁式”修法，不能只靠隐藏 / 跳过 / 去重显示来糊过去
3. 如果重复来自上游数据而不是 UI 渲染，也要在 handoff 里明确指出根因和证据

## 当前依赖

- `docs/coordinator-handoff.md`
- `docs/coordinator-agent-prompt.md`
- `plan.md`
- 当前集成基线：`coord/phase1-stabilization` @ `4ff9346`
- 先读：
  - `app/ui/proof/h_proof.py`
  - `app/ui/proof/v_proof.py`
  - 如有必要，可读 `app/core/project_store.py`、`app/models/` 相关结构，但不要改

## 边界

### ✅ 可以改

- `app/ui/proof/h_proof.py`
- `app/ui/proof/v_proof.py`（只在确认与重复渲染有关时再动）

### ❌ 不要改

- `app/core/`
- `app/services/`
- `app/engines/`
- `app/models/`
- `app/controllers/`
- `tests/test_core.py`

## 协作限制

- 这是 **横校重复内容排查** 任务
- 不要用“界面上去重一下”这种方式糊弄过去
- 先判断重复来自：
  - `load_pages()` / 行列表构建
  - block/line 数据源本身重复
  - 还是 UI 组件重复渲染
- 如果根因不在 UI，停止越界修复，在 handoff 里明确交回

## 具体任务

1. 复现“横校内容重复了一遍”的问题。
2. 明确判断重复发生在哪一层：
   - 数据层重复
   - UI 列表构建重复
   - 单行组件渲染重复
3. 只在确定根因位于 UI 层时修改 `h_proof.py` / 相关 UI 代码。
4. 如果重复其实是上游数据问题，在 handoff 里给出最小复现路径和证据，不要硬修 UI 表象。

## 验收标准

- 能明确说明横校重复内容的根因
- 若根因在 UI 层，重复显示被真正修掉
- 若根因不在 UI 层，handoff 能明确交回给 coordinator / GPT

## 测试与验证

- 自动化测试：`python tests/test_core.py`
- 如需手工验证：
  - 打开版面分析页，测试左拖 / 右画 / Delete 删除
  - 打开横校、纵校页面，对比图像区域是否明显更自然

## 交付要求

完成后：

1. 提交到 `claude/editable-canvas`
2. 保持 handoff 输出
3. 明确说明哪些点已完成，哪些点因边界限制未动

```text
[Handoff]
Task:
Done:
- ...
Files:
- ...
Decisions:
- ...
Risks:
- ...
Validate:
- ...
Next:
- ...
```
