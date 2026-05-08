请在 **一次** gh copilot ask 中完成以下全部内容。

你的分支：`claude/editable-canvas`
工作目录：`D:\project\ocr_process\worktrees\claude`

## 任务目标

做一轮 **UI / 交互 / 轻样式收口**，只解决用户最后明确提到的界面问题，不碰 core / engine / model adaptation。

用户最后确认的问题是：

1. 横校和纵校的图像呈现仍然“老样子”，其中 **纵校更严重**
2. 不要保留可见的删除按钮，改为 **Delete 键删除**
3. 顶部已经有导航栏，面板里重复的步骤名/标题占位应该清理
4. 版面编辑默认 **左键拖动图片**，**右键画框**
5. 版面分析页需要 **真实进度条**，不要伪进度
6. 新建框时要支持不同块类型（如 `text/title/table/equation/...`）
7. Gemini 原本负责的轻量样式收口并入本轮，由你顺手处理当前界面里明显“旧工具感”的控件观感，但不要做大范围设计重构

## 当前依赖

- `docs/coordinator-handoff.md`
- `docs/coordinator-agent-prompt.md`
- `plan.md`
- 当前集成基线：`coord/phase1-stabilization` @ `4ff9346`
- 先读：
  - `app/ui/proof/h_proof.py`
  - `app/ui/proof/v_proof.py`
  - `app/ui/recognize/layout_panel.py`
  - `app/ui/widgets/image_viewer.py`
  - `ui.jpg`

## 边界

### ✅ 可以改

- `app/ui/proof/h_proof.py`
- `app/ui/proof/v_proof.py`
- `app/ui/recognize/layout_panel.py`
- `app/ui/widgets/image_viewer.py`
- `app/ui/recognize/import_panel.py`
- `app/ui/recognize/ocr_panel.py`
- `app/ui/style.py`（仅限为本轮 UI 页面补轻量样式收口）

### ❌ 不要改

- `app/core/`
- `app/services/`
- `app/engines/`
- `app/models/`
- `app/controllers/`
- `app/ui/main_window.py`
- `tests/test_core.py`

## 协作限制

- 这是 **UI-only** 任务
- 不要改 PaddleOCR 模型适配逻辑
- 不要碰 `workflow_controller.py`、`main_window.py`
- 如果发现真实进度条缺后端信号，先在 handoff 里说明，不要越界补 core/controller
- 如果某项在当前基线已满足，就不要制造无意义 diff，但仍要在 handoff 里明确说明
- 若改 `app/ui/style.py`，只做与本轮页面直接相关的轻量收口，不要扩成独立视觉改版

## 具体任务

1. 复核并收口版面编辑交互：
   - 左键拖动画面
   - 右键画框
   - 删除依赖 Delete 键
   - 不再保留显式删除按钮
2. 让新建框的类型选择真正落地到框创建流程，支持现有 `BlockType` 里的常见类型。
3. 清理和顶部导航重复的面板标题占位。
4. 改善横校 / 纵校的图像呈现与观感，重点看 `v_proof.py`。
5. 如果 UI 侧已经能接到真实进度信号，就把版面分析进度条完整显示出来；否则在 handoff 里明确缺口。
6. 如有必要，可在 `app/ui/style.py` 中补少量 QSS，让本轮涉及页面的进度条、列表、编辑区、滚动条观感不再明显落后。

## 验收标准

- 版面编辑没有显式删除按钮，但 Delete 键仍可删框
- 左键 / 右键交互符合要求
- 新建框能使用所选块类型
- 重复标题被清理
- 横校 / 纵校界面观感有改善，纵校不再明显落后

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
