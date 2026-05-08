请在 **一次** gh copilot ask 中完成以下全部内容。

你的分支：`claude/editable-canvas`
工作目录：`D:\project\ocr_process\worktrees\claude`

## 任务目标

做一轮 **UI / 交互 / 轻样式收口**，重点解决“版面分析启动后，用户看到的流程与反馈不清晰”的问题，不碰 core / engine / model adaptation。

用户最后确认的问题是：

1. 版面分析启动按钮改成 **功能栏里的绿色三角图标**，不要文字描述
2. OCR 响应慢时，UI 需要有明确的 **OCR 进度反馈**
3. 用户确认现在 **已经可以进入横校**，所以不要再围绕“进不了横校”做表面修饰；只处理真实存在的 UI 反馈问题
4. 不要保留可见的删除按钮，改为 **Delete 键删除**
5. 顶部已经有导航栏，面板里重复的步骤名/标题占位应该清理
6. 版面编辑默认 **左键拖动图片**，**右键画框**
7. 新建框时要支持不同块类型（如 `text/title/table/equation/...`）
8. 横校和纵校的图像呈现仍然“老样子”，其中 **纵校切图问题仍然明显**
9. Gemini 原本负责的轻量样式收口并入本轮，由你顺手处理当前界面里明显“旧工具感”的控件观感，但不要做大范围设计重构

## 当前依赖

- `docs/coordinator-handoff.md`
- `docs/coordinator-agent-prompt.md`
- `plan.md`
- 当前集成基线：`coord/phase1-stabilization` @ `4ff9346`
- 先读：
  - `app/ui/main_window.py`
  - `app/ui/proof/h_proof.py`
  - `app/ui/proof/v_proof.py`
  - `app/ui/recognize/layout_panel.py`
  - `app/ui/recognize/ocr_panel.py`
  - `app/ui/widgets/image_viewer.py`
  - `ui.jpg`

## 边界

### ✅ 可以改

- `app/ui/proof/h_proof.py`
- `app/ui/proof/v_proof.py`
- `app/ui/main_window.py`
- `app/ui/recognize/layout_panel.py`
- `app/ui/recognize/ocr_panel.py`
- `app/ui/widgets/image_viewer.py`
- `app/ui/recognize/import_panel.py`
- `app/ui/style.py`（仅限为本轮 UI 页面补轻量样式收口）

### ❌ 不要改

- `app/core/`
- `app/services/`
- `app/engines/`
- `app/models/`
- `app/controllers/`
- `tests/test_core.py`

## 协作限制

- 这是 **UI-only** 任务
- 不要改 PaddleOCR 模型适配逻辑
- 不要碰 `workflow_controller.py`
- 如果发现真实进度条缺后端信号，先在 handoff 里说明，不要越界补 core/controller
- 如果某项在当前基线已满足，就不要制造无意义 diff，但仍要在 handoff 里明确说明
- 若改 `app/ui/style.py`，只做与本轮页面直接相关的轻量收口，不要扩成独立视觉改版
- 不允许打补丁式修法；如果纵校切图问题在 UI 层只是结果表现，必须在 handoff 里把根因缺口写清楚

## 具体任务

1. 把版面分析启动入口改成工具栏里的绿色三角图标按钮：
   - 保留 tooltip
   - 不再显示“▶ 分析”这类文字按钮
2. 收口 OCR 期间的 UI 反馈：
   - 如果 controller 已提供真实 OCR 进度，就在 UI 上显示
   - 只修真实存在的反馈问题，不要围绕已恢复的横校入口做表面处理
3. 复核并收口版面编辑交互：
   - 左键拖动画面
   - 右键画框
   - 删除依赖 Delete 键
   - 不再保留显式删除按钮
4. 让新建框的类型选择真正落地到框创建流程，支持现有 `BlockType` 里的常见类型。
5. 清理和顶部导航重复的面板标题占位。
6. 改善横校 / 纵校的图像呈现与观感，重点看 `v_proof.py` 的切图与展示。
7. 如果 UI 侧已经能接到真实进度信号，就把版面分析 / OCR 的进度反馈完整显示出来；否则在 handoff 里明确缺口。
8. 如有必要，可在 `app/ui/style.py` 中补少量 QSS，让本轮涉及页面的进度条、列表、编辑区、滚动条观感不再明显落后。

## 验收标准

- 版面分析入口是绿色三角图标，不再是文字按钮
- OCR 慢时有明确进度反馈
- 版面编辑没有显式删除按钮，但 Delete 键仍可删框
- 左键 / 右键交互符合要求
- 新建框能使用所选块类型
- 重复标题被清理
- 横校 / 纵校界面观感有改善，尤其纵校切图不再维持旧问题

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
