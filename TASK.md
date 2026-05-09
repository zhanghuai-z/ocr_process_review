请在 **一次** gh copilot ask 中完成以下全部内容。

你的分支：`claude/ocr-inspector-logic`  
工作目录：`D:\project\ocr_process\worktrees\claude`  
当前基线：`coord/phase1-stabilization` @ `c0164e0`

## 本轮性质

这是 **OCR Inspector 收口 follow-up**，不是新开一轮大改。  
用户刚才的直接反馈是：

1. **目前图像还是打不开**
2. 图像选择的默认格式要明确包含 **`.tif`**
3. 你顺手把这轮相关的明显未收口项再检查一遍，能一起收掉的就一起收掉

## 当前已知上下文

coord 刚在你当前工作树里做了两类**未提交**直修，请你直接接着做，不要回退：

1. `PPStructureV3` 缺失时，已加本地兼容回退
2. OCR Inspector 已接入主程序共享的 API 设置 / `/layout-parsing` 通路

相关变更当前就在工作树里，主要涉及：

- `tools/ocr_inspector/ui/panels/run_ocr.py`
- `tools/ocr_inspector/ui/__init__.py`
- `tests/test_core.py`

你的任务是基于这个当前状态继续收口。

## 本轮目标

你自己决定具体实现，但最终必须满足：

1. 用户在 OCR Inspector 里**真的能把图像打开/设进去**
2. 所有相关图像文件选择入口的默认格式都明确包含 **`*.tif`**（不要只有 `*.tiff`）
3. 你自己检查这一轮 OCR Inspector 还有没有和“打开 / 加载 / 显示 / 调试流”直接相关的明显缺口，有就一并收掉
4. 不要把这轮扩展成无边界重构；重点是**把用户当前还能踩到的坑收口**

## 必读

- `D:\project\ocr_process\worktrees\coord\ocr_tool_prompt.md`
- `/root/github/label-studio`
- `D:\project\ocr_process\worktrees\coord\standard\ocr-paddle-standard.md`
- `D:\project\ocr_process\docs\Paddle_api_details\README.md`

## 文件边界

你可以按需要修改：

- `tools/`
- `build.bat`
- `build.sh`
- `ocr_process.spec`
- `app/`（仅当确实影响 inspector 打开/设置/显示链路）
- `tests/test_core.py`

不要修改：

- `D:\project\ocr_process\AGENT.md`
- `worktrees/coord/standard/` 里的标准文件

## 过程要求

1. 你自己做判断和审查，不要把“请 coord 给技术方案”当默认下一步
2. **你自己跑现有测试 / 启动 / 构建链，再 handoff**
3. 不要把重点又转回“再接一层 live Paddle”；这轮是收口和可用性修补
4. 如果你发现当前工作树里 coord 的未提交修改需要一起整理，就直接整理进你的提交里

## handoff 必须写清楚

1. 图像原来为什么打不开，你怎么收的
2. 哪些入口现在支持 `*.tif`
3. 这轮你顺手还收了哪些相关小问题
4. 你自己跑了哪些测试 / 启动 / 构建
5. 还剩哪些问题没收

## 交付要求

完成后：

1. 提交到 `claude/ocr-inspector-logic`
2. 输出 handoff
3. 明确说明这轮“图像打开 + `.tif` + 收口检查”的完成情况
4. **完成后必须 ask request 等待**
