请在 **一次** gh copilot ask 中完成以下全部内容。

你的分支：`claude/ocr-inspector-logic`  
工作目录：`D:\project\ocr_process\worktrees\claude`

## 当前状态

你这条线最近已经连续提交了：

- `741e18d` `fix(inspector): image open + *.tif support + coord follow-up changes`
- `6459648` `fix: close all remaining loose ends — 73/73 tests pass`

说明你已经连续在做收口。  
**但当前这条 Claude 线仍然没有达到可交付水准。**

## coord 已复现到的现象

1. `build.bat --clean` 没有顺利完成
2. 实际卡点出现在清理旧构建目录时：
   - `dist\ocr_process\ocr_inspector.exe - Access is denied`
3. coord 复现时，Windows 下确实还能看到旧进程在运行：
   - `ocr_inspector.exe`
   - `ocr_process.exe`
4. `build.bat` 的 clean 模式标题还存在一个批处理语法问题：
   - `echo  Mode: FULL (--clean)` 会把 `)` 吞进 `if (...)` 语法，导致输出异常
5. 增量构建能进入 PyInstaller 主过程，说明当前不是“代码一启动就炸”的类型，而是**构建链收口不完整**
6. 用户刚刚又在真实运行中打到新的直接报错：
   - `Running OCR on: D:/project/ocr_process/worktrees/claude/file/244771纵校/120167.tif`
   - `[ERROR] Cannot read image: D:/project/ocr_process/worktrees/claude/file/244771纵校/120167.tif`
7. 这说明当前工具对 **`.tif` + 中文路径/真实用户路径** 的处理仍然没有收完，不能按“已交付”对待

## 你的本轮目标

你自己决定怎么实现，但最终必须满足：

1. 这条 Claude 分支的 **build + runtime** 都要达到可交付水准
2. `build.bat --clean` 至少不能再以现在这种方式卡死/假死/报不清楚
3. 如果旧 exe 正在运行，构建脚本要么能稳妥处理，要么**明确、可理解地失败并给出可执行提示**
4. OCR Inspector 对真实用户路径下的图像读取必须可靠，至少这类：
   - `.tif`
   - 中文/非 ASCII 路径
   - 真实工作目录下的用户文件路径
5. 不能把你前面刚做好的图像打开 / `.tif` / inspector 其他收口改坏
6. 不要再交一个“自评完成但用户一跑就报错”的版本；这轮目标是**完全收口**

## 必读

- `D:\project\ocr_process\worktrees\coord\ocr_tool_prompt.md`
- `/root/github/label-studio`
- `D:\project\ocr_process\worktrees\coord\standard\ocr-paddle-standard.md`
- `D:\project\ocr_process\docs\Paddle_api_details\README.md`

## 范围

你可以按需要修改：

- `build.bat`
- `build.sh`
- `ocr_process.spec`
- `tools/`
- `app/`（仅当确实影响打包或启动链）
- `tests/test_core.py`

不要修改：

- `D:\project\ocr_process\AGENT.md`
- `worktrees/coord/standard/` 里的标准文件

## 过程要求

1. 你自己判断并自审，不要把“请 coord 再分析一轮”当默认下一步
2. **你自己跑现有测试 / 启动 / 构建链，再 handoff**
3. 这轮重点不是再做新功能，而是把当前分支的交付阻塞项收干净
4. 如果你顺手发现和这次 build / 图像读取错误直接相关的小坑，可以一并处理
5. 不要把问题留给 coord 或用户二次验证后再暴露；你自己把真实运行路径走通

## handoff 必须写清楚

1. 构建失败的根因是什么
2. 你如何处理“旧 exe 占用导致 clean 失败/卡住”这个问题
3. `Cannot read image: ...120167.tif` 这类真实运行错误的根因是什么，你怎么收的
4. `build.bat` / 打包链 / 图像读取链具体改了什么
5. 图像打开 / `.tif` / 中文路径 / inspector 本轮已有修复是否全部保持正常
6. 你自己跑了哪些测试 / 启动 / 构建 / 真实路径验证
7. 还剩哪些问题没收；如果你认为没有，就明确写“当前已无已知用户可见阻塞”

## 交付要求

完成后：

1. 提交到 `claude/ocr-inspector-logic`
2. 输出 handoff
3. 明确说明“构建失败 + `.tif`/中文路径图像读取报错”这两件事是否彻底收口
4. **完成后必须 ask request 等待**
