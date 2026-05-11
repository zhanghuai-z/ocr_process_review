请在 **一次** gh copilot ask 中完成以下全部内容。

你的分支：`claude/ocr-inspector-char-crops`  
工作目录：`D:\project\ocr_process\worktrees\claude`  
当前基线：`coord/phase1-stabilization` @ `d51255e`

## 本轮目标

继续开发 OCR Inspector，但这轮重点不是 build 了，而是把**字级可视化与切图能力**做实。

用户当前明确要的东西：

1. 画布上现在没有真正精确到字的框，`chars` 看起来还是 `fallback`
2. 需要一个**根据 bbox 输出切图**的模块
3. 需要一个能力：**根据我输入的文本，输出对应 bbox 的图像**

## 你要解决的问题

你自己决定具体方案，但最终必须满足：

1. 画布里字符层不能只停留在“fallback 状态看起来像有 chars”
2. 用户能清楚分辨：
   - 哪些 char/token bbox 是 OCR 真实给的
   - 哪些是 fallback / 推导出来的
3. 工具里要能基于 bbox 直接产出对应切图
4. 工具里要能基于用户输入的文本，找到对应 bbox，并输出对应图像
5. 这轮仍然要保持 OCR Inspector 的调试台属性：参数 / 元素 / 图像 / JSON / IR 关系不能被做坏

## 当前用户痛点

1. chars 的状态现在看起来还是 `fallback`
2. 用户没法直接拿 bbox 切图
3. 用户没法输入一段文本后，直接看到对应 bbox 的图像结果

## 必读

- `D:\project\ocr_process\worktrees\coord\ocr_tool_prompt.md`
- `/root/github/label-studio`
- `D:\project\ocr_process\worktrees\coord\standard\ocr-paddle-standard.md`
- `D:\project\ocr_process\docs\Paddle_api_details\README.md`

## 文件边界

你可以按需要修改：

- `tools/`
- `app/`（仅当确实影响 inspector 所需的共享能力）
- `tests/test_core.py`
- `build.bat`
- `build.sh`
- `ocr_process.spec`

不要修改：

- `D:\project\ocr_process\AGENT.md`
- `worktrees/coord/standard/` 里的标准文件

## 过程要求

1. 你自己判断并自审，不要把“请 coord 给技术方案”当默认下一步
2. **你自己跑现有测试 / 启动 / 构建链，再 handoff**
3. 不要只做一个“能演示”的半成品；要按用户真实调试场景收口
4. 如果 `char` 层精度受限于源数据，你也必须把“真实 bbox / fallback bbox”的区别在工具里做清楚

## handoff 必须写清楚

1. 现在字级 bbox 是怎么呈现的
2. 哪些是 OCR 真实 bbox，哪些是 fallback
3. bbox 切图模块怎么用
4. 文本输入 -> bbox 图像输出怎么用
5. 你自己跑了哪些测试 / 启动 / 构建
6. 还剩哪些问题没收

## 交付要求

完成后：

1. 提交到 `claude/ocr-inspector-char-crops`
2. 输出 handoff
3. 明确说明：
   - 字级框现在怎么展示
   - bbox 切图怎么做
   - 文本找图怎么做
   - 真实 bbox / fallback bbox 如何区分
4. **完成后必须 ask request 等待**
