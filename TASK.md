请在 **一次** gh copilot ask 中完成以下全部内容。

你的分支：`claude/ocr-inspector-logic`  
工作目录：`D:\project\ocr_process\worktrees\claude`  
基线：`coord/phase1-stabilization` @ `c0164e0`

## 本轮目标

你这轮不要再把重点放在“再接一层 Paddle”。  
用户的明确要求是：**把 OCR Inspector 做成一个真正看得清参数、元素、逻辑关系的调试台。**

也就是说，这轮你负责的是：

1. 把参数都贴出来
2. 把整体逻辑规划清楚
3. 保证程序调试各项功能正常
4. 让用户看得见：
   - 哪个参数起什么作用
   - 能得到哪些元素

## 必读

- `D:\project\ocr_process\worktrees\coord\ocr_tool_prompt.md`
- `/root/github/label-studio`
- `D:\project\ocr_process\worktrees\coord\standard\ocr-paddle-standard.md`
- `D:\project\ocr_process\docs\Paddle_api_details\README.md`

## 用户当前反馈

1. 小工具目前 OCR 功能还只是占位
2. 根据 JSON 文件在图像上预览坐标详情还不太明朗
3. 我们不能再陷入另一个“重新接入 Paddle”的坑
4. 工具的目的很明确：
   - 我要看到哪个参数起什么作用
   - 可以得到哪些元素
   - 各项调试功能都能正常工作

## 你的交付

你自己决定怎么实现，但最终必须满足：

1. 参数在工具里是可见的，不是藏在代码里
2. 元素来源与层级在工具里是可见的
3. JSON / OCR_IR / 图像三者的关系在工具里是可观察的
4. 不是继续做 live-Paddle 接入，而是优先把现有 JSON / IR 调试能力做扎实

handoff 里必须明确写出：

- 现在工具里能看到哪些参数
- 每类参数会影响什么
- 现在工具里能看到哪些元素
- 整体调试逻辑是怎么规划的

## 文件边界

这轮没有其他实现 agent 与你抢文件。  
你可以按需要修改：

- `tools/`
- `build.bat`
- `build.sh`
- `ocr_process.spec`
- `app/`（如果启动链确实需要）
- `tests/test_core.py`

不要修改：

- `D:\project\ocr_process\AGENT.md`
- `worktrees/coord/standard/` 里的标准文件

## 过程要求

1. 你自己做审查，不要把“请 coord 给技术方向”当成默认下一步
2. **你自己跑现有测试 / 启动 / 构建链，再 handoff**
3. 不要再把重点放在“再接一个 live OCR 请求”
4. handoff 里必须说明当前工具对参数、元素、逻辑关系分别看到了什么

## 验收

coord 只看这些：

1. 分支能否顺利合并
2. 代码是否可构建
3. 工具里是否真的能看到参数 / 元素 / 逻辑关系
4. handoff 是否把整体调试规划写清楚

## 交付要求

完成后：

1. 提交到 `claude/ocr-inspector-logic`
2. 输出 handoff
3. 明确给出：
   - 工具里现在能看到哪些参数
   - 这些参数分别影响什么
   - 工具里现在能看到哪些元素
   - 整体调试逻辑怎么规划
   - 你自己跑了哪些测试/启动/构建
   - 剩余问题还有哪些
4. **完成后必须 ask request 等待**
