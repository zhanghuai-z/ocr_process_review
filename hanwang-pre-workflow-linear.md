# Hanwang 前工作流 Review 手册

本文给 reviewer / subagent 使用。目标不是复述流程，而是让别人能复核当前代码到底做了什么，避免“看到的 UI 状态”和“实际送入 Hanwang 的数据”不一致。

## 当前真值

当前主链是：

```text
图片
  -> PaddleOCR-VL-1.6 版面分析
  -> Block / ppvl_parsing_res_list
  -> PP-OCRv5 整页行识别
  -> attach_page_ocr_line_routes()
  -> _layout_line_routes
  -> text_slice_routes_for_block()
  -> Hanwang recblocks_xyxy
  -> Hanwang SegImg group
  -> Hanwang Recog / CharRcg
  -> Block.lines
```

不要用 UI 颜色猜来源。现在“框链”调试入口已经移除，review 必须看字段、脚本输出或保存的 JSON。

## 关键数据名

```text
VL1.6 顶层块
  page.ppvl_parsing_res_list 里的 parsing_res_list 记录。
  字段通常有 block_label / block_bbox / block_content。

Block
  程序内部版面块，给 UI、保存、人工编辑和后续 OCR 使用。
  Block.raw_payload 保留 Paddle 原始记录副本。

_route_subblocks
  父级文字块内部的特殊区域列表。
  对 Hanwang 来说，这是“不要当普通正文送入”的区域。
  inline_formula 会保留 Paddle/VL 公式真值，最后拼回。

PP-OCRv5 行
  Line.text / Line.bbox / Line.confidence。
  当前 PP-OCRv5 word box 不作为真值。

_layout_line_routes
  每个文字块内的物理行路由。
  有 PP-OCRv5 行时，应由 PP-OCRv5 行框重建。

route_text_slice_bbox
  真正送入 Hanwang SegImg 的正文切片。
  来源是 _layout_line_routes[*].segments[kind="text"].bbox。

Hanwang group
  Hanwang SegImg 返回的识别组。
  程序会把 group 限制在 route_text_slice_bbox 内。

Char
  最终校对使用的字符结构。
  Hanwang 字符 source 通常是 hanwang:micro_recblock。
  行内公式拼回 carrier 的 source 是 paddle_inline_formula，不等于送入 Hanwang。
```

## 当前必须成立的断言

### 1. PP-OCRv5 行必须能解析出来

PP-OCRv5 真实响应可能是：

```text
result.ocrResults[].prunedResult.rec_texts
result.ocrResults[].prunedResult.rec_scores
result.ocrResults[].prunedResult.rec_boxes / dt_polys
```

不是所有响应都会包在 `overall_ocr_res` 下。解析入口是：

```text
app/core/paddle_response.py::overall_ocr_res()
```

如果这里没读到 `prunedResult.rec_texts`，后果是：

```text
page_ocr_lines 为空
  -> attach_page_ocr_line_routes() 不会覆盖初始路由
  -> Hanwang 可能使用 VL1.6 初始窄框
```

120169 曾经的问题就是这个。

### 2. footnote / vision_footnote 现在应走 Hanwang

当前标签分类在：

```text
app/core/paddle_labels.py
```

必须满足：

```text
footnote        in PADDLE_HANWANG_TEXT_LABELS
vision_footnote in PADDLE_HANWANG_TEXT_LABELS
footnote        not in PADDLE_HANWANG_SKIP_LABELS
vision_footnote not in PADDLE_HANWANG_SKIP_LABELS
```

另外：

```text
app/core/paddle_line_routing.py
```

`_FORMULA_STYLE_POSITION_LABELS` 当前只包含 `footer`，不应包含 `footnote`。否则包含公式样式的脚注会被误判为公式块并绕开 Hanwang。

### 3. 行内公式不应送入 Hanwang

行内公式应该作为 `formula` segment 拼回，不作为 `text` segment 送入 Hanwang。

判断方法：

```text
_layout_line_routes[*].segments
```

其中：

```text
kind=text     -> 送 Hanwang
kind=formula  -> 不送 Hanwang，保留 Paddle/VL 公式文本
kind=skip     -> 不送 Hanwang
```

如果看到公式有橙色字框，不要立刻判断为送入 Hanwang。先看 Char source：

```text
paddle_inline_formula  -> 公式 carrier，是拼回结果，不是 Hanwang 识别
hanwang:micro_recblock -> Hanwang 识别字符
```

版面分析界面的普通字框图层已经过滤 `paddle_inline_formula`，不应再显示公式 carrier 字框。

### 4. “旧项目结果”不能代表当前代码

保存过的项目可能保留旧 OCR 结果。代码改完后，需要重新跑 OCR 才能验证：

```text
PP-OCRv5 prepass
  -> attach_page_ocr_line_routes()
  -> Hanwang micro-recblock
```

如果 UI 仍看到旧窄框或旧文本，先确认当前页是否重新 OCR。

## 120169 样例复核

### 正文段落

运行：

```bash
python scripts/hook_120169_hanwang_route.py
```

输出：

```text
debug/120169_route_hook/120169_route_hook.md
debug/120169_route_hook/120169_route_hook.json
```

关键结论：

```text
VL 初始路由：
[192, 2658, 434, 2710] h=52
[547, 2658, 2051, 2710] h=52

PP-OCRv5 attach 后：
[296, 2642, 434, 2718] h=76
[547, 2642, 2043, 2718] h=76
```

这说明 PP-OCRv5 行框正常，旧问题是 PP-OCRv5 行没有被解析进路由。

### footnote

运行：

```bash
python scripts/experiment_120169_footnote_echo.py
```

该脚本会真实调用 Hanwang probe。输出：

```text
debug/120169_footnote_hanwang_echo/120169_footnote_hanwang_echo.md
debug/120169_footnote_hanwang_echo/120169_footnote_hanwang_echo.json
debug/120169_footnote_hanwang_echo/120169_footnote_hanwang_echo_overlay.png
```

120169 当前 footnote 的 Hanwang 前输入：

```text
vision_footnote idx7:
block_bbox = [281, 2017, 1621, 2080]
Hanwang pre route = [288, 2013, 1617, 2077]

footnote idx13:
block_bbox = [189, 2936, 2050, 3062]
Hanwang pre route = [279, 2929, 2041, 2994]
Hanwang pre route = [196, 3005, 883, 3058]

footnote idx14:
block_bbox = [278, 3075, 1105, 3134]
Hanwang pre route = [354, 3071, 1099, 3129]
Hanwang pre route = [283, 3076, 338, 3125]
```

这说明 footnote 已经不是直接用 block_bbox 送入 Hanwang，而是使用 PP-OCRv5 行框路由后的 text slice。

实验后处理结果：

```text
idx7 vision_footnote:
  已送 Hanwang；最终只采用 Hanwang 文本，不再用 PPVL 文本兜底。

idx13 footnote:
  已送 Hanwang；最终只采用 Hanwang 文本，不再用 PPVL 文本兜底。

idx14 footnote:
  完整走 Hanwang，得到字符框。
```

文字路径现在只有一个原则：VL 提供块和 inline 真值，PP-OCRv5 提供行框，正文切片送入 Hanwang；Hanwang 文本短或为空也不回退 PPVL 文本。

## 120194 中英混排 Eng20 复核

本节只讨论 Hanwang 内部英文引擎 Eng20，不讨论 Paddle 版面块。

### 先验结论

反编译资料和实际 hook 一致：

```text
Hanwang 中文主链会在内部调用 Eng20。
它不是暴露给我们的独立“中英混合识别 API”。
它的工作方式是：
  中文主链先切出疑似英文/符号候选段
  -> 调用 Eng20 的 *_ENGSTR 接口
  -> 根据内部质量判断决定是否合并回中文识别链
```

因此不能再判断为“程序没有接 Eng20”。真实问题是：Hanwang 内部挑出的英文候选段，在 120194 的混合正文行上不稳定。

### 实验文件

```text
Eng20 直接 probe:
  scripts/eng20_probe.cs
  resources/hanwang_native/bin/eng20_probe.exe

Eng20 hook 代理:
  scripts/eng20_proxy.cpp
  scripts/eng20_proxy.def
  scripts/build_eng20_proxy.bat
  debug/eng20_hook_bin/Eng20.dll
  debug/eng20_hook_bin/Eng20_real.dll
  debug/eng20_hook_bin/eng20_hook.log

120194 样例输出:
  debug/120194_latin_hanwang/
  debug/120194_latin_hanwang_with_footnote/
  debug/120194_latin_binding_alignment_eng20_with_footnote/
```

`Eng20.dll` hook 只用于实验目录 `debug/eng20_hook_bin`，没有替换正式 `resources/hanwang_native/bin/Eng20.dll`。

### 纯英文 crop 结果

纯英文或较干净的拉丁片段可以被 Eng20 识别：

```text
specialization.png
  Eng20 返回：:Specialization,

pevc.png
  Eng20 返回：IPE/VC

tfp.png
  Eng20 返回：TFP'

alperovych.png
  Eng20 返回：,Alperovych:
```

这说明 Eng20 引擎本身可用，至少对独立英文 crop 有效。

### 混合正文行结果

同一页的混合整行走 linecut 中文主链时，Eng20 确实被调用，但输入不是整行，而是 linecut 内部挑出的局部 `recblock`。

例子：

```text
line_row02_06_273_1670_2146_1744.png
  Eng20 recblock 实际候选区域：
    x=1480..1701, y=13..64
  Eng20 返回：
    ~yLv|~|~ja2]|m|~

line_row03_02_274_2217_2147_2290.png
  Eng20 recblock 实际候选区域：
    x=87..218, y=23..58
    x=1687..1755, y=13..57
  Eng20 返回：
    FiIID~Kn|C
    ~~~

line_row04_03_274_2680_2148_2759.png
  Eng20 recblock 实际候选区域：
    x=167..388, y=22..68
    x=460..603, y=15..60
  Eng20 返回：
    ~I~~|rlr~yyrlH
    ~|~rjmN~
```

这里 `~` 可以理解为 Hanwang 英文链里未能稳定解释的候选字符。它不是我们的 UI 自行生成的占位。

### 当前判断

```text
不是：
  没有调用 Eng20
  后续 UI 把正确英文弄坏
  linecut 完全没有中英混排机制

而是：
  linecut 内部英文候选段选择和合并机制不够可靠
  对包含中文、英文、数字、公式/符号的正文行，Eng20 原始返回已经不干净
```

这也解释了为什么“单独 crop 能识别”，但“程序内混合正文仍然串框/误识别”：两者进入 Eng20 的前置状态不同。

### 对主程序的含义

短期不要引入“程序自己按中英文切分”的复杂后处理。这个方向维护成本高，且会偏离当前主链。

更稳的方向是：

```text
1. VL / 人工框负责结构真值：
   text / formula / table / figure / inline_formula

2. PP-OCRv5 只负责文字行几何：
   哪一行，行框在哪里。

3. Hanwang 负责正文汉字字符框：
   公式、表格、图片不强行送入 Hanwang。

4. 对明显英文短语：
   不依赖 linecut 内部候选段。
   用 Paddle/VL 父块真值文本和 Hanwang 回显文本做对齐。
   找回 Latin span 的 bbox 后，独立 crop -> Eng20。
```

### 120194 AutoRec-lite Latin recovery 结果

实验文件：

```text
scripts/experiment_120194_dispatch_units.py
scripts/experiment_120194_latin_hanwang.py
scripts/experiment_120194_latin_binding.py
```

最新输出：

```text
debug/120194_autorec_lite_dispatch/120194_dispatch_units.json
debug/120194_latin_hanwang_with_footnote/120194_latin_hanwang.json
debug/120194_latin_binding_alignment_eng20_with_footnote/120194_latin_binding.json
debug/120194_latin_binding_alignment_eng20_with_footnote/120194_latin_binding.md
```

结果：

```text
Latin hints:                 24
Hanwang final text exact:    20
alignment fallback:           4
unmatched:                    0
Eng20 on recovered crops:    24/24 correct
```

需要注意：

```text
HHI 在 Hanwang 最终文本里被写成 脚
TFP 在 Hanwang 最终文本里被写成 ”7P / ”FP
```

因此“只从 Hanwang 最终文本里找英文词”不够；必须保留 Paddle/VL 父块文本作为结构真值，然后做文本对齐找回错写 span 的几何框。

这仍然是实验结论，不代表已经接入主程序。进入产品代码前，应先抽出独立模块，例如：

```text
app/core/latin_span_recovery.py
```

首版只进入调试/横校并排展示，不直接覆盖正文 OCR。

### 30 页扩大实验结论

扩大实验报告：

```text
latin-recovery-batch-report.md
debug/latin_recovery_batch_prose_all_v2/latin_recovery_batch_summary.md
```

最终结果：

```text
30 页样例
22 页含正文 Latin hint
189 个正文 Latin hint
189/189 成功找回 bbox
174 个 exact
15 个 alignment fallback
177/189 Eng20 严格等于原 token
185/189 Eng20 输出包含目标 token
```

注意这里已经排除了 `$...$` 内部公式 token。公式内部的 `ln / TFP / it / GGF / Post-short` 等不属于 Latin recovery；它们应该继续走 Paddle inline formula 真值。

剩余风险：

```text
1. Eng20 有时会粘前后字符：
   GDP -> rGDP
   Age -> (Age)

2. 个别 token 内容级误识别：
   PE/VC -> PE!VC / PEfVC
   Lerner -> Lemer

3. 跨行 token 不能直接 union 后整块送 Eng20：
   上一行末尾 PE/
   下一行开头 VC
```

因此产品化时，Latin recovery 首版应进入调试/横校辅助，不直接覆盖正文 OCR。

## Review 步骤

### 第一步：看标签分类

检查：

```text
app/core/paddle_labels.py
```

确认文本类和跳过类符合预期。尤其是：

```text
footnote / vision_footnote -> text path
inline_formula / display_formula / table / figure -> skip or formula path
```

### 第二步：看 PP-OCRv5 是否产生行

检查：

```text
app/core/paddle_response.py::overall_ocr_res()
app/core/ocr_ir_builder.py::build_ir_lines_from_item()
```

确认 `prunedResult.rec_texts` 这种结构可以变成 `Line`。

测试名：

```text
test_api_ocr_engine_reads_direct_pruned_ppocr_rows
```

### 第三步：看路由是否被 PP 行覆盖

检查：

```text
app/core/paddle_line_routing.py::attach_page_ocr_line_routes()
```

关注：

```text
line_hints
assigned
block[LAYOUT_LINE_ROUTES_FIELD]
```

如果目标块没有 `_layout_line_routes`，Hanwang 就会退回 block bbox 或旧初始路由。

### 第四步：看真正送 Hanwang 的框

检查：

```text
app/core/paddle_line_routing.py::text_slice_routes_for_block()
app/engines/hanwang/micro_recblock.py::run_micro_recblock()
```

真正传入 Hanwang SegImg 的参数是：

```text
recblocks_xyxy = [route.bbox for route in text_routes]
```

不是 UI 上看到的 block bbox。

### 第五步：看 Hanwang 后结果

检查：

```text
BlockResult.source
BlockResult.raw_block["_hanwang_bbox_audit"]
```

常见 source：

```text
hanwang 使用 Hanwang 识别结果；文本短或为空也保持 Hanwang 结果
ppvl    非文字结构路径，例如公式、表格、图片等跳过 Hanwang
```

## 诊断表

| 现象 | 优先检查 |
|---|---|
| PP 行框看起来正常，但 Hanwang 输入框很窄 | `overall_ocr_res()` 是否读到了 PP-OCRv5 行；`_layout_line_routes` 是否被 PP 行覆盖 |
| footnote 没有字符框 | 看 Hanwang group 是否为空、Recog 是否返回 chars；不再用 PPVL 文本兜底 |
| 行内公式出现字符框 | 看 char.bbox_source；`paddle_inline_formula` 是拼回 carrier，不是 Hanwang |
| UI 显示和脚本输出不一致 | 当前项目可能保存了旧 OCR 结果，重新跑 OCR |
| 公式/表格被 Hanwang 识别 | 检查 `_route_subblocks` 和 `_layout_line_routes[*].segments` |

## 推荐测试命令

```bash
env QT_QPA_PLATFORM=offscreen pytest -q tests/test_core.py
```

关键定向测试：

```bash
env QT_QPA_PLATFORM=offscreen pytest -q tests/test_core.py -k "direct_pruned_ppocr_rows or marker_formula_from_120169 or footnote_labels_route_through_hanwang or excludes_inline_formula_carriers"
```

实样例脚本：

```bash
python scripts/hook_120169_hanwang_route.py
python scripts/experiment_120169_footnote_echo.py
```

## 文件索引

```text
Paddle 响应解析：
  app/core/paddle_response.py
  app/core/ocr_ir_builder.py

标签分类：
  app/core/paddle_labels.py
  app/models/enums.py

PP 行到 Hanwang 路由：
  app/core/paddle_line_routing.py

Hanwang 前后处理：
  app/engines/hanwang/micro_recblock.py

版面 UI：
  app/ui/recognize/layout_panel.py
  app/ui/widgets/image_viewer.py

120169 正文路由 hook：
  scripts/hook_120169_hanwang_route.py

120169 footnote 前后 echo：
  scripts/experiment_120169_footnote_echo.py

120194 Eng20 实验：
  scripts/eng20_probe.cs
  scripts/eng20_proxy.cpp
  scripts/experiment_120194_dispatch_units.py
  scripts/experiment_120194_latin_hanwang.py
  scripts/experiment_120194_latin_binding.py
  debug/120194_latin_hanwang/
  debug/120194_latin_hanwang_with_footnote/
  debug/120194_latin_binding_alignment_eng20_with_footnote/
  debug/eng20_hook_bin/eng20_hook.log
```
