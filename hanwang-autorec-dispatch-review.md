# Hanwang AutoRec 调度层复盘

本文用于接手 Hanwang 前工作流。目标是回答：源程序是否已经有中英混排调度方案，以及我们应该完整模拟 AutoRec，还是抽取一个 AutoRec-lite 调度层。

## 结论

源程序有调度方案，但它依赖 AutoRec 内部对象链、对象 flag、临时文件和全局状态。我们现在从 Paddle/VL 接管了版面分析，不建议完整复刻 AutoRec。

推荐路线：

```text
Paddle/VL         -> 结构真值：text / formula / table / figure / inline_formula
PP-OCRv5          -> 正文行几何
AutoRec-lite      -> 生成可审计的 dispatch units
Hanwang linecut   -> 中文正文行/字符框
Eng20             -> 只处理明确的英文/拉丁 crop
人工框            -> 补足 Paddle 漏框后的结构真值
```

这里的 AutoRec-lite 是我们自己的调度层，不是 Hanwang 原 AutoRec 的一比一复刻。

## 原生证据

### 1. ClientRecogEntry 是任务入口

文件：

```text
/mnt/d/project/ocr_process/null/2026-06-05-project-revival/root-stale-source/hanwang/deepseek_hanwang_reverse_guidance/decompiled/AutoRec.dll/10001290.c
```

关键行为：

```text
读取 OCR1.0 任务文件
  -> 解析每个任务
  -> FUN_10011b20 / FUN_100127f0 准备任务状态
  -> FUN_10012110(param_3, param_3)
  -> FUN_10011ec0 收尾
```

这说明 AutoRec 是调度入口，不是某个单独识别引擎。它先准备任务状态，再让 `FUN_10012110` 执行识别主链。

### 2. Init 同时加载 CHN40 和 IntegratRcg

文件：

```text
AutoRec.dll/10001bd0.c
```

关键行为：

```text
HW_CHN40_Init(...)
LoadLibraryW("IntegratRcg.dll")
GetProcAddress("InitialCharRcg")
构造全局 AutoRec 状态对象 DAT_10052e00
```

这对应我们现在的两类 native 能力：

```text
linecut / CHN40 负责切行、识别
IntegratRcg.CharRcg 负责候选和字符框增强
```

### 3. FUN_10012110 是主识别分支

文件：

```text
AutoRec.dll/10012110.c
```

核心结构：

```text
if language == English:
  FUN_10013400(...)
  -> Eng20 全英文识别
  -> postprocess / output assembly

else:
  FUN_10013950(...)
  -> 非英文分支里的 Eng20 候选预处理

  FUN_10010580(...)
  -> 构造 recblock，调用 SegImg

  FUN_1000ec50(...)
  -> 按语言调用 CHN40/JPN40/KR40 Recog
  -> 中文分支再调用 IntegratRcg.CharRcg

  postprocess / output assembly
```

因此 AutoRec 的价值不只是调用 DLL，而是把“对象链 -> recblock -> SegImg -> Recog -> CharRcg -> 后处理”串起来。

## 原生对象链

AutoRec 维护一条对象链，反编译里常见遍历方式是：

```text
object_list_head = *(param + 0x30 + 8)
cursor           = *(param + 0x30 + 0x10)
object           = *cursor
object = next object
```

每个 object 里至少能看到这些字段用途：

```text
object[0]      flags
object[1..4]   bbox / 几何信息
object[5]      识别数量或字符数量
object[10]     扩展对象/子对象/字符结果容器
```

反编译里坐标字段顺序有些变量名不可靠，但结构意图清楚：AutoRec 不是直接识别整页，而是先把页面拆成 object，再按 object 构造 recblock。

### 关键 flag

目前能确认的 flag 语义：

```text
flags & 0x1f != 0
  该 object 会作为独立识别对象参与 Eng20 或 recblock 构造。

flags & 0x10 != 0
  非英文分支里的 Eng20 预处理重点。
  FUN_10013950 只把这类 object 转成 recblockeng 后送 Eng20。

flags & 0x20 != 0
  object 有子对象/容器结构。
  FUN_10010580 / FUN_10013400 会进入 object[10] 指向的子链。
```

还不能完全确认的部分：

```text
flags & 0x03
flags & 0x0c
```

这些在输出/方向/字符链写入中出现，但暂时不作为我们主链设计依据。

## SegImg 调度

文件：

```text
AutoRec.dll/10010580.c
```

行为：

```text
复制图像 buffer
遍历 object 链
  如果普通 object:
    用 object bbox 构造 recblock
  如果 object 是 0x20 容器:
    遍历子对象 bbox 构造 recblock

按语言调用：
  HW_JPN40_SegImg
  HW_KR40_SegImg
  HW_CHN40_SegImg
```

映射到当前项目：

```text
当前的 text_slice_routes_for_block()
  ~= AutoRec object / child object -> recblock

当前的 run_linecut_segimg(recblocks_xyxy=...)
  ~= AutoRec 的 HW_CHN40_SegImg 调用
```

也就是说，我们已经在做一个简化版 SegImg 调度，但它缺少明确的“对象模型”命名，导致逻辑散在 Paddle 路由和 micro_recblock 里。

## Recog / CharRcg 调度

文件：

```text
AutoRec.dll/1000ec50.c
```

行为：

```text
language == 4:
  HW_JPN40_Recog(...)

language == 5:
  HW_KR40_Recog(...)

language == 1:
  HW_CHN40_Recog(..., mode=0x4b, ...)
  IntegratRcg.CharRcg(...)

else:
  HW_CHN40_Recog(..., mode=0x47, ...)
  IntegratRcg.CharRcg(...)
```

当前项目里：

```text
app/engines/hanwang/native_bridge.py

RECOG_MODE_SIMPLIFIED  = 0x47
RECOG_MODE_TRADITIONAL = 0x4B
run_linecut_recog(..., with_charrcg=True)
```

这部分方向是对的：中文正文应该走 CHN40/linecut + CharRcg，而不是直接依赖 Paddle 文本兜底。

## Eng20 调度

### 全英文分支

文件：

```text
AutoRec.dll/10013400.c
```

行为：

```text
遍历 object 链
  普通 object -> recblockeng
  0x20 容器 -> 子对象 recblockeng

调用：
  HW_ENG20_RECOGIMG_ENGSTR(...)

若返回有效：
  把 Eng20 返回链写回 object / 子对象的识别数量字段
```

这说明纯英文页/块是另一条主链，不是中文 linecut 的附属逻辑。

### 非英文分支里的 Eng20 候选

文件：

```text
AutoRec.dll/10013950.c
```

行为：

```text
遍历 object 链
  只处理 flags & 0x10 的 object
  构造 recblockeng

调用：
  HW_ENG20_RECOGIMG_ENGSTR(...)

然后把返回链逐个写回 object + 0x14
```

这说明 Hanwang 原程序的中英混排不是“自动对整行跑英文 OCR”，而是先由 AutoRec 对象链标记出疑似英文对象，再把这些对象交给 Eng20。

### linecut 内部补调 Eng20

文件：

```text
linecut.dll/1000fe90.c
linecut.dll/10012580.c
linecut.dll/10011730.c
```

行为：

```text
HW_ENG20_RECOGLINE_ENGSTR(...)
FUN_100142e0(...) 转换 Eng20 返回结构
FUN_10011730(...) 尝试合并/替换进 linecut 字符链
```

这就是 120194 hook 里看到的情况：我们虽然没有显式调 Eng20，但 linecut 会在内部调。问题是它挑出的候选段在混合正文里不稳定。

## 对 120194 的解释

120194 实验证明：

```text
纯英文 crop:
  Eng20 可以识别 Specialization / IPE/VC / TFP / Alperovych

混合正文整行:
  linecut 会内部调用 Eng20
  但传入的是 linecut 自己挑出的局部候选段
  Eng20 原始返回已经出现 ~ 和错字母
```

因此问题不是“没有调用 Eng20”，而是：

```text
AutoRec 原本依赖 object flags 先定位英文对象；
我们现在没有这条 object flags 链；
linecut 自己兜底挑候选段时不够可靠。
```

## 当前项目映射

当前项目已有的等价物：

```text
AutoRec object chain
  -> ppvl_parsing_res_list + page.blocks + _route_subblocks

AutoRec recblock
  -> text_slice_routes_for_block()

AutoRec SegImg
  -> native_bridge.run_linecut_segimg()

AutoRec Recog + CharRcg
  -> native_bridge.run_linecut_recog(with_charrcg=True)

AutoRec Eng20
  -> 当前只有实验 probe：scripts/eng20_probe.cs
```

当前缺口：

```text
缺少一个明确的 dispatch unit 层。

现在逻辑分散在：
  app/core/paddle_line_routing.py
  app/engines/hanwang/micro_recblock.py
  app/core/paddle_artifact_index.py

导致“谁该送 Hanwang、谁该保留 Paddle、谁该走 Eng20、谁需要人工校验”没有统一真值对象。
```

## AutoRec-lite 对象模型

建议新增概念，不急于马上写代码：

```text
DispatchUnit:
  id
  parent_block_id
  kind:
    text
    inline_formula
    display_formula
    formula_number
    table
    figure
    latin_span
    skip
  bbox
  text_truth
  geometry_source:
    paddle_vl
    ppocrv5_line
    manual_draw
    manual_binding
  text_source:
    paddle_vl
    hanwang
    eng20
    manual_empty
  engine:
    hanwang_chn
    eng20
    paddle_truth
    none
  review_flags
```

生成规则：

```text
1. Paddle/VL 顶层块生成 parent units。

2. PP-OCRv5 行框覆盖 text parent 的行几何。

3. inline_formula / table / figure / display_formula 不送 Hanwang。

4. text 行扣除 inline_formula / skip 子块后，生成 text slice unit。

5. latin_span 暂不自动生成主链单元。
   先通过实验和人工框确认，再决定是否启用。

6. 人工画框不直接等于 OCR 结果。
   它只提高 geometry authority，并触发绑定或 review。
```

## Latin span 策略

短期不要做“程序自己切中英文”的大策略。原因：

```text
1. Hanwang 原程序也不是无条件切中英文，而是依赖 object flag。
2. 当前 Paddle/VL 有结构能力，PP-OCRv5 有行几何能力，没必要再造复杂字符级语言判断。
3. 错切英文会破坏中文正文字符链，维护成本高。
```

推荐分阶段：

```text
Phase 0:
  保持当前主链不变。
  只保留 Eng20 实验工具和 120194 证据。

Phase 1:
  对人工明确标注为 latin_span / english_text 的框，
  单独 crop -> Eng20 -> 生成 eng20 chars。
  不自动覆盖中文 Hanwang 结果，只在调试/横校中并排展示。

Phase 2:
  对 Paddle/VL 或 PP-OCRv5 明确给出英文片段的区域，
  生成 latin_span dispatch unit。
  要求有置信门槛和 review flag。

Phase 3:
  若 Phase 1/2 稳定，再把 Eng20 结果合并进正文行。
  合并规则必须按 bbox/reading order，不按字符串猜测。
```

## 对代码结构的建议

后续如果开工，建议不要直接在 `micro_recblock.py` 里继续加判断。

建议新增模块：

```text
app/core/ocr_dispatch_units.py
```

职责：

```text
输入：
  page.ppvl_parsing_res_list
  page.blocks
  PP-OCRv5 page_ocr_lines
  manual binding

输出：
  DispatchUnit[]
```

然后 `micro_recblock.py` 只消费 dispatch units：

```text
text units        -> linecut SegImg + Recog + CharRcg
formula/table/... -> Paddle truth / manual review
latin_span units  -> Eng20 experiment path
```

这样职责会接近 AutoRec，但比 AutoRec 更可审计。

## 120194 只读实验结果

已完成两个只读实验，实验脚本和输出为：

```text
scripts/experiment_120194_dispatch_units.py
  -> debug/120194_autorec_lite_dispatch/120194_dispatch_units.json
  -> debug/120194_autorec_lite_dispatch/120194_dispatch_units.md

scripts/experiment_120194_latin_binding.py
  -> debug/120194_latin_binding_alignment_eng20_with_footnote/120194_latin_binding.json
  -> debug/120194_latin_binding_alignment_eng20_with_footnote/120194_latin_binding.md
```

结果：

```text
120194 latin hints:        24
Hanwang final text exact:  20
alignment fallback match:   4
still unmatched:            0
Eng20 on matched crops:    24/24 correct
```

关键失败点：

```text
HHI  在 Hanwang 最终文本里变成了 脚
TFP  在 Hanwang 最终文本里变成了 ”7P / ”FP
PSM  在 footnote 块里可以 exact 找回
```

关键结论：

```text
1. Eng20 对准确 crop 的能力足够。
   CIC2 / Specialization / SPE / Diversification / DIV / TFP / PE/VC / Alperovych
   在 120194 样例上都能正确返回。

2. 只靠 Hanwang 最终文本回捞不够。
   因为 HHI / TFP 可能已经被 Hanwang 最终文本改写成中文或错误符号。

3. 可行召回路径是：
   Paddle/VL 父块真值文本
     -> Hanwang 回显文本
     -> 文本对齐
     -> 找回错写 span 的 bbox
     -> crop
     -> Eng20
```

这个实验解释了为什么源程序 AutoRec 里需要调度层：

```text
不是简单地“中文行走 CHN40，英文走 Eng20”。
真正需要的是一个中间对象层，记录：
  哪些内容来自结构真值
  哪些几何来自 Hanwang 字符框
  哪些文本需要 Eng20 重识别
  哪些位置需要人工复核
```

## 下一步产品化建议

不建议继续在 `micro_recblock.py` 里追加临时判断。建议把这次实验收敛成一个明确模块：

```text
app/core/latin_span_recovery.py
```

输入：

```text
text_parent unit 的 Paddle/VL 真值文本
Hanwang 回显文本和字符 bbox
Latin hints
人工新增/修正的 latin_span 框
```

输出：

```text
LatinSpanRecovery[]
  block_idx
  expected_text
  binding_status
  source_text_span
  hanwang_text_span
  bbox
  crop_path/debug only
  eng20_text
  review_flags
```

首版只进入调试和横校并排展示，不直接覆盖主 OCR 文本。

必须保留的风险标记：

```text
alignment_fallback_match
eng20_disagrees_with_expected
manual_box_requires_ocr_rerun
```

120194 的 footnote 已在包含 footnote 的回显实验中闭环，不再作为当前样例的阻塞点。

## 30 页扩大实验

扩大实验报告：

```text
latin-recovery-batch-report.md
debug/latin_recovery_batch_prose_all_v2/latin_recovery_batch_summary.md
debug/latin_recovery_batch_prose_all_v2/latin_recovery_batch_summary.json
```

最终统计：

```text
pages:                  30
pages_with_latin:       22
prose latin_hints:     189
matched:               189 / 189
exact match:           174
alignment fallback:     15
unmatched:               0
Eng20 strict ok:       177 / 189
Eng20 normalized ok:   181 / 189
Eng20 contains token:  185 / 189
```

实验中修正了三个判断：

```text
1. 绑定目标文本必须来自 Hanwang chars，而不是 line.text。
   line.text 可能包含 Paddle inline formula carrier，和 chars 数量不一致。

2. Latin recovery 默认跳过 $...$ 公式内部 token。
   公式变量由 Paddle inline formula 负责，不拆给 Eng20。

3. figure_title / caption / table_title 也要作为文本类进入 Hanwang 回显实验。
```

扩大实验结论：

```text
正文 Latin span 的 bbox 召回链路成立。
Eng20 可以作为辅助识别，但不能无条件覆盖正文。
跨行 token、Eng20 不一致、过宽 crop 需要 review flag。
```
