# Hanwang AutoRec / Eng20 Deeper Trace

本文继续追 AutoRec 对英文候选、Eng20 调度和回写的处理方式，并更新当前项目的接入判断。

## 结论

AutoRec 的中英混排不是简单的“整行判断中英文”，而是对象驱动：

```text
外部任务/版面对象
  -> AutoRec object flag
  -> recblock / recblockeng
  -> CHN40 / linecut / CharRcg / Eng20
  -> 写回 object 或 linecut 字符链
```

目前能确认：

```text
flags & 0x10
  被 AutoRec 视为非英文主链中的 Eng20 候选 object。

flags & 0x20
  表示有子对象/容器结构，常见于 table/complex object。

linecut 内部也会调用 Eng20。
  但它以 `~` 作为占位/未知字符，再尝试局部重识别和替换。
```

这说明我们的 AutoRec-lite 不应该依赖“程序自己猜中英边界”。更稳的方式是显式生成可审计对象：

```text
text line
inline formula
latin span
table
figure
manual object
review object
```

## 关键证据

### 1. `FUN_10013950`: 非英文分支 Eng20 候选

文件：

```text
/mnt/d/project/ocr_process/null/2026-06-05-project-revival/root-stale-source/hanwang/deepseek_hanwang_reverse_guidance/decompiled/AutoRec.dll/10013950.c
```

关键逻辑：

```text
遍历 AutoRec object 链
  if object.flags & 0x10:
    object bbox -> recblockeng

HW_ENG20_RECOGIMG_ENGSTR(...)

遍历 object 链
  if object.flags & 0x1f:
    把 Eng20 返回数量写到 object + 0x14
```

所以 `0x10` object 是 AutoRec 主调度层明确认可的英文候选。

### 2. `FUN_100152d0`: 外部任务对象创建

文件：

```text
AutoRec.dll/100152d0.c
```

关键逻辑：

```text
读取任务/版面结构中的 object type
pvVar3 = object type / flags
FUN_100170a0(pvVar3, bbox, ...)

if pvVar3 & 0x20:
  继续解析子对象表
```

这说明 object flag 很大一部分来自外部任务/版面文件，而不是 OCR 后文本识别出来的。

因此 `flags & 0x10` 的来源更像是：

```text
上游版面/任务对象标记出的英文候选
```

而不是：

```text
linecut 临时判断某几个字符像英文
```

### 2.1 当前样例 PST 没有给出 Latin 子对象

对真实样例 `file/244771纵校/*.tif.pst` 做了只读解析。文件头符合 `PST31TIF` / `PST31BKI`，并且能按 `100152d0.c` 的结构读出 top-level object：

```text
header:
  uint[0..1]  signature
  uint[6]     page width
  uint[8]     page height
  uint[9]     top-level object count

object record:
  uint[0] left
  uint[1] right
  uint[2] top
  uint[3] bottom
  uint[4] extra/count
  uint[5] flag
  uint[6..9] extra fields
```

抽查结果：

```text
120166: flags = 0x1
120169: flags = 0x1, 0x1, 0x1, 0x1
120186: flags = 0x1, 0x1, 0x1
120192: flags = 0x1
120194: flags = 0x1, 0x1, 0x1, 0x1
```

这些样例的 PST 顶层对象没有 `0x10` 英文候选，也没有可直接拿来替代 Paddle token 的 Latin 子对象。

这意味着：

```text
PST 可以作为 Hanwang 原生大块/页面结构参考；
但当前样例里，它不能解决正文 Latin token truth 问题。
```

### 3. `FUN_1000e8c0`: AutoRec 自带分区也会生成 object

文件：

```text
AutoRec.dll/1000e8c0.c
```

关键逻辑：

```text
_seg(...)
  -> CArea list
  -> CArea.type 映射成 object flag
  -> FUN_10012af0(flag, AutoRecState)
```

当前看到的映射：

```text
default / type 0 / type 2 -> 0x1
type 1                  -> 0x200
type 3                  -> 0x8
type 4                  -> 0x20
```

这里没有直接看到 `0x10`。所以不能把 `0x10` 归因到这条 `_seg` 普通分区路径。

### 4. `FUN_10015ad0`: object 字段初始化

文件：

```text
AutoRec.dll/10015ad0.c
```

关键结构：

```text
object[0]    = flags
object[1..4] = bbox
object[5..9] = index / source / extra geometry fields
object[10]   = attr object
```

如果 `flags & 0x1f != 0`，创建 `SegAttr_T`。

如果 `flags == 0x20`，创建 `TableAttr_T`。

这和我们之前对 object 的判断一致。

### 5. `linecut.dll/100142e0`: Eng20 返回转成 linecut 链

文件：

```text
linecut.dll/100142e0.c
```

关键逻辑：

```text
遍历 Eng20 返回结构
  如果返回字符是 ASCII 且可直接表示:
    写入该 ASCII
  如果返回字符不能直接表示:
    写入 '~'
    可能把原始 code 写到 +0x18

同时写入字符 bbox:
  x/y/width/height
```

这解释了为什么实验里会看到大量 `~`。

`~` 不是普通文本字符，它更像 Eng20/linecut 之间的占位符。

### 6. `linecut.dll/10011730`: 处理 `~` 并尝试替换

文件：

```text
linecut.dll/10011730.c
```

关键逻辑：

```text
遍历 linecut 字符链
  if char == '~':
    根据 bbox 找前后节点
    构造局部区域
    先尝试中文识别
    再调用 FUN_10012580 -> Eng20 recogline
    如果结果可用:
      替换/合并链表节点
    否则:
      删除或保留原节点
```

这说明 linecut 内部确实有“英文补救”机制，但它不是稳定的高层边界算法，而是对 `~` 占位区域的局部修复。

## 对当前方案的影响

### 不要把 Paddle token 做成唯一真值

用户指出的问题成立：

```text
如果所有 Latin bbox 都依赖 Paddle token，
Paddle 一崩，Latin 逻辑全崩。
```

因此需要把“文本真值来源”和“几何来源”拆开：

```text
TokenTruthSource:
  paddle_vl
  hanwang_engcut_consensus
  manual

GeometrySource:
  ppocrv5_line
  engcut_char_boxes
  hanwang_char_boxes
  manual_box
```

### 不用评分系统，使用确定性状态机

不要引入综合评分。使用离散状态：

```text
ACCEPT:
  token truth 和 char boxes 精确匹配。

REVIEW:
  无 token truth；
  或多源文本不一致；
  或跨行；
  或 EngCut/Hanwang 只有一方识别出 Latin；
  或符号不一致，例如 PE/VC -> PE!VC。

MANUAL:
  人工画框或人工指定 token。
```

## 建议的新接入顺序

### 情况 A：Paddle/VL 正常

```text
Paddle/VL parent text
  -> extract Latin token
  -> PP-OCRv5 line crop
  -> EngCut line char boxes
  -> exact token match
  -> accept EngCut bbox
```

这里 Paddle 只提供 token truth，不提供 Latin bbox。

### 情况 B：Paddle/VL 没给出 Latin token

不能直接相信 EngCut 的随机 Latin 噪声。只允许确定性共识：

```text
Hanwang final line text extracts Latin token
EngCut line text extracts same Latin token
same line
same reading order
exact token text equal
  -> accept as hanwang_engcut_consensus
else
  -> review
```

这不是评分，是双源一致性。

### 情况 C：Paddle/VL 给 token，但 EngCut 无 exact match

```text
不做模糊修正
不自动 PE!VC -> PE/VC
不自动 Lemer -> Lerner
进入 review
```

### 情况 D：人工框

```text
manual box + manual label/token
  -> highest geometry authority
  -> EngCut/Hanwang 只做识别回显
  -> 不自动覆盖 manual token
```

## 更新后的 AutoRec-lite 目标

AutoRec 原程序依赖：

```text
object flag
```

我们的等价物不应该只有 Paddle token，而应该是：

```text
dispatch unit + truth source + geometry source + review state
```

核心对象：

```text
LatinSpanUnit:
  parent_block_id
  line_id
  expected_text
  text_truth_source
  bbox
  geometry_source
  char_source
  status:
    accepted
    review
    manual
  reason
```

这样即使 Paddle/VL 失败，也不会全链路崩溃：

```text
Paddle/VL 成功:
  用 Paddle token 精确绑定 EngCut 字框。

Paddle/VL 失败:
  只接受 Hanwang 与 EngCut 的 exact consensus。
  没有 consensus 就 review，不胡猜。
```

## 下一步实验

需要做一个新的批量实验，不再使用 Hanwang 字符框作为 Latin 主定位：

```text
输入：
  30 页真实样例

路径 1:
  Paddle token -> PP-OCRv5 line -> EngCut line chars -> exact match

路径 2:
  无 Paddle token时：
    Hanwang line Latin token
    EngCut line Latin token
    exact consensus

输出：
  paddle_token_engcut_exact_count
  hanwang_engcut_consensus_count
  review_count
  cross_line_count
  symbol_mismatch_count
```

这个实验能直接回答：

```text
EngCut 是否能成为正文 Latin bbox 主路；
Paddle 失败时，有多少样例能靠 Hanwang+EngCut 双源一致性保住。
```
