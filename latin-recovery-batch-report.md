# Latin Recovery 批量实验报告

本文记录 120166~120197 共 30 页真实样例上的 Latin span recovery 扩大实验。

## 实验目标

验证这条链路是否可行：

```text
Paddle/VL 父块文本
  -> 抽取正文 Latin hint
  -> Hanwang 回显字符框
  -> exact / 文本对齐找回 bbox
  -> crop
  -> Eng20
```

这里的 Latin recovery 只针对正文里的英文词、缩写、邮箱、机构名等，不针对 `$...$` 里的公式变量。公式变量已经由 Paddle inline formula 负责，不应该拆成单独 Latin token 再送 Eng20。

## 实验修正

扩大实验中修正了三个实验脚本问题：

```text
1. Hanwang 字符流必须由 line.chars[*].text 组成。
   不能用 line.text 做下标，因为 line.text 可能包含 Paddle inline formula carrier。

2. Latin hint 默认跳过 $...$ 内部 token。
   例如 \ln / TFP / it / GGF / Post-short 在公式中不是正文 Latin span。

3. Hanwang 回显目标标签补齐 figure_title / caption / table_title 等文本类标签。
   之前 120189 / 120191 的 PE/VC 未召回，是实验目标列表漏了 figure_title。
```

## 最终结果

最终输出：

```text
debug/latin_recovery_batch_prose_all_v2/latin_recovery_batch_summary.md
debug/latin_recovery_batch_prose_all_v2/latin_recovery_batch_summary.json
```

统计：

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

结论：

```text
1. 定位链路成立。
   正文 Latin hint 在这批样例里可以 189/189 找回 bbox。

2. alignment fallback 是必要的。
   15 个 token 需要靠 Paddle/VL 真值文本和 Hanwang 回显文本对齐找回。

3. Eng20 对干净独立 crop 效果较好，但不能盲目覆盖正文。
   严格等于 token 是 177/189；多数 mismatch 是前后粘连字符或括号。
```

## 剩余问题

严格 mismatch 中，很多是可容忍的边界字符：

```text
GDP       -> rGDP
Age       -> (Age)
zju.edu.cn -> )zju.edu.cn
nsd.pku.edu.cn -> )nsd.pku.edu.cn
```

真正需要人工/规则复核的内容级差异：

```text
120186 PE/VC  -> PE!VC
120186 Lerner -> Lemer
120186 PE/VC  -> VC~~...Carpenter...Petersen...
120187 PE/VC  -> PEfVC
```

其中第三个 `PE/VC` 是换行断裂导致的过宽 crop：

```text
上一行末尾：PE/
下一行开头：VC
```

这类跨行 token 不能直接取 union 后整块送 Eng20，否则会吃进下一行正文。产品化时需要按行拆 crop，或保持为 review-only。

## AutoRec-lite 含义

扩大实验进一步支持 AutoRec-lite 的设计：

```text
Paddle/VL:
  提供父块文本真值和 inline formula 真值。

Hanwang:
  提供正文字符框和中文 OCR。

Latin recovery:
  只为正文 Latin span 找回 bbox，并独立验证 Eng20。

Eng20:
  不能直接覆盖正文，只能作为调试/横校辅助结果。
```

首版产品化建议：

```text
1. 新增 app/core/latin_span_recovery.py。
2. 默认跳过 $...$ 公式内部 token。
3. 对 exact / alignment fallback 都生成 review flag。
4. Eng20 结果先进入调试视图或横校并排显示，不直接替换 Hanwang 文本。
5. 跨行 token、Eng20 不一致、过宽 crop 必须人工复核。
```

## Native 风险

批量实验中 Hanwang `linecut_recogimg_probe.exe` 多次触发 `AccessViolationException`，但当前 `run_micro_recblock` 有 guard，会禁用 batch 并降级继续。

产品化时应保留这个策略：

```text
batch 识别失败
  -> 自动禁用 batch
  -> 逐组/小批量调用
  -> 不让 native crash 中断整页处理
```
