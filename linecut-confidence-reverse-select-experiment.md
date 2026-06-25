# Linecut Confidence Reverse Select Experiment

## 问题

用户提出的新思路：

> linecut 有精确字符框，可以知道哪些是汉字；再用置信度过滤掉英文被强行识别成汉字的情况；剩下区域交给 EngCut 处理，得到更可靠的 Latin/英文框。

这次实验验证两点：

1. Hanwang/linecut 的 `confidence` 是否能区分“真实中文/标点”和“英文被误识别成中文/标点”。
2. 用高置信中文/标点框做占用区后，EngCut 反选出来的 Latin 框是否会挡掉真实英文。

## 置信度来源

当前 `confidence` 字段来自 Hanwang native 输出的 `scores[0]`，在 [app/engines/hanwang/micro_recblock.py](/mnt/d/project/ocr_process/worktrees/coord/app/engines/hanwang/micro_recblock.py:478) 中归一化：

```python
confidence = 1.0 - score / 100.0
```

也就是说：

- native score 越小越好。
- 项目内显示为 0 到 1 的 confidence。
- 这不是我们后处理随便构造的字段。

## 实验脚本

新增脚本：

- [scripts/experiment_linecut_conf_reverse_select.py](/mnt/d/project/ocr_process/worktrees/coord/scripts/experiment_linecut_conf_reverse_select.py)

输入：

- `debug/latin_recovery_batch_prose_all_v2/<page>/hanwang_echo.json`
- `debug/engcut_line_binding_batch_v2/<page>/engcut_line_binding.json`

输出：

- `debug/linecut_conf_reverse_select_batch_v1/linecut_conf_reverse_select_summary.md`
- `debug/linecut_conf_reverse_select_batch_v1/<page>/linecut_conf_reverse_select.json`
- `debug/linecut_conf_reverse_select_batch_v1/<page>/reverse_select_t0.50.png`

overlay 颜色：

- 蓝框：已知 Latin 真值框，即 Paddle token + EngCut exact 绑定结果。
- 绿色：Hanwang 高置信中文/中文标点占用框。
- 红色：反选保留下来的 EngCut Latin 字符框。
- 黄色：被中文占用区挡掉的 EngCut Latin 字符框。

## 22 页统计

样本为已有 `hanwang_echo.json` 的 22 页。

| threshold | truth chars | recalled chars | blocked truth chars | truth tokens | recalled tokens | blocked tokens | selected chars | selected extra chars | occupied boxes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 927 | 909 | 18 | 180 | 172 | 8 | 1727 | 818 | 15279 |
| 0.30 | 927 | 927 | 0 | 180 | 180 | 0 | 1944 | 1017 | 15018 |
| 0.50 | 927 | 927 | 0 | 180 | 180 | 0 | 1979 | 1052 | 14848 |
| 0.70 | 927 | 927 | 0 | 180 | 180 | 0 | 2244 | 1317 | 13925 |
| 0.85 | 927 | 927 | 0 | 180 | 180 | 0 | 5189 | 4262 | 10848 |
| 0.95 | 927 | 927 | 0 | 180 | 180 | 0 | 11465 | 10538 | 2380 |

解读：

- 阈值 `0.00` 会把低置信误识别中文也当占用区，因此挡掉 18 个真实 Latin 字符、8 个真实 Latin token。
- 阈值 `0.30` 到 `0.70` 都没有挡掉任何已知 Latin 真值。
- 阈值越高，中文占用区越少，EngCut 在中文区域吐出的 ASCII 噪声越多。
- 当前看 `0.30` 或 `0.50` 更合理。

## 置信度分布

| bucket | count | mean confidence |
| --- | ---: | ---: |
| Latin 真值区域内，Hanwang top 是 Latin ASCII | 911 | 0.463 |
| Latin 真值区域内，Hanwang top 是中文/中文标点 | 8 | 0.190 |
| Latin 真值区域外，Hanwang top 是中文/中文标点 | 15271 | 0.858 |
| Latin 真值区域外，Hanwang top 是 Latin ASCII | 847 | 0.652 |
| Latin 真值区域外，Hanwang top 是其他字符 | 298 | 0.584 |

关键点：

- 英文被 Hanwang 误识别成中文/中文标点时，当前样本里置信度非常低，均值约 `0.19`。
- 真正中文/中文标点的置信度明显高，均值约 `0.858`。
- 所以 confidence 可以参与中文占用区判断。

## 120194 例子

120194 单页在阈值 `0.50` 下：

- 已知 Latin token：24/24 召回。
- 已知 Latin char：109/109 召回。
- 没有真实 Latin 被中文占用区挡掉。

典型现象：

- `TFP` 被 Hanwang 正确识别为 `T/F/P` 时，置信度约 `0.19`，不会作为中文占用区。
- 另一个 `TFP` 被 Hanwang 识别成 `”7P` 时，`”` 的置信度也是 `0.19`，阈值 `0.50` 不会挡住 EngCut 的 `TFP`。
- `HHI` 一处被 Hanwang top 识别为 `脚`，置信度 `0.19`，同样不会挡住 EngCut。

## 重要限制

这个方案不能直接等价于“剩下的就是英文”。

原因是 EngCut 会在中文区域产生 ASCII 噪声，例如：

- `j`
- `p`
- `Bl`
- `WJ`
- 数字年份
- 公式下标或变量碎片

阈值 `0.50` 时，22 页里 927 个已知 Latin 字符全部召回，但还会额外留下 1052 个 EngCut ASCII 字符。这些额外字符不一定全是错误，其中有年份、变量、未被 Paddle token 覆盖的英文，也有噪声。

## 当前结论

1. 用户提出的方向成立：linecut 高置信中文/标点框可以作为“中文占用区”。
2. Hanwang confidence 在当前样本里对“真实中文”和“英文误判中文”有明显区分度。
3. 建议阈值从 `0.50` 起步实验，`0.30` 到 `0.70` 都没有挡掉已知 Latin 真值。
4. 反选结果不能直接落地为英文真值，必须进入下一层 token 校验。
5. 最可控的路线是：
   - Hanwang/linecut：提供中文/标点占用区。
   - EngCut：提供剩余区域的 Latin 字符候选框。
   - Paddle/VL、Hanwang 文本、人工框：提供 token 级真值或校验。
   - exact 通过则落地；不通过则进入人工 review。

## 建议接入形态

新增一个候选层，而不是直接改 OCR 主文本：

```text
physical line
  -> Hanwang CharRcg boxes
      -> high confidence CJK / Chinese punctuation = occupied boxes
  -> EngCut full line boxes
      -> remove chars blocked by occupied boxes
      -> group remaining Latin chars into candidates
  -> exact validate by Paddle/Hanwang/manual truth
      -> accepted Latin geometry
      -> review candidates
```

这比单纯依赖 Paddle token 更强，因为 Paddle token 崩掉时，仍然能从 Hanwang+EngCut 双路结果里生成候选；但它仍然不是无校验自动真值。

## 复验命令

```bash
python scripts/experiment_linecut_conf_reverse_select.py --out-dir debug/linecut_conf_reverse_select_batch_v1 --overlay-threshold 0.5
```
