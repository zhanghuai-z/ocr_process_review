# EngCut Line Binding Experiment

## 目的

这次实验只验证一个问题：当 Paddle/VL 给出正文里的 Latin token，但没有给出可靠字符框时，能否用“整行裁图 -> Eng20/EngCut 切字 -> token 精确绑定”的方式拿到稳定几何框。

实验不改主程序，只写入 `debug/engcut_line_binding_batch_v2/`。

## 输入和输出

- 输入 1：`debug/latin_recovery_batch_prose_all_v2/<page>/dispatch_units.json`
  - 来自 Paddle/VL 的 block 文本。
  - `latin_hints` 是从 Paddle/VL 文本中抽出的 Latin token，例如 `PE/VC`、`TFP`、`Specialization`。
- 输入 2：`debug/latin_recovery_batch_prose_all_v2/<page>/hanwang_echo.json`
  - 当前 Hanwang 前工作流输出的物理行框。
  - 这里暂时把它当作“整行裁图来源”，不是最终方案承诺。
- 输出：
  - `debug/engcut_line_binding_batch_v2/engcut_line_binding_summary.md`
  - `debug/engcut_line_binding_batch_v2/<page>/engcut_line_binding.json`
  - `debug/engcut_line_binding_batch_v2/<page>/page_overlay.png`
  - `debug/engcut_line_binding_batch_v2/<page>/lines/*.png`
  - `debug/engcut_line_binding_batch_v2/<page>/lines/*.eng20.overlay.png`

## 当前算法

1. 按 `hanwang_echo.json` 的物理行框从原图裁整行。
2. 调用 `resources/hanwang_native/bin/eng20_probe.exe` 的 `recogline_engstr`。
3. Eng20 返回字符码和字符框。
4. 非 ASCII/未知字符统一映射成单字符占位符 `~`，保证“文本 offset”和“字符框数组 offset”一一对应。
5. 对 Paddle token 只做 exact substring 匹配。
6. 加入 source-order 约束：当前 token 不能越过后续 token 去匹配，否则拒绝。
   - 例：如果前一个 `PE/VC` 被 Eng20 读成 `PEfVC`，不能把它错绑到后面的 `OtherPE/VCFunds` 里面。
7. 匹配失败、不完整、顺序冲突都不自动接收，进入 review/manual。

## 统计结果

样本：已有 `hanwang_echo.json` 的 22 页。

| 指标 | 数量 |
| --- | ---: |
| pages | 22 |
| physical lines | 365 |
| Paddle Latin hints | 189 |
| exact accepted | 180 |
| not found/rejected | 9 |
| incomplete bbox | 0 |
| Hanwang+Eng20 consensus candidates | 70 |
| consensus-only candidates | 13 |

120194 单页结果：24/24 exact accepted。

## 失败项

剩余 9 个未接收项集中在这些类型：

- `PE/VC` 被 Eng20 识别为 `PEfVC`、`PE!VC`、`PElVC` 等变体。
- `Lerner` 被 Eng20 识别为 `Lemer`。
- `VC` 出现在低质量或粘连文本中，exact 找不到。
- `120184 block=1 PE/VC` 被 source-order 约束拒绝：它本来应该匹配前面的 `PE/VC`，但 Eng20 只在后面的 `OtherPE/VCFunds` 中 exact 找到 `PE/VC`，自动接收会错绑。

这些失败不证明行框不可用，更多是 Eng20 英文识别文本和 Paddle token 的 exact 字符串不一致。

## 当前结论

1. “整行 Eng20 切字 + Paddle token exact 绑定”可以作为 Latin 字符框召回主方案之一。
2. 必须保留 source-order 约束，否则重复 token 和复合词会产生错绑。
3. 不能把 fuzzy、归一化、编辑距离直接放进自动接收路径；目前只建议进入 review/manual。
4. 当 Paddle token 崩掉时，不能全靠 Paddle。可以补一个保守候选来源：
   - 同一物理行内，Hanwang 文本和 Eng20 文本出现完全一致的 Latin token。
   - 这类结果标记为 `hanwang_engcut_consensus_only`。
   - 它不是自动真值，只能作为 UI 待确认候选。

## 建议接入方案

先不改 OCR 主文本流，只接几何候选层：

1. `paddle_vl` 仍是 Latin token 的第一真值来源。
2. `engcut_line_exact` 负责给 Paddle token 找字符框。
3. exact 成功：写入 Latin token 的 geometry source。
4. exact 失败：不修改文本，不生成伪框，进入人工 review。
5. Paddle 没给 token，但 Hanwang+Eng20 同行 exact 共识存在：作为候选提示，不自动落地。

这条路线符合当前原则：VL 给 block/inline，行框来自物理行，Latin 字符框来自 Eng20，所有不确定项都不自动吞掉。

## 复验命令

```bash
python scripts/experiment_engcut_line_binding.py --out-dir debug/engcut_line_binding_batch_v2 --pages 120167 120168 120171 120172 120173 120174 120175 120183 120184 120185 120186 120187 120188 120189 120190 120191 120192 120193 120194 120195 120196 120197
```

如果要强制重新调用 native Eng20：

```bash
python scripts/experiment_engcut_line_binding.py --out-dir debug/engcut_line_binding_batch_v2 --force --pages 120167 120168 120171 120172 120173 120174 120175 120183 120184 120185 120186 120187 120188 120189 120190 120191 120192 120193 120194 120195 120196 120197
```
