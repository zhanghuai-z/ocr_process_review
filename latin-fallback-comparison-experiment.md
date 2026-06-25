# Latin Fallback Comparison Experiment

## 目的

旧方案里 Latin token 有 15 个 `alignment_fallback_match`，也就是旧的 Hanwang 字符流 exact 找不到，只能靠 alignment fallback 勉强绑定。

这次实验验证新方案能否收口这些旧 fallback：

```text
primary:
  Paddle token + EngCut exact

fallback:
  linecut high-confidence CJK/punctuation occupancy
  + EngCut reverse-selected Latin candidates
```

## 实验脚本

新增：

- [scripts/experiment_old_fallback_vs_new_latin_strategy.py](/mnt/d/project/ocr_process/worktrees/coord/scripts/experiment_old_fallback_vs_new_latin_strategy.py)

输出：

- `debug/old_fallback_vs_new_latin_strategy/old_fallback_vs_new_latin_strategy.md`
- `debug/old_fallback_vs_new_latin_strategy/old_fallback_vs_new_latin_strategy.json`

## 结果

| 类别 | 数量 | 含义 |
| --- | ---: | --- |
| old alignment fallback items | 15 | 旧方案需要 fallback 的 Latin token |
| primary exact | 11 | 新主方案直接 exact 解决 |
| fallback exact after takeover | 1 | 新主方案失败，但反选 fallback 整行接管后 exact 解决 |
| fallback variant review | 3 | fallback 给出有用候选，但文本不 exact，必须 review |
| unresolved | 0 | 没有完全找不到候选的案例 |

## 逐项结论

| page | block | line | token | 旧 Hanwang span | 新主方案 | 新 fallback | 结论 |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| 120184 | 1 | 6 | `Other` | `Other` | exact | not needed | 主方案解决 |
| 120184 | 4 | 4 | `PE/VC` | `PE/VC` | exact | not needed | 主方案解决 |
| 120184 | 4 | 5 | `TFP` | `”7P` | exact | not needed | 主方案解决 |
| 120186 | 3 | 1 | `PE/VC` | `PE!VC` | failed | `PE` + `VC` | review，不自动修成 `PE/VC` |
| 120186 | 4 | 1 | `Lerner` | `Lemer` | failed | `Lemer` | review，不自动修成 `Lerner` |
| 120186 | 4 | 5 | `Guariglia` | `Gua吨lia` | exact | not needed | 主方案解决 |
| 120186 | 5 | 4 | `PE/VC` | `PE/VC` | failed | exact `PE/VC` | fallback 接管后解决 |
| 120187 | 2 | 2 | `PE/VC` | `PEfVC` | failed | `PEfVC` | review，不自动修成 `PE/VC` |
| 120192 | 2 | 5 | `PSM` | `PSM` | exact | not needed | 主方案解决 |
| 120192 | 7 | 7 | `Age` | `\de` | exact | not needed | 主方案解决 |
| 120192 | 7 | 7 | `Share` | `bAare` | exact | not needed | 主方案解决 |
| 120194 | 7 | 3 | `HHI` | `脚` | exact | not needed | 主方案解决 |
| 120194 | 7 | 7 | `TFP` | `”7P` | exact | not needed | 主方案解决 |
| 120194 | 10 | 2 | `TFP` | `”7P` | exact | not needed | 主方案解决 |
| 120194 | 10 | 4 | `TFP` | `”FP` | exact | not needed | 主方案解决 |

## 关键判断

1. 新主方案已经明显强于旧 Hanwang charstream 绑定。
   - 旧 15 个 fallback 中，11 个被 `Paddle token + EngCut exact` 直接解决。

2. 新 fallback 有价值，但必须保持轻量。
   - 120186 block 5 line 4 的 `PE/VC`，主方案失败，但反选 fallback 能 exact 找回。

3. fallback 不能做文本纠错。
   - `PE!VC`、`PEfVC`、`Lemer` 这些候选有几何价值，但不能自动改成 `PE/VC` 或 `Lerner`。
   - 这些必须进入 review/manual。

4. “fallback 全盘接手”的边界成立。
   - 当某物理行 primary 任意 token 失败，应丢弃该行 primary 落地结果。
   - fallback 接管整条 physical line，输出 exact accepted 或 review candidates。
   - 不允许同一行半段用 primary、半段用 fallback 将就拼接。

## 建议落地状态机

```text
for each physical line:
  run primary: Paddle token + EngCut exact

  if all Paddle Latin tokens exact:
      status = primary_accepted
      text_source = paddle_vl
      geometry_source = engcut_line_exact

  else:
      discard primary accepted fragments for this line
      run fallback takeover:
          linecut confidence occupied boxes
          EngCut reverse-selected Latin candidates

      if candidate exact matches trusted token:
          status = fallback_verified
          geometry_source = linecut_reverse_engcut_exact
      else:
          status = fallback_review
          keep candidate geometry and evidence
```

## 落地边界

- 自动落地只接受 exact。
- 不做 fuzzy。
- 不把 `PEfVC`、`PE!VC` 自动归一成 `PE/VC`。
- 不把 `Lemer` 自动改成 `Lerner`。
- fallback 默认接管粒度是 physical line。
- block/page 只用于上下文和排序，不能用于自动 token 搜索。

## 复验命令

```bash
python scripts/experiment_old_fallback_vs_new_latin_strategy.py
```
