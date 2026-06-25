# Hanwang Native Probe 调用次数实验

日期：2026-06-18

## 目标

验证 `linecut_recogimg_probe.exe` 的调用次数是否是 Hanwang 识别速度瓶颈，并确认可行的降调用方案。

当前主流程的安全形态是：每个 Hanwang 前输入单元 crop 单独启动一次 `linecut_recogimg_probe.exe`。这会导致一页 25-40 个 route 时启动 25-40 次 native 进程。

## 已验证的不可行路径

多 recblock 单次调用不可用：

- full page + 多个 page-space recblock：2 个及以上 recblock 触发 `System.AccessViolationException`
- vertical collage + 多个 recblock：2 个及以上 recblock 触发 `System.AccessViolationException`
- horizontal collage + 多个 recblock：2 个及以上 recblock 触发 `System.AccessViolationException`
- via-seg / no-via-seg、with-charrcg / no-charrcg、postprocess、split_mode 变体均未解决崩溃

结论：不能再沿用“拼图或多 recblock 后一次 native 调用”的 batch 方向。

## 可行路径

新增实验 wrapper：`linecut_recog_batch_probe.exe`

它不是把多个框传给一次 `Recog`，而是：

1. 启动一个 32 位 wrapper 进程
2. 初始化 Hanwang native 组件一次
3. 读取 task list
4. 对每个 crop 仍按单 recblock 安全调用 `SegImg -> Recog -> CharRcg`
5. 每个 crop 单独输出 JSON

这等价于“单进程循环处理多张 crop”，避免 native 多 recblock 崩溃，同时减少进程启动和初始化次数。

## 真实样例结果

实验输出目录：

- `debug/linecut_recog_probe_batch_full/120186/linecut_recog_probe_shapes.md`
- `debug/linecut_recog_probe_batch_full/120169/linecut_recog_probe_shapes.md`

### 120186

| route 数 | 当前 per-line 调用 | batch-list 调用 | 当前耗时 | batch 耗时 | 加速比 | 字符数一致 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 8 | 1 | 2.105s | 0.904s | 2.33x | 是 |
| 16 | 16 | 1 | 3.851s | 1.506s | 2.56x | 是 |
| 24 | 24 | 1 | 5.770s | 2.255s | 2.56x | 是 |
| 32 | 32 | 1 | 7.572s | 3.083s | 2.46x | 是 |
| 35 | 35 | 1 | 8.041s | 3.142s | 2.56x | 是 |

### 120169

| route 数 | 当前 per-line 调用 | batch-list 调用 | 当前耗时 | batch 耗时 | 加速比 | 字符数一致 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 8 | 1 | 1.942s | 0.799s | 2.43x | 是 |
| 16 | 16 | 1 | 3.668s | 1.287s | 2.85x | 是 |
| 24 | 24 | 1 | 5.526s | 2.090s | 2.64x | 是 |
| 28 | 28 | 1 | 6.312s | 2.224s | 2.84x | 是 |

## 20 页估算

现有对齐样例共 20 页、655 个 route，平均每页 32.75 个 route。

- 当前方式：约 655 次 `linecut_recogimg_probe.exe` 启动
- batch-list：约 20 次 wrapper 启动
- native 进程调用次数下降约 97%
- 按满页点估算，当前约 0.228s/route，batch-list 约 0.085s/route
- 20 页纯 LineCut Recog 段估算：约 149s -> 55s

这个估算只覆盖 Hanwang native recog 调用，不包含 Paddle、SegImg、裁图、UI 刷新、保存和 EngCut。

## 主程序路径 smoke

使用 `scripts/benchmark_hanwang_micro_recblock.py` 跑 120186，关闭 native cache。

输入：

- 图像：`file/244771纵校/120186.tif`
- layout：`file/244771纵校/120186.layout-api.json`

输出：

- batch-list：`debug/linecut_recog_probe_batch_full/120186/mainline_batch_smoke.json`
- per-line：`debug/linecut_recog_probe_batch_full/120186/mainline_per_line_smoke.json`

| 模式 | group 数 | Recog probe 调用 | SegImg | Recog | 总耗时 | group 失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| per-line | 35 | 35 | 4.82s | 9.94s | 20.51s | 0 |
| batch-list | 35 | 1 | 5.01s | 3.94s | 14.83s | 0 |

主程序路径中，batch-list 把 Recog 段降到原来的约 40%，总耗时降到原来的约 72%。剩余耗时主要来自 SegImg、EngCut、裁图/转换、Python 组装等非 Recog probe 调用。

## 结论

速度瓶颈中有一部分明确来自 native probe 调用次数。当前工程里已有的 collage batch 思路方向错误，应废弃。可接入的方案是 batch-list wrapper：每页或每批 route 只启动一次进程，进程内逐个安全处理 crop。

接入策略建议：

1. `native_bridge` 增加 `run_linecut_recog_batch_list()`，内部保持与 `run_linecut_recog()` 相同的缓存口径。
2. `micro_recblock` 在 group recog 阶段优先使用 batch-list，不再使用 multi-recblock/collage batch。
3. 如果 batch-list wrapper 缺失、失败或单个 task 报错，回退到当前 per-line 安全路径。
4. 保留 timing JSONL，后续用真实项目测量接入后 UI 端耗时。

## 当前接入状态

- 桌面应用入口 `main.py` 默认设置 `HANWANG_MICRO_RECBLOCK_BATCH=1`，正常启动程序时启用 batch-list。
- 测试和脚本默认不强行启用，仍可用环境变量显式控制。
- 如需临时关闭：启动前设置 `HANWANG_MICRO_RECBLOCK_BATCH=0`。
- WSL 开发环境已修正 native probe 路径传递：临时文件默认落到项目 `.cache/hanwang_native_work`，传给 Windows probe 时转换为 Windows 可读路径。
