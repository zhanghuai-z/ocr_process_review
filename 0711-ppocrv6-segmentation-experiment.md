# PP-OCRv6 字符分流扩大实验

日期：2026-07-11

## 范围

本轮使用 17 页真实样本：`file/0627` 5 页、`file/test2.pdf` 6 页、`file/43jpg(1).pdf` 6 页。PP-OCRv6 原始返回缓存在 `debug/extended_segmentation_ppocrv6/*/ppocrv6_raw`，不写入项目真值。

审计脚本 `scripts/audit_production_text_partition.py` 直接调用生产分流纯函数，不复制算法。扩展样本共 1,639 条 PP 物理行，其中 937 条含 ASCII 字母/数字并进入分区；发现 2 条旋转图表边缘小字产生 Latin mask 重叠，作为页级 validation issue 暴露，不做整行兜底。另对 `binary_600dpi` 13 页、345 条含 Latin/digit 行重放，路由问题为 0。

## 已进入生产的结论

- 行图根据边界背景自动选择黑底白字或白底黑字前景，深色页眉不再判空。
- 一个连通域只允许一个 token owner，避免相邻 Latin mask 重复认领。
- 超长横线不能成为 Latin owner，表题下横线不再扩大或重叠英文 mask。
- Latin 字符与标点物理黏连时，仅取组件与 Latin proposal 的交集。
- 偏移标点依据已归属组件的邻近关系收回自身墨迹，不再吃掉后续 `China` 的 `C` 或窄字母 `i`。
- 纯英文行与中英混合行使用同一分区器；只有 Latin/digit mask 进入 EngCut，标点留给 LineCut。
- 一个符号 token 含多个独立符号时，按真实墨迹间最大空白拆成独立 LineCut 区域；PP token 仅提供符号数量，不提供最终 OCR 文本。

证据：

- `debug/production_text_partition_binary13_final/page_0002_binary_600dpi/routed_lines/line_002_production_partition.png`
- `debug/production_text_partition_all_latin_final/0627/0002/routed_lines/line_047_production_partition.png`
- `debug/production_text_partition_all_latin_final/0627/0006/routed_lines/line_044_production_partition.png`
- `debug/production_text_partition_all_latin_final/{0627,test2,43jpg6}/batch_summary.json`

## 下游实验结论

- 斜体英文：EngCut 字符几何可用，但文本不稳定。真实行中 `Psychiatry` 被识别为 `PsVchiatty`，不能仅靠继续调整 mask 修复。
- 邮箱：混合路由可稳定保留 `E-mail`、`zhengyi0928`、`126`、`com`，`@` 和句点不进入 Latin mask。
- 表格数字：`8.68%` 整行进入 EngCut 会得到 `8.68qc`，因此整行 EngCut 路径已退出生产。当前数字进入 EngCut，符号区域进入 LineCut；表格仍须先由 `LayoutSnapshot` 排除出正文 CharOCR 路由。
- 公式：PP-OCR 文本和 word box 不能提供可靠公式语义。公式必须先由 `LayoutSnapshot` 作为结构原子扣除，再走公式识别；不应拆成普通字符竞争 LineCut/EngCut。

## 当前边界

PP-OCR token 文本仍是瞬时路由观察，不进入校对或导出真值。旧的“识别后用 PP 文本回填 EngCut”不能恢复。若后续引入 Latin word carrier，必须建立新的 typed token-hint contract、来源标记和 review 可见性，并先登记 fallback。

纯英文符号不再随整行进入 EngCut。实际页验证表明，组合区域 `” “` 一次送 LineCut 会得到四个低置信碎片框；按自然空白拆成两个几何簇后，LineCut 分别输出 `”`（0.89）与 `“`（0.88）的完整框。该过程不使用 PP 文本回填，也不增加字符替换枚举。
