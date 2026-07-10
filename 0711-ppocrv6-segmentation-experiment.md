# PP-OCRv6 字符分流扩大实验

日期：2026-07-11

## 范围

本轮使用 17 页真实样本：`file/0627` 5 页、`file/test2.pdf` 6 页、`file/43jpg(1).pdf` 6 页。PP-OCRv6 原始返回缓存在 `debug/extended_segmentation_ppocrv6/*/ppocrv6_raw`，不写入项目真值。

审计脚本 `scripts/audit_production_mixed_text_partition.py` 直接调用生产分流纯函数，不复制算法。最终审计覆盖 1,639 条 PP 物理行，其中 283 条为汉字与 ASCII 字母/数字混合行，路由问题为 0。既有 13 页项目使用 `LayoutSnapshot + PP-OCRv6` 完整重放，13/13 页可调度且无 validation issue。

## 已进入生产的结论

- 行图根据边界背景自动选择黑底白字或白底黑字前景，深色页眉不再判空。
- 一个连通域只允许一个 token owner，避免相邻 Latin mask 重复认领。
- 超长横线不能成为 Latin owner，表题下横线不再扩大或重叠英文 mask。
- Latin 字符与标点物理黏连时，仅取组件与 Latin proposal 的交集。
- 偏移标点只认领最近锚点及同一水平连接组件，不再吃掉后续 `China` 的 `C`。

证据：

- `debug/production_mixed_partition/0627/0001/mixed_lines/line_002_production_partition.png`
- `debug/production_mixed_partition/0627/0002/mixed_lines/line_047_production_partition.png`
- `debug/production_mixed_partition/0627/0006/mixed_lines/line_044_production_partition.png`
- `debug/production_mixed_partition/{0627,test2,43jpg6}/batch_summary.json`

## 下游实验结论

- 斜体英文：EngCut 字符几何可用，但文本不稳定。真实行中 `Psychiatry` 被识别为 `PsVchiatty`，不能仅靠继续调整 mask 修复。
- 邮箱：混合路由可稳定保留 `E-mail`、`zhengyi0928`、`126`、`com`，`@` 和句点不进入 Latin mask。
- 表格数字：`8.68%` 整行进入 EngCut 会得到 `8.68qc`。单独 LineCut 在足够 padding 下可识别 `%`，但句点、逗号和分号在高分辨率小框或整行符号蒙版中均不稳定。
- 公式：PP-OCR 文本和 word box 不能提供可靠公式语义。公式必须先由 `LayoutSnapshot` 作为结构原子扣除，再走公式识别；不应拆成普通字符竞争 LineCut/EngCut。

## 当前边界

PP-OCR token 文本仍是瞬时路由观察，不进入校对或导出真值。旧的“识别后用 PP 文本回填 EngCut”不能恢复。若后续引入 Latin word carrier，必须建立新的 typed token-hint contract、来源标记和 review 可见性，并先登记 fallback。

纯英文符号暂不改走 LineCut：现有证据只证明 `%` 个例有效，不能证明通用标点路径可靠。下一阶段应比较独立标点分类器或更可靠的 OCR 组件，不应增加字符替换枚举。
