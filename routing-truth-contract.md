# CharOCR 路由真值契约

本文档记录当前代码已实现的 CharOCR 输入边界。它不定义 UI 行为，也不替代代码或测试。

## 真值与派生物

| 对象 | 归属 | 是否可写 | 用途 |
| --- | --- | --- | --- |
| `LayoutSnapshot` | 版面编辑 | 仅版面编辑链路 | 当前页面的块类型、几何和 OCR 策略真值。 |
| Paddle VL 1.6 响应 | 原始外部事实 | 否 | 版面块、公式/表格/图片等结构事实及其文本。 |
| PP-OCRv6 prepass | 本次 OCR 输入事实 | 否 | 正文物理行与 word-box 几何提议；其文本不进入校对或导出真值。 |
| `PageRoutingPlan` | 路由编译器 | 否 | 由当前快照和本次 prepass 导出的、页面级、一次性派生计划。 |
| CharOCR observation | Hanwang/EngCut 输出 | 仅 OCR 写入链路 | 正文字、字符框和识别置信度。 |

`PageRoutingPlan` 不是布局真值，也不会写回 `LayoutSnapshot`、Paddle 原始响应或项目模型。计划无效只阻断对应页面的 CharOCR；不会猜测整块 crop 继续执行。

## 当前生产链

1. Paddle VL 1.6 产生并确认当前 `LayoutSnapshot`。
2. PP-OCRv6 对整页产生行框和 word-box 提议。
3. `compile_page_routing_plan()` 先把行框裁到允许文本 OCR 的布局块内，再扣除公式、表格、图片和其他结构区域。
4. 对剩余正文区域：
   - 不含拉丁字母或数字：一个 `text_other` slice，交给 LineCut。
   - 纯拉丁/数字行：整条物理行作为一个 `text_latin` route，交给 EngCut。
   - 汉字与拉丁/数字混合行：PP-OCR word-box 只提供 token proposal。行图先按边界背景自动判断黑底白字或白底黑字；每个连通域只能归属一个 token。中心落入 proposal 是第一优先级；偏移标点只能认领最近锚点及与其水平相连的同一符号组件，不能扩大认领邻近字母；剩余墨迹才由非符号 token 在各自搜索窗内认领。超长横线保留给非 Latin 区域，不能扩大 EngCut mask。若拉丁字符与标点物理黏连，只取黏连组件和该拉丁 proposal 的交集，不迁移完整标点组件。相邻 Latin mask 之间只有空白时合并；存在其他墨迹时生成 `text_other` slice。
5. `HanwangMicroRecBlockEngine` 只接收显式 `PageRoutingPlan`。native 行通过布局块 UID 映射到 typed routes，不能从 PP-VL 原始字典读取路由字段。缺少对应 text route 的 native 行会报错。
6. CharOCR 输出写入 OCR observation；公式文本仍由公式分支保有，不由 Hanwang 回填。

EngCut 只处理已编译为 `text_latin` 的 crop。旧的“Hanwang 识别完成后按
PP-OCR/文本猜测回填 EngCut 字符”的链路已删除，不能作为失败兜底重新接入。

路由 segment 的 `bbox` 是行内 mask/crop 几何；公式 segment 另外保留 `content_bbox`，即完整的布局公式框。两者在 PP-OCR 行只覆盖公式局部时会不同：前者保证 Hanwang 不读取公式像素，后者保证公式 atom、横校和导出仍使用真实公式位置。

## 2026-07-10 混合行实验结论

- 纯中文、纯英文/数字和混合行都必须先经过同一阶段的结构扣除；公式优先级高于文字分流。
- PP-OCR word-box 只承担几何 proposal，不承担最终文字真值。
- PP-OCR 的拉丁 token 只能生成自己的 mask；标点、中文和 token 外区域默认属于 LineCut，不参与 EngCut 的归属竞争。
- PP-OCR 的 word-token 串无法复现其 line text、出现 CJK/Latin 混合 token，或某个拉丁 token 没有可归属墨迹时，视为路由不完整并阻断该页。当前不使用“整行回退”“PP-OCR 文本回填”或静默猜测。
- 深色底白字和表格横线属于同一连通域分流算法的输入场景，不建立页面类型特判。
- PP token proposal 与黏连组件存在交集时可生成受限片段；片段仍只是本次路由 mask，不进入 OCR 文本、校对或导出真值。

## 退役路径

`_layout_line_routes` 和由 PP-VL 原始字典推导的旧 route 仅用于旧诊断/低层测试，不是生产 CharOCR 输入，也不应再被新代码写入或读取。新功能必须从 `LayoutSnapshot + PP-OCRv6 prepass -> PageRoutingPlan` 开始。

当前尚未解决的英文复杂场景（邮箱、网址、斜体和公式误分类）必须作为新的、可验证的路由/公式识别策略处理；不得重新启用旧 raw route 或把 PP-OCR 文本提升为业务真值。
