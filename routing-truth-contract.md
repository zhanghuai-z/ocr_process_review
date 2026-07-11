# CharOCR 路由真值契约

本文档记录当前代码已实现的 CharOCR 输入边界。它不定义 UI 行为，也不替代代码或测试。

## 真值与派生物

| 对象 | 归属 | 是否可写 | 用途 |
| --- | --- | --- | --- |
| `LayoutSnapshot` | 版面编辑 | 仅版面编辑链路 | 当前页面的块类型、几何和 OCR 策略真值。 |
| Paddle VL 1.6 响应 | 原始外部事实 | 否 | 版面块、公式/表格/图片等结构事实及其文本。 |
| PP-OCRv6 prepass | 本次 OCR 输入事实 | 否 | 正文物理行、word-box、独立标点和受限拉丁 token observation。 |
| `PageRoutingPlan` | 路由编译器 | 否 | 由当前快照和本次 prepass 导出的、页面级、一次性派生计划。 |
| CharOCR observation | Hanwang/EngCut/显式 PP token 策略输出 | 仅 OCR 写入链路 | 正文字、字符框、来源和审计标记。 |

`PageRoutingPlan` 不是布局真值，也不会写回 `LayoutSnapshot`、Paddle 原始响应或项目模型。计划无效只阻断对应页面的 CharOCR；不会猜测整块 crop 继续执行。

## 当前生产链

1. Paddle VL 1.6 产生并确认当前 `LayoutSnapshot`。
2. PP-OCRv6 对整页产生行框和 word-box 提议。
3. `compile_page_routing_plan()` 先把行框裁到允许文本 OCR 的布局块内，再扣除公式、表格、图片和其他结构区域。
4. 对剩余正文区域：
   - 不含拉丁字母或数字：一个 `text_other` slice，交给 LineCut。
   - 纯拉丁/数字行：整条物理行作为一个 `text_latin` route，交给 EngCut。
   - 汉字与拉丁/数字混合行：行图先按边界背景自动判断黑底白字或白底黑字；每个连通域只能归属一个 token。中心落入 proposal 是第一优先级；符号侧向回收只根据初始 token 归属执行一次，不能借已回收组件向邻近单词级联。超长横线保留给非 Latin 区域。若拉丁字符与标点物理黏连，只取黏连组件和该拉丁 proposal 的交集。相邻 Latin mask 之间只有空白时合并；存在其他墨迹时生成独立 route。
   - 可唯一恢复完整组件的独立标点编译为 `text_symbol`，使用 PP 标点文本和组件联合 bbox 形成显式 observation，不进入 LineCut/EngCut。其空格仍作为无 bbox 的结构字符保留。
   - `text_latin` 只进入 EngCut。路由同时保留逐 token 的 PP 文本和组件 mask；当一个 EngCut group 与一个 PP token 通过中心几何唯一一对一绑定，且 EngCut 字符框重叠或文本不一致时，降级为带审计标记的 PP word observation。绑定不唯一时禁止替换。
5. `HanwangMicroRecBlockEngine` 只接收显式 `PageRoutingPlan`。native 行通过布局块 UID 映射到 typed routes，不能从 PP-VL 原始字典读取路由字段。缺少对应 text route 的 native 行会报错。
6. CharOCR 输出写入 OCR observation；公式文本仍由公式分支保有，不由 Hanwang 回填。

EngCut 只处理已编译为 `text_latin` 的 crop。PP token fallback 只能读取 typed
route 中与当前 group 唯一几何绑定的 observation；禁止恢复旧的全页文本搜索、模糊匹配或按字符串猜位置链路。

路由 segment 的 `bbox` 是行内 mask/crop 几何；公式和 grouped symbol 可另外保留 `content_bbox`。公式使用完整布局框；symbol 使用完整组件联合框。`text_latin` 还可携带只读 `ppocr_latin_tokens`，它们不是布局或校对真值。

## 2026-07-10 混合行实验结论

- 纯中文、纯英文/数字和混合行都必须先经过同一阶段的结构扣除；公式优先级高于文字分流。
- PP-OCR word-box 默认承担几何 proposal；只有 typed `text_symbol` 和唯一几何绑定的拉丁 word fallback 可以产生明确标源的 OCR observation。
- PP-OCR 的拉丁 token 只能生成自己的 mask；中文和 token 外区域属于 LineCut。可完整恢复的独立标点由 `text_symbol` 直接接管，不参与两个 native 引擎竞争。
- PP-OCR 的 word-token 串无法复现其 line text、出现 CJK/Latin 混合 token，或某个拉丁 token 没有可归属墨迹时，视为路由不完整并阻断该页。当前不使用“整行回退”“PP-OCR 文本回填”或静默猜测。
- 深色底白字和表格横线属于同一连通域分流算法的输入场景，不建立页面类型特判。
- PP token proposal 与黏连组件存在交集时可生成受限片段。除上述两种显式 observation 外，片段仍只是本次路由 mask。

## 退役路径

`_layout_line_routes` 和由 PP-VL 原始字典推导的旧 route 仅用于旧诊断/低层测试，不是生产 CharOCR 输入，也不应再被新代码写入或读取。新功能必须从 `LayoutSnapshot + PP-OCRv6 prepass -> PageRoutingPlan` 开始。

邮箱、网址和斜体均使用同一几何绑定契约，不按页面类型或字体名特判。公式误分类仍必须由布局/公式链处理；不得重新启用旧 raw route。
