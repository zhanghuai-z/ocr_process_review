# CharOCR 路由真值契约

本文档记录当前代码已实现的 CharOCR 输入边界。它不定义 UI 行为，也不替代代码或测试。

## 真值与派生物

| 对象 | 归属 | 是否可写 | 用途 |
| --- | --- | --- | --- |
| `LayoutSnapshot` | 版面编辑 | 仅版面编辑链路 | 当前页面的块类型、几何和 OCR 策略真值。 |
| Paddle VL 1.6 响应 | 原始外部事实 | 否 | 版面块、公式/表格/图片等结构事实及其文本。 |
| PP-OCRv6 prepass | 本次 OCR 输入事实 | 否 | 正文物理行、word-box、独立标点和受限拉丁 token observation。 |
| `RoutingObservationBundle` | OCR 路由观察层 | 否 | 将本次整页 PP 观察、当前范围 VL 观察、图像 hash 和版面 fingerprint 封装为同一运行范围。 |
| `PageRoutingPlan` | 路由编译器 | 否 | 由当前快照和本次 prepass 导出的、页面级、一次性派生计划。 |
| `OcrRoutingRunAudit` | 诊断审计 | 仅追加 | 持久化本次观察范围、精确对齐状态、路由摘要和失败原因，不参与业务决策。 |
| CharOCR observation | Hanwang/EngCut/显式 PP 例外输出 | 仅 OCR 写入链路 | 正文字、字符框、来源和审计标记；唯一绑定的 PP/VL 文本分歧只可成为候选，唯一行归属且不与 native atom 相交的 PP 独立标点可成为明确标源的缺失 atom。 |

`PageRoutingPlan` 不是布局真值，也不会写回 `LayoutSnapshot`、Paddle 原始响应或项目模型。计划无效只阻断对应页面的 CharOCR；不会猜测整块 crop 继续执行。

## 当前生产链

1. Paddle VL 1.6 产生并确认当前 `LayoutSnapshot`。版面编辑只能修改该快照；框几何或属性变化会使该块旧 OCR 观察失效。
2. 每次 OCR 都重新取得整页 PP-OCRv6 行/word 观察。未修改且范围完全相同的自动版面块可复用原始 VL 文本；新建、移动、缩放或改类型的文本块必须按当前 bbox 裁图重新请求 VL，失败重试一次。
3. `RoutingObservationBundle` 绑定 image hash、layout fingerprint、PP run 和块级 VL 观察。成功但为空的 VL 观察只产生非阻断诊断；传输或解析失败阻断该页。
4. VL 块文本与其内部 PP 物理行只允许块内、单调、去空白后的精确对齐。对齐不唯一或文本不等时阻断该页，不做模糊匹配。VL 明确给出的圈号/括号编号可在精确正文对齐后建立 LineCut 前缀所有权。
5. `compile_page_routing_plan()` 先在版面块内解析物理行，再扣除公式、表格、图片和目录装饰线，最后编译 CharOCR 输入。
   - PP 行框只是物理行 seed。最终行几何由页面前景组件解析，但组件不得越过版面块所有权、结构排除区或另一条物理行的中心。
   - 同一基线的 PP 碎片可以合成一条物理行；跨行页边线、表格线和其他纵向规则不能扩张行框。
   - 物理行解析只改变本次路由几何，不写回 `LayoutSnapshot` 或 PP 原始响应。
6. 对剩余正文区域：
   - 不含拉丁字母或数字：保留一个 `text_other` 行路由，交给 LineCut。
   - 含拉丁字母或数字：PP word box 只提供 token proposal，页面前景组件提供实际墨迹。每个完整组件只能有一个所有者；拉丁/数字 token 以 word 为单位生成 `text_latin` 路由，剩余墨迹仍属于同一物理行的 LineCut 输入。
   - 标点不是第三个 native 分支。拉丁 token 内的 ASCII 标点可以随 word 进入 EngCut；外部标点和中文标点留在 LineCut。原生探针已验证 EngCut 可逐字符返回 `[`、`]` 和 `/`。
   - PP 标点 token 的空白范围不得切伤相邻拉丁字形。只有与唯一拉丁所有者共享水平墨迹投影、且未被显式 LineCut 所有权占用的完整组件可以归还该拉丁 token；竞争归属保持未决。单字符拉丁 token 只拥有小块墨迹时，仅在另外一个组件满足同字形的方向、间距和整体尺寸约束且候选唯一时补全几何，并记录 `single_latin_token_geometry_completed`；否则产生阻断 issue `incomplete_single_latin_token_ink`，不以残缺框继续 OCR。
   - 可唯一恢复完整组件的独立标点可保留为 `PpOcrSymbolObservation`。若 observation 与一个 native atom 严格一对一绑定且文本不同，只成为带独立来源和 bbox 的 `OcrCandidate`。若它唯一归属一条物理行、且 bbox 与任何 native atom 均不相交，可按 bbox 横向顺序创建明确标源的缺失符号 atom。多个行所有者、多个 native atom 所有者或与未认领 native atom 相交时均保持未绑定；不得替换或重排 native atom，也不能按文本邻近关系猜位置。
   - `text_latin` 只进入 EngCut。路由同时保留逐 token 的 PP 文本、原始 proposal bbox 和组件 mask。EngCut native group 的字符 bbox 不重叠时，CharOCR 文本和字符框保持主观察；唯一一对一且等长的 PP 文本分歧只保存为候选。若 native group 内字符 bbox 重叠，则字符几何和对应文本同时降级；只有该 segment 恰有一个 PP word token、且已有唯一前景所有权生成的 route bbox 时，才聚合同一 token 的连续 native groups，以 PP token 文本和 route 前景闭合 bbox 生成一次明确标源的 word atom，聚合 native 文本只作为外部候选。PP 原始 word bbox 仍是只读 observation，不成为最终 atom 几何。没有唯一 PP token、没有前景闭合几何或同 token groups 不连续时阻断，不保留损坏字符框。
7. `HanwangMicroRecBlockEngine` 只接收显式 `PageRoutingPlan`。native 行通过布局块 UID 映射到 typed routes，不能从 Paddle 原始字典读取路由字段。缺少对应 text route 的 native 行会报错。
8. 路由运行摘要以追加记录持久化；CharOCR 输出写入 OCR observation。PP 标点候选绑定、缺失 atom 插入和未绑定分别进入运行统计；插入 atom 的 `text/bbox/source` 持久化到 OCR observation。`ppocr_symbol_missing_native_atom` 是执行期行 review flag，当前 `OcrLine` 持久化模型不保存行 review flags。公式文本仍由公式分支保有，不由 CharOCR 回填。

EngCut 只处理已编译为 `text_latin` 的 crop。EngCut 对已有拉丁 route 完全无输出时，可采用该 typed route 自带的 PP 文本生成明确标源的 word fallback；EngCut native group 字符框重叠时，可由唯一绑定的 PP word token 接管文本，并由该 token 唯一前景所有权生成的 route bbox 接管最终 word 几何。原始 PP word bbox 不直接成为最终 atom 几何。除此之外，非空且几何可靠的 CharOCR 文本不得与 PP/VL 择优；禁止恢复旧的全页文本搜索、模糊匹配、整行静默回填或按字符串猜位置链路。

路由 segment 的 `bbox` 是行内 mask/crop 几何；公式可另外保留完整布局框 `content_bbox`。`text_latin` 可携带只读 `ppocr_latin_tokens`，行还可携带 `PpOcrSymbolObservation`；它们都不是布局或校对真值。

## 已采纳的实验结论

- 纯中文、纯英文/数字和混合行都必须先经过同一阶段的结构扣除；公式优先级高于文字分流。
- PP-OCR word-box 默认承担几何 proposal；唯一绑定的等长拉丁文本分歧只能产生明确标源的 `OcrCandidate`。独立标点除候选路径外，只在唯一行归属且与全部 native atom 不相交时生成明确标源的缺失 atom，不能替换已有 CharOCR 正文。
- PP-OCR 的拉丁 token 只能生成自己的 word mask；中文、外部标点和 token 外区域属于 LineCut。独立标点 observation 只提供文本与组件证据，不成为第三个 native 分支。
- PP-OCR 的 word-token 串无法复现其 line text、出现 CJK/Latin 混合 token，或某个拉丁 token 没有可归属墨迹时，视为路由不完整并阻断该页。除 EngCut 空结果的显式 route-local word fallback 和上述独立标点缺失 atom 外，当前不使用整行回退、PP 文本回填或静默猜测。
- 深色底白字和表格横线属于同一连通域分流算法的输入场景，不建立页面类型特判。
- PP token proposal 与黏连组件存在交集时可生成受限片段。除上述两种显式 observation 外，片段仍只是本次路由 mask。
- 路由失败只阻断对应页面；同批其他页面可继续。失败页必须保留可审计 issue，不能退回整块 OCR。

## 退役路径

`_layout_line_routes`、`linecut_token_indices`、脚注数字位置猜测和由 Paddle 原始字典推导 route 的路径均已退出生产契约。新功能必须从 `LayoutSnapshot + RoutingObservationBundle -> PageRoutingPlan` 开始。

邮箱、网址和斜体均使用同一几何绑定契约，不按页面类型或字体名特判。公式误分类仍必须由布局/公式链处理；不得重新启用旧 raw route。
