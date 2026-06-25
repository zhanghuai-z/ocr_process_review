# Paddle/Hanwang 对齐离线实验

实验时间：2026-06-09

实验脚本：`scripts/experiment_paddle_hanwang_alignment.py`

实验样例：

- 图片：`tests/fixtures/real_samples/244771/120166.tif`
- Paddle VL 版面 fixture：`tests/fixtures/layout/120166-layout-api-fixture.json`
- Paddle OCR 行提示 fixture：`tests/fixtures/layout/120166-page-ocr-lines.json`

本实验只读本地 fixture，不调用 Paddle 或汉王外部服务。

## 目标

验证人工画出的公式、表格、图片框，是否可以反查 Paddle 已返回的内容，并与 Hanwang 的文字切片机制对齐。

本次样例只有行内公式，没有表格/图片真实块，因此本报告只能确认公式对齐方案；表格和图片仍需要后续补充真实 fixture。

## 关键链路

当前链路可以拆成四层：

1. Paddle 父级版面块
   - 来源：`parsing_res_list`
   - 关键字段：`block_label`、`block_bbox`、`block_content`
   - 作用：提供父级段落框和完整文本内容，其中 `block_content` 保留了 LaTeX 公式文本。

2. Paddle 几何子块
   - 来源：`layout_det_res.boxes`
   - 关键字段：`label=inline_formula/display_formula/formula_number/...`、`coordinate`
   - 作用：提供行内公式等子元素的位置。

3. Paddle OCR 行提示
   - 来源：`overall_ocr_res` 或测试 fixture 中抽出的行级 OCR 结果
   - 关键字段：`text`、`bbox`、`score`
   - 作用：把父级段落拆成真实物理行，帮助 route 知道公式属于哪一行。

4. Hanwang text slice
   - 来源：`text_slice_routes_for_block`
   - 作用：只把公式左右两侧的文字区域送入汉王，公式段由 Paddle 文本插回。

因此正确策略不是让 Hanwang 识别公式，而是：

- Paddle 已检出的公式：用 Paddle 几何框定位，用 Paddle 父级文本恢复公式内容。
- Paddle 漏几何框但父级文本有公式：用人工框的物理行位置反查父级公式 span。
- 只有真正无法反查时，才进入人工补录或外部公式 OCR。

## 120166 样例观测

目标父级 text block：

- bbox：`[291, 1813, 2158, 2501]`
- 父级 `block_content` 内公式 span 数：8
- Paddle 几何 `inline_formula` 小框数：7
- Paddle OCR 行提示数：7
- Hanwang 将收到的 text slice 数：14

父级公式 span：

1. `$ Y_{ct} $`
2. `$ Incentive_{c} \times Post_{t} $`
3. `$ Incentive_{c} $`
4. `$ Post_{t} $`
5. `$ X_{ct} $`
6. `$ \delta_{c} $`
7. `$ \varphi_{t} $`
8. `$ \varepsilon_{ct} $`

Paddle 几何小框只覆盖了其中 7 个，漏掉的是第 3 个：

- `$ Incentive_{c} $`
- 物理行：第 3 行
- Paddle OCR 行文本：`Incentive。表示2016年增值税分成改革下各地级市结构性财政激励程度，具体构造`

这说明 Paddle OCR 行文本自身已经损坏了下标，但父级 `block_content` 仍然保存了 `$ Incentive_{c} $`。因此不能只信 OCR 行文本，也不能只信几何小框；必须把父级文本、几何小框、物理行三者合并判断。

## 当前 route 结果

第 1 行：

- text `[395, 1808, 1466, 1873]` -> Hanwang
- formula `[1466, 1808, 1518, 1873]` -> `$ Y_{ct} $`
- text `[1518, 1808, 2151, 1873]` -> Hanwang

第 2 行：

- text `[297, 1886, 1042, 1953]` -> Hanwang
- formula `[1042, 1886, 1144, 1953]` -> `$ Incentive_{c} \times Post_{t} $`
- text `[1144, 1886, 2149, 1953]` -> Hanwang

第 3 行：

- text `[294, 1962, 2151, 2031]` -> Hanwang
- 这里缺失 `$ Incentive_{c} $` 的几何公式段

第 4 行：

- text `[297, 2042, 844, 2107]` -> Hanwang
- formula `[844, 2042, 946, 2107]` -> `$ Post_{t} $`
- text `[946, 2042, 2151, 2107]` -> Hanwang

第 5 行：

- text `[297, 2120, 604, 2185]` -> Hanwang
- formula `[604, 2120, 665, 2185]` -> `$ X_{ct} $`
- text `[665, 2120, 2151, 2185]` -> Hanwang

第 6 行：

- text `[297, 2193, 2149, 2263]` -> Hanwang

第 7 行：

- text `[294, 2271, 623, 2343]` -> Hanwang
- formula `[623, 2271, 669, 2343]` -> `$ \delta_{c} $`
- text `[669, 2271, 729, 2343]` -> Hanwang
- formula `[729, 2271, 776, 2343]` -> `$ \varphi_{t} $`
- text `[776, 2271, 1737, 2343]` -> Hanwang
- formula `[1737, 2271, 1793, 2343]` -> `$ \varepsilon_{ct} $`
- text `[1793, 2271, 2153, 2343]` -> Hanwang

## 人工框 probe 结果

### 1. 覆盖已检出的 `Incentive x Post`

人工框：

- bbox：`[1030, 1888, 1165, 1960]`

结果：

- 状态：`PADDLE_GEOMETRY_HIT`
- 命中：`$ Incentive_{c} \times Post_{t} $`
- line：2
- score：0.896

结论：

- 可直接使用 Paddle 父级文本恢复公式内容。
- 该区域应从 Hanwang text slice 中挖掉，避免 Hanwang 误识别公式。

### 2. 覆盖 Paddle 漏几何框的 `$ Incentive_c $`

人工框：

- bbox：`[292, 1958, 620, 2038]`

结果：

- 状态：`PARENT_TEXT_LINE_INFERRED`
- 无 Paddle 几何小框
- 同一物理行存在唯一未解析公式：`$ Incentive_{c} $`

结论：

- 可实现反查。
- 绑定来源应记录为“人工几何 + Paddle 父级文本”。
- 必须打 review flag，因为它不是 Paddle 已检出的几何小框。

### 3. 宽框同时覆盖 `delta` 和 `varphi`

人工框：

- bbox：`[610, 2278, 790, 2348]`

结果：

- 状态：`AMBIGUOUS_GEOMETRY_HIT`
- 候选：
  - `$ \varphi_{t} $`，score=0.745
  - `$ \delta_{c} $`，score=0.744

结论：

- 不能静默合并为一个公式。
- UI 应提示多个候选，或要求用户缩小框/选择公式文本。

### 4. 人工框无父级公式真值

人工框：

- bbox：`[1880, 2045, 2050, 2105]`

结果：

- 状态：`EMPTY_FORMULA_REVIEW_BOX`
- 没有 Paddle 几何命中
- 没有可唯一召回的父级公式真值

结论：

- 不触发 crop 重识别作为主路径。
- 保留空公式校验框，等待人工补录或后续专门公式 OCR。

## 扩大实验面汇总

当前仓库只有 1 份真实 layout fixture，因此这次“扩大实验面”先扩成自动扫描器和自动 probe。后续只要把更多真实样例 fixture 放入 `tests/fixtures/layout`，脚本会自动纳入统计。

运行命令：

```bash
python scripts/experiment_paddle_hanwang_alignment.py
```

扫描结果：

- layout fixture 数：1
- 含公式上下文的父级记录数：3
- 父级公式真值 span 总数：10
- Paddle 几何公式框总数：7
- route formula segment 总数：7
- 未被几何小框覆盖但可进入父级召回检查的 span 总数：1
- 已检公式自动 probe：`PADDLE_PARENT_FORMULA_HIT=2`，`PADDLE_GEOMETRY_HIT=7`
- 漏几何公式自动 probe：`PARENT_TEXT_LINE_INFERRED=1`

逐记录结果：

- `record=7`，`display_formula`，父框自身就是公式：`PADDLE_PARENT_FORMULA_HIT`
- `record=9`，正文 text block：8 个父级公式 span，7 个几何公式框，1 个漏几何但可父级召回
- `record=14`，页脚 formula-style block：`PADDLE_PARENT_FORMULA_HIT`

修正后的统计口径：

- `display_formula` 和 formula-style 页脚不算行内漏框，因为它们的父框自身已经是公式。
- 真正的行内漏框只有正文第 3 行 `$ Incentive_{c} $`。
- 当前没有出现“一行漏多个公式”的真实证据。

## `file/244771纵校` 真实文件实验

用户要求不只使用 `tests/fixtures`，因此已直接对 `file/244771纵校` 下 30 张真实 `tif` 调用 PaddleOCR-VL-1.6，并在同目录生成本地缓存：

- 输入：`file/244771纵校/*.tif`
- 输出：`file/244771纵校/*.layout-api.json`
- 页数：30
- 命令：

```bash
python scripts/experiment_paddle_hanwang_alignment.py --file-dir 'file/244771纵校' --generate-file-layouts --token-from-sample-script
python scripts/experiment_paddle_hanwang_alignment.py --file-dir 'file/244771纵校'
```

全量扫描摘要：

- 生成 Paddle layout 缓存：30 个
- 含公式/表格/图表上下文的页：21 个
- 含公式上下文的父级记录：51 条
- 父级公式真值 span：142 个
- Paddle 几何公式框：45 个
- route formula segment：51 个
- 父框自身公式命中：31 条
- 父级 table 命中：9 条
- chart 父框：3 条

修正后的缺口分类：

- 正文实质公式缺口：3 条
- 脚注公式缺口：1 条
- 脚注/显著性标记缺口：18 个
- 表格内部公式不计入行内公式缺口，表格按父级 table 真值整体绑定。

正文实质公式缺口：

1. `120166.layout-api.json`
   - record：9
   - label：`text`
   - 缺口：1
   - 父级 span 包含：`$ Incentive_{c} $`
   - 说明：正文行内公式内容存在于父级 `block_content`，但缺少对应 `inline_formula` 几何框。

2. `120193.layout-api.json`
   - record：3
   - label：`text`
   - 缺口：1
   - 父级 span：`$ GGF_{it}^{Post-short} $` / `$ GGF_{it}^{Post-long} $` 重复出现
   - 说明：父级真值存在，几何框少一个；人工框需要按父级公式顺序或更细粒度上下文召回。

3. `120193.layout-api.json`
   - record：12
   - label：`text`
   - 缺口：1
   - 父级 span：`$ y, k, l, m $`、`$ \omega_{i} $`
   - 说明：两个公式 span 中只有一个几何框，人工框可召回父级真值；若无法唯一确定则保持候选/空校验。

脚注公式缺口：

- `120169.layout-api.json`
  - record：13
  - label：`footnote`
  - span：`$ 0.133\times0.373\div0.147\approx0.337 $`
  - 说明：脚注内公式父级真值存在，但没有独立几何框。

脚注/显著性标记缺口主要是：

- `$ ^{①} $`
- `$ ^{②} $`
- `$ ^{*} $`
- `$ ^{**} $`
- `$ ^{***} $`

这些更像注释标记或显著性标记，不应等同于正文公式漏识别；UI 上可以默认不进入强校对，仅在用户手动画框时召回或补录。

表格实验结论：

- 9 个 table 父框都有 Paddle 父级内容。
- table 的 `block_content` 直接是 HTML/markdown-like 表格真值，例如 `<table>...</table>`。
- 人工表格框应该绑定父级 table block，而不是把表格内部 `$...$` 公式拆成行内公式缺口。

图表实验结论：

- 发现 3 个 `chart` 父框。
- 当前 `chart` 的父级文本内容基本为空。
- 人工图/图表框可绑定 Paddle 几何父框；若无文本真值，保持空图表校验框，不强行 OCR。

全量真实文件实验后的策略收敛：

1. 人工公式框优先召回父级 `block_content` 中的公式真值。
2. 如果父级有唯一可召回 span，则写入公式文本并打 review flag。
3. 如果父级有多个候选但不能唯一确定，则保留候选/空公式校验框。
4. 如果父级无真值，则保持空公式校验框。
5. 人工表格框绑定父级 table 真值。
6. 人工图/图表框绑定父级几何，内容为空时不触发主路径 OCR。

## 可实现方案

建议增加一个页级 Paddle 对齐索引，名称可以是 `PaddleArtifactIndex` 或类似结构。它不需要改变 Paddle 调用，只是把已有返回结果组织成可查询索引。

索引建议包含：

- `parent_blocks`
  - 父级 `parsing_res_list` 记录
  - `block_label`
  - `block_bbox`
  - `block_content`

- `geometry_candidates`
  - `inline_formula`
  - `display_formula`
  - `formula_number`
  - `table`
  - `figure/image`
  - 原始 bbox、label、score、parent_block_id

- `formula_spans`
  - 从父级 `block_content` 提取的 LaTeX span
  - 所属 parent block
  - 推断物理行 line index
  - 是否已经被几何小框消费

- `route_lines`
  - 当前 `_layout_line_routes`
  - formula segment
  - text segment
  - Hanwang text slice

人工框落地时，按以下优先级绑定：

1. 几何命中
   - 对人工框和 Paddle candidate 计算 overlap、coverage、IoU、vertical overlap。
   - 单一高置信命中时，直接绑定 Paddle candidate。
   - 公式内容优先来自父级 `block_content` 恢复出的 LaTeX 文本，而不是 OCR 行文本。

2. 父级文本行反查
   - 仅用于 formula。
   - 当人工框没有几何命中时，先定位到物理行。
   - 如果该行只有一个未被几何消费的公式 span，则绑定该公式。
   - 绑定来源记为 `manual_geometry+paddle_parent_text`。

3. 歧义处理
   - 人工框覆盖多个公式候选：不自动合并。
   - 同一物理行存在多个未消费公式 span：不自动猜。
   - 进入 UI 候选选择或人工补录。

4. 无父级真值
   - 没有几何命中，也没有可唯一召回的父级公式 span。
   - 保留空公式校验框。
   - 不把 crop 到 Paddle 重识别作为主路径，只保留为后续可选能力。

落到 `Block.raw_payload` 时建议保留：

```json
{
  "paddle_binding": {
    "source": "paddle_geometry+parent_text",
    "parent_block_index": 9,
    "line_index": 1,
    "segment_index": 1,
    "label": "inline_formula",
    "text": "$ Incentive_{c} \\times Post_{t} $",
    "candidate_bbox": [1042, 1886, 1144, 1953],
    "manual_bbox": [1030, 1888, 1165, 1960],
    "score": 0.896,
    "review_flags": ["manual_paddle_binding"]
  }
}
```

漏几何框的人工公式建议保留：

```json
{
  "paddle_binding": {
    "source": "manual_geometry+paddle_parent_text",
    "parent_block_index": 9,
    "line_index": 2,
    "text": "$ Incentive_{c} $",
    "manual_bbox": [292, 1958, 620, 2038],
    "review_flags": ["manual_formula_from_parent_text"]
  }
}
```

无父级真值的人工公式建议保留：

```json
{
  "paddle_binding": {
    "source": "manual_geometry_empty_formula_review",
    "text": "",
    "manual_bbox": [1880, 2045, 2050, 2105],
    "review_flags": ["manual_formula_needs_text"]
  }
}
```

## 对 Hanwang 的影响

公式框绑定成功后：

- 该公式框不应进入 Hanwang OCR。
- 对应行的 Hanwang recblock 应只包含公式左右两侧文字。
- 最终行文本由 `layout_route+hanwang` 组装：
  - text segment：来自 Hanwang
  - formula segment：来自 Paddle binding

如果用户手动画出漏检公式 `$ Incentive_{c} $`，则第 3 行应从：

```text
[整行 text] -> Hanwang
```

变为：

```text
[左侧 text] -> Hanwang
[manual formula] -> Paddle parent text: $ Incentive_{c} $
[右侧 text] -> Hanwang
```

这才是“Paddle 一部分、Hanwang 一部分”的正确对齐方式。

## 风险与边界

- Paddle 父级 `block_content` 如果本身没有公式文本，则无法反查，保持空公式校验框。
- 如果同一物理行存在多个未消费公式 span，仅凭行位置不够，需要 UI 选择或更细粒度字符定位。
- 现在只有公式样例，表格/图片需要补真实 fixture 后再验证。
- 当前 route 已能服务自动 Paddle 几何公式，但手动画框后的 route 重建还需要补实现。

## 结论

可实现。

120166 证明了两条路径都成立：

- Paddle 检出的行内公式：可以通过几何命中稳定绑定。
- Paddle 漏检几何框但父级文本保留公式：可以通过人工框物理行 + 父级公式 span 唯一反查。
- 父级无可召回真值：保持空公式校验框，不强行 crop 重识别。

后续实现重点不是让 Hanwang 识别公式，而是让人工框参与 route 重建，让公式框从 Hanwang text slice 中被排除，并把 Paddle 公式内容插回最终行文本。
