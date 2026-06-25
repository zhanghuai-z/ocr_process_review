# 真实样例 120166 参数传递说明

本文档记录当前基线下，真实样例 `120166` 从页面输入到 PaddleOCR-VL 版面解析、行级路由、Hanwang 识别的参数传递过程。

## 样例范围

当前只把一个代表页放入自动测试：

- 图像：`tests/fixtures/real_samples/244771/120166.tif`
- PaddleOCR-VL 版面响应：`tests/fixtures/layout/120166-layout-api-fixture.json`
- 页面 OCR 行提示：`tests/fixtures/layout/120166-page-ocr-lines.json`

原始 `file/244771纵校/` 仍保留为人工视觉测试和补充验证素材，不会整体硬编码进测试。自动测试只验证当前关键风险点，避免大量样例全固定后造成“看起来全覆盖”的错觉。

2026-06-08 live smoke 已用 PaddleOCR-VL-1.6 jobs API 跑过 `120166.tif`。返回结构可被当前归一化层解析：

```text
layoutParsingResults = 1
prunedResult keys = height, layout_det_res, model_settings, page_count, parsing_res_list, width
parsing_res_list = 15
layout geometry records = 22
label counts = inline_formula: 7, text: 6, paragraph_title: 4, header: 1,
               display_formula: 1, formula_number: 1, number: 1, footer: 1
```

这次 live 结果仍然只有 7 个 `inline_formula` 框，因此 120166 缺少单独 `$ Incentive_{c} $` 行内公式框的问题，至少不是旧 1.5 fixture 解析链路独有的问题；VL1.6 线上模型当前也没有把它拆成独立框。

同日又做了 6 组 VL1.6 输入干预实验，结果如下：

```text
A_full_default          inline_formula = 7
B_full_temperature0     inline_formula = 7
C_crop_default          inline_formula = 7
D_crop_1.5x_default     inline_formula = 6
D_crop_2x_default       inline_formula = 6
E_crop_contrast_sharp   inline_formula = 7
```

所有变体的 `parsing_res_list` 文本都包含 `$ Incentive_{c} $`，但没有任何变体返回单独对应这个 span 的 `inline_formula` 几何框。放大 crop 还会减少一个公式框，说明对这类弱公式形态，输入预处理不是可靠补框方案。实验材料和统计 JSON 放在 ignored 目录 `null/vl16_formula_experiments/`。

## 1. 页面输入

测试构造 `Page` 时传入：

```text
image_path = tests/fixtures/real_samples/244771/120166.tif
width = 2356
height = 3424
page_number = 1
```

实际运行时，`LayoutAnalyzer._api_analyze()` 会读取 `page.display_image_path`，将图像无损编码为 PNG。当前 live 调用目标是 PaddleOCR-VL-1.6 官方 jobs API，会把 PNG bytes 作为 multipart 文件提交。fixture 测试为了稳定，不访问线上 API，而是把历史真实 VL 返回冻结在 JSON 中，用于复盘路由逻辑。

## 2. PaddleOCR-VL-1.6 live 请求

当前 layout role 会解析到：

```text
POST https://paddleocr.aistudio-app.com/api/v2/ocr/jobs
Authorization = bearer <api_token>
Content-Type = multipart/form-data
file = page.png，PNG bytes
model = PaddleOCR-VL-1.6
optionalPayload = {
  "useDocOrientationClassify": false,
  "useDocUnwarping": false,
  "useChartRecognition": false
}
```

提交后会轮询：

```text
GET https://paddleocr.aistudio-app.com/api/v2/ocr/jobs/{jobId}
```

任务完成后下载 `resultUrl.jsonUrl` 指向的 JSONL，再归一化为旧版下游可读的：

```text
result.layoutParsingResults[]
```

如果是 VL profile，不发送 OCR detector/recognizer 参数，也不发送 `returnWordBox`。

注意：`returnWordBox` 已退出主链。当前主线不再把 Paddle wordbox 作为字符框真值。

## 3. PaddleOCR-VL 响应进入版面结构

入口是 `LayoutAnalyzer._extract_api_blocks(page, response)`。

它从响应中读取：

- `layoutParsingResults`
- `prunedResult.parsing_res_list`
- `layout_det_res.boxes`
- `layout_det_res.formula`

处理步骤：

1. 通过 `_detect_api_canvas_scale()` 判断 VL 坐标空间和页面像素空间是否一致。
2. 通过 `_append_api_block()` 把 `parsing_res_list` 转为 `Block`，保留 `raw_payload`。
3. 通过 `_attach_route_subblocks()` 把公式、表格等下游特殊框挂回最近的非跳过父块。
4. 将原始 VL records 保存到 `page.ppvl_parsing_res_list`，供 Hanwang 阶段继续使用。

120166 的目标段落是：

```text
block_label = text
block_bbox = [291, 1813, 2158, 2501]
block_content 包含 $ Y_{ct} $、$ Incentive_{c} \times Post_{t} $ 等公式文本
```

历史 VL fixture 给这个段落挂了 7 个 `inline_formula` 子框：

```text
[1466, 1818, 1518, 1869]
[1042, 1897, 1144, 1950]
[844, 2052, 946, 2103]
[604, 2131, 665, 2188]
[1737, 2293, 1793, 2337]
[623, 2287, 669, 2341]
[729, 2295, 776, 2343]
```

## 4. 页面 OCR 行提示进入路由

`tests/fixtures/layout/120166-page-ocr-lines.json` 提供 7 条真实行级 OCR 框。它只保留：

```text
text
bbox
score
```

不再使用 Paddle `text_word` 或 `text_word_boxes`。

这些行进入 `attach_page_ocr_line_routes(ppvl_blocks, page_ocr_lines, width, height)` 后：

1. 每条 OCR 行转成 `PageOcrLineHint`。
2. 根据行框和父块框的交叠，把行分配给对应 VL 父块。
3. 对每个父块读取 `_route_subblocks`。
4. 调用 `recover_inline_formula_segments()`，把父块里的 LaTeX 公式文本和实际公式框重新配对。
5. 生成 `_layout_line_routes`，供 Hanwang 阶段切块和回填使用。

120166 的关键问题是：父块文本里有 8 个 LaTeX 公式片段，但历史 VL fixture 只给了 7 个公式框。缺失的是单独的 `$ Incentive_{c} $` 框。

修复后的结果不会顺序漂移。最终路由是：

```text
行 0: text + $ Y_{ct} $ + text
行 1: text + $ Incentive_{c} \times Post_{t} $ + text
行 2: text
行 3: text + $ Post_{t} $ + text
行 4: text + $ X_{ct} $ + text
行 5: text
行 6: text + $ \delta_{c} $ + text + $ \varphi_{t} $ + text + $ \varepsilon_{ct} $ + text
```

其中第 6 行三个公式的框被提升到同一行带：

```text
$ \delta_{c} $      [623, 2271, 669, 2343]
$ \varphi_{t} $     [729, 2271, 776, 2343]
$ \varepsilon_{ct} $ [1737, 2271, 1793, 2343]
```

## 5. Hanwang 阶段如何消费这些参数

入口是 `HanwangMicroRecBlockEngine.recognize_page_blocks(image_bgr, page)`。

它传给 `run_micro_recblock()` 的关键参数是：

```text
image_bgr = 当前页图像
ppvl_blocks = page.ppvl_parsing_res_list
seg_timeout = 120.0
recog_timeout = 60.0
include_chars = true
page_ocr_lines = 从 page.blocks 中收集到的已有行框
```

`run_micro_recblock()` 的处理策略：

1. 如果有 `page_ocr_lines`，先调用 `attach_page_ocr_line_routes()` 补齐行级 route。
2. 按 `block_label` 分类：
   - text-like 块交给 Hanwang。
   - formula/table/image/footer 等跳过块保留 VL 文本和框。
3. 对 text-like 块，优先使用 `_layout_line_routes` 切成更细的识别区域。
4. 调用 Hanwang native `run_linecut_segimg(image_bgr, recblocks_xyxy=...)`。
5. 对分出来的 group 调用 `run_linecut_recog(..., with_charrcg=True)`，取 Hanwang 字符级结果。
6. 如果这个 text 块包含 route，调用 `_assemble_layout_route_lines()` 将 Hanwang 文本和 VL 公式段拼回同一行。

公式段在拼回时不会交给 Hanwang 识别，而是作为 VL route token 插入：

```text
source = paddle_inline_formula
bbox_granularity = word
token_text = 公式 LaTeX 文本
```

这就是“行内公式降级为 wordbox/token 级”的当前实现口径：它不再是顶层文字块，也不参与 Hanwang 文字识别，但会落在正确的行和正确位置。

## 6. 当前真实样例测试覆盖什么

120166 相关测试主要覆盖：

- VL 顶层公式和公式编号仍作为只读 overlay 暴露。
- 公式子框只挂到正确父块，不再污染其他块。
- 缺失一个公式框时，后续公式文本不发生顺序漂移。
- 路由结果能表达“文本片段 + 公式 token + 文本片段”的顺序。
- 自动测试使用真实页图像，但只固定一个代表页。

它不覆盖：

- PaddleOCR-VL-1.6 线上模型当前是否改变了输出结构。
- Hanwang native 在所有 120166 到 120197 页面上的识别质量。
- 所有公式、表格、页脚、脚注样式。

这些仍应由 `file/244771纵校/` 做人工视觉回归和按需补充测试。后续每发现一类稳定 bug，再从 `file/` 中挑 1 个最小代表样例进入 `tests/fixtures`，不要整批搬入。
