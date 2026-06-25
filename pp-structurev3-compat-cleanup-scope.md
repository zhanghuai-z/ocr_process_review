# PP-StructureV3 兼容清理范围

日期：2026-06-15

执行状态：已完成第一批、第二批，以及第三批中“主链 v1.6 字段收窄”的核心部分。当前主链不再暴露 PP-StructureV3 / VL1.5 / 旧 VL profile，不再发送 `/layout-parsing` JSON 请求，不再把 `ocrResults` 或 `layout_det_res` 几何框伪装成顶层版面块。

## 结论

PP-StructureV3 可以从主链中清除。它和 PaddleOCR-VL-1.6 都处在“版面分析模型”这个逻辑位，现在 v1.6 已经接管主链，继续保留 PP-StructureV3 会造成两个问题：

- 配置层仍暴露多个同位模型，实际又强制重定向到 v1.6，容易让后续开发误判真实链路。
- 解析层为了兼容旧端点保留了大量泛字段识别，导致“哪些字段是真值、哪些是历史兜底”不清楚。

但不能按字段名粗暴删除。v1.6 当前真实返回仍然使用 `layoutParsingResults`、`prunedResult`、`parsing_res_list`、`layout_det_res.boxes` 这些名字；这些不是 PP-StructureV3 遗留项。

## 当前 v1.6 真值边界

从现有真实缓存和 fixture 看，v1.6 的版面返回主结构是：

- `result.layoutParsingResults[]`：页面级结果列表。
- `layoutParsingResults[].prunedResult.parsing_res_list[]`：顶层阅读顺序块，包含 `block_label`、`block_bbox`、`block_content`、`block_order`、`block_id`、`block_polygon_points` 等。
- `layoutParsingResults[].prunedResult.layout_det_res.boxes[]`：更细的版面几何框，包含 `label`、`coordinate`、`score`、`cls_id`、`order`、`polygon_points` 等。
- `layoutParsingResults[].markdown.text`：Markdown / LaTeX 内容。

因此下面这些名字虽然历史味重，但本轮不应删除：

- `layoutParsingResults`
- `prunedResult`
- `parsing_res_list`
- `layout_det_res`
- `Page.ppvl_parsing_res_list` / `ppvl_parsing_res_list_json`
- `_route_subblocks` / `_layout_line_routes`
- `source_label` / `raw_payload` / `app_payload`

其中 `ppvl_parsing_res_list` 可以后续重命名为更中性的 `layout_parsing_records`，但这属于数据库迁移，不应和 PP-StructureV3 清理混在一次提交里。

## 确认可清理范围

### 1. API profile 与 endpoint 兼容层

文件：`app/core/api_profiles.py`

当前问题：

- `API_MODEL_PROFILES` 仍保留 `pp-structurev3`、`paddleocr-vl`、`paddleocr-vl-1.5`。
- `KNOWN_API_ENDPOINT_SUFFIXES` 仍把 `/layout-parsing` 当成主流程可接受端点。
- `get_api_model_profile_url()`、`get_api_model_profile()` 的默认 fallback 仍落到 `pp-structurev3`。
- `resolve_api_endpoint_for_role()` 一边保留旧官方 profile，一边强制重定向到 v1.6。
- `get_api_request_options()` 对未知 profile fallback 到 `pp-structurev3`，会把旧 OCR detector 参数混入旧 layout request 语义。

建议清理：

- 删除 `pp-structurev3`、`paddleocr-vl`、`paddleocr-vl-1.5` profile。
- layout 角色只保留 `paddleocr-vl-1.6`。
- OCR 角色继续保留 `pp-ocrv5`，因为它仍负责行框/文本 OCR 辅助链路。
- 默认 profile fallback 改为 v1.6 或显式报错，不再 fallback 到 PP-StructureV3。
- `/layout-parsing` 不再作为主链 endpoint 后缀；如果考虑老项目配置迁移，只在配置加载/迁移层识别一次，然后转成 v1.6 jobs 根地址。

### 2. 旧 JSON `/layout-parsing` 请求路径

文件：`app/core/layout_analyzer.py`

当前问题：

- `_api_analyze()` 已经优先走 `PaddleV16LayoutClient`，但仍保留 `else` 分支，使用 base64 JSON + `post_json_without_env_proxy()` 调旧 `/layout-parsing`。
- `_build_api_payload()` / `_build_api_request_body()` 主要服务旧 JSON endpoint。
- `encode_image_b64_for_paddle`、`get_api_request_options`、`post_json_without_env_proxy` 在 layout 主链里因此继续存在。

建议清理：

- layout API 主链只允许 v1.6 jobs client。
- 非 v1.6 endpoint 直接报错，提示“版面分析只支持 PaddleOCR-VL-1.6 jobs API”。
- 删除 layout_analyzer 内旧 JSON body 构造函数。
- 测试连接和主流程保持一致，不再出现“测试可以走旧端点，主流程又重定向”的双语义。

### 3. API 设置窗口的旧测试分支

文件：`app/ui/widgets/api_settings_dialog.py`

当前问题：

- UI 文案已经写明固定 v1.6 + Hanwang，但 `_test_connection()` 仍保留非 v1.6 JSON POST 分支。
- `build_api_payload()` 只服务旧 endpoint 测试。
- 模型下拉虽然隐藏，但仍从 `get_api_model_profile_options()` 读取旧 profile。

建议清理：

- 测试连接只用 `PaddleV16LayoutClient`。
- 删除 `build_api_payload()`。
- 删除旧 branch 的 `post_json_without_env_proxy`、`detect_api_result_kind` 依赖。
- 如果保留模型下拉代码，也只允许 v1.6 和 pp-ocrv5；更建议直接移除隐藏下拉，避免“隐藏兼容代码”长期存在。

### 4. Paddle response 泛 schema 层

文件：`app/core/paddle_response.py`

当前问题：

- 同一个 helper 同时支持 `ocrResults`、`layoutParsingResults`、`overall_ocr_res`、直接 `rec_texts/rec_boxes`、`layout_boxes/regions/blocks/formula` 等多种 schema。
- 对 layout 主链来说，v1.6 应该只读 `prunedResult.parsing_res_list` 和 `prunedResult.layout_det_res.boxes`。
- `iter_layout_records_from_item()` 在没有 `parsing_res_list` 时 fallback 到 geometry records，这会把“顶层块”和“细粒度检测框”混为同一层级。

建议清理：

- 拆成两个模块或两组函数：
  - v1.6 layout parser：只解析 `parsing_res_list` 和 `layout_det_res.boxes`。
  - ppocr text parser：继续解析 OCR 行结果。
- layout 主链没有 `parsing_res_list` 时应报错或标记接口异常，不要自动拿 geometry records 顶替顶层块。
- `ocrResults` fallback 不应出现在版面分析块生成路径中。

### 5. Layout record 字段别名

文件：`app/core/paddle_layout_schema.py`

当前问题：

- 文本字段兼容 `block_content`、`text`、`content`、`markdown`。
- score 字段兼容 `score`、`confidence`、`layout_score`、`cls_score`、`block_score`、`prob`、`probability`。
- v1.6 实际需要的字段更窄：
  - `parsing_res_list`: `block_label`、`block_bbox`、`block_content`、`block_polygon_points`
  - `layout_det_res.boxes`: `label`、`coordinate`、`polygon_points`、`score`

建议清理：

- 对 v1.6 parser 使用精确字段。
- `content`、`markdown`、`confidence`、`layout_score`、`cls_score`、`block_score`、`prob`、`probability` 等作为 legacy alias 移出主链。
- 如果要保留调试兼容，放到显式 `legacy_paddle_parser.py`，不要混在主链 parser 里。

### 6. BBox 泛兼容层

文件：`app/core/bbox_extraction.py`

当前问题：

- `BBOX_FIELD_KEYS` 同时包含 `coordinate`、`bbox`、`box`、`block_bbox`、`block_box`、`polygon`、`poly`、`points`、`block_polygon_points`、`rec_box`、`rec_bbox`、`rec_poly`、`rec_polys`。
- 这不是单纯 PP-StructureV3 代码，因为 OCR/PP-OCRv5 也会用到部分字段。
- 但 layout 主链不应继续通过一个“大杂烩 extractor”猜字段。

建议清理：

- 新增或拆出 v1.6 layout bbox extractor：
  - 顶层块只认 `block_bbox` / `block_polygon_points`。
  - 细框只认 `coordinate` / `polygon_points`。
- OCR 字符/行框解析继续保留在 OCR parser 中。
- 删除 layout 主链对 `bbox`、`box`、`block_box`、`poly`、`points` 等旧别名的依赖。

### 7. Label authority 与分类映射

文件：`app/core/paddle_labels.py`、`app/models/enums.py`

当前问题：

- `PADDLE_LABEL_AUTHORITY_KEYS` 兼容 `type`、`category`、`category_name`、`cls_name`、`layout_label`。
- v1.6 顶层块 label 是 `block_label`，细框 label 是 `label`。
- `BlockType.from_paddle()` 注释仍写“PP-Structure 返回的 type”。
- `BlockType.from_paddle()` 当前对 `reference_content` 有潜在误分类：`content` 模糊规则在 `reference` 规则之前，可能先落到正文。

建议清理：

- 主链 label authority 只保留：
  - parsing record: `block_label`
  - geometry box: `label`
- `type/category/category_name/cls_name/layout_label` 移到 legacy parser 或测试 fixture，不进入主链。
- `BlockType` 只保留大类文件夹语义：text、title、figure、table、reference、equation、unknown。
- 新增 subtype registry，保留 v1.6 原始小类，例如：
  - text: `text`、`abstract`、`header`、`footer`、`footnote`、`vision_footnote`、`number`
  - title: `doc_title`、`paragraph_title`
  - reference: `reference_content`
  - figure: `chart`、`figure_title`
  - equation: `display_formula`、`inline_formula`、`formula_number`
  - table: `table`
- UI 视觉用大类分组；业务路由以 subtype/source_label 为依据，不把大类当成唯一事实。

### 8. 旧脚本

文件：

- `scripts/PP-StructureV3.sh`
- `scripts/PaddleOCR-VL.sh`
- `scripts/PaddleOCR-VL-1.5.sh`

建议清理：

- `PP-StructureV3.sh` 可以删除或移入 null/archive。
- `PaddleOCR-VL.sh` 和 `PaddleOCR-VL-1.5.sh` 同属旧 layout 位，也应移出主仓主脚本区。
- `scripts/PaddleOCR-VL-1.6.sh` 保留为当前 API 调用样例。
- `scripts/PP-OCRv5.sh` 保留，因为 ppocrv5 仍是文字行框链路的一部分。

### 9. 测试兼容项

文件：`tests/test_core.py`

需要改的测试类型：

- API profile helper 测试：目前仍断言 `PP-StructureV3`、`paddleocr-vl`、`paddleocr-vl-1.5`。
- endpoint role resolution 测试：目前验证旧官方预设被重定向到 v1.6。清理后应改成“旧 profile 不存在 / 旧 endpoint 被迁移或拒绝”。
- request builder 测试：目前验证 `pp-structurev3` layout body 和 old VL body。
- layout analyzer 旧 schema 测试：`category_name`、`layout_label`、OCR fallback、geometry fallback 需要删除或迁移到 legacy parser 测试。
- v1.6 schema 测试保留并加强：必须覆盖 `parsing_res_list` 顶层块、`layout_det_res.boxes` 路由子框、`block_label` / `label` 分别作为权威来源。

## 趋同性 / 无意义兼容代码

| 范围 | 当前状态 | 为什么趋同 | 建议 |
| --- | --- | --- | --- |
| 多个 layout profile | `pp-structurev3`、`paddleocr-vl`、`paddleocr-vl-1.5`、`paddleocr-vl-1.6` 同时存在 | 它们都是版面分析逻辑位，主链实际只允许 v1.6 | 只留 v1.6 |
| `/layout-parsing` endpoint | profile、URL normalize、测试连接、layout_analyzer 仍识别 | 旧端点已经不再是主链入口 | 配置迁移一次后拒绝 |
| layout JSON POST | `layout_analyzer` 和 `api_settings_dialog` 各有一套旧 JSON body | v1.6 jobs 已经是唯一 layout 请求方式 | 删除旧 body builder |
| 泛 schema parser | 一个 parser 同时猜 layout、OCR、旧字段别名 | 主链事实来源应是 v1.6 schema，不应靠猜 | 拆 v1.6 layout parser / ppocr parser |
| label authority aliases | `type/category/category_name/cls_name/layout_label` | v1.6 主链不需要这些字段 | 移到 legacy parser |
| bbox aliases | layout 主链通过通用 bbox extractor 猜字段 | v1.6 字段固定，猜字段会掩盖接口异常 | layout parser 精确字段 |
| OCR fallback 生成 layout block | 没有 page blocks 时从 `ocrResults` 合成 block | 版面分析失败时不该伪装成版面成功 | 删除或改成显式错误 |
| local PaddleOCR layout 模式 | `_local_analyze()` 仍按老 PaddleOCR `type/bbox` | 当前产品链路固定 v1.6 API + Hanwang | 后续可删除或降级为开发实验入口 |

## 不建议本轮清理的内容

这些内容看起来像历史兼容，但当前仍被主链使用：

- `ppvl_parsing_res_list`：保存 v1.6 `parsing_res_list` 真值。
- `_route_subblocks` / `_layout_line_routes`：当前用于行内公式、表格、图片等结构块路由。
- `raw_payload` / `app_payload` 分离：用于把 Paddle 原始事实和程序状态分开。
- `source_label`：保留 Paddle 原始标签，是后续 subtype registry 的基础。
- `overall_ocr_res` 相关函数：版面主链可以去掉依赖，但 PP-OCRv5 / OCR IR 路径仍可能需要。
- `PP-OCRv5` profile 和脚本：它不属于 PP-StructureV3，同样不能因为清 layout 兼容而删除。

## 建议执行顺序

### 第一批：清 profile 和入口

目标：让配置层不再出现 PP-StructureV3 干扰。

- 删除旧 layout profiles。
- 删除旧 layout 脚本或移入 null/archive。
- layout endpoint 只接受 v1.6 jobs。
- 设置窗口测试连接只走 v1.6 jobs。
- 更新相关 tests。

风险：低到中。主要风险在旧项目配置如果存了 `/layout-parsing`，需要迁移或明确报错。

### 第二批：清 layout 请求旧分支

目标：主流程没有旧 JSON `/layout-parsing` 分支。

- 删除 `LayoutAnalyzer._build_api_payload()` / `_build_api_request_body()`。
- 删除 `_api_analyze()` 非 v1.6 branch。
- 删除 layout 主链里的 `encode_image_b64_for_paddle` / `post_json_without_env_proxy` 依赖。

风险：中。需要确保 v1.6 jobs client 测试覆盖正常/鉴权失败/结果为空。

### 第三批：拆 v1.6 parser

目标：数据模型只承认 v1.6 真值，不再靠旧字段猜。

- 新增 v1.6 layout parser。
- `parsing_res_list` 缺失时显式错误。
- `layout_det_res.boxes` 只作为子框/overlay/路由来源，不顶替顶层 block。
- 将 OCR parser 留给 ppocrv5。

风险：中高。这里会触及版面分析、路由、调试回显、测试 fixture，需要配合真实样例验证。

### 第四批：标签 registry

目标：大类是 UI 文件夹，小类是 Paddle 真值。

- `BlockType` 保持粗粒度。
- 新增 subtype/source_label registry。
- 修正 `reference_content` 这类小类分类。
- 版面分析右侧属性按钮按大类平铺，小类用于 hover/detail/导出/路由。

风险：中。主要影响 UI 显示、路由判断和导出语义。

## 后续验收口径

清理完成后，需要满足：

- 主链无法再选择或隐式 fallback 到 PP-StructureV3。
- 版面分析请求只走 PaddleOCR-VL-1.6 jobs API。
- v1.6 返回缺少 `parsing_res_list` 时，不伪造成功版面块。
- 顶层块来源只有 `parsing_res_list`。
- 子结构框来源只有 `layout_det_res.boxes`。
- 大类只负责 UI 分组，小类保留 Paddle 原始语义。
- 旧字段兼容如果必须保留，只能在 legacy parser 或迁移层，不进入主链业务判断。
