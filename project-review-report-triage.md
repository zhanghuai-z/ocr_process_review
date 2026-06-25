# Project Review Report Triage

来源报告：

```text
/root/.gemini/antigravity-cli/brain/be57c09a-dc9b-46c1-a4a1-0c8ab3ad177a/project_review_report.md
```

本文是对该报告的代码复核结论。目标是给后续 reviewer / subagent 一个可对照的判断：哪些问题已经被当前代码反证，哪些问题确实需要进入 backlog，哪些属于策略讨论而不是已复现 bug。

## 当前修订

2026-06-10 已采纳新的文字路径原则：

```text
VL1.6 提供 block 和 inline 结构真值。
PP-OCRv5 提供 page line 几何框。
程序只做 route / carve / assemble。
普通文字最终只采用 Hanwang 识别结果。
Hanwang 文本短或为空时，不再回退 PPVL 文本。
```

因此本文中原报告涉及 `ppvl_fallback`、`short_hanwang_text`、`empty_hanwang_text` 的判断只作为历史背景，不再代表当前主链。

## 总体判断

这份报告能确认我们当前主线的几个关键点：

- `footnote` / `vision_footnote` 已经归入 Hanwang text path。
- `_FORMULA_STYLE_POSITION_LABELS` 当前不含 `footnote`。
- PP-OCRv5 `prunedResult.rec_texts` 路径已经存在。
- 行内公式作为 `formula` segment，不应作为普通 text slice 送入 Hanwang。
- UI 层已经过滤 `paddle_inline_formula` carrier 字框。

但报告里的 S 级问题混入了旧文档推断和模板化判断，不能直接按严重级别执行。下面是复核后的分级。

## 成立或基本成立

### S-5 Hanwang group 失败后的显式降级还不够清晰

代码证据：

```text
app/engines/hanwang/micro_recblock.py:1013-1032
```

单个 group 的 `run_linecut_recog()` 失败后，目前会记录 warning 并令 `raw = {}`，随后 `_line_results_from_recog()` 会生成空结果。旧版本后面还会通过文本长度 fallback 回退 PPVL；当前主链已经移除该 fallback。

这不是“完全静默丢失”，但报告提出的风险成立：失败计数没有作为 block 级显式诊断暴露给 UI / result note。建议保留为 P1：把 group 失败数量写入 `segimg_group_audits` 或 block note，便于用户知道是 Hanwang 识别失败，不是 Paddle 路由问题。

### S-6 未知 Paddle 标签默认走 text path，缺少审计

代码证据：

```text
app/engines/hanwang/micro_recblock.py:196-197
app/core/paddle_labels.py:89-96
```

当前逻辑是：

```text
_is_text_label(label) = label in TEXT_LABELS or not _is_skip_label(label)
```

因此未知标签只要不以 formula / table / figure / image / chart / seal / stamp 等前缀开头，就会默认进入 Hanwang text path。

这个默认值有一定合理性，因为文本漏识别的代价通常高于多送一次 Hanwang；但缺少 warning / diagnostics。建议保留为 P1：新增 unknown label audit，而不是立即改默认策略。

### S-7 多引擎降级策略需要写成统一真值

旧代码层曾有多处 fallback：

```text
PP-OCRv5 缺失 -> 使用初始 VL route
Hanwang 短文本 -> ppvl_fallback（已移除）
无 text route -> empty_text_routes fallback（已移除）
skip label -> ppvl
```

当前仍需要写成统一真值的是：PP-OCRv5 缺失时的 route 行为、inline 公式拼回、非文字结构 `ppvl` 路径。Hanwang 文本 fallback 已不再是主链策略。

## 部分成立，需要更精确定义

### S-1 PP-OCRv5 空列表 fallback

代码证据：

```text
app/core/paddle_response.py:31-43
```

当前 `overall_ocr_res()` 会识别直接位于 `prunedResult` 下的：

```text
rec_texts / rec_boxes / rec_polys / dt_polys
```

如果 `prunedResult.rec_texts` 存在但为空，函数会返回该 `prunedResult`，随后 `iter_ocr_records_from_item()` 得到 0 条 OCR line。

这不是 120169 已复现问题的同类 bug。120169 的根因是直接 `prunedResult.rec_texts` 没被读取；这个已经修了。

真正需要讨论的是策略：

- 如果 PP-OCRv5 真实返回空行，是否应该信任“空”？
- 如果 `ocrResults` 为空但 `layoutParsingResults` 有内容，是否应该 fallback 到 layout？
- 如果某个 `ocrResults` item 为空，但同 item 里存在 direct / overall 的备份字段，是否应该继续查找？

建议保留为 P2 策略测试，不建议按 S 级立即改。否则可能把“模型确认无文字”的结果误解释成解析失败。

### S-4 Hanwang Session 过期重试

当前主线使用本地 native probe：

```text
app/engines/hanwang/native_bridge.py
```

不是 HTTP session 调用。报告引用的 session 文档很可能来自旧架构或旧 docs。除非当前仍存在另一条 Hanwang HTTP path，否则这一项不能作为当前主链 S 级问题。

建议：先定位是否仍有生产路径使用 Hanwang session；没有的话，该项归档为旧文档债。

## 当前代码已反证，属于误报或过期来源

### S-2 Hanwang API 调用缺少请求超时

代码证据：

```text
app/engines/hanwang/native_bridge.py:45-64
app/engines/hanwang/native_bridge.py:108-142
app/engines/hanwang/micro_recblock.py:889-965
```

当前 Hanwang 主链不是 requests API，而是 `subprocess.run(..., timeout=timeout)` 调本地 probe。

默认超时：

```text
SegImg: 120s
Recog: 60s
```

所以“API 调用未设置请求超时”不成立。可以讨论是否需要更短 timeout 或 UI 取消机制，但不是缺少 timeout。

### S-3 crop 坐标越界未处理

代码证据：

```text
app/engines/hanwang/micro_recblock.py:513-523
app/engines/hanwang/micro_recblock.py:989-1017
```

Hanwang SegImg group bbox 会先：

```text
_bbox_tuple()
  -> _clamp_xyxy(..., width, height)
  -> _intersect_xyxy(raw_group_bbox, recblock)
  -> bbox invalid 时 continue
```

最终 `image_bgr[top:bottom, left:right]` 使用的是已经 clamp / intersect 后的 bbox。

因此报告里的“未校验 recblocks_xyxy 坐标是否在图片范围内”不适用于当前 group crop 路径。

需要注意的是：送入 `run_linecut_segimg()` 的 route recblocks 本身来自 `text_slice_routes_for_block()`，这些 route 应继续通过脚本验证是否都在页面内；但不是当前报告描述的未处理越界。

## 对后续工作的建议顺序

### P0 保持当前已修主线稳定

- 不要回滚 PP-OCRv5 direct `prunedResult` 解析。
- 不要把 `footnote` / `vision_footnote` 放回 skip。
- 不要把 `footnote` 放入 `_FORMULA_STYLE_POSITION_LABELS`。
- 不要恢复 UI “框链”开关。

### P1 可做的小修

- 给 unknown Paddle label 增加 diagnostics / warning。
- 将 Hanwang group 失败计数写入 block result，便于 UI 或调试报告展示。
- 给 Hanwang 空结果 / group 失败做诊断字段，但不要回退 PPVL 文本。

### P2 策略测试

- 构造 PP-OCRv5 `rec_texts=[]` 的 fixtures，明确是否 fallback。
- 构造 `ocrResults` 为空但 `layoutParsingResults` 有内容的 fixtures，明确是否 fallback。
- 构造 unknown label fixtures，明确默认走 text path 还是进入人工审核。

## 和 120169 footnote 实验的关系

120169 的最新实验输出位于：

```text
debug/120169_footnote_hanwang_echo/120169_footnote_hanwang_echo.md
debug/120169_footnote_hanwang_echo/120169_footnote_hanwang_echo.json
debug/120169_footnote_hanwang_echo/120169_footnote_hanwang_echo_overlay.png
```

实验结论：

- `vision_footnote idx7`、`footnote idx13`、`footnote idx14` 都有 `has_ppocr_routes=True`。
- Hanwang 前输入已经是 PP-OCRv5 route 后的 text slice。
- 旧实验中 `idx7` / `idx13` 因 `short_hanwang_text` 使用 `ppvl_fallback`；当前代码已移除该行为，需要重新跑实验确认最新输出。
- `idx14` 最终采用 Hanwang，字符框来源为 `hanwang:micro_recblock`。

因此其他 agent review 时必须区分：

```text
文字路径送入 Hanwang
  =
最终采用 Hanwang 文本
```
