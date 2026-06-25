# OCR Throughput Experiment 2026-06-16

## Scope

目标是评估百页级项目的版面分析与汉王 OCR 吞吐风险，并找出可以工程化接入的加速方向。

本轮已实测 Hanwang native 链路与 PaddleOCR-VL-1.6 live jobs API。Paddle 结论只针对当前账号、当前网络、当前官方 jobs API 表现。

## Search UI Changes

- 查找结果列表只显示文本预览，不再显示页码和属性标签。
- 查找批量赋值进入撤销栈。
- 跨页批量赋值现在是一条撤销记录：一次 `Ctrl+Z`/点击“撤销”会恢复本次批量操作涉及的所有页面。

## Paddle Layout Findings

已确认代码事实：

- 当前主程序在 `LayoutWorker` 中按 `Page.display_image_path` 逐页提交 PaddleOCR-VL-1.6 jobs。
- `layout_concurrency` 已接入，默认 2，上限 4；设置窗口已有入口。
- 上传图片使用 PNG lossless，不是 JPEG，不会因上传编码损失 OCR 精度。
- 官方示例脚本只展示单个本地 `file` 或单个 `fileUrl` 输入；jobs API 会返回 `extractProgress.totalPages/extractedPages`，说明“单个多页文件”是 API 形态上支持的，但不等价于“一个 multipart job 传多个本地文件”。

### Live Jobs API Timing

2026-06-16 追加实测，使用 `scripts/benchmark_paddle_v16_jobs.py`，请求禁用环境代理并复用 `requests.Session`。

| Input | File size | Paddle pages | Submit | Poll/wait | Download | Total | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| `120186.tif` | 200KB | 1 | 8.78s | 0.39s | 1.14s | 10.32s | 单页已超过 5s |
| 2 TIF, repeated `file` field | 400KB | 1 | 35.68s | 0.48s | 1.61s | 37.77s | 只处理 1 页，不能作为多文件 job |
| 2 TIF, repeated `file[]` field | 400KB | 1 | 9.46s | 1.09s | 1.21s | 11.76s | 只处理 1 页，不能作为多文件 job |
| 2 TIF, repeated `files` field | 400KB | 0 | - | - | - | HTTP 400 | 服务端返回“空文件” |
| 5 TIF as 5 separate jobs, 5 workers | 896KB | 5 | varies | varies | varies | 23.44s | 并发能压缩总耗时，但不是常数级 |
| 10 TIF as 10 separate jobs, 10 workers | 1.68MB | 10 | varies | varies | varies | 28.54s | 无失败；仍远超 5s |
| Official demo URL, lean output | remote URL | 1 | 3.51s | 44.79s | 1.91s | 50.21s | 无本地上传仍慢，主要卡在 pending |
| `120186.tif`, lean output | 200KB | 1 | 13.81s | 2.38s | 1.21s | 17.40s | 关闭结果图片/美化没有提速 |
| `120186.tif`, env proxy | 200KB | 1 | 4.86s | 0.48s | 1.28s | 6.61s | 当前代理路径比强制绕代理快 |
| 10 TIF as 10 separate jobs, 10 workers, env proxy | 1.68MB | 10 | varies | varies | varies | 21.01s | 比无代理 28.54s 更快 |
| 30 TIF as 30 separate jobs, 10 workers, env proxy | 5.36MB | 30 | varies | varies | varies | 73.06s | 无失败；出现 40s+ 服务端长尾 |
| 5 TIF as 5 separate jobs, 5 workers, env proxy, `batchId` | 896KB | 5 | varies | varies | varies | 11.10s | `batchId` 可用，但只是批量查询/归组 |
| 30 TIF as 30 separate jobs, 15 workers, env proxy | 5.36MB | 27 | varies | varies | varies | 13.93s | 3 个 429；速度很快但失败率不可接受，必须有退避重试 |
| 30 TIF as 30 separate jobs, 30 workers, env proxy | 5.36MB | 16 | varies | varies | varies | 10.92s | 14 个 429；请求被服务端拒绝，不是处理完成 |
| 30-page multi-page TIFF | 3.9MB | 1 | 184.76s | 5.28s | 1.21s | 191.25s | TIFF 多页被当作单页，不能用 |
| 5-page PDF | 5.4MB | 5 | 196.30s | 12.92s | 2.40s | 211.63s | PDF 多页可识别，但上传/提交极慢 |
| 30-page PDF, PyMuPDF embeds original TIFs | 3.8MB | 30 | 65.40s | 31.65s | 6.14s | 103.19s | 服务端解析约 32s，主要瓶颈在 submit/upload |
| 30-page PDF split into 4 PDFs, 4 workers, env proxy | 3.8MB | 30 | 19.80-40.34s | 25.15-30.31s | 1.83-2.42s | 67.33s | 无错误；比整包 PDF 快，但慢于高并发单页 |
| 30-page PDF split into 4 PDFs, 4 workers, env proxy, 4 runs | 3.8MB | 30/run | varies | varies | varies | 28.96-67.33s | 四轮全成功；mean 43.48s，median 38.81s，submit 波动很大 |
| 30-page PDF split into 8 PDFs, 8 workers, env proxy | 3.8MB | 30 | 3.28-19.36s | 17.40-30.44s | 1.28-2.37s | 47.95s | 无错误；比 4 份首轮快，但仍慢于高并发单页成功页 |
| 30-page PDF split into 3 PDFs, 3 workers, env proxy | 3.8MB | 30 | 79.44-86.71s | 29.76-51.51s | 2.80-3.44s | 138.96s | 无错误；10 页包 submit 长尾太重 |
| 30-page PDF split into 5 PDFs, 5 workers, env proxy | 3.8MB | 30 | 30.94-58.02s | 26.16-46.22s | 1.53-2.19s | 106.44s | 无错误；第 3 包拖尾明显 |
| 30-page PDF split into 6 PDFs, 6 workers, env proxy | 3.8MB | 30 | 48.10-67.53s | 22.40-29.92s | 1.60-1.84s | 99.14s | 无错误；submit 普遍偏重 |
| 30-page PDF split into 8 PDFs, 8 workers, env proxy, rerun | 3.8MB | 30 | 25.43-56.88s | 0.39-0.96s | 1.45-1.84s | 58.96s | 无错误；多数处理发生在 submit 请求返回前 |
| 30-page PDF split into 8 PDFs, 2 workers, env proxy | 3.8MB | 30 | 14.55-52.03s | 0.48-1.44s | varies | 174.43s | 无错误；worker 太少导致多轮排队 |
| 30-page PDF split into 10 PDFs, 10 workers, env proxy | 3.8MB | 30 | 14.67-42.83s | 16.15-22.08s | varies | 62.76s | 无错误；job 数更多，未优于 8 份 |

结论：

- “Paddle 完成百页版面分析 5s 内”不可行；当前环境下单页总耗时已约 10s，10 个单页 job 并发也需要约 28.5s。
- 用户关于“时间不应完全线性累加”的判断是对的：10 个 job 的单 job 累计耗时约 135.9s，被 10 并发压缩到 28.5s。但这不是“同批只增加 20%”，而是被最慢提交、服务端排队、下载结果共同限制。
- multipart 多本地文件上传在已测字段下不可用：`file`/`file[]` 只返回 1 页，`files` 被服务端判为空文件。
- 官方文档也只把 `file` 和 `fileUrl` 定义为二选一输入；`batchId` 是批量查询任务状态的归组字段，不是多文件合并提交。
- 官方 jobs 状态中的 `pending` 明确定义为排队中。本轮 fileUrl 样本无本地上传，仍等待约 45s，证明慢点不只是客户端上传。
- 30 页/10 workers/代理样本总耗时 73.06s，无 429/队列满错误；单页总耗时 median 15.68s、P90 25.67s、max 58.55s。长尾主要来自 40s+ wait。
- 当前环境下代理路径反而更快：单页从无代理约 10-17s 降到 6.61s；10 页从无代理 28.54s 降到 21.01s。主程序当前强制绕过环境代理，可能不是最快路径。
- `returnMarkdownImages=false`、`visualize=false`、`prettifyMarkdown=false` 对单页总耗时没有正向效果，说明结果图片/Markdown 美化不是主瓶颈。
- PDF 单 job 能返回多页结果，说明产品设计上可以研究“整份 PDF 单 job”，但它解决的是 job 数量和状态管理问题，不是 5s 级速度问题。用 PyMuPDF 将 30 张原始 TIF 封装为 3.8MB PDF 后，服务端解析 30 页只约 32s，但 submit/upload 花 65s，总耗时 103s。
- PDF 拆分为 4 份后总耗时降到 67s，无 429；这是稳定性更好的批量策略候选，但仍明显慢于单页 15 并发的 13.9s。单页 15 并发会触发 429，必须加退避重试才能工程化。
- 4 份 PDF 多轮波动很大：`67.33s / 35.77s / 28.96s / 41.85s`。第 2-4 轮首次 poll 基本已 done，说明 submit 返回前服务端可能已经完成解析，结果受远端缓存/短期队列/上传链路影响很大。
- 8 份 PDF 首轮 `47.95s`，比 4 份首轮快，但还不能证明长期稳定优于 4 份；它只是说明“适度分包”能降低单个 submit 长尾。
- 追加矩阵后，3/5/6 份 PDF 都慢于 8 份；10 份无 429，但没有比 8 份更快。当前最稳妥的 PDF 分包候选是每包约 3-4 页、8 workers 左右。
- PDF jobs 的 `submit_seconds` 不能简单理解为纯上传：8 份复测中 `wait_seconds` 只有 0.39-0.96s，但 submit 花 25.43-56.88s，说明服务端可能在 submit 阶段已经完成了大部分解析/排队/处理。
- PDF 分包和多并发不冲突：分包决定每个 job 内有多少页，多并发决定同时提交多少个 job。工程接入时必须保存 manifest：`package_index/page_start/page_count/original_page_uid`，不能依赖文件名字典序。
- 单页 30 并发结果不可采信为成功性能：10.9s 内只有 16 页成功，14 页在 submit 阶段被 `12002 请求频率过高` 拒绝。
- 30 页多页 TIFF 被 Paddle 视为 1 页，不能作为多页方案。
- 当前主程序的“逐页 jobs + 并发 + 429 退避重试”仍是最快短期策略；“PDF 分包 jobs”适合做稳定兜底或低失败率模式。

### Official Docs Check

- `PaddleOCR-VL_API` 文档描述的是同步 JSON `/layout-parsing` 接口：请求体放 base64/URL，`fileType=0` 表示 PDF，`fileType=1` 表示图像。
- `异步API使用文档` 描述的是当前主程序使用的 `/api/v2/ocr/jobs`：本地文件走 multipart，URL 文件走 JSON，`batchId` 仅用于 `/api/v2/ocr/jobs/batch/{batchId}` 批量查询。
- 异步 API 文档给出的限制是：单次请求最大支持 1000 页 PDF，URL 文件不超过 200MB，本地上传文件不超过 50MB。
- 异步 API 文档明确 `12002` 为请求频率过高，对应本轮 15/30 并发的 429 结果。
- 同步 `/layout-parsing` 在本地 120186 单页 base64 探针中超过 90s 未返回，暂不作为主路线；后续除非拿到官方当前 VL1.6 专属同步 URL 并复测，否则主线继续使用 jobs API。

主要风险：

- 百页 PDF 当前会被导入服务渲染成 100 张 PNG，然后主程序提交 100 个 jobs。
- 这会放大 job 提交、排队、轮询、下载 JSONL 的固定开销；并发可以压缩墙钟时间，但会受远端排队和网络波动影响。
- 后续更优路线之一是按原始 `source_path` 聚合同一个 PDF，尝试一次提交原始 PDF，再按 JSONL 页序映射回 `Page`。但这必须拿真实原始 PDF 重测，不能用本轮 PIL 重新封装 PDF 的慢结果直接下结论。

待做 live benchmark：

- 原始导入 PDF job：记录总页数、每页平均耗时、服务端页序和坐标空间。
- 对比 `layout_concurrency=1/2/4/6/8/10` 的远端限流、失败率、平均耗时、P95 耗时。
- 主程序增加 Paddle API 代理策略开关或自动探测：`direct`、`env_proxy`，记录 submit/wait/download 三段耗时。
- 用 `batchId` 改造百页轮询：仍逐页提交 job，但用 `/api/v2/ocr/jobs/batch/{batchId}` 批量查询状态，减少 GET 风暴和线程等待。
- 测试 URL 模式：若文件已在对象存储，`fileUrl` 可能绕过本地上传瓶颈，但这只是把上传成本转移到前置存储链路。

## Hanwang OCR Experiments

测试样本来自 `file/244771纵校`，使用真实 `.tif` 和对应 `.layout-api.json`。

### Single Page 120186

| Mode | Total | SegImg | Recog | Probe calls | Notes |
|---|---:|---:|---:|---:|---|
| default | 21.85s | 5.39s | 9.55s | 35 | batch disabled |
| batch env enabled | 20.63s | 5.35s | 8.40s | 35 | 34/35 chunks still guarded |
| no chars diagnostic | 13.15s | 4.92s | 8.23s | 35 | skips char/EngCut use; not acceptable for proof UI |
| 120170 default | 18.15s | 4.70s | 7.66s | 36 | batch disabled, EngCut 29 calls |
| 120170 batch env enabled | 18.19s | 4.59s | 7.86s | 36 | 35/36 chunks guarded, no real batch |
| 120170 no chars diagnostic | 12.35s | 4.56s | 7.79s | 36 | saves EngCut/char geometry cost, not acceptable for proof UI |

结论：

- 默认链路每页约 35 个 group，每个 group 触发一次 `linecut_recogimg_probe.exe`。
- `no chars` 只用于诊断；纵校需要字符框，不能作为产品路径。
- default 与 no-chars 差值提示 EngCut/字符精修链路有明显成本，后续应做缓存和更精确的调用条件，而不是粗暴关闭。
- 120170 的 `default` vs `no chars` 差值约 5.8s/页，其中 `recog_seconds` 基本不变，说明主要节省来自 EngCut/字符几何增强，而不是中文识别 probe 本身。

### Collage Batch Risk

实验设置：临时放宽 `MAX_RECOG_COLLAGE_WIDTH` 与 `MAX_RECOG_COLLAGE_ASPECT`，并通过 `HANWANG_MICRO_RECBLOCK_BATCH=1` 打开 native batch。

| Groups per batch | Result |
|---:|---|
| 6 | `System.AccessViolationException` |
| 2 | `System.AccessViolationException` |
| 120170, 4 | `System.AccessViolationException` on first chunk `1852x239`; fallback to single, total 20.82s |
| 120170, 2 | `System.AccessViolationException` on first chunk `1852x120`; fallback to single, total 20.42s |

结论：

- Hanwang `linecut_recogimg_probe.exe` 的多 recblock/collage batch 仍不稳定。
- 不建议把 `HANWANG_MICRO_RECBLOCK_BATCH=1` 作为主线优化。
- 当前保守阈值虽然导致 batch 基本无效，但避免了 native 崩溃。
- 当前正文行宽约 1850px，默认 `MAX_RECOG_COLLAGE_WIDTH=1600` 和 `MAX_RECOG_COLLAGE_ASPECT=4.5` 会把绝大多数单行 chunk 直接 guard 掉；30 页 workers=4 样本中 892 个 batch chunk 有 839 个被 guard。
- 放宽 guard 后 native batch 在真实宽行上崩溃，因此速度优化不应从“打开 batch 阈值”入手。

### Page-Level Concurrency

| Pages | Workers | Total | Effective sec/page | Errors |
|---:|---:|---:|---:|---|
| 3 | 1 | 56.50s | 18.83s | 0 |
| 3 | 2 | 38.96s | 12.99s | 0 |
| 3 | 3 | 27.81s | 9.27s | 0 |
| 4 | 4 | 32.51s | 8.13s | 0 |

2026-06-16 追加枚举，使用真实 `file/244771纵校` 样本和已保存的 Paddle layout JSON：

| Pages | Workers | Total | Page avg | Page max | Groups | EngCut calls | Latin exact/review | Group failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 127.26s | 15.82s | 18.84s | 240 | 180 | 75/2 | 0 |
| 8 | 2 | 74.41s | 18.25s | 21.72s | 240 | 180 | 75/2 | 0 |
| 8 | 4 | 55.62s | 26.80s | 31.59s | 240 | 180 | 75/2 | 0 |
| 8 | 6 | 52.62s | 31.86s | 40.12s | 240 | 180 | 75/2 | 0 |
| 8 | 8 | 46.50s | 42.23s | 46.35s | 240 | 180 | 75/2 | 0 |
| 8 | 4, no chars | 37.64s | 18.56s | 20.87s | 240 | 0 | 0/0 | 0 |
| 30 | 1 | 485.26s | 16.15s | 20.31s | 915 | 713 | 617/12 | 1 |
| 30 | 2 | 290.61s | 19.26s | 24.35s | 915 | 713 | 617/12 | 1 |
| 30 | 4 | 213.47s | 27.14s | 35.12s | 915 | 713 | 617/12 | 1 |
| 30 | 6 | 187.61s | 36.14s | 46.03s | 915 | 713 | 617/12 | 1 |
| 30 | 8 | 180.89s | 44.65s | 61.06s | 915 | 713 | 617/12 | 1 |

结论：

- 页级并发是当前 Hanwang 最可靠的短期加速方向。
- 并发越高，单页内部耗时会变慢，说明 CPU/WSL interop/磁盘临时文件存在竞争。
- 但总吞吐仍显著改善，4 workers 小样本无 native 错误。
- 8 页样本里 workers=8 最快且无错误，但 30 页全量里 workers=1/2/4/6/8 都出现同一个 native group failure。因此当前不能简单把并发当作唯一风险源。
- 30 页全量 workers=8 只比 workers=6 快约 6.72s，且单页平均耗时放大到 44.65s；从工程上看，workers=4 是更平衡的高性能档，workers=2 是更温和的保守档。
- `--no-chars` 在 8 页 workers=4 下从 55.62s 降到 37.64s，说明字符框/EngCut 几何约增加 18s/8页。但纵校需要字符框，不能作为产品路径，只能用于诊断。
- EngCut 对拉丁/数字兼容有明确收益：30 页得到 713 次 EngCut 行探针，617 个 exact token，12 个 review token；这说明当前“Paddle token + EngCut exact”为主路径是有效的。

### Native Failure Probe

固定失败点：

- page: `120183`
- block label: `doc_title`
- route text slice bbox: `[284, 560, 2001, 796]`
- SegImg group bbox: `[284, 567, 1991, 659]`
- padded recog crop: `[282, 565, 1993, 661]`
- error: `System.AccessViolationException` inside `LinecutRecogNative.Recog`

追加枚举：

- 重复跑 120183 共 7 次，7/7 都在同一个 `doc_title` group 失败。
- 直接测试 crop 参数共 96 组：
  - `padded`、`raw_group`、`route_block`：全部失败。
  - `padded_shrink_y`、`padded_left_half`、`padded_right_half`：各 8/16 成功。
  - 成功组合均要求 `postprocess=1`；`postprocess=0` 全失败。
  - `with_charrcg`、简体/繁体 mode、split_mode 不决定是否崩。
- y 裁剪网格 100 组里 70 组成功。最小稳定修改是只把上边界下移 3px：`[282,565,1993,661] -> [282,568,1993,661]`。
- 上边界下移 3px 后文本识别为 `政府引导基金、产业集聚与制造业企业`，比左右半切更可靠。底部裁剪过多会把 `、` 识别成 `＼`，不宜做对称大裁剪。

工程含义：

- Hanwang 加速不能只靠提高页并发；必须对 native group failure 做局部 retry。
- 推荐 fallback 顺序：原 crop 失败 -> 同 crop 上边界下移 3px 重试 -> 仍失败再标记该 line/block 进入人工 review，不应静默丢行。
- 该 fallback 应只在 native exception/timeout 后触发，并记录状态栏非阻塞警告和 debug audit，避免把数据错误误判为新逻辑错误。

2026-06-16 追加落地验证：

- 已在主路径实现 native group failure retry：原 crop 失败后，仅将同 group 上边界下移 3px 重试。
- 真实 `120183` 复跑：`recog_group_failures=0`，`recog_group_retry_attempts=1`，`recog_group_retry_successes=1`。
- 30 页 workers=4 复跑：`209.43s`，`group_fail=0`，`retry_attempts=1`，`retry_successes=1`，`retry_failures=0`。
- 对比旧 workers=4：`213.47s` 且 `group_fail=1`。补丁没有引入可见耗时回退，并修复了固定丢行风险。

### Hanwang Bottleneck Breakdown

基于 30 页 workers=4 retry patch 样本：

- 总墙钟：`209.43s`，页级 elapsed 累计 `796.94s`，平均 `26.56s/page`。
- `SegImg` 累计 `227.27s`，平均 `7.58s/page`。
- `Recog` 累计 `337.73s`，平均 `11.26s/page`。
- `linecut_recogimg_probe.exe` 调用 `916` 次，平均 `30.53/page`。
- `eng20_probe.exe` 调用 `713` 次，平均 `23.77/page`。
- EngCut 绑定结果：`617` 个 exact token，`12` 个 review token。
- 页面耗时相关性：`recog_seconds=0.946`，`EngCut calls=0.902`，`group count=0.882`，`crop pixels=0.749`，`SegImg=0.427`。

最慢页：

| Page | Elapsed | Groups | SegImg | Recog | EngCut calls | Exact/review |
|---|---:|---:|---:|---:|---:|---:|
| 120184 | 35.58s | 37 | 7.85s | 14.26s | 37 | 44/1 |
| 120185 | 34.53s | 37 | 7.80s | 14.09s | 34 | 19/0 |
| 120180 | 33.32s | 39 | 7.95s | 14.00s | 39 | 148/0 |
| 120186 | 33.21s | 35 | 8.32s | 14.75s | 33 | 28/2 |
| 120176 | 32.81s | 36 | 7.69s | 13.99s | 33 | 6/0 |

阻塞点判断：

- 第一阻塞点是 native probe 次数，不是单次识别特别慢。每个 group 目前基本会启动一次 `linecut_recogimg_probe.exe`，每个需要拉丁/数字几何的行又启动一次 `eng20_probe.exe`。
- 第二阻塞点是 WSL/临时文件通信：每次 probe 都要写 PNG、写 TSV、启动 exe、读 JSON、删除临时文件。这个成本会被 900+ 次调用放大。
- 第三阻塞点是并发竞争。单页 120170 默认约 `18.15s`，但 30 页 workers=4 中同页约 `30.89s`；页级并发压缩了墙钟时间，但会抬高单页耗时。
- `SegImg` 是固定成本但不是主瓶颈；它每页约 6-8s，和总耗时相关性较低。
- EngCut 有真实价值，不能简单关闭；但它是可优化点，后续应做缓存、减少重复探针，或把 EngCut 纳入常驻 bridge。

### PP-OCRv5 Route Token Alignment Probe

2026-06-17 追加离线实验，使用 `scripts/experiment_ppocr_route_token_alignment.py`：

- 输入：`file/244771纵校/*.layout-api.json`、`/mnt/d/project/ocr_process/null/claude/.cache/*_ppocrv5_return_word_box.json`、现有 `debug/engcut_line_binding_batch_v2`。
- 方法：复用主程序 `attach_page_ocr_line_routes()`，将 PP-OCRv5 行框重新挂到 VL1.6 父框；再从 Paddle/VL 父框文本抽取英文/数字 token，在同 block 的 PP-OCRv5 route-line 文本流里做 source-order exact/variant 绑定。
- 范围：20 页有 PP-OCRv5 缓存的真实样本。
- 结果：`450` 个正文英文/数字 token 中，`436` 个 exact，`1` 个跨行 variant，`13` 个 unresolved；成功率 `97.11%`。
- 公式剥离证据：20 页中 `10` 条 route line 被拆成 `text/formula/text`，route segment 总数为 `text=666`、`formula=13`。例如 120166 中 `$ Y_{ct} $`、`$ Incentive_{c} \times Post_{t} $`、`$ \delta_c $ / $ \varphi_t $ / $ \varepsilon_{ct} $` 都在 PP-OCRv5 行内被剥离，Hanwang 只会收到左右 text slice。
- 120186 重点页：`31` 个 token 全部绑定成功，其中 `PE/VC` 有 1 处跨 PP-OCRv5 两行，被识别为 `PE/\nVC`，绑定到两条 route line；`Lerner`、`Guariglia` 都能落到正确 route line。
- 与旧整行 EngCut 对比：在同时存在旧 `engcut_line_binding_batch_v2` 的 12 页上，旧路径需要 `204` 次整行 EngCut；按 token 所在 route line 调用只需要 `115` 次，调用量下降 `43.63%`。

结论：

- 当前主程序的设计事实成立：PP-OCRv5 提供行级几何，Paddle/VL 提供公式/表格等结构，`attach_page_ocr_line_routes()` 在 Hanwang 前把行内公式剥离出去。
- 英文/数字优化不应盲扫所有正文行。更合理的工程策略是：先用 Paddle/VL token + PP-OCRv5 route line 做行级定位，再对包含 token 的 route line 调 EngCut。
- `reference_content` 的英文密集参考文献仍有风险：PP-OCRv5 行文本会出现漏字/粘连，例如 `Urban Crisis` 变成类似 `Uran Cris`，导致 token exact 无法作为唯一依据。该场景应走“参考文献行整行 EngCut”或 source-order fallback，并由 EngCut exact 结果做最终确认。

### Route-Token EngCut Runtime Probe

2026-06-17 继续实测最新方案真实 EngCut 调用效率，使用：

- `scripts/benchmark_route_token_engcut.py`
- `scripts/experiment_route_token_engcut_binding.py`

口径说明：

- `97.11%` 不是最终 OCR 正文准确率，也不是页面成功率。
- 在上一节中，`97.11% = 437 / 450`，含义是：Paddle/VL 抽出的正文英文/数字 token，有多少能在 PP-OCRv5 route-line 文本流里定位到对应行。
- 本节进一步读取真实 EngCut 输出后重新计算：`97.11% = (432 exact + 5 variant) / 450`，含义变成：Paddle token 有多少能在 EngCut 字符流中完成 exact/source-order 绑定。
- 表格、公式 HTML、独立公式块不进入该统计；统计对象是会影响正文/校对几何的英文与数字 token。

效率实测：

| Mode | Pages | Calls | Workers | Time | Errors | Notes |
|---|---:|---:|---:|---:|---:|---|
| latest route-token | 20 | 217 | 4 | 20.66s | 0 | 只跑 token 所在 route line |
| latest route-token + unresolved block lines | 20 | 217 | 4 | 22.64s | 0 | unresolved 所在行已被其他 token 纳入，调用数未增加 |
| latest route-token + unresolved block lines | 20 | 217 | 8 | 18.60s | 0 | 并发 8 有收益，但单任务耗时上升 |
| latest route-token + unresolved block lines | 20 | 217 | 12 | 17.32s | 0 | 继续提升很小，单任务耗时明显上升 |
| latest route-token + unresolved block lines | 20 | 217 | 16 | 17.15s | 0 | 当前最佳墙钟，但只比 12 workers 快约 0.17s |
| latest route-token + unresolved block lines | 20 | 217 | 24 | 19.53s | 0 | 开始退化，进程/IO/CPU 竞争吞掉收益 |
| old full-line EngCut | 22 | 365 | 4 | 33.55s | 0 | 旧整行方案全量旧样本 |
| old full-line EngCut, same pages | 12 | 204 | 4 | 19.03s | 0 | 与新方案同页集合 |
| latest route-token, same pages | 12 | 115 | 4 | 13.82s | 0 | 同页集合调用数下降 43.63%，墙钟下降约 27.36% |

实际绑定结果：

- 20 页共 `450` 个 token。
- `432` 个 EngCut exact。
- `5` 个 EngCut variant。
- `13` 个 unresolved。
- 成功率 `97.11%`。

重要样例：

- `Urban` / `Crisis`：PP-OCRv5 route text 曾 miss，但真实 EngCut 字符流里已经 exact 绑定成功，说明“route-line 进入 EngCut 后继续匹配”这条 fallback 是有效的。
- 剩余 unresolved 主要集中在 120180 的英文参考文献和少量正文英文。它们不是新方案静默引入的错误，而是新口径把旧统计未纳入失败分母的 token 暴露出来了：旧 30 页聚合里 120180 为 `148/0`，新绑定结果为 `148 exact + 11 unresolved`。
  - `Macroeconomics` 被 EngCut 切成近似 `Macroecortomics`。
  - `Unbalanced` 被切成 `Unbalan.ced`。
  - `Governments` 被切成近似 `Governme,zts`。
  - `Lerner` 被切成 `Lemer`。

结论：

- 最新方案的效率收益成立：同页集合下 EngCut 调用从 `204` 降到 `115`，真实墙钟从 `19.03s` 降到 `13.82s`。
- 当前 EngCut workers 工程上限建议为 `12~16`；`24` workers 已退化。若和 Hanwang page workers 同时运行，应优先保守设为 `8~12`，避免 native probe 互相抢资源。
- 最新方案不是新建 OCR 真值源，仍然是 Paddle token 主导；EngCut 负责字符几何与 token bbox 绑定。
- 剩余 2.89% 失败不是“静默错误”，应进入 review 或后续轻量 fallback；不能用模糊识别静默改写 Paddle token。

### Token-Targeted EngCut Probe

2026-06-17 追加实验，目标是验证“Paddle token 定向 fallback + linecut 低置信 span 小 crop EngCut”能否替代当前整行 EngCut。

输入样本：

- 页面：`debug/engcut_line_binding_batch_v2` 中已有的 22 页。
- 当前整行 EngCut 行数：`365`。
- Paddle token 数：`332`。
- 已知 EngCut exact 真值 token：`180`。
- linecut 低置信/ASCII 可疑 span：`534`。
- 与 Paddle token 真值命中的可疑 span：`163`。

规则：

- linecut ASCII 字符直接视为候选。
- linecut 置信度 `< 0.30` 的非占用字符也进入候选；这能覆盖 `Guariglia -> Gua吨lia` 这类误识别。
- 高置信中文/中文标点作为占用区。
- 只有和 Paddle token 建立定向关系的小 span 才送 EngCut；盲扫全部低置信 span 不作为产品路线。

120186 单页验证：

- 初版“低置信且候选含 Latin”规则：`14` 次小 crop，`12` exact，`Guariglia` 失败。
- 改为“低置信非占用字符也进入候选”：`13` 次小 crop，`13/13` exact。
- 关键修复：`Gua吨lia -> Guariglia`。

22 页 native 计时：

| Mode | Workers | Calls | Wall time | Result |
|---|---:|---:|---:|---|
| token-targeted small crop | 1 | 163 | 27.44s | 139 exact, 23 variant, 1 compound miss |
| token-targeted small crop | 4 | 163 | 11.66s | same result |
| token-targeted small crop | 8 | 163 | 9.11s | same result |
| current full-line EngCut | 4 | 365 | 38.23s | baseline |

结论：

- EngCut 子阶段从当前整行模式改为 token 定向小 crop，在 4 workers 下约 `38.23s -> 11.66s`，减少约 `69.5%`。
- 8 workers 可继续降到 `9.11s`，但单调用平均耗时上升，说明 native/WSL 临时文件仍有竞争；8 workers 可作为实验档，不宜直接默认。
- 唯一 miss 是 `OtherPE/VCFunds` vs `Other/PE/VC/Funds` 的 token 粒度差异，属于 compound variant/review，不是小 crop OCR 失败。
- 如果盲扫全部低置信 span，需要 `534` 次调用，调用量会超过当前整行 `365` 次；因此必须坚持 Paddle token 定向触发。
- 按 30 页 workers=4 样本粗估，EngCut 子阶段可节省约 `50s` 量级；折算到完整 Hanwang OCR 墙钟，预期总耗时下降约 `15%-25%`。实际收益取决于页级并发、token 密度和 native I/O 竞争。

### Pure English Line Direct EngCut Probe

脚本：`scripts/experiment_pure_english_linecut_vs_direct_engcut.py`

输出：

- `debug/pure_english_linecut_vs_direct_engcut/pure_english_linecut_vs_direct_engcut_summary.md`
- `debug/pure_english_linecut_vs_direct_engcut/120180/*_direct_vs_linecut.png`

候选判定：

- CJK 字符数为 `0`。
- 英文字母数 `>= 10`。
- `(英文字母 + 数字) / 非空白字符数 >= 0.45`。

120180 结果：

- 自动筛出 `10` 条纯英文 reference 行。
- 路径 A：PP-OCR route line bbox -> EngCut。
- 路径 B：PP-OCR route line bbox -> linecut group bbox -> EngCut。
- linecut 会系统性收紧 route line bbox。例如 `[32] Baumol...` 从 `[378,2377,2142,2437]` 收紧到 `[378,2387,2134,2430]`，上下共少 `17px`，右侧少 `8px`。
- 10 条中大多数 direct EngCut 与 linecut > EngCut 文本完全一致；第一条 direct 输出 `UrbanCrisis`，linecut 路径输出 `UrbanCrisits`，说明 linecut 收紧后的 crop 可能在行尾引入额外误切。
- `Macroecortomics`、`Unbalan.ced`、`Incen.tives`、`Governme,zts`、`Larzd`、`Secon.d` 等问题在 direct 和 linecut 两条路径都存在，不能归因于 linecut；这是 EngCut 对该英文参考文献印刷质量/字距的自身误切。

120186 结果：

- 当前规则筛出 `0` 条纯英文行。
- 120186 的 `PE/VC`、`Gompers/Lerner`、`Guariglia` 等都在中文混排行内；不能整行绕过 linecut。

工程判断：

- 可以新增“纯英文整行”快路径：符合上述规则时，直接使用 PP-OCR route line bbox 调 EngCut，不再先经过 linecut group 裁剪。
- 混排行仍走 Paddle token 定位 + EngCut token crop；找不到 token 时再进入 linecut 置信度反选 fallback。
- direct EngCut 的文本不能成为权威 OCR 文本，只作为字符几何框来源；权威文本仍以 Paddle token/Hanwang OCR/人工校对链路为准。

### EngCut Word-Box Fallback Probe

脚本：`scripts/render_engcut_word_fallback_overlay.py`

输出：

- `debug/engcut_word_fallback_120180/engcut_word_fallback_summary.md`
- `debug/engcut_word_fallback_120180/120180/*_word_fallback.png`

120180 观察：

- EngCut 的 `group_index` 在纯英文 reference 行里基本已经是词级分组。
- 当单字符框因为斜体/字距产生重叠或小间隙时，可以把同一 group 内的字框合并为词框。
- 标点处理规则：
  - 首尾标点拆成独立框，例如 `[32]` -> `[` / `32` / `]`，`Growth:` -> `Growth` / `:`。
  - 字母之间的内部点号保留在词框内，例如 `Unbalan.ced`、`Incen.tives`，因为这类点号大概率是 EngCut 误切，不应生成假标点框。
- 第一行 `[32] Baumol...`：合并后得到 `14` 个词框、`7` 个标点框，`Unbalan.ced` 虽然文本错误，但词边界清楚。
- 第三条 `[33] Han...`：`Incen.tives`、`Governme,zts` 仍有文本误切，但词级边界明显比单字框稳定。

工程判断：

- 对纯英文行，可以采用“EngCut char boxes -> word/punctuation boxes”的几何 fallback。
- fallback 触发条件可以是：相邻字符框重叠、相邻字符间隙过小、或 token exact 失败但 group 级几何连续。
- fallback 输出只能影响几何显示/校对选区；不能把 `Unbalan.ced` 这类 EngCut 文本回写为 OCR 真值。

百页粗估：

- 串行：约 27 分钟。
- 2 workers：约 16 分钟。
- 4 workers：约 12 分钟。
- 6 workers：约 10.4 分钟。
- 8 workers：约 10.0 分钟。

以上按 30 页实测线性估算。注意：本轮 30 页在 workers=1/2/4/6/8 下均有同一个 group native failure；没有 group retry 前，任何并发档都不能视为完整可靠路径。

## Engineering Recommendations

### Implemented After Follow-up

- 主程序新增 `paddle_api_network_mode`：`auto` / `env_proxy` / `direct`。
  - 默认 `auto`：有系统代理时先走代理；连接类失败会退回 direct。
  - Paddle VL1.6 client 会记录 `submit_seconds`、`wait_seconds`、`download_seconds`、`poll_count`、实际网络路径。
- 设置窗口新增“Paddle 网络”选项，版面请求并发上限从 4 提到 10。
- `LayoutWorker` 为同一批版面分析 job 生成统一 `batchId`，便于后续接入 batch status 批量轮询。
- `PaddleV16LayoutClient.get_batch_status(batch_id)` 已准备好；当前主链仍按单 job 轮询，后续可以基于它减少百页项目的 GET 轮询风暴。
- Hanwang micro-recblock 增加 native group retry：
  - 原 group crop 失败后，上边界下移 3px 重试。
  - retry 成功不计入 `recog_group_failures`，但写入 bbox audit。
  - retry 失败才计入 group failure，并保留原错误和 retry 错误。

1. 短期接入 Hanwang 页级并发：
   - 新增 `ocr_page_concurrency` 配置，默认 2，上限 4；workers=6/8 可以保留为实验档，不建议默认开放。
   - 仅对 Hanwang hybrid/page-block OCR 启用。
   - 进度、错误、保存必须按 page index 回填，不能依赖完成顺序。
   - native group failure retry 已完成；下一步是把页级并发配置接入主程序 worker，并把 retry warning 显示为非阻塞状态栏提示。

2. 保持 Hanwang collage batch 关闭：
   - 已复现 native AccessViolation。
   - 后续如果要重启，必须先做独立 native 稳定性验证。

3. 给 OCR worker 增加 per-page telemetry：
   - `page_elapsed_seconds`
   - `seg_seconds`
   - `recog_seconds`
   - `recog_probe_calls`
   - `latin_engcut_probe_calls`
   - `error`

4. Paddle 下一步做多页 job 实验：
   - 原始 PDF 单 job vs 当前每页 PNG 多 job。
   - 必须确认坐标空间、页序、失败回填和缓存文件策略。
   - 当前 PDF 稳定路径候选是每包约 3-4 页、8 workers 左右；接入前必须保存分包 manifest，并验证失败重试后能按原始 page UID 回填。
   - 逐页 jobs 仍保留为快速路径；PDF 分包更适合作为“低 429、低失败率”的稳定模式。

5. 中长期降低 Hanwang 通信损耗：
   - 目前每次 probe 都是子进程 + PNG/TSV/JSON 临时文件。
   - 更彻底的方案是常驻 native bridge/daemon，或更直接的 DLL 调用层。
   - native collage batch 已在真实宽行上复现崩溃，不能作为主加速线；daemon 的目标应是减少进程启动和临时文件往返，而不是复用不稳定的多 recblock batch。

### Hanwang LineCut Daemon/Cache Probe

2026-06-17 追加。

SegImg daemon 复测：

- 输入：120186 真实页面，9 个 PP-OCR/VL route recblocks。
- spawn-per-call：5 次，中位数 `5.053s`。
- `linecut_segimg_probe.exe --daemon`：启动到 READY `0.085s`，5 次中位数 `5.075s`。
- 结论：SegImg 的耗时主要在 `linecut.dll` 内部分割，不在进程启动；SegImg daemon 对当前真实输入没有收益，短期不应作为主线。

已接入 LineCut native cache：

- 代码：
  - `app/engines/hanwang/native_cache.py`
  - `app/engines/hanwang/native_bridge.py`
- 覆盖：
  - `run_linecut_segimg()`
  - `run_linecut_recog()`
- cache key 包含：
  - 图像内容 hash、shape、dtype。
  - recblocks。
  - 识别参数：`mode`、`postprocess`、`split_mode`、`with_charrcg`。
  - native 指纹：probe exe、`linecut.dll`、`IntegratRcg.dll`。
- 默认开启；可用 `HANWANG_NATIVE_CACHE=0` 关闭。
- cache 目录默认 `.cache/hanwang_native`；可用 `HANWANG_NATIVE_CACHE_DIR` 指定。

120186 micro-recblock 两连跑：

| Run | Total | SegImg | Recog | Groups | Recog calls | Latin EngCut calls | Cache files |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `22.346s` | `5.374s` | `10.187s` | 35 | 35 | 33 | 36 |
| 2 | `8.010s` | `0.092s` | `1.364s` | 35 | 35 | 33 | 36 |

解释：

- LineCut 段从约 `15.561s` 降到约 `1.456s`。
- 第二跑剩余墙钟主要来自 Latin EngCut，这轮未纳入 LineCut cache。
- cache 对首次 OCR 没有收益；它主要解决同页重跑、OCR 失败后重试、人工调框后重复验证、开发阶段反复测试。

### EngCut Mainline Integration And Cache

2026-06-17 追加。

代码：

- `app/engines/hanwang/native_bridge.py`
- `app/engines/hanwang/micro_recblock.py`
- `app/core/latin_span_recovery.py`
- `tests/test_hanwang_native_cache.py`
- `tests/test_core.py`

接入策略：

- EngCut 仍只作为英文/数字字符几何来源，不能作为 OCR 文本真值。
- 主程序不再因为 block 内存在 Paddle/VL token 就 probe 全部 Hanwang 行。
- 新调度先从 Paddle/VL `block_text` 提取英文/数字 token，再用 Hanwang 行文本做 source-order 目标行选择。
- 若 Hanwang 行文本已有英文/数字片段，也会被纳入轻量 fallback；这保留 `Gua吨lia` -> `Guariglia` 这类“中间被误识别成汉字”的修复机会。
- 找不到 exact 的 token 不静默改写文本，只打 review 或保持 Hanwang 结果。
- `PE/VC` 这类 token 增加跨行变体：`PE/\nVC`、`PE\nVC`。跨行命中只进入 review，不自动改写。

cache 覆盖：

- `run_linecut_segimg()`
- `run_linecut_recog()`
- `run_eng20_recogline()`

cache key：

- 图像内容：shape、dtype、像素 sha256。
- 识别区域/参数：
  - SegImg：`recblocks_xyxy`。
  - Recog：`recblocks_xyxy`、`mode`、`postprocess`、`split_mode`、`with_charrcg`。
  - EngCut：`recogline_engstr`、`packed`、`tbrl`。
- native 指纹：
  - LineCut：probe exe、`linecut.dll`、`IntegratRcg.dll`。
  - EngCut：`eng20_probe.exe`、`Eng20.dll`、`CutEng.dll`、`EngDigital.dll`、`HWEng20.db`、`ENWList.db`。
- cache schema：`hanwang-native-probe-cache-v1`。

cache 使用/刷新/重置规则：

- 默认启用；`HANWANG_NATIVE_CACHE=0` 可关闭。
- 默认目录：`.cache/hanwang_native`；`HANWANG_NATIVE_CACHE_DIR` 可指定目录。
- 同一 crop、同一 bbox、同一参数、同一 native 文件指纹会直接复用。
- 版面框、行框、人工画框、公式剥离结果、识别参数、native dll/db/exe 任一变化，都会生成新 key，旧 cache 不会被命中。
- 项目保存/加载、UI 缩放、校对窗口切换不会刷新 cache。
- 如怀疑 native 数据库未被 key 覆盖、cache 文件损坏或磁盘占用过大，直接删除 `.cache/hanwang_native` 即可重置。
- cache 不改变 OCR 真值，只跳过相同 native probe 的重复子进程调用。

120186 主线实测：

| Mode | Total | SegImg | Recog | Groups | Recog calls | EngCut calls | Exact | Review |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no cache, new target-line EngCut | `19.765s` | `4.855s` | `9.373s` | 35 | 35 | 19 | 28 | 2 |
| cache run 1 | `20.938s` | `4.849s` | `10.207s` | 35 | 35 | 19 | 28 | 2 |
| cache run 2 | `4.403s` | `0.156s` | `2.110s` | 35 | 35 | 19 | 28 | 2 |

120169 主线实测：

| Mode | Total | SegImg | Recog | Groups | Recog calls | EngCut calls | Exact | Review |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no cache, new target-line EngCut | `14.784s` | `4.664s` | `6.871s` | 27 | 27 | 11 | 12 | 0 |

结论：

- 120186 的 EngCut 调用量从此前记录的 `33` 降到 `19`。
- EngCut cache 接入后，第二次同页重跑能把 120186 从约 `20.9s` 降到 `4.4s`。
- 当前统计里的 `latin_engcut_probe_calls` 是逻辑调用数，不区分 cache hit；真实 cache 命中通过墙钟和 cache 文件确认。

### Hanwang Page Concurrency Probe

2026-06-17 追加。测试目标是确认“既然单页 Hanwang native 处理速度不可控，是否能通过页级并发让总时间不累加”。

测试口径：

- `HANWANG_NATIVE_CACHE=0`，避免 cache 命中污染首次 OCR 结果。
- 使用 `scripts/evaluate_hanwang_page_concurrency.py`。
- 包含 `SegImg`、`Recog`、`CharRcg`、Latin EngCut。
- 不包含 Paddle 请求、DB 保存、UI 回填。
- 避开 120180 参考文献页，优先使用正文页。

4 页样本：120186-120189。

| Workers | Total | Throughput | Max page time | Errors |
|---:|---:|---:|---:|---:|
| 1 | `76.017s` | `3.16 pages/min` | `21.69s` | 0 |
| 2 | `41.721s` | `5.75 pages/min` | `22.41s` | 0 |
| 4 | `30.918s` | `7.76 pages/min` | `30.86s` | 0 |
| 6 | `31.445s` | `7.63 pages/min` | `31.36s` | 0 |
| 8 | `31.636s` | `7.59 pages/min` | `31.55s` | 0 |

4 页结论：

- `2 workers` 收益稳定，几乎不拖慢单页。
- `4 workers` 继续降低总墙钟，但单页耗时已经从约 20s 上升到约 31s。
- `6/8 workers` 对 4 页没有收益。

8 页样本：120186-120193。

| Workers | Total | Throughput | Max page time | Errors |
|---:|---:|---:|---:|---:|
| 4 | `59.441s` | `8.08 pages/min` | `32.17s` | 0 |
| 6 | `54.415s` | `8.82 pages/min` | `42.08s` | 0 |
| 8 | `50.097s` | `9.58 pages/min` | `49.95s` | 0 |

8 页结论：

- `8 workers` 吞吐最高，但每页都被明显拖慢；这是 native/CPU/IO 竞争，不是线性加速。
- `6 workers` 介于保守和激进之间，总时间比 4 workers 少约 `5s`，但最大单页时间增加约 `10s`。
- 没有出现 native 错误，但样本仍不足以证明长期百页稳定。

工程建议：

- 交互式/开发默认：`2 workers`。
- 普通批处理默认：`4 workers`。
- 大批量实验档：`6 workers`。
- `8 workers` 只适合用户明确选择“吞吐优先”，不建议默认；它牺牲单页延迟换总吞吐。
- 继续拉到 `10/12+` 前，需要至少 16 页、30 页压力测试，并加入失败页自动重试、进度按 page UID 回填、native 子进程总数上限。

### Hanwang Native Direct TIF Probe

2026-06-17 追加，目标是验证“不经过 Paddle/VL，直接把 TIF 输入 Hanwang 原生链路”的速度和稳定性。

脚本：`scripts/experiment_hanwang_native_direct.py`

样本：120186、120169，`HANWANG_NATIVE_CACHE=0`。

120186：正文页。

| Mode | Total | Layout | OCR | Blocks | Lines | Chars | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| direct full-page linecut | `34.491s` | - | - | - | 70 | 1196 | 整页一个 recblock，触发 AccessViolation 后逐字符 fallback |
| native docseg -> OCR | `20.790s` | `5.315s` | `15.475s` | 5 | 35 | 1153 | Hanwang 自己分版面后逐块 OCR |
| PPVL micro-recblock current | `23.883s` | - | - | 9 | 35 | 1155 | 当前 Paddle/VL 辅助主线 |

120169：标题、正文、表格、独立公式、脚注混合页。

| Mode | Total | Layout | OCR | Blocks | Lines | Chars | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| direct full-page linecut | `34.559s` | - | - | - | 90 | 1018 | 多个大 group 触发 AccessViolation 后逐字符 fallback |
| native docseg -> OCR | `19.179s` | `4.735s` | `14.443s` | 9 | 36 | 935 | 可切出表格区域，但仍是普通文本行语义 |
| PPVL micro-recblock current | `16.676s` | - | - | 16 | 30 | 1341 | 保留 Paddle/VL 结构语义，表格/公式由 VL payload 承担 |

输出：

- `debug/hanwang_native_direct/120186/hanwang_native_direct.md`
- `debug/hanwang_native_direct/120186/120186_direct_full_page_overlay.png`
- `debug/hanwang_native_direct/120186/120186_native_docseg_then_ocr_overlay.png`
- `debug/hanwang_native_direct/120186/120186_ppvl_micro_recblock_overlay.png`
- `debug/hanwang_native_direct/120169/hanwang_native_direct.md`
- `debug/hanwang_native_direct/120169/120169_direct_full_page_overlay.png`
- `debug/hanwang_native_direct/120169/120169_native_docseg_then_ocr_overlay.png`
- `debug/hanwang_native_direct/120169/120169_ppvl_micro_recblock_overlay.png`

观察：

- 整页直接塞给 linecut 不稳定。120186/120169 都触发 `System.AccessViolationException`，随后进入逐字符 fallback，且 overlay 容易出现孤立异常框。
- Hanwang 自己的 `docseg -> OCR` 速度有价值：120186 为 `20.790s`，比当前 PPVL micro-recblock 的 `23.883s` 略快；120169 为 `19.179s`，但慢于当前 PPVL micro-recblock 的 `16.676s`。
- Hanwang docseg 的正文行框可用，但只提供原生版面/行几何，不提供 Paddle/VL 的公式、表格、图片、标题、脚注等语义。
- 当前主线 PPVL micro-recblock 的优势是结构语义和可控路由，而不是单页纯文本速度。

工程判断：

- “整页 TIF 直接 linecut”不适合作主线。
- “Hanwang docseg -> OCR”可以作为后续速度参考或 Paddle 失败时的文字兜底。
- 对正式工作流，仍应保留 Paddle/VL 版面作为结构真值；可研究是否用 Hanwang docseg 的行/块几何辅助减少 PPVL route 侧的 Recog 调用或校验异常块。
