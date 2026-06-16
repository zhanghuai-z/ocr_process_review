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
| 30-page multi-page TIFF | 3.9MB | 1 | 184.76s | 5.28s | 1.21s | 191.25s | TIFF 多页被当作单页，不能用 |
| 5-page PDF | 5.4MB | 5 | 196.30s | 12.92s | 2.40s | 211.63s | PDF 多页可识别，但上传/提交极慢 |

结论：

- “Paddle 完成百页版面分析 5s 内”不可行；当前环境下单页总耗时已约 10s，10 个单页 job 并发也需要约 28.5s。
- 用户关于“时间不应完全线性累加”的判断是对的：10 个 job 的单 job 累计耗时约 135.9s，被 10 并发压缩到 28.5s。但这不是“同批只增加 20%”，而是被最慢提交、服务端排队、下载结果共同限制。
- multipart 多本地文件上传在已测字段下不可用：`file`/`file[]` 只返回 1 页，`files` 被服务端判为空文件。
- 官方文档也只把 `file` 和 `fileUrl` 定义为二选一输入；`batchId` 是批量查询任务状态的归组字段，不是多文件合并提交。
- 官方 jobs 状态中的 `pending` 明确定义为排队中。本轮 fileUrl 样本无本地上传，仍等待约 45s，证明慢点不只是客户端上传。
- 30 页/10 workers/代理样本总耗时 73.06s，无 429/队列满错误；单页总耗时 median 15.68s、P90 25.67s、max 58.55s。长尾主要来自 40s+ wait。
- 当前环境下代理路径反而更快：单页从无代理约 10-17s 降到 6.61s；10 页从无代理 28.54s 降到 21.01s。主程序当前强制绕过环境代理，可能不是最快路径。
- `returnMarkdownImages=false`、`visualize=false`、`prettifyMarkdown=false` 对单页总耗时没有正向效果，说明结果图片/Markdown 美化不是主瓶颈。
- PDF 单 job 能返回多页结果，说明产品设计上可以研究“整份 PDF 单 job”，但它解决的是 job 数量和状态管理问题，不是 5s 级速度问题；本轮生成的 5 页 PDF 上传/提交极慢，不能直接替换当前策略。
- 30 页多页 TIFF 被 Paddle 视为 1 页，不能作为多页方案。
- 当前主程序的“逐页 jobs + 并发”方向是可用的短期策略；需要把并发上限作为可调参数继续压测，而不是切到 multipart 多文件。

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

结论：

- 默认链路每页约 35 个 group，每个 group 触发一次 `linecut_recogimg_probe.exe`。
- `no chars` 只用于诊断；纵校需要字符框，不能作为产品路径。
- default 与 no-chars 差值提示 EngCut/字符精修链路有明显成本，后续应做缓存和更精确的调用条件，而不是粗暴关闭。

### Collage Batch Risk

实验设置：临时放宽 `MAX_RECOG_COLLAGE_WIDTH=2400`、`MAX_RECOG_COLLAGE_ASPECT=30`。

| Groups per batch | Result |
|---:|---|
| 6 | `System.AccessViolationException` |
| 2 | `System.AccessViolationException` |

结论：

- Hanwang `linecut_recogimg_probe.exe` 的多 recblock/collage batch 仍不稳定。
- 不建议把 `HANWANG_MICRO_RECBLOCK_BATCH=1` 作为主线优化。
- 当前保守阈值虽然导致 batch 基本无效，但避免了 native 崩溃。

### Page-Level Concurrency

| Pages | Workers | Total | Effective sec/page | Errors |
|---:|---:|---:|---:|---|
| 3 | 1 | 56.50s | 18.83s | 0 |
| 3 | 2 | 38.96s | 12.99s | 0 |
| 3 | 3 | 27.81s | 9.27s | 0 |
| 4 | 4 | 32.51s | 8.13s | 0 |

结论：

- 页级并发是当前 Hanwang 最可靠的短期加速方向。
- 并发越高，单页内部耗时会变慢，说明 CPU/WSL interop/磁盘临时文件存在竞争。
- 但总吞吐仍显著改善，4 workers 小样本无 native 错误。

百页粗估：

- 串行：约 31 分钟。
- 2 workers：约 22 分钟。
- 3 workers：约 15 分钟。
- 4 workers：约 14 分钟。

以上只是按本样本线性估算；真实百页需要至少 30 页连续跑验证失败率和内存峰值。

## Engineering Recommendations

1. 短期接入 Hanwang 页级并发：
   - 新增 `ocr_page_concurrency` 配置，默认 2，上限 4。
   - 仅对 Hanwang hybrid/page-block OCR 启用。
   - 进度、错误、保存必须按 page index 回填，不能依赖完成顺序。

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

5. 中长期降低 Hanwang 通信损耗：
   - 目前每次 probe 都是子进程 + PNG/TSV/JSON 临时文件。
   - 更彻底的方案是常驻 native bridge/daemon，或更直接的 DLL 调用层。
