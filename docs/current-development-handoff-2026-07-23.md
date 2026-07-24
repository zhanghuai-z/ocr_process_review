# 当前开发交接（2026-07-23）

## 用途

本文供新会话快速恢复当前开发上下文。它记录当前代码基线、已验证的设计约束、用户心智和下一步入口，不替代 `AGENTS.md`，也不把历史实验文档当作生产真值。

新会话应先核对代码和测试；本文与代码冲突时，以代码行为为证据并更新本文。

## 当前基线

- 分支：`refactor-remove-runtime-projection`
- 本轮研究起始基线：`2620f29 fix: use routed foreground for degraded words`
- 本轮研究开始时，该分支领先远端 45 个提交。
- 工作区包含大量用户样例、实验脚本、未跟踪文件和已删除的旧报告。不要批量清理、恢复或提交它们。
- 最近完整回归：`344 passed in 19.65s`。

开始工作前执行：

```bash
git status --short --branch
git log -8 --oneline --decorate
python -m pytest -q
```

## 产品心智

本项目是可恢复、可审计、可人工修正的 OCR 工作台，不是一次性 OCR 脚本。

主工作流为：

```text
导入页面
  -> Paddle VL 版面观察
  -> 采纳为 LayoutSnapshot
  -> 人工版面编辑
  -> PP-OCRv6 行/词观察
  -> 页面级路由编译
  -> CharOCR（LineCut / EngCut）
  -> 不可变 OCR observations
  -> 横校 / 纵校共同编辑 ProofState
  -> ExportSnapshot
  -> 各格式导出
```

人工编辑必须改变明确的业务事实；缓存、调试图、UI 控件状态和外部原始响应都不能暗中成为第二事实源。

## 当前事实边界

`ProjectSession` 是项目聚合根，分别持有页面、版面、Paddle artifact、OCR observation、proof、binding 和表格文本仓库。

| 领域 | 当前权威事实 | 不是权威事实 |
|---|---|---|
| 页面 | `PageRecord`，稳定 `page.uid` | UI 当前页索引、文件名 |
| 版面 | `LayoutSnapshot`，由 `LayoutRepository` 按 revision 管理 | Paddle raw dict、调试 overlay、旧 `Page.blocks` 投影 |
| 外部版面证据 | `PaddleArtifact` | 人工编辑后的版面 |
| OCR | `OcrRun/OcrBatch/OcrRegion/OcrLine/OcrAtom` 和 active pointer | proof 文本、PP-OCR mask、CharOCR 调试框 |
| 校对 | `ProofState` 与其中的 `ProofTextUnit` | HProof/VProof editor、字符索引、缩略图 |
| 导出 | 从当前 session 捕获的不可变 `ExportProjectSnapshot` | UI 状态、旧 exporter 临时字段 |
| 持久化 | `.ocrproj` 加同名 `.assets`，由 `ProjectFileService` 原子安装 | session 临时目录、实验缓存 |

业务身份使用稳定 UID；SQLite rowid、bbox、order 和数组下标都不能作为跨模块身份。

## 版面与 OCR 路由

当前权威链路由 `OcrJobService` 组织：

1. `prepare_page()` 捕获不可变 `PageRecord + LayoutSnapshot + PaddleArtifact + image bytes` 和 active pointer CAS 状态。
2. `execute_page()` 获取 PP-OCRv6 prepass 与必要的 VL observations。
3. `acquire_routing_observation_bundle()` 将观察绑定到 page UID、image hash 和 layout fingerprint。
4. `compile_page_routing_plan()` 编译页面局部、不可变的路由计划。
5. 仅可分派计划进入 `compile_charocr_page_request()` 和 CharOCR。
6. `commit_page()` 在应用线程校验 page/layout/artifact/pointer 后原子采纳 OCR observations，并创建或标记 proof rebind。

已接受的路由原则：

- `LayoutSnapshot` 决定块类型和几何；PP-OCR 行/词框只是 OCR 几何观察。
- 公式、表格、图片和装饰所有权先于文本分流；表格和图片不进入普通文本路由。
- 拉丁字母与数字进入 EngCut；剩余中文及其上下文符号进入 LineCut。
- 混合行是物理行状态，不是一个 `text_mixed` 目标。它应在 CharOCR 前拆为有明确所有权的 segment。
- 路由不完整或归属含糊时只阻断当前页，并保留可审计 issue；不得静默整行 fallback 或修改版面真值。单字符拉丁 token 的残缺墨迹仅在第二组件满足同字形几何约束且候选唯一时补全，否则以 `incomplete_single_latin_token_ink` 阻断当前页。CharOCR 几何可靠时，非空正文不得由 PP/VL 替换或重排，唯一几何绑定的文本分歧只保存为明确标源的候选；EngCut native group 内字符 bbox 重叠时，字符几何与文本同时降级，仅允许恰有一个 PP word token 且已有唯一前景所有权 route bbox 的 segment 聚合自身连续 native groups，并以 PP 文本和 route 前景闭合 bbox 生成一次明确标源的 word atom。PP 原始 word bbox 仍是只读 observation；无唯一 token、无前景闭合几何或 groups 穿插则阻断。若一个完整组件 PP 独立标点 observation 唯一归属一条物理行、且与任何 native atom 均不相交，可按几何顺序创建明确标源的缺失符号 atom。候选绑定、缺失 atom 插入和未绑定使用不同运行计数；插入 atom 的来源与 bbox 进入 OCR observation，执行期行 review flag 当前不写入 `OcrLine`。EngCut 对已有拉丁 route 完全无输出时仍保留显式 PP word fallback。
- 原 native adapter 临时 PP-VL row carrier 已删除；生产 runner 直接接收 `CharOcrInputRow`，厂商字典不再携带 layout UID、策略或 authorship 隐藏键。

生产契约以 `routing-truth-contract.md` 和当前测试为准；`ocr-routing-experiment-conclusions-2026-07-09.md` 仅保存实验背景。

### 斜体 token 退级研究（诊断阶段）

`scripts/audit_italic_token_fallback.py` 是离线诊断入口，只读取 `test3.ocrproj`、缓存 Layout/PP-OCR observation 和 `file/1`、`file/2` 样例；它在进程内捕获现有路由与 EngCut 输出，结束时恢复 monkeypatch，不提交 OCR、不保存项目，也不被生产代码引用。

- 已从代码和回放确认：当前生产“EngCut 有输出但字符几何退化”的 word 退级触发器仍只有同组相邻字符 bbox 的严格横向重叠；EngCut 完全无输出是另一条既有显式 fallback。重叠判据能处理已有重叠退级，但不能发现 `Unbalanced -> Unbalan,ced`、`Finance -> Financce`、`Incentives -> Irtcentives` 这类“没有 bbox 重叠、一个前景组件却被多个字符框切分”的退化。
- 已从 2026-07-23 全量诊断确认：`test3` 连续 3 次输出稳定；`file/1` 与 `file/2` 共 111 页中 108 页完成、3 页复现既有生产阻断（`120193.tif`、`120196.tif`、`T00036_00.jpg`），成功页采集 3377 个单 PP token。当前规则产生 121 个 word fallback；诊断阈值 conservative/balanced/broad 分别会产生 332/401/1224 个。
- 已从回显图确认：前景组件切分比例能覆盖上述目标斜体碎框，但也会命中正常数字和部分正常拉丁词。因此三个扩大策略都只是对比样本，当前证据不足以选择生产阈值；旧斜率分桶也不能单独作为斜体判据。
- 已从代码和回放确认：`Growth:`、`Governments:` 一类问题是 PP 独立标点被拉丁 route 的矩形包络吞入，属于符号所有权冲突，不得借 word fallback 静默吸收。诊断对所有候选策略都返回 `routing_symbol_conflict`。
- 2026-07-24 拉丁倾斜门控实验：`scripts/experiment_latin_token_slant_gate.py` 在上一轮不可变诊断 JSON 上去重，只测量至少含两个拉丁字母、无符号所有权冲突且已有唯一 owned foreground 的 token。它用剪切校正后的纵向投影集中度估计方向，并要求同时存在前景组件切分冲突。1319 个可测 token 中，右倾 `slope>=0.12`、投影改善 `>=0.05` 的对比门控选出 38 个：24 个已由当前规则退级，新增 14 个只来自 `120166/120180` 的斜体区域；4 个已知坏框全部命中，3 个已知正常对照全部保留。绝对斜率会额外误收 `23AXW003`、`very`、`WMF` 等 5 个左倾峰样本，故不能使用绝对值。该结果支持“右倾证据 + 组件切分冲突”的组合方向，但栅格斜率峰集中量化在 `0.25`，全量没有人工字体标签，尚不足以确定生产阈值或接入生产。
- 2026-07-24 字符框碎片质量实验已纠正口径：最初“把完整 owned component 归给主字符框、其余框视为外来碎片”的 v1 指标并不等价于矩形裁片切断墨迹，原 27 个候选和比例结论已作废。v2 直接测量字符矩形竖边是否切穿同一连通墨迹，已知目标和 3 个正常对照的受切字符比例都接近或等于 `1.0`，说明切穿事实普遍存在、单独使用会饱和。v3 再统计每个字符裁片中主墨迹之外的小连通组件，并为 `i/j` 允许一个合理上点；该指标在 4 个已知坏框上均为 `0`，因为视觉碎片仍与主墨迹连通，也不能解释目标退化。当前脚本保留三类测量供回显核查，但没有可进入生产的碎片阈值。
- 已从原始 PP JSON 与回放确认：`T00031_00.jpg` 的 PP word 是 `goubmieibsout`，bbox 为 `[942,1747,1263,1795]`，后处理 route 扩为 `[926,1746,1286,1809]`；扩出的右侧前景被 EngCut 识别为额外 `]`，产生 `goubmieibsout]`。相邻右引号 `”` 在 PP 中是独立 word，`[`/左引号属于周边行上下文。现有 `symbol_conflicts` 只覆盖已配对的相邻 PP symbol，未标出这个越界样例；这是诊断覆盖缺口，不是字符碎片退级证据。
- 尚未实施：没有修改 OCR 路由、EngCut 退级、atom 文本/几何、Proof 或持久化。生产实现前仍需获得可区分目标碎框与正常字框的判据，并单独研究独立标点所有权。

完整 JSON 和四联回显图位于 `D:\project\ocr_process\worktrees\coord\debug\italic_token_fallback_study_20260723`；该目录是诊断产物，不是生产依赖或权威事实。

拉丁倾斜门控报告与候选/方向拒绝回显位于 `D:\project\ocr_process\worktrees\coord\debug\latin_token_slant_gate_20260724`，同样只属于诊断证据。

字符框碎片质量 JSON、目标/对照热图、候选回显和 `T00031` 的 PP/route/native 专项图位于 `D:\project\ocr_process\worktrees\coord\debug\latin_charbox_fragment_quality_20260724`，只属于诊断证据；报告 schema v3 明确废弃 v1/v2 的候选解释。

## 校对心智

横校和纵校是同一 `ProofState` 的两个视图，不维护两套业务文本。

### 横校

- 以 OCR 行为工作单位，对照原稿行图与 proof 文本。
- 中文、英文、数字、标点、表格和公式通过 atom/view 表达。
- `granularity="word"` 的 atom 以完整 bbox 和文本范围生成一个 word overlay；横校不推断词内字符位置，单击选择整词，双击编辑整词范围。char atom 仍保持逐字槽位。
- 行内公式使用气泡编辑；独立公式使用原稿、渲染结果、源码三栏。
- 公式序号是可丢弃的软链接，不合并两个 OCR/proof 事实。

### 纵校

- 字符/词索引顺序为：汉字、字母、数字、标点/其他；多字符 word atom 作为一个词项参与字母组，不拆成无框字符项。
- “字符/词索引”表示不同校对文本 occurrence 集合；右侧索引表示当前字符或词在项目中的实例。
- 每个实例通过稳定 UID、page、line/text unit、bbox 和 crop 元数据引用上下文，不持有独立文本副本。
- OCR 文本上下文按整页呈现并高亮当前字符或词；原稿图同样由 page+bbox 定位。word occurrence 替换其显式 `[char_index, char_end)` 范围。
- 修改最终提交到共享 `ProofTextUnit`，横校和纵校随后读取同一新工作区。

当前实现边界：`ProofWorkspaceView` 是一次性只读投影；HProof/VProof 只持有焦点、选择和未提交 editor 状态；所有修改通过 `ProofSessionService` 的 CAS 命令提交。不得恢复旧 `Page/Line` 直接修改、panel bus、`merge_pages()` 或 stale-editor 补丁链。

## UI 等价状态

以旧成熟 UI `7134bfc^` 为对照，已核对 24 项能力：22 项等价恢复。

已恢复但仍需 Windows 交互验收：纵校 `Shift+Alt+左右`、`Ctrl+Alt+左右` 高级实例选择快捷键，以及横校左侧页面目录折叠。两者只维护面板 UI 状态。

旧 quality-probe 假字注入、可变页面合并和兼容字段诊断不是应恢复的用户能力。

## 性能基线

样例：`file/0722test/test1.ocrproj`，1 页、33 行、931 个纵校字符项、6.69 MiB。以下为 WSL 相对基线，不是 Windows 发行 SLA：

| 操作 | 当前测量 |
|---|---:|
| 模块切换 P50 / P95 | 1.3 / 3.4 ms |
| 横校普通行切换 P50 | 约 64 ms |
| 纵校首次显示 | 约 306 ms |
| 纵校字符桶切换 P50 / P95 | 103 / 132 ms |
| 纵校同字实例切换 P50 | 1.5 ms |
| 单行修改并同步两视图 | 312-454 ms |
| 仅加载业务 session | 1.32-1.57 s |
| 打开项目并发布全部 UI | 3.95-4.19 s |
| 保存服务本身 | 0.86-1.07 s |
| 用户可见保存反馈 | 4.8-11.2 s |

已完成：纵校同页图片只解码一次并缓存 crop；横校只渲染当前公式行。20 个真实裁图新旧像素 20/20 一致。

## 下一阶段入口

P1 代码已完成，待 Windows 样例验收：

1. 公式预览由单一后台 worker 生成 SVG/RGBA payload，GUI 线程只物化 `QPixmap`；结果按源码 hash 匹配，过期结果丢弃。
2. Proof 编辑提交后发布 `ProofWorkspacePatch`，只为变更的 `proof_uid/text_unit_uid` 重建字符索引和行投影；H/V 合并同一补丁，没有新增文本事实源。
3. 同目标普通保存对路径未变的既有资产使用暂存硬链接，链接不可用时显式退回复制；另存为仍复制，原校验、安装和回滚语义保留。

P2 代码已完成：LayoutPanel 相同投影/页面短路并删除任务完成后的重复完整发布；横校目录折叠和纵校高级快捷键已恢复；CharOCR native PP-VL row carrier 已删除。`scripts/benchmark_release_workflow.py` 已提供进程冷启动、P50/P95 和峰值 RSS 采集。仍待完成的是在 Windows 发行机上用 1/30/100 页项目执行并归档结果。

## 阅读顺序

1. `AGENTS.md`：长期治理规则。
2. 本文：当前会话入口。
3. `docs/charocr-routing-architecture-review-2026-07-20.md`：当前 CharOCR 路由边界。
4. `routing-truth-contract.md`：生产路由契约。
5. `docs/ui-equivalence-performance-audit-2026-07-22.md`：proof UI 等价与性能证据。
6. `CURRENT_TRUTH_MAP.md`：较完整但较早的 7 月 7 日事实地图，只作索引，必须对照当前代码。
7. `ocr-routing-experiment-conclusions-2026-07-09.md`：历史实验背景，不是生产权威。

## 新会话启动提示词

```text
先读取 AGENTS.md 和 docs/current-development-handoff-2026-07-23.md，
再按任务读取 routing-truth-contract.md、
docs/charocr-routing-architecture-review-2026-07-20.md 或
docs/ui-equivalence-performance-audit-2026-07-22.md。
当前代码基线为 04b1498。先用代码、持久化路径和测试核对文档事实；
不要恢复旧 Page/Block/Line 投影、proof 双文本、静默 fallback 或兼容链。
工作区有大量用户样例和实验文件，不要批量清理。
```

## 修改完成时必须回答

- 改动触碰了哪个事实边界和工作流阶段？
- 是否新增了兼容、fallback、缓存或第二事实源？
- 哪条旧路径被删除、保留或缩窄？
- 哪些契约、负例和用户行为由测试覆盖？
- 跑了哪些命令？
- 还剩哪些风险，属于 P0/P1/P2 哪一级？
