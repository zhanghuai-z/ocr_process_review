# 当前开发交接（2026-07-23）

## 用途

本文供新会话快速恢复当前开发上下文。它记录当前代码基线、已验证的设计约束、用户心智和下一步入口，不替代 `AGENTS.md`，也不把历史实验文档当作生产真值。

新会话应先核对代码和测试；本文与代码冲突时，以代码行为为证据并更新本文。

## 当前基线

- 分支：`refactor-remove-runtime-projection`
- 本轮研究起始基线：`2620f29 fix: use routed foreground for degraded words`
- 本轮研究开始时，该分支领先远端 45 个提交。
- 工作区包含大量用户样例、实验脚本、未跟踪文件和已删除的旧报告。不要批量清理、恢复或提交它们。
- 最近完整回归：`366 passed in 22.13s`（2026-07-24，WSL、`QT_QPA_PLATFORM=offscreen`）。

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
6. `commit_page()` 在应用线程校验 page/layout/artifact/pointer 后原子采纳成功的 OCR observations，并创建或标记 proof rebind。执行失败则由 `commit_page_failure()` 采纳空的 failed run/batch 并推进同一 active pointer；失败不创建或修改 `ProofState`，保存重开后仍可重试，成功重试继续通过 pointer CAS 接管。

已接受的路由原则：

- `LayoutSnapshot` 决定块类型和几何；PP-OCR 行/词框只是 OCR 几何观察。
- 公式、表格、图片和装饰所有权先于文本分流；表格和图片不进入普通文本路由。
- 拉丁字母与数字进入 EngCut；剩余中文及其上下文符号进入 LineCut。
- 混合行是物理行状态，不是一个 `text_mixed` 目标。它应在 CharOCR 前拆为有明确所有权的 segment。
- 路由不完整或归属含糊时只阻断当前页，并保留可审计 issue；不得静默整行 fallback 或修改版面真值。单字符拉丁 token 的残缺墨迹仅在第二组件满足同字形几何约束且候选唯一时补全，否则以 `incomplete_single_latin_token_ink` 阻断当前页。CharOCR 几何可靠时，非空正文不得由 PP/VL 替换或重排，唯一几何绑定的文本分歧只保存为明确标源的候选；EngCut native group 内字符 bbox 重叠时，字符几何与文本同时降级，仅允许恰有一个 PP word token 且已有唯一前景所有权 route bbox 的 segment 聚合自身连续 native groups，并以 PP 文本和 route 前景闭合 bbox 生成一次明确标源的 word atom。PP 原始 word bbox 仍是只读 observation；无唯一 token、无前景闭合几何或 groups 穿插则阻断。若一个完整组件 PP 独立标点 observation 唯一归属一条物理行、且与任何 native atom 均不相交，可按几何顺序创建明确标源的缺失符号 atom。候选绑定、缺失 atom 插入和未绑定使用不同运行计数；插入 atom 的来源与 bbox 进入 OCR observation，执行期行 review flag 当前不写入 `OcrLine`。EngCut 对已有拉丁 route 完全无输出时仍保留显式 PP word fallback。
- 原 native adapter 临时 PP-VL row carrier 已删除；生产 runner 直接接收 `CharOcrInputRow`，厂商字典不再携带 layout UID、策略或 authorship 隐藏键。

生产契约以 `routing-truth-contract.md` 和当前测试为准；`ocr-routing-experiment-conclusions-2026-07-09.md` 仅保存实验背景。

### 斜体 token 退级与 CJK 字框后处理（已接入生产）

`scripts/audit_italic_token_fallback.py` 是离线诊断入口，只读取 `test3.ocrproj`、缓存 Layout/PP-OCR observation 和 `file/1`、`file/2` 样例；它在进程内捕获现有路由与 EngCut 输出，结束时恢复 monkeypatch，不提交 OCR、不保存项目，也不被生产代码引用。

- 历史基线：生产“EngCut 有输出但字符几何退化”的 word 退级最初只有同组相邻字符 bbox 严格横向重叠；EngCut 完全无输出是另一条既有显式 fallback。重叠判据能处理已有重叠退级，但不能发现 `Unbalanced -> Unbalan,ced`、`Finance -> Financce`、`Incentives -> Irtcentives` 这类“没有 bbox 重叠、一个前景组件却被多个字符框切分”的退化。
- 已从 2026-07-23 全量诊断确认：`test3` 连续 3 次输出稳定；`file/1` 与 `file/2` 共 111 页中 108 页完成、3 页复现既有生产阻断（`120193.tif`、`120196.tif`、`T00036_00.jpg`），成功页采集 3377 个单 PP token。当前规则产生 121 个 word fallback；诊断阈值 conservative/balanced/broad 分别会产生 332/401/1224 个。
- 已从回显图确认：前景组件切分比例能覆盖上述目标斜体碎框，但也会命中正常数字和部分正常拉丁词。因此三个扩大策略都只是对比样本，当前证据不足以选择生产阈值；旧斜率分桶也不能单独作为斜体判据。
- 已从代码和回放确认：`Growth:`、`Governments:` 一类问题是 PP 独立标点被拉丁 route 的矩形包络吞入，属于符号所有权冲突，不得借 word fallback 静默吸收。诊断对所有候选策略都返回 `routing_symbol_conflict`。
- 2026-07-24 拉丁倾斜门控实验：`scripts/experiment_latin_token_slant_gate.py` 在上一轮不可变诊断 JSON 上去重，只测量至少含两个拉丁字母、无符号所有权冲突且已有唯一 owned foreground 的 token。它用剪切校正后的纵向投影集中度估计方向，并要求同时存在前景组件切分冲突。1319 个可测 token 中，右倾 `slope>=0.12`、投影改善 `>=0.05` 的对比门控选出 38 个：24 个已由当前规则退级，新增 14 个只来自 `120166/120180` 的斜体区域；4 个已知坏框全部命中，3 个已知正常对照全部保留。绝对斜率会额外误收 `23AXW003`、`very`、`WMF` 等 5 个左倾峰样本，故不能使用绝对值。该轮当时只支持“右倾证据 + 组件切分冲突”的组合方向；后续 `120180.tif` 全词审查与用户确认才将生产口径调整为召回优先的 slope-only 后判断。
- 2026-07-24 字符框碎片质量实验已纠正口径：最初“把完整 owned component 归给主字符框、其余框视为外来碎片”的 v1 指标并不等价于矩形裁片切断墨迹，原 27 个候选和比例结论已作废。v2 直接测量字符矩形竖边是否切穿同一连通墨迹，已知目标和 3 个正常对照的受切字符比例都接近或等于 `1.0`，说明切穿事实普遍存在、单独使用会饱和。v3 再统计每个字符裁片中主墨迹之外的小连通组件，并为 `i/j` 允许一个合理上点；该指标在 4 个已知坏框上均为 `0`，因为视觉碎片仍与主墨迹连通，也不能解释目标退化。当前脚本保留三类测量供回显核查，但没有可进入生产的碎片阈值。
- 2026-07-24 EngCut 去斜反事实实验：`scripts/experiment_latin_engcut_deslant_counterfactual.py` 从同一不可变审计 JSON 读取唯一 owned Latin route，只把该 route 的源像素复制到白底隔离画布；原始组与逆剪切组调用同一 EngCut，校正框只存在于实验坐标，不回写 OCR、Proof 或项目。完整 `1.0` 倍去斜在 test3 48 个目标/对照中只保持 24 个 native 文本不变，PP 文本匹配从 40 降到 27，且 14 个样本框几何变差，已否决“去斜后 EngCut 输出直接替代原输出”。四档曲线中，4 个已知坏框都对 `0.25` 倍轻扰动改字，3 个已知正常对照均不改字；扩到 108 个成功页的右倾候选后，“右倾 + 结构冲突”组 38 个有 15 个改字（file1 `2/7`、test3 `13/31`），“右倾 + 结构正常”组 57 个只有 2 个改字（file1 `1/43`、test3 `1/14`）。这支持把“轻扰动响应敏感 + 既有结构冲突”继续作为退化诊断探针；它不证明字体类别、不选择 OCR 文本，也不提供可持久化的新 bbox，尚未接入生产。
- 2026-07-24 `120180.tif` 后判断专项回显：`scripts/render_120180_italic_word_fallback_candidates.py` 不纠斜、不重跑 OCR。完整主册按页面顺序覆盖全部 77 个至少含两个拉丁字母的 token，并附 33 个单字母 Latin token，避免只看已命中样本。结构组合层从既有倾斜报告和 fallback 审计选择“右倾 `>=0.12` + 投影改善 `>=0.05` + 原始结构冲突”的 31 个 token：19 个已由当前重叠规则降级，另有 12 个新增候选；新增组包含 `Unbalanced`、`Finance`、一个 `Incentives` 等已知目标，也包含 `China`、`Crisis`、`Evidence`、`General`、`Generation`、`Money`、`Zoning`、`and` 等原字符结果仍可能可用的正文词。完整主册进一步发现 10 个只通过右倾/投影阈值、但没有结构冲突的明显同族斜体词：`Urban`、`Anatomy`、`Policy`、`Choices`、两个 `Fiscal`、`Public`、`How`、`Interest`、`Federalism`；它们以独立黄色 `SLOPE RECALL` 层展示。按用户本轮召回优先的审查标准，正体 Latin 被额外判断为 word 可以接受，不计作该实验失败；但黄色层仍与结构证据层分开，且所有彩色 route 框都只是诊断性 PP 文本 + route 前景 bbox，不是持久化结果。
- 已从原始 PP JSON 与回放确认：`T00031_00.jpg` 的 PP word 是 `goubmieibsout`，bbox 为 `[942,1747,1263,1795]`，后处理 route 扩为 `[926,1746,1286,1809]`；扩出的右侧前景被 EngCut 识别为额外 `]`，产生 `goubmieibsout]`。相邻右引号 `”` 在 PP 中是独立 word，`[`/左引号属于周边行上下文。现有 `symbol_conflicts` 只覆盖已配对的相邻 PP symbol，未标出这个越界样例；这是诊断覆盖缺口，不是字符碎片退级证据。
- 2026-07-24 CJK 字框清理实验：`scripts/experiment_cjk_slot_bbox_cleanup.py` 只读 `file/0724test/test1.ocrproj` 中持久化的 LineCut atom，不改变文本、顺序、项目或 Proof。被导入的源图是纯 `0/255` 二值图。首轮“字符中心中点槽 + 槽内全前景”会让 560 个 CJK atom 全部改框并从原框外吸入墨迹，已否决。保守版只在相邻可见 atom 中心之间搜索纵向投影空谷，并且候选 bbox 只能在原 LineCut bbox 内收缩：560 个 CJK atom 中 64 个排除原框边缘墨迹，其中 58 个位于不穿墨的空谷之外；39 个样例的槽边仍有墨迹、4 个样例搜索区没有空谷，均只标风险。26 个 `族` 中 5 个属于空谷外多余墨迹，另有 2 个涉及槽边风险。生产仅采纳“空缝且槽边无墨迹”的内收候选；风险样例保持原框。
- 同一 CJK 实验已扩到 `file/2/temp/68_page.ocrproj` 的全部 68 页。该项目仍是旧 `page/block/line/char_` 表结构；脚本通过只读、仅诊断的 `legacy_v1_diagnostic_adapter` 按显式 `bbox_source=hanwang:micro_recblock` 归一化输入，没有修改生产持久化或迁移器。32,846 个可测 CJK atom 中 9,200 个候选 bbox 发生变化，但只有 159 个实际排除黑色墨迹，118 个属于空谷外墨迹；2,289 个存在槽边墨迹风险，45 个搜索区没有空谷。绝大多数“变化”只是收紧原 bbox 内空白，不得等同于质量修复。逐页 overlay、分页裁片、风险分册、JSON 摘要和 HTML 索引已经生成；用户已逐页查看并反馈未见明显副作用，形成了上述保守生产接受条件。
- 2026-07-24 生产接入：`app/engines/hanwang/geometry_postprocess.py` 提供不改图、不改文本的像素测量；`micro_recblock.py` 在 EngCut 输出完成后增加右倾 word 退级，在物理路由行组装后增加 CJK bbox 内收。右倾只处理至少两个拉丁字母、唯一 PP token 和唯一 route foreground bbox；命中后沿用既有明确标源 PP word atom 路径，native 文本/框保留为外部候选。与本行明确 `PpOcrSymbolObservation` 相交的额外符号阻止右倾退级。CJK 清理后的 atom 使用 `hanwang:micro_recblock:cjk_empty_seam_cleanup`，原 LineCut bbox 作为外部候选保留。两条路径都不改变 Layout、routing plan、Proof 或 PP 原始 observation，实验脚本/JSON 不参与运行时。
- 生产斜率实现已对离线报告中的 1319 个可测 token 复算：右倾 `>=0.12` 且投影改善 `>=0.05` 的 95 个 slope-only 判定与实验实现逐项一致。当前按用户确认的召回优先标准不再要求结构冲突；正体 Latin 被额外合并为 word 可接受。独立标点所有权仍不得由该退级路径推断或吞入。

完整 JSON 和四联回显图位于 `D:\project\ocr_process\worktrees\coord\debug\italic_token_fallback_study_20260723`；该目录是诊断产物，不是生产依赖或权威事实。

拉丁倾斜门控报告与候选/方向拒绝回显位于 `D:\project\ocr_process\worktrees\coord\debug\latin_token_slant_gate_20260724`，同样只属于诊断证据。

字符框碎片质量 JSON、目标/对照热图、候选回显和 `T00031` 的 PP/route/native 专项图位于 `D:\project\ocr_process\worktrees\coord\debug\latin_charbox_fragment_quality_20260724`，只属于诊断证据；报告 schema v3 明确废弃 v1/v2 的候选解释。

碎片指标失效原因的专门回显入口位于 `D:\project\ocr_process\worktrees\coord\debug\latin_fragment_metric_failure_evidence_20260724\index.html`：第一组对照 v2 在已知坏框和正常框上同时饱和，第二组展示 v3 漏掉全部 4 个已知坏框，第三组展示 v3 实际主要命中已有 fallback 的严重样本；末尾 5 页覆盖全部 43 个既有 word fallback，三栏分别显示原图上下文、红色 EngCut 字符框/蓝色 PP 原框、绿色最终 word atom/橙色其他最终 atom。43 个最终 word bbox 全部等于 route 前景框，1 个 `120192.tif / pj` 与仍保留的 `o` char atom 相交；是否吞入未成 atom 的邻接墨迹仍由上下文图人工核查。该入口由 `scripts/render_latin_fragment_metric_failure_evidence.py` 从现有 v3 JSON 和不可变 fallback 审计 JSON 只读生成。

EngCut 四档去斜曲线位于 `D:\project\ocr_process\worktrees\coord\debug\latin_engcut_deslant_counterfactual_test3_20260724`，全批次 `0.25` 倍轻扰动目标/对照报告位于 `D:\project\ocr_process\worktrees\coord\debug\latin_engcut_light_shear_batch_20260724`；两者都是诊断产物，不是字体、文本或 bbox 权威事实。

`120180.tif` 的后判断 word fallback 专项入口位于 `D:\project\ocr_process\worktrees\coord\debug\120180_italic_word_fallback_candidates_20260724\index.html`；依次展示 12 个结构组合新增候选、19 个当前生产 word fallback、10 个黄色高召回候选、全部 77 个英文词的页面顺序主册和 33 个单字母 Latin token 附录。

CJK 字框清理 JSON、`族` 对照、空谷候选和边界风险回显位于 `D:\project\ocr_process\worktrees\coord\debug\cjk_slot_bbox_cleanup_0724test_20260724`，只属于诊断投影，不是新的 atom 几何真值。

68 页 CJK 扩面报告和人工审查入口位于 `D:\project\ocr_process\worktrees\coord\debug\cjk_slot_bbox_cleanup_68pages_20260724`；`index.html` 按页链接 full-page overlay、全部 changed 裁片、空谷候选和风险分册。

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

OCR 提交已恢复为“当前页优先，其余待处理页随后”的批次；待处理包括首次进入、输入变更后失效和最近一次执行失败的页面。失败状态属于 OCR run/batch observation，不复用页面级 `PageRecord.error`，也不覆盖已有人工校对文本。

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
