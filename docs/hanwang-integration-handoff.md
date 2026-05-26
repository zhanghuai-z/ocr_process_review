# 汉王 OCR 模块接入手册（Handoff）

> 接收方：负责主程序集成的 agent
> 提供方：本次实现的封装层（无 C# 改动，全部 Python 翻译壳 + 复用已有 probe.exe）
> 日期：2026-05-22

---

## 0. 当前主线状态（2026-05 mainline closure）

当前主程序里的 `"hanwang"` 已经不是旧的“汉王 docseg 版面 + 汉王 OCR”全原生链路。现行主链是：

1. 设置页固定为 **PP-VL / API layout + Hanwang micro-recblock OCR**，用户只维护 API URL 与 Token。
2. 版面分析阶段由 `LayoutAnalyzer` 调 PP-VL/API，保留 `Page.ppvl_parsing_res_list` 与每个活跃 `Block.source_label/raw_payload`。
3. OCR 阶段由 `HanwangMicroRecBlockEngine` 以页级方式消费 PP-VL `parsing_res_list`：
   - `TEXT/TITLE/REFERENCE/公式说明/图表说明` 等文字类块走 Hanwang SegImg + Recog；
   - 公式、表格、图片等非正文块保留 PP-VL 文本/属性；
   - Hanwang 为空或长度低于 PP-VL 文本 85% 时 fallback 到 PP-VL `block_content` 并写入 review flag。
4. UI 不再切到全页 OCR 占位页；OCR step 保留版面工作区，进度只显示在左下/状态栏边缘。

因此，后续 agent 不要再把 `"hanwang"` 理解为“禁用 API”或“只用汉王原生 docseg”。如果要改配置/UI，必须保持这一语义。

---

## 1. 关于实现语言

**没有写任何 C#。** 32 位的 `*.exe` 探针是 [hanwang-gpt-review/](../../hanwang-gpt-review/) 早期已经编译好的 .NET Framework 4.0 x86 工件（`docseg_probe.exe`、`linecut_segimg_probe.exe`、`linecut_recogimg_probe.exe`），直接从那里拷过来用。本次新增的全部是 Python：

- 一层 subprocess 桥（64 位 Python → 32 位 probe.exe）
- 一层 JSON → `app.models` 翻译
- 两个引擎适配器（`OcrEngine` / `LayoutEngine` 协议）
- 一处工厂注册

如果以后要把 probe.exe 也维护进本仓库，源码在 [hanwang-gpt-review/](../../hanwang-gpt-review/)`*_probe.cs`，构建脚本同目录的 `*.ps1`。

---

## 2. 资产清单

落到 [resources/hanwang_native/bin/](../resources/hanwang_native/bin/)，共 62 MB / 113 个文件。

| 类别 | 内容 |
|---|---|
| 探针 (x86) | `docseg_probe.exe`、`linecut_segimg_probe.exe`、`linecut_recogimg_probe.exe` |
| DLL | `linecut.dll`、`IntegratRcg.dll`、`doc_seg.dll`、`mp30.dll`、`mp60.dll` + 其他全部旁加载 DLL |
| 字典/模型 | `*.lib`、`*.db`、`*.DAT`、`*.sdic`、`*.tag`、`Dic/*` |

**约束**：probe.exe **必须**以 `bin/` 作为 CWD 运行（DLL 旁加载需要），临时图片也要落到 `bin/`。这两件事桥层已经包好。

---

## 3. 新增/改动文件

| 路径 | 用途 |
|---|---|
| [app/engines/hanwang/__init__.py](../app/engines/hanwang/__init__.py) | 公共出口：`verify_hanwang_assets`、`get_hanwang_bin_dir` |
| [app/engines/hanwang/paths.py](../app/engines/hanwang/paths.py) | 资产路径解析 + 启动期完整性校验 |
| [app/engines/hanwang/native_bridge.py](../app/engines/hanwang/native_bridge.py) | subprocess 桥；`run_docseg / run_linecut_segimg / run_linecut_recog` |
| [app/engines/hanwang/translator.py](../app/engines/hanwang/translator.py) | probe JSON → `Block / Line / Char` |
| [app/engines/hanwang/micro_recblock.py](../app/engines/hanwang/micro_recblock.py) | 当前主线页级混合引擎：PP-VL block routing + Hanwang text OCR + fallback + group progress |
| [app/engines/hanwang_ocr_engine.py](../app/engines/hanwang_ocr_engine.py) | `HanwangOcrEngine`：两阶段 segimg→逐行 recog；`bbox_space = "crop"` |
| [app/engines/hanwang_layout_engine.py](../app/engines/hanwang_layout_engine.py) | `HanwangLayoutEngine`：doc_seg → blocks |
| [app/engines/real_ocr_adapter.py](../app/engines/real_ocr_adapter.py) | `create_engine` / `get_engine_description` 增 `"hanwang"` 分支 |
| [app/services/ocr_pipeline.py](../app/services/ocr_pipeline.py) | 页级 hybrid seam：`prefer_page_hybrid_blocks` 引擎可直接写回 `page.blocks` |
| [app/core/project_store.py](../app/core/project_store.py) | schema v6：持久化 `Page.ppvl_parsing_res_list` 与 `Block.source_label/raw_payload` |
| [app/ui/main_window.py](../app/ui/main_window.py) | OCR 进度改为状态栏小组件，OCR step 不覆盖版面工作区 |
| [app/core/ocr_runner.py](../app/core/ocr_runner.py) | legacy runner 也认 `"hanwang"` |
| [build_spec_common.py](../build_spec_common.py) | PyInstaller datas 加 `resources/hanwang_native/**` |
| [scripts/smoke_hanwang_engines.py](../scripts/smoke_hanwang_engines.py) | E2E 冒烟脚本 |

---

## 4. 启用方式（最小改动）

### 4.1 启用当前主线 Hanwang hybrid
配置里把 `ocr_mode` 改成 `"hanwang"` 即可。`create_engine()` 工厂会自动返回 `HanwangMicroRecBlockEngine`，并声明 `prefer_page_hybrid_blocks=True`。

```python
# app/core/app_config.py 已经支持任意字符串 mode，无需改 schema
# 用户态：把保存的配置中 "ocr_mode" 设为 "hanwang"
```

### 4.2 版面分析（当前主线）
[app/core/layout_analyzer.py](../app/core/layout_analyzer.py) 的 `mode == "hanwang"` 分支当前应走 PP-VL/API layout，而不是旧 docseg。关键原因是 micro-recblock 需要消费 PP-VL 的 `parsing_res_list`，并依赖其 `block_label/block_bbox/block_content` 做 block routing 与 fallback。

切到 `ocr_mode="hanwang"` 之后，版面分析应显示为“PP-VL 版面分析（汉王混合）”，OCR 显示为“汉王 micro-recblock OCR”。

> 注：不要拆出用户可见的 `layout_mode` / `ocr_mode` 双开关。当前产品语义是“选择 hanwang 就按 PP-VL + Hanwang 的固定步骤运行”。

### 4.3 UI 入口
设置页已经收敛为固定 API scheme：隐藏 local/api/hanwang 多模式并列选择，只维护 API URL 和 Token，保存时固定 `mode=hanwang`。Hanwang 模式必须允许填写和测试 API 连接，因为 layout 依赖 PP-VL/API。

---

## 5. 数据流（关键约定）

```
Page (整页 page 坐标)
   │
   ▼
LayoutAnalyzer._api_analyze(image_path, role=layout)
   │   ├─ PP-VL /layout-parsing → parsing_res_list
   │   ├─ Page.ppvl_parsing_res_list = raw parsing_res_list
   │   └─ active Block.source_label/raw_payload 保留 PP-VL 原始属性
   ▼
HanwangMicroRecBlockEngine.recognize_page_blocks(page, image_path)
   │   ├─ text-like block → Hanwang run_micro_recblock()
   │   ├─ formula/table/figure-like block → keep PP-VL content
   │   └─ fallback: Hanwang 文本过短/为空 → keep PP-VL block_content + review flag
   ▼
run_micro_recblock(image_bgr, ppvl_blocks=parsing_res_list)
   │   ┌─ linecut_segimg_probe.exe   crop.png → segimg.json
   │   ├─ emit progress: SegImg / group n/m
   │   ├─ For each group:
   │   │     line_crop = group bbox union + margin, clamp 到 recblock_xyxy
   │   │     └─ linecut_recogimg_probe.exe → LineResult / CharResult
   │   └─ Line.bbox / Char.bbox 最终写回 page 坐标
   ▼
page.blocks = hybrid Block 列表，保留 raw_block/source_label/raw_payload
```

legacy OCR runner 仍保留旧 `HanwangOcrEngine` crop-space 语义；主程序新 pipeline 应优先走 `prefer_page_hybrid_blocks` 页级 seam。

---

## 6. 翻译规则（参数细节）

| 项 | 取值 | 来源 |
|---|---|---|
| `mode` | `0x47=71`（简体）/ `0x4B=75`（简繁） | hanwang_native_workflow `AUTOREC_CHINESE_LANGUAGE_MODES` |
| `postprocess` | `1` | 同上 |
| `split_mode` | `0` | 行 crop 已经是单行，不再切 |
| `chars[].codes[]` | little-endian GBK int，top-10 候选 | probe BuildJson |
| `chars[].scores[]` | 越**小**越好（mp30 原始分） | probe |
| Char `confidence` | `clamp(1 - score/100, 0, 1)` | translator |
| Line `confidence` | char 置信度平均 | translator |
| `proof_status` | `< 0.80` 标 `NEEDS_REVIEW` | 对齐 `real_ocr_adapter.AUTO_FLAG_THRESHOLD` |
| `Char.bbox_source` | `"hanwang:CharRcg"` | translator |
| `Block.source` | `BlockSource.HANWANG_DOCSEG` | 如枚举不存在请补 |

> ⚠️ 上面最后一行：[app/engines/hanwang/translator.py](../app/engines/hanwang/translator.py) 假设 `app.models.BlockSource` 里有 `HANWANG_DOCSEG`。如果接入时报枚举不存在，请在 `app/models/project.py` 给 `BlockSource` 补上这一项；若现有枚举命名不同，也可以直接改 translator 用现成值。

---

## 7. 已知问题 / 容忍策略

0. **属性保真很容易被写回 OCR 时丢掉**：PP-VL 的 `parsing_res_list` 不只是文本，还含有 block label、bbox、分数、公式/表格等结构属性。
   - **当前策略**：`Block.source_label/raw_payload` 在 layout 阶段写入并由 schema v6 持久化；micro-recblock 写回 `page.blocks` 时必须继续带上 `raw_block`，不能只创建裸 `Block(type,bbox,lines)`。

1. **少数 line crop 触发 `System.AccessViolationException`**：跟 hanwang_native_workflow 上观察到的现象一致（probe 内部 mp30 偶发崩溃）。
   - **当前策略**：[hanwang_ocr_engine.py::_char_fallback](../app/engines/hanwang_ocr_engine.py) 已实现字符级 fallback——line 崩了自动迭代 `group.chars` 逐字 crop+recog，每个救回的字生成一个单字 Line，`bbox_source="hanwang:CharRcg:char_fallback"`。120168.tif 实测把回收率从 35 lines 提到 70 lines / 948→1042 字。

2. **subprocess 启动开销**：每行一次 .exe + DLL 装载（~200ms），单页 35 行 → 约 7s 被 spawn 吃掉。
   - **优化方向**：把 3 个 probe.exe 改成 daemon 模式（while-loop + stdin/stdout JSON RPC），主程序生命周期内只 spawn 一次，DLL 装载一次。预计单页降到 0.5-1s。改造工作量 1-2 天，需要 .NET Framework SDK 编译环境。**当前未做**。

3. **不能跨平台**：probe.exe 是 Win32 PE 文件。Linux 端跑要靠 WSL 的 binfmt_misc Windows interop 或 wine。生产环境就是 Windows，所以没问题。

4. **进度看起来卡住**：micro-recblock 单页内 Recog group 很慢，页级进度不足以反馈“正在工作”。
   - **当前策略**：`run_micro_recblock(progress_callback=...)` 在 SegImg 和每个 Recog group 前后发事件；`OcrPipeline` 转成 `OcrProgress(current_block,total_blocks,message)`；UI 状态栏小进度条消费该信号。

5. **OCR step 不应覆盖整个工作区**：旧 UI 会跳到全页“正在 OCR”占位页，用户无法继续看版面。
   - **当前策略**：`STEP_OCR` 的 stack widget 映射到 `LayoutPanel`，OCR 进度小组件放在 status bar；OCR 完成后只隐藏进度，不强制跳 HProof。

---

## 8. 验证步骤

```bash
# 1) 资产校验
cd worktrees/coord
PYTHONPATH=. python -c "from app.engines.hanwang import verify_hanwang_assets; verify_hanwang_assets(); print('ok')"

# 2) E2E 冒烟（旧原生引擎，任意 tif/png 均可）
PYTHONPATH=. python scripts/smoke_hanwang_engines.py /path/to/page.tif

# 3) 主线回归
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_core.py -q -k "micro_recblock or ppvl_parsing_res_list or raw_parsing_res_list or ocr_finished_preserves_current_step"
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_core.py -q
QT_QPA_PLATFORM=offscreen python tests/test_core.py
```

参考结果（120168.tif，3425×2360）：
- layout: 3 blocks
- ocr: 35 lines / 948 chars
- 文本质量正常（"过比值调节捕捉区域间服务业供给能力差异..."）

---

## 9. 打包验证（PyInstaller）

[build_spec_common.py](../build_spec_common.py) 已经把 113 个 hanwang 资产塞到 datas。打包后路径为 `<dist>/ocr_process/resources/hanwang_native/bin/...`，[paths.py](../app/engines/hanwang/paths.py) 的候选根优先用 `sys._MEIPASS/resources/hanwang_native`。

**还没在 Windows 上实跑 build 验证**，建议接入后做一次：
```bat
build.bat
dist\ocr_process\ocr_process.exe  :: 切到 mode=hanwang 测一页
```

如果 `_MEIPASS` 路径解析有问题，可临时 set 环境变量 `HANWANG_NATIVE_DIR=...\bin` 兜底。
