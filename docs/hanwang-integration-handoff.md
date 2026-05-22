# 汉王 OCR 模块接入手册（Handoff）

> 接收方：负责主程序集成的 agent
> 提供方：本次实现的封装层（无 C# 改动，全部 Python 翻译壳 + 复用已有 probe.exe）
> 日期：2026-05-22

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
| [app/engines/hanwang_ocr_engine.py](../app/engines/hanwang_ocr_engine.py) | `HanwangOcrEngine`：两阶段 segimg→逐行 recog；`bbox_space = "crop"` |
| [app/engines/hanwang_layout_engine.py](../app/engines/hanwang_layout_engine.py) | `HanwangLayoutEngine`：doc_seg → blocks |
| [app/engines/real_ocr_adapter.py](../app/engines/real_ocr_adapter.py) | `create_engine` / `get_engine_description` 增 `"hanwang"` 分支 |
| [app/core/ocr_runner.py](../app/core/ocr_runner.py) | legacy runner 也认 `"hanwang"` |
| [build_spec_common.py](../build_spec_common.py) | PyInstaller datas 加 `resources/hanwang_native/**` |
| [scripts/smoke_hanwang_engines.py](../scripts/smoke_hanwang_engines.py) | E2E 冒烟脚本 |

---

## 4. 启用方式（最小改动）

### 4.1 仅切换 OCR 引擎
配置里把 `ocr_mode` 改成 `"hanwang"` 即可。`create_engine()` 工厂会自动返回 `HanwangOcrEngine`。

```python
# app/core/app_config.py 已经支持任意字符串 mode，无需改 schema
# 用户态：把保存的配置中 "ocr_mode" 设为 "hanwang"
```

### 4.2 接版面分析（**已接好**）
[app/core/layout_analyzer.py](../app/core/layout_analyzer.py) 的 `LayoutAnalyzer.analyze()` 已经加了 `mode == "hanwang"` 分支，调用 `_hanwang_analyze` → `HanwangLayoutEngine().analyze(page.display_image_path)`，blocks 直接塞回 `Page.blocks`（page 坐标）。

切到 `ocr_mode="hanwang"` 之后，**版面分析也会自动走汉王**，跟 OCR 一起替换 PaddleOCR。

> 注：当前 layout 的 mode 复用了同一个 `ocr_mode` 配置项。如果以后要支持「OCR 用汉王、版面用 paddle」这种交叉组合，需要拆出 `layout_mode` 独立配置项；现在没拆。

### 4.3 UI 入口（可选）
现在切换需要手改 config。如果要在设置面板加选项，参考已有 `mode=local/api/mock` 的下拉，多加一个 `"hanwang"` 选项即可。

---

## 5. 数据流（关键约定）

```
Page (整页 page 坐标)
   │
   ▼
HanwangLayoutEngine.analyze(image_path)
   │   ┌─ docseg_probe.exe page.png → docseg.json
   │   └─ translator.translate_docseg → List[Block]  (page 坐标, BlockType.TEXT)
   ▼
For each block:
   crop = page[block.bbox]               ← 调用方负责裁
   ▼
HanwangOcrEngine.recognize(crop, ctx)   ← bbox_space = "crop"
   │   ┌─ linecut_segimg_probe.exe   crop.png → segimg.json
   │   │       └→ lines[area].groups[line].chars[].bbox  (crop 坐标)
   │   ├─ For each group:
   │   │     line_bbox = chars union + margin=12  (clamp 到 crop 内)
   │   │     line_crop = crop[line_bbox]
   │   │     ┌─ linecut_recogimg_probe.exe line_crop.png recblock=full → linecut.json
   │   │     │     └→ lines[].groups[].chars[{codes:int[10], scores:int[10], bbox}]
   │   │     └─ translator.translate_linecut → Line
   │   └─ 把 Line.bbox / Char.bbox 平移 +line_bbox.left/top 回 crop 坐标
   ▼
List[Line]   (crop 坐标, 由调用方再 +block.bbox 平移回 page)
```

OCR runner [app/core/ocr_runner.py](../app/core/ocr_runner.py) 已经会做最后一次平移；如果你走新 pipeline，注意 `bbox_space="crop"` 这个语义。

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

1. **少数 line crop 触发 `System.AccessViolationException`**：跟 hanwang_native_workflow 上观察到的现象一致（probe 内部 mp30 偶发崩溃）。
   - **当前策略**：[hanwang_ocr_engine.py::_char_fallback](../app/engines/hanwang_ocr_engine.py) 已实现字符级 fallback——line 崩了自动迭代 `group.chars` 逐字 crop+recog，每个救回的字生成一个单字 Line，`bbox_source="hanwang:CharRcg:char_fallback"`。120168.tif 实测把回收率从 35 lines 提到 70 lines / 948→1042 字。

2. **subprocess 启动开销**：每行一次 .exe + DLL 装载（~200ms），单页 35 行 → 约 7s 被 spawn 吃掉。
   - **优化方向**：把 3 个 probe.exe 改成 daemon 模式（while-loop + stdin/stdout JSON RPC），主程序生命周期内只 spawn 一次，DLL 装载一次。预计单页降到 0.5-1s。改造工作量 1-2 天，需要 .NET Framework SDK 编译环境。**当前未做**。

3. **不能跨平台**：probe.exe 是 Win32 PE 文件。Linux 端跑要靠 WSL 的 binfmt_misc Windows interop 或 wine。生产环境就是 Windows，所以没问题。

---

## 8. 验证步骤

```bash
# 1) 资产校验
cd worktrees/coord
PYTHONPATH=. python -c "from app.engines.hanwang import verify_hanwang_assets; verify_hanwang_assets(); print('ok')"

# 2) E2E 冒烟（任意 tif/png 均可）
PYTHONPATH=. python scripts/smoke_hanwang_engines.py /path/to/page.tif
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
