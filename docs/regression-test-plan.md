# 回归测试与验收方案

本文档定义后续重构必须保持的测试策略。执行 agent 不应只依赖手工打开 UI 判断成功；每个核心功能都要有自动化测试或明确可重复的手工回归步骤。

## 1. 当前基线

当前基础测试：

```bash
cd /mnt/d/project/ocr_process
python tests/test_core.py
```

若已安装 pytest：

```bash
python -m pytest -q
```

当前 `tests/test_core.py` 覆盖：

- 模型基础行为。
- ProjectStore 保存/加载。
- ProofEngine 低置信标记。
- TXT/XML/HTML 导出。

## 2. 测试样本

样本目录：

```text
file/
├── 244771纵校.rar
└── 244771纵校/
    ├── 120166.tif
    ├── 120166.tif.dat
    ├── 120166.tif.pst
    ├── 120166.tif.vcr
    └── ...
```

用途建议：

- `.tif`：真实图片导入、版面、人工画框、自动收紧、OCR crop、横校/纵校视觉回归。
- `.dat/.pst/.vcr/.bki/.CHK/.con/.mdb/.ocj/.ocr`：历史 OCR 软件相关文件，第一阶段不解析，但不要删除；后续可研究汉王式数据结构和校对逻辑。

## 3. 测试分层

### 3.1 Unit tests

不启动 GUI、不依赖 OCR、不依赖外网。

必须覆盖：

- `BBox` normalize、clamp、expand、area、iou。
- `BBoxTightener` 合成图收紧。
- `ImportService` 文件分类和错误收集。
- `ProjectStore` schema migration。
- `ProjectStore` 重新保存不会留下旧 block/line。
- `ProofEngine` 疑点标记。
- LLM fake adapter。
- OCR fake adapter。
- 导出器读取最终文本。

### 3.2 Integration tests

使用 fake OCR、fake LLM，不联网。

必须覆盖完整闭环：

```text
导入样本图片
  -> 生成 Page
  -> fake layout 生成 Block
  -> 人工添加 Block
  -> fake OCR 生成 Line
  -> fake LLM 生成 suggestion
  -> 人工接受/拒绝
  -> 保存 .ocrproj
  -> 重新打开
  -> 导出 TXT/XML/HTML
  -> 校验内容和状态
```

### 3.3 UI tests

可用 `pytest-qt`。

必须覆盖：

- 初始步骤只有导入可用。
- 导入后版面可用。
- 无 block 时 OCR 按钮不可用或提示。
- OCR 完成后点击“进入校对”只跳转，不重复触发完成逻辑。
- 画框后框列表增加。
- 删除框后保存并重开不出现旧框。
- LLM 预审关闭时 UI 不显示阻断错误。

### 3.4 Manual regression

对于难以自动化的视觉交互，保留手工步骤：

1. 打开应用。
2. 新建项目。
3. 导入 `file/244771纵校/120166.tif`。
4. 进入版面页。
5. 手动画一个大致文字区域。
6. 自动收紧后确认框仍包住文字。
7. 运行 fake OCR 或真实 OCR。
8. 进入横校。
9. 修改一行，保存。
10. 导出 TXT。
11. 关闭再打开项目，确认修改仍存在。

## 4. Fake OCR 设计

Fake OCR 不能随机，必须稳定。

示例行为：

```python
class FakeOcrEngine:
    def recognize(self, image_bgr, context):
        return [
            Line(text="测试OCR文本", confidence=0.95, bbox=BBox(10, 10, 100, 20)),
            Line(text="低置信文本", confidence=0.60, bbox=BBox(10, 40, 100, 20)),
        ]
```

用途：

- 测 OCR pipeline。
- 测校对。
- 测导出。
- 测 LLM 预审输入。

验收：

- 每次运行输出一致。
- 不读取外网。
- 不依赖 PaddleOCR。

## 5. Fake Layout 设计

Fake layout engine 根据图片尺寸生成固定块：

```python
Block(
    block_type=BlockType.TEXT,
    bbox=BBox(20, 20, width - 40, height // 3),
    source=BlockSource.AUTO_LAYOUT,
)
```

验收：

- 生成 bbox 在图像边界内。
- 保存后重开仍存在。
- 重跑 layout 后旧 block 不残留。

## 6. Fake LLM 设计

Fake LLM 输入固定文本，输出固定建议：

| 输入 | 建议 | flags |
| --- | --- | --- |
| `错別字` | `错别字` | `ocr_typo` |
| 空字符串 | 空字符串 | `empty_text` |
| confidence < 0.8 | 原文 | `low_confidence` |

必须测试：

- 开关关闭不调用 fake LLM。
- 开关开启调用 fake LLM。
- 建议不覆盖最终文本。
- 人工接受后才更新最终文本。

## 7. BBoxTightener 测试

使用 OpenCV/Numpy 生成合成图，不依赖真实样本。

### 7.1 单行文本框

生成白底黑字或黑色矩形模拟文本：

```text
rough bbox 比内容大 30px
期望 tightened bbox 靠近内容，且仍包含内容
```

断言：

- `changed is True`
- new bbox 面积 < rough bbox 面积
- new bbox 包含所有前景像素外接矩形
- padding 生效

### 7.2 空白框

断言：

- `changed is False`
- 返回原框
- reason 包含“未检测到足够内容”

### 7.3 噪声

生成少量随机黑点。

断言：

- 噪点不应导致框收紧到一个小点。
- 如果无主体内容，应保留原框。

### 7.4 表格线

生成网格线。

断言：

- table 类型不应过滤掉细线。
- padding 比 text 更大。

### 7.5 边界框

rough bbox 超出图像边界。

断言：

- bbox 被 clamp。
- 不抛异常。

## 8. ProjectStore 回归

### 8.1 保存/加载

步骤：

1. 创建项目。
2. 添加 page/block/line/char。
3. 保存。
4. 重新加载。
5. 比对字段。

### 8.2 重跑版面不残留旧数据

步骤：

1. 保存 page，包含 block A。
2. 修改内存为只包含 block B。
3. 保存。
4. 重新加载。
5. 断言只有 block B，没有 block A。

### 8.3 迁移

步骤：

1. 构造 v1 SQLite。
2. 用新 ProjectStore 打开。
3. 断言 `meta.schema_version` 更新。
4. 原有数据仍可读取。

## 9. 工作流回归

### 9.1 步骤启用

| 状态 | 导入 | 版面 | OCR | 校对 | 导出 |
| --- | --- | --- | --- | --- | --- |
| 无项目 | 可用/新建 | 禁用 | 禁用 | 禁用 | 禁用 |
| 有页面 | 可用 | 可用 | 禁用或提示 | 禁用 | 禁用 |
| 有 block | 可用 | 可用 | 可用 | 禁用 | 禁用 |
| OCR 完成 | 可用 | 可用 | 可用 | 可用 | 可用但需检查 |

### 9.2 OCR 完成跳转

断言：

- Worker 完成只调用一次 `on_ocr_done`。
- 点击“进入校对”不会再次调用 `on_ocr_done`。
- 当前 stacked widget index 变为横校页。

## 10. LLM 预审回归

### 10.1 开关关闭

输入：项目有 OCR 行，`llm_pre_review_enabled=False`。

断言：

- LLM adapter 调用次数为 0。
- 行状态为 DISABLED。
- 可进入人工校对。

### 10.2 开关开启

输入：fake LLM。

断言：

- adapter 被调用。
- `llm_suggestion` 有值。
- `line.text` 仍是 OCR 原文。

### 10.3 接受建议

断言：

- 点击接受后 `line.text == line.llm_suggestion`。
- `proof_status == MODIFIED` 或新增 `LLM_ACCEPTED` 状态。
- 导出包含接受后的文本。

### 10.4 拒绝建议

断言：

- `line.text` 不变。
- suggestion 保留用于审计。
- 导出不包含 suggestion。

### 10.5 LLM 失败

fake adapter 抛异常。

断言：

- OCR 结果保留。
- 可进入校对。
- 状态为 FAILED。
- 错误被记录但不暴露 token。

## 11. 导出回归

必须测试所有导出器至少不崩溃：

- TXT
- XML
- HTML
- RTF
- DOCX
- PDF

最小断言：

- 输出文件存在。
- 包含人工最终文本。
- 不包含未接受的 LLM 建议。
- XML/HTML 保留 bbox 和 confidence。

PDF 注意：

- 需要 CJK 字体。
- 如果字体缺失，应抛出明确错误，不要生成乱码 PDF。

## 12. 性能和稳定性

批量测试：

- 导入 `file/244771纵校/` 中至少 10 张 `.tif`。
- fake OCR 全部页面。
- 保存项目。
- 重新打开。

验收：

- 不崩溃。
- UI 不长时间无响应；耗时操作必须在 worker 中。
- 保存后数据完整。

## 13. 推荐命令

轻量测试：

```bash
python tests/test_core.py
```

完整测试：

```bash
python -m pytest -q
```

仅核心服务：

```bash
python -m pytest tests/test_core.py tests/test_bbox_tightener.py tests/test_project_store.py -q
```

UI 测试：

```bash
python -m pytest tests/test_ui_workflow.py -q
```

注意：文件名是建议，后续实现时需要新增对应测试文件。

## 14. 测试通过定义

一个阶段不能只说“看起来能跑”。必须满足：

- 自动化测试通过。
- 手工回归步骤能重复执行。
- 保存/重开一致。
- fake adapter 离线可跑。
- 真实外部依赖失败时有明确提示且不破坏主流程。
