# OCR 后处理工程化升级技术实施文档

本文档面向后续接手实现的 agent。请严格按“先闭环、再增强”的顺序执行，不要一开始就重写全部 UI。项目当前是可运行原型，目标是逐步把它改造成稳定的 OCR 后处理工作台。

## 1. 当前状态速览

### 1.1 已存在的关键文件

```text
main.py
app/models/project.py
app/models/enums.py
app/core/ocr_runner.py
app/core/layout_analyzer.py
app/core/project_store.py
app/core/proof_engine.py
app/ui/main_window.py
app/ui/recognize/import_panel.py
app/ui/recognize/layout_panel.py
app/ui/recognize/ocr_panel.py
app/ui/widgets/image_viewer.py
app/ui/proof/h_proof.py
app/ui/proof/v_proof.py
app/ui/export/export_dialog.py
app/export/*.py
tests/test_core.py
file/244771纵校/
```

### 1.2 当前架构逻辑

- `main.py` 创建 `QApplication`，应用主题，启动 `MainWindow`。
- `MainWindow` 直接管理项目、步骤跳转、worker、保存和导出入口。
- `ImportPanel` 允许导入图片/PDF，但 PDF 未真正转换。
- `LayoutPanel` 展示页面、版面框和属性；目前只能看框，不能编辑框。
- `ImageViewer` 支持图片显示、缩放、平移、框叠加；尚不支持画框和编辑框。
- `LayoutAnalyzer` 调用本地 PaddleOCR 或 API 做版面分析。
- `OcrRunner` 调用本地 PaddleOCR 或 API 做块内 OCR。
- `ProjectStore` 用 SQLite 保存项目、页面、块、行、字符。
- `HProofPanel` 和 `VProofPanel` 是横校/纵校原型。
- `ExportDialog` 选择格式并调用导出器。

### 1.3 当前必须修复的问题

1. 步骤状态不可靠：用户可能跳到不该跳的页。
2. OCR 完成后“进入校对”重复触发 `_on_ocr_done`，而不是跳转到横校。
3. PDF 虽然可选，但后续按图片处理，实际不能走通。
4. 重新版面分析可能留下旧 block/line 脏数据。
5. 配置只存在内存，关闭程序丢失。
6. `requests` 被代码使用但未在 `pyproject.toml` 声明。
7. 人工画框和自动收紧完全缺失。
8. LLM 预审还未接入；需要作为可关闭的中间处理层，而不是替代人工。

## 2. 第一阶段验收目标

第一阶段不要追求完整美化。必须先保证以下闭环稳定：

```text
导入图片/PDF
  -> 创建项目与页面
  -> 版面分析或人工确认区域
  -> OCR
  -> 可选 LLM 预审
  -> 人工横校/纵校终审
  -> 导出
  -> 保存项目
  -> 重新打开项目仍能看到一致结果
```

### 2.1 第一阶段完成标准

- 图片导入可用。
- PDF 导入至少能渲染每页为图片并进入页面列表。
- 自动版面分析失败时，用户可以手动画一个文字块继续 OCR。
- OCR 完成后自动进入校对入口，不重复触发识别完成逻辑。
- 项目保存后关闭再打开，页面、块、行、状态、人工修改结果不丢失。
- 导出前能提示未 OCR、未校对、疑点未处理数量。
- 测试可在不安装 PaddleOCR、不联网、没有真实 LLM key 的情况下通过。

## 3. 目标模块设计

建议新增或重构为以下模块。路径只是建议，执行时可按现有风格微调，但不要把所有逻辑继续堆在 `MainWindow`。

```text
app/
├── controllers/
│   ├── project_controller.py
│   └── workflow_controller.py
├── services/
│   ├── import_service.py
│   ├── image_preprocessor.py
│   ├── bbox_tightener.py
│   ├── ocr_pipeline.py
│   ├── proof_queue.py
│   └── llm_pre_review.py
├── engines/
│   ├── ocr_base.py
│   ├── paddle_ocr_engine.py
│   ├── aistudio_ocr_engine.py
│   ├── fake_ocr_engine.py
│   ├── llm_base.py
│   ├── llm_http_engine.py
│   └── fake_llm_engine.py
├── core/
│   ├── app_config.py
│   ├── logging.py
│   └── project_store.py
└── ui/
    ├── widgets/editable_image_viewer.py
    └── ...
```

### 3.1 为什么要拆 controller

当前 `MainWindow` 同时负责 UI、项目状态、存储、worker、步骤切换。这样容易出现“按钮触发了不该触发的业务逻辑”的 bug。正确做法是：

- UI 只发出用户意图，例如“用户点击开始 OCR”。
- Controller 判断当前状态是否允许。
- Service 执行业务。
- Store 保存数据。
- UI 根据 controller 发出的状态刷新。

核心原则：界面按钮不应该直接决定项目状态，项目状态应该由 workflow controller 统一维护。

## 4. 数据模型升级

### 4.1 当前模型

当前模型位于 `app/models/project.py`：

- `BBox(x, y, w, h)`
- `Char(char, confidence, bbox)`
- `Line(text, confidence, bbox, chars, proof_status, original_text)`
- `Block(block_type, bbox, lines, order)`
- `Page(image_path, width, height, blocks, page_number)`
- `OcrProject(name, pages, created_at, updated_at, db_path)`

### 4.2 建议新增枚举

在 `app/models/enums.py` 增加：

```python
class PageStatus(str, Enum):
    IMPORTED = "imported"
    PREPROCESSED = "preprocessed"
    LAYOUT_DONE = "layout_done"
    LAYOUT_CONFIRMED = "layout_confirmed"
    OCR_DONE = "ocr_done"
    PRE_REVIEW_DONE = "pre_review_done"
    PROOFING = "proofing"
    PROOF_DONE = "proof_done"
    ERROR = "error"

class BlockSource(str, Enum):
    AUTO_LAYOUT = "auto_layout"
    MANUAL_DRAW = "manual_draw"
    AUTO_TIGHTENED = "auto_tightened"
    USER_EDITED = "user_edited"

class LlmReviewStatus(str, Enum):
    DISABLED = "disabled"
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
```

### 4.3 建议扩展字段

`Page`：

- `source_path`: 原始导入文件路径。
- `source_type`: `image` 或 `pdf`。
- `source_page_index`: PDF 页码，从 0 或 1 开始必须全局统一，建议数据库存 1-based 给用户看。
- `cache_image_path`: PDF 渲染或预处理后的实际工作图片。
- `thumbnail_path`: 缩略图路径。
- `status`: 页面状态。
- `error_message`: 当前页失败原因，供 UI 显示。

`Block`：

- `source`: 自动版面/人工画框/自动收紧/用户编辑。
- `is_locked`: 用户锁定后自动分析不覆盖。
- `recognizable`: 是否送 OCR。图片、公式、装饰线默认 false。
- `note`: 用户备注或系统说明。

`Line`：

- `ocr_text`: OCR 原始文本。
- `llm_suggestion`: LLM 预审建议文本。
- `final_text`: 人工终审文本。为了兼容现有代码，可以短期继续让 `text` 表示 final text，但必须写清迁移逻辑。
- `llm_reason`: LLM 修改原因。
- `llm_review_status`: 预审状态。
- `review_flags`: JSON 字符串或独立表，记录错别字、低置信、疑似断行等标签。

### 4.4 文本分层规则

必须遵守：

```text
OCR 原文:       ocr_text / original_text
LLM 预审建议:  llm_suggestion
人工终审文本:  text 或 final_text
导出文本:      只读取人工终审文本
```

易错点：

- 不要让 LLM 返回值直接覆盖 `line.text`，否则用户无法区分机器改了什么。
- 不要在导出器里临时调用 LLM。
- 不要丢掉 OCR 原文，否则无法做修改对比。
- 不要把 `original_text == ""` 当作未修改的唯一判断，因为空文本本身可能是合法 OCR 结果。

## 5. SQLite 存储与迁移

### 5.1 当前问题

`ProjectStore.save_project()` 当前是“有 id 更新、无 id 插入”，但没有删除项目中已经移除的旧 page/block/line/char。如果用户重新跑版面分析，数据库可能保留旧块，重新打开项目时混入过期数据。

### 5.2 建议 schema 管理

新增 `meta` 表：

```sql
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

存储：

```text
schema_version = 2
app_version = 0.2.0
```

启动 `ProjectStore.open()` 后：

1. 开启 `PRAGMA foreign_keys=ON`。
2. 读取 `meta.schema_version`。
3. 如果没有，视为 v1。
4. 顺序执行迁移到当前版本。
5. 迁移必须在事务里完成。

### 5.3 保存策略

第一阶段推荐保守方案：保存项目时对每个 page 的 block/line/char 做“删除后重建”，避免复杂 upsert 出错。

伪逻辑：

```python
with transaction:
    upsert project
    for page in project.pages:
        upsert page
        DELETE FROM char WHERE line_id IN (...)
        DELETE FROM line WHERE block_id IN (...)
        DELETE FROM block WHERE page_id = page.id
        insert current blocks/lines/chars
```

注意：

- 删除和插入必须在同一事务中，失败要回滚。
- 重新插入后要把新的 id 回写到内存对象。
- 如果后续要支持操作历史，可改为 soft delete，但第一阶段不建议增加复杂度。

### 5.4 推荐新增表

`operation_log`：

```sql
CREATE TABLE operation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    page_id INTEGER,
    object_type TEXT NOT NULL,
    object_id INTEGER,
    action TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
```

用途：

- 撤销/重做。
- 审计人工修改。
- 生成校对报告。

`settings`：

```sql
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

项目级设置，例如默认导出格式、是否启用 LLM 预审、预审模型名。

全局设置建议用 `QSettings`，项目级设置放 SQLite。

## 6. 工作流控制器

### 6.1 状态流转

建议状态流转：

```text
NO_PROJECT
  -> PROJECT_OPENED
  -> IMPORT_READY
  -> LAYOUT_READY
  -> LAYOUT_CONFIRMED
  -> OCR_RUNNING
  -> OCR_DONE
  -> LLM_PRE_REVIEW_RUNNING (optional)
  -> LLM_PRE_REVIEW_DONE / LLM_PRE_REVIEW_SKIPPED
  -> PROOFING
  -> PROOF_DONE
  -> EXPORT_READY
```

### 6.2 UI 步骤启用规则

```text
导入页:   有项目即可进入
版面页:   至少有一页
OCR页:    至少有一个 recognizable block
校对页:   至少有 OCR 行
导出:     至少有 OCR 行；如果疑点未处理，弹确认提示
```

### 6.3 OCR 完成跳转修复

当前 `OcrPanel.recognition_done` 会连接到 `_on_ocr_done`，这会导致“进入校对”按钮重复走完成逻辑。应改为：

- OCR worker 完成时，controller 调用 `on_ocr_done()`，并加载校对数据。
- `OcrPanel` 的按钮只发出 `go_to_proof_requested`。
- `MainWindow` 或 `WorkflowController` 收到后跳转到横校页。

伪代码：

```python
class OcrPanel(QWidget):
    go_to_proof_requested = Signal()
    ...
    self._btn_next.clicked.connect(self.go_to_proof_requested)
```

易错点：

- 不要把“业务完成事件”和“用户点击下一步事件”复用成同一个 signal。
- Worker signal 只应该由后台任务发出。
- UI 下一步 signal 只表达导航意图。

## 7. 导入与预处理

### 7.1 ImportService 接口

```python
class ImportService:
    def import_paths(self, paths: list[str], project_cache_dir: str) -> list[Page]:
        ...
```

处理逻辑：

1. 校验路径存在。
2. 按扩展名分类。
3. 图片直接读取尺寸，复制或引用原路径。
4. PDF 用 PyMuPDF 渲染为 PNG/TIFF，输出到缓存目录。
5. 对每一页生成 `Page`。
6. 返回成功页和失败列表；不要因为一个文件失败中断全部导入。

### 7.2 PDF 渲染建议

使用 PyMuPDF：

```python
import fitz

doc = fitz.open(pdf_path)
for index, page in enumerate(doc):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    out_path = cache_dir / f"{stem}_p{index+1:04d}.png"
    pix.save(out_path)
```

注意：

- `Matrix(2, 2)` 大约是 144 DPI；如需更清晰可用 3 倍，但文件更大。
- 输出文件名必须稳定，避免重复导入生成大量垃圾。
- PDF 页码给用户显示建议从 1 开始。

### 7.3 图像预处理

第一阶段只做安全预处理，不要过度增强：

- 读取尺寸。
- 生成缩略图。
- 可选去黑边。
- 可选方向检测提示，但不要自动旋转用户图，除非可撤销。

后续可增加：

- 灰度增强。
- 自适应二值化。
- 倾斜校正。
- 页面边缘检测。

## 8. 可编辑画布

### 8.1 基础交互模式

`ImageViewer` 建议扩展或派生为 `EditableImageViewer`。

模式：

```python
class CanvasMode(str, Enum):
    PAN = "pan"
    SELECT = "select"
    DRAW_BLOCK = "draw_block"
    EDIT_BLOCK = "edit_block"
```

信号：

```python
block_created = Signal(object)      # Block
block_selected = Signal(object)     # Block
block_changed = Signal(object)      # Block
block_deleted = Signal(object)      # Block
```

### 8.2 坐标转换

QGraphicsView 中坐标有三层：

```text
viewport/screen 坐标 -> scene 坐标 -> image 像素坐标
```

现有代码将 pixmap 放在 scene 原点，因此 scene 坐标基本等于图像像素坐标。但后续如果加缩放、旋转、预处理图映射，必须显式封装坐标转换函数。

易错点：

- 鼠标事件里不要直接用 `event.pos()` 当图像坐标。
- 必须用 `mapToScene(event.pos())`。
- BBox 保存前要 clamp 到图片边界。
- 宽高不能为负数；从右下往左上拖拽时要 normalize。

### 8.3 画框流程

1. 用户选择块类型，例如文字/标题/表格。
2. 切换到 `DRAW_BLOCK`。
3. 鼠标按下记录起点。
4. 鼠标移动显示临时框。
5. 鼠标释放得到粗框。
6. 如果“自动收紧”开关开启，调用 `BBoxTightener.tighten()`。
7. 创建 `Block`，source 为 `MANUAL_DRAW` 或 `AUTO_TIGHTENED`。
8. 加入当前页面 blocks，刷新框列表。
9. 标记项目 dirty，触发自动保存。

### 8.4 撤销/重做

第一阶段可以只实现内存命令栈：

```python
class Command:
    def do(self): ...
    def undo(self): ...
```

命令类型：

- AddBlockCommand
- DeleteBlockCommand
- MoveBlockCommand
- ResizeBlockCommand
- ChangeBlockTypeCommand

后续再把命令记录同步到 `operation_log`。

## 9. 粗画框自动收紧

### 9.1 目标

用户只需框出大概范围，程序自动把框收紧到真实文字/表格/图形内容附近。收紧必须保守，不能裁掉文字。

### 9.2 BBoxTightener 接口

```python
@dataclass
class TightenOptions:
    padding: int = 6
    min_content_area_ratio: float = 0.002
    max_shrink_ratio: float = 0.85
    block_type: BlockType = BlockType.TEXT

@dataclass
class TightenResult:
    bbox: BBox
    changed: bool
    confidence: float
    reason: str

class BBoxTightener:
    def tighten(self, image_bgr: np.ndarray, rough_bbox: BBox, options: TightenOptions) -> TightenResult:
        ...
```

### 9.3 算法步骤

1. 把 rough bbox clamp 到图像范围。
2. 裁剪 ROI。
3. 转灰度。
4. 高斯模糊去噪。
5. Otsu 二值化。
6. 判断背景黑白，如果前景比例异常则反色。
7. 形态学闭运算连接文字笔画。
8. 找连通域或轮廓。
9. 过滤噪声小区域。
10. 合并剩余区域外接矩形。
11. 加 padding。
12. 转回整页坐标。
13. 若结果异常，返回原框和 reason。

### 9.4 失败保护

必须保护这些情况：

- ROI 为空。
- 用户画框太小。
- 找不到前景。
- 收紧后面积小于原框面积的某个极端比例，疑似裁掉内容。
- 收紧后宽高小于最小值。
- 表格线/图片区域被过度过滤。

建议失败时：

```text
changed = False
bbox = rough_bbox
reason = "未检测到足够内容，保留原框"
```

不要静默失败。

## 10. OCR 管线

### 10.1 OcrEngine 接口

```python
class OcrEngine(Protocol):
    def recognize(self, image_bgr: np.ndarray, context: OcrContext) -> list[Line]:
        ...

class LayoutEngine(Protocol):
    def analyze(self, image_path: str) -> list[Block]:
        ...
```

### 10.2 OcrPipeline 职责

`OcrPipeline` 不直接关心 UI。它负责：

- 遍历 page/block。
- 跳过不可识别块。
- 裁剪 block。
- 扩边。
- 调用 engine。
- 将行 bbox 从 crop 坐标转换回 page 坐标。
- 设置 proof status。
- 记录失败。
- 发出进度。

### 10.3 块级 OCR

必须支持：

- 识别整页所有块。
- 只识别当前选中块。
- 重跑失败块。
- 重跑后替换该块 lines，并删除旧 lines。

易错点：

- Crop 坐标转整页坐标时，要加上 block bbox 的 x/y。
- 扩边 crop 后，回写坐标要加 crop 起点，而不是原 block 起点。
- OCR 失败不要清空旧人工校对结果，除非用户明确确认重跑覆盖。

## 11. LLM 文本流预审接入

详细设计见 `docs/llm-pre-review.md`。这里列出主流程中的位置：

```text
OCR raw lines
  -> 保存 OCR 原文
  -> 如果 LLM 预审关闭: 标记 DISABLED，进入人工校对
  -> 如果开启: 发送文本流给 LLM
  -> 保存建议，不覆盖最终文本
  -> 人工在校对界面接受/拒绝/修改
  -> 导出人工最终文本
```

第一阶段建议默认关闭，只提供配置、数据结构、fake adapter 和 UI 占位。不要因为 LLM 功能拖慢主闭环。

## 12. 校对工作台

### 12.1 疑点队列

疑点来源：

- OCR 置信度低于阈值。
- OCR 文本为空。
- LLM 认为疑似错字。
- 行 bbox 太小或太大。
- 同页行框重叠。
- block OCR 失败。
- 用户标记“待复查”。

`ProofQueue` 输出：

```python
@dataclass
class ProofItem:
    page_index: int
    block_id: int | None
    line_id: int | None
    severity: str
    reason: str
```

### 12.2 人工终审原则

- 只有人工点击确认、保存修改或接受 LLM 建议后，才能改变最终文本状态。
- LLM 建议可一键接受，但接受动作仍算人工动作。
- 导出前如果有 `AUTO_FLAGGED` 或 `PENDING`，需要提示。

## 13. 导出

### 13.1 导出前检查

导出前统计：

- 总页数。
- 总行数。
- 未 OCR 块数。
- 未确认行数。
- 低置信行数。
- LLM 建议未处理数量。

如果存在风险，弹窗：

```text
仍有 12 行未确认、3 个块未识别。是否继续导出？
```

### 13.2 导出内容来源

必须统一：

```python
def get_export_text(line: Line) -> str:
    return line.text  # 或 future line.final_text
```

不要在不同导出器里各自决定读取 `ocr_text` 或 `llm_suggestion`。

## 14. 配置和安全

### 14.1 配置分层

全局配置：

- OCR 模式。
- API URL。
- 超时。
- UI 主题。
- LLM 预审默认开关。
- LLM endpoint。

项目配置：

- 当前项目是否启用 LLM 预审。
- 默认块类型。
- 导出格式偏好。

### 14.2 敏感信息

不要：

- 在源码中写默认真实 API 地址。
- 在日志中输出 token。
- 把用户原始文件绝对路径发给 LLM。
- 在测试快照里保存真实 token。

可以：

- 本机 `QSettings` 保存非敏感配置。
- Token 用环境变量或系统 keyring；如果暂时用 QSettings，必须在文档和 UI 中说明风险。

## 15. 推荐实施顺序

### Phase 0: 文档和基线

- 确认当前 `python tests/test_core.py` 可通过。
- 补齐 README 和技术文档。
- 明确样本目录 `file/`。

### Phase 1: 最小闭环

1. 补齐依赖：`requests`。
2. 修复步骤状态和 OCR 完成跳转。
3. 接入 PDF 转图。
4. 修复 ProjectStore 保存脏数据。
5. 导出前增加状态检查。
6. 所有外部 OCR 用 fake adapter 可测试。

### Phase 2: 人工画框

1. 可画框。
2. 可选中、移动、缩放、删除。
3. 框属性可改。
4. 保存/打开一致。

### Phase 3: 自动收紧

1. 实现 `BBoxTightener`。
2. 画完即收紧。
3. 批量收紧。
4. 合成图测试。

### Phase 4: OCR 和 LLM 预审

1. OCR engine adapter。
2. OCR pipeline。
3. LLM pre-review adapter。
4. fake LLM 测试。
5. 校对界面显示建议。

### Phase 5: 校对和 UI

1. 疑点队列。
2. 快捷键。
3. 完成度面板。
4. 工作台式 UI。

## 16. 高风险易错点清单

- PDF 路径不能直接传给 `cv2.imread()`。
- QGraphicsView 坐标不能直接当图像像素。
- OCR crop 扩边后 bbox 回写坐标容易偏移。
- 重新版面分析必须清理旧 block/line。
- LLM 建议不能覆盖人工最终文本。
- 导出不能读取 LLM suggestion 作为最终结果。
- 测试不能依赖真实 PaddleOCR、真实 LLM、外网。
- Token 不能写入源码和日志。
- 自动收紧不能过度裁剪。
- Worker 线程中不要直接操作 Qt UI。
- Signal 要区分“任务完成”和“用户请求下一步”。
- `.ocrproj` 是 SQLite，迁移失败要回滚，不能半升级。

## 17. 接手 agent 执行规则

1. 每做一个 phase 前先运行当前测试。
2. 每次只改一个明确模块，不要跨太多 UI 和核心逻辑。
3. 新功能必须有 fake adapter 或纯函数测试。
4. 引入新字段必须同步：模型、存储、加载、导出、测试。
5. UI 行为变化必须写测试或至少写可重复手工回归步骤。
6. 不要删除 `file/` 下样本。
7. 不要修改 build/dist 生成物。
8. 如果发现真实目录和文档不一致，以实际项目 `/mnt/d/project/ocr_process` 为准。
