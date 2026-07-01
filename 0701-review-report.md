我先给结论：**当前程序的数据模型不是“字段不够”，而是“核心模型被外部格式、UI 状态、OCR 路由、人工编辑、导出需求共同污染了”。**

这会导致你现在看到的过拟合：

```text
一个字段兼顾多种含义
raw_payload / app_payload 越来越大
保存时还要清理运行时字段
OCR 前还要判断到底用 Paddle 原始结构还是人工 layout
下游函数到处从 source_label / raw_payload / app_payload / block_type 里猜语义
```

我没有在容器里成功 clone 并运行全仓库测试，所以以下是基于 GitHub 可见源码和项目文档做的静态架构 review。项目 README 明确说这是 OCR 后处理桌面工具，包含导入、版面块、OCR、校对、导出等流程；`CURRENT_TRUTH_MAP.md` 也已经把 `Page.display_image_path`、`Page.uid`、`Block.uid`、`page.ppvl_parsing_res_list`、`block.raw_payload`、`block.app_payload` 等列成“当前真值图”，并说明它不是最终架构，而是现状事实图。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/tree/review-baseline-2026-07-01))

------

# 1. 总体诊断：现在的模型混了 6 类职责

当前 `Page / Block / Line / Char` 看似简单，但实际承担了这些职责：

```text
1. 外部 OCR 原始结果
2. 当前版面模型
3. 人工编辑状态
4. OCR 调度策略
5. OCR 识别结果
6. 校对最终文本
7. 临时路由 / 中间计算状态
```

最典型的是 `Block`。它现在有 `block_type`、`bbox`、`lines`、`order`、`source`、`recognizable`、`note`、`source_label`、`raw_payload`、`app_payload`、`uid` 等字段。这个结构已经不是单纯的“版面块”，而是把来源、当前状态、OCR 策略、外部引擎字段、应用派生状态都塞进了同一个对象。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/models/project.py))

`Line` 也有类似问题：`text`、`original_text`、`ocr_text`、`proof_state` 同时存在，而且 `__post_init__` 会用 `text` 补 `ocr_text`，再用 `ocr_text` 补 `original_text`，这说明 OCR 原始文本、当前显示文本、人工最终文本之间的边界已经模糊了。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/models/project.py))

------

# 2. 最严重的过拟合点

## 问题 A：`raw_payload` 和 `app_payload` 成了“万能垃圾桶”

`block_payload.py` 已经把很多应用内部字段集中到了 `app_payload`：包括 Paddle 绑定、OCR 文本失效、人工合并来源、人工画框 bbox、UI 生成的 inline formula、删除 inline formula、Hanwang bbox audit，以及 `_layout_line_routes`、`_route_subblocks` 这类运行时路由字段。代码里还专门有 `split_legacy_raw_payload`，把以前误塞进 `raw_payload` 的应用字段迁移到 `app_payload`。这说明问题已经被代码本身承认了：**raw 和 app 的边界曾经坏过，现在 app_payload 又成了新的混杂层。**([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/core/block_payload.py))

这类字段不应该都在 `Block.app_payload` 里。它们其实属于不同模型：

```text
paddle_binding                → BlockOrigin / RawRef
ocr_text_invalidated          → OcrState
manual_merge_from             → LayoutEditEvent
manual_draw_bbox              → BlockOrigin 或 LayoutEditEvent
ui_inline_formula_*           → InlineFormulaAnchor / InlineSegment
hanwang_bbox_audit            → OcrAuditArtifact
_layout_line_routes           → Runtime RoutingPlan
_route_subblocks              → Runtime RoutingPlan 或 OcrJobSnapshot
```

`app_payload` 的存在让程序继续逃避建模。短期方便，长期会把所有边界打穿。

------

## 问题 B：外部 OCR label 已经侵入核心枚举

`BlockType.from_paddle()` 里面直接把大量 Paddle / PP-VL / Hanwang / 兼容 label 映射进核心枚举，包括 `paragraph_text`、`doc_title`、`page_number`、`header`、`footer`、`inline_formula`、`formula_number`、`table_cell` 等，还带 substring fallback。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/models/enums.py))

这说明 `BlockType` 现在不是纯业务枚举，而是在替外部 OCR adapter 做兼容工作。

核心层不应该知道这么多外部 label。应该改成：

```text
PaddleLayoutImporter 负责识别 Paddle label
HanwangLayoutImporter 负责识别 Hanwang label
核心 BlockKind 只表达应用自己的稳定类型
```

也就是：

```python
# 核心层
class BlockKind(Enum):
    TEXT = "text"
    TITLE = "title"
    FIGURE = "figure"
    TABLE = "table"
    FORMULA = "formula"
    CAPTION = "caption"
    REFERENCE = "reference"
    MARGINALIA = "marginalia"
    UNKNOWN = "unknown"
```

然后外部 label 存在 provenance 里：

```python
source_label = "inline_formula"
source_engine = "paddleocr-vl"
```

不要让 `BlockKind` 去兼容未来所有 OCR 引擎。

------

## 问题 C：OCR 调度策略靠多个字段兜底猜语义

`ocr_dispatch_policy.py` 的 `OcrDispatchBlock` 协议要求 `block_type`、`source_label`、`raw_payload`、`app_payload`、`recognizable`。`authoritative_block_label()` 会依次从 `source_label`、`app_payload`、`raw_payload`、`block_type.value` 找“最可信 label”，然后再决定是否送 OCR。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/core/ocr_dispatch_policy.py))

这正是你说的“一个字段兼顾多种情况 + 多场景兜底”。

当前逻辑是：

```text
source_label 有就用 source_label
否则 app_payload 里找 paddle_binding
否则 raw_payload 里找 Paddle label
否则退回 block_type
再结合 recognizable
```

这说明系统没有一个明确字段表达：

```text
这个 block 当前是否应该进入文字 OCR？
为什么？
用什么 OCR 策略？
```

`recognizable: bool` 太弱。它至少混了这些含义：

```text
这个块是不是文字类？
用户有没有手动关闭 OCR？
这个块是否已经识别过？
这个块是否因编辑失效？
这个块是否应该保留但不识别？
公式/表格/图片是不是走特殊 OCR？
```

应该删除 `recognizable: bool`，改成明确策略：

```python
class OcrPolicy(Enum):
    TEXT_OCR = "text_ocr"
    FORMULA_PRESERVE = "formula_preserve"
    TABLE_PRESERVE = "table_preserve"
    FIGURE_SKIP = "figure_skip"
    IGNORE = "ignore"
    MANUAL_ONLY = "manual_only"
```

再配一个状态：

```python
class OcrState:
    status: Literal["never_run", "valid", "invalidated", "running", "failed"]
    last_run_id: str | None
    invalidated_by_edit_uid: str | None
    invalidation_reason: str | None
```

这样 OCR 调度不再到处猜 label。

------

## 问题 D：运行时路由状态被写进了持久化模型

`paddle_line_routing.py` 定义了 `_route_subblocks`、`_layout_line_routes`、`_layout_route_source` 等路由字段，这些明显是 OCR 组装过程的中间状态。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/core/paddle_line_routing.py))

更严重的是 `ProjectStore._save_block()` 保存 block 时会调用 `_strip_runtime_layout_routes_from_dict(block.raw_payload)` 和 `_strip_runtime_layout_routes_from_dict(block.app_payload)`；如果发现这些运行时 route，且 block 里有 lines，就会直接 `block.lines = []`，并把 `OCR_TEXT_INVALIDATED_KEY` 写入 `app_payload`。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/core/project_store.py))

这是一个非常危险的边界问题：

```text
持久化层不应该修改业务内容。
保存项目不应该清空 OCR lines。
清理运行时字段不应该触发文本失效。
```

这说明当前模型里运行时临时状态、OCR 结果、持久化快照混在一起了。

正确做法：

```text
RoutingPlan 是运行时对象
OcrJobSnapshot 可以保存某次 OCR 任务的输入快照
Block 不保存 _layout_line_routes
ProjectStore 只保存，不修改业务语义
```

保存时发现脏数据，可以拒绝保存或记录 migration warning，但不要在 `_save_block()` 里隐式清空文本。

------

## 问题 E：存储层过度容错掩盖坏数据

`ProjectStore` 里的 `_json_to_list()` 和 `_json_to_dict()` 遇到 JSON 错误或类型不对时直接返回 `[]` / `{}`。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/core/project_store.py))

这也是过拟合式兼容：

```python
坏了也能读
类型错了也能跑
字段不对也兜底
```

但核心模型不应该这么做。正确边界是：

```text
migration/importer 可以容错
核心 ProjectStore 读取当前 schema 时必须严格
```

建议改成：

```python
def parse_json_dict_strict(value: str, *, field: str) -> dict:
    try:
        parsed = json.loads(value)
    except JSONDecodeError as exc:
        raise ProjectDataError(f"{field} invalid json") from exc

    if not isinstance(parsed, dict):
        raise ProjectDataError(f"{field} must be dict")

    return parsed
```

旧项目兼容放在：

```text
LegacyProjectMigrator
```

不要放在正常读取路径。

------

# 3. 重新树立业务边界

你现在需要把模型拆成这几层：

```text
Raw OCR Artifact
    ↓ importer / adapter
Block Origin / Provenance
    ↓
Current Layout Model
    ↓
OCR Job / OCR Observation
    ↓
Proof / Final Text
    ↓
Export
```

对应责任如下。

------

## 边界 1：Raw OCR Artifact，只保存外部事实

职责：

```text
保存 Paddle / Hanwang / 其他 OCR 的原始输出
可回溯
可 debug
可重新 normalize
不参与画布、OCR 调度、导出主流程
```

建议模型：

```python
@dataclass(frozen=True)
class RawOcrArtifact:
    uid: str
    page_uid: str
    engine: str
    engine_version: str
    run_id: str
    artifact_path: str
    artifact_hash: str
    created_at: float
```

当前 `Page.ppvl_parsing_res_list` 应该迁走。现在的 truth map 也承认 `page.ppvl_parsing_res_list` 是 Paddle 原始 list，而 `block.raw_payload` 是 vendor fact；这些不应该长期挂在 active Page/Block 上。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/CURRENT_TRUTH_MAP.md))

处理决定：

```text
Page.ppvl_parsing_res_list：从 Page active model 删除
迁移到 raw_ocr_artifact 或 raw_ocr_page_result
Page 只保留 raw_ocr_run_id / raw_artifact_ref
```

------

## 边界 2：BlockOrigin，只表达“这个 block 从哪里来”

职责：

```text
记录 block 创建来源
记录 OCR 原始 bbox/type/label/confidence
指向 raw artifact
不表达当前状态
不表达人工编辑
```

建议模型：

```python
@dataclass(frozen=True)
class BlockOrigin:
    created_by: Literal[
        "paddle_layout",
        "manual_draw",
        "split",
        "merge",
        "imported_project"
    ]
    source_engine: str | None
    source_run_id: str | None
    source_label: str | None
    source_confidence: float | None
    original_bbox: BBox | None
    original_kind: BlockKind | None
    raw_ref: RawRef | None
```

字段迁移：

```text
Block.source           → 拆成 origin.created_by + edit history
Block.source_label     → 移到 BlockOrigin.source_label
Block.raw_payload      → 移到 RawOcrArtifact，通过 raw_ref 引用
manual_draw_bbox       → 如果是初始人工创建，进入 origin.original_bbox
paddle_binding         → 进入 origin.raw_ref
```

`BlockSource` 当前有 `AUTO_LAYOUT`、`MANUAL_DRAW`、`AUTO_TIGHTENED`、`USER_EDITED`，但它混淆了“创建来源”和“编辑状态”。`USER_EDITED` 不是来源，`AUTO_TIGHTENED` 也不是来源，它们是事件或修订。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/models/enums.py))

------

## 边界 3：LayoutBlock，只表达当前有效版面

职责：

```text
画布显示
人工拖拽
OCR 输入
导出布局
质量检查
```

它必须非常干净。

建议模型：

```python
@dataclass
class LayoutBlock:
    uid: str
    page_uid: str

    kind: BlockKind
    bbox: BBox
    reading_order: int

    deleted: bool = False
    ocr_policy: OcrPolicy = OcrPolicy.TEXT_OCR
    revision: int = 0

    origin: BlockOrigin | None = None
    user_note: str = ""
    flags: list[BlockFlag] = field(default_factory=list)
```

核心规则：

```text
bbox 永远是当前有效 bbox
bbox 永远是 work image 坐标
画布坐标不入库
Paddle bbox 不放在 bbox，而放在 origin.original_bbox
人工修改 bbox 只改 current bbox，并记录 LayoutEditEvent
```

字段处理：

```text
Block.block_type       → 改名 kind / layout_kind
Block.bbox             → 保留，定义为 current bbox
Block.order            → 改名 reading_order
Block.uid              → 保留，业务 ID
Block.id               → 保留但仅限数据库 rowid，不参与业务判断
Block.note             → 拆成 user_note + system_flags
Block.recognizable     → 删除，改 ocr_policy + ocr_state
Block.lines            → 从 LayoutBlock 移走
```

`Block.lines` 不应该挂在版面块里。它让版面结构和 OCR 观察结果绑定在一起。应该改成：

```text
LayoutBlock 只知道自己是一个版面块
OcrLine 通过 block_uid 关联它
ProofLineState 通过 line_uid 关联 OCR line
```

------

## 边界 4：LayoutEditEvent，人工编辑是事件，不是字段补丁

职责：

```text
记录人工操作
支持回滚
支持 diff
支持解释为什么 OCR 失效
支持判断哪些 block 被改过
```

建议模型：

```python
@dataclass(frozen=True)
class LayoutEditEvent:
    uid: str
    page_uid: str
    target_uid: str | None

    op: Literal[
        "create_block",
        "move_block",
        "resize_block",
        "change_kind",
        "delete_block",
        "restore_block",
        "merge_blocks",
        "split_block",
        "change_order",
        "change_ocr_policy"
    ]

    before: dict
    after: dict
    actor: Literal["user", "system", "migration"]
    created_at: float
```

字段迁移：

```text
MANUAL_MERGE_FROM_KEY      → LayoutEditEvent(op="merge_blocks")
MANUAL_DRAW_BBOX_KEY       → LayoutEditEvent(op="create_block") 或 BlockOrigin
USER_EDITED                → 不存字段，通过是否有 user edit event 推导
AUTO_TIGHTENED             → LayoutEditEvent(op="resize_block", actor="system")
ocr_text_invalidated       → OcrState，由 edit event 触发
```

这样你就不会再需要：

```python
if _page_layout_has_user_edits(page):
    用人工 layout
else:
    用 Paddle 原始 layout
```

统一变成：

```python
layout_blocks = build_current_layout(page)
```

未编辑 block：

```text
current == origin.original
```

已编辑 block：

```text
current = original + edit events 后的快照
```

------

## 边界 5：OCR Job / OCR Observation，只表达识别结果

职责：

```text
记录某次 OCR 使用了什么输入
记录 OCR 结果
保留行、字符、token、bbox、confidence
不表达人工最终文本
不表达版面来源
```

建议模型：

```python
@dataclass(frozen=True)
class OcrRun:
    uid: str
    page_uid: str
    engine: str
    engine_version: str
    input_layout_revision: int
    created_at: float

@dataclass
class OcrLine:
    uid: str
    run_uid: str
    page_uid: str
    block_uid: str

    raw_text: str
    bbox: BBox
    confidence: float
    reading_order: int
    tokens: list[OcrToken]
```

`Char` 现在有 `char`、`bbox_source`、`bbox_granularity`、`token_text`，这说明它有时是字符，有时是 token，有时又承载公式片段。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/models/project.py))

建议改名为 `OcrToken`：

```python
@dataclass
class OcrToken:
    uid: str
    line_uid: str

    text: str
    bbox: BBox | None
    confidence: float

    granularity: Literal["char", "word", "formula", "line"]
    source: str
```

这样就不需要：

```text
char + token_text + bbox_granularity
```

改成一个清晰概念：

```text
OCR token observation
```

------

## 边界 6：Proof / Final Text，只表达人工最终文本

职责：

```text
记录人工校对结果
记录最终文本是否确认
记录校对状态
导出优先使用这里
```

你现在已经有独立的 `proof_line_state` 表，包含 `final_text`、`final_text_set`、`proof_status`、`alignment_state`、`updated_at`。这个方向是对的。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/core/project_store.py))

但 `Line.text / original_text / ocr_text / proof_state` 又把这些概念混回了 Line 里。建议收紧：

```python
@dataclass
class ProofLineState:
    line_uid: str
    final_text: str
    final_text_set: bool
    proof_status: ProofStatus
    updated_at: float
```

显示文本统一由函数计算：

```python
def display_text(ocr_line: OcrLine, proof: ProofLineState | None) -> str:
    if proof and proof.final_text_set:
        return proof.final_text
    return ocr_line.raw_text
```

删除或降级这些字段：

```text
Line.text           → 删除，或仅作为 display cache，不能持久化为权威
Line.original_text  → 删除，除非明确定义为 migration snapshot
Line.ocr_text       → 改名 OcrLine.raw_text
Line.proof_state    → 不嵌入 Line，按 line_uid 关联
```

------

# 4. 字段级裁决表

## Page

| 当前字段                                        | 裁决              | 新定义                                                    |
| ----------------------------------------------- | ----------------- | --------------------------------------------------------- |
| `Page.uid`                                      | 保留              | 页面业务 ID，跨数据库 rowid 稳定                          |
| `Page.id`                                       | 保留但降级        | SQLite rowid，仅存储层使用                                |
| `image_path`                                    | 改名 / 收敛       | 不再作为主图路径，迁入 `source.path` 或 `work_image.path` |
| `cache_image_path`                              | 改名              | 迁入 `work_image.path`                                    |
| `display_image_path`                            | 保留为只读属性    | 返回 `work_image.path`，不作为持久化核心字段              |
| `width / height`                                | 移入 `work_image` | 明确这是当前工作图坐标系尺寸                              |
| `source_path / source_type / source_page_index` | 保留              | 导入来源，不参与画布坐标                                  |
| `ppvl_parsing_res_list`                         | 删除 active 字段  | 迁入 `RawOcrArtifact`                                     |
| `status`                                        | 拆分              | `layout_status / ocr_status / proof_status`               |
| `ocr_invalidated_reason`                        | 下沉              | 迁入 `OcrState`，不要挂 Page 字符串                       |

当前代码已经把 `display_image_path` 定义为 layout、OCR、crop、render、export geometry 的工作图路径，这个方向是对的；问题是 `image_path/cache_image_path/source_path` 的职责还不够强约束。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/models/project.py))

------

## Block

| 当前字段       | 裁决        | 新定义                                  |
| -------------- | ----------- | --------------------------------------- |
| `uid`          | 保留        | block 业务 ID                           |
| `id`           | 保留但降级  | DB rowid                                |
| `block_type`   | 改名        | `kind: BlockKind`，只表达核心业务类型   |
| `bbox`         | 保留        | 当前有效 bbox，work image 坐标          |
| `order`        | 改名        | `reading_order`                         |
| `lines`        | 移除        | OCR 结果通过 `block_uid` 关联           |
| `source`       | 删除 / 拆分 | `origin.created_by` + `LayoutEditEvent` |
| `recognizable` | 删除        | `ocr_policy` + `ocr_state`              |
| `note`         | 拆分        | `user_note` + `system_flags`            |
| `source_label` | 移动        | `BlockOrigin.source_label`              |
| `raw_payload`  | 从核心删除  | 进入 RawOcrArtifact，通过 raw_ref 查    |
| `app_payload`  | 从核心删除  | 拆成 typed model / event / state        |

------

## Line / Char

| 当前字段             | 裁决         | 新定义                                |
| -------------------- | ------------ | ------------------------------------- |
| `Line.uid`           | 保留         | OCR line 或 proof line 业务 ID        |
| `Line.text`          | 删除权威性   | 不作为持久化权威文本                  |
| `Line.ocr_text`      | 改名         | `OcrLine.raw_text`                    |
| `Line.original_text` | 删除         | 若保留，只作 migration/debug snapshot |
| `Line.proof_state`   | 外置         | `ProofLineState(line_uid=...)`        |
| `Char.char`          | 改名         | `OcrToken.text`                       |
| `Char.token_text`    | 删除         | 被 `OcrToken.text` 替代               |
| `bbox_source`        | 收敛         | `OcrToken.source` 或 `GeometrySource` |
| `bbox_granularity`   | 保留但枚举化 | `char / word / formula / line`        |

------

# 5. 新的核心模型建议

可以先不用一次性全部数据库化，但 Python 领域模型应该先这样定。

```python
@dataclass(frozen=True)
class WorkImage:
    path: str
    width: int
    height: int
    dpi: int | None = None
    file_hash: str | None = None
@dataclass
class Page:
    uid: str
    page_number: int

    source: PageSource
    work_image: WorkImage

    layout_status: LayoutStatus
    ocr_status: OcrStatus
    proof_status: ProofStatus

    last_error: PageError | None = None
@dataclass
class LayoutBlock:
    uid: str
    page_uid: str

    kind: BlockKind
    bbox: BBox
    reading_order: int

    deleted: bool = False
    ocr_policy: OcrPolicy = OcrPolicy.TEXT_OCR
    ocr_state: OcrState = field(default_factory=OcrState)

    origin: BlockOrigin | None = None
    revision: int = 0

    user_note: str = ""
    flags: list[BlockFlag] = field(default_factory=list)
@dataclass(frozen=True)
class BlockOrigin:
    created_by: CreatedBy
    source_engine: str | None
    source_run_id: str | None
    source_label: str | None
    source_confidence: float | None
    original_bbox: BBox | None
    original_kind: BlockKind | None
    raw_ref: RawRef | None
@dataclass(frozen=True)
class RawRef:
    artifact_uid: str
    json_path: str
    raw_index: int | None = None
@dataclass(frozen=True)
class LayoutEditEvent:
    uid: str
    page_uid: str
    target_uid: str | None
    op: str
    before: dict
    after: dict
    actor: str
    created_at: float
@dataclass
class OcrLine:
    uid: str
    run_uid: str
    page_uid: str
    block_uid: str
    raw_text: str
    bbox: BBox
    confidence: float
    tokens: list[OcrToken]
@dataclass
class ProofLineState:
    line_uid: str
    final_text: str
    final_text_set: bool
    proof_status: ProofStatus
    updated_at: float
```

------

# 6. 新数据库边界

当前 SQLite schema 把 `ppvl_parsing_res_list_json` 存在 `page` 表，把 `raw_payload_json/app_payload_json` 存在 `block` 表，把 `proof_line_state` 单独存表。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/core/project_store.py))

建议目标 schema：

```text
page
  id
  uid
  page_number
  source_path
  source_type
  source_page_index
  work_image_path
  work_image_width
  work_image_height
  layout_status
  ocr_status
  proof_status
  last_error_json

raw_ocr_artifact
  id
  uid
  page_uid
  engine
  engine_version
  run_id
  artifact_path
  artifact_hash
  created_at

layout_block
  id
  uid
  page_uid
  kind
  x
  y
  w
  h
  reading_order
  deleted
  ocr_policy
  ocr_state_json
  revision
  user_note

block_origin
  block_uid
  created_by
  source_engine
  source_run_id
  source_label
  source_confidence
  original_x
  original_y
  original_w
  original_h
  original_kind
  raw_artifact_uid
  raw_json_path
  raw_index

layout_edit_event
  id
  uid
  page_uid
  target_uid
  op
  before_json
  after_json
  actor
  created_at

ocr_run
  id
  uid
  page_uid
  engine
  engine_version
  input_layout_revision
  created_at

ocr_line
  id
  uid
  run_uid
  page_uid
  block_uid
  raw_text
  x
  y
  w
  h
  confidence
  reading_order

ocr_token
  id
  uid
  line_uid
  text
  x
  y
  w
  h
  confidence
  granularity
  source

proof_line_state
  line_uid
  final_text
  final_text_set
  proof_status
  alignment_state
  updated_at
```

`proof_line_state` 现有方向可以保留，只需要从 `Line` 内嵌关系里解耦出来。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/core/project_store.py))

------

# 7. 你现在应该立即改的几个代码点

## 第一优先级：禁止核心流程直接消费 `ppvl_parsing_res_list`

目标：

```python
layout_blocks = build_current_layout(page)
```

不要再有：

```python
if page_has_user_edits:
    use_page_blocks_from_layout()
else:
    use_page.ppvl_parsing_res_list
```

`ppvl_parsing_res_list` 只在 importer / migration / debug 使用。

------

## 第二优先级：冻结 `app_payload` 的新增写入

`app_payload` 可以暂时保留用于兼容旧项目，但新代码不要再写新 key。

加规则：

```text
核心代码禁止新增 app_payload key
新增状态必须建 typed field / typed model
```

然后建立迁移映射：

```text
paddle_binding              → BlockOrigin.raw_ref
ocr_text_invalidated        → OcrState.status = invalidated
ocr_invalidation_kind       → OcrState.invalidation_reason
manual_merge_from           → LayoutEditEvent
manual_draw_bbox            → BlockOrigin / LayoutEditEvent
ui_inline_formula_*         → InlineFormulaAnchor
ui_deleted_inline_formula   → LayoutEditEvent(op="delete_inline_formula")
hanwang_bbox_audit          → OcrAuditArtifact
_layout_line_routes         → Runtime RoutingPlan
_route_subblocks            → Runtime RoutingPlan
```

------

## 第三优先级：把 `BlockType.from_paddle()` 移出核心枚举

改成：

```text
app/adapters/paddle/layout_importer.py
app/adapters/hanwang/layout_importer.py
```

核心枚举只保留业务类型。

```python
def map_paddle_label_to_block_kind(label: str) -> BlockKind:
    ...
```

这个函数属于 adapter，不属于 `BlockKind`。

------

## 第四优先级：`ProjectStore` 不允许有业务副作用

当前 `_save_block()` 里发现 runtime route 后会清理 payload，并清空 `block.lines`、写入 OCR invalidated 标记。这个必须移走。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/core/project_store.py))

目标：

```text
ProjectStore.save(project)
只做序列化
不改变 project 内容
不清空 lines
不触发 OCR 失效
不修复业务状态
```

如果发现 runtime route 泄漏：

```python
raise ProjectDataError("runtime routing data leaked into persistent block")
```

或者在 migration 层处理。

------

## 第五优先级：把 `recognizable` 替换成 `ocr_policy`

不要再用 bool。

```python
class OcrPolicy(Enum):
    TEXT_OCR = "text_ocr"
    PRESERVE_AS_FORMULA = "preserve_as_formula"
    PRESERVE_AS_TABLE = "preserve_as_table"
    SKIP = "skip"
    MANUAL_ONLY = "manual_only"
```

OCR dispatch 只看：

```python
block.ocr_policy
block.deleted
block.ocr_state
```

不要再从 `source_label / app_payload / raw_payload / block_type` 里猜。当前调度策略依赖这些字段组合，是模型边界不清的直接表现。([GitHub](https://github.com/zhanghuai-z/ocr_process_review/blob/review-baseline-2026-07-01/app/core/ocr_dispatch_policy.py))

------

# 8. 新的业务不变量

你需要给 agent 和代码都立这几条硬规则：

```text
1. Raw OCR 结果不可变。
2. Page / Block active model 不直接持有 Paddle 原始 JSON。
3. LayoutBlock.bbox 是唯一当前有效 bbox。
4. BlockOrigin.original_bbox 是模型初始 bbox。
5. 人工编辑只改 current snapshot，并追加 LayoutEditEvent。
6. OCR 输入只来自 current layout。
7. OCR 结果不挂在 LayoutBlock 内部，通过 block_uid 关联。
8. Proof final text 不写回 OCR raw text。
9. Runtime route 不入库，不进入 Block payload。
10. ProjectStore 不做业务修复。
11. fallback 只能在 importer / migration 层出现，核心层必须严格。
```

------

# 9. 迁移路线

## Phase 1：先立刻止血

不大改数据库，只加约束：

```text
新代码禁止写 raw_payload / app_payload
新代码禁止直接读 ppvl_parsing_res_list 做 OCR 输入
新代码禁止在 ProjectStore 里修改业务对象
```

同时加 validator：

```python
validate_page_model(page)
validate_block_model(block)
validate_no_runtime_routes_in_persistent_payload(block)
```

------

## Phase 2：引入新模型，但兼容旧字段

写一个转换层：

```python
def migrate_legacy_page(page: Page) -> NewPageModel:
    ...
```

迁移规则：

```text
page.ppvl_parsing_res_list        → RawOcrArtifact
block.raw_payload                 → RawOcrArtifact raw_ref
block.app_payload.paddle_binding  → BlockOrigin.raw_ref
block.source_label                → BlockOrigin.source_label
block.source                      → BlockOrigin.created_by 或 LayoutEditEvent
block.bbox                        → LayoutBlock.bbox
block.block_type                  → LayoutBlock.kind
block.order                       → LayoutBlock.reading_order
block.recognizable                → LayoutBlock.ocr_policy
block.lines                       → OcrLine / OcrToken
line.proof_state                  → ProofLineState
```

------

## Phase 3：替换核心读路径

按这个顺序替换：

```text
Canvas 渲染        → LayoutBlock
人工编辑           → LayoutBlock + LayoutEditEvent
OCR 输入构造       → build_current_layout(page)
OCR 结果写入       → OcrRun / OcrLine / OcrToken
校对显示           → display_text(ocr_line, proof_state)
导出               → proof final 优先，否则 ocr raw
```

------

## Phase 4：删除旧字段

等所有读路径迁完后再删：

```text
Page.ppvl_parsing_res_list
Block.raw_payload
Block.app_payload
Block.recognizable
Block.source
Block.lines
Line.text
Line.original_text
Line.ocr_text
Char.token_text
```

其中 `raw_payload/app_payload` 可以保留在 `legacy_payload_json` 里一段时间，但不要再进入核心模型。

------

# 10. 最终目标模型心智

你现在要从这个模型：

```text
Page
  ppvl_parsing_res_list
  blocks
    block_type
    bbox
    source
    recognizable
    source_label
    raw_payload
    app_payload
    lines
      text
      original_text
      ocr_text
      proof_state
```

转成这个模型：

```text
RawOcrArtifact
  原始外部 OCR 输出，只读

BlockOrigin
  这个 block 从哪里来，原始 bbox/type/label 是什么

LayoutBlock
  当前有效版面，画布和 OCR 输入只看它

LayoutEditEvent
  人工如何修改了 layout

OcrRun / OcrLine / OcrToken
  某次 OCR 的观察结果

ProofLineState
  人工最终文本

Exporter
  只读 current layout + proof display text
```

一句话：

```text
raw 是证据。
origin 是来源。
layout 是当前事实。
edit 是变更历史。
ocr 是观察结果。
proof 是人工结论。
export 是消费端。
```

当前程序过拟合的根源，就是这些概念被压进了 `Page / Block / Line` 和两个万能 dict 里。下一步不是继续加字段兜底，而是把这些职责拆开，让核心模型变窄、变硬、变可验证。