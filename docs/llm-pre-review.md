# LLM 文本流预审设计

LLM 预审是 OCR 之后、人工校对之前的中间处理层。它的定位是“提供建议”，不是“替代校对”。人工终审永远优先，导出永远读取人工最终文本。

## 1. 功能定位

### 1.1 用户价值

OCR 后文本常见问题：

- 同形字、近形字识别错误。
- 标点错漏。
- 断行不合理。
- 竖排文本顺序不自然。
- 专有名词或古籍用字疑似错误。
- 低置信行需要优先复核。

LLM 可以在人工校对前做预审：

- 给出疑似错误位置。
- 提供建议文本。
- 解释修改理由。
- 生成疑点标签。
- 帮助排序校对优先级。

### 1.2 必须遵守的边界

- 功能必须可关闭。
- 默认建议关闭，直到基础闭环稳定。
- LLM 失败不能阻断 OCR、校对和导出。
- LLM 不直接改最终文本。
- 人工点击接受建议才算最终修改。
- 所有测试必须可用 fake LLM，不依赖外网。

## 2. 主流程

```text
OCR 完成
  -> Line.ocr_text = OCR 原文
  -> Line.text = OCR 原文   # 当前兼容做法
  -> 如果 LLM 预审关闭:
        Line.llm_review_status = DISABLED
        进入人工校对
     如果开启:
        构造文本批次
        调用 LLM adapter
        保存 Line.llm_suggestion / Line.llm_reason / review_flags
        Line.text 仍保持 OCR 原文
        进入人工校对
  -> 人工接受/拒绝/修改
  -> 导出 Line.text 或 future Line.final_text
```

## 3. 配置项

建议配置：

```python
{
    "llm_pre_review_enabled": False,
    "llm_provider": "fake",       # fake/http/openai-compatible/local
    "llm_endpoint": "",
    "llm_model": "",
    "llm_timeout": 30,
    "llm_batch_lines": 30,
    "llm_send_context": True,
    "llm_send_page_text_only": True,
}
```

说明：

- `llm_pre_review_enabled` 是总开关。
- `fake` provider 用于测试和离线开发。
- `llm_batch_lines` 控制一次发送多少行，避免 prompt 过大。
- 第一阶段只发送文本，不发送图片。

## 4. 数据结构

### 4.1 Line 扩展字段

建议字段：

```python
ocr_text: str
llm_suggestion: str
llm_reason: str
llm_review_status: LlmReviewStatus
review_flags: list[str]
```

短期兼容：

- 现有 `Line.text` 仍表示当前最终文本。
- OCR 完成时 `Line.original_text` 或 `Line.ocr_text` 保存 OCR 原文。
- LLM 建议保存到新字段，不覆盖 `Line.text`。

### 4.2 LLM 输入

```python
@dataclass
class LlmPreReviewLine:
    page_number: int
    block_order: int
    line_index: int
    text: str
    confidence: float
    block_type: str
    previous_text: str | None = None
    next_text: str | None = None
```

### 4.3 LLM 输出

```python
@dataclass
class LlmPreReviewSuggestion:
    page_number: int
    block_order: int
    line_index: int
    original_text: str
    suggested_text: str
    confidence: float
    reason: str
    flags: list[str]
```

必须要求 LLM 返回结构化 JSON，不要让 UI 从自然语言段落里解析。

## 5. Adapter 设计

### 5.1 基础接口

```python
class LlmPreReviewEngine(Protocol):
    def review_lines(
        self,
        lines: list[LlmPreReviewLine],
        options: LlmPreReviewOptions,
    ) -> list[LlmPreReviewSuggestion]:
        ...
```

### 5.2 Fake adapter

Fake adapter 是必需的，不是可选项。

用途：

- 回归测试。
- UI 联调。
- 无外网环境开发。

行为建议：

- 对包含固定标记的文本返回固定建议，例如 `错別字` -> `错别字`。
- 对低置信行加 `low_confidence` flag。
- 对空文本加 `empty_text` flag。

### 5.3 HTTP adapter

HTTP adapter 必须：

- 设置 timeout。
- 捕获网络错误并返回明确失败状态。
- 不在日志输出 token。
- 校验响应 JSON。
- 对响应条数和输入条数不一致做容错。

## 6. Prompt 要求

系统提示应明确：

```text
你是 OCR 文本预审助手。你的任务是指出疑似 OCR 错误并给出建议。
不要改写文风，不要补充原文没有的信息。
如果无法确定，请保持原文并添加 uncertain 标记。
只返回 JSON。
```

用户内容建议：

```json
{
  "task": "pre_review_ocr_lines",
  "rules": [
    "只纠正明显 OCR 识别错误",
    "不要扩写",
    "不要翻译",
    "不要删除不确定的生僻字",
    "保留原始换行粒度"
  ],
  "lines": [
    {
      "page_number": 1,
      "block_order": 3,
      "line_index": 12,
      "text": "OCR文本",
      "confidence": 0.72,
      "previous_text": "...",
      "next_text": "..."
    }
  ]
}
```

输出格式：

```json
{
  "suggestions": [
    {
      "page_number": 1,
      "block_order": 3,
      "line_index": 12,
      "original_text": "OCR文本",
      "suggested_text": "OCR文本",
      "confidence": 0.6,
      "reason": "未发现明确错误",
      "flags": []
    }
  ]
}
```

## 7. UI 设计

校对界面应显示：

```text
OCR 原文
LLM 建议
人工最终文本编辑框
按钮：接受建议 / 拒绝建议 / 保存人工修改 / 标记待复查
```

状态颜色：

- 未预审：灰色。
- 预审建议：蓝色。
- 预审失败：黄色。
- 人工接受：绿色。
- 人工拒绝：灰色。

易错点：

- “接受建议”必须是人工动作，点击后才写入最终文本。
- “拒绝建议”只改变建议状态，不改变最终文本。
- 如果用户已经人工修改，再返回 LLM 建议，不要覆盖用户输入。

## 8. 存储设计

推荐短期在 `line` 表增加字段：

```sql
ALTER TABLE line ADD COLUMN ocr_text TEXT NOT NULL DEFAULT '';
ALTER TABLE line ADD COLUMN llm_suggestion TEXT NOT NULL DEFAULT '';
ALTER TABLE line ADD COLUMN llm_reason TEXT NOT NULL DEFAULT '';
ALTER TABLE line ADD COLUMN llm_review_status TEXT NOT NULL DEFAULT 'disabled';
ALTER TABLE line ADD COLUMN review_flags_json TEXT NOT NULL DEFAULT '[]';
```

如果后续需要多轮 LLM 建议，新增表：

```sql
CREATE TABLE llm_review (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id INTEGER NOT NULL REFERENCES line(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    original_text TEXT NOT NULL,
    suggested_text TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    flags_json TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);
```

第一阶段建议先用 line 字段，降低复杂度。

## 9. 错误处理

LLM 预审失败时：

- 不弹阻断式错误，除非用户主动点击“立即预审”。
- 在状态栏显示“LLM 预审失败，已跳过，可继续人工校对”。
- 对每行设置 `llm_review_status = FAILED` 或保持 PENDING 并记录批次错误。
- 日志记录技术原因。

不能：

- 清空 OCR 文本。
- 阻止用户进入校对。
- 阻止导出。

## 10. 测试要求

必须覆盖：

1. 开关关闭时不调用 LLM adapter。
2. 开关开启时 fake adapter 返回建议并保存。
3. LLM 建议不覆盖 `Line.text`。
4. 人工接受建议后才更新 `Line.text`。
5. LLM 异常时流程继续。
6. 导出读取人工最终文本，不读取未接受建议。
7. token 不出现在日志。
8. 输入批次过大时能分批。

## 11. 隐私和安全

第一阶段只发送文本，不发送图片。

不发送：

- 用户文件绝对路径。
- 项目数据库路径。
- Token。
- 原图。
- 大段无关上下文。

发送：

- 当前行文本。
- 上下行文本。
- 置信度。
- 页码和块序号。

如果未来接入多模态图像 LLM，必须增加独立开关：

```text
允许发送图像片段给 LLM 进行预审
```

并在 UI 明确告知用户。
