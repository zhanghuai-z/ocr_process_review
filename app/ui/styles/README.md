# app/ui/styles — 全局主题与样式约定

## 模块

- `__init__.py` — `LIGHT_TOKENS` 字典 + `_QSS_TEMPLATE` + `apply_theme(app, name)` 入口。

## 用法

```python
from app.ui.styles import apply_theme
apply_theme(app, "light")
```

读取/持久化用户偏好：

```python
from app.core.app_config import AppConfig
from app.ui.styles import apply_theme

cfg = AppConfig.instance()
apply_theme(app, cfg.get("theme"))   # 自动 normalize；旧 dark/dark_teal 会回落到 light
```

## token 命名约定

| 类别 | 命名 | 用途 |
|---|---|---|
| 背景 | `bg_root` / `bg_panel` / `bg_card` / `bg_hover` / `bg_selected` / `bg_input` / `bg_input_dis` / `bg_status` | 三层结构：root 是整窗背景，panel 是面板，card 是浮起卡片 |
| 边框 | `border` / `border_input` / `border_focus` / `border_subtle` | 常规 / 输入框 / 焦点态 / 细分割 |
| 文本 | `text_primary` / `text_secondary` / `text_muted` / `text_disabled` / `text_on_brand` | 三阶 + 禁用 + 反白 |
| 品牌 | `brand` / `brand_hover` / `brand_pressed` / `brand_disabled` / `brand_border_lt` | 主操作色 + 状态 |
| 状态 | `accent_green` / `danger` / `danger_bg` / `danger_border` | 运行 / 危险 |
| 字号 | `font_size_xs` (12) / `font_size_sm` (13) / `font_size` (14) / `font_size_lg` (16) / `font_size_xl` (20) | |
| 圆角 | `radius_sm` (2) / `radius_md` (4) / `radius_lg` (8) / `radius_xl` (16) | |

## objectName 约定

任何 widget 想被全局 QSS 主题样式化，**必须** 设置 `setObjectName(...)`，否则会用 fallback 通用样式。

### 容器
- `headerBar` — 面板顶栏（白底，下边框）
- `workflowSegment` — 顶部流程分段按钮容器
- `topBarLeft` / `topBarActions` — 顶部栏左右透明分区
- `sidebarBar` — 左侧侧栏（白底，右边框）
- `toolbar` — 面板内工具条（白底，下边框）
- `card` — 浮起卡片（白底 + radius_lg + border）；属性 `selected="true"` 切换高亮
- `linePair` — 横校原文/校对行对容器；属性 `active="true"` 切换高亮

### 标签
- `sectionTitle` — 卡片标题（品牌色 14px bold）
- `sectionDesc` — 卡片描述（muted 12px）
- `fieldLabel` — 表单字段标签（secondary 13px）
- `noteLabel` / `muted` — 注释 / 弱化文字（muted 12px）
- `pageTitle` — 大标题（品牌色 18px bold）
- `stepInfo` — 步骤提示（品牌色 12px）
- `pageSep` — 页分隔条
- `proofEmpty` — 校对空状态居中文字

### 按钮
- `primaryBtn` — 主操作（品牌底白字）
- `ghostBtn` — 次操作（浅蓝底蓝字）；支持 `:checked`
- `dangerBtn` — 危险操作（白底红字红边）
- `runBtn` — 圆形启动按钮（绿色）
- `workflowStepBtn` — 顶部流程步骤按钮；支持 `:checked` / `:disabled`
- `toolToggle` — 切换型工具按钮（如 `⊞ 字框`）；支持 `:checked`

### 默认控件
`QLineEdit / QComboBox / QSpinBox / QPlainTextEdit / QTextEdit / QListWidget / QTreeWidget / QTableWidget / QProgressBar / QScrollBar / QSplitter / QTabWidget / QGraphicsView / QToolTip` 已统一样式，无需 objectName。

## 新增 objectName 流程

1. 在本 README 表格中先登记名字 + 用途。
2. 在 `__init__.py` 的 `_QSS_TEMPLATE` 中添加对应规则，**只使用 token 占位符**，不硬编码颜色。
3. 在 widget 代码 `setObjectName("xxx")`。
4. 验证浅色主题 (`apply_theme(app, "light")`) 正常。

## 严禁

- 在 widget 代码 `setStyleSheet("color: #1a73e8")` 硬编码颜色 → 必须走 objectName + 全局 QSS。
- 在 `__init__.py` 外硬编码 token 值。
