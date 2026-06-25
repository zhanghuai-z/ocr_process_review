# UI 顾问入口说明

本文档用于让 UI 顾问基于参考图和当前程序实际界面，输出可执行的 UI 调整方案。目标不是重写业务逻辑，而是明确：哪些文件控制哪些界面区域、哪些参数可以调整、哪些约束不能破坏。

## 参考材料

- 目标参考图：`question_img/ui.png`
- 当前异常截图：`question_img/3.png`
- 图标素材雪碧图：`question_img/素材.png`
- 测试项目样例：`question_img/项目测试1.ocrproj`

当前主要问题：顶部流程入口已经移动到上方，但整体视觉仍不像参考图。原因是工作台三栏、边界过渡、右侧工具栏、底部栏仍然是旧 UI 壳层，不能只靠顶栏改造达到参考图质感。

## 启动入口

主程序入口：

```bash
python main.py
```

入口文件：

- `main.py`
  - 创建 `QApplication`
  - 调用 `app.ui.style.apply_light_theme(app)`
  - 实例化并显示 `app.ui.main_window.MainWindow`

注意：当前存在两套样式系统。

- `app/ui/style.py`
  - 当前 `main.py` 启动时实际使用。
  - 如果只运行 `python main.py`，优先看这里的 QSS。
- `app/ui/styles/__init__.py`
  - 新主题系统，包含 `LIGHT_TOKENS / DARK_TOKENS / apply_theme()`。
  - 菜单里的主题切换和部分测试会走这里。

UI 顾问如果建议修改样式，需要明确：是只改当前启动样式 `app/ui/style.py`，还是同步修改新主题系统 `app/ui/styles/__init__.py`。为了避免行为不一致，建议两个位置同步。

## 主窗口壳层

文件：`app/ui/main_window.py`

核心类：

- `MainWindow(QMainWindow)`
  - `_build_ui()`：主窗口布局。
  - 当前结构是：
    - 顶部：`TopBar`
    - 中间：`QStackedWidget`
    - 底部：`QStatusBar`
  - 左侧旧竖向导航 `NavRail` 已删除，不应恢复。

窗口尺寸相关：

- `_resize_for_initial_import_page()`
  - 导入页初始尺寸目标：`960 x 540`
- `_resize_for_initial_workbench_page()`
  - 工作台初始尺寸目标：屏幕可用区域的 `80%`
  - 不超过屏幕可用区域减 `40px`
- `_set_centered_window_size(target_w, target_h)`
  - 负责从中心向四边扩展
  - 已加入 `minimumSizeHint()` 约束

流程页对应关系：

- `STEP_IMPORT` -> `ImportPanel`
- `STEP_LAYOUT` / `STEP_OCR` -> `LayoutPanel`
- `STEP_HPROOF` -> `HProofPanel`
- `STEP_VPROOF` -> `VProofPanel`

约束：

- 不要让 UI 直接修改 workflow 状态。
- 顶部流程按钮点击必须继续走 `self._controller.request_step(step)`。
- 不要重新引入左侧竖向主导航。

## 顶部流程栏

文件：`app/ui/widgets/top_bar.py`

核心类：

- `TopBar(QWidget)`

当前布局：

- 左区：项目名 / 当前步骤 / 状态徽章
- 中区：流程分段按钮
- 右区：运行版面分析 / 导出

当前关键参数：

```python
TopBar height = 56
project label min width = 160
project label max width = 360
workflow segment margins = 4, 4, 4, 4
workflow segment spacing = 2
版面分析 button width = 96
横校 button width = 74
纵校 button width = 74
workflow button height = 30
运行版面分析 width = 138
导出 width = 76
right action button height = 32
```

当前 objectName：

- `headerBar`
- `topBarLeft`
- `workflowSegment`
- `workflowStepBtn`
- `topBarActions`
- `crumbProject`
- `crumbSep`
- `crumbStep`
- `statusPill`
- `primaryBtn`
- `ghostBtn`

顾问重点：

- 参考图中顶部流程更像“全局工作流标签”，不是普通按钮。
- 当前顶栏已经比裸按钮好，但视觉仍偏扁、偏普通。
- 可以建议高度、圆角、背景、按钮间距、选中态、项目面包屑布局。
- 不建议把菜单栏也自绘，除非明确要放弃系统菜单。

## 导入页

文件：`app/ui/recognize/import_panel.py`

角色：

- 初始界面。
- 用户导入图片/PDF 后生成项目页面。

顾问只需要关注：

- 导入页是否需要跟工作台统一背景。
- 空状态区域、按钮、文件列表的视觉是否需要调整。

约束：

- 不要改变导入信号 `images_ready`。
- 不要改项目创建逻辑。

## 版面分析工作台

文件：`app/ui/recognize/layout_panel.py`

核心类：

- `LayoutPanel(QWidget)`

当前结构：

- 主布局：`QVBoxLayout`
- 主区域：`QSplitter(Qt.Horizontal)`
  - 左：页面目录 / 标题目录 `QTabWidget`
  - 中：viewer 工具条 + `ImageViewer`
  - 右：框类型工具栏 + 项目统计
- 底部：翻页 / 状态 / 取消 / 完成本页 / 提交并进入 OCR

当前三栏参数：

```python
splitter stretch = [1, 9, 2]
splitter sizes = [220, 980, 320]
right_scroll min width = 280
right_scroll max width = 380
viewer toolbar height = 34
right sidebar header height = 44
bottom bar height = 46
page nav button size = 30 x 30
progress bar size = 120 x 6
```

当前可调对象：

- `layoutLeftTabs`
- `headingOutlineTree`
- `viewerToolbar`
- `toolToggle`
- `secondaryBtn`
- `typeBadge`
- `layoutSidebarHeader`
- `sidebarTitle`
- `sidebarBar`
- `layoutStatsPanel`
- `layoutToolScroll`
- `toolbar`
- `pageNavBtn`
- `pageNavLabel`
- `primaryBtn`

右侧工具栏业务内容：

- 核心文本：正文 / 摘要 / 公式
- 标题层级：H1-H6
- 图表与浮动：图片 / 图题 / 表格 / 表题 / 图表
- 边注与引用：页眉 / 页脚 / 页码 / 脚注 / 参考文献

约束：

- 右侧按钮数量和绑定逻辑先不改。
- 可以重做分组视觉、间距、颜色、滚动区域、标题样式。
- 不要删除 `Shift+左键拖拽`、`Space 长按` 等操作逻辑。
- 画布交互由 `ImageViewer` 控制，不要在样式方案中改变交互语义。

顾问重点：

- 参考图的三栏边界更柔和，当前三栏像硬拼接。
- 当前右侧工具栏按钮密度高、视觉层级弱。
- 顶部 viewer 工具条、右侧工具栏、底部栏之间缺少统一的工作台容器感。

## 图像画布与框交互

文件：`app/ui/widgets/image_viewer.py`

核心类：

- `ImageViewer(QGraphicsView)`
- `BBoxItem(QGraphicsRectItem)`

角色：

- 显示页面图像。
- 显示/编辑版面框和字符框。
- 处理缩放、拖拽、新建框、右键框选等交互。

约束：

- UI 顾问不建议改这里的算法逻辑。
- 如果只做视觉，可以建议：
  - 画布背景色
  - 画布外边距
  - 框线粗细和颜色
  - hover / selected 状态
  - 浮动工具条位置

当前样式入口：

- 全局 `QGraphicsView` 样式在 `app/ui/style.py` 和 `app/ui/styles/__init__.py`
- 具体框线颜色多半在 `image_viewer.py` 内部逻辑中，不完全由 QSS 控制。

## 页面目录

文件：`app/ui/widgets/page_directory.py`

核心类：

- `PageDirectoryList(QListWidget)`
- `_PageRow(QWidget)`

当前参数：

```python
thumbnail width = 38
thumbnail height = 46
row height = 62
list min width = 180
list max width = 240
```

被复用位置：

- 版面分析左侧页面目录
- 横校左侧页面目录

顾问重点：

- 参考图左侧页面栏更像独立浅色面板。
- 当前页缩略图和页信息较简陋，可以建议：
  - 缩略图尺寸
  - 行高
  - 选中态
  - 页码与文件名排版
  - 面板标题与搜索/筛选入口是否需要

## 横校界面

文件：`app/ui/proof/h_proof.py`

核心类：

- `HProofPanel(QWidget)`
- `_LinePair(QFrame)`
- `_SlotLineEditor(QWidget)`

当前结构：

- 主体：`QSplitter(Qt.Horizontal)`
  - 左：`PageDirectoryList`
  - 中：行级校对列表
  - 右：操作 / 当前 / 统计 / 调试
- 底部：统计状态栏

当前关键参数：

```python
IMAGE_ROW_H = 32
right dock min width = 220
right dock max width = 300
splitter sizes = [180, 9999, 180]
statusbar height = 28
line active bar width = 4
```

顾问重点：

- 横校要求“图和字 y 轴对齐，输入框以字符为单位，校对以行为单位，字号与图中近似”。
- 当前行对组件 `_LinePair` 是核心视觉单元。
- 不要把编辑文本和最终文本语义混在 UI 建议里；只建议视觉参数。

## 纵校界面

文件：`app/ui/proof/v_proof.py`

核心类：

- `VProofPanel(QWidget)`
- `_GalleryListView(QListView)`
- `_GalleryDelegate(QStyledItemDelegate)`
- `_SlotAwareTextEdit(QPlainTextEdit)`

当前结构：

- 顶部工具栏：上一页 / 下一页 / 保存 / 页码 / 置信度
- 主体：`QSplitter(Qt.Horizontal)`
  - 左：字符索引
  - 右：gallery + 候选字 + OCR 文本 + 图像查看

当前关键参数：

```python
CHAR_LIST_THUMB = 18
GALLERY_THUMB = 56
toolbar height = 46
left box min width = 110
left box max width = 170
main splitter stretch = [1, 7]
content split sizes = [proof_min_w + 40, 720]
candidate panel height = 64
gallery thumb display size approx = GALLERY_THUMB + 14
```

顾问重点：

- 纵校目前信息密度高，适合做“工具型工作台”，不要做营销式大卡片。
- 左侧字符索引、gallery、文本区、图片区需要更清晰的层级。
- 可以建议面板标题、空状态、选中态、候选按钮样式。

## 导出对话框

文件：`app/ui/export/export_dialog.py`

角色：

- 导出设置和导出动作入口。

当前阶段不是 UI 重构主线，但顾问可以顺手指出：

- 设置项分组
- 文件格式入口
- 进度/错误提示

## 设置对话框

文件：`app/ui/widgets/api_settings_dialog.py`

角色：

- OCR/API/模型配置。

注意：

- 这是独立复杂弹窗。
- 不建议第一轮一起重构。
- 如果顾问给建议，应单独列为第二阶段。

## 样式入口与 objectName

当前样式主要通过 Qt QSS 和 `objectName` 控制。

活跃启动样式：

- `app/ui/style.py`

新主题系统：

- `app/ui/styles/__init__.py`
- `app/ui/styles/README.md`

重要 objectName：

- 全局壳层：
  - `headerBar`
  - `toolbar`
  - `sidebarBar`
  - `card`
- 顶部流程：
  - `workflowSegment`
  - `workflowStepBtn`
  - `topBarLeft`
  - `topBarActions`
- 标签：
  - `sectionTitle`
  - `sectionDesc`
  - `fieldLabel`
  - `noteLabel`
  - `muted`
  - `sidebarTitle`
  - `statusPill`
- 按钮：
  - `primaryBtn`
  - `ghostBtn`
  - `secondaryBtn`
  - `dangerBtn`
  - `toolToggle`
  - `pageNavBtn`
- 版面分析：
  - `layoutLeftTabs`
  - `viewerToolbar`
  - `layoutSidebarHeader`
  - `layoutStatsPanel`
  - `layoutToolScroll`
  - `blockTypeGroup`
  - `blockTypeButton`
  - `blockTypeGroupTitle`
  - `typeBadge`
- 页面目录：
  - `pageDirectoryList`
  - `pageRow`
  - `pageThumb`
  - `pageRowTitle`
  - `pageRowFile`
  - `pageBadge`
- 横校：
  - `linePair`
  - `hproofSplitter`
  - `hproofRightDock`
  - `hproofModeBanner`
- 纵校：
  - `candidatePanel`

## 图标素材

路径：`question_img/素材.png`

内容：

- 4 x 4 图标雪碧图
- 包括打开、保存、导出、撤销、重做、搜索、删除、文档图片、版面框、文本、公式、表格、图片、标题、放大、移动等图标风格

建议：

- 顾问可以指定哪些按钮需要图标。
- 不建议直接把整张图塞进 UI。
- 如果要接入，需要后续单独切图或做 `QIcon` sprite mapper。

## 可生成当前截图的最小脚本

如果需要在无显示环境下生成当前主窗口截图，可用：

```bash
QT_QPA_PLATFORM=offscreen python - <<'PY'
from tests.test_core import _get_qapp
from app.ui.styles import apply_theme
from app.ui.main_window import MainWindow
from app.controllers.workflow_controller import STEP_LAYOUT, STEP_VPROOF

app = _get_qapp()
apply_theme(app, "light")
w = MainWindow()
w.resize(1536, 860)
w._top_bar.set_project_name("项目测试1")
w._top_bar.set_status("idle", "未运行")
w._top_bar.set_enabled_up_to(STEP_VPROOF)
w._controller.set_current_step(STEP_LAYOUT)
w.show()
app.processEvents()
w.grab().save("/tmp/ocr_ui_preview.png")
w.close()
print("/tmp/ocr_ui_preview.png")
PY
```

注意：

- 这只是截图脚本，不代表正式运行路径。
- 正式运行仍是 `python main.py`。
- 如果顾问要对比真实用户体验，应在 Windows 桌面直接运行程序或打包 exe。

## 测试入口

全量测试：

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```

UI 相关定向测试：

```bash
QT_QPA_PLATFORM=offscreen pytest -q tests/test_core.py -k "top_bar or main_window"
```

版面交互相关测试：

```bash
QT_QPA_PLATFORM=offscreen pytest -q tests/test_core.py -k "layout_panel or image_viewer"
```

说明：

- 当前全量测试基线：`629 passed, 18 warnings`
- warnings 为已有 `QMouseEvent` deprecated，不是 UI 重构新增阻断。

## 顾问输出期望

请 UI 顾问不要只给“好看/不好看”的评价，而是输出可落地参数。

建议输出格式：

1. 总体布局方案
   - 顶部栏高度、左右中三区比例、流程按钮视觉。
   - 左/中/右工作台宽度建议。
   - 底部栏是否保留、如何统一。

2. 文件级操作清单
   - 例如：`app/ui/widgets/top_bar.py` 调整哪些宽高。
   - 例如：`app/ui/recognize/layout_panel.py` 调整 splitter、右侧工具栏、viewer toolbar。
   - 例如：`app/ui/style.py` / `app/ui/styles/__init__.py` 增加哪些 objectName 样式。

3. 参数表
   - 颜色：背景、面板、边框、hover、selected、primary。
   - 间距：外边距、面板内边距、按钮间距。
   - 圆角：顶栏、分段控件、面板、按钮。
   - 字号：菜单、顶栏、面板标题、按钮、状态字。
   - 阴影：是否使用；若使用，Qt QSS 是否可行。

4. 分阶段建议
   - 第一阶段：只改顶栏和三栏壳层。
   - 第二阶段：右侧工具栏分组和按钮视觉。
   - 第三阶段：横校/纵校内部组件。
   - 第四阶段：图标素材接入。

5. 风险说明
   - 哪些建议只改样式。
   - 哪些建议会碰布局结构。
   - 哪些建议会碰交互逻辑，暂缓。

## 明确约束

- 不要恢复左侧主导航。
- 不要改变 `WorkflowController` 的流程语义。
- 不要改 OCR、版面框、校对保存、导出逻辑。
- 不要把右侧工具栏业务按钮删掉。
- 不要把开发调试需求包装成面向普通用户的大段说明文字。
- 不要做营销式首页；这个软件是工具型工作台。
- UI 方案必须服务于长时间人工校对，优先清晰、稳定、低干扰。
