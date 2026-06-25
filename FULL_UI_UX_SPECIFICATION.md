# OCR 项目全局 UI/UX 视觉重构规范 (终极融合超长版 V5)

> 顾问按：上一版由于直接覆盖导致大量组件级 CSS 丢失，实在抱歉。本版 V5 已将**所有像素级 CSS、网格间距体系**与**您的全部补充意见（入口切流、图标降级、19 个按钮排布、验收脚本）**进行了究极融合。请项目 Agent 将此文档视为不可违背的“圣经”。

在实际的 PySide6 与多系统环境下，真正的 Pixel-perfect 是极其困难且容易产生大量 hardcode 的，因此本文档的核心诉求是：**绝对保证比例、色彩、状态反馈、间距体系的高度一致性，彻底消除原生 Qt 的违和感。**

---

## 零、 阶段声明 (Phasing Strategy)
- **第一阶段 (Phase 1)**：仅聚焦于全局样式框架、**TopBar (顶栏)**、**左侧目录**以及**版面分析视图 (LayoutPanel 及其右侧工具栏)**。
- **第二阶段 (Phase 2)**：暂缓处理 `h_proof.py` (横校) 和 `v_proof.py` (纵校) 中极其复杂的画廊布局、字符索引，避免第一阶段步子迈得太大导致崩溃。

---

## 一、 系统入口与样式接管方案
当前系统存在新旧两套样式入口：`app/ui/style.py` 与 `app/ui/styles/__init__.py`。Agent 必须在代码中完成无缝接管：
1. **保留主入口逻辑**：`main.py` 依然可以调用 `apply_light_theme(app)`。
2. **底层切流**：进入 `app/ui/style.py` 中，将 `apply_light_theme` 函数内部的旧版 `LIGHT_QSS` 强行覆盖逻辑删除，**直接桥接调用新主题系统**：
   ```python
   from app.ui.styles import apply_theme

   def apply_light_theme(app) -> None:
       """在 QApplication 上应用全局浅色主题。"""
       apply_theme(app, "light")
   ```

---

## 二、 设计哲学与全局规范 (Design Philosophy & Global Rules)

### 2.1 容器内边距定律与 8pt 网格
- **8pt 网格系统**：所有的 padding、margin、width、height 必须是 4 或 8 的倍数（如 4, 8, 12, 16, 24, 32）。
- **The Padding Law**：除了用于充满屏幕的 `QSplitter` 或 `ScrollArea` 外，任何包含子控件的面板容器，**绝对禁止**出现 `setContentsMargins(0, 0, 0, 0)`。必须留有呼吸感（例如 `(16, 12, 16, 12)`）。

### 2.2 核心色彩体系 (Color Tokens)
Agent 必须将以下色值严格定义在 `app/ui/styles/__init__.py` 的 `LIGHT_TOKENS` 中：
- `bg_root`: `#F0F2F5` (底层画布背景)
- `bg_panel`: `#FFFFFF` (侧边栏、主工具栏、弹窗底色)
- `bg_hover`: `#F5F5F5` (任意透明控件悬停态)
- `bg_selected`: `#E6F7FF` (列表选中态、按钮按下态)
- `bg_input`: `#FFFFFF` (输入框底色)
- `border_subtle`: `#F0F0F0` (极弱分割线)
- `border`: `#E8E8E8` (全局 1px 常规面板分割线)
- `border_input`: `#D9D9D9` (输入框、普通按钮边框)
- `text_primary`: `#1F2937` (重要标题、正文文本)
- `text_secondary`: `#4B5563` (次级信息)
- `text_muted`: `#8C8C8C` (分组说明、弱提示文本)
- `brand`: `#1677FF` (主色调、主按钮、激活高亮)
- `brand_hover`: `#4096FF`

---

## 三、 消除 QSplitter 双重边框与布局重症
导致界面粗糙的第一元凶是 `QSplitter` 会与侧边栏自带的 border 产生双重黑线。
1. **剔除冗余**：取消所有直接挂载在 `QSplitter` 左右的 Widget 的 `border-right` 和 `border-left`（如 `sidebarBar`、`headerBar`）。
2. **Splitter 唯一化**：将把手 (Handle) 作为全场唯一分割线：
```css
QSplitter::handle { background: #E8E8E8; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
```

---

## 四、 顶栏模块 (TopBar)
- **外层约束**：`setFixedHeight(56)`，底层背景 `#FFFFFF`，底部边框 `1px solid #E8E8E8`。内部 `QHBoxLayout` 的 `margin=(16, 0, 16, 0)`。
- **胶囊分段控制器 (`#workflowSegment`)**：
  使用 `QFrame` 配合 `QPushButton`：
  ```css
  QFrame#workflowSegment {
      background: #F0F2F5; border: 1px solid #E8E8E8; border-radius: 16px; padding: 2px;
  }
  QPushButton#workflowStepBtn {
      border: none; border-radius: 14px; padding: 4px 16px; color: #4B5563; background: transparent; font-size: 13px; font-weight: 500;
  }
  QPushButton#workflowStepBtn:checked {
      background: #FFFFFF; color: #1677FF; border: 1px solid #D9D9D9; font-weight: 600;
  }
  ```

---

## 五、 左侧目录与中间画布 (Left Sidebar & Canvas)

### 5.1 左侧 Tabs (`#layoutLeftTabs`)
```css
QTabWidget#layoutLeftTabs::pane { border: none; border-top: 1px solid #E8E8E8; }
QTabWidget#layoutLeftTabs QTabBar::tab { background: transparent; border: none; padding: 8px 16px; border-bottom: 2px solid transparent; color: #4B5563; }
QTabWidget#layoutLeftTabs QTabBar::tab:selected { color: #1677FF; border-bottom: 2px solid #1677FF; font-weight: 600; }
```

### 5.2 左侧页面列表 (`QListWidget#pageDirectoryList`)
必须通过 QSS 切底透明化，依靠外层接管高亮：
```css
QListWidget#pageDirectoryList { background: #FFFFFF; border: none; outline: 0; }
QListWidget#pageDirectoryList::item { background: transparent; padding: 0; margin: 0; border: none; }
QListWidget#pageDirectoryList::item:hover { background: #F5F5F5; }
QListWidget#pageDirectoryList::item:selected { background: #E6F7FF; color: #1677FF; border-left: 3px solid #1677FF; }
/* 用于承载略缩图的自绘Widget需声明 WA_TranslucentBackground */
QWidget#pageRow { background: transparent; }
```

### 5.3 中间画布区域背景
放置图像的区域背景必须是深一层级的 `#F0F2F5`，让 A4 画布像一张纸一样浮现出来。 Toolbar 高度 `44px` 或 `48px`。

---

## 六、 右侧属性面板：19个按钮的精细排布 (重灾区)
当前 `layout_panel.py` 中存在 19 个框型按钮，**一个都不能少**。外层宽度至少 `280px`。内部 `QScrollArea` 必须无边框透明。

### 6.1 分组阵列架构 (`BLOCK_TYPE_BUTTON_GROUPS`)
Agent 需按以下 4 组严格排列，并设置对应的 Grid 列数：
1. **核心与结构** (4个)：正文, 摘要, 公式, 参考文献。 **(2列)**
2. **标题层级** (6个)：H1 ~ H6。 **(3列)**
3. **图表元素** (5个)：图片, 图题, 表格, 表题, 图表。 **(2列，末尾“图表” span=2 独占一行)**
4. **页边元素** (4个)：页眉, 页脚, 页码, 脚注。 **(2列)**

### 6.2 间距与样式细节
- 分组外层 `QVBoxLayout`：`setContentsMargins(16, 8, 16, 16)`。
- 分组小标题 (`QLabel`)：`font-size: 12px; color: #8C8C8C; margin-bottom: 8px;`
- Grid 内部：`setSpacing(8)`。
- **按钮尺寸极限**：所有 `#blockTypeButton` 必须 `setFixedHeight(32)`，不可拉伸过高。
- **按钮 QSS**：
  ```css
  QPushButton#blockTypeButton {
      background: #FFFFFF; border: 1px solid #D9D9D9; border-radius: 6px; color: #4B5563; font-size: 13px;
  }
  QPushButton#blockTypeButton:hover { border-color: #4096FF; color: #4096FF; }
  QPushButton#blockTypeButton:checked {
      background: #E6F7FF; border: 1px solid #1677FF; color: #1677FF; font-weight: 600;
  }
  ```

---

## 七、 弹窗、输入框与基础按钮 (Forms & Dialogs)
所有的弹窗 (`layoutFindDialog`)、下拉菜单均必须消除 Windows 原生 3D 内凹阴影。
```css
QDialog#layoutFindDialog { background: #FFFFFF; border: 1px solid #E8E8E8; border-radius: 8px; }
QLineEdit, QComboBox, QTextEdit {
    background: #FFFFFF; border: 1px solid #D9D9D9; border-radius: 6px; padding: 6px 12px; min-height: 24px; color: #1F2937;
}
QLineEdit:focus, QComboBox:focus { border: 1px solid #1677FF; }
/* QComboBox下拉菜单修复 */
QComboBox QAbstractItemView { background: #FFFFFF; border: 1px solid #E8E8E8; selection-background-color: #E6F7FF; selection-color: #1677FF; }

/* Primary 主按钮 (蓝底白字) */
QPushButton#primaryBtn { background: #1677FF; color: white; border: none; border-radius: 6px; padding: 6px 16px; }
QPushButton#primaryBtn:hover { background: #4096FF; }

/* Default 次按钮 (白底边框) */
QPushButton#defaultBtn { background: #FFFFFF; color: #1F2937; border: 1px solid #D9D9D9; border-radius: 6px; padding: 6px 16px; }
QPushButton#defaultBtn:hover { border-color: #4096FF; color: #4096FF; }
```

---

## 八、 图标资产引入方案 (Asset Management)
考虑到当前资产包的限制，**允许项目 Agent 降级使用现有的 PNG 切图素材（如 `question_img/素材.png`）作为第一阶段的替代方案**。
但约束如下：
1. **强制尺寸约束**：即便是切片 PNG，也必须在代码或 QSS 中严格绑定尺寸，如 `setIconSize(QSize(16, 16))`，防止图片在大按钮中被无限拉伸导致模糊。
2. **重心微调**：如果在带图标的按钮（如 Toolbar 里的“字框”）发现图标偏上/偏下，必须通过 QSS 的 `padding-top` 或 `padding-bottom` 1px 微调重心。
3. **架构预留**：代码结构依然推荐写为类似 `#btn { qproperty-icon: url(...) }` 的形式，以便未来直接将 `.png` 替换为 `.svg` 实现无缝进化。

---

## 九、 验收与测试脚本 (Acceptance Checklist)
接手此规范的 Agent 必须使用下述脚本输出无头视口截图，确认：(1) 没有原生黑框; (2) 19个按钮排布精准; (3) 左栏列表无遮挡黑块。

**在项目根目录创建 `preview.py` 并运行 `QT_QPA_PLATFORM=offscreen python preview.py`**：
```python
import sys
import os
from tests.test_core import _get_qapp
from app.ui.styles import apply_theme
from app.ui.main_window import MainWindow
from app.controllers.workflow_controller import STEP_LAYOUT

app = _get_qapp()
# 强制桥接至新系统
apply_theme(app, "light")
w = MainWindow()
w.resize(1536, 860)

w._top_bar.set_project_name("验收测试项目 - 样式检查")
w._controller.set_current_step(STEP_LAYOUT)
w.show()
app.processEvents()

output_path = "ui_preview_acceptance.png"
w.grab().save(output_path)
print(f"✅ 验收截图已保存至: {output_path}")
w.close()
```
