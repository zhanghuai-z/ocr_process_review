# OCR 后处理桌面软件

这是一个面向图片/PDF OCR 后处理的桌面工具原型，目标是把“导入建档、版面分析、人工画框、OCR 识别、LLM 预审、人工终审、导出归档”做成可恢复、可批处理、可人工干预的工作台。

当前项目已具备 PySide6 桌面界面、SQLite `.ocrproj` 项目文件、PaddleOCR/AiStudio API 调用、版面块模型、横校/纵校原型和多格式导出雏形，但仍处于重构前阶段。后续开发请先阅读 `docs/technical-implementation.md` 和 `docs/regression-test-plan.md`。

## 当前目录结构

```text
.
├── app/
│   ├── core/          # OCR、版面分析、项目存储、校对规则
│   ├── export/        # TXT/XML/HTML/PDF/DOCX/RTF 导出器
│   ├── models/        # OcrProject/Page/Block/Line/Char/BBox 等领域模型
│   └── ui/            # PySide6 主窗口、导入/版面/OCR/校对/导出界面
├── docs/              # 技术实施文档、测试回归文档、LLM 预审设计
├── file/              # 测试样本和历史 OCR 相关文件
├── resources/         # 字体等资源
├── tests/             # 当前基础单元测试
├── main.py            # 程序入口
├── pyproject.toml     # Python 项目配置
├── build.bat          # Windows .exe 唯一正式打包入口
└── build.sh           # WSL 下转调 build.bat 的桥接脚本
```

## 环境要求

- Python 3.10+
- Windows 或 Linux/WSL 开发环境
- 基础依赖：PySide6、qt-material、opencv-python、Pillow、lxml、fpdf2、python-docx、Jinja2、PyMuPDF、numpy、requests
- 可选 OCR 依赖：paddlepaddle、paddleocr

注意：当前 `pyproject.toml` 还缺少 `requests` 声明，后续工程化第一阶段需要补齐。文档任务不修改源码，因此这里先记录为已知问题。

## 运行

建议使用虚拟环境：

```bash
cd /mnt/d/project/ocr_process
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python main.py
```

如果本地 PaddleOCR 安装困难，可以先使用 API/mock 方案推进工作流。真实 OCR 引擎不应成为 UI、存储、导出和回归测试的阻塞条件。

## 打包

Windows `.exe` **只使用 `build.bat`**：

```bat
build.bat
build.bat --clean
```

如果你在 WSL 中工作，可以运行：

```bash
./build.sh
./build.sh --clean
```

但 `build.sh` 现在只是桥接到 `build.bat`，**不会再自行运行 Linux PyInstaller**，也不再尝试生成 Linux 二进制后冒充 Windows 打包结果。

## 当前可执行测试

当前环境若未安装 `pytest`，可以直接运行基础测试：

```bash
python tests/test_core.py
```

如果已安装开发依赖：

```bash
python -m pytest -q
```

测试样本位于 `file/` 目录，其中 `file/244771纵校/` 包含一批 `.tif` 与历史 OCR 相关文件，可作为后续纵校、人工画框、自动收紧和导出回归素材。

## 目标工作流

1. 用户导入图片/PDF，程序创建或打开 `.ocrproj` 项目。
2. PDF 渲染为项目缓存图片，图片生成缩略图和基础预处理图。
3. 自动版面分析生成块，用户可人工新增、调整、删除、重排块。
4. 用户粗画区域后，程序可自动收紧到真实内容边界。
5. OCR 按页或按块执行，失败块可重试或跳过。
6. 可选 LLM 文本流预审对 OCR 文本给出纠错建议；该功能默认可关闭，且只作为预审。
7. 人工横校/纵校做终审，最终文本以人工确认结果为准。
8. 导出 TXT/XML/HTML/PDF/DOCX/RTF，并生成必要的校对状态提示。

## 文档入口

- `docs/technical-implementation.md`：重构级技术实施文档，包含模块设计、数据模型、工作流、易错点和交接说明。
- `docs/llm-pre-review.md`：LLM 文本流预审功能设计，强调可关闭、预审不覆盖人工终审。
- `docs/regression-test-plan.md`：回归测试方案，包含单元、集成、UI、OCR fake adapter、LLM fake adapter 和样本测试建议。
- `AGENT.md`：双 agent / 多 agent 协作协议，包含 worktree、分支命名、共享 `plan.md`、交接模板和集成节奏。

## 多 agent 协作

当前项目推荐使用 **Coordinator + Agent A + Agent B** 的并行模式：

1. Coordinator：计划、集成、冲突处理、回归门禁
2. Agent A：OCR / 版面 / bbox / 数据流
3. Agent B：UI / 交互 / 画布 / 校对工作台

建议使用 `git worktree` 并行工作，而不是多个 agent 共用一个工作区。完整规则见 `AGENT.md`。

## 重要原则

- 先跑通完整闭环，再做 UI 美化和高级功能。
- OCR 原文、LLM 建议、人工最终文本必须分层保存。
- 导出只使用人工最终文本，除非用户明确接受建议。
- 所有外部服务都要有 fake/mock adapter，测试不得依赖外网。
- 自动收紧宁可多留边距，也不能裁掉文字。
- 不要把 Token、真实 API 地址、用户文件绝对路径等敏感信息写入文档、日志或源码默认值。
