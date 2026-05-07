你用 `claude/editable-canvas` 分支，改 `app/ui/` 下的文件。

## 背景

当前代码已包含：
- Claude Opus 上次的 UI 重构（横校、纵校、画布、可编辑框）
- GPT 后续加的模型选择下拉框（`api_settings_dialog.py` 里的 QComboBox + 4 个模型预设）

## 参考

当前目录有 `ui.jpg` 概念图，请参考图中的风格来实现。

## 任务：按概念图优化设置对话框

`app/ui/widgets/api_settings_dialog.py` 目前功能完整但视觉基础。
参考 `ui.jpg` 的风格美化它，同时**保留 GPT 加的模型下拉框和自动填 URL 功能**。

## 完成后
提交到 `claude/editable-canvas` 分支。
