#!/usr/bin/env bash
# ============================================================
#  OCR后处理 —— 打包脚本 (WSL 专用)
#
#  本脚本在 WSL 中通过 cmd.exe 调用 Windows 侧的 build.bat，
#  使用 Windows Python + PyInstaller 打包，输出真正的 .exe 文件。
#
#  前提：
#    - 在 WSL (Windows Subsystem for Linux) 中运行
#    - Windows 侧已安装 Python 3.10+（可通过 python.org 或 Microsoft Store 安装）
#    - cmd.exe 可从 WSL 调用（默认支持）
# ============================================================
set -e

PROJ_ROOT="$(cd "$(dirname "$0")"; pwd)"
cd "$PROJ_ROOT"

# ── 检查 WSL 环境 ─────────────────────────────────────────────
if ! command -v cmd.exe &>/dev/null; then
    echo "[ERROR] 未检测到 cmd.exe。"
    echo "        此脚本仅限 WSL 使用，请在 WSL 终端中运行。"
    echo "        纯 Linux 环境下，PyInstaller 只能生成 Linux 二进制，无法生成 Windows EXE。"
    exit 1
fi

# ── 将 WSL 路径转换为 Windows 路径 ───────────────────────────
WIN_PATH=$(wslpath -w "$PROJ_ROOT")
WIN_BAT="${WIN_PATH}\\build.bat"

echo "============================================================"
echo "  WSL Build Bridge — 调用 Windows build.bat"
echo "  项目路径 (Windows): $WIN_PATH"
echo "============================================================"
echo ""

# ── 通过 cmd.exe 在 Windows 环境执行打包 ────────────────────
cmd.exe /c "\"${WIN_BAT}\""

echo ""
echo "============================================================"
echo "  Windows 打包完成。"
echo "  输出目录（Windows）：${WIN_PATH}\\dist\\ocr_process\\"
echo "  WSL 路径：${PROJ_ROOT}/dist/ocr_process/"
echo "============================================================"
