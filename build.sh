#!/usr/bin/env bash
# ============================================================
#  OCR后处理 —— 打包脚本 (WSL 专用，增量模式)
#
#  用法:
#    ./build.sh            增量打包（快速）
#    ./build.sh --clean    全量重新打包
#    ./build.sh --help     显示帮助
#
#  本脚本在 WSL 中通过 cmd.exe 调用 Windows 侧的 build.bat，
#  使用 Windows Python + PyInstaller 打包，输出真正的 .exe 文件。
# ============================================================
set -e

PROJ_ROOT="$(cd "$(dirname "$0")"; pwd)"
cd "$PROJ_ROOT"

# ── 参数解析 ──────────────────────────────────────────────────
CLEAN_FLAG=""
case "${1:-}" in
    --clean|-c)
        CLEAN_FLAG="--clean"
        ;;
    --help|-h)
        echo "用法: $0 [--clean|--help]"
        echo "  (无参数)  增量打包，保留 build/ 缓存"
        echo "  --clean   全量重新打包"
        echo "  --help    显示此帮助"
        exit 0
        ;;
esac

# ── 检查 WSL 环境 ─────────────────────────────────────────────
if ! command -v cmd.exe &>/dev/null; then
    echo "[ERROR] 未检测到 cmd.exe。"
    echo "        此脚本仅限 WSL 使用。"
    exit 1
fi

# ── 调用 Windows build.bat ──────────────────────────────────
WIN_PATH=$(wslpath -w "$PROJ_ROOT")
WIN_BAT="${WIN_PATH}\\build.bat"

echo "============================================================"
echo "  WSL Build Bridge — 调用 Windows build.bat"
echo "  项目路径 (Windows): $WIN_PATH"
if [ -n "$CLEAN_FLAG" ]; then
    echo "  模式: 全量打包"
else
    echo "  模式: 增量打包"
fi
echo "============================================================"
echo ""

# 传递参数到 Windows
cmd.exe /c "\"${WIN_BAT}\" ${CLEAN_FLAG}"

echo ""
echo "============================================================"
echo "  Windows 打包完成。"
echo "  输出：${PROJ_ROOT}/dist/ocr_process/"
echo "============================================================"
