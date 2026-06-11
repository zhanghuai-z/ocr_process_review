#!/usr/bin/env bash
# ============================================================
#  OCR后处理 —— WSL 主程序打包桥接脚本
# ============================================================
set -e

PROJ_ROOT="$(cd "$(dirname "$0")"; pwd)"
cd "$PROJ_ROOT"

case "${1:-}" in
    --help|-h)
        echo "用法: $0 [--clean]"
        echo "  (无参数)  调用 Windows build.bat 增量打包主程序"
        echo "  --clean   调用 Windows build.bat --clean"
        exit 0
        ;;
esac

if ! command -v cmd.exe >/dev/null 2>&1; then
    echo "[ERROR] 未检测到 cmd.exe。"
    echo "        当前脚本只能在 WSL 中作为 build.bat 的桥接层使用。"
    exit 1
fi

WIN_PROJ_ROOT="$(wslpath -w "$PROJ_ROOT")"
WIN_BAT="${WIN_PROJ_ROOT}\\build.bat"
ARG=""

case "${1:-}" in
    --clean|-c)
        ARG=" --clean"
        ;;
    "")
        ;;
    *)
        echo "[ERROR] 不支持的参数：$1"
        echo "        仅支持 --clean / --help"
        exit 1
        ;;
esac

if [ -n "$ARG" ]; then
    cmd.exe /c "${WIN_BAT}" "$ARG"
else
    cmd.exe /c "${WIN_BAT}"
fi
