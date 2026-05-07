#!/usr/bin/env bash
# ============================================================
#  OCR后处理 —— WSL 打包桥接脚本
#
#  约定：
#    - 生成 Windows .exe 的唯一正式入口是 build.bat
#    - 本脚本不再直接运行 Linux PyInstaller
#    - 在 WSL 中使用时，仅负责转调 Windows cmd.exe / build.bat
#
#  用法：
#    ./build.sh            等价于 build.bat
#    ./build.sh --clean    等价于 build.bat --clean
#    ./build.sh --help     显示帮助
# ============================================================
set -e

PROJ_ROOT="$(cd "$(dirname "$0")"; pwd)"
cd "$PROJ_ROOT"

case "${1:-}" in
    --help|-h)
        echo "用法: $0 [--clean]"
        echo "  (无参数)  调用 Windows build.bat 增量打包"
        echo "  --clean   调用 Windows build.bat --clean"
        echo "  --help    显示此帮助"
        echo ""
        echo "说明：Windows .exe 的正式打包入口只有 build.bat。"
        echo "      本脚本仅在 WSL 中作为桥接层使用。"
        exit 0
        ;;
esac

if ! command -v cmd.exe >/dev/null 2>&1; then
    echo "[ERROR] 未检测到 cmd.exe。"
    echo "        当前脚本只能在 WSL 中作为 build.bat 的桥接层使用。"
    echo "        如需生成 Windows .exe，请在 Windows 中直接运行 build.bat。"
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

echo "============================================================"
echo "  WSL Build Bridge -> build.bat"
echo "  Project: $WIN_PROJ_ROOT"
echo "============================================================"
echo ""

cmd.exe /c "\"${WIN_BAT}\"${ARG}"
