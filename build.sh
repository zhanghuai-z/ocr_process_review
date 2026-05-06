#!/usr/bin/env bash
# ============================================================
#  OCR后处理 —— 打包脚本 (WSL 专用)
#
#  WSL 下的 /mnt/d/ (DrvFS) 不支持 chmod，PyInstaller 打包
#  最后一步会失败。解决办法：在 Linux 临时目录构建，再复制回来。
#
#  用法:
#    ./build.sh            增量打包
#    ./build.sh --clean    全量重新打包
#    ./build.sh --help     显示帮助
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
    --wsl)
        # 旧模式：直接调用 Windows build.bat
        exec "$0" "${@:2}"
        ;;
    --help|-h)
        echo "用法: $0 [--clean]"
        echo "  (无参数)  增量打包"
        echo "  --clean   全量重新打包"
        echo "  --help    显示此帮助"
        echo ""
        echo "注意：此脚本在 WSL 中运行，使用 Linux Python + PyInstaller"
        echo "      在 Linux 临时目录构建后复制到项目 dist/。"
        echo "      如需生成 Windows .exe，请用 cmd.exe 直接运行 build.bat"
        exit 0
        ;;
esac

# ── 检查是否 WSL ──────────────────────────────────────────────
IS_WSL=false
if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=true
fi

# ── 检查 Python + PyInstaller ────────────────────────────────
echo "[1/3] Checking environment..."
python3 --version
if ! python3 -m PyInstaller --version &>/dev/null; then
    echo "[INFO] Installing PyInstaller..."
    pip3 install "pyinstaller>=6.0"
fi
echo "  PyInstaller $(python3 -m PyInstaller --version)"

# ── 依赖检查 ─────────────────────────────────────────────────
echo "[2/3] Checking dependencies..."
python3 -c "import PySide6, lxml, PIL, cv2, numpy, fitz, requests" 2>/dev/null || {
    echo "[INFO] Installing missing dependencies..."
    pip3 install PySide6 qt-material lxml fpdf2 python-docx Jinja2 Pillow opencv-python numpy PyMuPDF requests
}

# ── 确定构建输出位置 ──────────────────────────────────────────
if [ "$IS_WSL" = true ] && [[ "$PROJ_ROOT" == /mnt/* ]]; then
    # 在 WSL 中且项目在 /mnt/ (Windows 盘) — 使用 /tmp 构建
    BUILD_ROOT=$(mktemp -d /tmp/ocr_build_XXXXX)
    echo "[INFO] 项目在 Windows 盘，使用 Linux /tmp 目录构建: $BUILD_ROOT"
    cp -a "$PROJ_ROOT"/* "$BUILD_ROOT/" 2>/dev/null || true
    cp -a "$PROJ_ROOT"/.gitignore "$BUILD_ROOT/" 2>/dev/null || true
    cd "$BUILD_ROOT"
else
    BUILD_ROOT="$PROJ_ROOT"
fi

# ── 运行 PyInstaller ──────────────────────────────────────────
echo "[3/3] Running PyInstaller..."

if [ "$CLEAN_FLAG" = "--clean" ]; then
    echo "  Full build (--clean)..."
    rm -rf build dist
    python3 -m PyInstaller ocr_process.spec --noconfirm --clean
else
    echo "  Incremental build..."
    python3 -m PyInstaller ocr_process.spec --noconfirm
fi

echo ""
echo "  Build completed successfully!"

# ── 复制回项目目录（如果在临时目录构建） ────────────────────
if [ "$BUILD_ROOT" != "$PROJ_ROOT" ]; then
    echo ""
    echo "  Copying result back to project directory..."
    rm -rf "$PROJ_ROOT/dist" "$PROJ_ROOT/build"
    cp -a "$BUILD_ROOT/dist" "$PROJ_ROOT/dist"
    cp -a "$BUILD_ROOT/build" "$PROJ_ROOT/build"
    rm -rf "$BUILD_ROOT"
    echo "  Copied to: $PROJ_ROOT/dist/"
fi

echo ""
echo "============================================================"
echo "  Build complete!"
echo "  输出: $PROJ_ROOT/dist/ocr_process/"
echo "  运行: $PROJ_ROOT/dist/ocr_process/ocr_process"
echo "============================================================"
