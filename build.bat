@echo off
chcp 65001 > nul
REM ============================================================
REM  OCR Post-Processing - Windows Build Script (增量模式)
REM
REM  用法:
REM    build.bat           增量打包（跳过已完成步骤，速度快）
REM    build.bat --clean   全量重新打包（彻底清理后从零构建）
REM    build.bat --help    显示帮助
REM
REM  增量原理：
REM    - PyInstaller 不使用 --clean 时会缓存分析结果
REM    - 不删除 build/ 目录，只更新有变化的文件
REM    - pip 只检查缺少的依赖，不重复安装
REM ============================================================
setlocal EnableDelayedExpansion

set PROJ_ROOT=%~dp0
cd /d "%PROJ_ROOT%"

REM 解析参数
set CLEAN_BUILD=0
if "%1"=="--clean" set CLEAN_BUILD=1
if "%1"=="-c" set CLEAN_BUILD=1
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help
if "%1"=="/?" goto :help

echo ============================================================
echo  OCR 后处理 — 打包脚本
if %CLEAN_BUILD%==1 (
    echo  模式：全量打包 (--clean)
) else (
    echo  模式：增量打包（首次或改依赖时建议用 --clean）
)
echo ============================================================
echo.

REM ── [1/4] 检查 Python ────────────────────────────────────────
echo [1/4] Checking Python...
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org and add to PATH.
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo   Found: %%v

REM ── [2/4] 检查 PyInstaller ───────────────────────────────────
echo [2/4] Checking PyInstaller...
python -m PyInstaller --version > nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing PyInstaller...
    pip install "pyinstaller>=6.0"
    if errorlevel 1 ( echo [ERROR] PyInstaller install failed. & pause & exit /b 1 )
)
for /f "tokens=*" %%v in ('python -m PyInstaller --version') do echo   Found: PyInstaller %%v

REM ── [3/4] 检查核心依赖（增量检查）────────────────────────────
echo [3/4] Checking core dependencies...
python -c "import PySide6, lxml, fpdf, docx, jinja2, PIL, cv2, fitz, numpy, requests" > nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing missing core dependencies...
    pip install PySide6 qt-material lxml fpdf2 python-docx Jinja2 Pillow opencv-python numpy PyMuPDF requests
    if errorlevel 1 (
        echo [ERROR] Core dependency install failed.
        pause & exit /b 1
    )
) else (
    echo   All core dependencies satisfied.
)

REM PaddleOCR（可选，失败不影响打包）
python -c "import paddleocr" > nul 2>&1
if errorlevel 1 (
    echo [INFO] PaddleOCR not found — skipping (optional).
    echo        Install: pip install paddlepaddle paddleocr
)

REM ── [4/4] 运行 PyInstaller ───────────────────────────────────
echo [4/4] Running PyInstaller...

if %CLEAN_BUILD%==1 (
    echo   Cleaning previous build artifacts...
    if exist dist\ocr_process rmdir /s /q dist\ocr_process
    if exist build\ocr_process rmdir /s /q build\ocr_process
    python -m PyInstaller ocr_process.spec --noconfirm --clean
) else (
    REM 增量模式：保留 build/ 缓存，不做 --clean
    python -m PyInstaller ocr_process.spec --noconfirm
)

if errorlevel 1 (
    echo [ERROR] Build failed. Try "build.bat --clean" for a full rebuild.
    pause & exit /b 1
)

echo.
echo ============================================================
echo  Build complete!
echo  输出: %PROJ_ROOT%dist\ocr_process\
echo  运行: %PROJ_ROOT%dist\ocr_process\ocr_process.exe
echo.
echo  提示：第二次打包用 "build.bat" 即可（增量模式）
echo        出问题时用 "build.bat --clean" 全量重来
echo ============================================================
explorer "%PROJ_ROOT%dist\ocr_process"
pause & exit /b 0

:help
echo.
echo OCR 后处理 — 打包脚本使用说明
echo.
echo   build.bat             增量打包（跳过已完成的步骤）
echo   build.bat --clean     全量重新打包（清理所有缓存）
echo   build.bat --help      显示此帮助
echo.
echo  输出目录：dist\ocr_process\
echo.
pause & exit /b 0
