@echo off
REM ============================================================
REM  OCR Post-Processing - Build Script
REM
REM  Usage:
REM    build.bat           Incremental build (fast)
REM    build.bat --clean   Full rebuild (clean)
REM    build.bat --help    Show help
REM
REM  First run: pip installs dependencies automatically.
REM  Subsequent runs: skip pip if already installed.
REM ============================================================
setlocal EnableDelayedExpansion

set PROJ_ROOT=%~dp0
cd /d "%PROJ_ROOT%"

set CLEAN_BUILD=0
if "%1"=="--clean" set CLEAN_BUILD=1
if "%1"=="-c" set CLEAN_BUILD=1
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help
if "%1"=="/?" goto :help

echo ============================================================
echo  OCR Post-Processing - Build
if %CLEAN_BUILD%==1 (
    echo  Mode: FULL rebuild (--clean)
) else (
    echo  Mode: INCREMENTAL (use --clean for full rebuild)
)
echo ============================================================
echo.

REM -- [1/4] Check Python -----------------------------------------
echo [1/4] Checking Python...
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org.
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo   Found: %%v

REM -- [2/4] Check PyInstaller ------------------------------------
echo [2/4] Checking PyInstaller...
python -m PyInstaller --version > nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing PyInstaller...
    pip install "pyinstaller>=6.0"
    if errorlevel 1 ( echo [ERROR] PyInstaller install failed. & pause & exit /b 1 )
)
for /f "tokens=*" %%v in ('python -m PyInstaller --version') do echo   Found: PyInstaller %%v

REM -- [3/4] Check dependencies (incremental) ---------------------
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

REM PaddleOCR (optional)
python -c "import paddleocr" > nul 2>&1
if errorlevel 1 (
    echo [INFO] PaddleOCR not found - skipping (optional).
)

REM -- [4/4] Run PyInstaller --------------------------------------
echo [4/4] Running PyInstaller...

if %CLEAN_BUILD%==1 (
    echo   Cleaning previous build artifacts...
    if exist dist\ocr_process rmdir /s /q dist\ocr_process
    if exist build\ocr_process rmdir /s /q build\ocr_process
    python -m PyInstaller ocr_process.spec --noconfirm --clean
) else (
    REM Incremental: keep build/ cache
    python -m PyInstaller ocr_process.spec --noconfirm
)

if errorlevel 1 (
    echo [ERROR] Build failed. Try: build.bat --clean
    pause & exit /b 1
)

echo.
echo ============================================================
echo  Build complete!
echo  Output: %PROJ_ROOT%dist\ocr_process\
echo  Run:    %PROJ_ROOT%dist\ocr_process\ocr_process.exe
echo.
echo  Tip: Use "build.bat" for fast incremental rebuilds.
echo       Use "build.bat --clean" if you hit strange errors.
echo ============================================================
explorer "%PROJ_ROOT%dist\ocr_process"
pause & exit /b 0

:help
echo.
echo OCR Post-Processing - Build Script
echo.
echo   build.bat             Incremental build (default)
echo   build.bat --clean     Full rebuild from scratch
echo   build.bat --help      Show this help
echo.
echo  Output: dist\ocr_process\
echo.
pause & exit /b 0
