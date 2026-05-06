@echo off
chcp 65001 > nul
REM ============================================================
REM  OCR Post-Processing - Windows Build Script
REM  Requirements: Python 3.10+ in PATH, internet for first run
REM  Usage: Double-click or run in cmd/PowerShell at project root
REM ============================================================
setlocal EnableDelayedExpansion

set PROJ_ROOT=%~dp0
cd /d "%PROJ_ROOT%"

echo [1/5] Checking Python...
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org and add to PATH.
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo   Found: %%v

echo [2/5] Checking PyInstaller...
python -m PyInstaller --version > nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing PyInstaller...
    pip install "pyinstaller>=6.0"
    if errorlevel 1 ( echo [ERROR] PyInstaller install failed. & pause & exit /b 1 )
)
for /f "tokens=*" %%v in ('python -m PyInstaller --version') do echo   Found: PyInstaller %%v

echo [3/5] Installing core dependencies...
pip install PySide6 qt-material lxml fpdf2 python-docx Jinja2 Pillow opencv-python numpy PyMuPDF requests
if errorlevel 1 (
    echo [ERROR] Core dependency install failed.
    pause & exit /b 1
)

echo [4/5] Installing PaddleOCR (optional, skip if it fails)...
pip install paddlepaddle paddleocr 2>&1
if errorlevel 1 (
    echo [WARN] PaddleOCR install failed - the program will run without OCR engine.
    echo        Install manually later: pip install paddlepaddle paddleocr
)

echo [5/5] Running PyInstaller...
if exist dist\ocr_process rmdir /s /q dist\ocr_process
if exist build\ocr_process rmdir /s /q build\ocr_process

python -m PyInstaller ocr_process.spec --noconfirm --clean
if errorlevel 1 (
    echo [ERROR] Build failed. See output above.
    pause & exit /b 1
)

echo.
echo ============================================================
echo  Build complete!
echo  Output: %PROJ_ROOT%dist\ocr_process\
echo  Run:    %PROJ_ROOT%dist\ocr_process\ocr_process.exe
echo ============================================================
explorer "%PROJ_ROOT%dist\ocr_process"
pause
