@echo off
chcp 65001 > nul
REM ============================================================
REM  OCR Process - Main App Build Script
REM
REM  Usage:
REM    build.bat           Incremental build (main app only)
REM    build.bat --clean   Full rebuild (main app only)
REM    build.bat --help    Show help
REM ============================================================
setlocal DISABLEDELAYEDEXPANSION

set PROJ_ROOT=%~dp0
cd /d "%PROJ_ROOT%"

set CLEAN_BUILD=0
if /i "%1"=="--clean" set CLEAN_BUILD=1
if /i "%1"=="-c" set CLEAN_BUILD=1
if /i "%1"=="--help" goto :help
if /i "%1"=="-h" goto :help
if /i "%1"=="/?" goto :help

echo ============================================================
echo  OCR Process - Main App Build
if %CLEAN_BUILD%==1 (
    echo  Mode: FULL ^(--clean^)
) else (
    echo  Mode: INCREMENTAL
)
echo ============================================================
echo.

echo [1/5] Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Python not found. Install Python 3.10+ from python.org
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo   %%v

echo [2/5] PyInstaller...
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo   Installing...
    pip install "pyinstaller>=6.0"
    if errorlevel 1 (
        echo [FAIL] PyInstaller install failed
        exit /b 1
    )
)
for /f "tokens=*" %%v in ('python -m PyInstaller --version') do echo   %%v

echo [3/5] Dependencies...
set NEED_PIP=0
python -c "import PySide6"       2>nul || (echo   [MISS] PySide6 & set NEED_PIP=1)
python -c "import lxml"          2>nul || (echo   [MISS] lxml & set NEED_PIP=1)
python -c "import fpdf"          2>nul || (echo   [MISS] fpdf2 & set NEED_PIP=1)
python -c "import docx"          2>nul || (echo   [MISS] python-docx & set NEED_PIP=1)
python -c "import jinja2"        2>nul || (echo   [MISS] Jinja2 & set NEED_PIP=1)
python -c "import PIL"           2>nul || (echo   [MISS] Pillow & set NEED_PIP=1)
python -c "import cv2"           2>nul || (echo   [MISS] opencv-python & set NEED_PIP=1)
python -c "import numpy"         2>nul || (echo   [MISS] numpy & set NEED_PIP=1)
python -c "import fitz"          2>nul || (echo   [MISS] PyMuPDF & set NEED_PIP=1)
python -c "import requests"      2>nul || (echo   [MISS] requests & set NEED_PIP=1)

if %NEED_PIP%==1 (
    echo.
    echo   Installing missing packages...
    pip install PySide6 lxml fpdf2 python-docx Jinja2 Pillow opencv-python numpy PyMuPDF requests
    if errorlevel 1 (
        echo [FAIL] pip install failed.
        exit /b 1
    )
) else (
    echo   All OK
)

python -c "import paddleocr" >nul 2>&1
if errorlevel 1 (
    echo [SKIP] PaddleOCR not installed ^(optional^)
)

echo [4/5] Preparing output directories...
echo   Stopping ocr_process.exe...
taskkill /F /IM ocr_process.exe >nul 2>&1
ping -n 2 127.0.0.1 >nul 2>&1

if exist "%PROJ_ROOT%dist\ocr_process" (
    rmdir /s /q "%PROJ_ROOT%dist\ocr_process" 2>nul
    if exist "%PROJ_ROOT%dist\ocr_process" (
        echo [FAIL] Cannot remove dist\ocr_process -- directory still locked.
        exit /b 1
    )
)

if %CLEAN_BUILD%==1 (
    echo   Cleaning build cache...
    if exist "%PROJ_ROOT%build\ocr_process" (
        rmdir /s /q "%PROJ_ROOT%build\ocr_process" 2>nul
    )
    echo   Done.
) else (
    echo   Using cached build cache...
)

echo [5/5] PyInstaller...
if %CLEAN_BUILD%==1 (
    python -m PyInstaller "%PROJ_ROOT%ocr_process.spec" --noconfirm --clean
) else (
    python -m PyInstaller "%PROJ_ROOT%ocr_process.spec" --noconfirm
)

if errorlevel 1 (
    echo [FAIL] Main app build failed.
    exit /b 1
)

echo.
echo ============================================================
echo  DONE! Output: %PROJ_ROOT%dist\ocr_process\
echo  Executable: ocr_process.exe
echo ============================================================
exit /b 0

:help
echo.
echo OCR Process - Main App Build Script
echo.
echo   build.bat             Incremental build ^(main app only^)
echo   build.bat --clean     Full rebuild ^(main app only^)
echo   build.bat --help      Show this help
echo.
echo  Output: dist\ocr_process\
echo.
exit /b 0
